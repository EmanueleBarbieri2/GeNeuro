import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import numpy as np
import argparse
from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]

# --- DIVERGENCE METRICS ---

def compute_mmd(z, z_prior):
    """Gaussian Kernel MMD for latent space regularization."""
    def gaussian_kernel(a, b):
        # Flattening ensures we handle [Batch*Modalities, Dim] correctly
        a_norm = a.pow(2).sum(dim=1, keepdim=True)
        b_norm = b.pow(2).sum(dim=1, keepdim=True)
        dist = a_norm + b_norm.transpose(0, 1) - 2.0 * torch.matmul(a, b.transpose(0, 1))
        return torch.exp(-dist) 

    k_zz = gaussian_kernel(z, z).mean()
    k_pp = gaussian_kernel(z_prior, z_prior).mean()
    k_zp = gaussian_kernel(z, z_prior).mean()
    return k_zz + k_pp - 2 * k_zp

def compute_kl(mu, logvar):
    """Standard VAE KL Divergence penalty."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()

def prom3e_recon_loss(pred_f, ground_truth_f, alpha, beta):
    """Contrastive cross-entropy loss based on pairwise distances."""
    dists = torch.cdist(pred_f, ground_truth_f, p=2)
    logits = alpha * dists + beta
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

# --- DATASET ---

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, allowed_ids=None):
        payload = torch.load(embeddings_path, map_location="cpu")
        embeddings = payload["embeddings"]
        
        by_id = defaultdict(dict)
        for idx, (mod, sid) in enumerate(zip(payload["labels"], payload["ids"])):
            if mod not in by_id[sid]: by_id[sid][mod] = embeddings[idx]

        self.samples = []
        for sid, mods in by_id.items():
            if allowed_ids is not None and sid not in allowed_ids: continue
            x = torch.zeros(4, embeddings.size(1))
            missing = torch.ones(4, dtype=torch.bool)
            for i, mod in enumerate(MOD_ORDER):
                if mod in mods:
                    x[i] = mods[mod]
                    missing[i] = False
            self.samples.append((sid, x, missing))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def load_split_ids(split_path):
    train_ids, val_ids = set(), set()
    if os.path.exists(split_path):
        with open(split_path) as f:
            mode = None
            for line in f:
                line = line.strip()
                if line == 'train_ids:': mode = 'train'; continue
                if line == 'val_ids:': mode = 'val'; continue
                if line and mode == 'train': train_ids.add(line)
                elif line and mode == 'val': val_ids.add(line)
    return train_ids, val_ids

# --- STEP LOGIC ---

def run_step(model, batch_embeddings, real_missing_mask, optimizer=None, 
             alpha=-5.0, beta=5.0, lambd=0.001, keep_prob=0.5, 
             is_train=True, divergence_type='mmd'):
    if is_train:
        model.train()
        if optimizer: optimizer.zero_grad()
    else:
        model.eval()
    
    # Dynamic Masking
    input_mask = real_missing_mask.clone()
    if is_train:
        for b in range(batch_embeddings.size(0)):
            visible = (~real_missing_mask[b]).nonzero(as_tuple=True)[0].tolist()
            if len(visible) > 1:
                t_mask = torch.zeros(4, dtype=torch.bool, device=batch_embeddings.device)
                for idx in visible:
                    if random.random() > keep_prob: t_mask[idx] = True
                if t_mask[visible].all(): t_mask[random.choice(visible)] = False
                input_mask[b] = real_missing_mask[b] | t_mask
    
    z_recon, mu, logvar = model(batch_embeddings, input_mask)

    # Divergence
    div_loss = torch.tensor(0.0, device=batch_embeddings.device)
    if divergence_type == 'mmd':
        std = torch.exp(0.5 * logvar)
        z_s = mu + std * torch.randn_like(std)
        div_loss = compute_mmd(z_s.view(-1, mu.size(-1)), 
                               torch.randn_like(z_s).view(-1, mu.size(-1)))
    elif divergence_type == 'kl':
        div_loss = compute_kl(mu, logvar)

    # Recon
    recon_total = 0.0
    mod_count = 0
    for i in range(4):
        active = ~real_missing_mask[:, i]
        if active.any():
            recon_total += prom3e_recon_loss(z_recon[active, i, :], batch_embeddings[active, i, :], alpha, beta)
            mod_count += 1
    avg_recon = recon_total / max(mod_count, 1)

    total_loss = avg_recon + (lambd * div_loss)
    
    if is_train and optimizer:
        total_loss.backward()
        optimizer.step()
        
    return total_loss.item(), avg_recon.item(), div_loss.item()

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambd', type=float, default=0.001)
    parser.add_argument('--divergence', choices=['mmd', 'kl', 'none'], default='mmd')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=-5.0)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--keep_prob', type=float, default=0.5)
    parser.add_argument('--kl_warmup', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_registers', type=int, default=4)
    parser.add_argument('--mlp_depth', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    args, _ = parser.parse_known_args()

    train_ids, val_ids = load_split_ids(args.split_path)
    train_dataset = EmbeddingDataset(args.embeddings_path, train_ids)
    val_dataset = EmbeddingDataset(args.embeddings_path, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    payload = torch.load(args.embeddings_path, map_location="cpu")
    model = ProM3E_Generator(
        embed_dim=payload["embeddings"].size(1), 
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers,
        num_registers=args.gen_num_registers if hasattr(args, 'gen_num_registers') else 4
    ).to(args.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    for epoch in range(args.epochs):
        warmup = min(1.0, (epoch + 1) / args.kl_warmup) if args.kl_warmup > 0 else 1.0
        
        train_m = np.zeros(3)
        for _, x, miss in train_loader:
            train_m += np.array(run_step(model, x.to(args.device), miss.to(args.device), optimizer, 
                                        args.alpha, args.beta, args.lambd*warmup, args.keep_prob, True, args.divergence))
        
        val_m = np.zeros(3)
        with torch.no_grad():
            for _, x, miss in val_loader:
                val_m += np.array(run_step(model, x.to(args.device), miss.to(args.device), None, 
                                          args.alpha, args.beta, args.lambd*warmup, args.keep_prob, False, args.divergence))
        
        avg_v = val_m[0] / len(val_loader)
        if avg_v < best_val:
            best_val = avg_v
            torch.save({"model_state": model.state_dict()}, args.out)
            
        print(f"Epoch {epoch+1:03d} | Val Loss: {avg_v:.4f} | {args.divergence.upper()} Val: {val_m[2]/len(val_loader):.4f}")