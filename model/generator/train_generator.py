import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, allowed_ids=None):
        print(f"Loading embeddings from {embeddings_path}...")
        payload = torch.load(embeddings_path, map_location="cpu")
        embeddings = payload["embeddings"]
        
        # Group by Subject ID
        by_id = defaultdict(dict)
        for idx, (mod, sid) in enumerate(zip(payload["labels"], payload["ids"])):
            if mod not in by_id[sid]:
                by_id[sid][mod] = embeddings[idx]

        self.samples = []
        for sid, mods in by_id.items():
            if allowed_ids is not None and sid not in allowed_ids: 
                continue
                
            x = torch.zeros(4, embeddings.size(1))
            missing = torch.ones(4, dtype=torch.bool)
            
            for i, mod in enumerate(MOD_ORDER):
                if mod in mods:
                    x[i] = mods[mod]
                    missing[i] = False
                    
            self.samples.append((sid, x, missing))
        print(f"  > Dataset loaded: {len(self.samples)} subjects.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def prom3e_recon_loss(pred_f, ground_truth_f, alpha, beta):
    """
    ProM3E Equation 4: Contrastive Reconstruction Loss.
    """
    # pairwise euclidean distances
    dists = torch.cdist(pred_f, ground_truth_f, p=2)
    # scaling/shifting
    logits = alpha * dists + beta
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def run_step(model, batch_embeddings, real_missing_mask, optimizer=None, 
             alpha=-2.0, beta=2.0, lambd=0.0005, is_train=True):
    """
    Core training step logic.
    """
    if is_train:
        model.train()
        if optimizer: optimizer.zero_grad()
    else:
        model.eval()
    
    B = batch_embeddings.size(0)
    
    # --- Dynamic Masking (Training Only) ---
    train_mask = torch.zeros_like(real_missing_mask)
    if is_train:
        for b in range(B):
            visible = (~real_missing_mask[b]).nonzero(as_tuple=True)[0].tolist()
            if len(visible) > 1:
                to_keep = random.choice(visible)
                for idx in visible:
                    if idx != to_keep: 
                        train_mask[b, idx] = True
    
    input_mask = real_missing_mask | train_mask
    z_recon, mu, logvar = model(batch_embeddings, input_mask)

    # 1. VIB Loss (KL Divergence)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # 2. Contrastive Recon Loss (Eq 4)
    recon_loss_total = 0.0
    mod_count = 0
    
    for i in range(4):
        active = ~real_missing_mask[:, i]
        if active.any():
            recon_loss_total += prom3e_recon_loss(
                z_recon[active, i, :], 
                batch_embeddings[active, i, :], 
                alpha, beta
            )
            mod_count += 1

    avg_recon = recon_loss_total / max(mod_count, 1)
    total_loss = avg_recon + (lambd * kl_loss)
    
    if is_train and optimizer:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    return total_loss.item(), avg_recon.item(), kl_loss.item()

def load_split_ids(split_path):
    train_ids, val_ids = set(), set()
    if os.path.exists(split_path):
        print(f"Loading split from {split_path}")
        with open(split_path) as f:
            mode = None
            for line in f:
                line = line.strip()
                if line == 'train_ids:': mode = 'train'; continue
                if line == 'val_ids:': mode = 'val'; continue
                if line and not line.startswith("#"):
                    if mode == 'train': train_ids.add(line)
                    elif mode == 'val': val_ids.add(line)
    return train_ids, val_ids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--csv_path', help="CSV path (for pipeline consistency)")

    # Hyperparameters from Pipeline
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4) # Lowered for 3-layer stability
    parser.add_argument('--alpha', type=float, default=-2.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--lambd', type=float, default=0.0005)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_known_args()[0]

    # 1. Load Data Splits
    train_ids, val_ids = load_split_ids(args.split_path)
    
    # 2. Create Datasets
    train_dataset = EmbeddingDataset(args.embeddings_path, allowed_ids=train_ids if train_ids else None)
    val_dataset = EmbeddingDataset(args.embeddings_path, allowed_ids=val_ids if val_ids else None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model & Optimizer (Force 3 layers for alignment depth)
    model = ProM3E_Generator(embed_dim=1024, num_layers=3).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training ProM3E Generator | Mode: {args.device} | α={args.alpha} | λ={args.lambd}")

    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        t_loss, t_rec, t_kl = 0, 0, 0
        for _, x, miss in train_loader:
            l, r, k = run_step(
                model, x.to(args.device), miss.to(args.device), 
                optimizer, 
                alpha=args.alpha, 
                beta=args.beta, 
                lambd=args.lambd, 
                is_train=True
            )
            t_loss += l; t_rec += r; t_kl += k
        
        v_loss, v_rec, v_kl = 0, 0, 0
        with torch.no_grad():
            for _, x, miss in val_loader:
                l, r, k = run_step(
                    model, x.to(args.device), miss.to(args.device), 
                    optimizer=None, 
                    alpha=args.alpha, 
                    beta=args.beta, 
                    lambd=args.lambd, 
                    is_train=False
                )
                v_loss += l; v_rec += r; v_kl += k

        avg_t_loss = t_loss/len(train_loader)
        avg_v_loss = v_loss/len(val_loader) if len(val_loader) > 0 else 0
        
        status = ""
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save({"model_state": model.state_dict()}, args.out)
            status = "⭐"

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train: {avg_t_loss:.4f} (Rec: {t_rec/len(train_loader):.2f}) | "
              f"Val: {avg_v_loss:.4f} (Rec: {v_rec/len(val_loader):.2f}) {status}")

    print(f"✅ Training complete. Best Generator saved to {args.out}")

if __name__ == "__main__": main()