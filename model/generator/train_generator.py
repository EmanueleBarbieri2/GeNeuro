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
        # Filter based on the split (Train vs Val)
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

def prom3e_recon_loss(pred_f, ground_truth_f, alpha=-2.0, beta=2.0):
    """
    ProM3E Equation 4: Contrastive Reconstruction Loss.
    Encourages the predicted embedding to be closer to its ground truth 
    than to the ground truths of other patients in the batch.
    """
    # pairwise euclidean distances
    dists = torch.cdist(pred_f, ground_truth_f, p=2)
    # scaling/shifting
    logits = alpha * dists + beta
    # labels are the diagonal (0, 1, 2...) because batch is aligned
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def run_step(model, batch_embeddings, real_missing_mask, optimizer=None, alpha=-2.0, beta=2.0, lambd=0.001, is_train=True):
    if is_train:
        model.train()
        if optimizer: optimizer.zero_grad()
    else:
        model.eval()
    
    B = batch_embeddings.size(0)
    
    # --- Dynamic Masking (Training Only) ---
    # We artificially hide available modalities to force the model to reconstruction them.
    train_mask = torch.zeros_like(real_missing_mask)
    if is_train:
        for b in range(B):
            visible = (~real_missing_mask[b]).nonzero(as_tuple=True)[0].tolist()
            # If we have >1 modality, keep one random and mask the rest
            if len(visible) > 1:
                to_keep = random.choice(visible)
                for idx in visible:
                    if idx != to_keep: 
                        train_mask[b, idx] = True
    
    # Input is the Union of "Real Missing" and "Artificially Hidden"
    input_mask = real_missing_mask | train_mask
    
    # Forward Pass
    z_recon, mu, logvar = model(batch_embeddings, input_mask)

    # 1. VIB Loss (KL Divergence)
    # Standard VAE KL: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # 2. Contrastive Recon Loss (Eq 4)
    # We calculate loss on ALL available modalities (Autoencoder style)
    # or specifically on the masked ones. ProM3E typically enforces consistency on all.
    recon_loss_total = 0.0
    mod_count = 0
    
    for i in range(4):
        # We compute loss for any modality that actually EXISTS in the ground truth
        active = ~real_missing_mask[:, i]
        if active.any():
            recon_loss_total += prom3e_recon_loss(
                z_recon[active, i, :], 
                batch_embeddings[active, i, :], 
                alpha, beta
            )
            mod_count += 1

    # Normalize by number of modalities processed
    avg_recon = recon_loss_total / max(mod_count, 1)
    
    # Total Loss
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
    parser.add_argument('--out', required=True, help="Path to save generator checkpoint")
    parser.add_argument('--split_path', default='data/unified_split.txt')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_known_args()[0]

    # 1. Load Data Splits
    train_ids, val_ids = load_split_ids(args.split_path)
    
    # 2. Create Datasets
    # Note: Val dataset does not need masking during training, but we track loss
    train_dataset = EmbeddingDataset(args.embeddings_path, allowed_ids=train_ids if train_ids else None)
    val_dataset = EmbeddingDataset(args.embeddings_path, allowed_ids=val_ids if val_ids else None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model & Optimizer
    model = ProM3E_Generator(embed_dim=1024).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training ProM3E Generator | Mode: {args.device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # --- TRAIN ---
        t_loss, t_rec, t_kl = 0, 0, 0
        for _, x, miss in train_loader:
            l, r, k = run_step(model, x.to(args.device), miss.to(args.device), optimizer, is_train=True)
            t_loss += l; t_rec += r; t_kl += k
        
        # --- VALIDATE ---
        v_loss, v_rec, v_kl = 0, 0, 0
        with torch.no_grad():
            for _, x, miss in val_loader:
                l, r, k = run_step(model, x.to(args.device), miss.to(args.device), optimizer=None, is_train=False)
                v_loss += l; v_rec += r; v_kl += k

        avg_t_loss = t_loss/len(train_loader)
        avg_v_loss = v_loss/len(val_loader) if len(val_loader) > 0 else 0
        
        # Save best model
        status = ""
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save({"model_state": model.state_dict()}, args.out)
            status = "⭐"

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train: {avg_t_loss:.4f} (Rec: {t_rec/len(train_loader):.2f}) | "
              f"Val: {avg_v_loss:.4f} (Rec: {v_rec/len(val_loader):.2f}) {status}")

    print(f"Training complete. Best Generator saved to {args.out}")

if __name__ == "__main__": main()