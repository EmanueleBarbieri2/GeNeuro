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
        payload = torch.load(embeddings_path, map_location="cpu")
        embeddings = payload["embeddings"]
        by_id = defaultdict(dict)
        for idx, (mod, sid) in enumerate(zip(payload["labels"], payload["ids"])):
            if mod not in by_id[sid]:
                by_id[sid][mod] = embeddings[idx]

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

def prom3e_recon_loss(pred_f, ground_truth_f, alpha=-2.0, beta=2.0):
    """ProM3E Equation 4: Contrastive Reconstruction Loss."""
    dists = torch.cdist(pred_f, ground_truth_f, p=2)
    logits = alpha * dists + beta
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def train_prom3e_step(model, batch_embeddings, real_missing_mask, optimizer, alpha=-5.0, beta=5.0, lambd=0.001):
    model.train()
    optimizer.zero_grad()
    B = batch_embeddings.size(0)
    
    # Dynamic Masking: force the model to practice reconstructing visible data
    train_mask = torch.zeros_like(real_missing_mask)
    for b in range(B):
        visible = (~real_missing_mask[b]).nonzero(as_tuple=True)[0].tolist()
        if len(visible) > 1:
            to_keep = random.choice(visible)
            for idx in visible:
                if idx != to_keep: train_mask[b, idx] = True

    z_recon, mu, logvar = model(batch_embeddings, real_missing_mask | train_mask)

    # 1. VIB Loss (KL)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # 2. Contrastive Recon Loss (Eq 4)
    recon_loss_total, mod_count = 0.0, 0
    for i in range(4):
        active = ~real_missing_mask[:, i]
        if active.any():
            recon_loss_total += prom3e_recon_loss(z_recon[active, i, :], batch_embeddings[active, i, :], alpha, beta)
            mod_count += 1

    total_loss = (recon_loss_total / max(mod_count, 1)) + (lambd * kl_loss)
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), (recon_loss_total / max(mod_count, 1)).item(), kl_loss.item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--out', required=True, help="Path to save generator checkpoint")
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--csv_path', help="Legacy/Optional CSV path from pipeline")
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args, unknown = parser.parse_known_args()

    # Load Split
    train_ids = set()
    if os.path.exists(args.split_path):
        with open(args.split_path) as f:
            mode = None
            for line in f:
                if 'train_ids' in line: mode = 'train'; continue
                if 'val_ids' in line: mode = 'val'; continue
                if line.strip() and mode == 'train': train_ids.add(line.strip())

    dataset = EmbeddingDataset(args.embeddings_path, allowed_ids=train_ids if train_ids else None)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ProM3E_Generator(embed_dim=1024).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training ProM3E Generator (Equation 4 Loss) | Mode: {args.device} | LR: {args.lr}")

    for epoch in range(args.epochs):
        total, total_rec, total_kl = 0, 0, 0
        for _, x, miss in loader:
            l, r, k = train_prom3e_step(model, x.to(args.device), miss.to(args.device), optimizer)
            total += l; total_rec += r; total_kl += k
        print(f"Epoch {epoch+1:02d}/{args.epochs} | total_loss={total/len(loader):.4f} | recon={total_rec/len(loader):.4f} | kl={total_kl/len(loader):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.out)
    print(f"Saved generator to {args.out}")

if __name__ == "__main__": main()