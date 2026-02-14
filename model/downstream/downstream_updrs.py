import csv
import os
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Logic for Target selection based on CSV columns
# 0=UPDRS1, 1=UPDRS2, 2=UPDRS3, 3=UPDRS4
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

def load_csv_targets(csv_path):
    targets = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            if not patno or not event_id: continue
            key, vals, valid = f"{patno}_{event_id}", [], True
            for t in TARGETS:
                v = row.get(t)
                try: vals.append(float(v) if v and v != "" else float('nan'))
                except: vals.append(float('nan'))
            if valid: targets[key] = torch.tensor(vals, dtype=torch.float32)
    return targets

class SmartUpdrsDataset(Dataset):
    def __init__(self, recon_demo_path, targets_dict, use_mask=True):
        """Loads pre-fused Smart Hybrid embeddings (Real + Hallucinated)."""
        self.samples = []
        data = torch.load(recon_demo_path, map_location="cpu")
        mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
        
        for key, target in targets_dict.items():
            if key not in data: continue
            
            patient_entry = data[key]
            hybrid_mods = patient_entry['recon']
            real_mods = patient_entry['real']
            
            # 1. Concatenate 4 modalities
            feat = torch.cat([hybrid_mods[m] for m in mod_order])
            
            # 2. Add availability mask
            if use_mask:
                mask = torch.tensor([1.0 if m in real_mods else 0.0 for m in mod_order])
                x = torch.cat([feat, mask])
            else:
                x = feat
                
            self.samples.append((key, x, target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx][1], self.samples[idx][2]

class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 4, 1) # Single target output
        )
    def forward(self, x): return self.net(x)

def train_eval(model, train_loader, val_loader, target_idx, device="cpu", epochs=50, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_r2, best_state = -float('inf'), None
    SCALE = 100.0 # Normalizes target range for neural net stability

    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            target = y[:, target_idx].unsqueeze(1) / SCALE
            mask = ~torch.isnan(target)
            if not mask.any(): continue

            pred = model(x)
            loss = F.mse_loss(pred[mask], target[mask])
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()
        
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for x, y in val_loader:
                preds.append(model(x.to(device)).cpu() * SCALE)
                trues.append(y[:, target_idx].unsqueeze(1))
            
            preds, trues = torch.cat(preds), torch.cat(trues)
            mask = ~torch.isnan(trues)
            if mask.sum() == 0:
                avg_v_mse, r2 = 0.0, 0.0
            else:
                avg_v_mse = F.mse_loss(preds[mask], trues[mask]).item()
                var = torch.var(trues[mask])
                r2 = (1 - avg_v_mse / (var.item() + 1e-8))

            if r2 > best_r2:
                best_r2 = r2; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                status = "⭐ Best!"
            else: status = ""

            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val MSE: {avg_v_mse:.4f} | Val R2: {r2:.4f} {status}")
    return best_state

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True) # Point to recon_demo.pt
    parser.add_argument('--updrs_ckpt', required=True)
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--target_idx', type=int, default=2) # 1=UPDRS2, 2=UPDRS3
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_mask', action='store_true', default=True)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()

    # 1. Load Split IDs
    train_ids, val_ids = set(), set()
    if os.path.exists(args.split_path):
        with open(args.split_path) as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    if mode == 'train': train_ids.add(line)
                    else: val_ids.add(line)

    # 2. Load Data
    targets = load_csv_targets(args.csv_path)
    dataset = SmartUpdrsDataset(args.embeddings_path, targets, use_mask=args.use_mask)
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, [i for i, s in enumerate(dataset.samples) if s[0] in train_ids]), batch_size=32, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, [i for i, s in enumerate(dataset.samples) if s[0] in val_ids]), batch_size=64)

    # 3. Initialize Model (Dynamic Dimension)
    if len(dataset) > 0:
        input_dim = dataset[0][0].shape[0]
    else:
        input_dim = 4096 + (4 if args.use_mask else 0)

    print(f"🚀 UPDRS Regressor initialized with Input Dim: {input_dim}")
    model = Regressor(input_dim, dropout=args.dropout).to(args.device)
    
    best = train_eval(model, train_loader, val_loader, args.target_idx, device=args.device, epochs=args.epochs, lr=args.lr)
    
    if best:
        os.makedirs(os.path.dirname(args.updrs_ckpt), exist_ok=True)
        torch.save({"model_state": best, "input_dim": input_dim, "target_idx": args.target_idx}, args.updrs_ckpt)
        print(f"Saved best regressor to {args.updrs_ckpt}")

if __name__ == "__main__": main()