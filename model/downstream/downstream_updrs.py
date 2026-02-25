import csv
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 0=UPDRS1, 1=UPDRS2, 2=UPDRS3, 3=UPDRS4
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

def load_csv_targets(csv_path):
    targets = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            if not patno or not event_id: continue
            vals = []
            for t in TARGETS:
                v = row.get(t)
                try: vals.append(float(v) if v and v != "" else float('nan'))
                except: vals.append(float('nan'))
            targets[f"{patno}_{event_id}"] = torch.tensor(vals, dtype=torch.float32)
    return targets

class SmartUpdrsDataset(Dataset):
    def __init__(self, recon_demo_path, targets_dict, use_mask=True):
        self.samples = []
        # Load to CPU to keep Blackwell VRAM clear for training
        data = torch.load(recon_demo_path, map_location="cpu")
        mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
        
        for key, target in targets_dict.items():
            if key not in data: continue
            entry = data[key]
            
            # Flatten to ensure 1D features per modality
            feat = torch.cat([entry['recon'][m].flatten() for m in mod_order])
            
            if use_mask:
                mask = torch.tensor([1.0 if m in entry['real'] else 0.0 for m in mod_order])
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
            nn.GELU(), # Blackwell-optimized activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 4, 1)
        )
    def forward(self, x): return self.net(x)

def train_eval(model, train_loader, val_loader, target_idx, device, epochs=50, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # Initialize trackers
    best_state = None
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')
    
    SCALE = 100.0 

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y[:, target_idx].unsqueeze(1) / SCALE
            mask = ~torch.isnan(target)
            if not mask.any(): continue

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred[mask], target[mask])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        scheduler.step()

        # Efficient Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.append(model(x.to(device)) * SCALE)
                trues.append(y[:, target_idx].unsqueeze(1))
            
            preds, trues = torch.cat(preds).cpu(), torch.cat(trues).cpu()
            mask = ~torch.isnan(trues)
            
            if mask.any():
                valid_preds = preds[mask]
                valid_trues = trues[mask]
                
                # Calculate all metrics
                mse_val = F.mse_loss(valid_preds, valid_trues).item()
                var_val = torch.var(valid_trues).item() + 1e-8
                
                r2 = 1 - (mse_val / var_val)
                mae = torch.mean(torch.abs(valid_preds - valid_trues)).item()
                rmse = torch.sqrt(torch.tensor(mse_val)).item()
                
                # Save based on best R2
                if r2 > best_r2:
                    best_r2 = r2
                    best_mae = mae
                    best_rmse = rmse
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    status = "⭐"
                else: 
                    status = ""

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | R2: {r2:.4f} | MAE: {mae:.4f} {status}")
                    
    # Pack metrics into a dictionary to return
    best_metrics = {
        "r2": best_r2,
        "mae": best_mae,
        "rmse": best_rmse
    }
    
    return best_state, best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--updrs_ckpt', required=True)
    parser.add_argument('--split_path', required=True)
    parser.add_argument('--target_idx', type=int, default=2) 
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_mask', action='store_true', help='Use mask for UPDRS dataset')
    parser.add_argument('--device', default='cuda')
    args, _ = parser.parse_known_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    # 2. Data Setup
    targets = load_csv_targets(args.csv_path)
    full_dataset = SmartUpdrsDataset(args.embeddings_path, targets, use_mask=args.use_mask)
    
    train_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in train_ids]
    val_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in val_ids]
    
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), 
                              batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), 
                            batch_size=args.batch_size, num_workers=2)

    input_dim = full_dataset[0][0].shape[0] if len(full_dataset) > 0 else 4100
    print(f"🚀 UPDRS Regressor | Dim: {input_dim} | Target: {TARGETS[args.target_idx]}")
    
    raw_model = Regressor(input_dim, dropout=args.dropout).to(device)
    model = raw_model
    
    # Train and evaluate
    best_state, best_metrics = train_eval(model, train_loader, val_loader, args.target_idx, device=device, epochs=args.epochs, lr=args.lr)
    
    target_name = TARGETS[args.target_idx].upper()
    
    # --- PRINT RICH STATISTICS AT THE END ---
    print(f"\n✅ Static Regression Complete [{target_name}].")
    print(f"Best R2:   {best_metrics['r2']:.4f}")
    print(f"Best MAE:  {best_metrics['mae']:.4f}")
    print(f"Best RMSE: {best_metrics['rmse']:.4f}")
    
    if best_state is not None:
        os.makedirs(os.path.dirname(args.updrs_ckpt), exist_ok=True)
        torch.save({
            "model_state": best_state, 
            "target_idx": args.target_idx,
            "metrics": best_metrics
        }, args.updrs_ckpt)
        print(f"✅ Saved best regressor to {args.updrs_ckpt}")
    
if __name__ == "__main__": 
    main()