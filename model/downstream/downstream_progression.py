import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import csv
import argparse
from collections import defaultdict

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

# --- Helper Functions ---
def parse_year(row):
    vd = row.get("visit_date")
    if vd and "/" in vd:
        try: m, yr = vd.split("/"); return float(yr) + (float(m) - 1) / 12.0
        except: return None
    y = row.get("YEAR")
    return float(y) if y else None

def load_csv_visits(csv_path):
    visits = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            year = parse_year(row)
            if not patno or year is None: continue
            vals = []
            for t in TARGETS:
                v = row.get(t)
                try: vals.append(float(v) if v and v != "" else float('nan'))
                except: vals.append(float('nan'))
            visits[patno].append({"key": f"{patno}_{event_id}", "year": year, "targets": torch.tensor(vals, dtype=torch.float32)})
    return visits

class SmartSequenceDataset(Dataset):
    def __init__(self, recon_demo_path, visits_by_patno, use_mask=True):
        self.samples = []
        data = torch.load(recon_demo_path, map_location="cpu")
        for patno, visits in visits_by_patno.items():
            visits = sorted(visits, key=lambda x: x["year"])
            if len(visits) < 2: continue
            seq_feat, valid_visits = [], []
            for v in visits:
                if v["key"] in data:
                    entry = data[v["key"]]
                    feat = torch.cat([entry['recon'][m].flatten() for m in MOD_ORDER])
                    if use_mask:
                        mask = torch.tensor([1.0 if m in entry['real'] else 0.0 for m in MOD_ORDER])
                        feat = torch.cat([feat, mask])
                    seq_feat.append(feat)
                    valid_visits.append(v)
            if len(seq_feat) < 2: continue
            for i in range(len(seq_feat) - 1):
                self.samples.append((seq_feat[:i+1], [v["year"] for v in valid_visits[:i+1]], 
                                    valid_visits[i+1]["year"] - valid_visits[i]["year"], valid_visits[i+1]["targets"]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    hist_embs, hist_times, dt_preds, targets = zip(*batch)
    lengths = torch.tensor([len(h) for h in hist_embs], dtype=torch.long)
    padded_embs = pad_sequence([torch.stack(h) for h in hist_embs], batch_first=True)
    padded_deltas = torch.zeros(len(batch), padded_embs.size(1))
    for i, times in enumerate(hist_times):
        t = torch.tensor(times)
        if len(t) > 1: padded_deltas[i, 1:len(t)] = t[1:] - t[:-1]
    return padded_embs, padded_deltas, lengths, torch.tensor(dt_preds).unsqueeze(1), torch.stack(targets)

class ForecastingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.compressor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.time_encoder = nn.Linear(1, 32)
        self.gru = nn.GRU(hidden_dim + 32, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 1, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, deltas, lengths, dt_pred):
        B, Seq, D = x.shape
        x_comp = self.compressor(x.reshape(-1, D)).view(B, Seq, -1)
        t_emb = F.gelu(self.time_encoder(deltas.unsqueeze(-1)))
        rnn_input = torch.cat([x_comp, t_emb], dim=2)
        packed = pack_padded_sequence(rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(torch.cat([h_n[-1], dt_pred], dim=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True); parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--progression_ckpt', required=True); parser.add_argument('--target_idx', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512); parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--split_path', required=True)
    parser.add_argument('--lr', type=float, default=5e-4); parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Tracker Variables
    best_state = None
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_rmse = float('inf')
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    visits = load_csv_visits(args.csv_path)
    
    train_pts, val_pts = set(), set()
    if os.path.exists(args.split_path):
        with open(args.split_path) as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    (train_pts if mode == 'train' else val_pts).add(line.split('_')[0])

    t_ds = SmartSequenceDataset(args.embeddings_path, {p: v for p, v in visits.items() if p in train_pts})
    v_ds = SmartSequenceDataset(args.embeddings_path, {p: v for p, v in visits.items() if p in val_pts})
    t_loader = DataLoader(t_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    v_loader = DataLoader(v_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    input_dim = t_ds[0][0][0].shape[0] if len(t_ds) > 0 else 4101
    model = ForecastingGRU(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    SCALE = 100.0
    for epoch in range(args.epochs):
        model.train(); total_loss = 0
        for x, dlt, lens, dt, y in t_loader:
            x, dlt, dt, y = x.to(device), dlt.to(device), dt.to(device), y.to(device)
            target = y[:, args.target_idx].unsqueeze(1) / SCALE
            mask = ~torch.isnan(target)
            if not mask.any(): continue
            optimizer.zero_grad(); pred = model(x, dlt, lens, dt); loss = F.mse_loss(pred[mask], target[mask])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += loss.item()
        scheduler.step()
        
        model.eval(); v_preds, v_trues = [], []
        with torch.no_grad():
            for x, dlt, lens, dt, y in v_loader:
                p = model(x.to(device), dlt.to(device), lens, dt.to(device))
                v_preds.append(p.cpu() * SCALE); v_trues.append(y[:, args.target_idx].unsqueeze(1))
        
        if v_preds:
            v_preds, v_trues = torch.cat(v_preds), torch.cat(v_trues)
            m = ~torch.isnan(v_trues)
            
            if m.any():
                valid_preds = v_preds[m]
                valid_trues = v_trues[m]
                
                # Metrics Calculation
                mse = F.mse_loss(valid_preds, valid_trues)
                var = torch.var(valid_trues) + 1e-8
                
                r2 = (1 - mse / var).item()
                mae = torch.mean(torch.abs(valid_preds - valid_trues)).item()
                rmse = torch.sqrt(mse).item()
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_mae = mae
                    best_rmse = rmse
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(t_loader):.4f} | Val R2: {r2:.4f} | MAE: {mae:.4f}")

    target_name = TARGETS[args.target_idx].upper()
    print(f"\n✅ Progression Training Complete [{target_name}].")
    print(f"Best R2:   {best_r2:.4f}")
    print(f"Best MAE:  {best_mae:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    if best_state is not None:
        os.makedirs(os.path.dirname(args.progression_ckpt), exist_ok=True)
        torch.save({
            "model_state": best_state, 
            "input_dim": input_dim, 
            "target_idx": args.target_idx,
            "metrics": {"r2": best_r2, "mae": best_mae, "rmse": best_rmse}
        }, args.progression_ckpt)
        print(f"✅ Saved best GRU sequence model to {args.progression_ckpt}")

if __name__ == "__main__": main()