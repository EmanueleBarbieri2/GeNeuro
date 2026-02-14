import csv
import os
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

# --- Smart Helper Functions ---
def parse_float(x):
    try: return float(x)
    except: return None

def parse_year(row):
    y = parse_float(row.get("YEAR"))
    if y is not None: return y
    vd = row.get("visit_date")
    if vd and "/" in vd:
        try: m, yr = vd.split("/"); return float(yr) + (float(m) - 1) / 12.0
        except: return None
    return None

def load_csv_visits(csv_path):
    visits = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            if not patno or not event_id: continue
            year = parse_year(row)
            if year is None: continue
            vals = []
            for t in TARGETS:
                v = row.get(t)
                try: vals.append(float(v) if v and v != "" else float('nan'))
                except: vals.append(float('nan'))
            visits[patno].append({
                "key": f"{patno}_{event_id}", 
                "year": year, 
                "targets": torch.tensor(vals, dtype=torch.float32)
            })
    return visits

class SmartSequenceDataset(Dataset):
    def __init__(self, recon_demo_path, visits_by_patno, use_mask=True):
        self.samples = []
        data = torch.load(recon_demo_path, map_location="cpu")
        
        for patno, visits in visits_by_patno.items():
            # Sort visits chronologically
            visits = sorted(visits, key=lambda x: x["year"])
            if len(visits) < 2: continue
            
            sequence_features = []
            valid_keys = []
            
            for v in visits:
                if v["key"] in data:
                    entry = data[v["key"]]
                    # Use the Smart Hybrid (Real + Hallucinated Fusion)
                    feat = torch.cat([entry['recon'][m] for m in MOD_ORDER])
                    if use_mask:
                        mask = torch.tensor([1.0 if m in entry['real'] else 0.0 for m in MOD_ORDER])
                        feat = torch.cat([feat, mask])
                    sequence_features.append(feat)
                    valid_keys.append(v)
            
            # We need at least one visit for history and one for target
            if len(sequence_features) < 2: continue

            for i in range(len(sequence_features) - 1):
                # History up to visit i
                history_feat = sequence_features[:i+1]
                history_years = [v["year"] for v in valid_keys[:i+1]]
                
                # Target at visit i+1
                target_visit = valid_keys[i+1]
                dt_pred = target_visit["year"] - history_years[-1]
                
                self.samples.append((
                    history_feat, 
                    history_years, 
                    dt_pred, 
                    target_visit["targets"]
                ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    hist_embs, hist_times, dt_preds, targets = zip(*batch)
    lengths = torch.tensor([len(h) for h in hist_embs], dtype=torch.long)
    padded_embs = pad_sequence([torch.stack(h) for h in hist_embs], batch_first=True)
    
    padded_deltas = torch.zeros(len(batch), padded_embs.size(1))
    for i, times in enumerate(hist_times):
        deltas = [0.0] + [times[j] - times[j-1] for j in range(1, len(times))]
        padded_deltas[i, :len(deltas)] = torch.tensor(deltas)
        
    return padded_embs, padded_deltas, lengths, torch.tensor(dt_preds, dtype=torch.float32).unsqueeze(1), torch.stack(targets)

# 
class ForecastingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.time_encoder = nn.Linear(1, 32)
        self.gru = nn.GRU(hidden_dim + 32, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, deltas, lengths, dt_pred):
        B, Seq, D = x.shape
        x_comp = self.compressor(x.view(-1, D)).view(B, Seq, -1)
        t_emb = F.relu(self.time_encoder(deltas.unsqueeze(-1)))
        
        rnn_input = torch.cat([x_comp, t_emb], dim=2)
        packed = pack_padded_sequence(rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        
        return self.head(torch.cat([h_n[-1], dt_pred], dim=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True) # recon_demo.pt
    parser.add_argument('--progression_ckpt', required=True)
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--target_idx', type=int, default=2) # 2=UPDRS3 Motor
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # 1. Data Loading
    visits = load_csv_visits(args.csv_path)
    
    # Split handling
    train_pts, val_pts = set(), set()
    if os.path.exists(args.split_path):
        with open(args.split_path, 'r') as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    patno = line.split('_')[0]
                    if mode == 'train': train_pts.add(patno)
                    else: val_pts.add(patno)

    t_ds = SmartSequenceDataset(args.embeddings_path, {p: v for p, v in visits.items() if p in train_pts})
    v_ds = SmartSequenceDataset(args.embeddings_path, {p: v for p, v in visits.items() if p in val_pts})
    
    t_loader = DataLoader(t_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    v_loader = DataLoader(v_ds, batch_size=64, collate_fn=collate_fn)

    # 2. Model Initialization (Dynamic Dimension)
    if len(t_ds) > 0:
        # Check first visit of first sample [0][0] is tensor list
        input_dim = t_ds[0][0][0].shape[0] 
    else:
        input_dim = 4100 # Fallback

    print(f"🚀 Progression GRU initialized with Input Dim: {input_dim}")
    model = ForecastingGRU(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
    
    # 3. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_r2, best_state = -float('inf'), None
    SCALE = 100.0

    for epoch in range(args.epochs):
        model.train(); total_loss = 0
        for x, deltas, lengths, dt_pred, y in t_loader:
            x, deltas, dt_pred, y = x.to(args.device), deltas.to(args.device), dt_pred.to(args.device), y.to(args.device)
            target = y[:, args.target_idx].unsqueeze(1) / SCALE
            mask = ~torch.isnan(target)
            if not mask.any(): continue
            
            # Note: Input x already contains the mask features if use_mask=True in dataset
            pred = model(x, deltas, lengths, dt_pred)
            loss = F.mse_loss(pred[mask], target[mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for x, deltas, lengths, dt_pred, y in v_loader:
                x, deltas, dt_pred = x.to(args.device), deltas.to(args.device), dt_pred.to(args.device)
                p = model(x, deltas, lengths, dt_pred)
                v_preds.append(p.cpu() * SCALE)
                v_trues.append(y[:, args.target_idx].unsqueeze(1))
        
        if len(v_preds) > 0:
            v_preds, v_trues = torch.cat(v_preds), torch.cat(v_trues)
            m = ~torch.isnan(v_trues)
            if m.sum() == 0: r2 = 0.0
            else:
                mse = F.mse_loss(v_preds[m], v_trues[m])
                var = torch.var(v_trues[m])
                r2 = (1 - mse / (var + 1e-8)).item()
        else:
            r2 = 0.0
        
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {total_loss/len(t_loader):.4f} | Val R2: {r2:.4f} {'⭐' if r2 == best_r2 else ''}")

    if best_state:
        os.makedirs(os.path.dirname(args.progression_ckpt), exist_ok=True)
        torch.save({"model_state": best_state, "target_idx": args.target_idx}, args.progression_ckpt)
        print(f"✅ Saved Forecasting Specialist to {args.progression_ckpt}")

if __name__ == "__main__":
    main()