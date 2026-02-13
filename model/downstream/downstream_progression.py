import csv
import os
import random
from collections import defaultdict
import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

# --- Helper Functions ---
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
                try: vals.append(float(v) if v else float('nan'))
                except: vals.append(float('nan'))
            visits[patno].append({"key": f"{patno}_{event_id}", "year": year, "targets": torch.tensor(vals, dtype=torch.float32)})
    return visits

def load_embeddings(path):
    payload = torch.load(path, map_location="cpu")
    by_id = defaultdict(dict)
    for idx, (mod, sid) in enumerate(zip(payload["labels"], payload["ids"])):
        if mod not in by_id[sid]: by_id[sid][mod] = payload["embeddings"][idx]
    return by_id

def hallucinate_missing_modalities(model, available, device="cpu"):
    model.eval()
    input_tensor = torch.zeros(1, 4, 1024, device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        for i, mod in enumerate(MOD_ORDER):
            if mod in available:
                input_tensor[0, i] = available[mod].to(device)
                mask[0, i] = False
        z_recon, _, _ = model(input_tensor, mask)
    return {mod: z_recon[0][i].detach().cpu() for i, mod in enumerate(MOD_ORDER)}

def build_visit_feature(generator, available, device="cpu"):
    recon = hallucinate_missing_modalities(generator, available, device=device)
    feat, mask_feat = [], []
    for mod in MOD_ORDER:
        if mod in available:
            feat.append(available[mod]); mask_feat.append(1.0)
        else:
            feat.append(recon[mod]); mask_feat.append(0.0)
    # Returns 4096 + 4 mask bits
    return torch.cat([torch.cat(feat, dim=0), torch.tensor(mask_feat, dtype=torch.float32)], dim=0)

class SequenceDataset(Dataset):
    def __init__(self, embeddings_by_id, visits_subset, generator, device="cpu"):
        self.samples = []
        for patno, visits in visits_subset.items():
            visits = sorted(visits, key=lambda x: x["year"])
            
            if len(visits) < 2: continue
            
            sequence_embeddings = []
            valid_sequence = True
            
            for v in visits:
                mods = embeddings_by_id.get(v["key"])
                if not mods: 
                    valid_sequence = False; break
                sequence_embeddings.append(build_visit_feature(generator, mods, device))
            
            if not valid_sequence: continue

            for i in range(len(visits) - 1):
                history_emb = sequence_embeddings[:i+1]
                history_times = [v["year"] for v in visits[:i+1]]
                
                target_visit = visits[i+1]
                target_score = target_visit["targets"]
                
                dt_pred = target_visit["year"] - history_times[-1]
                
                self.samples.append((
                    history_emb,     
                    history_times,   
                    dt_pred,        
                    target_score    
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

# --- ARCHITECTURE: The Successful Forecasting GRU ---
class ForecastingGRU(nn.Module):
    def __init__(self, input_dim=4100, hidden_dim=256, dropout=0.5):
        super().__init__()
        
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.time_encoder = nn.Linear(1, 32)
        
        self.gru = nn.GRU(
            hidden_dim + 32, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout
        )
        
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
        
        final_state = h_n[-1]
        return self.head(torch.cat([final_state, dt_pred], dim=1))

def train_eval(model, train_loader, val_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_r2, best_state = -float('inf'), None
    SCALE = 100.0 

    for epoch in range(args.epochs):
        model.train(); total_loss = 0
        for x, deltas, lengths, dt_pred, y in train_loader:
            x, deltas, dt_pred, y = x.to(args.device), deltas.to(args.device), dt_pred.to(args.device), y.to(args.device)
            
            target = y[:, args.target_idx].unsqueeze(1) / SCALE
            mask = ~torch.isnan(target)
            if not mask.any(): continue
            
            x_pure = x[:, :, :-4] 
            
            pred = model(x_pure, deltas, lengths, dt_pred)
            loss = F.mse_loss(pred[mask], target[mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for x, deltas, lengths, dt_pred, y in val_loader:
                x, deltas, dt_pred = x.to(args.device), deltas.to(args.device), dt_pred.to(args.device)
                
                x_pure = x[:, :, :-4]
                p = model(x_pure, deltas, lengths, dt_pred)
                
                v_preds.append(p.cpu() * SCALE)
                v_trues.append(y[:, args.target_idx].unsqueeze(1))
        
        v_preds, v_trues = torch.cat(v_preds), torch.cat(v_trues)
        m = ~torch.isnan(v_trues)
        
        if m.sum() == 0: r2 = 0.0
        else:
            mse = F.mse_loss(v_preds[m], v_trues[m])
            var = torch.var(v_trues[m])
            r2 = (1 - mse / (var + 1e-8)).item()
        
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Val R2: {r2:.4f} {'⭐' if r2 == best_r2 else ''}")

    return best_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--progression_ckpt', required=True)
    parser.add_argument('--target_idx', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    
    # Original Tuned Parameters that worked
    parser.add_argument('--lr', type=float, default=2e-4) 
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    visits = load_csv_visits(args.csv_path)
    embs = load_embeddings(args.embeddings_path)
    gen = ProM3E_Generator(embed_dim=1024, num_layers=3).to(args.device)
    gen.load_state_dict(torch.load(args.generator_ckpt, map_location=args.device)["model_state"])

    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_pts, val_pts = set(), set()
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    train_pts.add(line.split('_')[0]) if mode == 'train' else val_pts.add(line.split('_')[0])

    t_ds = SequenceDataset(embs, {p: v for p, v in visits.items() if p in train_pts}, gen, args.device)
    v_ds = SequenceDataset(embs, {p: v for p, v in visits.items() if p in val_pts}, gen, args.device)
    
    t_loader = DataLoader(t_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    v_loader = DataLoader(v_ds, batch_size=64, collate_fn=collate_fn)

    model = ForecastingGRU(input_dim=4096, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
    best = train_eval(model, t_loader, v_loader, args)

    if best:
        os.makedirs(os.path.dirname(args.progression_ckpt), exist_ok=True)
        torch.save({"model_state": best, "target_idx": args.target_idx}, args.progression_ckpt)
        print(f"✅ Saved Forecasting Specialist to {args.progression_ckpt}")

if __name__ == "__main__": main()