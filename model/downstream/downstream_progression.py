import csv
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
TARGETS = ["updrs1_score", "updrs2_score", "updrs3_score", "updrs4_score"]

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
    """REVERTED: Returns 4096D Reconstructions"""
    model.eval()
    input_tensor, mask = torch.zeros(1, 4, 1024, device=device), torch.ones(1, 4, dtype=torch.bool, device=device)
    with torch.no_grad():
        for i, mod in enumerate(MOD_ORDER):
            if mod in available:
                input_tensor[0, i] = available[mod].to(device)
                mask[0, i] = False
        z_recon, _, _ = model(input_tensor, mask)
    return {mod: z_recon[0][i].detach().cpu() for i, mod in enumerate(MOD_ORDER)}

def build_visit_feature(generator, available, delta_prev, device="cpu"):
    recon = hallucinate_missing_modalities(generator, available, device=device)
    feat, mask_feat = [], []
    for mod in MOD_ORDER:
        if mod in available:
            feat.append(available[mod]); mask_feat.append(1.0)
        else:
            feat.append(recon[mod]); mask_feat.append(0.0)
    x = torch.cat([torch.cat(feat, dim=0), torch.tensor(mask_feat, dtype=torch.float32), torch.tensor([delta_prev], dtype=torch.float32)], dim=0)
    return x

class SequenceTransitionDataset(Dataset):
    def __init__(self, embeddings_by_id, visits_subset, generator, device="cpu"):
        self.samples = []
        for patno, visits in visits_subset.items():
            visits = sorted(visits, key=lambda x: x["year"])
            if len(visits) < 2: continue
            for idx, (v_cur, v_next) in enumerate(zip(visits[:-1], visits[1:])):
                history = []
                prev_yr = None
                for v in visits[:idx+1]:
                    mods = embeddings_by_id.get(v["key"])
                    if not mods or len(mods) == 0: prev_yr = v["year"]; continue
                    delta_p = 0.0 if prev_yr is None else (v["year"] - prev_yr)
                    history.append(build_visit_feature(generator, mods, delta_p, device))
                    prev_yr = v["year"]
                if history: self.samples.append((history, v_next["year"] - v_cur["year"], v_next["targets"]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class RNNRegressor(nn.Module):
    def __init__(self, input_dim=4101, hidden_dim=256): # REVERTED TO FULL DIM + TIME
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4)
        )
    def forward(self, x, lengths, dt):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.head(torch.cat([h[-1], dt], dim=1))

def collate_sequences(batch):
    hist, dts, ys = zip(*batch)
    lengths = torch.tensor([len(h) for h in hist], dtype=torch.long)
    padded = pad_sequence([torch.stack(h, dim=0) for h in hist], batch_first=True)
    return padded, lengths, torch.tensor(dts, dtype=torch.float32).unsqueeze(1), torch.stack(ys, dim=0)

def train_eval(model, train_loader, val_loader, device="cpu", epochs=50, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_mse, best_state = float('inf'), None

    def masked_metrics(pred, target):
        mask = ~torch.isnan(target)
        mse_list, r2_list = [], []
        for i in range(target.shape[1]):
            m = mask[:, i]
            if m.any():
                mse = F.mse_loss(pred[m, i], target[m, i])
                mse_list.append(mse)
                var = torch.var(target[m, i], unbiased=False)
                r2_list.append(1 - mse / (var + 1e-8))
            else:
                mse_list.append(torch.tensor(float('nan'))); r2_list.append(torch.tensor(float('nan')))
        return torch.stack(mse_list), torch.stack(r2_list)

    for epoch in range(epochs):
        model.train(); total_loss, n_batches = 0.0, 0
        for x, lengths, dt, y in train_loader:
            x, lengths, dt, y = x.to(device), lengths.to(device), dt.to(device), y.to(device)
            mask = ~torch.isnan(y)
            pred = model(x, lengths, dt)
            loss = F.mse_loss(pred[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item(); n_batches += 1

        model.eval()
        with torch.no_grad():
            v_preds, v_trues = [], []
            for x, lengths, dt, y in val_loader:
                v_preds.append(model(x.to(device), lengths.to(device), dt.to(device)).cpu()); v_trues.append(y)
            v_preds, v_trues = torch.cat(v_preds), torch.cat(v_trues)
            mse_per, r2_per = masked_metrics(v_preds, v_trues)
            avg_v_mse = mse_per[~torch.isnan(mse_per)].mean().item()
            
            status = "⭐ Best!" if avg_v_mse < best_mse else ""
            if status: best_mse = avg_v_mse; best_state = model.state_dict()

            print(f"Epoch {epoch+1:02d}/{epochs} | train_mse={total_loss/max(n_batches,1):.4f} | val_mse={avg_v_mse:.4f} | val_rmse={torch.sqrt(mse_per).tolist()} | val_r2={r2_per.tolist()} {status}")
    return best_state

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--progression_ckpt', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()

    visits = load_csv_visits(args.csv_path)
    embs = load_embeddings(args.embeddings_path)
    gen = ProM3E_Generator(1024)
    gen.load_state_dict(torch.load(args.generator_ckpt, map_location="cpu")["model_state"])

    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_pts, val_pts = set(), set()
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    patno = line.split('_')[0]
                    if mode == 'train': train_pts.add(patno)
                    else: val_pts.add(patno)

    train_set = SequenceTransitionDataset(embs, {p: v for p, v in visits.items() if p in train_pts}, gen, args.device)
    val_set = SequenceTransitionDataset(embs, {p: v for p, v in visits.items() if p in val_pts}, gen, args.device)
    
    t_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_sequences)
    v_loader = DataLoader(val_set, batch_size=64, collate_fn=collate_sequences)

    model = RNNRegressor(4096 + 4 + 1).to(args.device) # REVERTED TO FULL DIM + MASK + TIME
    best = train_eval(model, t_loader, v_loader, device=args.device, epochs=args.epochs, lr=args.lr)
    if best:
        os.makedirs(os.path.dirname(args.progression_ckpt), exist_ok=True)
        torch.save({"model_state": best, "input_dim": (4096+4+1)}, args.progression_ckpt)
        print(f"Saved best regressor to {args.progression_ckpt}")

if __name__ == "__main__": main()