import csv
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
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
                try: vals.append(float(v) if v else float('nan'))
                except: valid = False; break
            if valid: targets[key] = torch.tensor(vals, dtype=torch.float32)
    return targets

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

class UpdrsDataset(Dataset):
    def __init__(self, embeddings_by_id, targets_dict, generator, device="cpu"):
        self.samples = []
        for key, target in targets_dict.items():
            mods = embeddings_by_id.get(key)
            if not mods or len(mods) == 0: continue
            recon = hallucinate_missing_modalities(generator, mods, device)
            feat, mask_feat = [], []
            for mod in MOD_ORDER:
                if mod in mods: feat.append(mods[mod]); mask_feat.append(1.0)
                else: feat.append(recon[mod]); mask_feat.append(0.0)
            self.samples.append((key, torch.cat([torch.cat(feat), torch.tensor(mask_feat)]), target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx][1], self.samples[idx][2]

class Regressor(nn.Module):
    def __init__(self, input_dim=4100): # REVERTED TO FULL DIM
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4)
        )
    def forward(self, x): return self.net(x)

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
        model.train(); total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = ~torch.isnan(y)
            loss = F.mse_loss(model(x)[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            preds, trues = [], []
            for x, y in val_loader:
                preds.append(model(x.to(device)).cpu()); trues.append(y)
            preds, trues = torch.cat(preds), torch.cat(trues)
            
            mse_per_mod, r2_per_mod = masked_metrics(preds, trues)
            avg_v_mse = mse_per_mod[~torch.isnan(mse_per_mod)].mean().item()
            
            status = ""
            if avg_v_mse < best_mse:
                best_mse = avg_v_mse; best_state = model.state_dict(); status = "⭐ Best!"

            print(f"Epoch {epoch+1:02d}/{epochs} | train_mse={total_loss/len(train_loader):.4f} | "
                  f"val_mse={avg_v_mse:.4f} | val_rmse={torch.sqrt(mse_per_mod).tolist()} | "
                  f"val_r2={r2_per_mod.tolist()} {status}")
    return best_state

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--updrs_ckpt', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()

    targets = load_csv_targets(args.csv_path)
    embs = load_embeddings(args.embeddings_path)
    gen = ProM3E_Generator(1024)
    gen.load_state_dict(torch.load(args.generator_ckpt, map_location="cpu")["model_state"])

    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_ids, val_ids = set(), set()
    with open(split_path) as f:
        mode = None
        for line in f.read().splitlines():
            if 'train_ids' in line: mode = 'train'
            elif 'val_ids' in line: mode = 'val'
            elif line.strip() and not line.startswith('#'):
                if mode == 'train': train_ids.add(line)
                else: val_ids.add(line)

    dataset = UpdrsDataset(embs, targets, gen, args.device)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, [i for i, s in enumerate(dataset.samples) if s[0] in train_ids]), batch_size=32, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, [i for i, s in enumerate(dataset.samples) if s[0] in val_ids]), batch_size=64)

    model = Regressor(4096+4).to(args.device) # REVERTED
    best = train_eval(model, train_loader, val_loader, device=args.device, epochs=args.epochs, lr=args.lr)
    if best:
        os.makedirs(os.path.dirname(args.updrs_ckpt), exist_ok=True)
        torch.save({"model_state": best, "input_dim": 4096+4}, args.updrs_ckpt)
        print(f"Saved best regressor to {args.updrs_ckpt}")

if __name__ == "__main__": main()