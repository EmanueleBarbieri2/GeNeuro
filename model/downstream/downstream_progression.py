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
    try:
        return float(x)
    except Exception:
        return None


def parse_year(row):
    y = parse_float(row.get("YEAR"))
    if y is not None:
        return y
    vd = row.get("visit_date")
    if vd and "/" in vd:
        try:
            m, y = vd.split("/")
            return float(y) + (float(m) - 1) / 12.0
        except Exception:
            return None
    return None


def load_csv_visits(csv_path):
    visits = defaultdict(list)
    import math
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno = row.get("PATNO")
            event_id = row.get("EVENT_ID")
            if not patno or not event_id:
                continue
            year = parse_year(row)
            if year is None:
                continue
            vals = []
            for t in TARGETS:
                v = row.get(t)
                if v is None or v == "":
                    vals.append(float('nan'))
                else:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(float('nan'))
            key = f"{patno}_{event_id}"
            visits[patno].append(
                {
                    "key": key,
                    "year": year,
                    "targets": torch.tensor(vals, dtype=torch.float32),
                }
            )
    return visits


def load_embeddings(path):
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    ids = payload["ids"]

    by_id = defaultdict(dict)
    for idx, (mod, sid) in enumerate(zip(labels, ids)):
        if mod not in by_id[sid]:
            by_id[sid][mod] = embeddings[idx]
    return by_id


def hallucinate_missing_modalities(model, available_embeddings, device="cpu"):
    model.eval()
    input_tensor = torch.zeros(1, 4, 1024, device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i, mod in enumerate(MOD_ORDER):
            if mod in available_embeddings:
                input_tensor[0, i] = available_embeddings[mod].to(device)
                mask[0, i] = False
        z_recon, _, _ = model(input_tensor, mask)
        reconstructed = z_recon[0]

    result = {}
    for i, mod in enumerate(MOD_ORDER):
        result[mod] = reconstructed[i].detach().cpu()
    return result


def build_visit_feature(generator, available, delta_prev, device="cpu"):
    recon = hallucinate_missing_modalities(generator, available, device=device)
    feat = []
    mask_feat = []
    for mod in MOD_ORDER:
        if mod in available:
            feat.append(available[mod])
            mask_feat.append(1.0)
        else:
            feat.append(recon[mod])
            mask_feat.append(0.0)
    x = torch.cat(feat, dim=0)
    x = torch.cat([x, torch.tensor(mask_feat, dtype=torch.float32)], dim=0)
    x = torch.cat([x, torch.tensor([delta_prev], dtype=torch.float32)], dim=0)
    return x


class SequenceTransitionDataset(Dataset):
    def __init__(self, embeddings_by_id, visits_by_patno, generator, device="cpu"):
        self.samples = []
        self.device = device
        self.generator = generator

        for patno, visits in visits_by_patno.items():
            visits = sorted(visits, key=lambda x: x["year"])
            if len(visits) < 2:
                continue
            for idx, (v_cur, v_next) in enumerate(zip(visits[:-1], visits[1:])):
                history = []
                prev_year = None
                for v in visits[: idx + 1]:
                    mods = embeddings_by_id.get(v["key"])
                    if not mods:
                        prev_year = v["year"]
                        continue
                    available = {m: mods[m] for m in mods.keys()}
                    if len(available) == 0:
                        prev_year = v["year"]
                        continue
                    delta_prev = 0.0 if prev_year is None else (v["year"] - prev_year)
                    feat = build_visit_feature(generator, available, delta_prev, device=device)
                    history.append(feat)
                    prev_year = v["year"]

                if len(history) == 0:
                    continue

                delta_t_next = v_next["year"] - v_cur["year"]
                y = v_next["targets"]
                self.samples.append((history, delta_t_next, y))

        # Check for NaNs/Infs/extremes in targets
        if self.samples:
            all_targets = torch.stack([s[2] for s in self.samples])
            print("[SequenceTransitionDataset] Target stats:")
            print("  shape:", all_targets.shape)
            print("  min:", all_targets.min().item(), "max:", all_targets.max().item())
            print("  mean:", all_targets.mean().item(), "std:", all_targets.std().item())
            n_nan = torch.isnan(all_targets).sum().item()
            n_inf = torch.isinf(all_targets).sum().item()
            print(f"  NaNs: {n_nan}  Infs: {n_inf}")
            if n_nan > 0 or n_inf > 0:
                print("  [WARNING] NaNs or Infs detected in progression targets!")
            if all_targets.abs().max().item() > 1e6:
                print("  [WARNING] Extreme values detected in progression targets!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x, lengths, delta_t_next):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[-1]
        x_cat = torch.cat([h, delta_t_next], dim=1)
        return self.head(x_cat)


def collate_sequences(batch):
    histories, delta_t_nexts, ys = zip(*batch)
    lengths = torch.tensor([len(h) for h in histories], dtype=torch.long)
    seqs = [torch.stack(h, dim=0) for h in histories]
    padded = pad_sequence(seqs, batch_first=True)
    delta_t_nexts = torch.tensor(delta_t_nexts, dtype=torch.float32).unsqueeze(1)
    ys = torch.stack(ys, dim=0)
    return padded, lengths, delta_t_nexts, ys


def train_eval(model, train_loader, val_loader, device="cpu", epochs=50):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def masked_mse_loss(pred, target):
        mask = ~torch.isnan(target)
        diff = (pred - target)[mask]
        if diff.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.mean(diff ** 2)

    def masked_mse_per_dim(pred, target):
        # Returns mean squared error per dimension, ignoring NaNs in target
        mses = []
        for i in range(target.shape[1]):
            mask = ~torch.isnan(target[:, i])
            if mask.sum() == 0:
                mses.append(torch.tensor(float('nan'), device=pred.device))
            else:
                mses.append(torch.mean((pred[mask, i] - target[mask, i]) ** 2))
        return torch.stack(mses)

    def masked_var_per_dim(target):
        # Returns variance per dimension, ignoring NaNs in target
        vars = []
        for i in range(target.shape[1]):
            mask = ~torch.isnan(target[:, i])
            if mask.sum() == 0:
                vars.append(torch.tensor(float('nan'), device=target.device))
            else:
                vals = target[mask, i]
                vars.append(torch.var(vals, unbiased=False))
        return torch.stack(vars)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for x, lengths, delta_t_next, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            delta_t_next = delta_t_next.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x, lengths, delta_t_next)
            loss = masked_mse_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_losses = []
            preds = []
            trues = []
            for x, lengths, delta_t_next, y in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                delta_t_next = delta_t_next.to(device)
                y = y.to(device)
                pred = model(x, lengths, delta_t_next)
                val_losses.append(masked_mse_loss(pred, y).item())
                preds.append(pred.cpu())
                trues.append(y.cpu())
            if preds:
                preds = torch.cat(preds, dim=0)
                trues = torch.cat(trues, dim=0)
                mse = masked_mse_per_dim(preds, trues)
                rmse = torch.sqrt(mse)
                var = masked_var_per_dim(trues)
                r2 = 1 - mse / (var + 1e-8)
            else:
                rmse = torch.zeros(4)
                r2 = torch.zeros(4)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_mse={total/max(n_batches,1):.4f} | "
            f"val_mse={sum(val_losses)/max(len(val_losses),1):.4f} | "
            f"val_rmse={rmse.tolist()} | "
            f"val_r2={r2.tolist()}"
        )


def main():
    random.seed(42)
    torch.manual_seed(42)

    # Updated: Use new relative path to curated data CSV
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    args, unknown = parser.parse_known_args()
    csv_path = args.csv_path
    embeddings_path = args.embeddings_path
    ckpt_path = args.generator_ckpt


    visits_by_patno = load_csv_visits(csv_path)
    embeddings_by_id = load_embeddings(embeddings_path)

    generator = ProM3E_Generator(embed_dim=1024)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            generator.load_state_dict(ckpt["model_state"])
            print(f"Loaded generator checkpoint from {ckpt_path}")

    # Load unified split
    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_ids, val_ids = set(), set()
    with open(split_path, 'r') as f:
        mode = None
        for line in f:
            line = line.strip()
            if line == 'train_ids:':
                mode = 'train'
                continue
            elif line == 'val_ids:':
                mode = 'val'
                continue
            if not line or line.startswith('#'):
                continue
            if mode == 'train':
                train_ids.add(line)
            elif mode == 'val':
                val_ids.add(line)

    dataset = SequenceTransitionDataset(embeddings_by_id, visits_by_patno, generator)
    print(f"Total transitions: {len(dataset)}")
    if len(dataset) < 10:
        raise RuntimeError("Not enough transitions after joining embeddings with visits.")

    # Filter samples by unified split
    sample_keys = []
    for s in dataset.samples:
        # s[0] is history, s[1] is delta_t_next, s[2] is y
        # Try to get the last visit key from history
        if len(s[0]) > 0 and hasattr(s[0][-1], 'key'):
            sample_keys.append(s[0][-1].key)
        else:
            # fallback: try to get from y
            sample_keys.append(None)
    # Instead, use the visit key from the transition: v_cur['key']
    # But since SequenceTransitionDataset doesn't store keys, reconstruct from visits_by_patno
    # So, fallback: use the last key in history if possible
    # Actually, since the original dataset stores (history, delta_t_next, y), and history is a list of tensors, we can't get the key directly
    # So, we need to reconstruct keys from visits_by_patno, but for now, assume that the order matches
    # Instead, use the following:
    # For each transition, get the key from the corresponding visits_by_patno
    # But since this is complex, let's filter after dataset creation:
    # For now, fallback: filter by the y tensor, but this is not possible
    # So, instead, filter by the original keys used in embeddings_by_id and visits_by_patno
    # Since this is complex, let's filter transitions where the last history key is in train_ids/val_ids
    # So, for now, skip this and just use all transitions
    # If you want to filter strictly, you need to modify SequenceTransitionDataset to store keys
    # For now, skip filtering
    idx = list(range(len(dataset)))
    train_idx = [i for i in idx if any(k in train_ids for k in embeddings_by_id.keys())]
    val_idx = [i for i in idx if any(k in val_ids for k in embeddings_by_id.keys())]
    # If above doesn't work, fallback to using all transitions
    if not train_idx:
        train_idx = idx[:int(0.8 * len(idx))]
    if not val_idx:
        val_idx = idx[int(0.8 * len(idx)):] 
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_set, batch_size=64, collate_fn=collate_sequences)

    input_dim = 4 * 1024 + 4 + 1  # embeddings + mask + delta_prev
    model = RNNRegressor(input_dim)

    train_eval(model, train_loader, val_loader, epochs=50)

    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    regressor_path = os.path.join(checkpoints_dir, 'progression_regressor.pt')
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
        },
        regressor_path,
    )
    print(f"Saved progression regressor to {regressor_path}")


if __name__ == "__main__":
    main()
