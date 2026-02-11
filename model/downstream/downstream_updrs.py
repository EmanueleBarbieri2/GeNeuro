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
        reader = csv.DictReader(f)
        for row in reader:
            patno = row.get("PATNO")
            event_id = row.get("EVENT_ID")
            if not patno or not event_id:
                continue
            key = f"{patno}_{event_id}"
            vals = []
            valid = True
            for t in TARGETS:
                v = row.get(t)
                if v is None or v == "":
                    valid = False
                    break
                try:
                    vals.append(float(v))
                except ValueError:
                    valid = False
                    break
            if valid:
                targets[key] = torch.tensor(vals, dtype=torch.float32)
    return targets


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


class UpdrsDataset(Dataset):
    def __init__(self, embeddings_by_id, targets_by_id, generator, device="cpu"):
        self.samples = []
        self.device = device
        self.generator = generator

        for key, target in targets_by_id.items():
            mods = embeddings_by_id.get(key)
            if not mods:
                continue

            available = {m: mods[m] for m in mods.keys()}
            if len(available) == 0:
                continue

            recon = hallucinate_missing_modalities(generator, available, device=device)

            # Build input: concatenated embeddings + mask
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

            self.samples.append((key, x, target))

        # Check for NaNs/Infs/extremes in targets
        all_targets = torch.stack([t for _, _, t in self.samples]) if self.samples else torch.empty(0)
        if all_targets.numel() > 0:
            print("[UpdrsDataset] Target stats:")
            print("  shape:", all_targets.shape)
            print("  min:", all_targets.min().item(), "max:", all_targets.max().item())
            print("  mean:", all_targets.mean().item(), "std:", all_targets.std().item())
            n_nan = torch.isnan(all_targets).sum().item()
            n_inf = torch.isinf(all_targets).sum().item()
            print(f"  NaNs: {n_nan}  Infs: {n_inf}")
            if n_nan > 0 or n_inf > 0:
                print("  [WARNING] NaNs or Infs detected in regression targets!")
            if all_targets.abs().max().item() > 1e6:
                print("  [WARNING] Extreme values detected in regression targets!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return (x, target) for DataLoader, but keep key for filtering
        _, x, target = self.samples[idx]
        return x, target


class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)


def train_eval(model, train_loader, val_loader, device="cpu", epochs=50):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            val_losses = []
            abs_err = []
            preds = []
            trues = []
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_losses.append(F.mse_loss(pred, y).item())
                abs_err.append(torch.abs(pred - y).mean(dim=0))
                preds.append(pred.cpu())
                trues.append(y.cpu())
            mae = torch.stack(abs_err).mean(dim=0) if abs_err else torch.zeros(4)
            if preds:
                preds = torch.cat(preds, dim=0)
                trues = torch.cat(trues, dim=0)
                mse = torch.mean((preds - trues) ** 2, dim=0)
                rmse = torch.sqrt(mse)
                var = torch.var(trues, dim=0, unbiased=False)
                r2 = 1 - mse / (var + 1e-8)
            else:
                rmse = torch.zeros(4)
                r2 = torch.zeros(4)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_mse={total/max(len(train_loader),1):.4f} | "
            f"val_mse={sum(val_losses)/max(len(val_losses),1):.4f} | "
            f"val_mae={mae.tolist()} | "
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


    targets = load_csv_targets(csv_path)
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

    dataset = UpdrsDataset(embeddings_by_id, targets, generator)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) < 10:
        raise RuntimeError("Not enough samples after joining embeddings with UPDRS targets.")

    # Filter samples by unified split
    sample_keys = [s[0] for s in dataset.samples]  # s[0] is now the subject key
    train_idx = [i for i, k in enumerate(sample_keys) if k in train_ids]
    val_idx = [i for i, k in enumerate(sample_keys) if k in val_ids]

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    input_dim = 4 * 1024 + 4
    model = Regressor(input_dim)

    train_eval(model, train_loader, val_loader, epochs=50)

    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    regressor_path = os.path.join(checkpoints_dir, 'updrs_regressor.pt')
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
        },
        regressor_path,
    )
    print(f"Saved regressor checkpoint to {regressor_path}")


if __name__ == "__main__":
    main()
