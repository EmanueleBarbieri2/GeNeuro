import csv
import os
import random
import sys
from collections import defaultdict, Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Robust import: ensure 'model' is importable both directly and via pipeline
try:
    from model.generator.generator import ProM3E_Generator
except ModuleNotFoundError:
    # Add workspace root to sys.path if not already present
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if workspace_root not in sys.path:
        sys.path.insert(0, workspace_root)
    from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]
CLASS_NAMES = ["Control", "PD", "Prodromal", "Non-PD"]
PD_GENES = ["LRRK2", "GBA", "SNCA", "PRKN", "PINK1", "PARK7", "VPS35"]


def label_from_subgroup(subgroup: str):
    if not subgroup:
        return None
    s = subgroup.upper()
    if "SWEDD" in s:
        return "Non-PD"
    if "HEALTHY CONTROL" in s or "NORMOSMIC" in s:
        return "Control"
    if "SPORADIC PD" in s:
        return "PD"
    for g in PD_GENES:
        if g in s:
            return "PD"
    if "RBD" in s or "HYPOSMIA" in s:
        return "Prodromal"
    return None


def load_csv_labels(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno = row.get("PATNO")
            event_id = row.get("EVENT_ID")
            subgroup = row.get("subgroup")
            if not patno or not event_id:
                continue
            label = label_from_subgroup(subgroup)
            if label is None:
                continue
            key = f"{patno}_{event_id}"
            labels[key] = label
    return labels


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


class ClassDataset(Dataset):
    def __init__(self, embeddings_by_id, labels_by_id, generator, device="cpu"):
        self.samples = []
        self.device = device
        self.generator = generator



        added_count = 0
        skipped_count = 0
        for key, label in labels_by_id.items():
            mods = embeddings_by_id.get(key)
            if not mods:
                if skipped_count < 10:

                    skipped_count += 1
                continue

            available = {m: mods[m] for m in mods.keys()}
            if len(available) == 0:
                if skipped_count < 10:

                    skipped_count += 1
                continue

            try:
                recon = hallucinate_missing_modalities(generator, available, device=device)
            except Exception as e:
                if skipped_count < 10:

                    skipped_count += 1
                continue

            feat = []
            mask_feat = []
            for mod in MOD_ORDER:
                value = available.get(mod, None)
                if value is not None:
                    feat.append(value)
                    mask_feat.append(1.0)
                else:
                    feat.append(recon[mod])
                    mask_feat.append(0.0)
            try:
                x = torch.cat(feat, dim=0)
                x = torch.cat([x, torch.tensor(mask_feat, dtype=torch.float32)], dim=0)
                y = CLASS_NAMES.index(label)
                self.samples.append((key, y, x))
                if added_count < 10:

                    added_count += 1
            except Exception as e:
                if skipped_count < 10:

                    skipped_count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return (x, y) for DataLoader, but keep key for filtering
        _, y, x = self.samples[idx]
        return x, y


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device="cpu"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.append(pred)
            trues.append(y)
    if not preds:
        return {}
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    acc = (preds == trues).float().mean().item()

    # Balanced accuracy and macro F1
    num_classes = len(CLASS_NAMES)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(trues, preds):
        conf[t, p] += 1
    recall = torch.diag(conf).float() / (conf.sum(dim=1).float().clamp_min(1))
    precision = torch.diag(conf).float() / (conf.sum(dim=0).float().clamp_min(1))
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    bal_acc = recall.mean().item()
    macro_f1 = f1.mean().item()

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "conf": conf,
    }


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


    generator = ProM3E_Generator(embed_dim=1024)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            generator.load_state_dict(ckpt["model_state"])
            print(f"Loaded generator checkpoint from {ckpt_path}")

    # 1. Load generator
    generator = ProM3E_Generator(embed_dim=1024)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            generator.load_state_dict(ckpt["model_state"])
            print(f"Loaded generator checkpoint from {ckpt_path}")

    # 2. Load labels and embeddings
    labels = load_csv_labels(csv_path)
    embeddings_by_id = load_embeddings(embeddings_path)
    dataset = ClassDataset(embeddings_by_id, labels, generator)
    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_ids, val_ids = set(), set()
    with open(split_path, 'r') as f:
        mode = None
        for line in f:
            line = line.strip()
            if line == 'train_ids:':
                mode = 'train'
                continue
            if line == 'val_ids:':
                mode = 'val'
                continue
            if not line or line.startswith('#'):
                continue
            if mode == 'train':
                train_ids.add(line)
            if mode == 'val':
                val_ids.add(line)


    # 4. Print embeddings debug info
    emb_data = torch.load(args.embeddings_path, map_location='cpu')
    emb_ids = emb_data['ids']


    # 5. Print CSV debug info
    df = pd.read_csv(args.csv_path)
    if 'PATNO' in df.columns and 'EVENT_ID' in df.columns:
        csv_ids = set(f"{int(row['PATNO'])}_{row['EVENT_ID']}" for _, row in df.iterrows())
        csv_ids_list = [f"{int(row['PATNO'])}_{row['EVENT_ID']}" for _, row in df.head(10).iterrows()]
    else:
        csv_ids = set()
        csv_ids_list = []


    # 6. Filter samples by unified split
    sample_keys = [s[0] for s in dataset.samples]

    train_idx = [i for i, k in enumerate(sample_keys) if k in train_ids]
    val_idx = [i for i, k in enumerate(sample_keys) if k in val_ids]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    # Define DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)



    # Fix: define model before optimizer
    input_dim = 4 * 1024 + 4
    model = Classifier(input_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        metrics = evaluate(model, val_loader)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={total/max(len(train_loader),1):.4f} | "
            f"val_acc={metrics.get('acc',0):.4f} | "
            f"val_bal_acc={metrics.get('bal_acc',0):.4f} | "
            f"val_macro_f1={metrics.get('macro_f1',0):.4f}"
        )


    # Save classifier checkpoint
    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    classifier_path = os.path.join(checkpoints_dir, 'classifier.pt')
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
        },
        classifier_path,
    )
    print(f"Saved classifier checkpoint to {classifier_path}")

    if metrics.get("conf") is not None:
        print("Confusion matrix (rows=true, cols=pred):")
        print(metrics["conf"])



if __name__ == "__main__":
    main()
