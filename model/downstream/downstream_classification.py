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
        for key, label in labels_by_id.items():
            mods = embeddings_by_id.get(key)
            if not mods:
                continue

            available = {m: mods[m] for m in mods.keys()}
            if len(available) == 0:
                continue

            try:
                recon = hallucinate_missing_modalities(generator, available, device=device)
            except Exception as e:
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
            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, y, x = self.samples[idx]
        return x, y

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        # Integrated Architecture: BatchNorm and Dropout to handle overfitting
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
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
    num_classes = len(CLASS_NAMES)
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(trues, preds):
        conf[t, p] += 1
        
    recall = torch.diag(conf).float() / (conf.sum(dim=1).float().clamp_min(1))
    precision = torch.diag(conf).float() / (conf.sum(dim=0).float().clamp_min(1))
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    
    return {
        "acc": acc,
        "bal_acc": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "conf": conf,
    }

def main():
    random.seed(42)
    torch.manual_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    args, unknown = parser.parse_known_args()

    # 1. Load generator
    generator = ProM3E_Generator(embed_dim=1024)
    if os.path.exists(args.generator_ckpt):
        ckpt = torch.load(args.generator_ckpt, map_location="cpu")
        generator.load_state_dict(ckpt["model_state"])
        print(f"Loaded generator checkpoint from {args.generator_ckpt}")

    # 2. Load data
    labels_dict = load_csv_labels(args.csv_path)
    embeddings_by_id = load_embeddings(args.embeddings_path)
    dataset = ClassDataset(embeddings_by_id, labels_dict, generator)
    
    # 3. Handle splits: Correct directory logic
    split_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'unified_split.txt'))
    train_ids, val_ids = set(), set()
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            mode = None
            for line in f:
                line = line.strip()
                if line == 'train_ids:': mode = 'train'; continue
                if line == 'val_ids:': mode = 'val'; continue
                if line and not line.startswith('#'):
                    if mode == 'train': train_ids.add(line)
                    elif mode == 'val': val_ids.add(line)
    
    sample_keys = [s[0] for s in dataset.samples]
    train_idx = [i for i, k in enumerate(sample_keys) if k in train_ids]
    val_idx = [i for i, k in enumerate(sample_keys) if k in val_ids]
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=256, shuffle=False)

    # 4. Class Weights for Balanced Accuracy
    train_labels = [dataset.samples[i][1] for i in train_idx]
    counts = Counter(train_labels)
    total_train = len(train_labels)
    num_classes = len(CLASS_NAMES)
    weights = torch.tensor([total_train / (num_classes * counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float32)
    print(f"Computed Class Weights: {weights.tolist()}")

    model = Classifier(4 * 1024 + 4)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_bal_acc = 0.0
    best_model_state = None
    
    # 5. Training Loop with exact original print formatting
    print("Starting Improved Balanced Training...")
    for epoch in range(30):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_train_loss += loss.item()

        metrics = evaluate(model, val_loader)
        curr_bal_acc = metrics.get('bal_acc', 0)
        
        # Best-state selection
        if curr_bal_acc > best_bal_acc:
            best_bal_acc = curr_bal_acc
            best_model_state = model.state_dict()
            status = " Best"
        else:
            status = ""

        # Print all metrics from your original script
        print(f"Epoch {epoch+1}/30 | train_loss={total_train_loss/len(train_loader):.4f} | "
              f"val_acc={metrics.get('acc',0):.4f} | val_bal_acc={curr_bal_acc:.4f} | "
              f"val_macro_f1={metrics.get('macro_f1',0):.4f} {status}")

    # 6. Correct Checkpoint Saving to standard directory
    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    classifier_path = os.path.join(checkpoints_dir, 'classifier.pt')
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save({"model_state": best_model_state, "input_dim": (4*1024+4)}, classifier_path)
        print(f"Saved best classifier checkpoint to {classifier_path}")

    final_metrics = evaluate(model, val_loader)
    if final_metrics.get("conf") is not None:
        print("Final Confusion Matrix (rows=true, cols=pred):")
        print(final_metrics["conf"])

if __name__ == "__main__":
    main()