import csv
import os
import random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.generator.generator import ProM3E_Generator

MOD_ORDER = ["SPECT", "MRI", "fMRI", "DTI"]

# CHANGED: Removed "Non-PD" (SWEDD) from the list.
CLASS_NAMES = ["Control", "PD", "Prodromal"]

# CHANGED: New logic using Cohort codes (1=PD, 2=HC, 4=Prodromal)
def label_from_cohort(cohort_val):
    if cohort_val is None: return None
    try:
        # Handle cases where cohort might be "1.0" string or integer
        c = int(float(cohort_val))
    except (ValueError, TypeError):
        return None

    if c == 1: return "PD"
    if c == 2: return "Control"
    if c == 4: return "Prodromal"
    return None # Returns None for everything else (SWEDD, etc), effectively excluding them

# CHANGED: Now looks for COHORT column instead of subgroup
def load_csv_labels(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Auto-detect column name case (COHORT vs cohort)
        keys = reader.fieldnames
        cohort_key = "COHORT" if "COHORT" in keys else "cohort"

        for row in reader:
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            cohort_val = row.get(cohort_key)
            
            if not patno or not event_id: continue
            
            label = label_from_cohort(cohort_val)
            if label: labels[f"{patno}_{event_id}"] = label
    return labels

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

class ClassDataset(Dataset):
    def __init__(self, embeddings_by_id, labels_dict, generator, device="cpu"):
        self.samples = []
        for key, label in labels_dict.items():
            mods = embeddings_by_id.get(key)
            if not mods or len(mods) == 0: continue
            recon = hallucinate_missing_modalities(generator, mods, device)
            feat, mask_feat = [], []
            for mod in MOD_ORDER:
                if mod in mods: feat.append(mods[mod]); mask_feat.append(1.0)
                else: feat.append(recon[mod]); mask_feat.append(0.0)
            x = torch.cat([torch.cat(feat), torch.tensor(mask_feat)])
            self.samples.append((key, CLASS_NAMES.index(label), x))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx][2], self.samples[idx][1]

class Classifier(nn.Module):
    # CHANGED: Default num_classes=3, matching CLASS_NAMES length
    def __init__(self, input_dim=4100, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(torch.argmax(model(x.to(device)), dim=1).cpu()); trues.append(y)
    if not preds: return {}
    preds, trues = torch.cat(preds), torch.cat(trues)
    
    acc = (preds == trues).float().mean().item()
    # CHANGED: Dynamic size for confusion matrix
    conf = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    for t, p in zip(trues, preds): conf[t, p] += 1
    recall = torch.diag(conf).float() / conf.sum(dim=1).float().clamp_min(1)
    precision = torch.diag(conf).float() / conf.sum(dim=0).float().clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"acc": acc, "bal_acc": recall.mean().item(), "macro_f1": f1.mean().item(), "conf": conf}

def main():
    random.seed(42); torch.manual_seed(42)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--generator_ckpt', required=True)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()

    labels = load_csv_labels(args.csv_path)
    print(f"Loaded labels for {len(labels)} subjects/events.")
    
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

    dataset = ClassDataset(embs, labels, gen, args.device)
    train_idx = [i for i, s in enumerate(dataset.samples) if s[0] in train_ids]
    val_idx = [i for i, s in enumerate(dataset.samples) if s[0] in val_ids]
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=256)

    # CHANGED: Dynamic weight calculation based on len(CLASS_NAMES)
    num_classes = len(CLASS_NAMES)
    counts = Counter([dataset.samples[i][1] for i in train_idx])
    weights = torch.tensor([len(train_idx) / (num_classes * counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float32).to(args.device)

    # CHANGED: Initialize model with correct number of classes (3)
    model = Classifier(4096+4, num_classes=num_classes).to(args.device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_bal_acc, best_state = 0.0, None
    for epoch in range(30):
        model.train(); total_loss = 0.0
        for x, y in train_loader:
            loss = criterion(model(x.to(args.device)), y.to(args.device))
            opt.zero_grad(); loss.backward(); opt.step(); total_loss += loss.item()
        
        metrics = evaluate(model, val_loader, args.device)
        status = "⭐ Best!" if metrics['bal_acc'] > best_bal_acc else ""
        if status: best_bal_acc = metrics['bal_acc']; best_state = model.state_dict()
        print(f"Epoch {epoch+1:02d}/30 | train_loss={total_loss/len(train_loader):.4f} | val_acc={metrics['acc']:.4f} | val_bal_acc={metrics['bal_acc']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} {status}")

    checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save({"model_state": best_state, "input_dim": 4096+4}, os.path.join(checkpoints_dir, 'classifier.pt'))
    
    model.load_state_dict(best_state)
    print("Final Confusion Matrix:\n", evaluate(model, val_loader, args.device)['conf'])

if __name__ == "__main__": main()