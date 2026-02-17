import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
from collections import Counter

# Logic using Cohort codes (1=PD, 2=HC, 4=Prodromal)
CLASS_NAMES = ["Control", "PD", "Prodromal"]

def load_csv_labels(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patno, event_id = row.get("PATNO"), row.get("EVENT_ID")
            cohort_val = row.get("COHORT") or row.get("cohort")
            if not patno or not event_id or not cohort_val: continue
            
            try:
                c = int(float(cohort_val))
            except: continue
                
            label = None
            if c == 1: label = "PD"
            elif c == 2: label = "Control"
            elif c == 4: label = "Prodromal"
            
            if label: labels[f"{patno}_{event_id}"] = label
    return labels

class SmartClassDataset(Dataset):
    def __init__(self, recon_demo_path, labels_dict, use_mask=True):
        self.samples = []
        data = torch.load(recon_demo_path, map_location="cpu")
        mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
        
        for key, label in labels_dict.items():
            if key not in data: continue
            
            patient_entry = data[key] 
            hybrid_mods = patient_entry['recon']
            real_mods = patient_entry['real']
            
            # 1. Concatenate the 4 modalities
            feat = torch.cat([hybrid_mods[m] for m in mod_order])
            
            # 2. Add availability mask
            if use_mask:
                mask = torch.tensor([1.0 if m in real_mods else 0.0 for m in mod_order])
                x = torch.cat([feat, mask])
            else:
                x = feat
                
            self.samples.append((key, CLASS_NAMES.index(label), x))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx][2], self.samples[idx][1]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 4, 3)
        )
    def forward(self, x): return self.net(x)

def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(torch.argmax(model(x.to(device)), dim=1).cpu())
            trues.append(y)
    if not preds: return {"bal_acc": 0}
    preds, trues = torch.cat(preds), torch.cat(trues)
    
    conf = torch.zeros(3, 3, dtype=torch.int64)
    for t, p in zip(trues, preds): conf[t, p] += 1
    recall = torch.diag(conf).float() / conf.sum(dim=1).float().clamp_min(1)
    return {"bal_acc": recall.mean().item(), "conf": conf}

def main():
    import argparse
    import os
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True) # Point this to recon_demo.pt
    parser.add_argument('--classifier_ckpt', default='model/checkpoints/classifier.pt') # FIXED: Added checkpoint arg
    parser.add_argument('--split_path', default='data/unified_split.txt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_mask', action='store_true', default=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # 1. Load Split IDs
    train_ids, val_ids = set(), set()
    if os.path.exists(args.split_path):
        with open(args.split_path) as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif line.strip() and not line.startswith('#'):
                    if mode == 'train': train_ids.add(line)
                    else: val_ids.add(line)
    
    # 2. Load Data
    labels = load_csv_labels(args.csv_path)
    dataset = SmartClassDataset(args.embeddings_path, labels, use_mask=args.use_mask)
    
    train_idx = [i for i, s in enumerate(dataset.samples) if s[0] in train_ids]
    val_idx = [i for i, s in enumerate(dataset.samples) if s[0] in val_ids]
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=256)

    # 3. Handle Imbalance
    counts = Counter([dataset.samples[i][1] for i in train_idx])
    weights = torch.tensor([len(train_idx) / (3 * counts.get(i, 1)) for i in range(3)]).to(args.device)

    # 4. Initialize Model (Dynamic Dimension)
    if len(dataset) > 0:
        input_dim = dataset[0][0].shape[0]
    else:
        input_dim = 4096 + (4 if args.use_mask else 0)

    print(f"🚀 Classifier initialized with Input Dim: {input_dim}")
    model = Classifier(input_dim, dropout=args.dropout).to(args.device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_bal_acc = 0.0
    best_state = None # FIXED: Properly initialize best_state

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            opt.step()
        
        metrics = evaluate(model, val_loader, args.device)
        if metrics['bal_acc'] > best_bal_acc:
            best_bal_acc = metrics['bal_acc']
            # FIXED: Actually extract and clone the PyTorch weights
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1:02d} | Val Bal. Acc: {metrics['bal_acc']:.4f}")

    # FIXED: Save the correct dictionary to the correct file path
    if best_state is not None:
        os.makedirs(os.path.dirname(args.classifier_ckpt), exist_ok=True)
        torch.save({
            "model_state": best_state, 
            "input_dim": input_dim,        # Crucial for the explainer!
            "best_bal_acc": best_bal_acc
        }, args.classifier_ckpt)
        print(f"✅ Saved Best Classifier to {args.classifier_ckpt}")

if __name__ == "__main__":
    main()