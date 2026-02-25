import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
from collections import Counter
import argparse
import numpy as np

# --- NEW IMPORTS FOR METRICS ---
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

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
        # Load once and keep on CPU to save VRAM for the model
        data = torch.load(recon_demo_path, map_location="cpu")
        mod_order = ["SPECT", "MRI", "fMRI", "DTI"]
        
        for key, label in labels_dict.items():
            if key not in data: continue
            
            patient_entry = data[key] 
            hybrid_mods = patient_entry['recon']
            real_mods = patient_entry['real']
            
            # 1. Concatenate the 4 modalities
            feat = torch.cat([hybrid_mods[m].flatten() for m in mod_order])
            
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
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 4, 3)
        )
    def forward(self, x): return self.net(x)

def evaluate(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            logits = model(x)
            
            # Get probabilities for AUC
            prob = F.softmax(logits, dim=1)
            
            preds.append(torch.argmax(logits, dim=1))
            trues.append(y)
            probs.append(prob)
            
    if not preds: 
        return {"bal_acc": 0.0, "acc": 0.0, "f1_macro": 0.0, "auc_macro": 0.0}
        
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    
    # Calculate Sklearn Metrics
    bal_acc = balanced_accuracy_score(trues, preds)
    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average='macro')
    
    # Multi-class AUC (One-vs-Rest)
    try:
        auc_macro = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc_macro = 0.0 # Failsafe if a class is entirely missing from a tiny validation fold
        
    return {
        "bal_acc": bal_acc, 
        "acc": acc, 
        "f1_macro": f1_macro, 
        "auc_macro": auc_macro
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--classifier_ckpt', required=True)
    parser.add_argument('--split_path', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_mask', action='store_true', default=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    full_dataset = SmartClassDataset(args.embeddings_path, labels, use_mask=args.use_mask)
    
    train_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in train_ids]
    val_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in val_ids]
    
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_idx), 
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, val_idx), 
        batch_size=args.batch_size,
        num_workers=2, pin_memory=True
    )

    # 3. Handle Imbalance
    counts = Counter([full_dataset.samples[i][1] for i in train_idx])
    weights = torch.tensor([len(train_idx) / (3 * counts.get(i, 1)) for i in range(3)]).to(device)

    input_dim = full_dataset[0][0].shape[0] if len(full_dataset) > 0 else 4100
    
    print(f"🚀 Classifier Running on {device} | Input Dim: {input_dim}")
    
    model = Classifier(input_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # --- BUG FIX: Initialize negative so it is guaranteed to save ---
    best_bal_acc = -1.0
    best_state = None
    best_metrics = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        metrics = evaluate(model, val_loader, device)
        
        # Save based on best Balanced Accuracy
        if metrics['bal_acc'] > best_bal_acc:
            best_bal_acc = metrics['bal_acc']
            best_metrics = metrics  # Store all metrics for this absolute best epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Bal. Acc: {metrics['bal_acc']:.4f}")

    # --- PRINT RICH STATISTICS AT THE END ---
    print(f"\n✅ Stage 3 Complete.")
    print(f"Best Balanced Accuracy: {best_metrics.get('bal_acc', 0):.4f}")
    print(f"Standard Accuracy:      {best_metrics.get('acc', 0):.4f}")
    print(f"Macro F1-Score:         {best_metrics.get('f1_macro', 0):.4f}")
    print(f"Macro AUC (OvR):        {best_metrics.get('auc_macro', 0):.4f}")

    # Checkpoint Saving Logic
    if best_state is not None:
        os.makedirs(os.path.dirname(args.classifier_ckpt), exist_ok=True)
        torch.save({
            "model_state": best_state,
            "input_dim": input_dim,
            "best_bal_acc": best_bal_acc,
            "metrics": best_metrics # Optional: saves these to the .pt file too
        }, args.classifier_ckpt)
        print(f"✅ Saved best classifier weights to {args.classifier_ckpt}")

if __name__ == "__main__":
    main()