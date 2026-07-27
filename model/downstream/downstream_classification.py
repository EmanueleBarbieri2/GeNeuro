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

# --- IMPORTS FOR METRICS ---
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

# Default global for external imports (like explainers)
CLASS_NAMES = ["Control", "PD", "Prodromal"]

def load_csv_labels(csv_path, drop_prodromal=False):
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
            elif c == 4: 
                if drop_prodromal: continue
                label = "Prodromal"
            
            if label: labels[f"{patno}_{event_id}"] = label
    return labels

class SmartClassDataset(Dataset):
    # 🌟 ADDED: ablate_modality and ablate_ratio to init signature
    def __init__(self, embeddings_path, labels_dict, active_mods, class_names, use_mask=True, zero_impute=False, ablate_modality=None, ablate_ratio=None):
        self.samples = []
        data = torch.load(embeddings_path, map_location="cpu")
        
        is_raw = "embeddings" in data and "labels" in data and "ids" in data
        mod_to_idx = {mod: i for i, mod in enumerate(active_mods)}
        
        if is_raw:
            pt_data = {}
            for i, (emb, label, pid) in enumerate(zip(data["embeddings"], data["labels"], data["ids"])):
                if pid not in pt_data: pt_data[pid] = {}
                pt_data[pid][label] = emb

            for key, label in labels_dict.items():
                if key not in pt_data: continue
                
                patient_mods = pt_data[key]
                feat_list, mask_list = [], []
                
                for m in active_mods:
                    if m in patient_mods:
                        feat_list.append(patient_mods[m].flatten())
                        mask_list.append(0.0)
                    else:
                        feat_list.append(torch.zeros(1024))
                        mask_list.append(1.0)
                        
                feat = torch.cat(feat_list)
                
                # 🌟 DYNAMIC ABLATION LOGIC FOR RAW 🌟
                if ablate_modality in mod_to_idx and ablate_ratio is not None:
                    if np.random.rand() > ablate_ratio:
                        m_idx = mod_to_idx[ablate_modality]
                        feat[m_idx*1024 : (m_idx+1)*1024] = 0
                        mask_list[m_idx] = 1.0 # Set mask to missing
                
                x = torch.cat([feat, torch.tensor(mask_list)]) if use_mask else feat
                self.samples.append((key, class_names.index(label), x))
                
        else:
            for key, label in labels_dict.items():
                if key not in data: continue
                
                patient_entry = data[key] 
                hybrid_mods = patient_entry['recon']
                real_mods = patient_entry['real']
                
                feat_list = [hybrid_mods[m].flatten() for m in active_mods]
                feat = torch.cat(feat_list)
                mask_list = [1.0 if m not in real_mods else 0.0 for m in active_mods]
                
                # 🌟 DYNAMIC ABLATION LOGIC FOR GENERATED 🌟
                if ablate_modality in mod_to_idx and ablate_ratio is not None:
                    if np.random.rand() > ablate_ratio: # e.g., if rand > 0.1, we ZERO IT OUT (keeps 10%)
                        m_idx = mod_to_idx[ablate_modality]
                        feat[m_idx*1024 : (m_idx+1)*1024] = 0
                        mask_list[m_idx] = 1.0 # Set mask to missing
                
                if use_mask:
                    mask = torch.tensor(mask_list)
                    x = torch.cat([feat, mask])
                else:
                    x = feat
                    
                self.samples.append((key, class_names.index(label), x))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx][2], self.samples[idx][1]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5, num_classes=3):
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
            nn.Linear(hidden_dim // 4, num_classes) # <-- Dynamic classes
        )
    def forward(self, x): return self.net(x)

def evaluate(model, loader, device, num_classes):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            
            preds.append(torch.argmax(logits, dim=1))
            trues.append(y)
            probs.append(prob)
            
    if not preds: 
        return {"bal_acc": 0.0, "acc": 0.0, "f1_macro": 0.0, "auc_macro": 0.0}
        
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()
    probs = torch.cat(probs).cpu().numpy()
    
    bal_acc = balanced_accuracy_score(trues, preds)
    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average='macro')
    
    try:
        # Dynamic AUC based on binary vs multi-class
        if num_classes == 2:
            auc_macro = roc_auc_score(trues, probs[:, 1])
        else:
            auc_macro = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc_macro = 0.0 
        
    return {"bal_acc": bal_acc, "acc": acc, "f1_macro": f1_macro, "auc_macro": auc_macro}

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
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--use_mask', dest='use_mask', action='store_true')
    mask_group.add_argument(
        '--no_missingness_mask',
        dest='use_mask',
        action='store_false',
        help='Exclude observed/reconstructed modality indicators from the downstream input.',
    )
    parser.set_defaults(use_mask=True)
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--exclude_modality', nargs='+', default=None)
    parser.add_argument('--disable_generator', action='store_true')
    parser.add_argument('--drop_prodromal', action='store_true', help="Convert to Binary PD vs Control")
    
    # 🌟 ADDED: The required arguments to accept commands from the orchestrator
    parser.add_argument('--ablate_modality', type=str, choices=['fMRI', 'DTI', 'MRI', 'SPECT'])
    parser.add_argument('--ablate_ratio', type=float)
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    active_mods = ["SPECT", "MRI", "fMRI", "DTI"]
    if args.exclude_modality:
        active_mods = [m for m in active_mods if m not in args.exclude_modality]

    # --- Handle Class Logic ---
    if args.drop_prodromal:
        print("⚠️ ABLATION: Dropping 'Prodromal' class for Binary Classification.")
        current_class_names = ["Control", "PD"]
    else:
        current_class_names = ["Control", "PD", "Prodromal"]
    num_classes = len(current_class_names)

    train_ids, val_ids, test_ids = set(), set(), set()
    if os.path.exists(args.split_path):
        with open(args.split_path) as f:
            mode = None
            for line in f.read().splitlines():
                if 'train_ids' in line: mode = 'train'
                elif 'val_ids' in line: mode = 'val'
                elif 'test_ids' in line: mode = 'test'
                elif line.strip() and not line.startswith('#'):
                    if mode == 'train': train_ids.add(line)
                    elif mode == 'val': val_ids.add(line)
                    elif mode == 'test': test_ids.add(line)
    
    labels = load_csv_labels(args.csv_path, drop_prodromal=args.drop_prodromal)
    
    # 🌟 ADDED: Pass the ablate flags directly into your SmartClassDataset
    full_dataset = SmartClassDataset(
        args.embeddings_path, labels, active_mods, class_names=current_class_names,
        use_mask=args.use_mask, zero_impute=args.disable_generator,
        ablate_modality=args.ablate_modality, ablate_ratio=args.ablate_ratio
    )
    
    if len(full_dataset) == 0:
        print("❌ Error: Dataset is empty after filtering.")
        return

    train_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in train_ids]
    val_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in val_ids]
    test_idx = [i for i, s in enumerate(full_dataset.samples) if s[0] in test_ids]

    # 🌟 PRESERVED DIAGNOSTIC BLOCK 🌟
    train_labels = [current_class_names[full_dataset.samples[i][1]] for i in train_idx]
    val_labels = [current_class_names[full_dataset.samples[i][1]] for i in val_idx]
    test_labels = [current_class_names[full_dataset.samples[i][1]] for i in test_idx]
    
    print("\n📊 DATASET DIAGNOSTICS:")
    print(f"   Train Set: {len(train_idx)} visits -> {dict(Counter(train_labels))}")
    print(f"   Val Set:   {len(val_idx)} visits -> {dict(Counter(val_labels))}")
    if test_ids:
        print(f"   Test Set:  {len(test_idx)} visits -> {dict(Counter(test_labels))}")
    print()
    # 🌟 ------------------------ 🌟
    
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_idx), batch_size=args.batch_size)

    counts = Counter([full_dataset.samples[i][1] for i in train_idx])
    weights = torch.tensor([len(train_idx) / (num_classes * counts.get(i, 1)) for i in range(num_classes)]).to(device)

    input_dim = full_dataset[0][0].shape[0]
    print(
        f"🚀 Classifier Running on {device} | Input Dim: {input_dim} | "
        f"Classes: {num_classes} | Missingness mask: {'included' if args.use_mask else 'excluded'}"
    )
    
    model = Classifier(input_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_bal_acc = -1.0
    best_state, best_metrics = None, {}

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
        
        metrics = evaluate(model, val_loader, device, num_classes)
        
        if metrics['bal_acc'] > best_bal_acc:
            best_bal_acc = metrics['bal_acc']
            best_metrics = metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val Bal. Acc: {metrics['bal_acc']:.4f}")

    reported_metrics = best_metrics
    if test_idx and best_state is not None:
        model.load_state_dict(best_state)
        reported_metrics = evaluate(model, test_loader, device, num_classes)
        print("\n🧪 Untouched Test-Set Metrics:")
    else:
        print("\n✅ Stage 3 Complete (validation metrics; no test_ids partition supplied).")
    print(f"Balanced Accuracy: {reported_metrics.get('bal_acc', 0):.4f}")
    print(f"Standard Accuracy: {reported_metrics.get('acc', 0):.4f}")
    print(f"Macro F1-Score:    {reported_metrics.get('f1_macro', 0):.4f}")
    print(f"Macro AUC:         {reported_metrics.get('auc_macro', 0):.4f}")

    if best_state is not None:
        os.makedirs(os.path.dirname(args.classifier_ckpt), exist_ok=True)
        torch.save({
            "model_state": best_state,
            "input_dim": input_dim,
            "num_classes": num_classes,          
            "class_names": current_class_names,  
            "best_bal_acc": best_bal_acc,
            "validation_metrics": best_metrics,
            "metrics": reported_metrics,
            "evaluation_split": "test" if test_idx else "validation",
            "use_missingness_mask": args.use_mask,
        }, args.classifier_ckpt)
        print(f"✅ Saved weights to {args.classifier_ckpt}")

if __name__ == "__main__":
    main()
