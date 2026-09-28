#!/usr/bin/env python3
"""Fold-local BrainNetCNN-style baselines for ADNI connectivity matrices."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adni.data import load_paired_connectomes
from model.adni.tasks import TASKS, prepare_task_labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri_path", default=str(PROJECT_ROOT / "data" / "ADNI_fMRI.npy"))
    parser.add_argument("--dti_path", default=str(PROJECT_ROOT / "data" / "ADNI_DTI.npy"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "model" / "checkpoints" / "adni_brainnetcnn"))
    parser.add_argument("--task", choices=sorted(TASKS), default="three_class")
    parser.add_argument("--modality", choices=["fmri", "dti", "early_fusion"], default="early_fusion")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--e2e_channels", type=int, default=16)
    parser.add_argument("--e2n_channels", type=int, default=32)
    parser.add_argument("--graph_channels", type=int, default=64)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EdgeToEdge(nn.Module):
    def __init__(self, in_channels, out_channels, nodes):
        super().__init__()
        self.nodes = nodes
        self.row = nn.Conv2d(in_channels, out_channels, kernel_size=(1, nodes))
        self.column = nn.Conv2d(in_channels, out_channels, kernel_size=(nodes, 1))

    def forward(self, matrix):
        row = self.row(matrix).expand(-1, -1, -1, self.nodes)
        column = self.column(matrix).expand(-1, -1, self.nodes, -1)
        return row + column


class BrainNetCNN(nn.Module):
    """Compact E2E/E2N/N2G connectome CNN following the BrainNetCNN design."""

    def __init__(
        self,
        in_channels,
        num_classes,
        nodes=90,
        e2e_channels=16,
        e2n_channels=32,
        graph_channels=64,
        dropout=0.3,
    ):
        super().__init__()
        self.e2e = EdgeToEdge(in_channels, e2e_channels, nodes)
        self.e2n = nn.Conv2d(e2e_channels, e2n_channels, kernel_size=(1, nodes))
        self.n2g = nn.Conv2d(e2n_channels, graph_channels, kernel_size=(nodes, 1))
        self.classifier = nn.Sequential(
            nn.Linear(graph_channels, graph_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_channels, num_classes),
        )

    def forward(self, matrix):
        x = F.leaky_relu(self.e2e(matrix), negative_slope=0.1)
        x = F.leaky_relu(self.e2n(x), negative_slope=0.1)
        x = F.leaky_relu(self.n2g(x), negative_slope=0.1)
        return self.classifier(x.flatten(1))


def stack_modalities(fmri, dti, modality):
    if modality == "fmri":
        return fmri[:, None]
    if modality == "dti":
        return np.log1p(np.clip(dti, 0.0, None))[:, None]
    transformed_dti = np.log1p(np.clip(dti, 0.0, None))
    return np.stack([fmri, transformed_dti], axis=1)


def normalize_from_training(matrices, train_indices):
    """Standardize each modality/edge using inner-training subjects only."""
    mean = matrices[train_indices].mean(axis=0, keepdims=True)
    std = matrices[train_indices].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((matrices - mean) / std).astype(np.float32), mean, std


def evaluate(logits, labels, num_classes):
    probabilities = F.softmax(logits, dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        result.update({
            "roc_auc": float(roc_auc_score(truth, probabilities[:, 1])),
            "pr_auc": float(average_precision_score(truth, probabilities[:, 1])),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
        })
    else:
        binary = label_binarize(truth, classes=np.arange(num_classes))
        result["macro_auc_ovr"] = float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        )
    return result, predicted, probabilities


def loader(matrices, labels, indices, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(matrices[indices]),
        torch.from_numpy(labels[indices]),
        torch.from_numpy(np.asarray(indices, dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def collect(model, data_loader, device):
    model.eval()
    logits, labels, indices = [], [], []
    with torch.no_grad():
        for matrices, targets, subject_ids in data_loader:
            logits.append(model(matrices.to(device)).cpu())
            labels.append(targets)
            indices.append(subject_ids)
    return torch.cat(logits), torch.cat(labels), torch.cat(indices)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri, dti, source_labels = load_paired_connectomes(args.fmri_path, args.dti_path)
    labels, eligible, task = prepare_task_labels(source_labels, args.task)
    raw_matrices = stack_modalities(fmri, dti, args.modality)
    requested = torch.device(args.device)
    device = torch.device("cpu") if requested.type == "cuda" and not torch.cuda.is_available() else requested
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, predictions = [], []
    print(
        f"BrainNetCNN task={task.name} modality={args.modality}: n={len(eligible)} "
        f"labels={dict(Counter(labels[eligible].tolist()))} device={device}"
    )

    for fold, (development_pos, test_pos) in enumerate(
        splitter.split(eligible, labels[eligible]), start=1
    ):
        development, test_indices = eligible[development_pos], eligible[test_pos]
        train_indices, val_indices = train_test_split(
            development,
            test_size=args.val_fraction,
            random_state=args.seed + fold,
            stratify=labels[development],
        )
        matrices, mean, std = normalize_from_training(raw_matrices, train_indices)
        set_seed(args.seed + fold)
        model = BrainNetCNN(
            in_channels=matrices.shape[1],
            num_classes=task.num_classes,
            e2e_channels=args.e2e_channels,
            e2n_channels=args.e2n_channels,
            graph_channels=args.graph_channels,
            dropout=args.dropout,
        ).to(device)
        counts = Counter(labels[train_indices].tolist())
        weights = torch.tensor(
            [len(train_indices) / (task.num_classes * counts[i]) for i in range(task.num_classes)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = loader(matrices, labels, train_indices, args.batch_size, True, args.seed + fold)
        val_loader = loader(matrices, labels, val_indices, args.batch_size, False, args.seed + fold)
        best_state, best_score, stale = None, -float("inf"), 0

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for batch_matrices, targets, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_matrices.to(device)), targets.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total += loss.item()
            val_logits, val_labels, _ = collect(model, val_loader, device)
            validation_metrics, _, _ = evaluate(val_logits, val_labels, task.num_classes)
            score = validation_metrics["macro_f1"]
            if score > best_score + 1e-6:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"  fold {fold} epoch {epoch + 1:03d}: "
                    f"loss={total / max(len(train_loader), 1):.4f} val_macro_f1={score:.4f}"
                )
            if stale >= args.patience:
                break

        model.load_state_dict(best_state)
        test_loader = loader(matrices, labels, test_indices, args.batch_size, False, args.seed + fold)
        test_logits, test_labels, test_ids = collect(model, test_loader, device)
        result, predicted, probabilities = evaluate(test_logits, test_labels, task.num_classes)
        result.update({"fold": fold, "best_validation_macro_f1": best_score})
        fold_metrics.append(result)
        print(
            f"  fold {fold} test bal_acc={result['balanced_accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f}"
        )
        for subject, truth, pred, probability in zip(
            test_ids.tolist(), test_labels.tolist(), predicted.tolist(), probabilities.tolist()
        ):
            row = {
                "fold": fold,
                "sample_index": subject,
                "true_label": truth,
                "predicted_label": pred,
            }
            row.update({f"prob_class_{i}": value for i, value in enumerate(probability)})
            predictions.append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "normalization_mean": torch.from_numpy(mean),
                "normalization_std": torch.from_numpy(std),
                "train_indices": train_indices.tolist(),
                "validation_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "task": task.name,
                "label_names": task.label_names,
                "config": vars(args),
                "test_metrics": result,
            },
            output_dir / f"fold_{fold}.pt",
        )

    metric_names = [key for key in fold_metrics[0] if key not in {"fold", "best_validation_macro_f1"}]
    aggregate = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_metrics])),
            "std": float(np.std([row[name] for row in fold_metrics], ddof=1)),
        }
        for name in metric_names
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "model": "BrainNetCNN-style E2E/E2N/N2G",
                "task": task.name,
                "modality": args.modality,
                "fold_metrics": fold_metrics,
                "aggregate": aggregate,
                "config": vars(args),
            },
            handle,
            indent=2,
        )
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(sorted(predictions, key=lambda row: row["sample_index"]))
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
