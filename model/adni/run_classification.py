#!/usr/bin/env python3
"""Leakage-resistant paired fMRI/DTI classification on the 407-subject ADNI data."""

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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adni.atlas import AAL90_REGIONS
from model.adni.data import LABEL_NAMES, PairedADNIDataset, load_paired_connectomes, paired_collate
from model.adni.models import ADNIClassifier, ConnectivityEncoder, symmetric_contrastive_loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri_path", default=str(PROJECT_ROOT / "data" / "ADNI_fMRI.npy"))
    parser.add_argument("--dti_path", default=str(PROJECT_ROOT / "data" / "ADNI_DTI.npy"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "model" / "checkpoints" / "adni_cv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--classifier_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--contrastive_lr", type=float, default=3e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--encoder_dropout", type=float, default=0.15)
    parser.add_argument("--classifier_dropout", type=float, default=0.30)
    parser.add_argument("--edge_drop_prob", type=float, default=0.15)
    parser.add_argument("--feature_jitter", type=float, default=0.01)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(fmri, dti, labels, indices, args, shuffle=False, augment=False):
    dataset = PairedADNIDataset(
        fmri,
        dti,
        labels,
        indices=indices,
        augment=augment,
        edge_drop_prob=args.edge_drop_prob,
        feature_jitter=args.feature_jitter,
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=paired_collate,
        generator=generator,
        pin_memory=args.device.startswith("cuda"),
    )


def train_encoders(fmri, dti, labels, train_indices, args, device):
    encoders = {
        "fmri": ConnectivityEncoder(
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_layers=args.gnn_layers,
            dropout=args.encoder_dropout,
        ).to(device),
        "dti": ConnectivityEncoder(
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_layers=args.gnn_layers,
            dropout=args.encoder_dropout,
        ).to(device),
    }
    parameters = list(encoders["fmri"].parameters()) + list(encoders["dti"].parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.contrastive_lr, weight_decay=args.weight_decay)
    loader = make_loader(fmri, dti, labels, train_indices, args, shuffle=True, augment=True)

    for epoch in range(args.contrastive_epochs):
        for model in encoders.values():
            model.train()
        total = 0.0
        for batch in loader:
            z_fmri = encoders["fmri"](batch["fmri"].to(device))
            z_dti = encoders["dti"](batch["dti"].to(device))
            loss = symmetric_contrastive_loss(z_fmri, z_dti, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
            total += loss.item()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    contrastive {epoch + 1:03d}/{args.contrastive_epochs}: {total / max(len(loader), 1):.4f}")
    return encoders


def encode_indices(encoders, fmri, dti, labels, indices, args, device):
    loader = make_loader(fmri, dti, labels, indices, args, shuffle=False, augment=False)
    for model in encoders.values():
        model.eval()
    all_indices, all_features, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            z_fmri = encoders["fmri"](batch["fmri"].to(device))
            z_dti = encoders["dti"](batch["dti"].to(device))
            all_features.append(torch.cat([z_fmri, z_dti], dim=1).cpu())
            all_indices.append(batch["index"])
            all_labels.append(batch["label"])
    return torch.cat(all_indices), torch.cat(all_features), torch.cat(all_labels)


def classifier_metrics(logits, labels):
    probabilities = F.softmax(logits, dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    y_binary = label_binarize(truth, classes=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "macro_auc_ovr": float(roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")),
    }, predicted, probabilities


def train_classifier(train_x, train_y, val_x, val_y, args, device):
    model = ADNIClassifier(
        input_dim=train_x.size(1), hidden_dim=args.hidden_dim, dropout=args.classifier_dropout
    ).to(device)
    counts = Counter(train_y.tolist())
    weights = torch.tensor(
        [len(train_y) / (3 * counts[class_id]) for class_id in range(3)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state, best_score, stale = None, -float("inf"), 0
    for epoch in range(args.classifier_epochs):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device)).cpu()
        metrics, _, _ = classifier_metrics(val_logits, val_y)
        score = metrics["macro_f1"]
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_score


def save_manifest(path: Path, labels: np.ndarray):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "sample_id", "label", "diagnosis"])
        for index, label in enumerate(labels):
            writer.writerow([index, f"ADNI_{index:04d}", int(label), LABEL_NAMES[int(label)]])


def main():
    args = parse_args()
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("--val_fraction must be between 0 and 0.5.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmri, dti, labels = load_paired_connectomes(args.fmri_path, args.dti_path)
    save_manifest(output_dir / "adni_manifest.csv", labels)
    with (output_dir / "aal90_regions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["zero_based_index", "aal_index", "region"])
        for index, region in enumerate(AAL90_REGIONS):
            writer.writerow([index, index + 1, region])

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = requested

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions, fold_metrics, fold_splits = [], [], []
    all_indices = np.arange(len(labels))
    print(f"ADNI: {len(labels)} subjects | labels={dict(Counter(labels.tolist()))} | device={device}")

    for fold, (development_indices, test_indices) in enumerate(splitter.split(all_indices, labels), start=1):
        train_indices, val_indices = train_test_split(
            development_indices,
            test_size=args.val_fraction,
            random_state=args.seed + fold,
            stratify=labels[development_indices],
        )
        print(
            f"\nFold {fold}/{args.folds}: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )
        fold_splits.append({
            "fold": fold,
            "train_indices": train_indices.tolist(),
            "validation_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
        })
        set_seed(args.seed + fold)
        encoders = train_encoders(fmri, dti, labels, train_indices, args, device)
        _, train_x, train_y = encode_indices(encoders, fmri, dti, labels, train_indices, args, device)
        _, val_x, val_y = encode_indices(encoders, fmri, dti, labels, val_indices, args, device)
        test_ids, test_x, test_y = encode_indices(encoders, fmri, dti, labels, test_indices, args, device)

        classifier, best_val_f1 = train_classifier(train_x, train_y, val_x, val_y, args, device)
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_x.to(device)).cpu()
        metrics, predicted, probabilities = classifier_metrics(test_logits, test_y)
        metrics.update({"fold": fold, "best_validation_macro_f1": best_val_f1})
        fold_metrics.append(metrics)
        print("    test " + " | ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key not in {"fold"}))

        for sample_id, truth, pred, probability in zip(
            test_ids.tolist(), test_y.tolist(), predicted.tolist(), probabilities.tolist()
        ):
            predictions.append({
                "fold": fold,
                "sample_index": sample_id,
                "sample_id": f"ADNI_{sample_id:04d}",
                "true_label": truth,
                "true_diagnosis": LABEL_NAMES[truth],
                "predicted_label": pred,
                "predicted_diagnosis": LABEL_NAMES[pred],
                "prob_ad": probability[0],
                "prob_mci": probability[1],
                "prob_cn": probability[2],
            })

        torch.save(
            {
                "fmri_encoder": encoders["fmri"].state_dict(),
                "dti_encoder": encoders["dti"].state_dict(),
                "classifier": classifier.state_dict(),
                "fold": fold,
                "test_metrics": metrics,
                "label_names": LABEL_NAMES,
                "aal90_regions": AAL90_REGIONS,
                "train_indices": train_indices.tolist(),
                "validation_indices": val_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "config": vars(args),
            },
            output_dir / f"fold_{fold}.pt",
        )

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"]
    aggregate = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_metrics])),
            "std": float(np.std([fold[name] for fold in fold_metrics], ddof=1)),
        }
        for name in metric_names
    }
    summary = {
        "dataset": "ADNI paired AAL90 fMRI+DTI",
        "subjects": len(labels),
        "label_names": LABEL_NAMES,
        "fold_splits": fold_splits,
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
        "config": vars(args),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(sorted(predictions, key=lambda row: row["sample_index"]))

    print("\nCross-validation summary")
    for name, values in aggregate.items():
        print(f"  {name}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
