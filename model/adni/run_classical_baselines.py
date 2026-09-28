#!/usr/bin/env python3
"""Leakage-resistant nested-CV baselines for paired ADNI connectomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adni.data import load_paired_connectomes
from model.adni.tasks import TASKS, prepare_task_labels


MODELS = ("dummy", "logistic", "pca_logistic", "rbf_svm", "random_forest")
MODALITIES = ("fmri", "dti", "early_fusion")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri_path", default=str(PROJECT_ROOT / "data" / "ADNI_fMRI.npy"))
    parser.add_argument("--dti_path", default=str(PROJECT_ROOT / "data" / "ADNI_DTI.npy"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "model" / "checkpoints" / "adni_baselines"))
    parser.add_argument("--task", choices=["all", *sorted(TASKS)], default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def edge_features(matrices: np.ndarray, *, log1p: bool = False) -> np.ndarray:
    rows, cols = np.triu_indices(matrices.shape[1], k=1)
    features = matrices[:, rows, cols].astype(np.float64, copy=True)
    if log1p:
        features = np.log1p(np.clip(features, 0.0, None))
    return features


def feature_sets(fmri: np.ndarray, dti: np.ndarray) -> dict[str, np.ndarray]:
    fmri_edges = edge_features(fmri)
    dti_edges = edge_features(dti, log1p=True)
    return {
        "fmri": fmri_edges,
        "dti": dti_edges,
        "early_fusion": np.concatenate([fmri_edges, dti_edges], axis=1),
    }


def estimator_and_grid(name: str, seed: int):
    if name == "dummy":
        return DummyClassifier(strategy="prior"), {}
    if name == "logistic":
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ])
        return estimator, {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    if name == "pca_logistic":
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(svd_solver="full")),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ])
        return estimator, {
            "pca__n_components": [0.80, 0.90, 0.95],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        }
    if name == "rbf_svm":
        estimator = Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    SVC(class_weight="balanced", random_state=seed),
                    cv=3,
                    ensemble=False,
                ),
            ),
        ])
        return estimator, {
            "model__estimator__C": [0.1, 1.0, 10.0],
            "model__estimator__gamma": ["scale", 1e-4, 1e-3],
        }
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        return estimator, {
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 3, 5],
        }
    raise ValueError(name)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    predicted = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
    }
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        result.update({
            "roc_auc": float(roc_auc_score(y_true, probabilities[:, 1])),
            "pr_auc": float(average_precision_score(y_true, probabilities[:, 1])),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
        })
    else:
        binary = label_binarize(y_true, classes=np.arange(num_classes))
        result["macro_auc_ovr"] = float(
            roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
        )
    return result, predicted


def aligned_probabilities(estimator, features: np.ndarray, num_classes: int) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    aligned = np.zeros((len(features), num_classes), dtype=np.float64)
    aligned[:, np.asarray(estimator.classes_, dtype=int)] = probabilities
    return aligned


def run_task(task_name, all_features, source_labels, args):
    labels, eligible, spec = prepare_task_labels(source_labels, task_name)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows, predictions = [], []
    print(f"\nTask {task_name}: n={len(eligible)} labels={dict(Counter(labels[eligible].tolist()))}")

    for modality in args.modalities:
        x = all_features[modality]
        for model_name in args.models:
            print(f"  {modality}/{model_name}", flush=True)
            for fold, (train_pos, test_pos) in enumerate(
                outer.split(eligible, labels[eligible]), start=1
            ):
                train_ids, test_ids = eligible[train_pos], eligible[test_pos]
                inner = StratifiedKFold(
                    n_splits=args.inner_folds,
                    shuffle=True,
                    random_state=args.seed + fold,
                )
                estimator, grid = estimator_and_grid(model_name, args.seed + fold)
                if grid:
                    fitted = GridSearchCV(
                        estimator,
                        grid,
                        scoring="balanced_accuracy",
                        cv=inner,
                        n_jobs=args.n_jobs,
                        refit=True,
                        error_score="raise",
                    )
                else:
                    fitted = estimator
                fitted.fit(x[train_ids], labels[train_ids])
                probabilities = aligned_probabilities(fitted, x[test_ids], spec.num_classes)
                fold_metrics, predicted = metrics(labels[test_ids], probabilities)
                best_params = fitted.best_params_ if hasattr(fitted, "best_params_") else {}
                rows.append({
                    "task": task_name,
                    "modality": modality,
                    "model": model_name,
                    "fold": fold,
                    "train_subjects": len(train_ids),
                    "test_subjects": len(test_ids),
                    "best_params": best_params,
                    **fold_metrics,
                })
                print(
                    f"    fold {fold}: bal_acc={fold_metrics['balanced_accuracy']:.4f} "
                    f"macro_f1={fold_metrics['macro_f1']:.4f}",
                    flush=True,
                )
                for subject, truth, pred, prob in zip(
                    test_ids, labels[test_ids], predicted, probabilities
                ):
                    prediction = {
                        "task": task_name,
                        "modality": modality,
                        "model": model_name,
                        "fold": fold,
                        "sample_index": int(subject),
                        "true_label": int(truth),
                        "predicted_label": int(pred),
                    }
                    prediction.update({f"prob_class_{i}": float(value) for i, value in enumerate(prob)})
                    predictions.append(prediction)
    return rows, predictions


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["modality"], row["model"])].append(row)
    output = []
    ignored = {"task", "modality", "model", "fold", "train_subjects", "test_subjects", "best_params"}
    for (task, modality, model), fold_rows in sorted(groups.items()):
        result = {"task": task, "modality": modality, "model": model, "folds": len(fold_rows)}
        for name in fold_rows[0]:
            if name in ignored:
                continue
            values = np.asarray([row[name] for row in fold_rows], dtype=float)
            result[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmri, dti, source_labels = load_paired_connectomes(args.fmri_path, args.dti_path)
    features = feature_sets(fmri, dti)
    task_names = sorted(TASKS) if args.task == "all" else [args.task]

    all_rows, all_predictions = [], []
    for task_name in task_names:
        rows, predictions = run_task(task_name, features, source_labels, args)
        all_rows.extend(rows)
        all_predictions.extend(predictions)

    summary = {
        "protocol": "nested stratified cross-validation; all scaling, PCA, and tuning fit inside training folds",
        "config": vars(args),
        "aggregate": aggregate(all_rows),
        "fold_metrics": all_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(output_dir / "fold_metrics.csv", all_rows)
    write_csv(output_dir / "predictions.csv", all_predictions)
    print(f"\nArtifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
