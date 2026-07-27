#!/usr/bin/env python3
"""Aggregate paired site-held-out results with and without explicit missingness masks."""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch


CHECKPOINTS = {
    "classification": "classifier.pt",
    "progression_u2": "prog_U2_ADL.pt",
    "progression_u3": "prog_U3_Motor.pt",
    "static_u2": "static_U2_ADL.pt",
    "static_u3": "static_U3_Motor.pt",
}

TASK_LABELS = {
    "classification": "Classification",
    "progression_u2": "Progression UPDRS-II",
    "progression_u3": "Progression UPDRS-III",
    "static_u2": "Static UPDRS-II",
    "static_u3": "Static UPDRS-III",
}

PRIMARY_METRICS = {
    "classification": ("bal_acc", "Balanced accuracy"),
    "progression_u2": ("r2", "$R^2$"),
    "progression_u3": ("r2", "$R^2$"),
    "static_u2": ("r2", "$R^2$"),
    "static_u3": ("r2", "$R^2$"),
}

CONDITIONS = ("with_mask", "without_mask")


def _patient_id(visit_id):
    return visit_id.split("_", 1)[0]


def _validate_site_split(path):
    partitions = {"train": set(), "val": set(), "test": set()}
    mode = None
    comments = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#"):
                comments.append(line)
            elif line in {"train_ids:", "val_ids:", "test_ids:"}:
                mode = line.split("_", 1)[0]
            elif line:
                if mode is None:
                    raise ValueError(f"ID before partition header in {path}: {line}")
                partitions[mode].add(line)
    patients = {
        partition: {_patient_id(visit_id) for visit_id in ids}
        for partition, ids in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[left] & patients[right]
        if overlap:
            raise ValueError(
                f"Patient leakage in {path}: {len(overlap)} patients overlap {left}/{right}."
            )
    if not all(partitions.values()):
        raise ValueError(f"Expected non-empty train, val, and test partitions in {path}.")
    if not any("site" in comment.lower() for comment in comments):
        raise ValueError(f"Split does not identify itself as site-grouped: {path}")
    return {
        partition: {"visits": len(ids), "patients": len(patients[partition])}
        for partition, ids in partitions.items()
    }


def _load_metrics(path, condition):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"Checkpoint has no metrics dictionary: {path}")
    if payload.get("evaluation_split") != "test":
        raise ValueError(
            f"Expected untouched test metrics in {path}, "
            f"found evaluation_split={payload.get('evaluation_split')!r}."
        )
    mask_flag = payload.get("use_missingness_mask")
    if condition == "without_mask" and mask_flag is not False:
        raise ValueError(
            f"Maskless checkpoint is not marked use_missingness_mask=False: {path}"
        )
    if condition == "with_mask" and mask_flag is False:
        raise ValueError(f"With-mask checkpoint is marked maskless: {path}")
    return payload["metrics"], mask_flag


def _fold_checkpoint(root, fold, filename):
    return os.path.join(root, f"fold_{fold}", "checkpoints", filename)


def _mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _format(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}"


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(path, summary_lookup):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Paired ablation of the explicit modality-availability mask under "
            r"five-fold site-grouped holdout. Values are mean $\pm$ standard deviation "
            r"across outer folds; $\Delta$ is without-mask minus with-mask performance.}"
        ),
        r"\label{tab:missingness_mask_site_holdout}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Metric & With mask & Without mask & $\Delta$ \\",
        r"\midrule",
    ]
    for task, (metric, metric_label) in PRIMARY_METRICS.items():
        with_row = summary_lookup[(task, metric, "with_mask")]
        without_row = summary_lookup[(task, metric, "without_mask")]
        delta_mean, delta_sd = _mean_sd(
            np.asarray(without_row["fold_values"]) - np.asarray(with_row["fold_values"])
        )
        lines.append(
            f"{TASK_LABELS[task]} & {metric_label} & "
            f"{_format(with_row['mean'], with_row['sd'])} & "
            f"{_format(without_row['mean'], without_row['sd'])} & "
            f"{_format(delta_mean, delta_sd)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate paired with-mask and maskless checkpoints from the same "
            "five site-held-out folds."
        )
    )
    parser.add_argument("--with_mask_root", required=True)
    parser.add_argument("--without_mask_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main():
    args = parse_args()
    with_mask_root = os.path.abspath(args.with_mask_root)
    without_mask_root = os.path.abspath(args.without_mask_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    split_audit = {}
    per_fold_rows = []
    mask_metadata = defaultdict(set)
    for fold in args.folds:
        split_path = os.path.join(with_mask_root, f"fold_{fold}", "site_grouped_split.txt")
        split_audit[str(fold)] = _validate_site_split(split_path)
        for condition, root in (
            ("with_mask", with_mask_root),
            ("without_mask", without_mask_root),
        ):
            for task, filename in CHECKPOINTS.items():
                path = _fold_checkpoint(root, fold, filename)
                metrics, mask_flag = _load_metrics(path, condition)
                mask_metadata[condition].add(mask_flag)
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        per_fold_rows.append(
                            {
                                "fold": fold,
                                "condition": condition,
                                "task": task,
                                "metric": metric,
                                "value": float(value),
                            }
                        )

    grouped = defaultdict(list)
    for row in per_fold_rows:
        grouped[(row["task"], row["metric"], row["condition"])].append(
            (row["fold"], row["value"])
        )

    summary_rows = []
    summary_lookup = {}
    for (task, metric, condition), fold_values in sorted(grouped.items()):
        ordered = [value for _, value in sorted(fold_values)]
        mean, sd = _mean_sd(ordered)
        row = {
            "task": task,
            "metric": metric,
            "condition": condition,
            "mean": mean,
            "sd": sd,
            "n_folds": len(ordered),
            "fold_values": ordered,
        }
        summary_rows.append(
            {
                **{key: value for key, value in row.items() if key != "fold_values"},
                "fold_values": ";".join(f"{value:.8g}" for value in ordered),
            }
        )
        summary_lookup[(task, metric, condition)] = row

    primary_rows = []
    for task, (metric, _) in PRIMARY_METRICS.items():
        with_row = summary_lookup[(task, metric, "with_mask")]
        without_row = summary_lookup[(task, metric, "without_mask")]
        deltas = np.asarray(without_row["fold_values"]) - np.asarray(with_row["fold_values"])
        delta_mean, delta_sd = _mean_sd(deltas)
        primary_rows.append(
            {
                "task": TASK_LABELS[task],
                "metric": metric,
                "with_mask_mean": with_row["mean"],
                "with_mask_sd": with_row["sd"],
                "without_mask_mean": without_row["mean"],
                "without_mask_sd": without_row["sd"],
                "delta_mean": delta_mean,
                "delta_sd": delta_sd,
                "n_folds": len(deltas),
            }
        )

    _write_csv(os.path.join(output_dir, "per_fold_metrics.csv"), per_fold_rows)
    _write_csv(os.path.join(output_dir, "all_metrics_summary.csv"), summary_rows)
    _write_csv(os.path.join(output_dir, "primary_metrics_table.csv"), primary_rows)
    _write_latex(
        os.path.join(output_dir, "missingness_mask_site_holdout_table.tex"),
        summary_lookup,
    )
    with open(os.path.join(output_dir, "aggregation_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "with_mask_root": with_mask_root,
                "without_mask_root": without_mask_root,
                "folds": args.folds,
                "split_type": "five-fold site-grouped holdout",
                "patient_overlap": 0,
                "split_audit": split_audit,
                "checkpoint_mask_metadata": {
                    condition: [
                        value if value is not None else "legacy_default_true"
                        for value in values
                    ]
                    for condition, values in mask_metadata.items()
                },
                "standard_deviation": "sample SD across outer folds (ddof=1)",
                "delta_definition": "without_mask minus with_mask, paired by fold",
            },
            handle,
            indent=2,
        )

    print("Paired five-fold site-held-out aggregation complete.")
    print("Patient overlap across train/validation/test: 0 in every fold.")
    for row in primary_rows:
        print(
            f"{row['task']} {row['metric']}: "
            f"with={_format(row['with_mask_mean'], row['with_mask_sd'])}, "
            f"without={_format(row['without_mask_mean'], row['without_mask_sd'])}, "
            f"delta={_format(row['delta_mean'], row['delta_sd'])}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
