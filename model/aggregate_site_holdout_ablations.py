#!/usr/bin/env python3
"""Aggregate matched five-fold site-held-out MINERVA ablations.

The script accepts completed ``run_5fold_site_cv.py`` output directories,
checks that every condition used the same site/patient partitions and untouched
test metrics, and writes CSV, JSON, Markdown, and LaTeX summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


FOLDS = tuple(range(1, 6))
CHECKPOINT_METRICS = (
    ("classifier.pt", "bal_acc", "classification_balanced_accuracy"),
    ("classifier.pt", "f1_macro", "classification_macro_f1"),
    ("classifier.pt", "auc_macro", "classification_macro_auc"),
    ("static_U2_ADL.pt", "r2", "severity_part_ii_r2"),
    ("static_U3_Motor.pt", "r2", "severity_part_iii_r2"),
    ("prog_U2_ADL.pt", "r2", "progression_part_ii_r2"),
    ("prog_U3_Motor.pt", "r2", "progression_part_iii_r2"),
)
OUTPUT_METRICS = tuple(item[2] for item in CHECKPOINT_METRICS)
MATCHED_CONFIGURATION_KEYS = (
    "seed",
    "validation_fraction",
    "exclude_modality",
    "alternate_hub",
    "drop_prodromal",
    "require_all_active",
    "strict_downstream",
    "cls_only",
    "contrastive_epochs",
    "generator_epochs",
    "cls_epochs",
    "prog_epochs",
    "updrs_epochs",
    "downstream_lr",
    "no_missingness_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_run_root(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "site_cv_summary.json").is_file():
        return candidate
    matches = sorted(
        child
        for child in candidate.glob("site_5fold_seed*")
        if (child / "site_cv_summary.json").is_file()
    )
    if not matches:
        raise FileNotFoundError(
            f"No site_cv_summary.json found in {candidate} or its site_5fold_seed* children."
        )
    return matches[-1]


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _expected_flags(condition: str) -> dict[str, bool]:
    return {
        "no_cl": {"skip_cl": True, "disable_generator": False},
        "no_gr": {"skip_cl": False, "disable_generator": True},
        "full": {"skip_cl": False, "disable_generator": False},
    }[condition]


def _load_condition(condition: str, path: str) -> dict:
    root = _resolve_run_root(path)
    summary = _load_json(root / "site_cv_summary.json")
    configuration = summary.get("configuration") or {}
    for name, expected in _expected_flags(condition).items():
        actual = bool(configuration.get(name, False))
        if actual != expected:
            raise ValueError(
                f"{root} is not a valid {condition} run: {name}={actual}, expected {expected}."
            )
    if not bool(configuration.get("no_missingness_mask", False)):
        raise ValueError(
            f"{root} used explicit observed/reconstructed indicators. "
            "Ablation comparisons require --no_missingness_mask."
        )

    results = {int(row["fold"]): row for row in summary.get("results", [])}
    if set(results) != set(FOLDS):
        raise ValueError(f"{root} must contain folds {FOLDS}; found {sorted(results)}.")

    split_hashes = {}
    metrics = {name: [] for name in OUTPUT_METRICS}
    per_fold = []
    for fold in FOLDS:
        result = results[fold]
        if result.get("status") != "ok":
            raise ValueError(
                f"{root} fold {fold} did not finish successfully: {result.get('status')!r}."
            )
        manifest = _load_json(Path(result["manifest"]))
        split_path = Path(manifest["split_path"])
        split_hashes[fold] = _sha256(split_path)
        checkpoint_payloads = result.get("checkpoint_metrics") or {}
        fold_row = {"fold": fold}
        for checkpoint, metric, output_name in CHECKPOINT_METRICS:
            payload = checkpoint_payloads.get(checkpoint)
            if not payload:
                raise ValueError(f"{root} fold {fold} is missing {checkpoint} metrics.")
            if payload.get("evaluation_split") != "test":
                raise ValueError(
                    f"{root} fold {fold} {checkpoint} reports "
                    f"evaluation_split={payload.get('evaluation_split')!r}, not 'test'."
                )
            value = (payload.get("metrics") or {}).get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"{root} fold {fold} {checkpoint} has invalid {metric}={value!r}."
                )
            value = float(value)
            metrics[output_name].append(value)
            fold_row[output_name] = value
        per_fold.append(fold_row)

    aggregate = {}
    for name, values in metrics.items():
        aggregate[name] = {
            "mean": statistics.fmean(values),
            # Preserve the convention already used by run_5fold_site_cv.py.
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "fold_values": values,
            "n_folds": len(values),
        }

    return {
        "condition": condition,
        "root": str(root),
        "configuration": configuration,
        "data_csv_sha256": summary.get("data_csv_sha256"),
        "split_algorithm_version": summary.get("split_algorithm_version"),
        "split_hashes": split_hashes,
        "aggregate": aggregate,
        "per_fold": per_fold,
    }


def _validate_matched_design(conditions: list[dict]) -> None:
    reference = conditions[0]
    for current in conditions[1:]:
        for key in ("data_csv_sha256", "split_algorithm_version", "split_hashes"):
            if current[key] != reference[key]:
                raise ValueError(
                    f"Ablation runs are not matched: {current['condition']} differs from "
                    f"{reference['condition']} in {key}."
                )
        for key in MATCHED_CONFIGURATION_KEYS:
            current_value = current["configuration"].get(key)
            reference_value = reference["configuration"].get(key)
            if current_value != reference_value:
                raise ValueError(
                    f"Ablation runs are not matched: {current['condition']} has "
                    f"{key}={current_value!r}, while {reference['condition']} has "
                    f"{key}={reference_value!r}."
                )


def _format_metric(payload: dict) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def _latex_metric(payload: dict) -> str:
    return f"${payload['mean']:.4f} \\pm {payload['std']:.4f}$"


def _condition_label(condition: str) -> str:
    return {
        "full": "MINERVA (Full)",
        "no_cl": "MINERVA (No CL)",
        "no_gr": "MINERVA (No GR)",
    }[condition]


def _summary_row(condition: dict) -> dict:
    row = {
        "condition": condition["condition"],
        "model": _condition_label(condition["condition"]),
        "modality": "Multi",
    }
    for name in OUTPUT_METRICS:
        payload = condition["aggregate"][name]
        row[f"{name}_mean"] = payload["mean"]
        row[f"{name}_std"] = payload["std"]
        row[name] = _format_metric(payload)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["condition", "model", "modality"]
    for name in OUTPUT_METRICS:
        fields.extend([name, f"{name}_mean", f"{name}_std"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, conditions: list[dict]) -> None:
    headers = (
        "Model",
        "Modality",
        "Bal. Acc.",
        "Macro F1",
        "Macro AUC",
        "Severity II",
        "Severity III",
        "Progression II",
        "Progression III",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for condition in conditions:
        aggregate = condition["aggregate"]
        values = [_condition_label(condition["condition"]), "Multi"]
        values.extend(_format_metric(aggregate[name]) for name in OUTPUT_METRICS)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, conditions: list[dict]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Site-held-out ablations. Values are mean $\pm$ standard deviation across five outer folds.}",
        r"\label{tab:site_holdout_ablations}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Classification}} & \multicolumn{2}{c}{\textbf{Severity ($R^2$)}} & \multicolumn{2}{c}{\textbf{Progression ($R^2$)}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"\textbf{Model} & \textbf{Modality} & \textbf{Bal. Acc.} & \textbf{Macro F1} & \textbf{Macro AUC} & \textbf{Part II} & \textbf{Part III} & \textbf{Part II} & \textbf{Part III} \\",
        r"\midrule",
    ]
    for condition in conditions:
        aggregate = condition["aggregate"]
        values = [_latex_metric(aggregate[name]) for name in OUTPUT_METRICS]
        lines.append(
            f"{_condition_label(condition['condition'])} & Multi & "
            + " & ".join(values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no_cl_root", required=True)
    parser.add_argument("--no_gr_root", required=True)
    parser.add_argument("--full_root")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    requested = []
    if args.full_root:
        requested.append(("full", args.full_root))
    requested.extend((("no_cl", args.no_cl_root), ("no_gr", args.no_gr_root)))
    conditions = [_load_condition(name, path) for name, path in requested]
    _validate_matched_design(conditions)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summary_row(condition) for condition in conditions]
    _write_csv(output_dir / "site_holdout_ablation_summary.csv", rows)
    _write_markdown(output_dir / "site_holdout_ablation_table.md", conditions)
    _write_latex(output_dir / "site_holdout_ablation_table.tex", conditions)
    with (output_dir / "site_holdout_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "design": "matched five-fold site-held-out evaluation",
                "std_convention": "population standard deviation across outer folds (ddof=0)",
                "conditions": conditions,
            },
            handle,
            indent=2,
        )
    print(f"Validated matched site splits for: {', '.join(name for name, _ in requested)}")
    print(f"LaTeX table: {output_dir / 'site_holdout_ablation_table.tex'}")
    print(f"CSV summary: {output_dir / 'site_holdout_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
