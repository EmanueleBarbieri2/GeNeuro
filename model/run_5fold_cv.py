# This file has been rewritten to execute the full pipeline per fold,
# propagate modality exclusions, and aggregate downstream metrics from fold-specific checkpoints.

#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "run_full_pipeline.py")
NUM_FOLDS = 5

FOLD_CHECKPOINT_FILES = {
    "classification": ["classifier.pt"],
    "progression_u2": ["prog_U2_ADL.pt"],
    "progression_u3": ["prog_U3_Motor.pt"],
    "updrs_u2": ["static_U2_ADL.pt"],
    "updrs_u3": ["static_U3_Motor.pt"],
}

TASK_READABLE_NAMES = {
    "classification": "Classification",
    "progression_u2": "Progression U2_ADL",
    "progression_u3": "Progression U3_Motor",
    "updrs_u2": "Static UPDRS U2_ADL",
    "updrs_u3": "Static UPDRS U3_Motor",
}

def _format_seconds(seconds):
    minutes, secs = divmod(float(seconds), 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:05.2f}s"
    return f"{secs:.2f}s"

def _fold_split_path(split_dir, fold_idx):
    return os.path.abspath(os.path.join(split_dir, f"unified_split_fold{fold_idx}.txt"))

def _load_metrics_from_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    payload = torch.load(checkpoint_path, map_location="cpu")
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not metrics and isinstance(payload, dict):
        metrics = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
    return metrics or None

def _run_pipeline_for_fold(fold_idx, args):
    fold_id = fold_idx + 1
    fold_split_path = _fold_split_path(args.split_dir, fold_idx)
    if not os.path.exists(fold_split_path):
        print(f"❌ Error: Missing split file: {fold_split_path}")
        return None

    fold_dir = os.path.join(args.logs_dir, f"fold_{fold_id}")
    fold_checkpoints_dir = os.path.join(fold_dir, "checkpoints")
    os.makedirs(fold_checkpoints_dir, exist_ok=True)

    print(f"\n{'='*80}\n🚀 STARTING Fold {fold_id}/{NUM_FOLDS}\n{'='*80}")
    print(f"Split: {fold_split_path}")
    print(f"Checkpoints: {fold_checkpoints_dir}")

    cmd = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--data_csv", args.data_csv,
        "--split_path", fold_split_path,
        "--checkpoints_dir", fold_checkpoints_dir,
        "--device", args.device,
        "--contrastive_epochs", str(args.contrastive_epochs),
        "--generator_epochs", str(args.generator_epochs),
        "--cls_epochs", str(args.cls_epochs),
        "--prog_epochs", str(args.prog_epochs),
        "--updrs_epochs", str(args.updrs_epochs),
        "--downstream_lr", str(args.downstream_lr),
    ]

    if args.exclude_modality:
        cmd.append("--exclude_modality")
        cmd.extend(args.exclude_modality)

    # propagate alternate hub selection to the pipeline
    if getattr(args, 'alternate_hub', None):
        cmd.append('--alternate_hub')
        cmd.append(args.alternate_hub)

    if args.skip_cl:
        cmd.append("--skip_cl")
    if args.disable_generator:
        cmd.append("--disable_generator")
    if args.drop_prodromal:
        cmd.append("--drop_prodromal")
    if args.require_all_active:
        cmd.append("--require_all_active")
    if args.strict_downstream:
        cmd.append("--strict_downstream")
    if args.cls_only:
        cmd.append("--cls_only")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    full_output = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        full_output.append(line)
    process.stdout.close()
    process.wait()
    elapsed = time.perf_counter() - t0

    stdout = "".join(full_output)
    log_path = os.path.join(fold_dir, "full_pipeline.log")
    with open(log_path, "w") as fh:
        fh.write(stdout)

    if process.returncode != 0:
        print(f"❌ Fold {fold_id} failed with exit code {process.returncode}. Log: {log_path}")
        return {
            "fold": fold_id,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "log_path": log_path,
            "checkpoints_dir": fold_checkpoints_dir,
        }

    fold_result = {
        "fold": fold_id,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "log_path": log_path,
        "checkpoints_dir": fold_checkpoints_dir,
        "tasks": {},
    }

    for task_name, checkpoint_files in FOLD_CHECKPOINT_FILES.items():
        task_metrics = {}
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(fold_checkpoints_dir, checkpoint_file)
            metrics = _load_metrics_from_checkpoint(checkpoint_path)
            if metrics:
                task_metrics[checkpoint_file] = metrics
        if task_metrics:
            fold_result["tasks"][task_name] = task_metrics

    return fold_result

def _flatten_metric_records(summary):
    records = []
    for fold_result in summary:
        if fold_result.get("status") != "ok":
            continue
        fold_id = fold_result["fold"]
        for task_name, checkpoint_map in fold_result.get("tasks", {}).items():
            for checkpoint_file, metrics in checkpoint_map.items():
                for metric_name, metric_value in metrics.items():
                    records.append(
                        {
                            "fold": fold_id,
                            "task": task_name,
                            "checkpoint_file": checkpoint_file,
                            "metric": metric_name,
                            "value": float(metric_value),
                        }
                    )
    return records

def _write_outputs(summary, args, run_id):
    os.makedirs(args.logs_dir, exist_ok=True)

    json_path = os.path.join(args.logs_dir, f"cv_full_pipeline_summary_{run_id}.json")
    csv_path = os.path.join(args.logs_dir, f"cv_full_pipeline_summary_{run_id}.csv")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exclude_modality": args.exclude_modality,
        "data_csv": args.data_csv,
        "split_dir": args.split_dir,
        "folds": summary,
    }

    with open(json_path, "w") as jf:
        json.dump(payload, jf, indent=2)

    records = _flatten_metric_records(summary)
    metric_groups = defaultdict(list)
    for row in records:
        key = (row["task"], row["checkpoint_file"], row["metric"])
        metric_groups[key].append(row["value"])

    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["task", "checkpoint_file", "metric", "mean", "std", "n"])
        for (task_name, checkpoint_file, metric_name), values in sorted(metric_groups.items()):
            arr = np.array([v for v in values if v is not None], dtype=float)
            if arr.size == 0:
                writer.writerow([task_name, checkpoint_file, metric_name, "nan", "nan", 0])
            else:
                writer.writerow(
                    [
                        task_name,
                        checkpoint_file,
                        metric_name,
                        float(np.nanmean(arr)),
                        float(np.nanstd(arr)),
                        int(arr.size),
                    ]
                )

    return json_path, csv_path

def _print_human_summary(summary, json_path, csv_path):
    print(
        f"\n{'='*100}\n🏁 5-Fold Full Pipeline CV Summary 🏁\nSaved JSON: {json_path}\nSaved CSV: {csv_path}\n{'='*100}"
    )

    grouped = defaultdict(lambda: defaultdict(list))
    for fold_result in summary:
        if fold_result.get("status") != "ok":
            continue
        for task_name, checkpoint_map in fold_result.get("tasks", {}).items():
            for checkpoint_file, metrics in checkpoint_map.items():
                for metric_name, metric_value in metrics.items():
                    grouped[(task_name, checkpoint_file)][metric_name].append(metric_value)

    for task_name in ["classification", "progression_u2", "progression_u3", "updrs_u2", "updrs_u3"]:
        related = [(k, v) for k, v in grouped.items() if k[0] == task_name]
        if not related:
            continue

        print(f"\n--- {TASK_READABLE_NAMES.get(task_name, task_name)} ---")
        for (task_key, checkpoint_file), metrics in related:
            print(f"[{checkpoint_file}]")
            for metric_name, values in metrics.items():
                arr = np.array([v for v in values if v is not None], dtype=float)
                if arr.size == 0:
                    print(f"  {metric_name}: no numeric values")
                else:
                    print(f"  {metric_name}: {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} (n={arr.size})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-fold CV over the full pipeline and aggregate downstream metrics.")
    parser.add_argument("--data_csv", default=os.path.abspath(os.path.join(BASE_DIR, "..", "data", "PPMI_Curated_Data_Cut_Public_20251112.csv")))
    parser.add_argument("--split_dir", default=os.path.abspath(os.path.join(BASE_DIR, "..", "data")))
    parser.add_argument("--logs_dir", default=os.path.join(BASE_DIR, "logs", "cv_full_pipeline"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude_modality", nargs="+", default=[], choices=["SPECT", "MRI", "fMRI", "DTI"])
    parser.add_argument("--alternate_hub", default=None, choices=["fMRI", "MRI", "SPECT", "DTI"],
                        help="If the requested hub is excluded, use this hub instead")
    parser.add_argument("--skip_cl", action="store_true")
    parser.add_argument("--disable_generator", action="store_true")
    parser.add_argument("--drop_prodromal", action="store_true")
    parser.add_argument("--require_all_active", action="store_true")
    parser.add_argument("--strict_downstream", action="store_true")
    parser.add_argument("--cls_only", action="store_true")

    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--generator_epochs", type=int, default=100)
    parser.add_argument("--cls_epochs", type=int, default=100)
    parser.add_argument("--prog_epochs", type=int, default=100)
    parser.add_argument("--updrs_epochs", type=int, default=100)
    parser.add_argument("--downstream_lr", type=float, default=0.01)

    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = []
    for fold_idx in range(NUM_FOLDS):
        fold_result = _run_pipeline_for_fold(fold_idx, args)
        if fold_result is not None:
            summary.append(fold_result)

    json_path, csv_path = _write_outputs(summary, args, run_id)
    _print_human_summary(summary, json_path, csv_path)