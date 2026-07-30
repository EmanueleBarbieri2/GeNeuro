#!/usr/bin/env python3
"""Reproduce the legacy embedding-level baselines on site-held-out folds.

By default each fold is trained end to end from raw graphs.  The resulting
fold-specific contrastive ``embeddings.pt`` are used by downstream heads
without generative reconstruction and compared with full-model heads trained
on the matching ``recon_demo.pt``.  Consequently, these are no-reconstruction
ablations, not no-contrastive raw-graph baselines.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_PATH = os.path.join(PROJECT_DIR, "model", "run_full_pipeline.py")
CLASSIFICATION_PATH = os.path.join(
    PROJECT_DIR, "model", "downstream", "downstream_classification.py"
)
STATIC_PATH = os.path.join(
    PROJECT_DIR, "model", "downstream", "downstream_updrs.py"
)
PROGRESSION_PATH = os.path.join(
    PROJECT_DIR, "model", "downstream", "downstream_progression.py"
)

MODALITIES = ("SPECT", "MRI", "fMRI", "DTI")
PARTITIONS = ("train", "val", "test")
TASKS = (
    "classification",
    "static_u2",
    "static_u3",
    "progression_u2",
    "progression_u3",
)
BASELINES = {
    "spect": {
        "model": "GCN",
        "modality": "SPECT",
        "active": ("SPECT",),
        "classification": "binary",
        "progression": True,
    },
    "mri": {
        "model": "GCN",
        "modality": "MRI",
        "active": ("MRI",),
        "classification": "three-class",
        "progression": False,
    },
    "fmri": {
        "model": "GINE",
        "modality": "fMRI",
        "active": ("fMRI",),
        "classification": "three-class",
        "progression": True,
    },
    "dti": {
        "model": "GINE",
        "modality": "DTI",
        "active": ("DTI",),
        "classification": "binary",
        "progression": True,
    },
    "multimodal": {
        "model": r"GCN\&GINE",
        "modality": "Multi",
        "active": MODALITIES,
        "classification": "three-class",
        "progression": False,
    },
}
CHECKPOINTS = {
    "classification": "classifier.pt",
    "static_u2": "static_U2_ADL.pt",
    "static_u3": "static_U3_Motor.pt",
    "progression_u2": "prog_U2_ADL.pt",
    "progression_u3": "prog_U3_Motor.pt",
}
TABLE_COLUMNS = (
    ("classification", "bal_acc", "Bal. Acc."),
    ("classification", "f1_macro", "Macro F1"),
    ("classification", "auc_macro", "Macro AUC"),
    ("static_u2", "r2", "Severity Part II"),
    ("static_u3", "r2", "Severity Part III"),
    ("progression_u2", "r2", "Progression Part II"),
    ("progression_u3", "r2", "Progression Part III"),
)


def _load_split(path):
    partitions = {partition: set() for partition in PARTITIONS}
    mode = None
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line in {"train_ids:", "val_ids:", "test_ids:"}:
                mode = line.split("_", 1)[0]
            elif line and not line.startswith("#"):
                if mode is None:
                    raise ValueError(f"ID occurs before a partition header in {path}")
                partitions[mode].add(line)
    if any(not partitions[partition] for partition in PARTITIONS):
        raise ValueError(f"Expected non-empty train/val/test partitions in {path}")
    patients = {
        partition: {visit.split("_", 1)[0] for visit in visits}
        for partition, visits in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[left] & patients[right]
        if overlap:
            raise ValueError(
                f"Patient leakage in {path}: {len(overlap)} patients overlap "
                f"{left}/{right}."
            )
    return partitions, patients


def _site_map(csv_path):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            patient = (row.get("PATNO") or "").strip()
            event = (row.get("EVENT_ID") or "").strip()
            site = (row.get("SITE") or "").strip()
            if patient and event and site:
                mapping[f"{patient}_{event}"] = site
    return mapping


def _audit_sites(partitions, site_by_visit, split_path):
    sites = {
        partition: {
            site_by_visit[visit]
            for visit in visits
            if visit in site_by_visit
        }
        for partition, visits in partitions.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sites[left] & sites[right]
        if overlap:
            raise ValueError(
                f"Site leakage in {split_path}: {sorted(overlap)} overlap "
                f"{left}/{right}."
            )
    return sites


def _fold_dirs(root, folds):
    output = {}
    for fold in folds:
        fold_dir = os.path.join(root, f"fold_{fold}")
        split_path = os.path.join(fold_dir, "site_grouped_split.txt")
        checkpoints = os.path.join(fold_dir, "checkpoints")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing fold split: {split_path}")
        output[fold] = {
            "fold_dir": fold_dir,
            "split_path": split_path,
            "checkpoints": checkpoints,
        }
    return output


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filtered_embeddings(source_path, output_path, active_modalities):
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    required = {"embeddings", "labels", "ids"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Not a raw contrastive embeddings artifact: {source_path}")
    active = set(active_modalities)
    indices = [
        index
        for index, modality in enumerate(payload["labels"])
        if modality in active
    ]
    if not indices:
        raise ValueError(
            f"No {sorted(active)} embeddings are present in {source_path}"
        )
    index_tensor = torch.tensor(indices, dtype=torch.long)
    filtered = {
        "embeddings": payload["embeddings"].index_select(0, index_tensor),
        "labels": [payload["labels"][index] for index in indices],
        "ids": [payload["ids"][index] for index in indices],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(filtered, output_path)
    return output_path


def _representation_ids(embeddings_path, reconstruction_path):
    embeddings = torch.load(
        embeddings_path, map_location="cpu", weights_only=False
    )
    reconstructions = torch.load(
        reconstruction_path, map_location="cpu", weights_only=False
    )
    embedding_ids = set(embeddings.get("ids", []))
    reconstruction_ids = (
        set(reconstructions) if isinstance(reconstructions, dict) else set()
    )
    if embedding_ids != reconstruction_ids:
        raise ValueError(
            "Raw contrastive and reconstructed representations do not cover the "
            f"same visits: raw-only={len(embedding_ids - reconstruction_ids)}, "
            f"reconstruction-only={len(reconstruction_ids - embedding_ids)}."
        )
    digest_source = "\n".join(sorted(embedding_ids)).encode("utf-8")
    return {
        "visits": len(embedding_ids),
        "ids_sha256": hashlib.sha256(digest_source).hexdigest(),
        "embeddings_sha256": _sha256_file(embeddings_path),
        "reconstruction_sha256": _sha256_file(reconstruction_path),
    }


def _checkpoint_result(path, expected_mask=True):
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    if payload.get("evaluation_split") != "test":
        return None
    if payload.get("use_missingness_mask") is not expected_mask:
        return None
    if not isinstance(payload.get("metrics"), dict):
        return None
    return {
        "metrics": payload["metrics"],
        "class_names": payload.get("class_names", []),
        "num_classes": payload.get("num_classes"),
        "use_missingness_mask": payload.get("use_missingness_mask"),
    }


def _source_sidecar(checkpoint_path):
    return f"{checkpoint_path}.source.json"


def _checkpoint_result_for_source(
    checkpoint_path, embeddings_sha256, expected_mask=True
):
    result = _checkpoint_result(checkpoint_path, expected_mask=expected_mask)
    sidecar_path = _source_sidecar(checkpoint_path)
    if result is None or not os.path.exists(sidecar_path):
        return None
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if sidecar.get("embeddings_sha256") != embeddings_sha256:
        return None
    return result


def _run(command):
    print("  command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _full_pipeline_ready(checkpoints_dir):
    artifacts = ("embeddings.pt", "encoders.pt", "generator.pt", "recon_demo.pt")
    if any(
        not os.path.exists(os.path.join(checkpoints_dir, filename))
        for filename in artifacts
    ):
        return False
    return all(
        _checkpoint_result(
            os.path.join(checkpoints_dir, filename),
            expected_mask=True,
        )
        for filename in CHECKPOINTS.values()
    )


def _prepare_fold_pipeline(fold, source, output_fold, args):
    target_checkpoints = os.path.join(output_fold, "full_model", "checkpoints")
    if args.reuse_existing_representations:
        ready = all(
            _checkpoint_result(
                os.path.join(target_checkpoints, filename),
                expected_mask=True,
            )
            for filename in CHECKPOINTS.values()
        )
    else:
        ready = _full_pipeline_ready(target_checkpoints)
    if ready and not args.restart:
        representation_checkpoints = (
            source["checkpoints"]
            if args.reuse_existing_representations
            else target_checkpoints
        )
        label = (
            "full downstream heads with reused representations"
            if args.reuse_existing_representations
            else "complete end-to-end full pipeline"
        )
        print(f"  fold {fold}: resumed {label}", flush=True)
        return target_checkpoints, representation_checkpoints
    os.makedirs(target_checkpoints, exist_ok=True)
    command = [
        sys.executable,
        PIPELINE_PATH,
        "--data_csv",
        os.path.abspath(args.data_csv),
        "--split_path",
        source["split_path"],
        "--checkpoints_dir",
        target_checkpoints,
        "--device",
        args.device,
        "--seed",
        str(args.seed + fold),
        "--contrastive_epochs",
        str(args.contrastive_epochs),
        "--generator_epochs",
        str(args.generator_epochs),
        "--cls_epochs",
        str(args.epochs),
        "--prog_epochs",
        str(args.epochs),
        "--updrs_epochs",
        str(args.epochs),
        "--downstream_lr",
        str(args.lr),
    ]
    if args.reuse_existing_representations:
        source_checkpoints = source["checkpoints"]
        for filename in ("embeddings.pt", "recon_demo.pt"):
            if not os.path.exists(os.path.join(source_checkpoints, filename)):
                raise FileNotFoundError(
                    f"Cannot reuse missing representation: "
                    f"{os.path.join(source_checkpoints, filename)}"
                )
        command.extend(
            ["--reuse_representations_dir", source_checkpoints]
        )
        representation_checkpoints = source_checkpoints
        print(
            f"  fold {fold}: reusing existing representations by explicit request",
            flush=True,
        )
    else:
        representation_checkpoints = target_checkpoints
        print(
            f"  fold {fold}: training full pipeline end to end from raw graphs",
            flush=True,
        )
    _run(command)
    if args.reuse_existing_representations:
        complete_heads = all(
            _checkpoint_result(
                os.path.join(target_checkpoints, filename),
                expected_mask=True,
            )
            for filename in CHECKPOINTS.values()
        )
        if not complete_heads:
            raise RuntimeError(
                f"Fold {fold} did not produce all full-model test checkpoints."
            )
    elif not _full_pipeline_ready(target_checkpoints):
        raise RuntimeError(
            f"Fold {fold} did not produce a complete end-to-end pipeline."
        )
    return target_checkpoints, representation_checkpoints


def _task_command(
    task,
    embeddings_path,
    checkpoint_path,
    split_path,
    baseline,
    fold,
    args,
):
    active = BASELINES[baseline]["active"]
    excluded = [modality for modality in MODALITIES if modality not in active]
    common = [
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--embeddings_path",
        embeddings_path,
        "--split_path",
        split_path,
        "--device",
        args.device,
        "--seed",
        str(args.seed + fold),
        "--disable_generator",
        "--use_mask",
    ]
    if excluded:
        common.extend(["--exclude_modality", *excluded])
    if task == "classification":
        command = [
            sys.executable,
            CLASSIFICATION_PATH,
            "--csv_path",
            os.path.abspath(args.data_csv),
            "--classifier_ckpt",
            checkpoint_path,
            *common,
        ]
        if BASELINES[baseline]["classification"] == "binary":
            command.append("--drop_prodromal")
        return command
    target_index = "1" if task.endswith("u2") else "2"
    if task.startswith("static"):
        return [
            sys.executable,
            STATIC_PATH,
            "--csv_path",
            os.path.abspath(args.data_csv),
            "--updrs_ckpt",
            checkpoint_path,
            "--target_idx",
            target_index,
            *common,
        ]
    return [
        sys.executable,
        PROGRESSION_PATH,
        "--csv_path",
        os.path.abspath(args.data_csv),
        "--progression_ckpt",
        checkpoint_path,
        "--target_idx",
        target_index,
        "--hidden_dim",
        str(args.hidden_dim),
        *common,
    ]


def _mean_sd(values):
    array = np.asarray(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    )


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows, folds):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        for metric, value in row["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[(row["baseline"], row["task"], metric)].append(
                    (row["fold"], float(value))
                )
    summary = []
    for baseline in [*BASELINES, "full_model"]:
        for task in TASKS:
            metrics = sorted(
                metric
                for candidate_baseline, candidate_task, metric in grouped
                if candidate_baseline == baseline and candidate_task == task
            )
            if not metrics:
                summary.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "metric": "",
                        "mean": "",
                        "sd": "",
                        "n_folds": 0,
                        "status": "not_evaluated",
                        "fold_values": "",
                    }
                )
                continue
            for metric in metrics:
                values = sorted(grouped[(baseline, task, metric)])
                mean, sd = _mean_sd([value for _, value in values])
                summary.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "n_folds": len(values),
                        "status": (
                            "ok"
                            if [fold for fold, _ in values] == sorted(folds)
                            else "incomplete"
                        ),
                        "fold_values": ";".join(
                            f"{fold}:{value:.8g}" for fold, value in values
                        ),
                    }
                )
    return summary


def _write_latex(path, summary):
    lookup = {
        (row["baseline"], row["task"], row["metric"]): row
        for row in summary
        if row["metric"]
    }
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Site-held-out legacy representation-level comparison. "
            r"GCN/GINE rows use fold-specific contrastively learned embeddings without "
            r"generative reconstruction; missing embeddings are zero-imputed and accompanied "
            r"by availability indicators, matching the historical downstream protocol. "
            r"The full model uses generated reconstructions. Values are mean $\pm$ sample "
            r"standard deviation across five untouched site folds. SPECT and DTI "
            r"classification ($^\dagger$) is binary Control versus PD. Multimodal "
            r"no-reconstruction progression is not evaluated because it does not provide "
            r"a consistently reconstructed multimodal time series.}"
        ),
        r"\label{tab:legacy_site_holdout}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\toprule",
        (
            r"& & \multicolumn{3}{c}{\textbf{Classification}} & "
            r"\multicolumn{2}{c}{\textbf{Severity ($R^2$)}} & "
            r"\multicolumn{2}{c}{\textbf{Progression ($R^2$)}} \\"
        ),
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        (
            r"\textbf{Model} & \textbf{Modality} & \textbf{Bal. Acc.} & "
            r"\textbf{Macro F1} & \textbf{Macro AUC} & \textbf{Part II} & "
            r"\textbf{Part III} & \textbf{Part II} & \textbf{Part III} \\"
        ),
        r"\midrule",
    ]
    for baseline in [*BASELINES, "full_model"]:
        if baseline == "full_model":
            lines.append(r"\midrule")
            model = r"\textbf{\modelName}"
            modality = r"\textbf{Multi}"
        else:
            model = BASELINES[baseline]["model"]
            modality = BASELINES[baseline]["modality"]
            if baseline in {"spect", "dti"}:
                modality += r"$^\dagger$"
        cells = []
        for task, metric, _ in TABLE_COLUMNS:
            row = lookup.get((baseline, task, metric))
            if row is None or row["status"] != "ok":
                cells.append("--")
            else:
                cells.append(
                    f"{float(row['mean']):.4f} $\\pm$ {float(row['sd']):.4f}"
                )
        lines.append(f"{model} & {modality} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the historical contrastive-embedding/no-reconstruction baselines "
            "on existing five-fold site-held-out splits."
        )
    )
    parser.add_argument("--site_cv_root", required=True)
    parser.add_argument(
        "--data_csv",
        default=os.path.join(
            PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            PROJECT_DIR, "model", "results", "legacy_site_holdout_baselines"
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--generator_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument(
        "--reuse_existing_representations",
        action="store_true",
        help=(
            "Reuse embeddings.pt/recon_demo.pt under --site_cv_root. By default "
            "the full pipeline is retrained end to end inside --output_dir."
        ),
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.site_cv_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folds = _fold_dirs(root, args.folds)
    site_by_visit = _site_map(args.data_csv)
    split_audit = {}
    representation_audit = {}
    rows = []

    print(
        "Running legacy site-held-out representation baselines.\n"
        "  Full pipeline preparation: "
        + (
            "reuse existing representations (explicit override)\n"
            if args.reuse_existing_representations
            else "end-to-end retraining from raw graphs\n"
        )
        + "  Baseline input: freshly matched contrastive embeddings.pt (no generator)\n"
        "  Full input: freshly matched recon_demo.pt\n"
        "  Explicit availability indicators: enabled (historical protocol)",
        flush=True,
    )

    for fold in args.folds:
        source = folds[fold]
        partitions, patients = _load_split(source["split_path"])
        sites = _audit_sites(partitions, site_by_visit, source["split_path"])
        split_audit[str(fold)] = {
            "split_path": source["split_path"],
            "patient_overlap": 0,
            "site_overlap": 0,
            "visits": {
                partition: len(partitions[partition]) for partition in PARTITIONS
            },
            "patients": {
                partition: len(patients[partition]) for partition in PARTITIONS
            },
            "sites": {
                partition: sorted(sites[partition]) for partition in PARTITIONS
            },
        }
        output_fold = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(output_fold, exist_ok=True)
        shutil.copy2(
            source["split_path"],
            os.path.join(output_fold, "site_grouped_split.txt"),
        )

        print(f"\n{'=' * 78}\nFold {fold}/{len(args.folds)}", flush=True)
        fold_rows = []
        full_checkpoints, representation_checkpoints = _prepare_fold_pipeline(
            fold, source, output_fold, args
        )
        representation_audit[str(fold)] = _representation_ids(
            os.path.join(representation_checkpoints, "embeddings.pt"),
            os.path.join(representation_checkpoints, "recon_demo.pt"),
        )
        embeddings_sha256 = representation_audit[str(fold)][
            "embeddings_sha256"
        ]
        for task, filename in CHECKPOINTS.items():
            result = _checkpoint_result(
                os.path.join(full_checkpoints, filename),
                expected_mask=True,
            )
            if result is None:
                raise RuntimeError(
                    f"Full-model checkpoint is invalid: "
                    f"{os.path.join(full_checkpoints, filename)}"
                )
            output = {
                "fold": fold,
                "baseline": "full_model",
                "task": task,
                "status": "ok",
                **result,
            }
            rows.append(output)
            fold_rows.append(output)

        source_embeddings = os.path.join(
            representation_checkpoints, "embeddings.pt"
        )
        for baseline, configuration in BASELINES.items():
            artifact_path = os.path.join(
                output_fold, "artifacts", f"{baseline}_embeddings.pt"
            )
            _filtered_embeddings(
                source_embeddings,
                artifact_path,
                configuration["active"],
            )
            for task in TASKS:
                if task.startswith("progression") and not configuration["progression"]:
                    output = {
                        "fold": fold,
                        "baseline": baseline,
                        "task": task,
                        "status": "not_evaluated",
                        "reason": (
                            "Historical protocol did not evaluate this "
                            "longitudinal configuration."
                        ),
                    }
                    rows.append(output)
                    fold_rows.append(output)
                    continue
                checkpoint_path = os.path.join(
                    output_fold,
                    baseline,
                    "checkpoints",
                    CHECKPOINTS[task],
                )
                existing = (
                    None
                    if args.restart
                    else _checkpoint_result_for_source(
                        checkpoint_path,
                        embeddings_sha256,
                        expected_mask=True,
                    )
                )
                if existing is None:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    print(f"  {baseline} | {task}", flush=True)
                    _run(
                        _task_command(
                            task,
                            artifact_path,
                            checkpoint_path,
                            source["split_path"],
                            baseline,
                            fold,
                            args,
                        )
                    )
                    existing = _checkpoint_result(
                        checkpoint_path, expected_mask=True
                    )
                    if existing is not None:
                        _atomic_json(
                            _source_sidecar(checkpoint_path),
                            {
                                "embeddings_sha256": embeddings_sha256,
                                "embeddings_path": os.path.abspath(
                                    source_embeddings
                                ),
                            },
                        )
                else:
                    print(f"  {baseline} | {task}: resumed", flush=True)
                if existing is None:
                    raise RuntimeError(
                        f"Task did not produce an untouched-test checkpoint: "
                        f"{checkpoint_path}"
                    )
                output = {
                    "fold": fold,
                    "baseline": baseline,
                    "task": task,
                    "status": "ok",
                    **existing,
                }
                rows.append(output)
                fold_rows.append(output)
        _atomic_json(
            os.path.join(output_fold, "fold_results.json"),
            {"fold": fold, "results": fold_rows},
        )

    per_fold = []
    for row in rows:
        base = {
            "fold": row["fold"],
            "baseline": row["baseline"],
            "task": row["task"],
            "status": row["status"],
            "reason": row.get("reason", ""),
            "class_names": "|".join(row.get("class_names", [])),
            "num_classes": row.get("num_classes", ""),
        }
        if row["status"] == "ok":
            for metric, value in row["metrics"].items():
                per_fold.append({**base, "metric": metric, "value": value})
        else:
            per_fold.append({**base, "metric": "", "value": ""})
    summary = _aggregate(rows, args.folds)
    _write_csv(os.path.join(output_dir, "per_fold_metrics.csv"), per_fold)
    _write_csv(os.path.join(output_dir, "legacy_site_holdout_summary.csv"), summary)
    _write_latex(
        os.path.join(output_dir, "legacy_site_holdout_table.tex"),
        summary,
    )
    _atomic_json(
        os.path.join(output_dir, "run_manifest.json"),
        {
            "site_cv_root": root,
            "data_csv": os.path.abspath(args.data_csv),
            "folds": args.folds,
            "split_audit": split_audit,
            "representation_audit": representation_audit,
            "protocol": {
                "representation_preparation": (
                    "reused existing fold artifacts"
                    if args.reuse_existing_representations
                    else "trained end to end from raw graphs in this run"
                ),
                "baseline_representation": (
                    "fold-specific contrastive embeddings.pt"
                ),
                "baseline_generator": False,
                "baseline_availability_indicators": True,
                "full_representation": "fold-specific recon_demo.pt",
                "full_availability_indicators": True,
                "multimodal_progression": "not evaluated",
                "comparison_type": (
                    "no-reconstruction ablation; not a no-contrastive baseline"
                ),
            },
            "configuration": vars(args),
        },
    )
    print("\nLegacy site-held-out experiment complete.", flush=True)
    print(
        f"LaTeX table: {os.path.join(output_dir, 'legacy_site_holdout_table.tex')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
