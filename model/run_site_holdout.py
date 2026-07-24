#!/usr/bin/env python3
"""Run a patient-disjoint, site-held-out evaluation of the full GeNeuro pipeline."""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PIPELINE_SCRIPT = os.path.join(BASE_DIR, "run_full_pipeline.py")
DEFAULT_CSV = os.path.join(PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv")
MODALITIES = ("SPECT", "MRI", "fMRI", "DTI")
CHECKPOINT_FILES = (
    "classifier.pt",
    "prog_U2_ADL.pt",
    "prog_U3_Motor.pt",
    "static_U2_ADL.pt",
    "static_U3_Motor.pt",
)


def _read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"SITE", "PATNO", "EVENT_ID", "COHORT"}
    missing = required.difference(rows[0] if rows else [])
    if missing:
        raise ValueError(f"CSV is empty or missing required columns: {sorted(missing)}")
    for row in rows:
        for column in required:
            row[column] = (row.get(column) or "").strip()
    return [row for row in rows if row["SITE"] and row["PATNO"] and row["EVENT_ID"]]


def _site_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["SITE"]].append(row)
    output = []
    for site, site_rows in grouped.items():
        output.append(
            {
                "site": site,
                "patients": len({row["PATNO"] for row in site_rows}),
                "visits": len({f"{row['PATNO']}_{row['EVENT_ID']}" for row in site_rows}),
                "cohorts": dict(sorted(Counter(row["COHORT"] for row in site_rows).items())),
            }
        )
    return sorted(output, key=lambda item: (-item["patients"], item["site"]))


def _print_sites(rows):
    print(f"{'SITE':>6} {'PATIENTS':>10} {'VISITS':>8}  COHORT VISITS")
    for item in _site_table(rows):
        print(f"{item['site']:>6} {item['patients']:>10} {item['visits']:>8}  {item['cohorts']}")


def _make_patient_split(rows, test_site, validation_fraction, seed):
    patients = defaultdict(list)
    for row in rows:
        patients[row["PATNO"]].append(row)

    test_patients = {
        patient
        for patient, patient_rows in patients.items()
        if test_site in {row["SITE"] for row in patient_rows}
    }
    if not test_patients:
        raise ValueError(f"SITE={test_site!r} does not occur in the CSV. Use --list_sites.")

    remaining_patients = sorted(set(patients).difference(test_patients))
    labels = []
    for patient in remaining_patients:
        cohorts = {row["COHORT"] for row in patients[patient]}
        if len(cohorts) != 1:
            raise ValueError(f"Patient {patient} has inconsistent COHORT values: {sorted(cohorts)}")
        labels.append(next(iter(cohorts)))

    patients_by_cohort = defaultdict(list)
    for patient, label in zip(remaining_patients, labels):
        patients_by_cohort[label].append(patient)
    rng = random.Random(seed)
    val_patients = set()
    for cohort, cohort_patients in sorted(patients_by_cohort.items()):
        rng.shuffle(cohort_patients)
        if len(cohort_patients) < 2:
            raise ValueError(
                f"COHORT={cohort} has fewer than two non-test patients; "
                "a patient-stratified train/validation split is impossible."
            )
        val_count = min(len(cohort_patients) - 1, max(1, round(len(cohort_patients) * validation_fraction)))
        val_patients.update(cohort_patients[:val_count])
    train_patients = set(remaining_patients).difference(val_patients)

    partitions = {"train": [], "val": [], "test": []}
    for row in rows:
        visit_id = f"{row['PATNO']}_{row['EVENT_ID']}"
        if row["PATNO"] in test_patients:
            partitions["test"].append(visit_id)
        elif row["PATNO"] in val_patients:
            partitions["val"].append(visit_id)
        elif row["PATNO"] in train_patients:
            partitions["train"].append(visit_id)

    for name in partitions:
        partitions[name] = sorted(set(partitions[name]))

    patient_sets = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }
    if any(patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Patient leakage detected while constructing the split.")

    non_test_site_visits = [
        row for row in rows if row["PATNO"] in test_patients and row["SITE"] != test_site
    ]
    return partitions, patient_sets, non_test_site_visits


def _write_split(path, partitions, test_site):
    with open(path, "w") as handle:
        handle.write(f"# Patient-disjoint site holdout; SITE={test_site} is the untouched test set.\n")
        for partition in ("train", "val", "test"):
            handle.write(f"{partition}_ids:\n")
            for visit_id in partitions[partition]:
                handle.write(f"{visit_id}\n")


def _partition_summary(rows, patient_sets, partitions):
    summary = {}
    for name in ("train", "val", "test"):
        patient_set = patient_sets[name]
        subset = [row for row in rows if row["PATNO"] in patient_set]
        summary[name] = {
            "patients": len(patient_set),
            "visits": len(partitions[name]),
            "sites": dict(sorted(Counter(row["SITE"] for row in subset).items())),
            "cohort_visits": dict(sorted(Counter(row["COHORT"] for row in subset).items())),
            "cohort_patients": dict(
                sorted(Counter(next(row["COHORT"] for row in rows if row["PATNO"] == patient) for patient in patient_set).items())
            ),
        }
    return summary


def _scan_availability(partitions):
    result = {}
    data_dir = os.path.join(PROJECT_DIR, "data")
    for name, visit_ids in partitions.items():
        visit_set = set(visit_ids)
        result[name] = {}
        for modality in MODALITIES:
            modality_dir = os.path.join(data_dir, modality)
            available = {
                os.path.splitext(filename)[0]
                for filename in os.listdir(modality_dir)
                if filename.endswith(".pt")
            } if os.path.isdir(modality_dir) else set()
            result[name][modality] = len(visit_set & available)
    return result


def _pipeline_command(args, split_path, checkpoints_dir):
    command = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--data_csv", os.path.abspath(args.data_csv),
        "--split_path", split_path,
        "--checkpoints_dir", checkpoints_dir,
        "--device", args.device,
        "--contrastive_epochs", str(args.contrastive_epochs),
        "--generator_epochs", str(args.generator_epochs),
        "--cls_epochs", str(args.cls_epochs),
        "--prog_epochs", str(args.prog_epochs),
        "--updrs_epochs", str(args.updrs_epochs),
        "--downstream_lr", str(args.downstream_lr),
    ]
    if args.exclude_modality:
        command.extend(["--exclude_modality", *args.exclude_modality])
    if args.alternate_hub:
        command.extend(["--alternate_hub", args.alternate_hub])
    for enabled, flag in (
        (args.skip_cl, "--skip_cl"),
        (args.disable_generator, "--disable_generator"),
        (args.drop_prodromal, "--drop_prodromal"),
        (args.require_all_active, "--require_all_active"),
        (args.strict_downstream, "--strict_downstream"),
        (args.cls_only, "--cls_only"),
    ):
        if enabled:
            command.append(flag)
    return command


def _load_checkpoint_metrics(checkpoints_dir):
    import torch

    output = {}
    for filename in CHECKPOINT_FILES:
        path = os.path.join(checkpoints_dir, filename)
        if not os.path.exists(path):
            continue
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            output[filename] = {
                "evaluation_split": payload.get("evaluation_split"),
                "metrics": payload.get("metrics"),
                "validation_metrics": payload.get("validation_metrics"),
            }
    return output


def _safe_site_name(site):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", site)


def main():
    parser = argparse.ArgumentParser(
        description="Train on non-held-out sites and evaluate once on an untouched site, grouped by patient."
    )
    parser.add_argument("--data_csv", default=DEFAULT_CSV)
    parser.add_argument("--test_site", help="SITE value to reserve as the external test set.")
    parser.add_argument("--list_sites", action="store_true", help="Print site sizes and exit.")
    parser.add_argument("--validation_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logs_dir", default=os.path.join(BASE_DIR, "logs", "site_holdout"))
    parser.add_argument("--prepare_only", action="store_true", help="Create and audit the split without training.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude_modality", nargs="+", default=[], choices=MODALITIES)
    parser.add_argument("--alternate_hub", choices=MODALITIES)
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

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation_fraction must be strictly between 0 and 1.")

    rows = _read_rows(os.path.abspath(args.data_csv))
    if args.list_sites:
        _print_sites(rows)
        return
    if not args.test_site:
        parser.error("--test_site is required unless --list_sites is used.")

    partitions, patient_sets, cross_site_rows = _make_patient_split(
        rows, str(args.test_site), args.validation_fraction, args.seed
    )
    partition_summary = _partition_summary(rows, patient_sets, partitions)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(
        os.path.join(args.logs_dir, f"site_{_safe_site_name(str(args.test_site))}_{run_id}")
    )
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=False)
    split_path = os.path.join(run_dir, "site_holdout_split.txt")
    manifest_path = os.path.join(run_dir, "split_manifest.json")
    _write_split(split_path, partitions, str(args.test_site))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_site": str(args.test_site),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "patient_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "held_out_patients_with_visits_at_other_sites": len({row["PATNO"] for row in cross_site_rows}),
        "partitions": partition_summary,
        "scan_availability": _scan_availability(partitions),
        "split_path": split_path,
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Created leakage-audited split: {split_path}")
    for name in ("train", "val", "test"):
        details = partition_summary[name]
        print(f"  {name:>5}: {details['patients']} patients, {details['visits']} visits")
    print("Patient overlap: train/val=0, train/test=0, val/test=0")
    print(f"Manifest: {manifest_path}")

    if args.prepare_only:
        print("Preparation only: training was not started.")
        return

    command = _pipeline_command(args, split_path, checkpoints_dir)
    log_path = os.path.join(run_dir, "full_pipeline.log")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            log.write(line)
        process.stdout.close()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"Full pipeline failed with exit code {return_code}. See {log_path}")

    results_path = os.path.join(run_dir, "test_metrics.json")
    with open(results_path, "w") as handle:
        json.dump(
            {
                "test_site": str(args.test_site),
                "split_manifest": manifest_path,
                "checkpoint_metrics": _load_checkpoint_metrics(checkpoints_dir),
            },
            handle,
            indent=2,
        )
    print(f"Untouched site-test metrics: {results_path}")


if __name__ == "__main__":
    main()
