#!/usr/bin/env python3
"""Run reproducible five-fold cross-validation with entire sites held out."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

from run_site_holdout import (
    DEFAULT_CSV,
    MODALITIES,
    PROJECT_DIR,
    _load_checkpoint_metrics,
    _partition_summary,
    _pipeline_command,
    _read_rows,
    _scan_availability,
)


NUM_FOLDS = 5
SPLIT_ALGORITHM_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_visit_ids():
    availability = {}
    data_dir = os.path.join(PROJECT_DIR, "data")
    for modality in MODALITIES:
        modality_dir = os.path.join(data_dir, modality)
        if not os.path.isdir(modality_dir):
            availability[modality] = set()
            continue
        availability[modality] = {
            os.path.splitext(filename)[0]
            for filename in os.listdir(modality_dir)
            if filename.endswith(".pt")
        }
    return availability


def _patient_metadata(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["PATNO"]].append(row)

    metadata = {}
    for patient, patient_rows in grouped.items():
        sites = {row["SITE"] for row in patient_rows}
        cohorts = {row["COHORT"] for row in patient_rows}
        if len(sites) != 1:
            raise ValueError(
                f"Patient {patient} occurs at multiple sites {sorted(sites)}; "
                "site-grouped CV requires one site per patient."
            )
        if len(cohorts) != 1:
            raise ValueError(f"Patient {patient} has inconsistent COHORT values: {sorted(cohorts)}")
        metadata[patient] = {
            "site": next(iter(sites)),
            "cohort": next(iter(cohorts)),
            "visits": {f"{row['PATNO']}_{row['EVENT_ID']}" for row in patient_rows},
        }
    return metadata


def _build_site_stats(patient_metadata, availability, allowed_sites=None):
    site_stats = defaultdict(
        lambda: {
            "patients": 0,
            "visits": 0,
            "cohorts": Counter(),
            "modalities": Counter(),
        }
    )
    for patient in patient_metadata.values():
        site = patient["site"]
        if allowed_sites is not None and site not in allowed_sites:
            continue
        stats = site_stats[site]
        stats["patients"] += 1
        stats["visits"] += len(patient["visits"])
        stats["cohorts"][patient["cohort"]] += 1
        for modality, available_ids in availability.items():
            stats["modalities"][modality] += len(patient["visits"] & available_ids)
    return dict(site_stats)


def _metric_spec(site_stats):
    cohorts = sorted({cohort for stats in site_stats.values() for cohort in stats["cohorts"]})
    metrics = [("patients", 1.0), ("visits", 0.20), ("site_count", 0.05)]
    metrics.extend((f"cohort:{cohort}", 1.0) for cohort in cohorts)
    metrics.extend((f"modality:{modality}", 0.25) for modality in MODALITIES)
    return metrics


def _site_vector(stats):
    vector = {
        "patients": stats["patients"],
        "visits": stats["visits"],
        "site_count": 1,
    }
    vector.update({f"cohort:{key}": value for key, value in stats["cohorts"].items()})
    vector.update({f"modality:{key}": value for key, value in stats["modalities"].items()})
    return vector


def _assignment_score(fold_vectors, targets, metric_weights):
    score = 0.0
    for metric, weight in metric_weights:
        target = targets[metric]
        if target <= 0:
            continue
        score += weight * sum(
            ((fold.get(metric, 0.0) - target) / target) ** 2
            for fold in fold_vectors
        )
    return score


def _assign_sites(site_stats, n_splits, seed):
    if len(site_stats) < n_splits:
        raise ValueError(f"Cannot create {n_splits} site folds from only {len(site_stats)} sites.")

    rng = random.Random(seed)
    ordered_sites = sorted(site_stats)
    rng.shuffle(ordered_sites)
    ordered_sites.sort(
        key=lambda site: (
            -site_stats[site]["patients"],
            -site_stats[site]["visits"],
        )
    )

    metric_weights = _metric_spec(site_stats)
    totals = Counter()
    site_vectors = {}
    for site, stats in site_stats.items():
        vector = _site_vector(stats)
        site_vectors[site] = vector
        totals.update(vector)
    targets = {metric: totals[metric] / n_splits for metric, _ in metric_weights}

    folds = [set() for _ in range(n_splits)]
    fold_vectors = [Counter() for _ in range(n_splits)]
    for site in ordered_sites:
        candidates = []
        candidate_order = list(range(n_splits))
        rng.shuffle(candidate_order)
        for fold_idx in candidate_order:
            proposed = [Counter(vector) for vector in fold_vectors]
            proposed[fold_idx].update(site_vectors[site])
            candidates.append(
                (
                    _assignment_score(proposed, targets, metric_weights),
                    fold_vectors[fold_idx]["patients"],
                    fold_idx,
                )
            )
        _, _, chosen_fold = min(candidates)
        folds[chosen_fold].add(site)
        fold_vectors[chosen_fold].update(site_vectors[site])

    if any(not fold for fold in folds):
        raise AssertionError("Site-balancing algorithm produced an empty fold.")
    if set().union(*folds) != set(site_stats):
        raise AssertionError("Not every site was assigned to exactly one fold.")
    if sum(len(fold) for fold in folds) != len(set().union(*folds)):
        raise AssertionError("A site was assigned to more than one fold.")
    return folds


def _fold_balance(folds, site_stats):
    output = []
    for fold_idx, sites in enumerate(folds, start=1):
        cohort_counts = Counter()
        modality_counts = Counter()
        patients = visits = 0
        for site in sites:
            stats = site_stats[site]
            patients += stats["patients"]
            visits += stats["visits"]
            cohort_counts.update(stats["cohorts"])
            modality_counts.update(stats["modalities"])
        output.append(
            {
                "fold": fold_idx,
                "sites": sorted(sites),
                "site_count": len(sites),
                "patients": patients,
                "visits": visits,
                "cohort_patients": dict(sorted(cohort_counts.items())),
                "modality_visits": dict(sorted(modality_counts.items())),
            }
        )
    return output


def _make_fold_split(rows, patient_metadata, outer_folds, fold_idx, validation_fraction, seed, availability):
    test_sites = set(outer_folds[fold_idx])
    development_sites = set().union(
        *(sites for index, sites in enumerate(outer_folds) if index != fold_idx)
    )
    inner_splits = max(2, round(1.0 / validation_fraction))
    inner_splits = min(inner_splits, len(development_sites))
    development_stats = _build_site_stats(patient_metadata, availability, development_sites)
    inner_folds = _assign_sites(development_stats, inner_splits, seed + 10_000 + fold_idx)
    val_sites = set(inner_folds[fold_idx % inner_splits])
    train_sites = development_sites - val_sites

    site_sets = {"train": train_sites, "val": val_sites, "test": test_sites}
    if any(site_sets[a] & site_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Site leakage detected while constructing a fold.")

    patient_sets = {
        name: {
            patient
            for patient, metadata in patient_metadata.items()
            if metadata["site"] in sites
        }
        for name, sites in site_sets.items()
    }
    if any(patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Patient leakage detected while constructing a fold.")

    partitions = {"train": [], "val": [], "test": []}
    for row in rows:
        visit_id = f"{row['PATNO']}_{row['EVENT_ID']}"
        site = patient_metadata[row["PATNO"]]["site"]
        for partition, sites in site_sets.items():
            if site in sites:
                partitions[partition].append(visit_id)
                break
    for partition in partitions:
        partitions[partition] = sorted(set(partitions[partition]))

    return partitions, patient_sets, site_sets, inner_splits


def _write_split(path, partitions, fold_number, test_sites, val_sites):
    with open(path, "w") as handle:
        handle.write(f"# Five-fold site-grouped CV, outer fold {fold_number}.\n")
        handle.write(f"# Validation sites: {','.join(sorted(val_sites))}\n")
        handle.write(f"# Untouched test sites: {','.join(sorted(test_sites))}\n")
        for partition in ("train", "val", "test"):
            handle.write(f"{partition}_ids:\n")
            for visit_id in partitions[partition]:
                handle.write(f"{visit_id}\n")


def _run_pipeline(command, log_path):
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
        return process.wait()


def _write_metric_summary(results, run_dir):
    records = defaultdict(list)
    for result in results:
        if result.get("status") != "ok":
            continue
        for checkpoint, payload in result.get("checkpoint_metrics", {}).items():
            for metric, value in (payload.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    records[(checkpoint, metric)].append(float(value))

    csv_path = os.path.join(run_dir, "site_cv_test_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "metric", "mean", "std", "n", "fold_values"])
        for (checkpoint, metric), values in sorted(records.items()):
            writer.writerow(
                [
                    checkpoint,
                    metric,
                    statistics.fmean(values),
                    statistics.pstdev(values) if len(values) > 1 else 0.0,
                    len(values),
                    json.dumps(values),
                ]
            )
    return csv_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run five-fold site-grouped CV. Entire sites are excluded from each outer test fold, "
            "and internal validation is also site-disjoint."
        )
    )
    parser.add_argument("--data_csv", default=DEFAULT_CSV)
    parser.add_argument("--logs_dir", default=os.path.join(PROJECT_DIR, "model", "logs", "site_5fold_cv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.20,
        help="Approximate fraction of the non-test sites used for internal validation.",
    )
    parser.add_argument("--prepare_only", action="store_true", help="Generate and audit folds without training.")
    parser.add_argument("--fold", type=int, choices=range(1, NUM_FOLDS + 1), help="Run only this outer fold.")
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
    return parser.parse_args()


def main():
    args = _parse_args()
    if not 0 < args.validation_fraction <= 0.5:
        raise SystemExit("--validation_fraction must be greater than 0 and no more than 0.5.")

    data_csv = os.path.abspath(args.data_csv)
    rows = _read_rows(data_csv)
    data_csv_sha256 = _sha256(data_csv)
    patient_metadata = _patient_metadata(rows)
    availability = _available_visit_ids()
    site_stats = _build_site_stats(patient_metadata, availability)
    outer_folds = _assign_sites(site_stats, NUM_FOLDS, args.seed)
    outer_balance = _fold_balance(outer_folds, site_stats)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.logs_dir, f"site_5fold_seed{args.seed}_{run_id}"))
    os.makedirs(run_dir, exist_ok=False)

    selected_indices = [args.fold - 1] if args.fold else list(range(NUM_FOLDS))
    results = []
    fold_manifests = []
    for fold_idx in selected_indices:
        fold_number = fold_idx + 1
        fold_dir = os.path.join(run_dir, f"fold_{fold_number}")
        checkpoints_dir = os.path.join(fold_dir, "checkpoints")
        os.makedirs(checkpoints_dir)

        partitions, patient_sets, site_sets, inner_splits = _make_fold_split(
            rows,
            patient_metadata,
            outer_folds,
            fold_idx,
            args.validation_fraction,
            args.seed,
            availability,
        )
        split_path = os.path.join(fold_dir, "site_grouped_split.txt")
        _write_split(split_path, partitions, fold_number, site_sets["test"], site_sets["val"])
        partition_summary = _partition_summary(rows, patient_sets, partitions)
        fold_manifest = {
            "fold": fold_number,
            "seed": args.seed,
            "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
            "data_csv": data_csv,
            "data_csv_sha256": data_csv_sha256,
            "inner_site_splits": inner_splits,
            "requested_validation_fraction": args.validation_fraction,
            "sites": {name: sorted(sites) for name, sites in site_sets.items()},
            "site_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "patient_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "partitions": partition_summary,
            "scan_availability": _scan_availability(partitions),
            "split_path": split_path,
        }
        manifest_path = os.path.join(fold_dir, "split_manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(fold_manifest, handle, indent=2)
        fold_manifests.append(manifest_path)

        print(f"\n{'=' * 80}")
        print(f"Outer fold {fold_number}/{NUM_FOLDS}")
        for partition in ("train", "val", "test"):
            details = partition_summary[partition]
            print(
                f"  {partition:>5}: {len(site_sets[partition]):2d} sites, "
                f"{details['patients']:4d} patients, {details['visits']:4d} visits"
            )
        print(f"  Test sites: {','.join(sorted(site_sets['test']))}")
        print("  Site overlap: 0 | Patient overlap: 0")
        print(f"  Manifest: {manifest_path}")

        if args.prepare_only:
            results.append({"fold": fold_number, "status": "prepared", "manifest": manifest_path})
            continue

        log_path = os.path.join(fold_dir, "full_pipeline.log")
        command = _pipeline_command(args, split_path, checkpoints_dir)
        return_code = _run_pipeline(command, log_path)
        if return_code:
            results.append(
                {
                    "fold": fold_number,
                    "status": "failed",
                    "return_code": return_code,
                    "manifest": manifest_path,
                    "log": log_path,
                }
            )
            print(f"Fold {fold_number} failed with exit code {return_code}. See {log_path}")
            continue
        results.append(
            {
                "fold": fold_number,
                "status": "ok",
                "manifest": manifest_path,
                "log": log_path,
                "checkpoint_metrics": _load_checkpoint_metrics(checkpoints_dir),
            }
        )

    summary_path = os.path.join(run_dir, "site_cv_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "design": "five-fold outer site-grouped CV with site-disjoint internal validation",
                "seed": args.seed,
                "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
                "data_csv": data_csv,
                "data_csv_sha256": data_csv_sha256,
                "configuration": vars(args),
                "outer_fold_balance": outer_balance,
                "fold_manifests": fold_manifests,
                "results": results,
            },
            handle,
            indent=2,
        )

    print(f"\nSummary: {summary_path}")
    if args.prepare_only:
        print("Preparation only: no model training was started.")
    else:
        metric_path = _write_metric_summary(results, run_dir)
        print(f"Aggregated untouched-test metrics: {metric_path}")


if __name__ == "__main__":
    main()
