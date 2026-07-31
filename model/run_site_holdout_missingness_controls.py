#!/usr/bin/env python3
"""Run site-held-out missingness controls without retraining other models.

Controls:
1. A three-class multinomial logistic regression using only four observed/not-
   observed modality bits.
2. A complete-case raw GCN/GINE baseline restricted to visits for which SPECT,
   MRI, fMRI, and DTI are all genuinely observed.  Its encoders and heads are
   initialized from scratch for every fold and downstream task.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model import run_site_holdout_baselines as raw  # noqa: E402


CONTROL_NAMES = ("availability_only", "complete_case_multimodal")
TASKS = raw.TASKS
TABLE_COLUMNS = raw.TABLE_COLUMNS


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(raw._json_safe(payload), handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


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


def _identity(ids):
    identifiers = sorted(ids)
    return {
        "test_samples": len(identifiers),
        "test_patients": len({raw._patient_id(identifier) for identifier in identifiers}),
        "test_ids_sha256": hashlib.sha256(
            "\n".join(identifiers).encode("utf-8")
        ).hexdigest(),
    }


def _availability_arrays(partitions, labels, graph_store):
    class_to_index = {name: index for index, name in enumerate(raw.CLASS_NAMES)}
    output = {}
    for partition in raw.PARTITIONS:
        rows = []
        for visit_id in sorted(partitions[partition]):
            label = labels.get(visit_id)
            if label not in class_to_index:
                continue
            bits = [
                int(graph_store.has(modality, visit_id))
                for modality in raw.MODALITIES
            ]
            if not any(bits):
                continue
            rows.append((visit_id, bits, class_to_index[label]))
        output[partition] = rows
    return output


def _availability_classifier(fold, partitions, labels, graph_store, seed):
    rows = _availability_arrays(partitions, labels, graph_store)
    counts = {
        partition: dict(
            sorted(
                Counter(raw.CLASS_NAMES[target] for _, _, target in values).items()
            )
        )
        for partition, values in rows.items()
    }
    expected = set(raw.CLASS_NAMES)
    if any(set(counts[partition]) != expected for partition in raw.PARTITIONS):
        return {
            "status": "not_evaluable",
            "reason": "At least one partition does not contain all three classes.",
            "counts": counts,
        }

    arrays = {}
    for partition, values in rows.items():
        arrays[partition] = (
            np.asarray([bits for _, bits, _ in values], dtype=np.float64),
            np.asarray([target for _, _, target in values], dtype=np.int64),
        )
    best = None
    for regularization in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
        model.fit(*arrays["train"])
        validation_prediction = model.predict(arrays["val"][0])
        score = balanced_accuracy_score(arrays["val"][1], validation_prediction)
        candidate = (float(score), -regularization, model, regularization)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    validation_score, _, model, regularization = best
    test_x, test_y = arrays["test"]
    probability = model.predict_proba(test_x)
    prediction = model.predict(test_x)
    metrics = raw._classification_metrics(
        test_y, prediction, probability, raw.CLASS_NAMES
    )
    return {
        "status": "ok",
        "task": "classification",
        "metrics": metrics,
        "class_names": list(raw.CLASS_NAMES),
        "n_classes": 3,
        "validation_best": validation_score,
        "selected_C": regularization,
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "feature_order": list(raw.MODALITIES),
        "counts": counts,
        "unique_train_patterns": len({tuple(bits) for _, bits, _ in rows["train"]}),
        **_identity([visit_id for visit_id, _, _ in rows["test"]]),
    }


def _complete_case_ids(partitions, graph_store):
    return {
        visit_id
        for partition in raw.PARTITIONS
        for visit_id in partitions[partition]
        if all(graph_store.has(modality, visit_id) for modality in raw.MODALITIES)
    }


def _complete_case_task(
    task,
    split_info,
    eligible_ids,
    labels,
    targets,
    visits,
    graph_store,
    args,
    device,
    seed,
):
    raw._set_seed(seed)
    if task == "classification":
        records = raw._build_visit_records(
            labels,
            split_info["visit_partition"],
            graph_store,
            raw.MODALITIES,
            eligible_ids=eligible_ids,
        )
        return raw._train_classification(
            records, raw.MODALITIES, graph_store, args, device, seed
        )
    target_index = 1 if task.endswith("u2") else 2
    if task.startswith("static"):
        records = raw._build_visit_records(
            targets,
            split_info["visit_partition"],
            graph_store,
            raw.MODALITIES,
            target_index=target_index,
            eligible_ids=eligible_ids,
        )
        return raw._train_static(
            records, raw.MODALITIES, graph_store, args, device, seed, task
        )
    records = raw._build_progression_records(
        visits,
        split_info["patient_partition"],
        graph_store,
        raw.MODALITIES,
        target_index,
        eligible_ids=eligible_ids,
    )
    return raw._train_progression(
        records, raw.MODALITIES, graph_store, args, device, seed, task
    )


def _aggregate(results, folds):
    grouped = defaultdict(list)
    statuses = defaultdict(list)
    for result in results:
        key = (result["control"], result["task"])
        statuses[key].append(result)
        if result["status"] != "ok":
            continue
        for metric, value in result["metrics"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[(result["control"], result["task"], metric)].append(
                    (result["fold"], float(value))
                )
    summary = []
    for control in CONTROL_NAMES:
        for task in TASKS:
            task_results = statuses[(control, task)]
            ok_folds = sorted(
                result["fold"] for result in task_results if result["status"] == "ok"
            )
            reasons = sorted(
                {
                    result.get("reason", result["status"])
                    for result in task_results
                    if result["status"] != "ok"
                }
            )
            metrics = sorted(
                metric
                for candidate, candidate_task, metric in grouped
                if candidate == control and candidate_task == task
            )
            if not metrics:
                summary.append(
                    {
                        "control": control,
                        "task": task,
                        "metric": "",
                        "mean": "",
                        "sd": "",
                        "n_folds": len(ok_folds),
                        "status": "not_evaluable",
                        "reason": " | ".join(reasons),
                        "fold_values": "",
                    }
                )
                continue
            for metric in metrics:
                values = sorted(grouped[(control, task, metric)])
                mean, sd = raw._mean_sd([value for _, value in values])
                summary.append(
                    {
                        "control": control,
                        "task": task,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "n_folds": len(values),
                        "status": "ok" if ok_folds == sorted(folds) else "incomplete",
                        "reason": " | ".join(reasons),
                        "fold_values": ";".join(
                            f"{fold}:{value:.8g}" for fold, value in values
                        ),
                    }
                )
    return summary


def _format(summary, control, task, metric):
    for row in summary:
        if (
            row["control"] == control
            and row["task"] == task
            and row["metric"] == metric
            and row["status"] == "ok"
        ):
            return f"{float(row['mean']):.4f} $\\pm$ {float(row['sd']):.4f}"
    return "--"


def _write_latex(path, summary):
    rows = []
    for control, model, modality in (
        ("availability_only", "Logistic regression", "Availability only"),
        (
            "complete_case_multimodal",
            r"GCN\&GINE",
            r"All four observed$^\dagger$",
        ),
    ):
        cells = [
            _format(summary, control, task, metric)
            for task, metric, _ in TABLE_COLUMNS
        ]
        rows.append(f"{model} & {modality} & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Site-held-out missingness controls. The availability-only "
            r"classifier receives four binary indicators denoting whether SPECT, MRI, "
            r"fMRI, and DTI were acquired and receives no imaging content. The complete-case "
            r"GCN/GINE baseline includes only visits with all four modalities genuinely "
            r"observed, uses no imputation or availability indicators, and is initialized "
            r"from scratch in every fold and task. Complete-case classification "
            r"($^\dagger$) is binary Control versus PD because no Prodromal visit has all "
            r"four modalities. Values are mean $\pm$ sample standard deviation across five "
            r"site-held-out folds; dashes indicate that an endpoint was not evaluable in "
            r"every fold. The two rows use different classification tasks and cohorts and "
            r"are not directly ranked.}"
        ),
        r"\label{tab:site_holdout_missingness_controls}",
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
            r"\textbf{Model} & \textbf{Input} & \textbf{Bal. Acc.} & "
            r"\textbf{Macro F1} & \textbf{Macro AUC} & \textbf{Part II} & "
            r"\textbf{Part III} & \textbf{Part II} & \textbf{Part III} \\"
        ),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete-case GCN/GINE and availability-only controls."
    )
    parser.add_argument("--site_cv_root", required=True)
    parser.add_argument("--data_root", default=os.path.join(PROJECT_DIR, "data"))
    parser.add_argument(
        "--data_csv",
        default=os.path.join(
            PROJECT_DIR, "data", "PPMI_Curated_Data_Cut_Public_20251112.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            PROJECT_DIR, "model", "results", "site_holdout_missingness_controls"
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--progression_batch_size", type=int, default=16)
    parser.add_argument("--encoder_hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--recurrent_dim", type=int, default=256)
    parser.add_argument("--edge_threshold", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_cache_graphs", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.set_defaults(multimodal_missingness_mask=False)
    return parser.parse_args()


def main():
    args = parse_args()
    site_cv_root = os.path.abspath(args.site_cv_root)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    folds = raw._find_fold_dirs(site_cv_root, args.folds)
    graph_store = raw.GraphStore(args.data_root, cache=not args.no_cache_graphs)
    visit_sites = raw._load_site_metadata(args.data_csv)
    labels = raw.load_csv_labels(args.data_csv, drop_prodromal=False)
    targets = raw.load_csv_targets(args.data_csv)
    visits = raw.load_csv_visits(args.data_csv)
    device = raw.torch.device(
        args.device
        if args.device.startswith("cuda") and raw.torch.cuda.is_available()
        else "cpu"
    )
    print(
        "Site-held-out missingness controls\n"
        "  availability-only: four binary acquisition indicators, no images\n"
        "  complete-case: four real modalities, no zero fill, no masks\n"
        "  complete-case GNN weights: new initialization per fold and task",
        flush=True,
    )

    results = []
    count_rows = []
    split_audits = {}
    for fold, paths in folds.items():
        partitions, patients, visit_partition, patient_partition, comments = raw._load_split(
            paths["split_path"]
        )
        sites = raw._audit_site_disjointness(
            partitions, visit_sites, paths["split_path"]
        )
        split_info = {
            "visit_partition": visit_partition,
            "patient_partition": patient_partition,
        }
        eligible_ids = _complete_case_ids(partitions, graph_store)
        split_audits[str(fold)] = {
            "split_path": paths["split_path"],
            "patient_overlap": 0,
            "site_overlap": 0,
            "sites": {partition: sorted(sites[partition]) for partition in raw.PARTITIONS},
        }
        for partition in raw.PARTITIONS:
            ids = sorted(eligible_ids & partitions[partition])
            class_counts = Counter(labels.get(identifier) for identifier in ids)
            count_rows.append(
                {
                    "fold": fold,
                    "partition": partition,
                    "visits": len(ids),
                    "patients": len({raw._patient_id(identifier) for identifier in ids}),
                    "control": class_counts.get("Control", 0),
                    "pd": class_counts.get("PD", 0),
                    "prodromal": class_counts.get("Prodromal", 0),
                }
            )
        print(
            f"\nFold {fold}: {len(eligible_ids)} complete-case visits across split",
            flush=True,
        )

        availability_path = os.path.join(
            output_dir, "runs", f"fold_{fold}", "availability_only.json"
        )
        if os.path.isfile(availability_path) and not args.restart:
            with open(availability_path, encoding="utf-8") as handle:
                availability = json.load(handle)
            print("  availability-only: resumed", flush=True)
        else:
            availability = _availability_classifier(
                fold,
                partitions,
                labels,
                graph_store,
                raw._stable_seed(args.seed, fold, "availability_only"),
            )
            availability.update(
                {
                    "fold": fold,
                    "control": "availability_only",
                    "task": "classification",
                    "input": "four binary modality-availability indicators",
                }
            )
            _atomic_json(availability_path, availability)
        results.append(availability)
        if availability["status"] == "ok":
            print(f"  availability-only test: {availability['metrics']}", flush=True)

        for task in TASKS:
            result_path = os.path.join(
                output_dir,
                "runs",
                f"fold_{fold}",
                "complete_case_multimodal",
                f"{task}.json",
            )
            if os.path.isfile(result_path) and not args.restart:
                with open(result_path, encoding="utf-8") as handle:
                    result = json.load(handle)
                if (
                    result.get("initialization") != "random_from_scratch"
                    or result.get("pretrained_weights_loaded") is not False
                ):
                    raise ValueError(
                        f"Refusing unverified resumed result: {result_path}"
                    )
                print(f"  complete-case {task}: resumed", flush=True)
            else:
                seed = raw._stable_seed(
                    args.seed, fold, "complete_case_multimodal", task
                )
                started = time.time()
                result = _complete_case_task(
                    task,
                    split_info,
                    eligible_ids,
                    labels,
                    targets,
                    visits,
                    graph_store,
                    args,
                    device,
                    seed,
                )
                result.update(
                    {
                        "fold": fold,
                        "control": "complete_case_multimodal",
                        "task": task,
                        "modalities": list(raw.MODALITIES),
                        "complete_case": True,
                        "zero_imputation": False,
                        "availability_indicators": False,
                        "initialization": "random_from_scratch",
                        "pretrained_weights_loaded": False,
                        "weights_shared_with_other_tasks": False,
                        "seed": seed,
                        "elapsed_seconds": time.time() - started,
                    }
                )
                _atomic_json(result_path, result)
            results.append(result)
            if result["status"] == "ok":
                print(f"  complete-case {task} test: {result['metrics']}", flush=True)
            else:
                print(
                    f"  complete-case {task}: {result['status']} - "
                    f"{result.get('reason', '')}",
                    flush=True,
                )

        for task in TASKS[1:]:
            results.append(
                {
                    "fold": fold,
                    "control": "availability_only",
                    "task": task,
                    "status": "not_evaluated",
                    "reason": "Availability-only control is classification only.",
                }
            )

    summary = _aggregate(results, args.folds)
    per_fold = []
    for result in results:
        base = {
            "fold": result["fold"],
            "control": result["control"],
            "task": result["task"],
            "status": result["status"],
            "reason": result.get("reason", ""),
            "test_samples": result.get("test_samples", ""),
            "test_patients": result.get("test_patients", ""),
            "n_classes": result.get("n_classes", ""),
        }
        if result["status"] == "ok":
            for metric, value in result["metrics"].items():
                per_fold.append({**base, "metric": metric, "value": value})
        else:
            per_fold.append({**base, "metric": "", "value": ""})
    _write_csv(os.path.join(output_dir, "per_fold_metrics.csv"), per_fold)
    _write_csv(os.path.join(output_dir, "missingness_controls_summary.csv"), summary)
    _write_csv(os.path.join(output_dir, "complete_case_counts.csv"), count_rows)
    _write_latex(os.path.join(output_dir, "missingness_controls_table.tex"), summary)
    _atomic_json(
        os.path.join(output_dir, "run_manifest.json"),
        {
            "site_cv_root": site_cv_root,
            "data_root": os.path.abspath(args.data_root),
            "data_csv": os.path.abspath(args.data_csv),
            "folds": args.folds,
            "split_audits": split_audits,
            "availability_only_model": (
                "multinomial logistic regression using four binary observed-modality bits; "
                "regularization selected on validation balanced accuracy"
            ),
            "complete_case_policy": (
                "Every included visit has genuinely observed SPECT, MRI, fMRI, and DTI; "
                "no zero imputation, availability flags, contrastive pretraining, generator, "
                "or shared model weights"
            ),
            "configuration": vars(args),
        },
    )
    print("\nMissingness controls complete.", flush=True)
    print(
        f"LaTeX table: {os.path.join(output_dir, 'missingness_controls_table.tex')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
