#!/usr/bin/env python3
"""Run a fresh site-held-out full-model versus raw-GNN benchmark.

Only the site-grouped split files are reused.  For every outer fold this
orchestrator first trains a new GeNeuro pipeline (contrastive encoders,
generator, reconstructions, and downstream heads).  It then launches the
independent raw-graph baselines, whose GCN/GINE encoders are randomly
initialized and optimized end to end for each supervised task.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FULL_PIPELINE = os.path.join(PROJECT_DIR, "model", "run_full_pipeline.py")
RAW_BASELINES = os.path.join(PROJECT_DIR, "model", "run_site_holdout_baselines.py")
FULL_ARTIFACTS = (
    "embeddings.pt",
    "encoders.pt",
    "generator.pt",
    "recon_demo.pt",
    "classifier.pt",
    "static_U2_ADL.pt",
    "static_U3_Motor.pt",
    "prog_U2_ADL.pt",
    "prog_U3_Motor.pt",
)


def _run(command):
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _split_path(site_cv_root, fold):
    path = os.path.join(site_cv_root, f"fold_{fold}", "site_grouped_split.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing site-held-out split: {path}")
    return path


def _full_fold_ready(checkpoints_dir):
    return all(
        os.path.isfile(os.path.join(checkpoints_dir, filename))
        for filename in FULL_ARTIFACTS
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the complete model and independent raw GCN/GINE baselines "
            "from scratch on the same site-held-out folds."
        )
    )
    parser.add_argument("--site_cv_root", required=True)
    parser.add_argument(
        "--data_root", default=os.path.join(PROJECT_DIR, "data")
    )
    parser.add_argument(
        "--data_csv",
        default=os.path.join(
            PROJECT_DIR,
            "data",
            "PPMI_Curated_Data_Cut_Public_20251112.csv",
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            PROJECT_DIR,
            "model",
            "results",
            "site_holdout_end_to_end",
        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--generator_epochs", type=int, default=100)
    parser.add_argument("--full_downstream_epochs", type=int, default=100)
    parser.add_argument("--full_downstream_lr", type=float, default=0.01)

    parser.add_argument("--baseline_epochs", type=int, default=100)
    parser.add_argument("--baseline_patience", type=int, default=20)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--baseline_weight_decay", type=float, default=1e-2)
    parser.add_argument("--baseline_batch_size", type=int, default=64)
    parser.add_argument("--progression_batch_size", type=int, default=16)
    parser.add_argument("--restart_full", action="store_true")
    parser.add_argument("--restart_baselines", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    site_cv_root = os.path.abspath(args.site_cv_root)
    data_root = os.path.abspath(args.data_root)
    data_csv = os.path.abspath(args.data_csv)
    output_dir = os.path.abspath(args.output_dir)
    full_model_root = os.path.join(output_dir, "full_model")
    baseline_output = os.path.join(output_dir, "raw_graph_baselines")
    os.makedirs(full_model_root, exist_ok=True)

    print(
        "Fresh end-to-end site-held-out benchmark\n"
        "  Full model: new contrastive encoders + generator + downstream heads\n"
        "  Baselines: randomly initialized raw GCN/GINE encoders trained "
        "supervised end to end\n"
        "  Shared inputs: split membership and raw graph files only\n"
        "  Availability indicators: disabled for both branches",
        flush=True,
    )

    for fold in args.folds:
        source_split = _split_path(site_cv_root, fold)
        target_fold = os.path.join(full_model_root, f"fold_{fold}")
        checkpoints_dir = os.path.join(target_fold, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        shutil.copy2(
            source_split, os.path.join(target_fold, "site_grouped_split.txt")
        )
        if _full_fold_ready(checkpoints_dir) and not args.restart_full:
            print(f"Fold {fold}: complete fresh full-model fold resumed", flush=True)
            continue
        print(f"\nFold {fold}: training full model from raw graphs", flush=True)
        _run(
            [
                sys.executable,
                FULL_PIPELINE,
                "--data_csv",
                data_csv,
                "--split_path",
                source_split,
                "--checkpoints_dir",
                checkpoints_dir,
                "--device",
                args.device,
                "--seed",
                str(args.seed + fold),
                "--contrastive_epochs",
                str(args.contrastive_epochs),
                "--generator_epochs",
                str(args.generator_epochs),
                "--cls_epochs",
                str(args.full_downstream_epochs),
                "--prog_epochs",
                str(args.full_downstream_epochs),
                "--updrs_epochs",
                str(args.full_downstream_epochs),
                "--downstream_lr",
                str(args.full_downstream_lr),
                "--no_missingness_mask",
            ]
        )
        if not _full_fold_ready(checkpoints_dir):
            raise RuntimeError(
                f"Full-model fold {fold} did not produce every required artifact."
            )

    print(
        "\nTraining independent raw-graph GCN/GINE baselines from scratch",
        flush=True,
    )
    baseline_command = [
        sys.executable,
        RAW_BASELINES,
        "--site_cv_root",
        site_cv_root,
        "--full_model_root",
        full_model_root,
        "--data_root",
        data_root,
        "--data_csv",
        data_csv,
        "--output_dir",
        baseline_output,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--folds",
        *[str(fold) for fold in args.folds],
        "--epochs",
        str(args.baseline_epochs),
        "--patience",
        str(args.baseline_patience),
        "--lr",
        str(args.baseline_lr),
        "--weight_decay",
        str(args.baseline_weight_decay),
        "--batch_size",
        str(args.baseline_batch_size),
        "--progression_batch_size",
        str(args.progression_batch_size),
        "--no-multimodal_missingness_mask",
    ]
    if args.restart_baselines:
        baseline_command.append("--restart")
    _run(baseline_command)

    promoted_outputs = {
        "site_holdout_baseline_table.tex": "site_holdout_end_to_end_table.tex",
        "site_holdout_baseline_summary.csv": "site_holdout_end_to_end_summary.csv",
        "per_fold_metrics.csv": "site_holdout_end_to_end_per_fold.csv",
        "multimodal_full_comparability_audit.csv": (
            "site_holdout_end_to_end_comparability_audit.csv"
        ),
    }
    for source_name, target_name in promoted_outputs.items():
        source = os.path.join(baseline_output, source_name)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Expected benchmark output is missing: {source}")
        shutil.copy2(source, os.path.join(output_dir, target_name))

    manifest = {
        "site_cv_root": site_cv_root,
        "full_model_root": full_model_root,
        "raw_baseline_output": baseline_output,
        "folds": args.folds,
        "seed": args.seed,
        "full_model_initialization": "new training per fold",
        "baseline_initialization": (
            "random initialization per fold, baseline, and downstream task; "
            "no weights or optimizer state shared between runs"
        ),
        "shared_between_branches": [
            "site_grouped_split.txt",
            "raw graph files",
            "clinical labels/targets",
        ],
        "not_shared_with_baselines": [
            "encoders.pt",
            "embeddings.pt",
            "generator.pt",
            "recon_demo.pt",
            "full-model downstream checkpoints",
        ],
        "missingness_indicators": False,
        "configuration": vars(args),
    }
    with open(
        os.path.join(output_dir, "end_to_end_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print("\nEnd-to-end benchmark complete.", flush=True)
    print(
        "LaTeX table: "
        + os.path.join(output_dir, "site_holdout_end_to_end_table.tex"),
        flush=True,
    )


if __name__ == "__main__":
    main()
