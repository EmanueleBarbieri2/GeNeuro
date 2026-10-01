#!/usr/bin/env python3
"""Run matched No-CL and No-GR ablations with five-fold site holdout.

Both conditions are launched through ``run_5fold_site_cv.py`` with identical
data, split seed, hyperparameters, and no observed/generated modality flag.
By default, No-GR mean-pools only genuinely observed embeddings instead of
zero-filling unavailable modality slots.
After training, the suite validates and aggregates untouched-test metrics into
the paper table requested by ``aggregate_site_holdout_ablations.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SITE_CV_SCRIPT = PROJECT_DIR / "model" / "run_5fold_site_cv.py"
AGGREGATE_SCRIPT = PROJECT_DIR / "model" / "aggregate_site_holdout_ablations.py"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_csv",
        default=str(PROJECT_DIR / "data" / "PPMI_Curated_Data_Cut_Public_20251112.csv"),
    )
    parser.add_argument(
        "--output_root",
        default=str(PROJECT_DIR / "model" / "logs" / "site_holdout_ablations"),
    )
    parser.add_argument("--full_root", help="Optional completed matched full-model run to include in the table.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.20)
    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--generator_epochs", type=int, default=100)
    parser.add_argument("--cls_epochs", type=int, default=100)
    parser.add_argument("--prog_epochs", type=int, default=100)
    parser.add_argument("--updrs_epochs", type=int, default=100)
    parser.add_argument("--downstream_lr", type=float, default=0.01)
    parser.add_argument(
        "--no_gr_strategy",
        choices=("observed_only", "zero_fill"),
        default="observed_only",
        help=(
            "Fusion used by the No-GR condition. observed_only mean-pools only "
            "available contrastive embeddings and is the manuscript default; "
            "zero_fill retains the legacy diagnostic."
        ),
    )
    return parser.parse_args()


def _run(command):
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _new_run(condition_dir: Path, before: set[Path]) -> Path:
    candidates = sorted(
        (
            path.resolve()
            for path in condition_dir.glob("site_5fold_seed*")
            if path.resolve() not in before and (path / "site_cv_summary.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one new completed run in {condition_dir}; found {len(candidates)}."
        )
    return candidates[0]


def _base_command(args, logs_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SITE_CV_SCRIPT),
        "--data_csv",
        str(Path(args.data_csv).expanduser().resolve()),
        "--logs_dir",
        str(logs_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--validation_fraction",
        str(args.validation_fraction),
        "--contrastive_epochs",
        str(args.contrastive_epochs),
        "--generator_epochs",
        str(args.generator_epochs),
        "--cls_epochs",
        str(args.cls_epochs),
        "--prog_epochs",
        str(args.prog_epochs),
        "--updrs_epochs",
        str(args.updrs_epochs),
        "--downstream_lr",
        str(args.downstream_lr),
        # The downstream models must not receive an observed/generated flag.
        "--no_missingness_mask",
    ]


def main():
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suite_dir = output_root / f"suite_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir()

    condition_flags = {
        "no_cl": ["--skip_cl"],
        "no_gr": ["--disable_generator"],
    }
    if args.no_gr_strategy == "observed_only":
        condition_flags["no_gr"].append("--observed_only_pooling")
    completed = {}
    commands = {}
    try:
        for condition, flags in condition_flags.items():
            condition_dir = output_root / condition
            condition_dir.mkdir(exist_ok=True)
            before = {path.resolve() for path in condition_dir.glob("site_5fold_seed*")}
            command = _base_command(args, condition_dir) + flags
            commands[condition] = command
            print(
                f"\n{'=' * 80}\nCondition: {condition} ({' '.join(flags)})",
                flush=True,
            )
            _run(command)
            completed[condition] = _new_run(condition_dir, before)

        aggregate_command = [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--no_cl_root",
            str(completed["no_cl"]),
            "--no_gr_root",
            str(completed["no_gr"]),
            "--output_dir",
            str(suite_dir),
        ]
        if args.full_root:
            aggregate_command.extend(
                ["--full_root", str(Path(args.full_root).expanduser().resolve())]
            )
        commands["aggregate"] = aggregate_command
        _run(aggregate_command)
        status = "complete"
    except BaseException:
        status = "failed"
        raise
    finally:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "design": "matched five-fold site-held-out No-CL and No-GR ablations",
            "std_convention": "population standard deviation across outer folds (ddof=0)",
            "configuration": vars(args),
            "condition_roots": {name: str(path) for name, path in completed.items()},
            "commands": commands,
        }
        (suite_dir / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(f"\nAblation suite complete: {suite_dir}", flush=True)
    print(f"Paper table: {suite_dir / 'site_holdout_ablation_table.tex'}", flush=True)


if __name__ == "__main__":
    main()
