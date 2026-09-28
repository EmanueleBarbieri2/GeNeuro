#!/usr/bin/env python3
"""Run or resume the leakage-resistant ADNI benchmark suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "model" / "checkpoints" / "adni_benchmark"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["classical", "neural", "brainnetcnn", "vae"],
        default=["classical", "neural", "brainnetcnn"],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["three_class", "ad_vs_cn", "mci_vs_cn", "ad_vs_mci"],
        default=["three_class", "ad_vs_cn", "mci_vs_cn", "ad_vs_mci"],
    )
    parser.add_argument(
        "--training_modes",
        nargs="+",
        choices=["frozen_contrastive", "supervised", "joint"],
        default=["frozen_contrastive", "supervised", "joint"],
    )
    parser.add_argument(
        "--brainnet_modalities",
        nargs="+",
        choices=["fmri", "dti", "early_fusion"],
        default=["fmri", "dti", "early_fusion"],
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true", help="Rerun jobs with an existing summary.json.")
    return parser.parse_args()


def run_job(name: str, command: list[str], result_dir: Path, force: bool, records: list[dict]):
    summary = result_dir / "summary.json"
    if summary.exists() and not force:
        print(f"\n[skip] {name}: {summary} already exists", flush=True)
        records.append({"name": name, "status": "skipped", "summary": str(summary)})
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[start] {name}\n  {' '.join(command)}", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    record = {
        "name": name,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "summary": str(summary),
        "command": command,
    }
    records.append(record)
    if completed.returncode:
        raise RuntimeError(f"Benchmark job {name!r} failed with exit code {completed.returncode}.")


def main():
    args = parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        if "classical" in args.stages:
            for task in args.tasks:
                result_dir = root / "classical" / task
                run_job(
                    f"classical/{task}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "model.adni.run_classical_baselines",
                        "--task",
                        task,
                        "--n_jobs",
                        str(args.n_jobs),
                        "--output_dir",
                        str(result_dir),
                    ],
                    result_dir,
                    args.force,
                    records,
                )

        if "neural" in args.stages:
            for task in args.tasks:
                for mode in args.training_modes:
                    result_dir = root / "neural" / task / mode
                    run_job(
                        f"neural/{task}/{mode}",
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "model.adni.run_classification",
                            "--device",
                            args.device,
                            "--task",
                            task,
                            "--training_mode",
                            mode,
                            "--output_dir",
                            str(result_dir),
                        ],
                        result_dir,
                        args.force,
                        records,
                    )

        if "brainnetcnn" in args.stages:
            for task in args.tasks:
                for modality in args.brainnet_modalities:
                    result_dir = root / "brainnetcnn" / task / modality
                    run_job(
                        f"brainnetcnn/{task}/{modality}",
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "model.adni.run_brainnetcnn",
                            "--device",
                            args.device,
                            "--task",
                            task,
                            "--modality",
                            modality,
                            "--output_dir",
                            str(result_dir),
                        ],
                        result_dir,
                        args.force,
                        records,
                    )

        if "vae" in args.stages:
            for task in args.tasks:
                result_dir = root / "vae" / task
                run_job(
                    f"vae/{task}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "model.adni.run_vae_reconstruction",
                        "--device",
                        args.device,
                        "--task",
                        task,
                        "--output_dir",
                        str(result_dir),
                    ],
                    result_dir,
                    args.force,
                    records,
                )
    finally:
        with (root / "suite_status.json").open("w") as handle:
            json.dump({"config": vars(args), "jobs": records}, handle, indent=2)

    print(f"\nBenchmark suite completed. Results: {root}")


if __name__ == "__main__":
    main()
