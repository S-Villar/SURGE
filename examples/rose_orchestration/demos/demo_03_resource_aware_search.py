#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from dataset_utils import build_training_parquet, read_iteration_state, training_row_plan, workspace_dir


def _available_cpus() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "CPUS_PER_TASK"):
        raw = os.environ.get(key)
        if raw and raw.isdigit():
            return max(1, int(raw))
    return max(1, os.cpu_count() or 1)


def _sample_architectures(count: int, seed: int) -> list[list[int]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    widths = [32, 48, 64, 96, 128]
    archs: list[list[int]] = []
    for _ in range(count):
        depth = int(rng.integers(1, 5))
        archs.append([int(rng.choice(widths)) for _ in range(depth)])
    return archs


def _trial_command(
    *,
    trial_id: int,
    family: str,
    dataset_path: Path,
    output_dir: Path,
    log_path: Path,
    random_state: int,
    hidden_layer_sizes: list[int] | None,
    sklearn_mlp_max_iter: int,
) -> list[str]:
    namespace = f"demo_03/trial_{trial_id:02d}_{family}"
    cmd = [
        sys.executable,
        str(_EXAMPLE_DIR / "surge_train.py"),
        "--workflow",
        "m3dc1_mlp" if family == "mlp" else "m3dc1_rf",
        "--iteration",
        str(trial_id),
        "--namespace",
        namespace,
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(_EXAMPLE_DIR),
        "--log-file",
        str(log_path),
        "--run-tag-prefix",
        "resource_scale",
        "--random-state",
        str(random_state),
    ]
    if family == "mlp" and hidden_layer_sizes:
        cmd.extend(
            [
                "--hidden-layer-sizes",
                ",".join(str(x) for x in hidden_layer_sizes),
                "--sklearn-mlp-max-iter",
                str(sklearn_mlp_max_iter),
                "--dataset",
                "m3dc1",
            ]
        )
    return cmd


def _run_trial(
    *,
    trial_id: int,
    family: str,
    dataset_path: Path,
    output_dir: Path,
    log_dir: Path,
    random_state: int,
    hidden_layer_sizes: list[int] | None,
    sklearn_mlp_max_iter: int,
) -> dict:
    namespace = f"demo_03/trial_{trial_id:02d}_{family}"
    log_path = log_dir / f"trial_{trial_id:02d}_{family}.log"
    cmd = _trial_command(
        trial_id=trial_id,
        family=family,
        dataset_path=dataset_path,
        output_dir=output_dir,
        log_path=log_path,
        random_state=random_state,
        hidden_layer_sizes=hidden_layer_sizes,
        sklearn_mlp_max_iter=sklearn_mlp_max_iter,
    )
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write("=" * 80 + "\n")
            handle.write("Driver command output\n")
            handle.write("=" * 80 + "\n")
            handle.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                handle.write("\n")
    metrics = read_iteration_state(trial_id, "surge_metrics", namespace=namespace)
    return {
        "trial_id": trial_id,
        "family": family,
        "hidden_layer_sizes": hidden_layer_sizes,
        "random_state": random_state,
        **metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Resource-aware parallel surrogate search on full M3DC1.")
    ap.add_argument("--cpus-per-trial", type=int, default=4)
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--rf-baselines", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sklearn-mlp-max-iter", type=int, default=400)
    args = ap.parse_args()

    available = _available_cpus()
    cpus_per_trial = max(1, int(args.cpus_per_trial))
    planned_trials = max(1, available // cpus_per_trial)
    if args.max_trials is not None:
        planned_trials = min(planned_trials, max(1, int(args.max_trials)))
    rf_baselines = min(max(1, int(args.rf_baselines)), planned_trials)
    mlp_trials = max(0, planned_trials - rf_baselines)

    output_dir = _HERE / "output" / "demo_03"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    namespace = "demo_03"
    plan = training_row_plan(0, dataset="m3dc1", use_full_dataset=True)
    dataset_path = build_training_parquet(0, dataset="m3dc1", use_full_dataset=True, namespace=namespace)
    total_rows = pd.read_parquet(dataset_path).shape[0]
    archs = _sample_architectures(mlp_trials, seed=args.seed + 17)

    trials: list[dict] = []
    trial_id = 0
    for idx in range(rf_baselines):
        trials.append(
            {
                "trial_id": trial_id,
                "family": "rf",
                "random_state": args.seed + idx,
                "hidden_layer_sizes": None,
            }
        )
        trial_id += 1
    for idx, arch in enumerate(archs):
        trials.append(
            {
                "trial_id": trial_id,
                "family": "mlp",
                "random_state": args.seed + 100 + idx,
                "hidden_layer_sizes": arch,
            }
        )
        trial_id += 1

    started = time.time()
    print(
        "Demo 3: Resource-aware surrogate race | "
        f"available_cpus={available} | cpus_per_trial={cpus_per_trial} | "
        f"parallel_trials={planned_trials} | total_rows={total_rows} | "
        f"log_dir={log_dir}",
        flush=True,
    )
    print(
        f"Full M3DC1 table -> SURGE split 60/20/20. "
        f"Planned trials: {rf_baselines} rf baseline(s) + {mlp_trials} randomized mlp trial(s).",
        flush=True,
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=planned_trials) as pool:
        futures = [
            pool.submit(
                _run_trial,
                trial_id=int(trial["trial_id"]),
                family=str(trial["family"]),
                dataset_path=dataset_path,
                output_dir=_EXAMPLE_DIR,
                log_dir=log_dir,
                random_state=int(trial["random_state"]),
                hidden_layer_sizes=trial["hidden_layer_sizes"],
                sklearn_mlp_max_iter=int(args.sklearn_mlp_max_iter),
            )
            for trial in trials
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            splits = row.get("splits", {})
            detail = f"arch={row['hidden_layer_sizes']}" if row["hidden_layer_sizes"] else "baseline"
            print(
                f"done trial={row['trial_id']:02d} family={row['family']} "
                f"val_r2={float(row['val_r2']):.5f} val_rmse={float(row['val_rmse']):.6f} "
                f"split={splits.get('train', '?')}/{splits.get('val', '?')}/{splits.get('test', '?')} "
                f"{detail}",
                flush=True,
            )

    results.sort(key=lambda item: float(item["val_r2"]), reverse=True)
    elapsed = time.time() - started
    print("\nDemo 3 summary")
    print(f"Wall time: {elapsed:.1f}s")
    print(
        f"{'rank':>4}  {'trial':>5}  {'family':<5}  {'train':>6}  {'val':>6}  {'test':>6}  "
        f"{'val_r2':>9}  {'val_rmse':>10}  {'detail'}"
    )
    for rank, row in enumerate(results, start=1):
        splits = row.get("splits", {})
        detail = "baseline_rf" if row["family"] == "rf" else f"mlp{row['hidden_layer_sizes']}"
        print(
            f"{rank:>4}  {int(row['trial_id']):>5}  {row['family']:<5}  "
            f"{int(splits.get('train', 0)):>6}  {int(splits.get('val', 0)):>6}  {int(splits.get('test', 0)):>6}  "
            f"{float(row['val_r2']):>9.5f}  {float(row['val_rmse']):>10.6f}  {detail}",
            flush=True,
        )

    payload = {
        "available_cpus": available,
        "cpus_per_trial": cpus_per_trial,
        "parallel_trials": planned_trials,
        "total_rows": total_rows,
        "row_policy": plan["row_policy"],
        "results": results,
    }
    results_path = output_dir / "summary.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Workspace: {workspace_dir(namespace)}")
    print(f"Summary JSON: {results_path}")
    print(f"Logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
