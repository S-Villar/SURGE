#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from dataset_utils import build_training_parquet, training_row_plan, workspace_dir


def _visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        return [piece.strip() for piece in raw.split(",") if piece.strip()]
    try:
        import torch

        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _sample_architectures(count: int, seed: int) -> list[list[int]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    widths = [64, 96, 128, 160]
    archs: list[list[int]] = []
    for _ in range(count):
        depth = int(rng.integers(2, 5))
        archs.append([int(rng.choice(widths)) for _ in range(depth)])
    return archs


def _run_trial(
    *,
    trial_id: int,
    gpu_slot: str | None,
    dataset_path: Path,
    output_dir: Path,
    summary_dir: Path,
    hidden_layers: list[int],
    random_state: int,
    n_epochs: int,
    batch_size: int,
    num_workers: int,
    allow_cpu_fallback: bool,
) -> dict:
    run_tag = f"gpu_aware_torch_trial_{trial_id:02d}"
    summary_path = summary_dir / f"trial_{trial_id:02d}.json"
    cmd = [
        sys.executable,
        str(_HERE / "demo_04_gpu_trial.py"),
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--summary-path",
        str(summary_path),
        "--run-tag",
        run_tag,
        "--trial-id",
        str(trial_id),
        "--device",
        "cuda:0" if gpu_slot is not None else ("cpu" if allow_cpu_fallback else "auto"),
        "--hidden-layers",
        ",".join(str(x) for x in hidden_layers),
        "--random-state",
        str(random_state),
        "--n-epochs",
        str(n_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    env = os.environ.copy()
    if gpu_slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["driver_output"] = completed.stdout
    payload["requested_gpu_slot"] = gpu_slot
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU-aware SURGE demo using pytorch.mlp on full M3DC1.")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--trials-per-gpu", type=int, default=1)
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    ap.add_argument("--n-epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gpu_ids = _visible_gpu_ids()
    if not gpu_ids and not args.allow_cpu_fallback:
        raise SystemExit("No visible GPUs. Re-run inside a GPU allocation or pass --allow-cpu-fallback.")

    if gpu_ids:
        planned_trials = max(1, len(gpu_ids) * max(1, int(args.trials_per_gpu)))
    else:
        planned_trials = 1
    if args.max_trials is not None:
        planned_trials = min(planned_trials, max(1, int(args.max_trials)))

    output_dir = _HERE / "output" / "demo_04"
    summary_dir = output_dir / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    namespace = "demo_04"
    plan = training_row_plan(0, dataset="m3dc1", use_full_dataset=True)
    dataset_path = build_training_parquet(0, dataset="m3dc1", use_full_dataset=True, namespace=namespace)
    total_rows = pd.read_parquet(dataset_path).shape[0]
    archs = _sample_architectures(planned_trials, seed=args.seed + 101)

    assignments: list[str | None] = []
    if gpu_ids:
        for idx in range(planned_trials):
            assignments.append(gpu_ids[idx % len(gpu_ids)])
    else:
        assignments = [None]

    started = time.time()
    print(
        "Demo 4: GPU-aware surrogate search | "
        f"visible_gpus={gpu_ids or ['cpu-only']} | trials={planned_trials} | "
        f"total_rows={total_rows}",
        flush=True,
    )
    print(
        "GPU-capable SURGE model keys in this repo: pytorch.mlp, gpflow.gpr, gpflow.multi_kernel. "
        "This demo uses pytorch.mlp because its device placement is explicit and now verified.",
        flush=True,
    )

    results: list[dict] = []
    max_workers = len(gpu_ids) if gpu_ids else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _run_trial,
                trial_id=idx,
                gpu_slot=assignments[idx],
                dataset_path=dataset_path,
                output_dir=_EXAMPLE_DIR,
                summary_dir=summary_dir,
                hidden_layers=archs[idx],
                random_state=args.seed + idx,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                allow_cpu_fallback=args.allow_cpu_fallback,
            )
            for idx in range(planned_trials)
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            used = row.get("resources_used", {})
            effective = ((used.get("effective") or {}).get("device"))
            print(
                f"done trial={row['trial_id']:02d} val_r2={float(row['val_r2']):.5f} "
                f"val_rmse={float(row['val_rmse']):.6f} requested_slot={row.get('requested_gpu_slot')} "
                f"effective_device={effective} arch={row['hidden_layers']}",
                flush=True,
            )

    results.sort(key=lambda item: float(item["val_r2"]), reverse=True)
    elapsed = time.time() - started
    print("\nDemo 4 summary")
    print(f"Wall time: {elapsed:.1f}s")
    print(
        f"{'rank':>4}  {'trial':>5}  {'device':<8}  {'train':>6}  {'val':>6}  {'test':>6}  "
        f"{'val_r2':>9}  {'val_rmse':>10}  {'arch'}"
    )
    for rank, row in enumerate(results, start=1):
        splits = row.get("splits", {})
        used = row.get("resources_used", {})
        effective = ((used.get("effective") or {}).get("device", "?"))
        print(
            f"{rank:>4}  {int(row['trial_id']):>5}  {str(effective):<8}  "
            f"{int(splits.get('train', 0)):>6}  {int(splits.get('val', 0)):>6}  {int(splits.get('test', 0)):>6}  "
            f"{float(row['val_r2']):>9.5f}  {float(row['val_rmse']):>10.6f}  {row['hidden_layers']}",
            flush=True,
        )

    payload = {
        "visible_gpus": gpu_ids,
        "planned_trials": planned_trials,
        "total_rows": total_rows,
        "row_policy": plan["row_policy"],
        "results": results,
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Workspace: {workspace_dir(namespace)}")
    print(f"Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
