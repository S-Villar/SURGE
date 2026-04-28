# -*- coding: utf-8 -*-
"""
Example 3 - **Capability: metric-driven stop + architecture search.**

Randomized **sklearn MLP** ``hidden_layer_sizes`` each iteration; ROSE stops when
**validation R2** (from SURGE / ``last_surge_metrics.json``) meets a threshold
(default R2 >= 0.9).

Each iteration:

1. **Simulation** - grow Parquet (``--dataset synthetic`` or ``--dataset m3dc1``).
2. **Training** - random depth (1-4) and layer widths; SURGE MLP spec patched in memory.
3. **Active learning** - placeholder.
4. **Criterion** - ``val_r2`` with explicit comparison operator.

If R2 never reaches the target, the loop ends after ``--max-iter`` iterations.

Run from this directory with ROSE + SURGE on your env::

    cd examples/rose_orchestration
    export PYTHONPATH=/path/to/SURGE:$PYTHONPATH
    python example_03_rose_mlp_random_r2_stop.py
    python example_03_rose_mlp_random_r2_stop.py --max-iter 60 --r2-threshold 0.9 --r2-operator ">="
    python example_03_rose_mlp_random_r2_stop.py --r2-operator ">" --r2-threshold 0.9
    python example_03_rose_mlp_random_r2_stop.py --dataset synthetic --max-iter 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from dataset_utils import (
    build_training_parquet,
    read_iteration_state,
    training_row_plan,
    workspace_dir,
    write_iteration_state,
)
from demo_common import add_demo_cli, add_reporting_cli, mlp_search_task_kwargs
from live_report import LiveProgress, capture_output_to_log, default_log_path, reset_log_file
from orch_report import (
    RunTimer,
    print_phase_progress,
    print_run_header,
    print_run_report,
    print_iteration_progress,
    snapshot_iteration,
)
from surge_train import run_mlp_surge_random_arch

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))


def _banner(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72 + "\n", flush=True)


def sample_hidden_layers(rng: np.random.Generator) -> list[int]:
    """Random MLP depth and widths (surrogate search toy example)."""
    n_layers = int(rng.integers(1, 5))
    widths = np.array([16, 24, 32, 48, 64, 96, 128], dtype=int)
    return [int(x) for x in rng.choice(widths, size=n_layers)]


async def _demo(
    *,
    max_iter: int,
    workers: int,
    base_seed: int,
    r2_threshold: float,
    r2_operator: str,
    sklearn_mlp_max_iter: int,
    quiet: bool,
    dataset: str,
    live_progress: bool,
    log_file: str | None,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend

    from rose.al.active_learner import SequentialActiveLearner
    from rose.learner import LearnerConfig, TaskConfig
    use_full_training_set = dataset == "m3dc1"
    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_03"
    log_path = Path(log_file) if log_file else default_log_path("example_03")
    if live_progress:
        reset_log_file(log_path)
    timer = RunTimer()
    if live_progress:
        print(
            f"Example 3: Random MLP HPO | dataset={dataset} | "
            f"target val_r2 {r2_operator} {r2_threshold} | log={log_path}",
            flush=True,
        )
    else:
        print_run_header(
            example="3 - random MLP hidden_layer_sizes + ROSE stop on val_r2",
            max_iter=max_iter,
            workers=workers,
            quiet=quiet,
            extra=(
                f"Dataset={dataset}. Policy=random search over sklearn MLP hidden_layer_sizes. "
                f"Stop when val_r2 {r2_operator} {r2_threshold}. "
                f"sklearn MLP max_iter={sklearn_mlp_max_iter}."
            ),
        )
    if not quiet and not live_progress:
        _banner(
            "ROSE + SURGE - Example 3: random MLP configs, stop on val R2 "
            f"({r2_operator} {r2_threshold})",
        )

    if r2_operator not in (">", ">="):
        raise ValueError(f"Unsupported r2_operator {r2_operator!r}; use '>' or '>='.")
    cmp_op = r2_operator

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    raw_kw = mlp_search_task_kwargs(max_iter, base_seed)
    sched = {i: TaskConfig(kwargs=kw) for i, kw in raw_kw.items()}
    initial = LearnerConfig(
        simulation=sched,
        training=sched,
        active_learn=sched,
    )

    @acl.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        seed = int(kwargs.get("base_seed", base_seed))
        plan = training_row_plan(
            it,
            dataset=dataset,
            use_full_dataset=use_full_training_set,
        )
        if not quiet and not live_progress:
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=1,
                phase_total=4,
                phase_name="prepare fixed candidate dataset",
                detail=f"seed={seed}; {plan['n_rows']} samples available to SURGE before its train/val/test split",
            )
        path = build_training_parquet(
            it,
            dataset=dataset,
            use_full_dataset=use_full_training_set,
        )
        n = pd.read_parquet(path).shape[0]
        meta = {
            "iteration": it,
            "n_rows": n,
            "n_rows_requested": plan["n_rows_requested"],
            "n_rows_total": plan["n_rows_total"],
            "row_policy": plan["row_policy"],
            "dataset": str(path),
            "base_seed": seed,
        }
        workspace_dir()
        write_iteration_state(it, "simulation", meta)
        if not quiet and not live_progress:
            print(f"  dataset ready  wrote {n} samples to {path.name}", flush=True)
        return meta

    @acl.training_task(as_executable=False)
    async def training(sim_result, **kwargs):
        it = int(kwargs.get("iteration", sim_result["iteration"]))
        bseed = int(kwargs.get("base_seed", sim_result.get("base_seed", base_seed)))
        rng = np.random.default_rng(bseed + 1_009 * it + 17)
        arch = sample_hidden_layers(rng)
        candidate_meta = {
            "iteration": it,
            "policy": "random_search",
            "base_seed": bseed,
            "candidate_id": f"mlp_candidate_{it:03d}",
            "hidden_layer_sizes": arch,
            "search_space": {"depth": [1, 4], "widths": [16, 24, 32, 48, 64, 96, 128]},
        }
        write_iteration_state(it, "candidate", candidate_meta)
        if not quiet and not live_progress:
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=2,
                phase_total=4,
                phase_name="evaluate MLP candidate with SURGE",
                detail=f"candidate={candidate_meta['candidate_id']}; arch={arch}",
            )
        out = run_mlp_surge_random_arch(
            it,
            arch,
            dataset=dataset,
            sklearn_mlp_max_iter=sklearn_mlp_max_iter,
            random_state=(bseed + it) % (2**31),
            namespace="example_03",
            dataset_path=sim_result["dataset"],
            output_dir=_EX,
            log_path=log_path if live_progress else None,
            verbose=not quiet and not live_progress,
        )
        return {"simulation": sim_result, "surge": out}

    @acl.active_learn_task(as_executable=False)
    async def active_learn(sim_result, train_bundle, **kwargs):
        it = int(kwargs.get("iteration", train_bundle["simulation"]["iteration"]))
        surge = train_bundle["surge"]
        if not quiet and not live_progress:
            splits = surge.get("splits", {})
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=3,
                phase_total=4,
                phase_name="record HPO candidate result",
                detail=(
                    f"arch={surge.get('hidden_layer_sizes')}; val_r2={float(surge['val_r2']):.5f}; "
                    f"split train/val/test={splits.get('train', '?')}/{splits.get('val', '?')}/{splits.get('test', '?')}"
                ),
            )
        write_iteration_state(
            it,
            "active",
            {"iteration": it, "policy": "random_search", "surge": surge, "note": "candidate result recorded"},
        )
        return {"iteration": it, "train": train_bundle}

    @acl.as_stop_criterion(
        metric_name="val_r2",
        threshold=r2_threshold,
        operator=cmp_op,
        as_executable=False,
    )
    async def stop_on_val_r2(*args, **kwargs):
        """ROSE treats ``val_r2`` as a custom metric; ``operator`` must be explicit."""
        crit_it = int(kwargs.get("iteration", -1))
        if crit_it >= 0:
            meta = read_iteration_state(crit_it, "surge_metrics")
        else:
            meta = json.loads((workspace_dir() / "last_surge_metrics.json").read_text(encoding="utf-8"))
            crit_it = int(meta.get("iteration", 0))
        r2 = float(meta["val_r2"])
        if not quiet and not live_progress:
            print_phase_progress(
                rose_iter=crit_it,
                max_iter=max_iter,
                phase_index=4,
                phase_total=4,
                phase_name="evaluate stop criterion",
                detail=f"val_r2={r2:.6f}; threshold {r2_operator} {r2_threshold}",
            )
        write_iteration_state(crit_it, "criterion", {"iteration": crit_it, "metric": "val_r2", "value": r2})
        return r2

    if not quiet and not live_progress:
        _banner("Starting learner - stops early when R2 criterion is met, else cap at max_iter")
    report_rows: list[dict] = []

    async def _run_stream(progress: LiveProgress | None = None) -> None:
        async for state in acl.start(max_iter=max_iter, initial_config=initial):
            row = snapshot_iteration(workspace_dir(), state.iteration)
            report_rows.append(row)
            if progress:
                progress.update(
                    1,
                    r2=f"{float(row.get('val_r2', 0.0)):.4f}",
                    best=f"{max(float(r.get('val_r2', 0.0)) for r in report_rows):.4f}",
                    arch=str(row.get("hidden_layer_sizes", "?")),
                )
            else:
                print_iteration_progress(
                    report_rows,
                    max_iter=max_iter,
                    score_metric="val_r2",
                    higher_is_better=True,
                    mode_label="hpo",
                    threshold=r2_threshold,
                    threshold_operator=r2_operator,
                )
                if not quiet:
                    print(
                        f"  orchestration  metric(val_r2)={state.metric_value!r}  should_stop={state.should_stop}",
                        flush=True,
                    )
            if state.iteration >= max_iter - 1:
                acl.stop()
                break

    if live_progress:
        with LiveProgress(total=max_iter, desc="MLP candidates", enabled=True, unit="model") as progress:
            with capture_output_to_log(log_path):
                await _run_stream(progress)
                await acl.shutdown()
    else:
        await _run_stream(None)
        await acl.shutdown()
    print_run_report(
        report_rows,
        title=f"Example 3 summary (random MLP, R2 {r2_operator} {r2_threshold})",
        elapsed_sec=timer.seconds(),
    )
    print(f"Log file: {log_path}")
    if not quiet and not live_progress:
        _banner("Done. Inspect workspace/last_surge_metrics.json and runs/rose_mlp_rand_*")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*GPflow.*", category=UserWarning)
    parser = argparse.ArgumentParser(
        description="Randomized MLP architectures; ROSE stop when val R2 passes threshold.",
    )
    add_demo_cli(parser)
    add_reporting_cli(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducible architecture draws per iteration.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=0.9,
        dest="r2_threshold",
        metavar="R2",
        help="Stop when val_r2 compares true against this (default 0.9).",
    )
    parser.add_argument(
        "--r2-operator",
        type=str,
        default=">=",
        choices=(">", ">="),
        dest="r2_operator",
        help="Comparison for val_r2 vs threshold (default >=). Use '>' for strict.",
    )
    parser.add_argument(
        "--sklearn-mlp-max-iter",
        type=int,
        default=800,
        metavar="N",
        help="sklearn MLPRegressor max_iter (default 800).",
    )
    args = parser.parse_args()
    # Higher default iteration cap than Examples 1-2 unless --max-iter is passed.
    if "--max-iter" not in sys.argv:
        args.max_iter = 48

    asyncio.run(
        _demo(
            max_iter=args.max_iter,
            workers=args.workers,
            base_seed=args.seed,
            r2_threshold=args.r2_threshold,
            r2_operator=args.r2_operator,
            sklearn_mlp_max_iter=args.sklearn_mlp_max_iter,
            quiet=args.quiet,
            dataset=args.dataset,
            live_progress=not args.no_live_progress,
            log_file=args.log_file,
        )
    )


if __name__ == "__main__":
    main()
