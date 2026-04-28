#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 5 - ROSE UQ learner monitors SURGE MLP ensemble uncertainty."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

from dataset_utils import (
    build_training_parquet,
    read_iteration_state,
    training_row_plan,
    workspace_dir,
    write_iteration_state,
)
from demo_common import add_dataset_cli, add_reporting_cli
from live_report import LiveProgress, capture_output_to_log, default_log_path, reset_log_file
from orch_report import RunTimer, print_run_header, target_bar
from surge_train import run_mlp_ensemble_uq

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))


async def _demo(
    *,
    dataset: str,
    max_iter: int,
    workers: int,
    ensemble_n: int,
    sklearn_mlp_max_iter: int,
    uncertainty_threshold: float,
    growing_pool: bool,
    live_progress: bool,
    log_file: str | None,
    quiet: bool,
) -> None:
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend
    from rose.learner import TaskConfig
    from rose.uq.uq_active_learner import SeqUQLearner
    from rose.uq.uq_learner import UQLearnerConfig

    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_05"
    namespace = "example_05/uq_ensemble"
    log_path = Path(log_file) if log_file else default_log_path("example_05")
    if live_progress:
        reset_log_file(log_path)
    timer = RunTimer()
    if not live_progress:
        print_run_header(
            example="5 - UQ-guided MLP ensemble monitor",
            max_iter=max_iter,
            workers=workers,
            quiet=quiet,
            extra=(
                f"Dataset={dataset}. SURGE sklearn.mlp ensemble_n={ensemble_n}. "
                f"ROSE UQ stop when mean validation std <= {uncertainty_threshold}."
            ),
        )
    else:
        print(
            f"Example 5: SURGE MLP ensemble UQ | dataset={dataset} | "
            f"ensemble_n={ensemble_n} | log={log_path}",
            flush=True,
        )

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    learner = SeqUQLearner(asyncflow)

    @learner.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        use_full_dataset = dataset == "m3dc1" and not growing_pool
        plan = training_row_plan(it, dataset=dataset, use_full_dataset=use_full_dataset)
        path = build_training_parquet(
            it,
            dataset=dataset,
            use_full_dataset=use_full_dataset,
            namespace=namespace,
        )
        n = pd.read_parquet(path).shape[0]
        meta = {
            "iteration": it,
            "dataset": str(path),
            "n_rows": n,
            "n_rows_total": plan["n_rows_total"],
            "row_policy": plan["row_policy"],
        }
        write_iteration_state(it, "simulation", meta, namespace=namespace)
        return meta

    @learner.training_task(as_executable=False)
    async def training(sim_result, **kwargs):
        it = int(kwargs.get("iteration", sim_result["iteration"]))
        out = run_mlp_ensemble_uq(
            it,
            dataset=dataset,
            namespace=namespace,
            dataset_path=sim_result["dataset"],
            output_dir=_EX,
            log_path=log_path if live_progress else None,
            ensemble_n=ensemble_n,
            max_iter=sklearn_mlp_max_iter,
            random_state=42 + it,
            run_tag_prefix="rose_uq_mlp_ensemble",
            verbose=not quiet and not live_progress,
        )
        return {"simulation": sim_result, "surge": out}

    @learner.prediction_task(as_executable=False)
    async def prediction(training_result, **kwargs):
        it = int(kwargs.get("iteration", 0))
        meta = read_iteration_state(it, "surge_metrics", namespace=namespace)
        payload = {
            "iteration": it,
            "uq_artifact": meta.get("uq_artifact"),
            "uq_val_std_mean": meta.get("uq_val_std_mean"),
            "uq_val_std_max": meta.get("uq_val_std_max"),
        }
        write_iteration_state(it, "prediction", payload, namespace=namespace)
        return payload

    @learner.uncertainty_quantification(
        uq_metric_name="mean_val_std",
        threshold=uncertainty_threshold,
        query_size=0,
        operator="<=",
        as_executable=False,
    )
    async def check_uncertainty(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        meta = read_iteration_state(it, "surge_metrics", namespace=namespace)
        value = float(meta.get("uq_val_std_mean", float("inf")))
        write_iteration_state(it, "uncertainty", {"iteration": it, "metric": "mean_val_std", "value": value}, namespace=namespace)
        return value

    @learner.active_learn_task(as_executable=False)
    async def active_learn(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        meta = read_iteration_state(it, "surge_metrics", namespace=namespace)
        value = float(meta.get("uq_val_std_mean", float("inf")))
        decision = {
            "iteration": it,
            "policy": "monitor_validation_uncertainty",
            "mean_val_std": value,
            "decision": "stop_when_uncertainty_below_threshold",
        }
        write_iteration_state(it, "active", decision, namespace=namespace)
        return decision

    @learner.as_stop_criterion(
        metric_name="val_r2",
        threshold=0.99,
        operator=">=",
        as_executable=False,
    )
    async def stop_on_r2(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        meta = read_iteration_state(it, "surge_metrics", namespace=namespace)
        r2 = float(meta["val_r2"])
        return 0.99 if it >= max_iter - 1 and r2 < 0.99 else r2

    schedule = {
        i: TaskConfig(kwargs={"iteration": i})
        for i in range(max_iter + 1)
    }
    config = UQLearnerConfig(
        simulation=schedule,
        training=schedule,
        prediction=schedule,
        uncertainty=schedule,
        active_learn=schedule,
        criterion=schedule,
    )

    rows: list[dict] = []

    async def _run_stream(progress: LiveProgress | None = None) -> None:
        async for state in learner.start(
            model_names=["surge_mlp_ensemble"],
            num_predictions=1,
            max_iter=max_iter,
            learning_config=config,
        ):
            meta = read_iteration_state(state.iteration, "surge_metrics", namespace=namespace)
            uq = float(meta.get("uq_val_std_mean", float("nan")))
            rows.append(meta)
            if progress:
                progress.update(
                    1,
                    r2=f"{float(meta['val_r2']):.4f}",
                    mean_std=f"{uq:.5f}",
                    target=f"{uncertainty_threshold:.5f}",
                )
            else:
                print(
                    f"uq iter={state.iteration} val_r2={float(meta['val_r2']):.5f} "
                    f"mean_val_std={uq:.6f} "
                    f"{target_bar(uq, uncertainty_threshold, higher_is_better=False)}",
                    flush=True,
                )
            if state.iteration >= max_iter - 1:
                learner.stop()
                break

    if live_progress:
        with LiveProgress(total=max_iter, desc="SURGE ensemble UQ", enabled=True, unit="iter") as progress:
            with capture_output_to_log(log_path):
                await _run_stream(progress)
                await learner.shutdown()
    else:
        await _run_stream(None)
        await learner.shutdown()
    print("\nExample 5 summary")
    print(f"Wall time: {timer.seconds():.1f}s")
    print(
        f"{'#':>3}  {'total':>8}  {'train':>8}  {'val':>8}  {'test':>8}  "
        f"{'val_r2':>9}  {'mean_std':>10}  {'max_std':>10}  {'uq_artifact'}"
    )
    for row in rows:
        splits = row.get("splits", {})
        train_n = int(splits.get("train", 0))
        val_n = int(splits.get("val", 0))
        test_n = int(splits.get("test", 0))
        total_n = train_n + val_n + test_n
        print(
            f"{int(row['iteration']):>3}  {total_n:>8}  {train_n:>8}  {val_n:>8}  {test_n:>8}  "
            f"{float(row['val_r2']):>9.5f}  "
            f"{float(row.get('uq_val_std_mean', float('nan'))):>10.6f}  "
            f"{float(row.get('uq_val_std_max', float('nan'))):>10.6f}  "
            f"{row.get('uq_artifact')}",
            flush=True,
        )
    print(f"Workspace: {workspace_dir('example_05')}")
    print(f"Log file: {log_path}")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*GPflow.*", category=UserWarning)
    parser = argparse.ArgumentParser(description="ROSE UQ learner around SURGE MLP ensemble UQ.")
    add_dataset_cli(parser)
    add_reporting_cli(parser)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ensemble-n", type=int, default=3)
    parser.add_argument("--sklearn-mlp-max-iter", type=int, default=250)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.015)
    args = parser.parse_args()
    asyncio.run(
        _demo(
            dataset=args.dataset,
            max_iter=args.max_iter,
            workers=args.workers,
            ensemble_n=args.ensemble_n,
            sklearn_mlp_max_iter=args.sklearn_mlp_max_iter,
            uncertainty_threshold=args.uncertainty_threshold,
            growing_pool=args.growing_pool,
            live_progress=not args.no_live_progress,
            log_file=args.log_file,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
