#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 4 - ROSE ParallelActiveLearner races multiple SURGE models."""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
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
from demo_common import add_dataset_cli, add_reporting_cli, canonical_workflow, workflow_fixed_rows
from live_report import LiveProgress, capture_output_to_log, default_log_path, reset_log_file
from orch_report import RunTimer, print_run_header, progress_bar

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))


def _candidate_configs(dataset: str, candidates: list[str], max_iter: int, growing_pool: bool):
    from rose.learner import LearnerConfig, TaskConfig

    configs = []
    for idx, family in enumerate(candidates):
        workflow = canonical_workflow(dataset, family)
        label = f"{idx}_{family}"
        kwargs = {
            "learner_label": label,
            "workflow": workflow,
            "dataset": dataset,
            "growing_pool": growing_pool,
        }
        schedule = {i: TaskConfig(kwargs={**kwargs, "iteration": i}) for i in range(max_iter + 1)}
        schedule[-1] = TaskConfig(kwargs={**kwargs, "iteration": max_iter})
        configs.append(
            LearnerConfig(
                simulation=schedule,
                training=schedule,
                active_learn=schedule,
                criterion=schedule,
            )
        )
    return configs


async def _demo(
    *,
    dataset: str,
    candidates: list[str],
    max_iter: int,
    workers: int,
    r2_threshold: float,
    growing_pool: bool,
    live_progress: bool,
    log_file: str | None,
    quiet: bool,
) -> None:
    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend
    from rose.al.active_learner import ParallelActiveLearner

    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_04"
    log_path = Path(log_file) if log_file else default_log_path("example_04")
    if live_progress:
        reset_log_file(log_path)
    timer = RunTimer()
    if not live_progress:
        print_run_header(
            example="4 - ParallelActiveLearner model race",
            max_iter=max_iter,
            workers=workers,
            quiet=quiet,
            extra=(
                f"Dataset={dataset}. Candidates={candidates}. "
                f"Pool policy={'growing subset' if growing_pool else 'full dataset'}. "
                f"Stop each learner when val_r2 >= {r2_threshold}."
            ),
        )
    else:
        print(
            f"Example 4: Parallel SURGE model race | dataset={dataset} | "
            f"candidates={candidates} | log={log_path}",
            flush=True,
        )

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    learner = ParallelActiveLearner(asyncflow)

    @learner.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        label = str(kwargs["learner_label"])
        ns = f"example_04/{label}"
        workflow = str(kwargs["workflow"])
        fixed_rows = workflow_fixed_rows(dataset, workflow, growing_pool=growing_pool)
        use_full_dataset = dataset == "m3dc1" and not growing_pool and fixed_rows is None
        plan = training_row_plan(
            it,
            dataset=dataset,
            fixed_rows=fixed_rows,
            use_full_dataset=use_full_dataset,
        )
        path = build_training_parquet(
            it,
            dataset=dataset,
            fixed_rows=fixed_rows,
            use_full_dataset=use_full_dataset,
            namespace=ns,
        )
        n = pd.read_parquet(path).shape[0]
        meta = {
            "iteration": it,
            "learner_label": label,
            "workflow": workflow,
            "dataset": str(path),
            "n_rows": n,
            "n_rows_total": plan["n_rows_total"],
            "row_policy": plan["row_policy"],
        }
        write_iteration_state(it, "simulation", meta, namespace=ns)
        return meta

    @learner.training_task(as_executable=False)
    async def training(sim_result, **kwargs):
        it = int(kwargs.get("iteration", sim_result["iteration"]))
        label = str(kwargs["learner_label"])
        ns = f"example_04/{label}"
        cmd = [
            sys.executable,
            str(_EX / "surge_train.py"),
            "--workflow",
            str(kwargs["workflow"]),
            "--iteration",
            str(it),
            "--namespace",
            ns,
            "--dataset-path",
            str(sim_result["dataset"]),
            "--output-dir",
            str(_EX),
            "--run-tag-prefix",
            f"rose_parallel_{label}",
        ]
        if live_progress:
            cmd.extend(["--log-file", str(log_path)])
        if not quiet and not live_progress:
            cmd.append("--verbose")
        env = os.environ.copy()
        root = str(_EX.parents[1])
        prev = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = root if not prev else root + os.pathsep + prev
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if completed.stdout and live_progress:
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write(f"Parallel learner {label} command output:\n")
                handle.write(completed.stdout)
        out = read_iteration_state(it, "surge_metrics", namespace=ns)
        return {"simulation": sim_result, "surge": out}

    @learner.active_learn_task(as_executable=False)
    async def active_learn(sim_result, train_bundle, **kwargs):
        it = int(kwargs.get("iteration", train_bundle["simulation"]["iteration"]))
        label = str(kwargs["learner_label"])
        ns = f"example_04/{label}"
        surge = train_bundle["surge"]
        decision = {
            "iteration": it,
            "learner_label": label,
            "policy": "monitor_best_val_r2",
            "val_r2": surge["val_r2"],
            "val_rmse": surge["val_rmse"],
            "splits": surge.get("splits", {}),
        }
        write_iteration_state(it, "active", decision, namespace=ns)
        return {"iteration": it, "learner_label": label, "train": train_bundle}

    @learner.as_stop_criterion(
        metric_name="val_r2",
        threshold=r2_threshold,
        operator=">=",
        as_executable=False,
    )
    async def stop_on_r2(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        label = str(kwargs["learner_label"])
        ns = f"example_04/{label}"
        meta = read_iteration_state(it, "surge_metrics", namespace=ns)
        r2 = float(meta["val_r2"])
        forced_stop = it >= max_iter - 1 and r2 < r2_threshold
        write_iteration_state(
            it,
            "criterion",
            {
                "iteration": it,
                "metric": "val_r2",
                "value": r2,
                "forced_stop_after_max_iter": forced_stop,
            },
            namespace=ns,
        )
        return r2_threshold if forced_stop else r2

    configs = _candidate_configs(dataset, candidates, max_iter, growing_pool)
    rows: list[dict] = []

    async def _run_stream(progress: LiveProgress | None = None) -> None:
        async for state in learner.start(
            parallel_learners=len(candidates),
            max_iter=max_iter,
            learner_configs=configs,
        ):
            label = candidates[int(state.learner_id)]
            ns = f"example_04/{state.learner_id}_{label}"
            meta = read_iteration_state(state.iteration, "surge_metrics", namespace=ns)
            rows.append({"learner": label, **meta})
            if progress:
                progress.update(
                    1,
                    learner=label,
                    r2=f"{float(meta['val_r2']):.4f}",
                    rmse=f"{float(meta['val_rmse']):.4g}",
                )
            else:
                print(
                    f"parallel {progress_bar(len(rows), len(candidates) * max_iter)} "
                    f"learner={label} iter={state.iteration} "
                    f"val_r2={float(meta['val_r2']):.5f} val_rmse={float(meta['val_rmse']):.5f} "
                    f"split={meta.get('splits', {}).get('train', '?')}/"
                    f"{meta.get('splits', {}).get('val', '?')}/"
                    f"{meta.get('splits', {}).get('test', '?')}",
                    flush=True,
                )
            if len(rows) >= len(candidates) * max_iter:
                learner.stop()
                break

    if live_progress:
        with LiveProgress(
            total=len(candidates) * max_iter,
            desc="SURGE candidates",
            enabled=True,
            unit="model",
        ) as progress:
            with capture_output_to_log(log_path):
                await _run_stream(progress)
                await learner.shutdown()
    else:
        await _run_stream(None)
        await learner.shutdown()
    rows.sort(key=lambda row: float(row["val_r2"]), reverse=True)
    print("\nExample 4 summary")
    print(f"Wall time: {timer.seconds():.1f}s")
    print(f"{'rank':>4}  {'learner':<8}  {'workflow':<12}  {'val_r2':>9}  {'val_rmse':>10}  {'run_tag'}")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4}  {row['learner']:<8}  {row['workflow']:<12}  "
            f"{float(row['val_r2']):>9.5f}  {float(row['val_rmse']):>10.6f}  {row['run_tag']}",
            flush=True,
        )
    print(f"Workspace: {workspace_dir('example_04')}")
    print(f"Log file: {log_path}")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*GPflow.*", category=UserWarning)
    parser = argparse.ArgumentParser(description="Parallel ROSE/Rhapsody orchestration of SURGE candidates.")
    add_dataset_cli(parser)
    add_reporting_cli(parser)
    parser.add_argument(
        "--candidates",
        default="rf,mlp",
        help="Comma-separated model families: rf,mlp,gpr,gpflow_gpr.",
    )
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--r2-threshold", type=float, default=0.95)
    args = parser.parse_args()
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    invalid = sorted(set(candidates) - {"rf", "mlp", "gpr", "gpflow_gpr"})
    if invalid:
        raise ValueError(f"Unsupported candidates: {invalid}; use rf and/or mlp.")
    if len(candidates) < 2:
        raise ValueError("Example 4 needs at least two candidates for ParallelActiveLearner.")
    asyncio.run(
        _demo(
            dataset=args.dataset,
            candidates=candidates,
            max_iter=args.max_iter,
            workers=args.workers,
            r2_threshold=args.r2_threshold,
            growing_pool=args.growing_pool,
            live_progress=not args.no_live_progress,
            log_file=args.log_file,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
