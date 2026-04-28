#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1 - ROSE ``SequentialActiveLearner`` with **in-process** tasks (no shell).

simulation -> training (SURGE) -> active_learn (stub) -> criterion (val MSE).
Default dataset is ``m3dc1`` when the PKL is installed.

On many clusters ``python`` is 2.7; this demo needs **Python 3.10+** (conda env or ``python3.11`` etc.).
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

import pandas as pd

from dataset_utils import (
    build_training_parquet,
    read_iteration_state,
    training_row_plan,
    workspace_dir,
    write_iteration_state,
)
from demo_common import add_demo_cli, add_reporting_cli, inprocess_task_kwargs, workflow_fixed_rows
from live_report import LiveProgress, capture_output_to_log, default_log_path, reset_log_file
from orch_report import (
    RunTimer,
    print_phase_progress,
    print_run_header,
    print_run_report,
    print_iteration_progress,
    snapshot_iteration,
)
from surge_train import run_one_surge

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))


def _banner(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72 + "\n", flush=True)


async def _demo(
    *,
    max_iter: int,
    workers: int,
    dataset: str,
    workflow_family: str,
    growing_pool: bool,
    live_progress: bool,
    log_file: str | None,
    quiet: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend

    from rose.al.active_learner import SequentialActiveLearner
    from rose.learner import LearnerConfig, TaskConfig
    from rose.metrics import MEAN_SQUARED_ERROR_MSE

    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_01"
    log_path = Path(log_file) if log_file else default_log_path("example_01")
    if live_progress:
        reset_log_file(log_path)
    timer = RunTimer()
    if live_progress:
        print(
            f"Example 1: Sequential ROSE+SURGE | dataset={dataset} | "
            f"workflow={workflow_family} | log={log_path}",
            flush=True,
        )
    else:
        print_run_header(
            example="1 - in-process SequentialActiveLearner + ThreadPoolExecutor",
            max_iter=max_iter,
            workers=workers,
            quiet=quiet,
            extra=(
                f"Dataset mode: {dataset}. Workflow family: {workflow_family}. "
                f"Pool policy: {'growing subset' if growing_pool else 'full dataset'}. "
                "Phases: sim -> train (SURGE) -> active_learn (stub) -> criterion."
            ),
        )
    if not quiet and not live_progress:
        _banner("ROSE + SURGE - Example 1: in-process orchestration")

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    raw_kw = inprocess_task_kwargs(max_iter, dataset, workflow_family)
    sched = {i: TaskConfig(kwargs=kw) for i, kw in raw_kw.items()}
    initial = LearnerConfig(
        simulation=sched,
        training=sched,
        active_learn=sched,
    )

    @acl.simulation_task(as_executable=False)
    async def simulation(*args, **kwargs):
        it = int(kwargs.get("iteration", 0))
        wf = kwargs.get("workflow", "rf")
        fixed_rows = workflow_fixed_rows(dataset, wf, growing_pool=growing_pool)
        use_full_dataset = dataset == "m3dc1" and not growing_pool and fixed_rows is None
        plan = training_row_plan(
            it,
            dataset=dataset,
            fixed_rows=fixed_rows,
            use_full_dataset=use_full_dataset,
        )
        if not quiet and not live_progress:
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=1,
                phase_total=4,
                phase_name="prepare dataset slice",
                detail=f"next workflow={wf}; {plan['n_rows']} samples selected",
            )
        path = build_training_parquet(
            it,
            dataset=dataset,
            fixed_rows=fixed_rows,
            use_full_dataset=use_full_dataset,
        )
        n = pd.read_parquet(path).shape[0]
        meta = {
            "iteration": it,
            "n_rows": n,
            "n_rows_requested": plan["n_rows_requested"],
            "n_rows_total": plan["n_rows_total"],
            "row_policy": plan["row_policy"],
            "dataset": str(path),
            "workflow": wf,
        }
        workspace_dir()
        write_iteration_state(it, "simulation", meta)
        if not quiet and not live_progress:
            available = meta["n_rows_total"]
            if available:
                detail = f"wrote {n}/{available} samples to {path.name}"
            else:
                detail = f"wrote {n} samples to {path.name}"
            print(f"  dataset ready  {detail}", flush=True)
        return meta

    @acl.training_task(as_executable=False)
    async def training(sim_result, **kwargs):
        it = int(kwargs.get("iteration", sim_result["iteration"]))
        wf = str(kwargs.get("workflow", "rf"))
        if not quiet and not live_progress:
            detail = f"workflow={wf}; SURGE receives {sim_result['n_rows']} samples"
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=2,
                phase_total=4,
                phase_name="train surrogate with SURGE",
                detail=detail,
            )
        out = run_one_surge(
            wf,
            it,
            namespace="example_01",
            dataset_path=sim_result["dataset"],
            output_dir=_EX,
            log_path=log_path if live_progress else None,
            verbose=not quiet and not live_progress,
        )
        return {"simulation": sim_result, "surge": out}

    @acl.active_learn_task(as_executable=False)
    async def active_learn(sim_result, train_bundle, **kwargs):
        it = int(kwargs.get("iteration", train_bundle["simulation"]["iteration"]))
        wf = train_bundle["surge"]["workflow"]
        if not quiet and not live_progress:
            vrmse = float(train_bundle["surge"]["val_rmse"])
            vr2 = float(train_bundle["surge"].get("val_r2", float("nan")))
            splits = train_bundle["surge"].get("splits", {})
            detail = f"workflow={wf}; val_rmse={vrmse:.5f}; val_r2={vr2:.4f}; acquisition=stub"
            if splits:
                detail += (
                    f"; split train/val/test="
                    f"{splits.get('train', '?')}/{splits.get('val', '?')}/{splits.get('test', '?')}"
                )
            print_phase_progress(
                rose_iter=it,
                max_iter=max_iter,
                phase_index=3,
                phase_total=4,
                phase_name="update active-learning state",
                detail=detail,
            )
        active_state = {
            "iteration": it,
            "policy": "stub",
            "surge": train_bundle["surge"],
        }
        write_iteration_state(it, "active", active_state)
        return {"iteration": it, "train": train_bundle}

    @acl.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=0.0,
        as_executable=False,
    )
    async def never_stop_until_max(*args, **kwargs):
        crit_it = int(kwargs.get("iteration", -1))
        if crit_it >= 0:
            meta = read_iteration_state(crit_it, "surge_metrics")
        else:
            meta = json.loads((workspace_dir() / "last_surge_metrics.json").read_text(encoding="utf-8"))
            crit_it = int(meta.get("iteration", 0))
        mse = float(meta["val_mse"])
        if not quiet and not live_progress:
            print_phase_progress(
                rose_iter=crit_it,
                max_iter=max_iter,
                phase_index=4,
                phase_total=4,
                phase_name="evaluate stop criterion",
                detail=f"val_mse={mse:.6e}; threshold=0",
            )
        write_iteration_state(crit_it, "criterion", {"iteration": crit_it, "metric": "val_mse", "value": mse})
        return mse

    if not quiet and not live_progress:
        _banner("Starting learner (sim -> train -> active -> criterion)")
    report_rows = []

    async def _run_stream(progress: LiveProgress | None = None) -> None:
        async for state in acl.start(max_iter=max_iter, initial_config=initial):
            row = snapshot_iteration(workspace_dir(), state.iteration)
            report_rows.append(row)
            if progress:
                progress.update(
                    1,
                    r2=f"{float(row.get('val_r2', 0.0)):.4f}",
                    rmse=f"{float(row.get('val_rmse', 0.0)):.4g}",
                )
            else:
                print_iteration_progress(
                    report_rows,
                    max_iter=max_iter,
                    score_metric="val_r2",
                    higher_is_better=True,
                    mode_label="campaign",
                )
                if not quiet:
                    print(
                        f"  orchestration  metric={state.metric_value!r}  should_stop={state.should_stop}",
                        flush=True,
                    )
            if state.iteration >= max_iter - 1:
                acl.stop()
                break

    if live_progress:
        with LiveProgress(total=max_iter, desc="ROSE iterations", enabled=True, unit="iter") as progress:
            with capture_output_to_log(log_path):
                await _run_stream(progress)
                await acl.shutdown()
    else:
        await _run_stream(None)
        await acl.shutdown()
    print_run_report(
        report_rows,
        title=f"Example 1 summary ({dataset})",
        elapsed_sec=timer.seconds(),
    )
    print(f"Log file: {log_path}")
    if not quiet and not live_progress:
        _banner("Done. SURGE runs under runs/rose_*")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*GPflow.*", category=UserWarning)
    parser = argparse.ArgumentParser(description="ROSE in-process + SURGE.")
    add_demo_cli(parser)
    add_reporting_cli(parser)
    args = parser.parse_args()
    asyncio.run(
        _demo(
            max_iter=args.max_iter,
            workers=args.workers,
            dataset=args.dataset,
            workflow_family=args.workflow_family,
            growing_pool=args.growing_pool,
            live_progress=not args.no_live_progress,
            log_file=args.log_file,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
