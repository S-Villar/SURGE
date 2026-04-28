#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1 - ROSE ``SequentialActiveLearner`` with **in-process** tasks (no shell).

simulation -> training (SURGE) -> active_learn (stub) -> criterion (val MSE).
Use ``--dataset synthetic`` (default) or ``--dataset m3dc1`` if the PKL is installed.

On many clusters ``python`` is 2.7; this demo needs **Python 3.10+** (conda env or ``python3.11`` etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from dataset_utils import build_training_parquet, workspace_dir
from demo_common import add_demo_cli, add_reporting_cli, inprocess_task_kwargs
from orch_report import (
    RunTimer,
    print_run_header,
    print_run_report,
    progress_label,
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


async def _demo(*, max_iter: int, workers: int, dataset: str, quiet: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend

    from rose.al.active_learner import SequentialActiveLearner
    from rose.learner import LearnerConfig, TaskConfig
    from rose.metrics import MEAN_SQUARED_ERROR_MSE

    timer = RunTimer()
    print_run_header(
        example="1 - in-process SequentialActiveLearner + ThreadPoolExecutor",
        max_iter=max_iter,
        workers=workers,
        quiet=quiet,
        extra=f"Dataset mode: {dataset}. Phases: sim -> train (SURGE) -> active_learn (stub) -> criterion.",
    )
    if not quiet:
        _banner("ROSE + SURGE - Example 1: in-process orchestration")

    engine = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    raw_kw = inprocess_task_kwargs(max_iter, dataset)
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
        if not quiet:
            print(
                f"{progress_label(it, max_iter, 'sim')} refresh Parquet; next SURGE spec={wf}",
                flush=True,
            )
        path = build_training_parquet(it, dataset=dataset)
        n = pd.read_parquet(path).shape[0]
        meta = {"iteration": it, "n_rows": n, "dataset": str(path)}
        workspace_dir()
        with (workspace_dir() / "last_simulation.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if not quiet:
            print(f"{progress_label(it, max_iter, 'sim')} wrote {n} rows -> {path.name}", flush=True)
        return meta

    @acl.training_task(as_executable=False)
    async def training(sim_result, **kwargs):
        it = int(kwargs.get("iteration", sim_result["iteration"]))
        wf = str(kwargs.get("workflow", "rf"))
        if not quiet:
            print(
                f"{progress_label(it, max_iter, 'train')} SURGE spec={wf}  rows={sim_result['n_rows']}",
                flush=True,
            )
        out = run_one_surge(wf, it, verbose=not quiet)
        return {"simulation": sim_result, "surge": out}

    @acl.active_learn_task(as_executable=False)
    async def active_learn(sim_result, train_bundle, **kwargs):
        it = int(kwargs.get("iteration", train_bundle["simulation"]["iteration"]))
        wf = train_bundle["surge"]["workflow"]
        if not quiet:
            vrmse = float(train_bundle["surge"]["val_rmse"])
            vr2 = float(train_bundle["surge"].get("val_r2", float("nan")))
            print(
                f"{progress_label(it, max_iter, 'active_learn')} last spec={wf}  "
                f"val_rmse={vrmse:.5f}  val_r2={vr2:.4f}  (stub)",
                flush=True,
            )
        return {"iteration": it, "train": train_bundle}

    @acl.as_stop_criterion(
        metric_name=MEAN_SQUARED_ERROR_MSE,
        threshold=0.0,
        as_executable=False,
    )
    async def never_stop_until_max(*args, **kwargs):
        wdir = workspace_dir()
        mp = wdir / "last_surge_metrics.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        mse = float(meta["val_mse"])
        crit_it = int(meta.get("iteration", 0))
        if not quiet:
            print(
                f"{progress_label(crit_it, max_iter, 'criterion')} val_mse={mse:.6e}  (threshold 0)",
                flush=True,
            )
        return mse

    if not quiet:
        _banner("Starting learner (sim -> train -> active -> criterion)")
    report_rows = []
    async for state in acl.start(max_iter=max_iter, initial_config=initial):
        report_rows.append(snapshot_iteration(workspace_dir(), state.iteration))
        if not quiet:
            print(
                f"\n{progress_label(state.iteration, max_iter, 'ORCHESTRATION')} "
                f"metric={state.metric_value!r}  should_stop={state.should_stop}",
                flush=True,
            )

    await acl.shutdown()
    print_run_report(
        report_rows,
        title=f"Example 1 summary ({dataset})",
        elapsed_sec=timer.seconds(),
    )
    if not quiet:
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
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
