# -*- coding: utf-8 -*-
"""
Example 2 - Same ROSE loop as Example 1; each stage is a **subprocess**.

Set PYTHONPATH to the SURGE repo root (see ``demo_common.SURGE_ROOT``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

from demo_common import SURGE_ROOT, add_demo_cli, add_reporting_cli, shell_task_kwargs
from orch_report import (
    RunTimer,
    print_run_header,
    print_run_report,
    progress_label,
    snapshot_iteration,
)

_EX = Path(__file__).resolve().parent
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))


def _banner(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72 + "\n", flush=True)


async def _demo(*, max_iter: int, workers: int, dataset: str, quiet: bool) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend

    from rose.al.active_learner import SequentialActiveLearner
    from rose.learner import LearnerConfig
    from rose.metrics import MEAN_SQUARED_ERROR_MSE

    ws = _EX / "workspace"
    timer = RunTimer()
    print_run_header(
        example="2 - SequentialActiveLearner + subprocess (ProcessPoolExecutor)",
        max_iter=max_iter,
        workers=workers,
        quiet=quiet,
        extra=f"Dataset={dataset}. Each phase spawns sim_surge_step, surge_train, active_surge_step, check_surge_metrics.",
    )
    if not quiet:
        _banner("ROSE + SURGE - Example 2: subprocess orchestration")

    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    raw_kw = shell_task_kwargs(max_iter, dataset)
    from rose.learner import TaskConfig
    sched = {i: TaskConfig(kwargs=kw) for i, kw in raw_kw.items()}
    initial = LearnerConfig(simulation=sched, training=sched, active_learn=sched)

    py = sys.executable
    ex = _EX
    vb = " --verbose" if not quiet else ""

    @acl.simulation_task()
    async def simulation(*args, **kwargs):
        it = kwargs.get("--iteration", "0")
        iit = int(it)
        cmd = f"{py} {ex / 'sim_surge_step.py'} --iteration {it} --dataset {dataset}{vb}"
        if not quiet:
            print(f"{progress_label(iit, max_iter, 'sim')} spawn -> {cmd}", flush=True)
        return cmd

    @acl.training_task()
    async def training(*args, **kwargs):
        it = kwargs.get("--iteration", "0")
        iit = int(it)
        wf = kwargs.get("--workflow", "rf")
        cmd = f"{py} {ex / 'surge_train.py'} --workflow {wf} --iteration {it}{vb}"
        if not quiet:
            print(f"{progress_label(iit, max_iter, 'train')} spec={wf} -> {cmd}", flush=True)
        return cmd

    @acl.active_learn_task()
    async def active_learn(*args, **kwargs):
        it = kwargs.get("--iteration", "0")
        iit = int(it)
        cmd = f"{py} {ex / 'active_surge_step.py'} --iteration {it}{vb}"
        if not quiet:
            print(f"{progress_label(iit, max_iter, 'active_learn')} stub -> {cmd}", flush=True)
        return cmd

    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.0)
    async def check_mse(*args, **kwargs):
        cmd = f"{py} {ex / 'check_surge_metrics.py'}{vb}"
        if not quiet:
            crit_it = 0
            mp = ws / "last_surge_metrics.json"
            if mp.exists():
                crit_it = int(json.loads(mp.read_text(encoding="utf-8")).get("iteration", 0))
            print(f"{progress_label(crit_it, max_iter, 'criterion')} -> {cmd}", flush=True)
        return cmd

    if not quiet:
        _banner("Learner starting")
    report_rows: list[dict] = []
    async for state in acl.start(max_iter=max_iter, initial_config=initial):
        report_rows.append(snapshot_iteration(ws, state.iteration))
        if not quiet:
            print(
                f"\n{progress_label(state.iteration, max_iter, 'ORCHESTRATION')} "
                f"metric={state.metric_value!r}  should_stop={state.should_stop}",
                flush=True,
            )

    await acl.shutdown()
    print_run_report(
        report_rows,
        title=f"Example 2 summary ({dataset}, subprocess)",
        elapsed_sec=timer.seconds(),
    )
    if not quiet:
        _banner("Finished subprocess demo")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*GPflow.*", category=UserWarning)
    parser = argparse.ArgumentParser(description="ROSE subprocess orchestration.")
    add_demo_cli(parser)
    add_reporting_cli(parser)
    args = parser.parse_args()

    root = str(SURGE_ROOT)
    prev = os.environ.get("PYTHONPATH", "").strip()
    if not prev:
        os.environ["PYTHONPATH"] = root
    elif root not in prev.split(os.pathsep):
        os.environ["PYTHONPATH"] = root + os.pathsep + prev

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
