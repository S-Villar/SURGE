# -*- coding: utf-8 -*-
"""
Example 2 - Same ROSE loop as Example 1; each stage is a **subprocess**.

Set PYTHONPATH to the SURGE repo root (see ``demo_common.SURGE_ROOT``).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path

from dataset_utils import read_iteration_state
from demo_common import SURGE_ROOT, add_demo_cli, add_reporting_cli, shell_task_kwargs
from orch_report import (
    RunTimer,
    print_run_header,
    print_run_report,
    print_iteration_progress,
    print_phase_progress,
    snapshot_iteration,
)

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
    quiet: bool,
) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from radical.asyncflow import WorkflowEngine
    from rhapsody.backends import ConcurrentExecutionBackend

    from rose.al.active_learner import SequentialActiveLearner
    from rose.learner import LearnerConfig
    from rose.metrics import MEAN_SQUARED_ERROR_MSE

    ws = _EX / "workspace"
    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_02"
    ws = ws / "example_02"
    timer = RunTimer()
    print_run_header(
        example="2 - SequentialActiveLearner + subprocess (ProcessPoolExecutor)",
        max_iter=max_iter,
        workers=workers,
        quiet=quiet,
        extra=(
            f"Dataset={dataset}. Workflow family: {workflow_family}. "
            f"Pool policy: {'growing subset' if growing_pool else 'full dataset'}. "
            "Each phase spawns sim_surge_step, surge_train, active_surge_step, check_surge_metrics."
        ),
    )
    if not quiet:
        _banner("ROSE + SURGE - Example 2: subprocess orchestration")

    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor(max_workers=workers))
    asyncflow = await WorkflowEngine.create(engine)
    acl = SequentialActiveLearner(asyncflow)

    raw_kw = shell_task_kwargs(max_iter, dataset, workflow_family)
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
        gp = " --growing-pool" if growing_pool else ""
        cmd = f"{py} {ex / 'sim_surge_step.py'} --iteration {it} --dataset {dataset}{gp}{vb}"
        if not quiet:
            print_phase_progress(
                rose_iter=iit,
                max_iter=max_iter,
                phase_index=1,
                phase_total=4,
                phase_name="spawn dataset preparation process",
                detail=cmd,
            )
        return cmd

    @acl.training_task()
    async def training(*args, **kwargs):
        it = kwargs.get("--iteration", "0")
        iit = int(it)
        wf = kwargs.get("--workflow", "rf")
        cmd = f"{py} {ex / 'surge_train.py'} --workflow {wf} --iteration {it}{vb}"
        if not quiet:
            print_phase_progress(
                rose_iter=iit,
                max_iter=max_iter,
                phase_index=2,
                phase_total=4,
                phase_name="spawn SURGE training process",
                detail=f"workflow={wf}; {cmd}",
            )
        return cmd

    @acl.active_learn_task()
    async def active_learn(*args, **kwargs):
        it = kwargs.get("--iteration", "0")
        iit = int(it)
        cmd = f"{py} {ex / 'active_surge_step.py'} --iteration {it}{vb}"
        if not quiet:
            print_phase_progress(
                rose_iter=iit,
                max_iter=max_iter,
                phase_index=3,
                phase_total=4,
                phase_name="spawn active-learning hook",
                detail=cmd,
            )
        return cmd

    @acl.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.0)
    async def check_mse(*args, **kwargs):
        crit_it = int(kwargs.get("--iteration", "0"))
        cmd = f"{py} {ex / 'check_surge_metrics.py'} --iteration {crit_it}{vb}"
        if not quiet:
            print_phase_progress(
                rose_iter=crit_it,
                max_iter=max_iter,
                phase_index=4,
                phase_total=4,
                phase_name="spawn criterion check",
                detail=cmd,
            )
        return cmd

    if not quiet:
        _banner("Learner starting")
    report_rows: list[dict] = []
    async for state in acl.start(max_iter=max_iter, initial_config=initial):
        report_rows.append(snapshot_iteration(ws, state.iteration))
        print_iteration_progress(
            report_rows,
            max_iter=max_iter,
            score_metric="val_r2",
            higher_is_better=True,
            mode_label="campaign",
        )
        if not quiet:
            try:
                surge_meta = read_iteration_state(state.iteration, "surge_metrics")
                print(
                    "  subprocess     "
                    f"split train/val/test={surge_meta.get('splits', {}).get('train', '?')}/"
                    f"{surge_meta.get('splits', {}).get('val', '?')}/"
                    f"{surge_meta.get('splits', {}).get('test', '?')}",
                    flush=True,
                )
            except FileNotFoundError:
                pass
            print(
                f"  orchestration  metric={state.metric_value!r}  should_stop={state.should_stop}",
                flush=True,
            )
        if state.iteration >= max_iter - 1:
            acl.stop()
            break

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
    os.environ["SURGE_ROSE_WORKSPACE_NAMESPACE"] = "example_02"

    asyncio.run(
        _demo(
            max_iter=args.max_iter,
            workers=args.workers,
            dataset=args.dataset,
            workflow_family=args.workflow_family,
            growing_pool=args.growing_pool,
            quiet=args.quiet,
        )
    )


if __name__ == "__main__":
    main()
