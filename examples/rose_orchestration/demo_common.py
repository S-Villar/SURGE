# -*- coding: utf-8 -*-
"""Shared CLI and schedule builders for ROSE + SURGE orchestration examples."""
from __future__ import annotations

import argparse
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SURGE_ROOT = EXAMPLE_DIR.parent.parent
DEFAULT_MAX_ITER = 3
DEFAULT_POOL_WORKERS = 4


def canonical_workflow(dataset: str, family: str = "rf") -> str:
    if dataset == "m3dc1":
        return f"m3dc1_{family}"
    return family


def inprocess_task_kwargs(max_iter: int, dataset: str, workflow_family: str = "rf") -> dict[int, dict]:
    workflow = canonical_workflow(dataset, workflow_family)
    return {i: {"iteration": i, "workflow": workflow} for i in range(max_iter)}


def shell_task_kwargs(max_iter: int, dataset: str, workflow_family: str = "rf") -> dict[int, dict]:
    workflow = canonical_workflow(dataset, workflow_family)
    return {
        i: {"--iteration": str(i), "--workflow": workflow}
        for i in range(max_iter)
    }


def mlp_search_task_kwargs(max_iter: int, base_seed: int) -> dict[int, dict]:
    return {i: {"iteration": i, "base_seed": base_seed} for i in range(max_iter)}


def add_dataset_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "m3dc1"),
        default="m3dc1",
        help="Training table: SPARC M3DC1 PKL (default) or built-in synthetic.",
    )
    parser.add_argument(
        "--growing-pool",
        action="store_true",
        help="Use the staged campaign pool-growth policy instead of the full available dataset.",
    )


def add_reporting_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less per-step printing; summary table when supported.",
    )
    parser.add_argument(
        "--no-live-progress",
        action="store_true",
        help="Disable refreshing progress bars and print normal line-oriented output.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write detailed ROSE/SURGE output to this log file.",
    )


def add_demo_cli(parser: argparse.ArgumentParser) -> None:
    add_dataset_cli(parser)
    parser.add_argument(
        "--workflow-family",
        choices=("rf", "mlp"),
        default="rf",
        help="SURGE workflow family for campaign examples (default rf).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        metavar="N",
        help=f"ROSE outer iterations (default {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_POOL_WORKERS,
        metavar="K",
        help=f"Backend pool workers (default {DEFAULT_POOL_WORKERS}).",
    )
