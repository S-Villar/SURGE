# -*- coding: utf-8 -*-
"""Shared CLI and schedule builders for ROSE + SURGE orchestration examples."""
from __future__ import annotations

import argparse
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SURGE_ROOT = EXAMPLE_DIR.parent.parent
DEFAULT_MAX_ITER = 3
DEFAULT_POOL_WORKERS = 4


def workflow_for_iteration(iteration: int, dataset: str) -> str:
    if dataset == "m3dc1":
        return "m3dc1_rf" if iteration % 2 == 0 else "m3dc1_mlp"
    return "rf" if iteration % 2 == 0 else "mlp"


def inprocess_task_kwargs(max_iter: int, dataset: str) -> dict[int, dict]:
    return {i: {"iteration": i, "workflow": workflow_for_iteration(i, dataset)} for i in range(max_iter)}


def shell_task_kwargs(max_iter: int, dataset: str) -> dict[int, dict]:
    return {
        i: {"--iteration": str(i), "--workflow": workflow_for_iteration(i, dataset)}
        for i in range(max_iter)
    }


def mlp_search_task_kwargs(max_iter: int, base_seed: int) -> dict[int, dict]:
    return {i: {"iteration": i, "base_seed": base_seed} for i in range(max_iter)}


def add_dataset_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "m3dc1"),
        default="synthetic",
        help="Training table: built-in synthetic (default) or SPARC M3DC1 PKL.",
    )


def add_reporting_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less per-step printing; summary table when supported.",
    )


def add_demo_cli(parser: argparse.ArgumentParser) -> None:
    add_dataset_cli(parser)
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
