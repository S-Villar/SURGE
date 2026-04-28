# -*- coding: utf-8 -*-
"""End-of-run reporting for ROSE + SURGE orchestration demos."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


def progress_label(rose_iter: int, n_rose_iters: int, phase: str) -> str:
    return f"[{rose_iter + 1}/{n_rose_iters} {phase}]"


def snapshot_iteration(workspace: Path, rose_iter: int) -> dict:
    row: dict = {"rose_iter": rose_iter}
    sp = workspace / "last_simulation.json"
    mp = workspace / "last_surge_metrics.json"
    if sp.exists():
        sim = json.loads(sp.read_text(encoding="utf-8"))
        row["n_rows"] = sim.get("n_rows")
        row["dataset"] = sim.get("dataset")
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        row["workflow"] = m.get("workflow")
        row["val_rmse"] = m.get("val_rmse")
        row["val_r2"] = m.get("val_r2")
        row["val_mse"] = m.get("val_mse")
        row["run_tag"] = m.get("run_tag")
        if "hidden_layer_sizes" in m:
            row["hidden_layer_sizes"] = m.get("hidden_layer_sizes")
    return row


def print_run_header(
    *,
    example: str,
    max_iter: int,
    workers: int,
    quiet: bool,
    extra: Optional[str] = None,
) -> None:
    lines = [
        f"Example: {example}",
        f"ROSE iterations (max_iter): {max_iter}  |  pool workers: {workers}",
        f"Demo log mode: {'quiet' if quiet else 'verbose'}",
        "Note: ROSE and SURGE may still print their own progress.",
    ]
    if extra:
        lines.append(extra)
    print("\n".join(lines), flush=True)
    print("-" * 72, flush=True)


def print_run_report(
    rows: list,
    *,
    title: str,
    dataset_line: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
) -> None:
    print("", flush=True)
    print("=" * 72, flush=True)
    print(title, flush=True)
    if dataset_line:
        print(dataset_line, flush=True)
    if elapsed_sec is not None:
        print(f"Wall time: {elapsed_sec:.1f}s", flush=True)
    print("=" * 72, flush=True)
    if not rows:
        print("(no iteration rows collected)", flush=True)
        return
    hdr = (
        f"{'#':>3}  {'rows':>6}  {'workflow':<12}  "
        f"{'val_rmse':>10}  {'val_r2':>8}  {'extra':<20}"
    )
    print(hdr, flush=True)
    print("-" * 72, flush=True)
    for row in rows:
        i = row.get("rose_iter", "?")
        n = row.get("n_rows", "?")
        wf = (row.get("workflow") or "?")[:12]
        rmse = row.get("val_rmse")
        r2 = row.get("val_r2")
        rmse_s = f"{float(rmse):.6f}" if rmse is not None else "?"
        r2_s = f"{float(r2):.4f}" if r2 is not None else "?"
        extra = row.get("run_tag") or ""
        if row.get("hidden_layer_sizes") is not None:
            extra = str(row["hidden_layer_sizes"])
        if len(extra) > 20:
            extra = extra[:17] + "..."
        print(f"{i!s:>3}  {n!s:>6}  {wf:<12}  {rmse_s:>10}  {r2_s:>8}  {extra:<20}", flush=True)
    print("", flush=True)


class RunTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self._t0
