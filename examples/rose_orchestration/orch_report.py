# -*- coding: utf-8 -*-
"""End-of-run reporting for ROSE + SURGE orchestration demos."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional


def progress_label(rose_iter: int, n_rose_iters: int, phase: str) -> str:
    return f"Iteration {rose_iter + 1}/{n_rose_iters} | {phase}"


def snapshot_iteration(workspace: Path, rose_iter: int) -> dict:
    row: dict = {"rose_iter": rose_iter}
    sp = workspace / "last_simulation.json"
    mp = workspace / "last_surge_metrics.json"
    if sp.exists():
        sim = json.loads(sp.read_text(encoding="utf-8"))
        row["n_rows"] = sim.get("n_rows")
        row["n_rows_total"] = sim.get("n_rows_total")
        row["row_policy"] = sim.get("row_policy")
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


def progress_bar(current: int, total: int | None, *, width: int = 24) -> str:
    if total is None or total <= 0:
        return "[" + "#" * width + "]"
    current = max(0, min(int(current), int(total)))
    filled = round(width * current / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {current}/{total}"


def target_bar(value: float, target: float, *, width: int = 24, higher_is_better: bool = True) -> str:
    if not math.isfinite(value) or not math.isfinite(target) or target == 0:
        return "[" + "." * width + "]"
    if higher_is_better:
        frac = max(0.0, min(value / target, 1.0))
    else:
        frac = max(0.0, min(target / value, 1.0)) if value != 0 else 1.0
    filled = round(width * frac)
    return "[" + "=" * filled + "." * (width - filled) + f"] {100.0 * frac:5.1f}%"


def metric_sparkline(values: list[float], *, width: int = 32, higher_is_better: bool = True) -> str:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo = min(vals)
    hi = max(vals)
    chars = " .:-=+*#%@"
    if hi == lo:
        return chars[-1] * len(vals)
    out = []
    for v in vals:
        x = (v - lo) / (hi - lo)
        if not higher_is_better:
            x = 1.0 - x
        out.append(chars[int(round(x * (len(chars) - 1)))])
    return "".join(out)


def best_so_far(rows: list[dict], *, metric: str, higher_is_better: bool) -> dict | None:
    candidates = []
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        fval = float(val)
        if math.isfinite(fval):
            candidates.append((fval, row))
    if not candidates:
        return None
    return (max if higher_is_better else min)(candidates, key=lambda item: item[0])[1]


def describe_row_count(row: dict) -> str:
    n = row.get("n_rows")
    total = row.get("n_rows_total")
    if n is None:
        return "rows=?"
    if total:
        pct = 100.0 * float(n) / float(total)
        return f"rows={n}/{total} ({pct:.1f}%)"
    return f"rows={n}"


def print_iteration_progress(
    rows: list[dict],
    *,
    max_iter: int,
    score_metric: str,
    higher_is_better: bool,
    mode_label: str = "campaign",
    threshold: float | None = None,
    threshold_operator: str | None = None,
) -> None:
    if not rows:
        return
    row = rows[-1]
    i = int(row.get("rose_iter", len(rows) - 1))
    metric_val = row.get(score_metric)
    metric_s = f"{float(metric_val):.6f}" if metric_val is not None else "?"
    best = best_so_far(rows, metric=score_metric, higher_is_better=higher_is_better)
    best_s = "?"
    best_i = "?"
    if best is not None:
        best_s = f"{float(best[score_metric]):.6f}"
        best_i = str(best.get("rose_iter", "?"))
    vals = [float(r[score_metric]) for r in rows if r.get(score_metric) is not None]
    spark = metric_sparkline(vals, higher_is_better=higher_is_better)
    extra = row.get("hidden_layer_sizes") or row.get("run_tag") or ""
    print(
        f"[{mode_label}] explored {progress_bar(i + 1, max_iter)}  "
        f"{describe_row_count(row)}  current {score_metric}={metric_s}  "
        f"best={best_s}@iter{best_i}",
        flush=True,
    )
    if threshold is not None and threshold_operator:
        print(
            f"[{mode_label}] target   {target_bar(float(metric_val), threshold, higher_is_better=higher_is_better)}  "
            f"{score_metric} {threshold_operator} {threshold}",
            flush=True,
        )
    if extra:
        print(f"[{mode_label}] latest   {extra}", flush=True)
    if spark:
        print(f"[{mode_label}] trend    {score_metric}: {spark}", flush=True)


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
    best_rmse = best_so_far(rows, metric="val_rmse", higher_is_better=False)
    best_r2 = best_so_far(rows, metric="val_r2", higher_is_better=True)
    if best_rmse or best_r2:
        print("-" * 72, flush=True)
    if best_rmse:
        print(
            f"Best val_rmse: {float(best_rmse['val_rmse']):.6f} "
            f"at iter {best_rmse.get('rose_iter')} ({describe_row_count(best_rmse)})",
            flush=True,
        )
    if best_r2:
        print(
            f"Best val_r2  : {float(best_r2['val_r2']):.6f} "
            f"at iter {best_r2.get('rose_iter')} ({describe_row_count(best_r2)})",
            flush=True,
        )
    r2_vals = [float(r["val_r2"]) for r in rows if r.get("val_r2") is not None]
    rmse_vals = [float(r["val_rmse"]) for r in rows if r.get("val_rmse") is not None]
    if r2_vals:
        print(f"R2 trend    : {metric_sparkline(r2_vals, higher_is_better=True)}", flush=True)
    if rmse_vals:
        print(f"RMSE trend  : {metric_sparkline(rmse_vals, higher_is_better=False)}", flush=True)
    print("", flush=True)


class RunTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self._t0
