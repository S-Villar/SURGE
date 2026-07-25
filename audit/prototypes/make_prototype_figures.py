"""Generate SURGE visual-system prototype figures from REAL run artifacts.

Nothing here is hand-encoded: every number is read from machine-readable
SURGE outputs (runs/<tag>/ artifacts and benchmark_reports/**/result.json).

Usage (from repo root):
    python audit/prototypes/make_prototype_figures.py \
        [--run runs/diabetes_rf] [--out audit/prototypes/output]

Produces, in light and dark modes, as deterministic PNG+SVG+PDF:
    parity_<mode>          regression parity plot with density + metrics
    leaderboard_<mode>     benchmark leaderboard: score ± std and runtime
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surge_style import surge_theme, save_figure, fmt_metric

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ parity figure

def parity_figure(run_dir: Path, mode: str):
    """Regression diagnostic: parity + residual histogram from run artifacts."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_key = next(iter(metrics))
    pred_path = run_dir / "predictions" / f"{model_key}_test.parquet"
    df = pd.read_parquet(pred_path)
    y_true = df[[c for c in df.columns if c.startswith("y_true")][0]].to_numpy()
    y_pred = df[[c for c in df.columns if c.startswith("y_pred")][0]].to_numpy()
    m = metrics[model_key]["test"]

    with surge_theme(mode) as p:
        fig, (ax, axr) = plt.subplots(
            1, 2, figsize=(6.6, 3.0), width_ratios=[1.0, 0.85])

        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
        # identity line first (recessive), marks on top
        ax.plot(lims, lims, color=p["axis"], lw=0.8, zorder=1)
        ax.scatter(y_true, y_pred, s=14, color=p["series"][0],
                   alpha=0.55, linewidths=0, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(f"Parity — {model_key} (test)")
        ax.text(0.03, 0.94,
                f"R² {fmt_metric(m['r2'])}   RMSE {fmt_metric(m['rmse'],'rmse')}"
                f"   MAE {fmt_metric(m['mae'],'rmse')}",
                transform=ax.transAxes, fontsize=8, color=p["ink2"],
                va="top")

        resid = y_pred - y_true
        axr.hist(resid, bins=21, color=p["series"][0], alpha=0.85)
        axr.axvline(0.0, color=p["axis"], lw=0.8)
        axr.set_xlabel("residual (pred − obs)")
        axr.set_ylabel("count")
        axr.set_title("Residuals")
        axr.text(0.03, 0.94,
                 f"mean {resid.mean():+.2f}   σ {resid.std():.2f}",
                 transform=axr.transAxes, fontsize=8, color=p["ink2"],
                 va="top")
        return fig


# -------------------------------------------------------- leaderboard figure

def collect_results(reports_dir: Path, benchmark_key: str):
    """Aggregate benchmark_reports/<key>/*/result.json -> per-model stats."""
    rows = defaultdict(lambda: {"scores": [], "runtimes": []})
    for rj in sorted((reports_dir / benchmark_key).glob("*/result.json")):
        d = json.loads(rj.read_text())
        model = d.get("model_key") or d.get("extra", {}).get("model_key")
        met = d.get("metrics", {})
        score = met.get("test_r2", met.get("test_accuracy"))
        if model is None or score is None:
            continue
        rows[model]["scores"].append(float(score))
        if met.get("runtime_s") is not None:
            rows[model]["runtimes"].append(float(met["runtime_s"]))
    out = []
    for model, v in rows.items():
        s = v["scores"]
        out.append({
            "model": model,
            "mean": statistics.fmean(s),
            "std": statistics.stdev(s) if len(s) > 1 else 0.0,
            "n": len(s),
            "runtime": statistics.fmean(v["runtimes"]) if v["runtimes"] else None,
        })
    out.sort(key=lambda r: r["mean"], reverse=True)
    return out


def leaderboard_figure(reports_dir: Path, benchmark_key: str, mode: str,
                       threshold: float | None = None, metric="test R²"):
    rows = collect_results(reports_dir, benchmark_key)[:10]
    if not rows:
        raise SystemExit(f"no results found for {benchmark_key}")

    with surge_theme(mode) as p:
        fig, (ax, axt) = plt.subplots(
            1, 2, figsize=(7.2, 0.42 * len(rows) + 1.3),
            width_ratios=[1.0, 0.38], sharey=True)
        ypos = np.arange(len(rows))[::-1]

        # single-hue bars: identity is the row label, color is not rank
        for y, r in zip(ypos, rows):
            ax.barh(y, r["mean"], height=0.62, color=p["series"][0],
                    alpha=0.9)
            if r["std"] > 0:
                ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none",
                            ecolor=p["ink2"], elinewidth=1.1, capsize=2.5)
            label = f'{fmt_metric(r["mean"])}'
            if r["std"] > 0:
                label += f' ± {r["std"]:.3f}'
            label += f'  (n={r["n"]})'
            ax.text(0.005, y, " " + r["model"], va="center", fontsize=8,
                    color=p["ink"], zorder=3)
            ax.text(min(r["mean"] + (r["std"] or 0) + 0.015, 1.0), y, label,
                    va="center", ha="left", fontsize=7, color=p["ink2"])
        if threshold is not None:
            ax.axvline(threshold, color=p["serious"], lw=1.0, ls=(0, (4, 3)),
                       zorder=1)
            ax.text(threshold, -0.62, f"threshold {threshold} ",
                    color=p["serious"], fontsize=7, va="top", ha="right")
        ax.set_yticks([])
        ax.set_xlabel(metric + "  (mean ± std over runs)")
        ax.set_title(f"Leaderboard — {benchmark_key}")
        ax.set_xlim(0, 1.02)

        # runtime companion axis (separate chart, same rows — never dual-axis)
        for y, r in zip(ypos, rows):
            if r["runtime"] is not None:
                axt.barh(y, r["runtime"], height=0.62,
                         color=p["series"][1], alpha=0.9)
                axt.text(r["runtime"], y, " " + fmt_metric(r["runtime"], "runtime"),
                         va="center", fontsize=7, color=p["ink2"])
        axt.set_title("runtime", fontsize=9)
        axt.set_xlabel("seconds")
        return fig


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "runs" / "diabetes_rf"))
    ap.add_argument("--benchmark", default="tabular.california_housing")
    ap.add_argument("--reports", default=str(REPO / "benchmark_reports"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "output"))
    args = ap.parse_args()

    out = Path(args.out)
    for mode in ("light", "dark"):
        fig = parity_figure(Path(args.run), mode)
        print("wrote", [str(x) for x in save_figure(fig, out / f"parity_{mode}")])
        plt.close(fig)

        fig = leaderboard_figure(Path(args.reports), args.benchmark, mode,
                                 threshold=0.75)
        print("wrote", [str(x) for x in save_figure(fig, out / f"leaderboard_{mode}")])
        plt.close(fig)


if __name__ == "__main__":
    main()
