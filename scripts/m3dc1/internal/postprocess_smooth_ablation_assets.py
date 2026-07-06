#!/usr/bin/env python3
"""Generate §9-style comparison figures for fieldloss smooth ablations (Run D / D')."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "docs" / "m3dc1" / "assets"

RUNS = {
    "smooth05": {
        "run": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "bench": "with_fieldloss_smooth05",
        "short": "fieldloss_smooth05",
    },
    "smooth0": {
        "run": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "bench": "with_fieldloss_smooth0",
        "short": "fieldloss_smooth0",
    },
}
BASELINE = "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss"
PEAK4 = "spectrum_fno48_floor6_smooth1_qc_peak4"
SPARC_FAMS = ["sparc_1427", "sparc_1430", "sparc_1530", "sparc_1500", "sparc_1524"]


def _rel_col(model: str) -> str:
    return f"relL2_{model}"


def _pairwise_wins_wide(rows: list[dict], model_a: str, model_b: str) -> tuple[int, int]:
    col_a, col_b = _rel_col(model_a), _rel_col(model_b)
    wins_a = wins_b = 0
    for row in rows:
        if col_a not in row or col_b not in row:
            continue
        va, vb = float(row[col_a]), float(row[col_b])
        if va < vb:
            wins_a += 1
        elif vb < va:
            wins_b += 1
    return wins_a, wins_b


def _plot_pairwise(out: Path, title: str, wins_new: int, wins_base: int, new_label: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = [new_label, "fieldloss baseline"]
    vals = [wins_new, wins_base]
    colors = ["#2ca02c", "#7f7f7f"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_ylabel("cases won (lower relL2)")
    ax.set_title(title, fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, str(v), ha="center", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_hist_overlay(out: Path, rows: list[dict], models: list[str], title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c44e52", "#4c72b0", "#2ca02c", "#9467bd"]
    for model, color in zip(models, colors):
        col = _rel_col(model)
        vals = [float(r[col]) for r in rows if col in r]
        short = model.replace("spectrum_fno48_floor6_smooth1_qc_peak4_", "")
        ax.hist(vals, bins=40, alpha=0.45, label=short, color=color, edgecolor="none")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="relL2=1")
    ax.set_xlabel("field relL2 (oracle phase)")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_sparc_frac(out: Path, bench_dir: Path, models: list[str], title: str):
    rows = list(csv.DictReader((bench_dir / "per_family.csv").open()))
    x = np.arange(len(SPARC_FAMS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, model in enumerate(models):
        fracs = []
        for fam in SPARC_FAMS:
            match = [r for r in rows if r["family"] == fam and r["model"] == model]
            fracs.append(float(match[0]["frac_relL2_gt_1"]) if match else np.nan)
        short = model.replace("spectrum_fno48_floor6_smooth1_qc_peak4_", "")
        ax.bar(x + j * width, fracs, width, label=short)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(SPARC_FAMS, rotation=20, ha="right")
    ax.set_ylabel("frac(relL2 > 1)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


def process_variant(key: str, cfg: dict) -> None:
    run_dir = REPO / "runs" / cfg["run"]
    bench_dir = REPO / "field_bench" / cfg["bench"]
    short = cfg["short"]
    model = cfg["run"]

    loss_src = run_dir / "loss_fno2d.png"
    if loss_src.exists():
        dst = ASSETS / f"loss_{short}_fno2d.png"
        shutil.copy2(loss_src, dst)
        print(f"Copied {dst}")

    for src_name, dst_name in [
        (f"relL2_hist_{model}.png", f"field_bench_relL2_hist_{short}.png"),
        (f"relL2_hist_{BASELINE}.png", f"field_bench_relL2_hist_{short}_vs_baseline.png"),
    ]:
        src = bench_dir / src_name
        if src.exists():
            dst = ASSETS / dst_name
            shutil.copy2(src, dst)
            print(f"Copied {dst}")

    with (bench_dir / "per_case.csv").open() as f:
        cases = list(csv.DictReader(f))

    wins_new, wins_base = _pairwise_wins_wide(cases, model, BASELINE)
    _plot_pairwise(
        ASSETS / f"field_bench_pairwise_wins_{short}.png",
        f"Pairwise wins vs fieldloss baseline (n=1994 test)\n{short}: {wins_new}  ·  baseline: {wins_base}",
        wins_new,
        wins_base,
        short,
    )

    _plot_hist_overlay(
        ASSETS / f"field_bench_relL2_hist_peak4_vs_{short}.png",
        cases,
        [PEAK4, BASELINE, model],
        f"Field relL2 distribution — peak4 vs fieldloss vs {short}",
    )

    _plot_sparc_frac(
        ASSETS / f"field_bench_frac_gt1_sparc_{short}.png",
        bench_dir,
        [PEAK4, BASELINE, model],
        f"frac(relL2>1) — selected sparc families ({short})",
    )

    lb = json.loads((bench_dir / "leaderboard.json").read_text())
    summary = ASSETS / f"field_bench_leaderboard_{short}.json"
    summary.write_text(json.dumps(lb, indent=2))
    print(f"Wrote {summary}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for key, cfg in RUNS.items():
        print(f"\n=== {key} ===")
        process_variant(key, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
