#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "label": row["label"],
                    "available_cpus": int(float(row["available_cpus"] or 0)),
                    "wall_time_seconds": float(row["wall_time_seconds"] or 0.0),
                    "trials_per_hour": float(row["trials_per_hour"] or 0.0),
                    "best_val_r2": float(row["best_val_r2"] or 0.0),
                }
            )
    rows.sort(key=lambda row: row["available_cpus"])
    return rows


def _plot_metric(rows, *, x_key, y_key, title, ylabel, output_path):
    x = [row[x_key] for row in rows]
    y = [row[y_key] for row in rows]
    labels = [row["label"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, y, marker="o", linewidth=2)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Allocated CPUs")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot Demo 3 scaling curves from scaling_report.csv.")
    ap.add_argument("--csv-path", default="demos/output/demo_03_scaling/scaling_report.csv")
    ap.add_argument("--output-dir", default="demos/output/demo_03_scaling/plots")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit("No scaling rows found in CSV.")

    _plot_metric(
        rows,
        x_key="available_cpus",
        y_key="wall_time_seconds",
        title="Demo 3 Scaling: Wall Time vs CPUs",
        ylabel="Wall Time (s)",
        output_path=output_dir / "wall_time_vs_cpus.png",
    )
    _plot_metric(
        rows,
        x_key="available_cpus",
        y_key="trials_per_hour",
        title="Demo 3 Scaling: Trials per Hour vs CPUs",
        ylabel="Trials per Hour",
        output_path=output_dir / "trials_per_hour_vs_cpus.png",
    )
    _plot_metric(
        rows,
        x_key="available_cpus",
        y_key="best_val_r2",
        title="Demo 3 Scaling: Best Validation R2 vs CPUs",
        ylabel="Best Validation R2",
        output_path=output_dir / "best_val_r2_vs_cpus.png",
    )

    print("Wrote", output_dir / "wall_time_vs_cpus.png")
    print("Wrote", output_dir / "trials_per_hour_vs_cpus.png")
    print("Wrote", output_dir / "best_val_r2_vs_cpus.png")


if __name__ == "__main__":
    raise SystemExit(main())
