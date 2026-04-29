#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
DEMOS = ROOT / "demos" / "output"
VIZ = ROOT / "viz"
DEMO2_UQ_THRESHOLD = 0.015


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_example_metrics(namespace: str) -> pd.DataFrame:
    files = sorted((WORKSPACE / namespace).glob("iter_*/surge_metrics.json"))
    rows = [_read_json(path) for path in files]
    return pd.DataFrame(rows).sort_values("iteration") if rows else pd.DataFrame()


def _load_example05() -> pd.DataFrame:
    base = WORKSPACE / "example_05" / "uq_ensemble"
    rows = []
    for metrics_path in sorted(base.glob("iter_*/surge_metrics.json")):
        iteration = int(metrics_path.parent.name.split("_")[-1])
        metric = _read_json(metrics_path)
        uq = _read_json(metrics_path.parent / "uncertainty.json")
        rows.append(
            {
                "iteration": iteration,
                "workflow": metric["workflow"],
                "val_r2": metric["val_r2"],
                "val_rmse": metric["val_rmse"],
                "train": metric.get("splits", {}).get("train"),
                "val": metric.get("splits", {}).get("val"),
                "test": metric.get("splits", {}).get("test"),
                "mean_std": uq["value"],
                "max_std": metric.get("uq_val_std_max"),
                "ensemble_n": metric.get("ensemble_n"),
                "hidden_layer_sizes": metric.get("hidden_layer_sizes"),
            }
        )
    return pd.DataFrame(rows).sort_values("iteration") if rows else pd.DataFrame()


def _load_example04() -> pd.DataFrame:
    rows = []
    base = WORKSPACE / "example_04"
    for path in sorted(base.glob("*_*/iter_*/surge_metrics.json")):
        payload = _read_json(path)
        learner_label = path.parents[1].name
        family = learner_label.split("_", 1)[1] if "_" in learner_label else learner_label
        rows.append(
            {
                "learner": family,
                "iteration": payload["iteration"],
                "workflow": payload["workflow"],
                "val_r2": payload["val_r2"],
                "val_rmse": payload["val_rmse"],
                "train": payload.get("splits", {}).get("train"),
                "val": payload.get("splits", {}).get("val"),
                "test": payload.get("splits", {}).get("test"),
                "run_tag": payload.get("run_tag"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["iteration", "learner"]) if not df.empty else df


def _load_demo_summary(name: str) -> dict | None:
    path = DEMOS / name / "summary.json"
    return _read_json(path) if path.exists() else None


def _save(fig: plt.Figure, name: str) -> Path:
    VIZ.mkdir(parents=True, exist_ok=True)
    path = VIZ / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_example_lines(df: pd.DataFrame, title: str, name_prefix: str) -> list[Path]:
    if df.empty:
        return []
    figs: list[Path] = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["iteration"], df["val_r2"], marker="o", label="val_r2")
    ax.set_title(f"{title}: Validation R2")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("val_r2")
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, f"{name_prefix}_val_r2.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["iteration"], df["val_rmse"], marker="o", color="tab:red", label="val_rmse")
    ax.set_title(f"{title}: Validation RMSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("val_rmse")
    ax.grid(True, alpha=0.3)
    figs.append(_save(fig, f"{name_prefix}_val_rmse.png"))
    return figs


def _plot_example05(df: pd.DataFrame) -> list[Path]:
    if df.empty:
        return []
    out: list[Path] = []
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["iteration"], df["val_r2"], marker="o", label="val_r2", color="tab:blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("val_r2", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["iteration"], df["mean_std"], marker="s", label="mean_std", color="tab:orange")
    ax2.set_ylabel("mean_std", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    fig.suptitle("Example 5: Validation R2 and Ensemble Mean Std")
    out.append(_save(fig, "example_05_uq_monitor.png"))
    return out


def _plot_example04(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    df = df.sort_values("val_r2", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df["learner"], df["val_r2"], color="tab:blue")
    axes[0].set_title("Validation R2")
    axes[0].set_ylabel("val_r2")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(df["learner"], df["val_rmse"], color="tab:red")
    axes[1].set_title("Validation RMSE")
    axes[1].set_ylabel("val_rmse")
    axes[1].grid(True, axis="y", alpha=0.3)

    split = f"split {int(df.iloc[0]['train'])}/{int(df.iloc[0]['val'])}/{int(df.iloc[0]['test'])}"
    fig.suptitle(f"Example 4: Parallel Model Race ({split})")
    return _save(fig, "example_04_model_race.png")


def _plot_demo01(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    df = df.sort_values("val_r2", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].barh(df["learner"], df["val_r2"], color="tab:green")
    axes[0].invert_yaxis()
    axes[0].set_title("Validation R2")
    axes[0].set_xlabel("val_r2")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(df["learner"], df["val_rmse"], color="tab:purple")
    axes[1].invert_yaxis()
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("val_rmse")
    axes[1].grid(True, axis="x", alpha=0.3)

    best = df.iloc[0]
    split = f"{int(best['train'])}/{int(best['val'])}/{int(best['test'])}"
    fig.suptitle(
        f"Demo 1: Better Surrogate Selection | winner={best['learner']} | split={split}"
    )
    return _save(fig, "demo_01_surrogate_selection.png")


def _plot_demo02(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    row = df.sort_values("iteration").iloc[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = ["val_r2", "mean_std", "max_std"]
    values = [float(row["val_r2"]), float(row["mean_std"]), float(row["max_std"])]
    colors = ["tab:blue", "tab:orange", "tab:red"]
    axes[0].bar(labels, values, color=colors)
    axes[0].axhline(DEMO2_UQ_THRESHOLD, color="tab:orange", linestyle="--", linewidth=1.5, label="UQ target")
    axes[0].set_title("Validation quality and uncertainty")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].axis("off")
    text = "\n".join(
        [
            "Demo 2: Smarter Campaign Control",
            "",
            f"workflow         : {row['workflow']}",
            f"ensemble_n       : {int(row['ensemble_n']) if pd.notna(row['ensemble_n']) else '?'}",
            f"hidden_layers    : {row['hidden_layer_sizes']}",
            f"train/val/test   : {int(row['train'])}/{int(row['val'])}/{int(row['test'])}",
            f"val_r2           : {float(row['val_r2']):.6f}",
            f"val_rmse         : {float(row['val_rmse']):.6f}",
            f"mean_std         : {float(row['mean_std']):.6f}",
            f"max_std          : {float(row['max_std']):.6f}",
            f"uq target        : {DEMO2_UQ_THRESHOLD:.6f}",
            "",
            "Interpretation:",
            "ROSE reads SURGE ensemble UQ artifacts and",
            "compares validation uncertainty to a decision threshold.",
        ]
    )
    axes[1].text(0.0, 1.0, text, va="top", ha="left", family="monospace")
    fig.suptitle("Demo 2: UQ-guided campaign summary")
    return _save(fig, "demo_02_uq_control.png")


def _plot_demo03(summary: dict) -> list[Path]:
    results = summary.get("results", [])
    if not results:
        return []
    df = pd.DataFrame(results).copy()
    out: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in df.groupby("family"):
        ax.scatter(group["val_rmse"], group["val_r2"], s=70, label=family, alpha=0.8)
    top = df.sort_values("val_r2", ascending=False).head(min(5, len(df)))
    for _, row in top.iterrows():
        ax.annotate(f"{int(row['trial_id']):02d}", (row["val_rmse"], row["val_r2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Demo 3: Trial quality frontier")
    ax.set_xlabel("val_rmse")
    ax.set_ylabel("val_r2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    resource_note = (
        f"available_cpus={summary.get('available_cpus')}  "
        f"cpus_per_trial={summary.get('cpus_per_trial')}  "
        f"parallel_trials={summary.get('parallel_trials')}  "
        f"rows={summary.get('total_rows')}"
    )
    fig.text(0.02, 0.01, resource_note, fontsize=9, family="monospace")
    out.append(_save(fig, "demo_03_quality_frontier.png"))

    arch_df = df.copy()
    arch_df["depth"] = arch_df["hidden_layer_sizes"].map(
        lambda hs: len(hs) if isinstance(hs, list) else 0
    )
    arch_df["total_units"] = arch_df["hidden_layer_sizes"].map(
        lambda hs: sum(hs) if isinstance(hs, list) else 0
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    mlp_df = arch_df[arch_df["family"] == "mlp"].copy()
    if not mlp_df.empty:
        scatter = ax.scatter(
            mlp_df["depth"],
            mlp_df["total_units"],
            c=mlp_df["val_r2"],
            s=90,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.9,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("val_r2")
        top = mlp_df.sort_values("val_r2", ascending=False).head(min(8, len(mlp_df)))
        for _, row in top.iterrows():
            label = f"{int(row['trial_id']):02d}"
            ax.annotate(
                label,
                (row["depth"], row["total_units"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    rf_df = arch_df[arch_df["family"] == "rf"]
    if not rf_df.empty:
        ax.scatter(
            [-0.1] * len(rf_df),
            [0] * len(rf_df),
            marker="s",
            s=110,
            color="tab:blue",
            label="rf baseline",
        )
        for _, row in rf_df.iterrows():
            ax.annotate(
                f"rf-{int(row['trial_id']):02d}",
                (-0.1, 0),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_title("Demo 3: Architecture map of tried configurations")
    ax.set_xlabel("MLP depth (# hidden layers)")
    ax.set_ylabel("Total hidden units")
    ax.set_xticks(sorted(set([0] + arch_df["depth"].tolist())))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.text(
        0.02,
        0.01,
        "MLP trials are colored by validation R2. RF baselines are shown separately.",
        fontsize=9,
    )
    out.append(_save(fig, "demo_03_architecture_map.png"))

    df_top = df.sort_values("val_r2", ascending=False).head(min(12, len(df))).copy()
    df_top["trial"] = df_top.apply(
        lambda row: f"{int(row['trial_id']):02d}-{row['family']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df_top["trial"], df_top["val_r2"], color=["tab:green" if fam == "mlp" else "tab:blue" for fam in df_top["family"]])
    ax.set_title("Demo 3: Top trials by validation R2")
    ax.set_xlabel("trial-family")
    ax.set_ylabel("val_r2")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    out.append(_save(fig, "demo_03_top_trials.png"))
    return out


def _plot_demo04(summary: dict) -> Path | None:
    results = summary.get("results", [])
    if not results:
        return None
    df = pd.DataFrame(results).sort_values("val_r2", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    df["trial"] = df["trial_id"].map(lambda x: f"trial_{x:02d}")
    axes[0].bar(df["trial"], df["val_r2"], color="tab:purple")
    axes[0].set_title("Validation R2 by GPU-aware trial")
    axes[0].set_ylabel("val_r2")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].axis("off")
    row = df.iloc[0]
    effective = (((row.get("resources_used") or {}).get("effective") or {}).get("device"))
    banner = (((row.get("resources_used") or {}).get("banner") or {}))
    text = "\n".join(
        [
            "Demo 4: GPU-aware SURGE search",
            "",
            f"visible_gpus      : {summary.get('visible_gpus')}",
            f"planned_trials    : {summary.get('planned_trials')}",
            f"effective_device  : {effective}",
            f"requested_slot    : {row.get('requested_gpu_slot')}",
            f"train/val/test    : {int(row['splits']['train'])}/{int(row['splits']['val'])}/{int(row['splits']['test'])}",
            f"hidden_layers     : {row.get('hidden_layers')}",
            f"val_r2            : {float(row['val_r2']):.6f}",
            f"val_rmse          : {float(row['val_rmse']):.6f}",
            "",
            "SURGE resources_used banner:",
            f"backend           : {banner.get('backend')}",
            f"device            : {banner.get('device')}",
            f"n_train           : {banner.get('n_train')}",
            f"n_features        : {banner.get('n_features')}",
        ]
    )
    axes[1].text(0.0, 1.0, text, va="top", ha="left", family="monospace")
    fig.suptitle("Demo 4: Device-aware training summary")
    return _save(fig, "demo_04_gpu_trials.png")


def main() -> int:
    generated: list[Path] = []

    generated.extend(_plot_example_lines(_load_example_metrics("example_01"), "Example 1", "example_01"))
    generated.extend(_plot_example_lines(_load_example_metrics("example_02"), "Example 2", "example_02"))

    ex3 = _load_example_metrics("example_03")
    if not ex3.empty:
        ex3 = ex3.copy()
        ex3["best_so_far_r2"] = ex3["val_r2"].cummax()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ex3["iteration"], ex3["val_r2"], marker="o", label="candidate")
        ax.plot(ex3["iteration"], ex3["best_so_far_r2"], linestyle="--", label="best_so_far")
        ax.set_title("Example 3: Random MLP Search")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("val_r2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        generated.append(_save(fig, "example_03_hpo_progress.png"))

    ex4 = _load_example04()
    path = _plot_example04(ex4)
    if path:
        generated.append(path)
        demo1 = _plot_demo01(ex4)
        if demo1:
            generated.append(demo1)

    ex5 = _load_example05()
    generated.extend(_plot_example05(ex5))
    demo2 = _plot_demo02(ex5)
    if demo2:
        generated.append(demo2)

    demo3 = _load_demo_summary("demo_03")
    if demo3:
        generated.extend(_plot_demo03(demo3))

    demo4 = _load_demo_summary("demo_04")
    if demo4:
        path = _plot_demo04(demo4)
        if path:
            generated.append(path)

    report = VIZ / "README.md"
    lines = [
        "# Visualization Outputs",
        "",
        "## Examples",
        "",
        "- [example_01_val_r2.png](example_01_val_r2.png)",
        "- [example_01_val_rmse.png](example_01_val_rmse.png)",
        "- [example_02_val_r2.png](example_02_val_r2.png)",
        "- [example_02_val_rmse.png](example_02_val_rmse.png)",
        "- [example_03_hpo_progress.png](example_03_hpo_progress.png)",
        "- [example_04_model_race.png](example_04_model_race.png)",
        "- [example_05_uq_monitor.png](example_05_uq_monitor.png)",
        "",
        "## Demos",
        "",
        "- [demo_01_surrogate_selection.png](demo_01_surrogate_selection.png)",
        "- [demo_02_uq_control.png](demo_02_uq_control.png)",
        "- [demo_03_quality_frontier.png](demo_03_quality_frontier.png)",
        "- [demo_03_architecture_map.png](demo_03_architecture_map.png)",
        "- [demo_03_top_trials.png](demo_03_top_trials.png)",
        "- [demo_04_gpu_trials.png](demo_04_gpu_trials.png)",
        "",
        f"Generated files: {len(generated)}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(generated)} plots under {VIZ}")
    print(f"Index: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
