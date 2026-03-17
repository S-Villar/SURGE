#!/usr/bin/env python3
"""
XGC A_parallel datastreamset generalization analysis.

Evaluates surrogate generalization across held-out dataset regions (datastreamsets).
Two modes:
  1. Use existing trained models: load from run_dir, predict on datastreamsets, report R²/RMSE.
  2. Train-on-datastreamset: train simple models (RF, MLP) with 10 HPO trials on datastreamset 0,
     evaluate on datastreamsets 1,2,... to assess generalization.

Usage:
  # Use existing models from a run
  python scripts/xgc_datastreamset_generalization.py --run-dir runs/xgc_aparallel_set1_v3 --datastreamset-eval

  # Train on datastreamset 0, evaluate on datastreamsets 1-4 (10 HPO trials)
  python scripts/xgc_datastreamset_generalization.py --data-dir /path/to/olcf_ai_hackathon_2025 \\
      --train-on-datastreamset --hpo-trials 10 --datastreamset-size 50000 --max-datastreamsets 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _main_use_existing(
    run_dir: Path,
    datastreamset_size: int,
    max_datastreamsets: int,
    output_dir: Path,
    eval_set_name: str | None = None,
) -> int:
    """Use existing models for datastreamset evaluation."""
    from surge.viz.run_viz import viz_datastreamset_evaluation

    out = output_dir if output_dir != Path("runs/xgc_datastreamset_analysis") else run_dir / "plots"
    result = viz_datastreamset_evaluation(
        run_dir=run_dir,
        output_dir=out,
        datastreamset_size=datastreamset_size,
        max_datastreamsets=max_datastreamsets,
        eval_set_name=eval_set_name,
    )
    print("Datastreamset evaluation (existing models):")
    print(json.dumps(result, indent=2))
    return 0


def _main_train_on_datastreamset(
    data_dir: Path,
    datastreamset_size: int,
    max_datastreamsets: int,
    hpo_trials: int,
    output_dir: Path,
) -> int:
    """Train simple models on datastreamset 0 with HPO, evaluate on other datastreamsets."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    from surge.dataset import SurrogateDataset

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("optuna required for train-on-datastreamset. pip install optuna", file=sys.stderr)
        return 1

    # Load datastreamset 0 as train
    hints = {"row_range": (0, datastreamset_size), "set_name": "set1"}
    dataset = SurrogateDataset.from_path(
        data_dir,
        format="xgc",
        sample=None,
        analyzer_kwargs={"hints": hints},
    )
    if dataset.df is None or len(dataset.df) == 0:
        print("Datastreamset 0 empty.", file=sys.stderr)
        return 1

    X_train = dataset.df[dataset.input_columns].values.astype(np.float64)
    y_train = dataset.df[dataset.output_columns].values.astype(np.float64)
    # A_parallel = output_1
    y_train_aparallel = y_train[:, 1]

    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)

    def objective(trial: "optuna.Trial") -> float:
        n_est = trial.suggest_int("n_estimators", 100, 300, step=50)
        depth = trial.suggest_int("max_depth", 10, 25)
        leaf = trial.suggest_int("min_samples_leaf", 2, 5)
        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_aparallel)
        pred = model.predict(X_train_scaled)
        return float(np.sqrt(mean_squared_error(y_train_aparallel, pred)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(n_startup_trials=3))
    study.optimize(objective, n_trials=hpo_trials, show_progress_bar=True)

    best_params = study.best_params
    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
    )
    model.fit(X_train_scaled, y_train_aparallel)

    # Evaluate on other datastreamsets
    results = {"datastreamset_size": datastreamset_size, "hpo_trials": hpo_trials, "datastreamsets": {}}
    r2_list = []
    rmse_list = []
    datastreamset_indices = []

    for datastreamset_idx in range(1, max_datastreamsets):
        start = datastreamset_idx * datastreamset_size
        end = start + datastreamset_size
        hints = {"row_range": (start, end), "set_name": "set1"}
        try:
            ds = SurrogateDataset.from_path(
                data_dir,
                format="xgc",
                sample=None,
                analyzer_kwargs={"hints": hints},
            )
        except Exception as e:
            print(f"Segment {datastreamset_idx} load failed: {e}", file=sys.stderr)
            continue
        if ds.df is None or len(ds.df) == 0:
            continue

        X = ds.df[ds.input_columns].values.astype(np.float64)
        y_true = ds.df[ds.output_columns].values[:, 1]
        X_scaled = input_scaler.transform(X)
        y_pred = model.predict(X_scaled)

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        datastreamset_key = f"datastreamset_{datastreamset_idx}_rows_{start}_{end}"
        results["datastreamsets"][datastreamset_key] = {"n_samples": len(ds.df), "r2": r2, "rmse": rmse}
        r2_list.append(r2)
        rmse_list.append(rmse)
        datastreamset_indices.append(datastreamset_idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "datastreamset_generalization_train_on_datastreamset.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out_json}")

    # Plot
    if datastreamset_indices and r2_list:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(datastreamset_indices, r2_list, "o-", color="steelblue", linewidth=2, markersize=10)
        ax1.set_ylabel("R² score")
        ax1.set_title("Generalization: model trained on datastreamset 0, evaluated on held-out datastreamsets")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        ax2.plot(datastreamset_indices, rmse_list, "o-", color="coral", linewidth=2, markersize=10)
        ax2.set_xlabel("Segment index")
        ax2.set_ylabel("RMSE")
        ax2.set_title("RMSE on held-out datastreamsets (A_parallel)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "datastreamset_generalization_r2_rmse.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot: {plot_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="XGC datastreamset generalization analysis")
    parser.add_argument("--run-dir", type=Path, help="Use existing models from this run")
    parser.add_argument("--data-dir", type=Path, help="XGC data dir (for train-on-datastreamset)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/xgc_datastreamset_analysis"))
    parser.add_argument("--datastreamset-size", type=int, default=50000)
    parser.add_argument("--max-datastreamsets", type=int, default=5)
    parser.add_argument("--datastreamset-eval", action="store_true", help="Evaluate existing models on datastreamsets")
    parser.add_argument(
        "--train-on-datastreamset",
        action="store_true",
        help="Train RF on datastreamset 0 with HPO, evaluate on datastreamsets 1+",
    )
    parser.add_argument("--hpo-trials", type=int, default=10)
    parser.add_argument(
        "--eval-set",
        type=str,
        default=None,
        help="Override set_name for eval (e.g. set2_beta0p5 for cross-set eval)",
    )
    args = parser.parse_args()

    if args.run_dir and args.datastreamset_eval:
        return _main_use_existing(
            args.run_dir,
            args.datastreamset_size,
            args.max_datastreamsets,
            args.output_dir,
            eval_set_name=args.eval_set,
        )
    if args.data_dir and args.train_on_datastreamset:
        return _main_train_on_datastreamset(
            args.data_dir, args.datastreamset_size, args.max_datastreamsets, args.hpo_trials, args.output_dir
        )

    parser.print_help()
    print("\nUse --run-dir + --datastreamset-eval OR --data-dir + --train-on-datastreamset", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
