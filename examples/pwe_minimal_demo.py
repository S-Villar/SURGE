#!/usr/bin/env python
"""
High-level SURGE workflow demo using the HHFW-NSTX PwE_ dataset.

The script highlights the new SurrogateEngine orchestration helpers:
  1. `prepare_workflow` loads the dataset, creates splits, and standardizes data.
  2. `workflow` accepts model specifications, runs CV/training/optimization,
     and saves artifacts (results, predictions, fitted estimators).
  3. A JSON summary with final metrics is produced for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surge import SurrogateEngine, get_data_path


def run_demo(output_root: Path, n_trials_rf: int = 50, n_trials_mlp: int = 50) -> None:
    dataset_path = get_data_path('HHFW-NSTX') / 'PwE_.pkl'
    print(f"📁 Loading dataset from {dataset_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"📂 Outputs will be saved under: {output_root}")

    engine = SurrogateEngine(dir_path=str(output_root))
    engine.prepare_workflow(
        dataset_path,
        test=0.2,
        val=0.2,
        standardize=True,
    )

    summary = engine.workflow(
        [
            {
                'name': 'RandomForest',
                'slug': 'random_forest',
                'key': 'random_forest',
                'params': {'n_estimators': 200, 'random_state': 42},
                'cross_validate': True,
                'cv_folds': 5,
                'optimize': {'n_trials': n_trials_rf, 'sampler': 'botorch', 'objective': 'r2'},
            },
            {
                'name': 'MLPRegressor',
                'slug': 'mlp',
                'key': 'sklearn.mlp',
                'params': {
                    'hidden_layer_sizes': (150, 75),
                    'random_state': 42,
                    'max_iter': 500,
                },
                'cross_validate': True,
                'cv_folds': 5,
                'optimize': {'n_trials': n_trials_mlp, 'sampler': 'botorch', 'objective': 'r2'},
            },
        ],
        cv_folds=5,
        predictions_scope='all',
        predictions_format='pickle',
        output_dir=output_root,
    )

    summary_path = output_root / 'workflow_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print("\n🎯 Final Performance")
    print("--------------------")
    rf_r2 = summary['RandomForest']['performance'].get('R2')
    mlp_r2 = summary['MLPRegressor']['performance'].get('R2')
    print(f"Random Forest R² (test): {rf_r2:.4f}" if rf_r2 is not None else "Random Forest R² unavailable")
    print(f"MLP R² (test):          {mlp_r2:.4f}" if mlp_r2 is not None else "MLP R² unavailable")
    print(f"\n📝 Summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SURGE PwE_ minimal workflow demo.")
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path.cwd() / 'runs' / 'pwe_demo',
        help="Directory where artifacts (results, models, predictions) will be saved.",
    )
    parser.add_argument('--rf-trials', type=int, default=50, help="Optuna trials for Random Forest.")
    parser.add_argument('--mlp-trials', type=int, default=50, help="Optuna trials for MLP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args.output_root, n_trials_rf=args.rf_trials, n_trials_mlp=args.mlp_trials)


if __name__ == '__main__':
    main()



