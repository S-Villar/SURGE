# -*- coding: utf-8 -*-
"""
Train several sklearn MLPs with different ``random_state`` on one SURGE split, then
report validation **mean** and **std** of predictions across members (deep-ensemble UQ).

Defaults target the **synthetic** ROSE demo Parquet and metadata in ``rose_orchestration``.

Run::

    cd examples/mlp_ensemble_uq
    export PYTHONPATH=/path/to/SURGE:$PYTHONPATH
    python mlp_ensemble_uq_demo.py --n-members 5

**Built-in:** :class:`surge.model.sklearn.MLPModel` supports ``ensemble_n`` in YAML/params;
with ``request_uncertainty: true`` the workflow writes ``*_val_uq.json``. See
``examples/surge_uq_demo/run_uq_demo.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EX_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EX_DIR.parent.parent
_ROSE_ORCH = _EX_DIR.parent / "rose_orchestration"
_DEFAULT_PARQUET = _ROSE_ORCH / "workspace" / "synthetic_active_learning.parquet"
_DEFAULT_METADATA = _ROSE_ORCH / "synthetic_demo_metadata.yaml"


def _ensure_surge_path() -> None:
    r = str(_REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def _ensure_parquet(parquet: Path, build_iteration: int) -> None:
    if parquet.is_file():
        return
    sys.path.insert(0, str(_ROSE_ORCH))
    from dataset_utils import build_synthetic_parquet  # noqa: WPS433

    build_synthetic_parquet(build_iteration)
    if not parquet.is_file():
        raise FileNotFoundError(f"Could not build Parquet at {parquet}")


def _member_seeds(base: int, n: int) -> list[int]:
    rng = np.random.default_rng(base)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(n)]


def main() -> int:
    _ensure_surge_path()

    parser = argparse.ArgumentParser(
        description="K-fold deep ensemble UQ: K sklearn MLPs, different seeds, same SURGE split.",
    )
    parser.add_argument("--parquet", type=Path, default=_DEFAULT_PARQUET)
    parser.add_argument("--metadata", type=Path, default=_DEFAULT_METADATA)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-members", type=int, default=5, metavar="K")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--build-iteration", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    _ensure_parquet(args.parquet.resolve(), args.build_iteration)

    from sklearn.metrics import mean_squared_error, r2_score

    from surge.dataset import SurrogateDataset
    from surge.engine import EngineRunConfig, ModelSpec, SurrogateEngine

    dataset = SurrogateDataset.from_path(
        args.parquet,
        format="auto",
        metadata_path=args.metadata,
    )
    engine = SurrogateEngine(
        run_config=EngineRunConfig(
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            standardize_inputs=True,
            standardize_outputs=True,
            random_state=args.split_seed,
        ),
    )
    engine.configure_dataframe(dataset.df, dataset.input_columns, dataset.output_columns)
    engine.prepare()

    raw = engine.get_raw_splits()
    y_val = np.asarray(raw.y_val)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)

    hidden = tuple(int(x) for x in args.hidden)
    seeds = _member_seeds(args.base_seed, args.n_members)

    member_metrics: list[dict[str, float]] = []
    val_preds: list[np.ndarray] = []

    print(
        f"Dataset rows={len(dataset.df)}  outputs={len(dataset.output_columns)}",
        flush=True,
    )
    print(
        f"split_seed={args.split_seed}  K={args.n_members}  hidden={hidden}",
        flush=True,
    )

    for k, rs in enumerate(seeds):
        spec = ModelSpec(
            key="sklearn.mlp",
            name=f"ens_mlp_{k}",
            params={
                "hidden_layer_sizes": hidden,
                "max_iter": int(args.max_iter),
                "random_state": int(rs),
                "ensemble_n": 1,
            },
            request_uncertainty=False,
            store_predictions=True,
        )
        result = engine.train_model(spec, record=False)
        pval = np.asarray(result.predictions["val"]["y_pred"])
        if pval.ndim == 1:
            pval = pval.reshape(-1, 1)
        val_preds.append(pval)
        member_metrics.append(
            {"r2": float(result.val_metrics["r2"]), "rmse": float(result.val_metrics["rmse"])},
        )
        print(
            f"  member {k + 1}/{args.n_members}  random_state={rs}  "
            f"val_r2={result.val_metrics['r2']:.4f}  val_rmse={result.val_metrics['rmse']:.6f}",
            flush=True,
        )

    stacked = np.stack(val_preds, axis=0)
    mean_pred = stacked.mean(axis=0)
    std_pred = stacked.std(axis=0, ddof=0)

    r2_mean = float(r2_score(y_val, mean_pred, multioutput="uniform_average"))
    rmse_mean = float(np.sqrt(mean_squared_error(y_val, mean_pred)))

    print("", flush=True)
    print(
        f"Ensemble val: R2={r2_mean:.4f}  RMSE={rmse_mean:.6f}",
        flush=True,
    )
    print(
        f"Across-member std: mean(std)={float(np.mean(std_pred)):.6f}  "
        f"max(std)={float(np.max(std_pred)):.6f}",
        flush=True,
    )

    summary = {
        "n_members": args.n_members,
        "member_seeds": seeds,
        "member_val_metrics": member_metrics,
        "ensemble_val_r2": r2_mean,
        "ensemble_val_rmse": rmse_mean,
        "parquet": str(args.parquet.resolve()),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
