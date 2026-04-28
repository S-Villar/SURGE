# -*- coding: utf-8 -*-
"""Build growing labeled tables (Parquet) for ROSE + SURGE demos: synthetic or optional M3DC1."""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

_PKL_NAME = "sparc-m3dc1-D1.pkl"

SYN_N_BASE = 50
SYN_N_STEP = 30
SYN_SHUFFLE_SEED = 42

M3DC1_N_BASE = 600
M3DC1_N_STEP = 600
M3DC1_SHUFFLE_SEED = 42

_repo_root_cache: Path | None = None
_m3dc1_pool: pd.DataFrame | None = None


def repo_root_from_here() -> Path:
    global _repo_root_cache
    if _repo_root_cache is None:
        _repo_root_cache = Path(__file__).resolve().parents[2]
    return _repo_root_cache


def example_dir() -> Path:
    return Path(__file__).resolve().parent


def workspace_dir() -> Path:
    d = example_dir() / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON handoff files atomically for ROSE/SURGE stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def m3dc1_pkl_path() -> Path:
    return repo_root_from_here() / "data" / "datasets" / "M3DC1" / _PKL_NAME


def m3dc1_metadata_path() -> Path:
    return repo_root_from_here() / "data" / "datasets" / "M3DC1" / "sparc_m3dc1_D1_metadata.yaml"


def _m3dc1_columns() -> list[str]:
    import yaml  # noqa: WPS433
    meta = yaml.safe_load(m3dc1_metadata_path().read_text(encoding="utf-8"))
    return list(meta["inputs"]) + list(meta["outputs"])


def _install_numpy_pickle_compat() -> None:
    """Let NumPy 1.x read pickles written by NumPy 2.x."""
    import sys

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = np.core.numeric
    if "numpy._core.multiarray" not in sys.modules:
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


def _load_m3dc1_pool(*, shuffle_seed: int = M3DC1_SHUFFLE_SEED) -> pd.DataFrame:
    global _m3dc1_pool
    if _m3dc1_pool is not None:
        return _m3dc1_pool
    pkl = m3dc1_pkl_path()
    if not pkl.is_file():
        raise FileNotFoundError(
            f"Missing M3DC1 PKL at {pkl}. Copy from SURGE, e.g.:\n"
            f"  cp /path/to/SURGE/data/datasets/SPARC/{_PKL_NAME} {pkl}\n"
            "Or use --dataset synthetic for Examples 1-3 without external data.",
        )
    _install_numpy_pickle_compat()
    df = pd.read_pickle(pkl)
    if "completeness" in df.columns:
        df = df.loc[df["completeness"] == 1].copy()
    cols = _m3dc1_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"M3DC1 PKL missing columns: {missing}")
    df = df[cols].dropna()
    rng = np.random.default_rng(int(shuffle_seed))
    order = rng.permutation(len(df))
    _m3dc1_pool = df.iloc[order].reset_index(drop=True)
    return _m3dc1_pool


def build_synthetic_parquet(
    iteration: int,
    *,
    seed: int = SYN_SHUFFLE_SEED,
    fixed_rows: int | None = None,
) -> Path:
    rng_seed = seed if fixed_rows is not None else seed + iteration
    rng = np.random.default_rng(rng_seed)
    n = int(fixed_rows) if fixed_rows is not None else SYN_N_BASE + iteration * SYN_N_STEP
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.12, size=n)
    y = 0.7 * np.sin(2.0 * x0) + 0.3 * (x1**2) + noise
    df = pd.DataFrame({"x0": x0, "x1": x1, "y": y})
    path = workspace_dir() / "synthetic_active_learning.parquet"
    df.to_parquet(path, index=False)
    return path


def build_m3dc1_parquet(
    iteration: int,
    *,
    fixed_rows: int | None = None,
    use_full_dataset: bool = False,
) -> Path:
    pool = _load_m3dc1_pool()
    if use_full_dataset:
        n = len(pool)
    elif fixed_rows is not None:
        n = min(int(fixed_rows), len(pool))
    else:
        n = min(M3DC1_N_BASE + iteration * M3DC1_N_STEP, len(pool))
    if n < 1:
        raise ValueError("M3DC1 pool is empty after filtering.")
    chunk = pool.iloc[:n].copy()
    path = workspace_dir() / "sparc_m3dc1_active_learning.parquet"
    chunk.to_parquet(path, index=False)
    return path


def training_row_plan(
    iteration: int,
    *,
    dataset: str,
    fixed_rows: int | None = None,
    use_full_dataset: bool = False,
) -> dict:
    """Return row-count context for the current demo iteration."""
    if dataset == "m3dc1":
        pool = _load_m3dc1_pool()
        if use_full_dataset:
            requested = len(pool)
            row_policy = "full shuffled complete M3DC1 pool for every candidate"
        elif fixed_rows is not None:
            requested = int(fixed_rows)
            row_policy = f"fixed prefix of shuffled complete M3DC1 pool; {requested} rows/iteration"
        else:
            requested = M3DC1_N_BASE + iteration * M3DC1_N_STEP
            row_policy = f"prefix of shuffled complete M3DC1 pool; +{M3DC1_N_STEP} rows/iteration"
        return {
            "n_rows_requested": requested,
            "n_rows": min(requested, len(pool)),
            "n_rows_total": len(pool),
            "row_policy": row_policy,
        }
    if dataset == "synthetic":
        if fixed_rows is not None:
            n = int(fixed_rows)
            row_policy = f"fixed synthetic sample; {n} rows for every candidate"
        else:
            n = SYN_N_BASE + iteration * SYN_N_STEP
            row_policy = f"fresh synthetic sample; +{SYN_N_STEP} rows/iteration"
        return {
            "n_rows_requested": n,
            "n_rows": n,
            "n_rows_total": None,
            "row_policy": row_policy,
        }
    raise ValueError(f"Unknown dataset mode {dataset!r}; use 'synthetic' or 'm3dc1'.")


def build_training_parquet(
    iteration: int,
    *,
    dataset: str,
    fixed_rows: int | None = None,
    use_full_dataset: bool = False,
) -> Path:
    if dataset == "m3dc1":
        return build_m3dc1_parquet(
            iteration,
            fixed_rows=fixed_rows,
            use_full_dataset=use_full_dataset,
        )
    if dataset == "synthetic":
        return build_synthetic_parquet(iteration, fixed_rows=fixed_rows)
    raise ValueError(f"Unknown dataset mode {dataset!r}; use 'synthetic' or 'm3dc1'.")
