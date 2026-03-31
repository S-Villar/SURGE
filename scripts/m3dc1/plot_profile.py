#!/usr/bin/env python3
"""
Plot δp_n,m(ψ_N) profile for a given (n, m) from per-mode predictions or dataset.

Usage:
  python plot_profile.py data/datasets/SPARC/delta_p_per_mode.pkl --n 9 --m -7
  python plot_profile.py runs/m3dc1_delta_p_per_mode/predictions/rf_per_mode_test.csv --n 9 --m -7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# psi_norm from HDF5: 200 points, typically 0.0001 to 1.0
PSI_NORM = np.linspace(0.0001, 1.0, 200)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot δp_n,m(ψ_N) profile")
    parser.add_argument("path", type=Path, help="Dataset .pkl / .parquet or predictions CSV")
    parser.add_argument("--n", type=float, required=True, help="Toroidal mode n")
    parser.add_argument("--m", type=int, required=True, help="Poloidal mode m")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Output plot path")
    parser.add_argument("--row", type=int, default=0, help="Row index if multiple matches")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    suf = path.suffix.lower()
    if suf in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    profile_cols = [c for c in df.columns if c.startswith("output_p_")]
    if not profile_cols:
        print("No output_p_* columns", file=sys.stderr)
        return 1

    mask = (np.isclose(df["n"], args.n)) & (df["m"] == args.m)
    if not mask.any():
        print(f"No rows with n={args.n}, m={args.m}", file=sys.stderr)
        print(f"Available n: {df['n'].unique()[:5]}... m: {df['m'].min()}..{df['m'].max()}")
        return 1

    rows = df[mask]
    row = rows.iloc[args.row]
    profile = row[profile_cols].values.astype(float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(PSI_NORM, profile, "b-", linewidth=1.5)
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$|\delta p_{n,m}|$")
    ax.set_title(f"δp profile: n={args.n}, m={args.m}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out or path.parent / f"delta_p_n{args.n}_m{args.m}.png"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
