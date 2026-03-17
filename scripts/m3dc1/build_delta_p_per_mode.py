#!/usr/bin/env python3
"""
Build delta p dataset in per-mode format: one row per (case, m).

Each row: inputs (eq_*, input_*, n, m) + outputs (200 profile values).
Model learns: given (equilibrium, n, m), predict δp_n,m(ψ_N) - the amplitude
profile vs psi for that mode. Native resolution (200 m-modes × 200 psi).

Usage:
  python build_delta_p_per_mode.py /pscratch/.../batch_16 --out data/datasets/SPARC/delta_p_per_mode.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from dataset_complex_v2 import load_per_mode_for_surge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build delta p per-mode dataset (n, m → profile)"
    )
    parser.add_argument("batch_dir", type=Path, help="Batch root")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output .pkl path")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if not args.batch_dir.exists():
        print(f"Error: {args.batch_dir} not found", file=sys.stderr)
        return 1

    df, input_cols, output_cols = load_per_mode_for_surge(
        args.batch_dir,
        max_cases=args.max_cases,
        verbose=not args.quiet,
    )

    if df.empty:
        print("Error: no data loaded", file=sys.stderr)
        return 1

    print(f"\nDataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Inputs: {len(input_cols)} (incl. n, m) | Outputs: {len(output_cols)} (profile pts)")
    print(f"Example: n={df['n'].iloc[0]}, m={df['m'].iloc[0]} → profile length {len(output_cols)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
