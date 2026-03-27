#!/usr/bin/env python3
"""
Build a profile-only dataset: predict delta_p amplitude vs psiN for a specific (n,m) mode.

The full spectrum is (n_modes x n_psi). With mode_step=4, psi_step=4 → 50x50.
Layout: output_p_0..49 = mode 0 profile, output_p_50..99 = mode 1, etc.

Usage:
  python build_profile_dataset.py /pscratch/.../batch_16 --mode 0 --out delta_p_mode0.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from dataset_complex_v2 import load_complex_v2_for_surge


def main() -> int:
    parser = argparse.ArgumentParser(description="Build delta_p profile dataset for a specific mode")
    parser.add_argument("batch_dir", type=Path, help="Batch root")
    parser.add_argument("--mode", type=int, default=0, help="Mode index (0=first/dominant)")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output .pkl path")
    parser.add_argument("--target-shape", type=str, default="50,50", metavar="N_MODES,N_PSI",
                        help="Fixed resolution (default: 50,50)")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    parts = [int(x.strip()) for x in args.target_shape.split(",")]
    target_shape = (parts[0], parts[1]) if len(parts) == 2 else (50, 50)
    df, input_cols, output_cols = load_complex_v2_for_surge(
        args.batch_dir,
        target_shape=target_shape,
        max_cases=args.max_cases,
        verbose=True,
    )
    if df.empty:
        print("No data loaded.", file=sys.stderr)
        return 1

    # spec shape (n_modes, n_psi) flattened C-order: mode0_psi0..psiN, mode1_psi0..
    n_total = len(output_cols)
    n_modes_actual, n_psi_actual = target_shape
    n_per_mode = n_psi_actual

    mode_idx = min(args.mode, n_modes_actual - 1)
    start = mode_idx * n_per_mode
    end = start + n_per_mode
    profile_cols = output_cols[start:end]

    df_profile = df[input_cols + profile_cols].copy()
    df_profile.to_pickle(args.out)
    print(f"Saved profile for mode {mode_idx}: {len(profile_cols)} psi points → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
