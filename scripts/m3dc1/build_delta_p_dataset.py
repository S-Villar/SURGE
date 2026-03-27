#!/usr/bin/env python3
"""
Build delta p spectra DataFrame from batch dirs with sdata_pertfields_grid_complex_v2.h5.

Usage:
  python build_delta_p_dataset.py ${SURGE_SCRATCH}/mp288/jobs/batch_16
  python build_delta_p_dataset.py /path/to/batch_16 --out data/datasets/SPARC/delta_p_batch16.pkl --max-cases 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure scripts/m3dc1 is on path
_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from dataset_complex_v2 import load_complex_v2_for_surge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build delta p spectra DataFrame from batch dir with sdata_pertfields_grid_complex_v2.h5"
    )
    parser.add_argument(
        "batch_dir",
        type=Path,
        help="Batch root (e.g. /pscratch/.../batch_16)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output path (.pkl or .parquet). If omitted, only prints summary.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases (for testing)",
    )
    parser.add_argument(
        "--mode-step",
        type=int,
        default=None,
        help="Downsample spectrum: take every Nth mode (e.g. 4 → 50x50 instead of 200x200)",
    )
    parser.add_argument(
        "--psi-step",
        type=int,
        default=None,
        help="Downsample spectrum: take every Nth psi",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        default=None,
        metavar="N_MODES,N_PSI",
        help="Fixed resolution for all cases (e.g. 50,50). Same m-spectrum resolution for every case.",
    )
    parser.add_argument(
        "--eigenmodes",
        action="store_true",
        help="Include eigenmode_amp_* from C1.h5 (requires m3dc1/fpy)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    batch_dir = args.batch_dir
    if not batch_dir.exists():
        print(f"Error: batch dir not found: {batch_dir}", file=sys.stderr)
        return 1

    target_shape = None
    if args.target_shape:
        parts = [int(x.strip()) for x in args.target_shape.split(",")]
        if len(parts) != 2:
            parser.error("--target-shape must be N_MODES,N_PSI (e.g. 50,50)")
        target_shape = (parts[0], parts[1])

    df, input_cols, output_cols = load_complex_v2_for_surge(
        batch_dir,
        max_cases=args.max_cases,
        mode_step=args.mode_step,
        psi_step=args.psi_step,
        target_shape=target_shape,
        include_eigenmodes=args.eigenmodes,
        verbose=not args.quiet,
    )

    if df.empty:
        print("Error: no cases loaded. Check batch dir has run*/sparc_*/sdata_pertfields_grid_complex_v2.h5", file=sys.stderr)
        return 1

    print(f"\nDataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Inputs: {len(input_cols)} | Outputs: {len(output_cols)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".pkl", ".pickle"):
            df.to_pickle(out)
            print(f"Saved: {out}")
        elif out.suffix.lower() in (".parquet", ".pq"):
            df.to_parquet(out, index=False)
            print(f"Saved: {out}")
        else:
            df.to_csv(out, index=False)
            print(f"Saved: {out} (CSV)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
