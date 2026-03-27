#!/usr/bin/env python3
"""
Debug script to inspect how the SURGE dataloader works for M3DC1 delta p spectra.

Shows:
- iter_batches (framework-agnostic numpy)
- to_dataloader (PyTorch, batch_size=32)
- Batch shapes and sample values
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts and project root
_scripts = Path(__file__).resolve().parent
_root = _scripts.parent.parent
for p in (str(_scripts), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

def main():
    from surge import M3DC1Dataset

    batch_dir = Path("${SURGE_SCRATCH}/mp288/jobs/batch_16")
    if not batch_dir.exists():
        print(f"Batch dir not found: {batch_dir}")
        print("Using extracted pkl if available...")
        pkl = _root / "data" / "datasets" / "SPARC" / "delta_p_batch16.pkl"
        if pkl.exists():
            import pandas as pd
            df = pd.read_pickle(pkl)
            from surge.dataset import SurrogateDataset
            inc = [c for c in df.columns if c.startswith("eq_") and c != "eq_id" or c.startswith("input_") or c in ("q0","q95","qmin","p0")]
            outc = [c for c in df.columns if c.startswith("output_p_")]
            dataset = SurrogateDataset.from_dataframe(df, input_columns=inc, output_columns=outc)
        else:
            print("No data. Run build_delta_p_dataset.py first.")
            return 1
    else:
        dataset = M3DC1Dataset.from_batch_dir(
            batch_dir,
            mode_step=4,
            psi_step=4,
            max_cases=30,
            verbose=True,
        )

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Rows: {len(dataset.df)}")
    print(f"Input columns: {len(dataset.input_columns)}")
    print(f"Output columns: {len(dataset.output_columns)}")
    print(f"Output shape per row: ({len(dataset.output_columns)},)")

    print("\n" + "="*60)
    print("1. iter_batches (framework-agnostic, batch_size=32)")
    print("="*60)
    n_batches = 0
    for X_batch, y_batch in dataset.iter_batches(batch_size=32, shuffle=False):
        n_batches += 1
        print(f"  Batch {n_batches}: X {X_batch.shape}, y {y_batch.shape}")
        if n_batches == 1:
            print(f"    X sample [0,:3]: {X_batch[0,:3]}")
            print(f"    y sample [0,:5]: {y_batch[0,:5]}")
    print(f"  Total batches: {n_batches}")

    print("\n" + "="*60)
    print("2. to_dataloader (PyTorch, batch_size=32)")
    print("="*60)
    try:
        loader = dataset.to_dataloader(batch_size=32, shuffle=True)
        print(f"  DataLoader: {loader}")
        print(f"  num_batches: {len(loader)}")
        for i, (X, y) in enumerate(loader):
            print(f"  Batch {i+1}: X {X.shape}, y {y.shape}")
            if i == 0:
                print(f"    X device: {X.device}, dtype: {X.dtype}")
            if i >= 2:
                break
    except ImportError as e:
        print(f"  PyTorch not available: {e}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
