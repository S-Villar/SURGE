#!/usr/bin/env python3
"""
Build delta p dataset in per-mode format: one row per (case, m).

Each row: inputs (eq_*, input_*, n, m) + outputs (200 profile values).
Model learns: given (equilibrium, n, m), predict δp_n,m(ψ_N) - the amplitude
profile vs psi for that mode. Native resolution (200 m-modes × 200 psi).

Usage:
  python build_delta_p_per_mode.py /pscratch/.../batch_16 --out data/datasets/SPARC/delta_p_per_mode.pkl
  python build_delta_p_per_mode.py /global/cfs/.../data/m3dc1 \\
      --out data/datasets/SPARC/delta_p_per_mode_cfs_sdata_complex_v2.parquet \\
      --filename sdata_complex_v2.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import pandas as pd

from dataset_complex_v2 import (
    COMPLEX_V2_FILENAME,
    PER_MODE_INPUT_COLS,
    find_complex_v2_files,
    load_per_mode_for_surge,
    load_single_complex_v2_per_mode,
)


def _write_parquet_chunked(
    batch_dir: Path,
    out: Path,
    *,
    filename: str,
    max_cases: Optional[int],
    chunk_cases: int,
    verbose: bool,
) -> Optional[Tuple[int, list[str], list[str]]]:
    """Stream HDF5 cases to a single Parquet file without holding ~2M rows in RAM."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        print("Error: pyarrow is required for Parquet output. pip install pyarrow", file=sys.stderr)
        raise SystemExit(1) from e

    paths = find_complex_v2_files(batch_dir, filename=filename)
    if not paths:
        print(f"Error: no {filename} under {batch_dir}", file=sys.stderr)
        return None
    if max_cases is not None:
        paths = paths[: max_cases]
    if verbose:
        print(f"Streaming {len(paths)} cases to {out} (chunk_cases={chunk_cases})")

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    input_cols: list[str] = []
    output_cols: list[str] = []
    total_rows = 0

    for start in range(0, len(paths), chunk_cases):
        chunk_paths = paths[start : start + chunk_cases]
        all_rows: list = []
        for p in chunk_paths:
            try:
                all_rows.extend(
                    load_single_complex_v2_per_mode(
                        p,
                        spectrum_field="p",
                        spectrum_time_idx=-1,
                        use_magnitude=True,
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"  Skip {p}: {e}")
        if not all_rows:
            continue
        df = pd.DataFrame(all_rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(out), schema, compression="snappy")
            input_cols = [c for c in PER_MODE_INPUT_COLS if c in df.columns]
            output_cols = [c for c in df.columns if c.startswith("output_p_")]
        else:
            table = table.cast(schema)
        writer.write_table(table)
        total_rows += len(df)
        if verbose and (start // chunk_cases + 1) % 20 == 0:
            print(f"  ... {total_rows} rows written")
        del df, table, all_rows

    if writer is not None:
        writer.close()
    if total_rows == 0:
        return None
    return total_rows, input_cols, output_cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build delta p per-mode dataset (n, m → profile)"
    )
    parser.add_argument("batch_dir", type=Path, help="Batch root")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output .pkl or .parquet path")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases")
    parser.add_argument(
        "--chunk-cases",
        type=int,
        default=150,
        help="HDF5 files per chunk when writing Parquet (lowers peak RAM). Ignored for .pkl.",
    )
    parser.add_argument(
        "--filename",
        default=COMPLEX_V2_FILENAME,
        help=f"HDF5 filename under each sparc dir (default: {COMPLEX_V2_FILENAME}). "
        "Use sdata_complex_v2.h5 for CFS bulk trees.",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if not args.batch_dir.exists():
        print(f"Error: {args.batch_dir} not found", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    use_parquet = out.suffix.lower() in (".parquet", ".pq")

    if use_parquet:
        result = _write_parquet_chunked(
            args.batch_dir,
            out,
            filename=args.filename,
            max_cases=args.max_cases,
            chunk_cases=args.chunk_cases,
            verbose=not args.quiet,
        )
        if result is None:
            print("Error: no data loaded", file=sys.stderr)
            return 1
        total_rows, input_cols, output_cols = result
        print(f"\nStreamed {total_rows} rows")
        print(f"Inputs: {len(input_cols)} | Outputs: {len(output_cols)} (profile pts)")
        print(f"Saved: {out}")
        return 0

    df, input_cols, output_cols = load_per_mode_for_surge(
        args.batch_dir,
        filename=args.filename,
        max_cases=args.max_cases,
        verbose=not args.quiet,
    )

    if df.empty:
        print("Error: no data loaded", file=sys.stderr)
        return 1

    print(f"\nDataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Inputs: {len(input_cols)} (incl. n, m) | Outputs: {len(output_cols)} (profile pts)")
    print(f"Example: n={df['n'].iloc[0]}, m={df['m'].iloc[0]} → profile length {len(output_cols)}")

    df.to_pickle(out)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
