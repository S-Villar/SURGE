#!/usr/bin/env python3
"""Build a per-case scalar dataset (one row per (run_id, eq_id)) from M3DC1
csdata HDF5 files, for scalar-target surrogates such as growth rate (gamma).

This deliberately reads only scalars (equilibrium shape, parset controls, flux
profile scalars, growth rate) and skips the 2D spectra, so it is fast over the
full batch (~10k cases). SURGE consumes the resulting table as a generic
dataframe (no M3DC1 import needed at train time).

Usage:
    python scripts/m3dc1/internal/build_case_scalar_dataset.py \
        /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
        --filename csdata_deltap_b_ver.h5 \
        --out data/datasets/SPARC/case_scalars_ver.parquet \
        --metadata-out data/datasets/SPARC/case_scalars_ver_metadata.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import yaml

# Candidate scalar input columns emitted by this builder.
INPUT_COLS: List[str] = [
    "eq_R0",
    "eq_a",
    "eq_kappa",
    "eq_delta",
    "q0",
    "q95",
    "qmin",
    "p0",
    "input_ntor",
    "input_pscale",
    "input_batemanscale",
]


def _decode(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "decode"):
        return value.decode()
    if isinstance(value, bytes):
        return value.decode()
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            item = arr.item()
            return item.decode() if isinstance(item, bytes) else str(item)
    except Exception:
        pass
    return str(value)


def _extract_row(path: Path) -> Optional[Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        if "runs" not in f:
            return None
        run_names = list(f["runs"].keys())
        if not run_names:
            return None
        rg = f["runs"][run_names[0]]

        row: Dict[str, Any] = {
            "run_id": _decode(rg.get("runID"), run_names[0]),
            "eq_id": _decode(rg.get("eqID"), "eq"),
        }

        if "miller" in rg:
            for src, dst in [("R0", "eq_R0"), ("a", "eq_a"), ("kappa", "eq_kappa"), ("delta", "eq_delta")]:
                if src in rg["miller"]:
                    row[dst] = float(rg["miller"][src][()])

        if "parset" in rg and "names" in rg["parset"] and "values" in rg["parset"]:
            names = rg["parset"]["names"]
            values = rg["parset"]["values"]
            for i, nm in enumerate(names):
                name = _decode(nm)
                if i < len(values):
                    row[f"input_{name}"] = float(values[i])

        if "flux_average" in rg:
            fa = rg["flux_average"]
            if "q" in fa and "profile" in fa["q"] and "psin" in fa["q"]:
                qprof = np.asarray(fa["q"]["profile"])
                psin = np.asarray(fa["q"]["psin"])
                if qprof.size > 0:
                    row["q0"] = float(qprof[0])
                    row["qmin"] = float(np.min(qprof))
                    if psin.size > 1:
                        idx95 = int(0.95 * (len(psin) - 1))
                        row["q95"] = float(qprof[min(idx95, len(qprof) - 1)])
            if "p" in fa and "profile" in fa["p"]:
                pprof = np.asarray(fa["p"]["profile"])
                if pprof.size > 0:
                    row["p0"] = float(pprof[0])
        if "q95" in rg:
            row["q95"] = float(rg["q95"][()])

        if "growth_rate" in rg:
            gr0 = rg["growth_rate"].get("0")
            if gr0 is not None:
                row["gamma"] = float(gr0[()])

        return row


def build(
    batch_dir: Path,
    filename: str,
    run_pattern: str = "run*",
    sparc_pattern: str = "sparc_*",
    max_cases: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    paths = sorted(batch_dir.glob(f"{run_pattern}/{sparc_pattern}/{filename}"))
    if not paths:
        raise FileNotFoundError(
            f"No {filename} under {batch_dir}/{run_pattern}/{sparc_pattern}"
        )
    if max_cases is not None:
        paths = paths[:max_cases]
    if verbose:
        print(f"Scanning {len(paths)} case files for scalars...")

    rows: List[Dict[str, Any]] = []
    n_skip = 0
    for i, p in enumerate(paths):
        try:
            row = _extract_row(p)
        except Exception as e:  # noqa: BLE001 - resilient bulk scan
            row = None
            if verbose and n_skip < 20:
                print(f"  skip {p}: {e}")
        if row is None:
            n_skip += 1
            continue
        rows.append(row)
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(paths)} scanned, {len(rows)} rows")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Built {len(df)} rows ({n_skip} skipped), {len(df.columns)} columns")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata-out", type=Path, default=None)
    ap.add_argument("--target", default="gamma", help="Output column (default: gamma)")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--run-pattern", default="run*")
    ap.add_argument("--sparc-pattern", default="sparc_*")
    args = ap.parse_args()

    df = build(
        args.batch_dir,
        filename=args.filename,
        run_pattern=args.run_pattern,
        sparc_pattern=args.sparc_pattern,
        max_cases=args.max_cases,
    )

    if args.target not in df.columns:
        raise SystemExit(
            f"Target '{args.target}' not present. Available: {sorted(df.columns)}"
        )
    before = len(df)
    df = df.dropna(subset=[args.target]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with missing {args.target}; {len(df)} remain")

    input_cols = [c for c in INPUT_COLS if c in df.columns]
    # Only keep input columns that are fully populated to avoid NaN-driven splits.
    input_cols = [c for c in input_cols if df[c].notna().all()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    elif args.out.suffix in (".pkl", ".pickle"):
        df.to_pickle(args.out)
    else:
        df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)")

    # SURGE's dataset analyzer reads manual columns from `inputs`/`outputs`.
    meta = {
        "inputs": input_cols,
        "outputs": [args.target],
        "group_columns": ["run_id", "eq_id"],
        "notes": f"Per-case scalars from {args.filename}; target={args.target}.",
    }
    meta_path = args.metadata_out or args.out.with_name(args.out.stem + "_metadata.yaml")
    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    print(f"Wrote {meta_path}")
    print(f"input_columns ({len(input_cols)}): {input_cols}")


if __name__ == "__main__":
    main()
