#!/usr/bin/env python3
"""
Discover M3DC1 postprocessed data across pscratch batch directories.

Scans run*/sparc_*/ for H5 files used in delta p surrogate training:
- sdata_pertfields_grid_complex_v2.h5 (delta p spectra, per-run - SURGE uses this)
- sdata_complex_v2.h5 (aggregated delta p)
- sdata*.h5 (legacy)
- C1.h5 (raw simulation output, not postprocessed for delta p)

Outputs a mapping: batch/location -> file counts by type.

Usage:
  python scripts/m3dc1/discover_m3dc1_data.py
  python scripts/m3dc1/discover_m3dc1_data.py ${SURGE_SCRATCH}/mp288/jobs
  python scripts/m3dc1/discover_m3dc1_data.py /path/to/jobs --out m3dc1_data_map.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Filenames we care about (delta p / complex postprocessing)
COMPLEX_V2 = "sdata_pertfields_grid_complex_v2.h5"  # per-run delta p spectra
SDATA_COMPLEX_V2 = "sdata_complex_v2.h5"             # aggregated
SDATA_PATTERN = "sdata*.h5"                          # legacy
C1_H5 = "C1.h5"                                      # raw sim (for reference)


def discover_in_dir(
    root: Path,
    *,
    run_pattern: str = "run*",
    sparc_pattern: str = "sparc_*",
    max_runs: int | None = None,
) -> dict[str, dict[str, int]]:
    """
    Scan root for run*/sparc_* and count H5 files by type.
    Returns {run_name: {file_type: count}}.
    """
    root = Path(root)
    if not root.exists():
        return {}

    results: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    run_dirs = sorted(root.glob(run_pattern))
    if not run_dirs:
        return dict(results)
    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        for sparc_dir in sorted(run_dir.glob(sparc_pattern)):
            if not sparc_dir.is_dir():
                continue
            loc = run_dir.name
            if (sparc_dir / COMPLEX_V2).exists():
                results[loc][COMPLEX_V2] += 1
            if (sparc_dir / SDATA_COMPLEX_V2).exists():
                results[loc]["sdata_complex_v2.h5"] += 1
            for f in sparc_dir.glob("sdata*.h5"):
                if f.name not in (COMPLEX_V2, SDATA_COMPLEX_V2):
                    results[loc]["sdata*.h5 (other)"] += 1
            if (sparc_dir / C1_H5).exists():
                results[loc][C1_H5] += 1

    return dict(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover M3DC1 complex_v2 and related H5 files across batch dirs"
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("${SURGE_SCRATCH}/mp288/jobs"),
        help="Root dir (e.g. pscratch/.../jobs) containing batch_* or run*/sparc_*",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Write mapping to YAML/JSON file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (default: human-readable)",
    )
    parser.add_argument(
        "--sample-runs",
        type=int,
        default=None,
        help="Limit runs scanned per batch (for speed on large batches)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: root not found: {root}", file=sys.stderr)
        return 1

    # Check structure: batch_* subdirs or run* directly?
    batch_dirs = sorted(root.glob("batch_*"))
    if batch_dirs:
        # Scan each batch
        all_results: dict[str, dict[str, dict[str, int]]] = {}
        total_complex_v2 = 0
        total_c1 = 0
        for bd in batch_dirs:
            if not bd.is_dir():
                continue
            batch_name = bd.name
            res = discover_in_dir(bd, max_runs=args.sample_runs)
            if res:
                # Aggregate per batch
                batch_totals: dict[str, int] = defaultdict(int)
                for loc, counts in res.items():
                    for ft, n in counts.items():
                        batch_totals[ft] += n
                all_results[batch_name] = {
                    "by_run": res,
                    "totals": dict(batch_totals),
                }
                total_complex_v2 += batch_totals.get(COMPLEX_V2, 0)
                total_c1 += batch_totals.get(C1_H5, 0)
        structure = "batch_*"
    else:
        # Flat: run* directly under root
        res = discover_in_dir(root, max_runs=args.sample_runs)
        all_results = {".": {"by_run": res, "totals": {}}}
        for loc, counts in res.items():
            for ft, n in counts.items():
                all_results["."]["totals"][ft] = all_results["."]["totals"].get(ft, 0) + n
        total_complex_v2 = all_results["."]["totals"].get(COMPLEX_V2, 0)
        total_c1 = all_results["."]["totals"].get(C1_H5, 0)
        structure = "run*/sparc_* (flat)"

    # Summary
    print("=" * 70)
    print("M3DC1 Data Discovery")
    print("=" * 70)
    print(f"Root: {root}")
    print(f"Structure: {structure}")
    print()

    for batch_name, data in sorted(all_results.items()):
        totals = data["totals"]
        n_complex = totals.get(COMPLEX_V2, 0)
        n_c1 = totals.get(C1_H5, 0)
        print(f"  {batch_name}:")
        print(f"    {COMPLEX_V2}: {n_complex}")
        print(f"    {C1_H5}: {n_c1}")
        for k, v in sorted(totals.items()):
            if k not in (COMPLEX_V2, C1_H5) and v > 0:
                print(f"    {k}: {v}")
        print()

    print("=" * 70)
    print(f"TOTAL {COMPLEX_V2}: {total_complex_v2}")
    print(f"TOTAL {C1_H5}: {total_c1}")
    print("=" * 70)

    if total_complex_v2 == 0:
        print("\nNo sdata_pertfields_grid_complex_v2.h5 found.")
        print("Delta p surrogate training requires these postprocessed files.")
        print("Check if postprocessing has been run on more batches.")
    elif total_complex_v2 < 100:
        print(f"\nOnly {total_complex_v2} complex_v2 files. Consider postprocessing more runs.")

    # Write output
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(root),
            "structure": structure,
            "totals": {"sdata_pertfields_grid_complex_v2.h5": total_complex_v2, "C1.h5": total_c1},
            "by_batch": {k: v["totals"] for k, v in all_results.items()},
            "detail": all_results,
        }
        if args.json or out.suffix == ".json":
            with open(out, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            import yaml
            try:
                with open(out, "w") as f:
                    yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                with open(out, "w") as f:
                    json.dump(payload, f, indent=2)
        print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
