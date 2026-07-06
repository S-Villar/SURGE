#!/usr/bin/env python3
"""Phase-honesty stress test: field relL2 with non-oracle phase injection.

Same IFFT convention as field_bench.py (native m-grid poloidal IFFT, max-normalized relL2).

Conditions:
  oracle     — true phase from HDF5 (reference, matches field_bench)
  zero       — φ=0 on all native coefficients
  random     — φ ~ Uniform(-π,π) per coefficient
  sigma_*    — true phase + Gaussian noise (radians), wrapped to [-π,π]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from field_bench import (  # noqa: E402
    CaseMeta,
    _ifft_field,
    interp_mag_to_native,
    load_case_meta,
    pred_spec_from_dex,
    rel_l2_raw,
)

CONDITIONS = ("oracle", "zero", "random", "sigma_0.25", "sigma_0.5", "sigma_1.0")


def _phase_for_condition(meta: CaseMeta, cond: str, rng: np.random.RandomState) -> np.ndarray:
    true = meta.phase.astype(np.float64)
    if cond == "oracle":
        return true
    if cond == "zero":
        return np.zeros_like(true)
    if cond == "random":
        return rng.uniform(-np.pi, np.pi, size=true.shape)
    if cond.startswith("sigma_"):
        sigma = float(cond.split("_", 1)[1])
        noisy = true + rng.normal(0.0, sigma, size=true.shape)
        return np.mod(noisy + np.pi, 2 * np.pi) - np.pi
    raise ValueError(cond)


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "frac_gt_1": float(np.mean(a > 1.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir with predictions_cache.npz")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"])
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all in split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    run = Path(args.run)
    z = np.load(run / "predictions_cache.npz", allow_pickle=True)
    split = z["split"].astype(str)
    keys = z["keys"].astype(str)
    paths = z["paths"].astype(str)
    pred = z["pred"].astype(np.float32)
    m_grid = z["m_grid"]
    psi_grid = z["psi_grid"]
    space = str(z["target_space"])
    spec_field = str(z["spectrum_field"]) if "spectrum_field" in z.files else "p"

    if args.split == "all":
        idx = np.arange(len(split))
    else:
        idx = np.where(split == args.split)[0]
    if args.max_cases and args.max_cases < len(idx):
        rng_pick = np.random.RandomState(args.seed)
        idx = np.sort(rng_pick.choice(idx, size=args.max_cases, replace=False))

    rng = np.random.RandomState(args.seed)
    meta_cache: dict[str, CaseMeta] = {}
    by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    per_case_rows = []

    print(f"run={run.name} split={args.split} n={len(idx)}", flush=True)
    for n_done, i in enumerate(idx, 1):
        meta = load_case_meta(paths[i], spec_field, meta_cache)
        ftrue = _ifft_field(meta.true_spec, meta.m_modes)

        row = {"key": keys[i], "family": keys[i].split("_", 1)[-1]}
        for cond in CONDITIONS:
            if cond == "oracle":
                pred_spec = pred_spec_from_dex(pred[i], meta, m_grid, psi_grid)
            else:
                mag_nat = interp_mag_to_native(pred[i], m_grid, psi_grid, meta)
                phase = _phase_for_condition(meta, cond, rng)
                pred_spec = mag_nat * np.exp(1j * phase)
            fpred = _ifft_field(pred_spec, meta.m_modes)
            rl2 = rel_l2_raw(fpred, ftrue)
            by_cond[cond].append(rl2)
            row[f"relL2_{cond}"] = rl2
        per_case_rows.append(row)
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(idx)}", flush=True)

    summary = {cond: summarize(v) for cond, v in by_cond.items()}
    out = Path(args.out) if args.out else run / "phase_stress_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": str(run),
        "split": args.split,
        "n_cases": len(idx),
        "conditions": summary,
        "notes": {
            "oracle": "true phase from HDF5 (same as field_bench)",
            "zero": "phi=0 on native (m, psi) coefficients",
            "random": "phi ~ Uniform(-pi,pi) per coefficient",
            "sigma_*": "true phase + Gaussian noise (radians), wrapped",
            "field_convention": "native m-grid poloidal IFFT, max-normalized relL2 (field_bench.py)",
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print(f"{'condition':14s} {'mean':>8s} {'p90':>8s} {'frac>1':>8s}")
    for cond in CONDITIONS:
        s = summary.get(cond, {})
        if not s:
            continue
        print(f"{cond:14s} {s['mean']:8.3f} {s['p90']:8.3f} {s['frac_gt_1']:8.3f}")

    csv_path = out.with_suffix(".per_case.csv")
    if per_case_rows:
        fields = sorted({k for r in per_case_rows for k in r})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_case_rows)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
