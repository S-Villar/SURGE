#!/usr/bin/env python3
"""Zero-GPU diagnosis: what drives per-case RZ field relL2 error?

Joins test-split aligned relL2 (from eval_gallery/percase CSV) with ground-truth
field structure: spectrum peak m, peak ψ_N, RZ high-spatial-frequency fraction,
peak amplitude, and linear stability γ.

Usage:
  python scripts/m3dc1/internal/analyze_rz_field_difficulty.py \\
      --run runs/rz_field_gaugefix_complex_g201 \\
      --out runs/rz_field_gaugefix_complex_g201/difficulty_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from gauge_fix import gauge_fix_field, _sanitize_field  # noqa: E402
import train_spectrum_image as Ts  # noqa: E402
import train_rz_field_image as Trz  # noqa: E402
from dataset_complex_v2 import find_complex_v2_files  # noqa: E402


def _gamma_from_h5(path: Path) -> float:
    with h5py.File(path, "r") as f:
        rg = f["runs"][list(f["runs"].keys())[0]]
        if "growth_rate" not in rg or "0" not in rg["growth_rate"]:
            return float("nan")
        return float(rg["growth_rate"]["0"][()])


def _spectrum_features(path: Path) -> Dict[str, float]:
    c = Ts._read_case(path, "p")
    if c is None:
        return {"peak_m": np.nan, "peak_psi": np.nan, "frac_hi": np.nan}
    mag = c["mag"]
    mm = c["m_modes"].astype(float)
    psi = c["psi"]
    E = (mag ** 2).sum(axis=1)
    if E.sum() <= 0:
        return {"peak_m": np.nan, "peak_psi": np.nan, "frac_hi": np.nan}
    peak_m = float(mm[int(np.argmax(E))])
    peak_psi = float(psi[int(np.argmax((mag ** 2).sum(axis=0)))])
    m_hi = 20.0
    frac_hi = float((mag ** 2)[np.abs(mm) > m_hi].sum() / (mag ** 2).sum())
    return {"peak_m": peak_m, "peak_psi": peak_psi, "frac_hi": frac_hi}


def _rz_spatial_features(
    path: Path,
    *,
    grid: int = 201,
    gauge_fix: bool = False,
    midplane_z: str = "axis",
    gauge_ref: str = "peak",
    peak_window: int = 3,
) -> Dict[str, float]:
    """High-frequency fraction, dominant k, peak ψ_N on gauge-fixed Re(δp) RZ field."""
    c = Trz._read_rz_case_complex(path)
    if c is None:
        return {"peak_psi_rz": np.nan, "hf_frac_rz": np.nan, "dom_k_norm": np.nan, "peak_amp": np.nan}
    fc = _sanitize_field(np.asarray(c["field_complex"], np.complex128))
    scale = float(np.nanmax(np.abs(fc))) or 1.0
    fn = fc / scale
    if gauge_fix:
        gf = gauge_fix_field(
            fn, Z=c.get("Z"), z_axis=float(c.get("z_axis", 0.0)),
            midplane_z=midplane_z, gauge_ref=gauge_ref, peak_window=peak_window,
        )
        fn = gf.field_gf
    re = Trz._resize_2d(np.real(fn).astype(np.float32), grid)
    re = np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)
    psin = Trz._resize_2d(Trz._finite_array(c["psin"]), grid)
    flat_i = int(np.argmax(np.abs(re)))
    peak_psi_rz = float(psin.ravel()[flat_i])
    peak_amp = float(scale)

    f2 = np.fft.rfft2(re - np.mean(re))
    H, W_r = f2.shape
    ky = np.fft.fftfreq(H)
    kx = np.arange(W_r) / max(H, 1)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    kmag = np.sqrt(KY ** 2 + KX ** 2)
    pw = (np.abs(f2) ** 2)
    pw[0, 0] = 0.0
    if pw.sum() <= 0:
        hf_frac = 0.0
        dom_k_norm = 0.0
    else:
        thr = np.percentile(pw.ravel(), 75)
        hf_frac = float(pw[pw >= thr].sum() / pw.sum())
        dom_k_norm = float(kmag.ravel()[int(np.argmax(pw))])

    return {"peak_psi_rz": peak_psi_rz, "hf_frac_rz": hf_frac, "dom_k_norm": dom_k_norm, "peak_amp": peak_amp}


def _load_relL2(run: Path) -> Dict[str, Dict[str, float]]:
    csv_path = run / "eval_gallery" / "percase_test_relL2.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}; run plot_rz_field_gallery.py first")
    out: Dict[str, Dict[str, float]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            out[row["key"]] = {
                "raw_relL2": float(row["raw_relL2"]),
                "aligned_relL2": float(row["aligned_relL2"]),
            }
    return out


def _build_key_paths(batch_dir: str, filename: str, keys_needed: set) -> Dict[str, Path]:
    paths = find_complex_v2_files(batch_dir, filename=filename)
    key_paths: Dict[str, Path] = {}
    for p in paths:
        # fast key from path: run{N}/sparc_{id}
        parts = Path(p).parts
        try:
            run_id = parts[-3]
            eq_id = parts[-2]
            key = f"{run_id}_{eq_id}"
        except IndexError:
            continue
        if key not in keys_needed:
            continue
        key_paths[key] = Path(p)
        if len(key_paths) >= len(keys_needed):
            break
    return key_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-dir", default=None)
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    out = Path(args.out) if args.out else run / "difficulty_analysis"
    out.mkdir(parents=True, exist_ok=True)
    batch_dir = args.batch_dir or cfg["batch_dir"]

    rel = _load_relL2(run)
    splits = json.loads((run / "splits.json").read_text())
    test_keys = splits["test_keys"]
    print(f"Test cases with relL2: {len(rel)}  split test_keys: {len(test_keys)}")

    key_paths = _build_key_paths(batch_dir, args.filename, set(test_keys))
    rows: List[dict] = []
    for i, key in enumerate(test_keys):
        if key not in rel:
            continue
        path = key_paths.get(key)
        if path is None:
            continue
        spec = _spectrum_features(path)
        rz = _rz_spatial_features(
            path, grid=cfg.get("grid", 201),
            gauge_fix=cfg.get("gauge_fix", False),
            midplane_z=cfg.get("midplane_z", "axis"),
            gauge_ref=cfg.get("gauge_ref", "peak"),
            peak_window=cfg.get("peak_window", 3),
        )
        gamma = _gamma_from_h5(path)
        rows.append({
            "key": key,
            "aligned_relL2": rel[key]["aligned_relL2"],
            "raw_relL2": rel[key]["raw_relL2"],
            "gamma": gamma,
            "unstable": bool(np.isfinite(gamma) and gamma > 0),
            **spec,
            **rz,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test_keys)}", flush=True)

    if not rows:
        raise SystemExit("No joined rows — check paths and percase CSV")

    aligned = np.array([r["aligned_relL2"] for r in rows])
    print(f"Joined n={len(rows)}  aligned med={np.median(aligned):.4f}")

    # Stratified tables
    def _med(mask: np.ndarray) -> float:
        a = aligned[mask]
        return float(np.median(a)) if mask.any() else float("nan")

    unstable = np.array([r["unstable"] for r in rows])
    stab_tbl = {
        "unstable": {"n": int(unstable.sum()), "median_aligned_relL2": _med(unstable)},
        "stable": {"n": int((~unstable).sum()), "median_aligned_relL2": _med(~unstable)},
    }

    peak_m = np.array([r["peak_m"] for r in rows])
    peak_m_abs = np.abs(peak_m)
    m_bins = [(0, 5), (5, 10), (10, 20), (20, np.inf)]
    m_tbl = []
    for lo, hi in m_bins:
        m = np.isfinite(peak_m_abs) & (peak_m_abs > lo) & (peak_m_abs <= hi)
        m_tbl.append({
            "bin": f"|peak_m| in ({lo},{hi}]",
            "n": int(m.sum()),
            "median_aligned_relL2": _med(m),
        })

    frac_hi = np.array([r["frac_hi"] for r in rows])
    hi_bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 1.01)]
    hi_tbl = []
    for lo, hi in hi_bins:
        m = np.isfinite(frac_hi) & (frac_hi >= lo) & (frac_hi < hi)
        hi_tbl.append({
            "bin": f"frac_hi in [{lo},{hi})",
            "n": int(m.sum()),
            "median_aligned_relL2": _med(m),
        })

    dom_k = np.array([r["dom_k_norm"] for r in rows])
    dk_bins = [(0, 0.15), (0.15, 0.35), (0.35, 0.55), (0.55, 1.01)]
    dk_tbl = []
    for lo, hi in dk_bins:
        m = np.isfinite(dom_k) & (dom_k >= lo) & (dom_k < hi)
        dk_tbl.append({
            "bin": f"dom_k_norm in [{lo},{hi})",
            "n": int(m.sum()),
            "median_aligned_relL2": _med(m),
        })

    peak_psi = np.array([r["peak_psi"] for r in rows])
    psi_bins = [(0, 0.7), (0.7, 0.9), (0.9, 1.01)]
    psi_tbl = []
    for lo, hi in psi_bins:
        m = np.isfinite(peak_psi) & (peak_psi >= lo) & (peak_psi < hi)
        psi_tbl.append({
            "bin": f"peak_psi in [{lo},{hi})",
            "n": int(m.sum()),
            "median_aligned_relL2": _med(m),
        })

    hf = np.array([r["hf_frac_rz"] for r in rows])
    hf_bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.01)]
    hf_tbl = []
    for lo, hi in hf_bins:
        m = np.isfinite(hf) & (hf >= lo) & (hf < hi)
        hf_tbl.append({
            "bin": f"hf_frac_rz in [{lo},{hi})",
            "n": int(m.sum()),
            "median_aligned_relL2": _med(m),
        })

    peak_psi_rz_arr = np.array([r["peak_psi_rz"] for r in rows])
    corr_psi_rz = (
        float(np.corrcoef(aligned[np.isfinite(peak_psi_rz_arr)], peak_psi_rz_arr[np.isfinite(peak_psi_rz_arr)])[0, 1])
        if np.isfinite(peak_psi_rz_arr).sum() > 2 else float("nan")
    )
    summary = {
        "run": str(run),
        "n_test_joined": len(rows),
        "aligned_relL2": {
            "median": float(np.median(aligned)),
            "p90": float(np.percentile(aligned, 90)),
            "min": float(aligned.min()),
            "max": float(aligned.max()),
        },
        "correlations": {
            "aligned_vs_abs_peak_m": float(np.corrcoef(aligned, peak_m_abs)[0, 1]) if np.isfinite(peak_m_abs).sum() > 2 else float("nan"),
            "aligned_vs_peak_psi": float(np.corrcoef(aligned[np.isfinite(peak_psi)], peak_psi[np.isfinite(peak_psi)])[0, 1]) if np.isfinite(peak_psi).sum() > 2 else float("nan"),
            "aligned_vs_hf_frac_rz": float(np.corrcoef(aligned[np.isfinite(hf)], hf[np.isfinite(hf)])[0, 1]) if np.isfinite(hf).sum() > 2 else float("nan"),
            "aligned_vs_dom_k_norm": float(np.corrcoef(aligned[np.isfinite(dom_k)], dom_k[np.isfinite(dom_k)])[0, 1]) if np.isfinite(dom_k).sum() > 2 else float("nan"),
            "aligned_vs_frac_hi": float(np.corrcoef(aligned, frac_hi)[0, 1]) if np.isfinite(frac_hi).sum() > 2 else float("nan"),
            "aligned_vs_peak_psi_rz": corr_psi_rz,
        },
        "by_stability": stab_tbl,
        "by_abs_peak_m": m_tbl,
        "by_frac_hi": hi_tbl,
        "by_dom_k_norm": dk_tbl,
        "by_peak_psi": psi_tbl,
        "by_hf_frac_rz": hf_tbl,
    }
    (out / "difficulty_summary.json").write_text(json.dumps(summary, indent=2))

    # CSV for downstream
    with (out / "percase_difficulty.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes[0, 0]
        ax.scatter(dom_k, aligned, s=6, alpha=0.35, c=unstable, cmap="coolwarm")
        ax.set_xlabel("RZ dominant k (normalized)")
        ax.set_ylabel("aligned relL2")
        ax.set_title("relL2 vs dominant spatial freq (red=unstable)")

        ax = axes[0, 1]
        ax.scatter(frac_hi, aligned, s=6, alpha=0.35, c=unstable, cmap="coolwarm")
        ax.set_xlabel("spectrum frac_hi (|m|>20)")
        ax.set_ylabel("aligned relL2")
        ax.set_title("relL2 vs spectrum hi-m tail")

        ax = axes[1, 0]
        peak_psi_rz = np.array([r["peak_psi_rz"] for r in rows])
        ax.scatter(peak_psi_rz, aligned, s=6, alpha=0.35, c=unstable, cmap="coolwarm")
        ax.set_xlabel("RZ peak ψ_N (gauge-fixed Re)")
        ax.set_ylabel("aligned relL2")
        ax.set_title("relL2 vs RZ peak location")
        ax.axvline(0.9, color="k", ls=":", lw=1)

        ax = axes[1, 1]
        for lab, mask, col in [("unstable", unstable, "C3"), ("stable", ~unstable, "C0")]:
            a = aligned[mask]
            if len(a):
                ax.hist(a, bins=30, alpha=0.5, label=f"{lab} n={mask.sum()}", color=col)
        ax.set_xlabel("aligned relL2")
        ax.set_title("by stability")
        ax.legend(fontsize=8)

        fig.suptitle(f"RZ field difficulty — {run.name} (n={len(rows)})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "difficulty_scatter.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {out / 'difficulty_scatter.png'}")
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out / 'difficulty_summary.json'}")


if __name__ == "__main__":
    main()
