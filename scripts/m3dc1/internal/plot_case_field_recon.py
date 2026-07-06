#!/usr/bin/env python3
"""Per-case 2x3 comparison panels (spectrum + RZ field) for a spectrum-image run.

For a trained ``|delta p|`` spectrum surrogate we already export a compact
``predictions_cache.npz`` (see ``export_predictions_cache.py``). This script picks
the worst / median / best validation cases by per-case *pattern* R2 and renders, for
each, a 2x3 figure:

    row 0 = spectrum |dp^|(m, psi_N):  ground truth | prediction | difference
    row 1 = field Re(dp)(R,Z):         ground truth | prediction | difference

The predicted field combines the model's predicted *magnitude* with the case's
*true phase* (phase is not learnable), rescaled to the case's physical peak, then
inverse-FFT'd along m and mapped onto the fpy PEST flux grid (falls back to the
poloidal (theta, psi) plane if fpy is unavailable). This mirrors the logic in
``m3dc1ml/notebooks/curate_validate_mlm3dc1_predictions.ipynb``.

Usage
-----
    python scripts/m3dc1/internal/plot_case_field_recon.py \
        --run runs/spectrum_image_full_maxnorm_log10 \
        --split val --out-dir docs/m3dc1/assets --tag maxnorm_log10

Output filenames include the run directory name (model id), e.g.
``rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_worst_maxnorm_log10.png``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "m3dc1ml" / "src"))


def _load_cache(run: Path):
    cache = run / "predictions_cache.npz"
    if not cache.exists():
        raise SystemExit(
            f"cache not found: {cache}\nRun export_predictions_cache.py first."
        )
    z = np.load(cache, allow_pickle=True)
    return z


def _pred_complex_native(b, pred_i, m_grid, psi_grid, target_space):
    from scipy.interpolate import RegularGridInterpolator

    pt = pred_i.astype(np.float64)
    mag_u = 10.0 ** pt if target_space == "log10" else np.clip(pt, 0, None)
    interp = RegularGridInterpolator(
        (m_grid, psi_grid), mag_u, bounds_error=False, fill_value=0.0
    )
    m_nat = np.asarray(b["m_modes"], float)
    psi_nat = np.asarray(b["psi_norm"], float)
    MM, PP = np.meshgrid(m_nat, psi_nat, indexing="ij")
    mag_nat = interp(np.stack([MM.ravel(), PP.ravel()], 1)).reshape(
        len(m_nat), len(psi_nat)
    )
    spec = np.asarray(b["spec_complex"])
    true_mag = np.abs(spec)
    mag_nat = mag_nat * (true_mag.max() + 1e-300)  # normalized -> physical peak
    phase = np.angle(spec)
    return mag_nat * np.exp(1j * phase)


def _recon_fields(path, pred_i, spec_field, m_grid, psi_grid, target_space):
    from m3dc1ml.io.sdata import load_complex_v2_case
    from m3dc1ml.viz.explore_case import (
        recon_real_from_spectrum,
        _flux_grid_for_bundle,
    )

    b = load_complex_v2_case(str(path), spectrum_field=spec_field)
    gt_field = recon_real_from_spectrum(b)
    pb = dict(b)
    pc = _pred_complex_native(b, pred_i, m_grid, psi_grid, target_space)
    pb["spec_complex"] = pc
    pb["spec_magnitude"] = np.abs(pc)
    pred_field = recon_real_from_spectrum(pb)
    RZ = None
    try:
        R, Zc = _flux_grid_for_bundle(b)
        if R.shape == gt_field.shape:
            RZ = (R, Zc)
    except Exception as exc:  # noqa: BLE001
        print(f"    [rz] flux grid unavailable ({exc}); using poloidal plane")
        RZ = None
    return gt_field, pred_field, RZ


def _panel(ax, data, *, RZ=None, extent=None, cmap="magma", vmin=None, vmax=None, title=""):
    if RZ is not None:
        im = ax.pcolormesh(RZ[0], RZ[1], data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal", adjustable="box")
    else:
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_case(i, z, out_path: Path):
    keys = z["keys"]; paths = z["paths"]; split = z["split"]
    gt = z["gt"].astype(np.float32); pred = z["pred"].astype(np.float32)
    m_grid = z["m_grid"]; psi_grid = z["psi_grid"]
    r2g = z["r2_global"]; r2p = z["r2_pattern"]
    target_space = str(z["target_space"])
    spec_field = str(z["spectrum_field"]) if "spectrum_field" in z.files else "p"

    ext = [0.0, 1.0, float(m_grid[0]), float(m_grid[-1])]
    g, p = gt[i], pred[i]
    d = p - g
    vlo, vhi = np.percentile(g, 2), g.max()
    dm = np.percentile(np.abs(d), 98) or 1.0

    fig, ax = plt.subplots(2, 3, figsize=(14, 8.6))
    _panel(ax[0, 0], g, extent=ext, vmin=vlo, vmax=vhi, title="spectrum: ground truth")
    _panel(ax[0, 1], p, extent=ext, vmin=vlo, vmax=vhi, title="spectrum: prediction")
    _panel(ax[0, 2], d, extent=ext, cmap="coolwarm", vmin=-dm, vmax=dm, title="spectrum: pred - GT")
    for a in ax[0]:
        a.set_xlabel(r"$\psi_N$"); a.set_ylabel("m")

    try:
        gf, pf, RZ = _recon_fields(paths[i], pred[i], spec_field, m_grid, psi_grid, target_space)
        fm = np.percentile(np.abs(gf), 99) or 1.0
        fext = None if RZ is not None else [0, 1, 0, gf.shape[0]]
        _panel(ax[1, 0], gf, RZ=RZ, extent=fext, cmap="RdBu_r", vmin=-fm, vmax=fm, title="field Re(δp): GT")
        _panel(ax[1, 1], pf, RZ=RZ, extent=fext, cmap="RdBu_r", vmin=-fm, vmax=fm, title="field Re(δp): pred")
        _panel(ax[1, 2], pf - gf, RZ=RZ, extent=fext, cmap="coolwarm", vmin=-fm, vmax=fm, title="field: pred - GT")
        if RZ is not None:
            for a in ax[1]:
                a.set_xlabel("R [m]"); a.set_ylabel("Z [m]")
    except Exception as exc:  # noqa: BLE001
        for a in ax[1]:
            a.text(0.5, 0.5, f"field recon unavailable:\n{exc}", ha="center", va="center",
                   transform=a.transAxes, fontsize=8)

    fig.suptitle(
        f"{keys[i]}  [{split[i]}]   global R²={r2g[i]:.3f}  pattern R²={r2p[i]:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=125)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _output_stem(run: Path, case_label: str, tag: str | None, model_label: str | None) -> str:
    """Build ``rz_case_<model>_<case>[_<tag>]`` without extension."""
    model = model_label or run.name
    parts = ["rz_case", model, case_label]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="docs/m3dc1/assets")
    ap.add_argument("--tag", default="maxnorm_log10",
                    help="optional suffix (e.g. sparc1530_diagnosis)")
    ap.add_argument("--model-label", default=None,
                    help="model id in output filename (default: run directory name)")
    ap.add_argument("--cases", type=int, nargs="*", default=None,
                    help="explicit cache indices instead of worst/median/best")
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = _load_cache(run)
    split = z["split"]; r2p = z["r2_pattern"]
    keys = z["keys"].astype(str)

    if args.cases:
        picks = [(str(keys[i]), i) for i in args.cases]
    else:
        idx = np.where(split == args.split)[0]
        rp = r2p[idx]
        fin = np.isfinite(rp)
        idx, rp = idx[fin], rp[fin]
        order = np.argsort(rp)
        picks = [
            ("worst", int(idx[order[0]])),
            ("median", int(idx[order[len(order) // 2]])),
            ("best", int(idx[order[-1]])),
        ]

    print(f"run={run.name} split={args.split} picks:",
          [(n, str(keys[i]), round(float(r2p[i]), 3)) for n, i in picks])
    for name, i in picks:
        out_path = out_dir / f"{_output_stem(run, name, args.tag, args.model_label)}.png"
        plot_case(i, z, out_path)


if __name__ == "__main__":
    main()
