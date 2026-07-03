#!/usr/bin/env python
"""Metric reality-check for a spectrum-image prediction cache.

Top: histogram of per-case metric. Bottom: gallery (worst→best percentiles) with
**6 panels per row** — spectrum GT/pred/diff and RZ field GT/pred/diff side by side.

Usage:
    python scripts/m3dc1/internal/metric_gallery.py \
        --cache runs/spectrum_fno48_floor6_smooth1_qc/predictions_cache.npz \
        --split test --metric r2_pattern \
        --out docs/m3dc1/assets/metric_reality_check_qc_combined.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path("/global/homes/a/asvillar/src/SURGE")
if str(REPO / "m3dc1ml" / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "m3dc1ml" / "src"))


def _recon_gt_pred_fields(path, pred_i, m_grid, psi_grid, space, spec_field):
    """Reconstruct GT and predicted Re(δp)(R,Z), each max-normalized to unit amplitude."""
    from m3dc1ml.io.sdata import load_complex_v2_case
    from m3dc1ml.viz.explore_case import recon_real_from_spectrum, _flux_grid_for_bundle
    from scipy.interpolate import RegularGridInterpolator

    b = load_complex_v2_case(str(path), spectrum_field=spec_field)
    gt_field = recon_real_from_spectrum(b)
    RZ = None
    try:
        R, Zc = _flux_grid_for_bundle(b)
        if R.shape == gt_field.shape:
            RZ = (R, Zc)
    except Exception:
        RZ = None

    mag_u = 10.0 ** pred_i.astype(np.float64) if space == "log10" else \
        np.clip(pred_i.astype(np.float64), 0, None)
    interp = RegularGridInterpolator((m_grid, psi_grid), mag_u,
                                     bounds_error=False, fill_value=0.0)
    m_nat = np.asarray(b["m_modes"], float)
    psi_nat = np.asarray(b["psi_norm"], float)
    MM, PP = np.meshgrid(m_nat, psi_nat, indexing="ij")
    mag_nat = interp(np.stack([MM.ravel(), PP.ravel()], 1)).reshape(len(m_nat), len(psi_nat))
    spec = np.asarray(b["spec_complex"])
    pb = dict(b)
    pb["spec_complex"] = mag_nat * np.exp(1j * np.angle(spec))
    pb["spec_magnitude"] = np.abs(pb["spec_complex"])
    pred_field = recon_real_from_spectrum(pb)

    gt_field = gt_field / (np.abs(gt_field).max() + 1e-30)
    pred_field = pred_field / (np.abs(pred_field).max() + 1e-30)
    return gt_field, pred_field, RZ


def _plot_spectrum(axs, gt_i, pred_i, ext, space, titles=False):
    g, p = gt_i, pred_i
    d = p - g
    vlo, vhi = float(np.percentile(g, 2)), float(g.max())
    dm = float(np.percentile(np.abs(d), 98)) or 1.0
    im0 = axs[0].imshow(g, origin="lower", aspect="auto", extent=ext,
                        cmap="magma", vmin=vlo, vmax=vhi)
    axs[1].imshow(p, origin="lower", aspect="auto", extent=ext,
                  cmap="magma", vmin=vlo, vmax=vhi)
    im2 = axs[2].imshow(d, origin="lower", aspect="auto", extent=ext,
                        cmap="coolwarm", vmin=-dm, vmax=dm)
    if titles:
        axs[0].set_title(f"spec GT ({space})", fontsize=9)
        axs[1].set_title("spec pred", fontsize=9)
        axs[2].set_title("spec Δ", fontsize=9)
    for a in axs:
        a.set_xlabel(r"$\psi_N$", fontsize=7)
        a.tick_params(labelsize=6)
    return im0, im2


def _plot_field(axs, gt_f, pred_f, RZ, titles=False):
    d = pred_f - gt_f
    rl2 = float(np.linalg.norm(d) / (np.linalg.norm(gt_f) + 1e-30))
    if RZ is not None:
        axs[0].pcolormesh(RZ[0], RZ[1], gt_f, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axs[1].pcolormesh(RZ[0], RZ[1], pred_f, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        im2 = axs[2].pcolormesh(RZ[0], RZ[1], d, shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        for a in axs:
            a.set_aspect("equal", adjustable="box")
            a.set_xlabel("R", fontsize=7)
    else:
        axs[0].imshow(gt_f, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        axs[1].imshow(pred_f, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        im2 = axs[2].imshow(d, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    if titles:
        axs[0].set_title("field GT", fontsize=9)
        axs[1].set_title("field pred", fontsize=9)
        axs[2].set_title(f"field Δ  relL2={rl2:.2f}", fontsize=9)
    for a in axs:
        a.tick_params(labelsize=6)
    return im2, rl2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    ap.add_argument("--metric", default="r2_pattern",
                    choices=["r2_pattern", "r2_global"])
    ap.add_argument("--n-gallery", type=int, default=6)
    ap.add_argument("--field", action="store_true",
                    help="field only (legacy); default is spectrum+field combined")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    gt = z["gt"].astype(np.float32)
    pred = z["pred"].astype(np.float32)
    split = z["split"].astype(str)
    keys = z["keys"].astype(str)
    metric = z[args.metric].astype(np.float32)
    m_grid, psi_grid = z["m_grid"], z["psi_grid"]
    space = str(z["target_space"])
    ext = [float(psi_grid[0]), float(psi_grid[-1]), float(m_grid[0]), float(m_grid[-1])]
    spec_field = str(z["spectrum_field"]) if "spectrum_field" in z.files else "p"
    paths = z["paths"].astype(str) if "paths" in z.files else None

    mask = np.ones(len(split), bool) if args.split == "all" else (split == args.split)
    idx = np.where(mask)[0]
    mv = metric[idx]
    order = idx[np.argsort(mv)]
    n = len(idx)
    label = args.metric.replace("r2_", "").replace("_", " ") + " R²"

    thr = [0.0, 0.5, 0.7, 0.8, 0.9]
    frac = {t: float((mv < t).mean()) for t in thr}

    pcts = np.linspace(2, 98, args.n_gallery)
    gal_rows = [int(np.clip(round(p / 100 * (n - 1)), 0, n - 1)) for p in pcts]
    gal_idx = [int(order[r]) for r in gal_rows]

    combined = not args.field
    ncols = 6 if combined else 3
    ng = len(gal_idx)
    col_w = 2.05 if combined else 2.8
    row_h = 2.8 if combined else 3.2
    fig = plt.figure(figsize=(col_w * ncols + 0.4, 2.4 + row_h * ng))
    gs = GridSpec(ng + 1, ncols, figure=fig,
                  height_ratios=[1.2] + [1] * ng,
                  hspace=0.32, wspace=0.12)

    axh = fig.add_subplot(gs[0, :])
    axh.hist(mv, bins=50, color="#4c72b0", edgecolor="k", linewidth=0.3, alpha=0.85)
    med = float(np.median(mv))
    axh.axvline(med, color="#c44e52", lw=2, label=f"median = {med:.3f}")
    ctext = "   ".join([f"<{t:g}: {100*frac[t]:.1f}%" for t in thr])
    axh.set_title(f"{Path(args.cache).parent.name}  ·  {args.split}  ·  "
                  f"{n} cases  ·  mean={mv.mean():.3f}  ·  {ctext}", fontsize=10)
    axh.set_xlabel(f"per-case {label}"); axh.set_ylabel("count")
    axh.legend(fontsize=8, loc="upper left")

    for r, ci in enumerate(gal_idx, start=1):
        tag = f"p{int(round(pcts[r-1]))}"
        row_label = (f"{tag}  {keys[ci]}\n{label}={metric[ci]:.3f}")

        if combined:
            sax = [fig.add_subplot(gs[r, c]) for c in range(3)]
            fax = [fig.add_subplot(gs[r, c]) for c in range(3, 6)]
            _plot_spectrum(sax, gt[ci], pred[ci], ext, space, titles=(r == 1))
            sax[0].set_ylabel(row_label, fontsize=7)
            try:
                gf, pf, RZ = _recon_gt_pred_fields(
                    paths[ci], pred[ci], m_grid, psi_grid, space, spec_field)
                _, rl2 = _plot_field(fax, gf, pf, RZ, titles=(r == 1))
                sax[0].set_ylabel(row_label + f"\nrelL2={rl2:.2f}", fontsize=7)
            except Exception as exc:  # noqa: BLE001
                for a in fax:
                    a.text(0.5, 0.5, str(exc)[:60], ha="center", va="center",
                           transform=a.transAxes, fontsize=6)
                print(f"  recon failed {keys[ci]}: {exc}")
        elif args.field:
            axs = [fig.add_subplot(gs[r, c]) for c in range(3)]
            try:
                gf, pf, RZ = _recon_gt_pred_fields(
                    paths[ci], pred[ci], m_grid, psi_grid, space, spec_field)
                _plot_field(axs, gf, pf, RZ, titles=(r == 1))
                axs[0].set_ylabel(row_label, fontsize=7)
            except Exception as exc:  # noqa: BLE001
                for a in axs:
                    a.text(0.5, 0.5, str(exc)[:60], ha="center", va="center",
                           transform=a.transAxes, fontsize=6)
        else:
            axs = [fig.add_subplot(gs[r, c]) for c in range(3)]
            _plot_spectrum(axs, gt[ci], pred[ci], ext, space, titles=(r == 1))
            axs[0].set_ylabel(row_label, fontsize=7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = ("spectrum + RZ field" if combined else
             ("RZ field" if args.field else "spectrum"))
    fig.suptitle(f"Does the metric match the picture? ({title})  "
                 f"worst→best across {args.split} split",
                 fontsize=12, y=1.002)
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.15)
    print(f"Wrote {out}")
    print(f"{args.split}: n={n}  median={med:.3f}  mean={mv.mean():.3f}")
    for t in thr:
        print(f"  {label} < {t:g}: {100*frac[t]:.1f}%  ({int((mv<t).sum())} cases)")


if __name__ == "__main__":
    main()
