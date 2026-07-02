#!/usr/bin/env python3
"""Head-to-head: plain-MSE baseline vs peak-weighted |delta p| spectrum surrogate.

Both runs export a ``predictions_cache*.npz`` (gt + pred in the max-normalized
log10 target space, plus per-case r2_pattern). This script aligns the two caches
on their common validation cases and answers the question global R2 *cannot*:
did up-weighting the high-amplitude ridge actually make the peak sharper and put
it in the right place?

Metrics per case (computed identically for both models):
  * pattern R2      -- as stored by the export (correlation of the shape)
  * SSIM            -- structural similarity on the log-space image
  * dpsi_peak       -- |psi_N(argmax pred) - psi_N(argmax gt)|  (peak-location err)
  * amp_at_truepeak -- predicted linear amplitude at the true peak (1.0 = perfect;
                       gt is max-normalized so the true peak is always 1.0)

Outputs a printed summary table and two figures:
  peakweight_metric_dists_<tag>.png       distributions (baseline vs peak)
  peakweight_spectra_<tag>.png            worst/median/best (GT | baseline | peak)

Usage:
  python scripts/m3dc1/internal/compare_peakweight.py \
    --baseline runs/spectrum_image_full_maxnorm_log10/predictions_cache.npz \
    --peak     runs/spectrum_image_full_maxnorm_log10_peak8_interactive/predictions_cache_val.npz \
    --out-dir  docs/m3dc1/assets --tag peak8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on two 2D images (gaussian-windowed)."""
    from scipy.ndimage import gaussian_filter

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    rng = hi - lo
    if rng <= 0:
        return 1.0
    a = (a - lo) / rng
    b = (b - lo) / rng
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = 1.5
    mu_a = gaussian_filter(a, s)
    mu_b = gaussian_filter(b, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = gaussian_filter(a * a, s) - mu_a2
    vb = gaussian_filter(b * b, s) - mu_b2
    vab = gaussian_filter(a * b, s) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * vab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (va + vb + C2)
    )
    return float(np.mean(ssim_map))


def _peak_psi(img_log: np.ndarray, psi_grid: np.ndarray) -> tuple:
    """Return (psi_N of the amplitude peak, m-index, psi-index).

    img_log is log10 of the max-normalized magnitude; exponentiate to get the
    linear amplitude, collapse over m (max), then argmax over psi_N.
    """
    lin = 10.0 ** img_log.astype(np.float64)
    marg = lin.max(axis=0)            # axis0 = m -> marginal over psi_N
    j = int(np.argmax(marg))
    return float(psi_grid[j]), j


def _amp_at(img_log_pred: np.ndarray, img_log_gt: np.ndarray) -> float:
    """Predicted linear amplitude at the GT peak pixel (gt peak == 1.0)."""
    lin_gt = 10.0 ** img_log_gt.astype(np.float64)
    i, j = np.unravel_index(int(np.argmax(lin_gt)), lin_gt.shape)
    return float(10.0 ** img_log_pred[i, j])


# --------------------------------------------------------------------------- #
# load + align
# --------------------------------------------------------------------------- #
def _load(path: str):
    z = np.load(path, allow_pickle=True)
    keys = z["keys"].astype(str)
    split = z["split"].astype(str)
    val = split == "val"
    return {
        "keys": keys[val],
        "gt": z["gt"][val].astype(np.float32),
        "pred": z["pred"][val].astype(np.float32),
        "r2p": z["r2_pattern"][val].astype(np.float32),
        "r2g": z["r2_global"][val].astype(np.float32),
        "psi_grid": z["psi_grid"].astype(np.float32),
        "m_grid": z["m_grid"].astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--peak", required=True)
    ap.add_argument("--out-dir", default="docs/m3dc1/assets")
    ap.add_argument("--tag", default="peak8")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    B = _load(args.baseline)
    P = _load(args.peak)
    psi_grid = P["psi_grid"]
    m_grid = P["m_grid"]

    # align on common val keys
    bk = {k: i for i, k in enumerate(B["keys"])}
    pk = {k: i for i, k in enumerate(P["keys"])}
    common = [k for k in P["keys"] if k in bk]
    print(f"baseline val={len(B['keys'])}  peak val={len(P['keys'])}  "
          f"common={len(common)}")

    rows = []
    for k in common:
        bi, pi = bk[k], pk[k]
        gt = B["gt"][bi]                      # gt identical across models
        pb = B["pred"][bi]
        pp = P["pred"][pi]
        psi_gt, _ = _peak_psi(gt, psi_grid)
        psi_b, _ = _peak_psi(pb, psi_grid)
        psi_p, _ = _peak_psi(pp, psi_grid)
        rows.append(dict(
            key=k,
            r2p_b=float(B["r2p"][bi]), r2p_p=float(P["r2p"][pi]),
            ssim_b=_ssim(gt, pb), ssim_p=_ssim(gt, pp),
            dpsi_b=abs(psi_b - psi_gt), dpsi_p=abs(psi_p - psi_gt),
            amp_b=_amp_at(pb, gt), amp_p=_amp_at(pp, gt),
        ))

    def col(name):
        return np.array([r[name] for r in rows], float)

    def med(name):
        return float(np.nanmedian(col(name)))

    print("\n================  BASELINE (plain MSE)  vs  PEAK-WEIGHTED  ================")
    print(f"{'metric':22s}{'baseline':>12s}{'peak-weighted':>16s}{'better?':>12s}")
    def line(label, b, p, higher_better):
        win = ("peak" if (p > b) == higher_better else "baseline")
        if abs(p - b) < 1e-6:
            win = "tie"
        print(f"{label:22s}{b:>12.4f}{p:>16.4f}{win:>12s}")
    line("pattern R2 (median)", med("r2p_b"), med("r2p_p"), True)
    line("SSIM (median)", med("ssim_b"), med("ssim_p"), True)
    line("dpsi_peak (median)", med("dpsi_b"), med("dpsi_p"), False)
    line("amp@truepeak (median)", med("amp_b"), med("amp_p"), True)  # want ->1.0
    # fraction of cases with peak within 0.05 in psi_N
    fb = float(np.mean(col("dpsi_b") < 0.05))
    fp = float(np.mean(col("dpsi_p") < 0.05))
    line("frac |dpsi|<0.05", fb, fp, True)
    print("(amp@truepeak: closer to 1.0 = peak amplitude better reproduced; "
          "gt peak == 1.0)")

    # -------------------------------------------------------------------- #
    # figure 1: metric distributions
    # -------------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    specs = [
        ("dpsi", "peak-location error  |Δψ_N|", False),
        ("r2p", "pattern R²", True),
        ("ssim", "SSIM", True),
        ("amp", "amp @ true peak (→1)", True),
    ]
    for a, (base, title, _hb) in zip(ax, specs):
        b = col(f"{base}_b"); p = col(f"{base}_p")
        lo = float(np.nanmin([b.min(), p.min()]))
        hi = float(np.nanmax([b.max(), p.max()]))
        bins = np.linspace(lo, hi, 40)
        a.hist(b, bins=bins, alpha=0.55, label=f"baseline (med {np.nanmedian(b):.3f})",
               color="#888888")
        a.hist(p, bins=bins, alpha=0.55, label=f"peak8 (med {np.nanmedian(p):.3f})",
               color="#1f77b4")
        a.set_title(title, fontsize=11)
        a.legend(fontsize=8)
    fig.suptitle("Validation-set per-case metrics: plain MSE vs peak-weighted", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = out_dir / f"peakweight_metric_dists_{args.tag}.png"
    fig.savefig(f1, dpi=125)
    plt.close(fig)
    print(f"\nwrote {f1}")

    # -------------------------------------------------------------------- #
    # figure 2: spectra GT | baseline | peak  for worst/median/best (by peak r2p)
    # -------------------------------------------------------------------- #
    r2p_p = col("r2p_p")
    order = np.argsort(r2p_p)
    picks = [("worst", order[0]), ("median", order[len(order) // 2]),
             ("best", order[-1])]
    ext = [0.0, 1.0, float(m_grid[0]), float(m_grid[-1])]
    fig, ax = plt.subplots(3, 3, figsize=(14, 12))
    for r, (name, oi) in enumerate(picks):
        k = rows[oi]["key"]
        bi, pi = bk[k], pk[k]
        gt = B["gt"][bi]; pb = B["pred"][bi]; pp = P["pred"][pi]
        vlo, vhi = np.percentile(gt, 2), gt.max()
        for c, (img, ttl) in enumerate([
            (gt, "ground truth"),
            (pb, f"baseline  r2p={rows[oi]['r2p_b']:.2f} SSIM={rows[oi]['ssim_b']:.2f}"),
            (pp, f"peak8  r2p={rows[oi]['r2p_p']:.2f} SSIM={rows[oi]['ssim_p']:.2f}"),
        ]):
            im = ax[r, c].imshow(img, origin="lower", aspect="auto", extent=ext,
                                 cmap="magma", vmin=vlo, vmax=vhi)
            ax[r, c].set_title(f"{name}: {ttl}", fontsize=9)
            ax[r, c].set_xlabel(r"$\psi_N$"); ax[r, c].set_ylabel("m")
            plt.colorbar(im, ax=ax[r, c], fraction=0.046, pad=0.04)
    fig.suptitle("Spectrum |δp̂|(m, ψ_N): worst / median / best (by peak8 pattern R²)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f2 = out_dir / f"peakweight_spectra_{args.tag}.png"
    fig.savefig(f2, dpi=125)
    plt.close(fig)
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
