#!/usr/bin/env python
"""Whole-spectrum (2D image) surrogate for M3DC1 |delta p hat|(m, psi_N).

Motivation
----------
The per-mode MLP approach tops out at ~0.36 test R2 because it treats each
(case, m) as an independent row and cannot see the coherent m-psi *ridge*
(m ~ n q(psi)) that dominates the spectrum. Here we instead predict the WHOLE
spectrum as a 2D image per case, conditioned on the equilibrium encoded as input
channels on the (m, psi_N) grid -- including the physics channel m - n q(psi).

Target: log10|delta p hat|(m, psi_N), phase-invariant magnitude. We report both
the global test R2 and the per-image de-meaned ("pattern") R2, which isolates how
well the spatial ridge structure is captured from the (unpredictable, arbitrary)
per-case overall amplitude offset.

Optional field-loss terms (--field-loss-weight, default 0) add a differentiable
IFFT-proxy relL2 on the training grid with **oracle true phase** from the HDF5
complex spectrum. This optimizes field quality given known phase — the same
post-hoc idealization as field_recon_compare.py. With all new flags at defaults,
behavior is unchanged from the magnitude-only recipe.

Architectures: SURGE backends pytorch.fno2d and pytorch.unet (conditioning as
input channels). Case-grouped split is trivial here (one image per case).

Usage:
  python scripts/m3dc1/internal/train_spectrum_image.py \
      --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
      --filename csdata_deltap_b_ver.h5 --n-cases 2500 --grid 128 \
      --models fno2d unet --epochs 80 --out runs/spectrum_image
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[2]  # .../SURGE/scripts
if str(_SCRIPTS / "m3dc1") not in sys.path:
    sys.path.insert(0, str(_SCRIPTS / "m3dc1"))

import h5py  # noqa: E402
from dataset_complex_v2 import find_complex_v2_files, _decode  # noqa: E402


def _read_case(path: Path, spectrum_field: str) -> Optional[Dict]:
    """Read one case: complex spectrum magnitude, axes, q/p profiles, shaping, n."""
    try:
        with h5py.File(path, "r") as f:
            if "runs" not in f:
                return None
            rname = list(f["runs"].keys())[0]
            rg = f["runs"][rname]
            if "spectrum" not in rg or spectrum_field not in rg["spectrum"]:
                return None
            sp = rg["spectrum"][spectrum_field]
            spec = np.asarray(sp["spec"])
            if spec.ndim == 3:
                spec = spec[-1]
            mag = np.abs(spec) if np.iscomplexobj(spec) else np.abs(np.asarray(spec, float))
            m_modes = np.asarray(sp["m_modes"]).astype(float).ravel()
            psi = (np.asarray(sp["psi_norm"], float).ravel()
                   if "psi_norm" in sp else np.linspace(1e-4, 1.0, mag.shape[1]))
            out: Dict = {"run_id": _decode(rg.get("runID", rname)),
                         "eq_id": _decode(rg.get("eqID", "eq")),
                         "mag": mag.astype(np.float64), "m_modes": m_modes, "psi": psi}
            if np.iscomplexobj(spec):
                out["phase"] = np.angle(spec).astype(np.float64)
                out["real"] = np.real(spec).astype(np.float64)
                out["imag"] = np.imag(spec).astype(np.float64)
            else:
                z = np.zeros_like(mag, dtype=np.float64)
                out["phase"] = z
                out["real"] = np.asarray(spec, float) if not np.iscomplexobj(spec) else z
                out["imag"] = z.copy()
            # shaping
            sh = {}
            if "miller" in rg:
                for k in ("R0", "a", "kappa", "delta"):
                    if k in rg["miller"]:
                        sh[k] = float(rg["miller"][k][()])
            if "R0" in sh and "a" in sh and sh["R0"]:
                sh["epsilon"] = sh["a"] / sh["R0"]
            n_val, pscale, bscale = 0.0, 1.0, 1.0
            if "parset" in rg:
                names = rg["parset"]["names"]; vals = rg["parset"]["values"]
                for i, nm in enumerate(names):
                    nm = _decode(nm)
                    if i < len(vals):
                        if nm == "ntor":
                            n_val = float(vals[i])
                        elif nm == "pscale":
                            pscale = float(vals[i])
                        elif nm == "batemanscale":
                            bscale = float(vals[i])
            sh["pscale"] = pscale; sh["batemanscale"] = bscale
            out["n"] = n_val; out["shaping"] = sh
            # q, p profiles
            qprof = qpsin = pprof = ppsin = None
            if "flux_average" in rg:
                fa = rg["flux_average"]
                if "q" in fa and "profile" in fa["q"]:
                    qprof = np.asarray(fa["q"]["profile"], float).ravel()
                    qpsin = np.asarray(fa["q"]["psin"], float).ravel() if "psin" in fa["q"] else None
                if "p" in fa and "profile" in fa["p"]:
                    pprof = np.asarray(fa["p"]["profile"], float).ravel()
                    ppsin = np.asarray(fa["p"]["psin"], float).ravel() if "psin" in fa["p"] else None
            out["qprof"], out["qpsin"] = qprof, qpsin
            out["pprof"], out["ppsin"] = pprof, ppsin
            # equilibrium geometry (RZ grid) for flux-expansion / LCFS channels
            if "equilibrium" in rg:
                eq = rg["equilibrium"]
                try:
                    out["equilibrium"] = {
                        "psi": np.asarray(eq["psi"], float),
                        "grad_psi_mag": np.asarray(eq["grad_psi_mag"], float),
                        "psi_lcfs": float(eq["psi_lcfs"][()]),
                    }
                except Exception:
                    out["equilibrium"] = None
            else:
                out["equilibrium"] = None
            return out
    except Exception:
        return None


def _interp_to(grid: np.ndarray, x: Optional[np.ndarray], y: Optional[np.ndarray],
               fill: float = 0.0) -> np.ndarray:
    if y is None or y.size == 0:
        return np.full_like(grid, fill)
    if x is None or x.size != y.size:
        x = np.linspace(0.0, 1.0, y.size)
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order])


def _geometry_profiles(c: Dict, psi_grid: np.ndarray) -> Dict[str, np.ndarray]:
    """1-D geometry profiles on psi_grid: shear, flux expansion, LCFS proximity.

    - shear ~ (psi_N / q) dq/dpsi_N  (magnetic shear proxy)
    - flux_exp ~ <|grad psi|>(psi_N) from equilibrium RZ grid (edge compression)
    - lcfs_prox = 1 - psi_N  (1 at axis, 0 at LCFS)
    """
    q_on = _interp_to(psi_grid, c.get("qpsin"), c.get("qprof"), fill=1.0)
    dq = np.gradient(q_on, psi_grid)
    shear = psi_grid * dq / np.maximum(np.abs(q_on), 1e-8)
    lcfs_prox = 1.0 - psi_grid

    flux_exp = np.ones_like(psi_grid)
    eq = c.get("equilibrium")
    if eq is not None:
        psi_f = eq["psi"]
        grad = eq["grad_psi_mag"]
        psi_lcfs = eq["psi_lcfs"]
        psi_axis = float(np.min(psi_f))
        denom = psi_lcfs - psi_axis
        if abs(denom) > 1e-12:
            psin_f = np.clip((psi_f - psi_axis) / denom, 0.0, 1.05)
            tol = max(0.5 / len(psi_grid), 0.005)
            for j, pg in enumerate(psi_grid):
                m = (psin_f >= pg - tol) & (psin_f <= pg + tol)
                flux_exp[j] = float(grad[m].mean()) if m.any() else np.nan
            # fill empty bins by linear interp
            ok = np.isfinite(flux_exp)
            if ok.any():
                flux_exp = np.interp(psi_grid, psi_grid[ok], flux_exp[ok])
            # normalize to [0,1] per case (scale is arbitrary; shape matters)
            fmax = float(flux_exp.max()) or 1.0
            flux_exp = flux_exp / fmax

    return {"shear": shear.astype(np.float32),
            "flux_exp": flux_exp.astype(np.float32),
            "lcfs_prox": lcfs_prox.astype(np.float32)}


def _interp_field_to_grid(
    field_2d: np.ndarray,
    m_modes: np.ndarray,
    psi: np.ndarray,
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
    grid: int,
) -> np.ndarray:
    """Interpolate native (m, psi) spectrum slice onto uniform training grid."""
    m_vals = m_modes
    tmp = np.vstack([_interp_to(psi_grid, psi, row) for row in field_2d])
    return np.vstack([_interp_to(m_grid, m_vals, tmp[:, j]) for j in range(grid)]).T


def build_dataset(
    batch_dir: str, filename: str, n_cases: Optional[int], grid: int,
    m_lo: float, m_hi: float, spectrum_field: str, eps: float,
    shaping_keys: Tuple[str, ...] = ("kappa", "delta", "epsilon", "pscale", "batemanscale"),
    target_norm: str = "none", target_space: str = "log10",
    target_floor: Optional[float] = None,
    target_smooth: Optional[float] = None,
    target_kind: str = "magnitude",
    exclude_keys: Optional[set] = None,
    geom_channels: bool = False,
    return_paths: bool = False,
    return_mag_grid: bool = False,
):
    """Return X (N,C,H,W), Y (N,H,W) target, channel names, case keys.

    target_kind : {"magnitude", "phase", "real", "imag"}
        magnitude — log10|δp̂| (default surrogate target)
        phase     — angle(δp̂) in radians on the training grid
        real/imag — Re/Im(δp̂) with optional per-case max scale from |δp̂| peak
    return_mag_grid : when target_kind=phase, also return Y_mag (N,H,W) max-normalized
        log10|δp̂| grids for ridge-weighting and honest field-loss magnitude arm.
    """
    paths = find_complex_v2_files(batch_dir, filename=filename)
    if n_cases:
        paths = paths[:n_cases]
    print(f"Building spectrum-image dataset from {len(paths)} cases "
          f"(grid={grid}x{grid}, m in [{m_lo},{m_hi}], target={target_kind})")
    psi_grid = np.linspace(0.0, 1.0, grid)
    m_grid = np.linspace(m_lo, m_hi, grid)
    M = np.repeat(m_grid[:, None], grid, axis=1)          # (H,W) m varies along rows
    PSI = np.repeat(psi_grid[None, :], grid, axis=0)      # (H,W) psi varies along cols
    chan_names = ["psi", "m", "q", "p", "res(m-nq)", "prox", *shaping_keys]
    if geom_channels:
        chan_names.extend(["shear", "flux_exp", "lcfs_prox"])
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    Ymags: List[np.ndarray] = []
    keys: List[str] = []
    kept: List[str] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        c = _read_case(Path(p), spectrum_field)
        if c is None:
            continue
        if exclude_keys and f"{c['run_id']}_{c['eq_id']}" in exclude_keys:
            continue                                       # quarantined bad-data case
        mag, m_modes, psi = c["mag"], c["m_modes"], c["psi"]
        sel = (m_modes >= m_lo) & (m_modes <= m_hi)
        if sel.sum() < 4:
            continue
        mag_sel = mag[sel, :]
        cmax = float(mag_sel.max()) if mag_sel.size else 0.0
        scale = cmax if (target_norm == "max" and cmax > 0) else 1.0

        if target_kind == "magnitude":
            field = mag_sel.copy()
            if target_norm == "max" and cmax > 0:
                field = field / cmax
            if target_space == "log10":
                field = np.log10(field + eps)
                if target_floor is not None and target_floor > 0:
                    field = np.maximum(field, -float(target_floor))
        elif target_kind == "phase":
            field = c["phase"][sel, :]
        elif target_kind == "real":
            field = c["real"][sel, :] / scale
        elif target_kind == "imag":
            field = c["imag"][sel, :] / scale
        else:
            raise ValueError(f"unknown target_kind={target_kind!r}")

        m_vals = m_modes[sel]
        img = _interp_field_to_grid(field, m_vals, psi, m_grid, psi_grid, grid)
        if target_kind == "magnitude" and target_smooth is not None and target_smooth > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=float(target_smooth))

        mag_img = None
        if return_mag_grid or target_kind == "phase":
            mag_norm = mag_sel / scale if scale > 0 else mag_sel
            mag_log = np.log10(mag_norm + eps)
            if target_floor is not None and target_floor > 0:
                mag_log = np.maximum(mag_log, -float(target_floor))
            mag_img = _interp_field_to_grid(mag_log, m_vals, psi, m_grid, psi_grid, grid)
            if target_smooth is not None and target_smooth > 0:
                from scipy.ndimage import gaussian_filter
                mag_img = gaussian_filter(mag_img, sigma=float(target_smooth))
        q_on = _interp_to(psi_grid, c["qpsin"], c["qprof"])
        p_on = _interp_to(psi_grid, c["ppsin"], c["pprof"])
        Q = np.repeat(q_on[None, :], grid, axis=0)        # (H,W) q(psi_j)
        P = np.repeat(p_on[None, :], grid, axis=0)
        n_val = c["n"]
        RES = M - n_val * Q                                # resonance detuning
        PROX = 1.0 / (1.0 + RES ** 2)                      # ridge proximity
        sh = c["shaping"]
        const = [np.full((grid, grid), float(sh.get(k, 0.0))) for k in shaping_keys]
        layers = [PSI, M, Q, P, RES, PROX, *const]
        if geom_channels:
            geo = _geometry_profiles(c, psi_grid)
            for gname in ("shear", "flux_exp", "lcfs_prox"):
                g1 = geo[gname]
                layers.append(np.repeat(g1[None, :], grid, axis=0))  # (H,W) psi-only
        X = np.stack(layers, axis=0).astype(np.float32)  # (C,H,W)
        Xs.append(X); Ys.append(img.astype(np.float32)); keys.append(f"{c['run_id']}_{c['eq_id']}")
        if mag_img is not None:
            Ymags.append(mag_img.astype(np.float32))
        kept.append(str(p))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    X = np.stack(Xs); Y = np.stack(Ys)
    print(f"  Built X={X.shape} Y={Y.shape} in {time.time()-t0:.0f}s")
    if return_mag_grid or target_kind == "phase":
        Y_mag = np.stack(Ymags)
        if return_paths:
            return X, Y, chan_names, keys, kept, Y_mag
        return X, Y, chan_names, keys, Y_mag
    if return_paths:
        return X, Y, chan_names, keys, kept
    return X, Y, chan_names, keys


def _psi_balance_weights(Y: np.ndarray, n_bins: int = 10,
                         strength: float = 1.0) -> np.ndarray:
    """Per-sample weights that cross-balance the peak-psi_N distribution.

    The dataset is dominated by edge/pedestal modes (peak at high psi_N); core-
    localized modes (low psi_N) are rare, so a plain-MSE model under-fits them.
    We bin each training image by the psi_N of its global-max pixel, then weight
    every sample by (1/bin_count)**strength (normalized to mean 1). Feeding these
    to a WeightedRandomSampler oversamples the rare core bins so each ridge
    location is seen ~equally -- balancing classes WITHOUT changing sampling of
    the underlying files. strength=1 => full inverse-frequency; 0 => uniform.
    """
    N, H, W = Y.shape
    col = np.argmax(Y.reshape(N, -1), axis=1) % W        # peak psi_N column
    psi_peak = col / max(W - 1, 1)                        # 0..1
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(psi_peak, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(b, minlength=n_bins).astype(float)
    inv = np.where(counts > 0, 1.0 / counts, 0.0) ** float(strength)
    w = inv[b]
    w *= len(w) / w.sum()                                 # mean weight -> 1
    return w.astype(np.float32), counts


def load_mag_pred_dex_from_run(
    mag_run: Path,
    keys: Sequence[str],
    X: np.ndarray,
    tr: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 16,
    model: str = "fno2d",
) -> np.ndarray:
    """Predict log10|δp̂| dex grids from a trained magnitude run (frozen), aligned to keys."""
    import torch

    mag_run = Path(mag_run)
    cfg = json.loads((mag_run / "run_config.json").read_text())
    ckpt = mag_run / f"ckpt_{model}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"magnitude checkpoint not found: {ckpt}")

    cache = mag_run / "predictions_cache.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        cache_keys = z["keys"].astype(str)
        pred = z["pred"].astype(np.float32)
        key_to_pred = {k: pred[i] for i, k in enumerate(cache_keys)}
        missing = [k for k in keys if k not in key_to_pred]
        if not missing:
            print(f"  [mag-run] loaded frozen |δp̂| preds from {cache.name} ({len(keys)} cases)")
            return np.stack([key_to_pred[k] for k in keys])
        print(f"  [mag-run] cache missing {len(missing)} keys — running inference")

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    net = _build_net(
        model,
        X.shape[1],
        fno_modes=int(cfg.get("fno_modes", 16)),
        fno_hidden=int(cfg.get("fno_hidden", 32)),
        grid=int(cfg.get("grid", X.shape[-1])),
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["state_dict"])
    net = net.to(dev).eval()

    xm = X[tr].mean((0, 2, 3), keepdims=True)
    xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs

    ym_path = mag_run / "norm_stats.json"
    if ym_path.is_file():
        ns = json.loads(ym_path.read_text())
        ym, ysd = float(ns["y_mean"]), float(ns["y_std"])
    else:
        print("  [mag-run] norm_stats.json missing — using train-split Y stats from magnitude rebuild")
        _, Y_mag, _, _ = build_dataset(
            cfg["batch_dir"], cfg["filename"], (cfg.get("n_cases") or None),
            cfg["grid"], float(cfg["m_window"][0]), float(cfg["m_window"][1]),
            cfg.get("spectrum_field", "p"), 1e-12,
            target_norm=cfg.get("target_norm", "none"),
            target_space=cfg.get("target_space", "log10"),
            target_floor=cfg.get("target_floor"),
            target_smooth=cfg.get("target_smooth"),
            target_kind="magnitude",
            exclude_keys=None,
            geom_channels=cfg.get("geom_channels", False),
        )
        ym = float(Y_mag[tr].mean())
        ysd = float(Y_mag[tr].std() + 1e-8)

    pred_std = _predict_net(net, Xn, batch_size)
    print(f"  [mag-run] inference from {ckpt.name} on {len(X)} cases ({dev})")
    return (pred_std * ysd + ym).astype(np.float32)


def _build_net(name: str, in_channels: int, fno_modes: int = 16,
               fno_hidden: int = 32, grid: int = 128):
    """Build a raw torch net from the SURGE backend modules (own training loop).

    fno_modes / fno_hidden control the FNO spectral-truncation width and channel
    width. On a 128 grid the FFT has ~64 modes/axis; n_modes=16 keeps only ~25%
    of the band (blurs sharp peaks), while 48 keeps ~75% (resolves the ridge).
    """
    if name == "fno2d":
        from surge.model.backends.fno2d import _FNO2dNet
        return _FNO2dNet(in_channels, 1, hidden_channels=fno_hidden,
                         n_modes=fno_modes, n_layers=4)
    if name == "unet":
        from surge.model.backends.unet import _UNetNet
        return _UNetNet(in_channels, 1, base_channels=48, depth=4)
    if name == "deeponet":
        import torch
        import torch.nn as nn
        from surge.model.backends.deeponet import DeepONet

        class _DeepONetNet(nn.Module):
            """Image-in/image-out DeepONet for the (m, psi_N) spectrum.

            branch: per-case conditioning read off the input channels sampled
            along psi_N (q/p profiles, shaping scalars, resonance at m_lo) ->
            latent. trunk: the 2-D query coordinate (m, psi_N) -> latent. The
            spectrum value at each grid point is their dot product. Plugs into
            the same loop as FNO/U-Net: (B,C,H,W) -> (B,1,H,W).
            """

            def __init__(self, in_ch: int, g: int):
                super().__init__()
                self.g = g
                self.net = DeepONet(
                    n_sensors=in_ch * g, n_query=g * g, n_basis=128,
                    branch_width=256, trunk_width=128, n_hidden=4, coord_dim=2)
                m = torch.linspace(-1.0, 1.0, g)
                p = torch.linspace(0.0, 1.0, g)
                MM, PP = torch.meshgrid(m, p, indexing="ij")     # (H=m, W=psi)
                self.register_buffer(
                    "coords", torch.stack([MM.reshape(-1), PP.reshape(-1)], 1))

            def forward(self, x):
                B, C, H, W = x.shape
                # conditioning is ~constant along m; sample channels at m_lo row
                u = x[:, :, 0, :].reshape(B, C * W)
                return self.net(u, self.coords).view(B, 1, H, W)

        return _DeepONetNet(in_channels, grid)
    return None


def _loss_plot(hist_path: Path, name: str, out: Path) -> None:
    """(Re)generate a train/val loss-curve PNG from the live history JSONL."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
    # Drop non-epoch marker rows (e.g. the {"early_stop": true} sentinel) that
    # lack the per-epoch loss keys.
    rows = [r for r in rows if "train_loss" in r and "val_loss" in r]
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [r["train_loss"] for r in rows], label="train")
    a1.plot(ep, [r["val_loss"] for r in rows], label="val")
    best = min(rows, key=lambda r: r["val_loss"])
    a1.axvline(best["epoch"], color="k", ls=":", lw=1, label=f"best ep {best['epoch']}")
    a1.set_xlabel("epoch"); a1.set_ylabel("MSE loss"); a1.set_yscale("log")
    a1.set_title(f"{name}: loss"); a1.legend()
    a2.plot(ep, [r["val_r2"] for r in rows], color="C2", label="val R2")
    a2.axhline(0.358, color="r", ls="--", lw=1, label="per-mode 0.358")
    a2.set_xlabel("epoch"); a2.set_ylabel("val R2"); a2.set_title(f"{name}: val R2")
    a2.legend()
    fig.tight_layout(); fig.savefig(out / f"loss_{name}.png", dpi=110); plt.close(fig)


def _train_net(net, name, out: Path, Xtr, Ytr, Xva, Yva, *,
               epochs: int, batch_size: int, lr: float, patience: int,
               gpu_cache: bool = True, resume: Optional[str] = None,
               ckpt_every: int = 0, peak_weight: float = 0.0, peak_pow: float = 1.0,
               loc_weight: float = 0.0, marg_weight: float = 0.0, loc_beta: float = 8.0,
               lr_schedule: str = "none", lr_min: float = 0.0,
               select_by: str = "mse", y_std: float = 1.0,
               grad_weight: float = 0.0, ssim_weight: float = 0.0,
               sample_w: Optional[np.ndarray] = None,
               time_budget_min: float = 0.0,
               field_loss_weight: float = 0.0, field_loss_warmup: int = 20,
               coherence_loss_weight: float = 0.0, coherence_cutoff: float = 0.25,
               phase_tr: Optional[np.ndarray] = None,
               target_floor: Optional[float] = None, y_mean: float = 0.0,
               field_select_n: int = 64,                field_select_every: int = 5,
               val_field_subset: Optional[np.ndarray] = None,
               val_paths: Optional[List[str]] = None,
               Yva_dex: Optional[np.ndarray] = None,
               m_grid: Optional[np.ndarray] = None,
               psi_grid: Optional[np.ndarray] = None,
               spectrum_field: str = "p",
               target_kind: str = "magnitude",
               Y_mag_train: Optional[np.ndarray] = None,
               mag_pred_dex: Optional[np.ndarray] = None):
    """Custom loop: per-epoch train+val loss/R2 -> live JSONL, best-val checkpoint,
    val early-stop, live loss plot. Returns (best_net, n_params).

    gpu_cache: keep the whole train/val set resident on the GPU (removes the
    per-batch host->device copy that otherwise dominates FNO/U-Net epoch time).
    resume: path to a checkpoint (.pt) to continue training from -- restores the
        model weights, the Adam optimizer state, the epoch counter, and the
        best-val-so-far, and *appends* to the existing history JSONL.
    ckpt_every: if >0, also write a periodic ckpt_<name>_ep<N>.pt every N epochs
        (in addition to the best-val ckpt_<name>.pt and the rolling
        ckpt_<name>_last.pt that always carries the latest resumable state).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = gpu_cache and dev.type == "cuda"
    net = net.to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Loss: plain MSE weights every pixel equally, so the ~90% noise-floor pixels
    # dominate and the sharp high-amplitude ridge/peak is under-fit (its location
    # and peak amplitude come out wrong). With peak_weight>0 we up-weight pixels by
    # the ground-truth amplitude (per-image min-max ranked, so the peak -> 1),
    # forcing the model to reproduce the peak/ridge accurately.
    _mse = torch.nn.MSELoss()
    is_phase = target_kind == "phase"
    Y_mag_t = None
    mag_pred_t = None
    if is_phase and Y_mag_train is not None:
        Y_mag_t = torch.tensor(Y_mag_train, dtype=torch.float32, device=dev if cache else "cpu")
    if is_phase and mag_pred_dex is not None:
        mag_pred_t = torch.tensor(mag_pred_dex, dtype=torch.float32, device=dev if cache else "cpu")

    # --- pixel term (plain or amplitude-weighted MSE) ---------------------- #
    def _pixel_loss(pred, target, idx=None):
        if is_phase:
            diff = pred - target
            circ = 1.0 - torch.cos(diff)
            if peak_weight and peak_weight > 0 and Y_mag_t is not None and idx is not None:
                mag = Y_mag_t[idx]
                tmin = mag.amin(dim=(1, 2), keepdim=True)
                tmax = mag.amax(dim=(1, 2), keepdim=True)
                s = ((mag - tmin) / (tmax - tmin + 1e-8)).clamp_(0.0, 1.0)
                w = 1.0 + peak_weight * s.pow(peak_pow)
                return (w * circ).mean()
            if Y_mag_t is not None and idx is not None:
                mag = Y_mag_t[idx]
                w = mag - mag.amin(dim=(1, 2), keepdim=True)
                w = w / (w.amax(dim=(1, 2), keepdim=True) + 1e-8)
                return ((0.25 + 0.75 * w) * circ).mean()
            return circ.mean()
        if peak_weight and peak_weight > 0:
            with torch.no_grad():
                tmin = target.amin(dim=(2, 3), keepdim=True)
                tmax = target.amax(dim=(2, 3), keepdim=True)
                s = ((target - tmin) / (tmax - tmin + 1e-8)).clamp_(0.0, 1.0)
                w = 1.0 + peak_weight * s.pow(peak_pow)
            return (w * (pred - target) ** 2).mean()
        return _mse(pred, target)

    # --- location-aware / marginal terms ---------------------------------- #
    # The pixel MSE (even amplitude-weighted) gives no direct gradient on *where*
    # the ridge sits, and in log space the noise floor dominates. These extra
    # terms optimize the shape explicitly:
    #   loc  = squared error of the soft-argmax psi_N of the peak (energy centroid
    #          of a temperature-sharpened softmax over the whole image). Standardi-
    #          zation/log are monotone so the max stays the max; this pulls the
    #          predicted mode to the correct radial location (core vs edge).
    #   marg = MSE of the psi-marginal and m-marginal profiles (energy vs psi_N and
    #          vs m), emphasizing the 1-D structure over the flat background.
    use_loc = loc_weight and loc_weight > 0
    use_marg = marg_weight and marg_weight > 0
    use_grad = grad_weight and grad_weight > 0
    use_ssim = ssim_weight and ssim_weight > 0

    def _grad_loss(pred, target):
        # match spatial gradients -> sharper ridge, less blur (edge fidelity)
        dpx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dtx = target[:, :, :, 1:] - target[:, :, :, :-1]
        dpy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dty = target[:, :, 1:, :] - target[:, :, :-1, :]
        return _mse(dpx, dtx) + _mse(dpy, dty)

    def _ssim_loss(pred, target):
        # 1 - SSIM with an 11x11 average window (differentiable, structure-driven)
        import torch.nn.functional as F
        k = 11; pad = k // 2
        mu_p = F.avg_pool2d(pred, k, 1, pad); mu_t = F.avg_pool2d(target, k, 1, pad)
        sp = F.avg_pool2d(pred * pred, k, 1, pad) - mu_p ** 2
        st = F.avg_pool2d(target * target, k, 1, pad) - mu_t ** 2
        spt = F.avg_pool2d(pred * target, k, 1, pad) - mu_p * mu_t
        C1 = 0.01 ** 2; C2 = 0.03 ** 2
        s = ((2 * mu_p * mu_t + C1) * (2 * spt + C2)) / \
            ((mu_p ** 2 + mu_t ** 2 + C1) * (sp + st + C2) + 1e-12)
        return 1.0 - s.mean()
    _psi_map = None  # lazily built (H*W,) psi_N coordinate for the soft-argmax

    def _psi_softloc(z, psi_flat):
        B = z.shape[0]
        zf = z.reshape(B, -1)
        zf = zf - zf.amax(dim=1, keepdim=True)
        p = torch.softmax(loc_beta * zf, dim=1)
        return (p * psi_flat).sum(dim=1)             # (B,) expected psi_N of peak

    def lossf(pred, target, idx=None):
        nonlocal _psi_map
        loss = _pixel_loss(pred, target, idx)
        if use_loc:
            if _psi_map is None:
                W = target.shape[-1]; H = target.shape[-2]
                psi = torch.linspace(0.0, 1.0, W, device=target.device)
                _psi_map = psi.view(1, W).expand(H, W).reshape(-1)
            lp = _psi_softloc(pred, _psi_map)
            with torch.no_grad():
                lt = _psi_softloc(target, _psi_map)
            loss = loss + loc_weight * ((lp - lt) ** 2).mean()
        if use_marg:
            # dim2 = m (rows), dim3 = psi (cols)
            marg = (_mse(pred.mean(dim=2), target.mean(dim=2))     # psi-marginal
                    + _mse(pred.mean(dim=3), target.mean(dim=3)))  # m-marginal
            loss = loss + marg_weight * marg
        if use_grad:
            loss = loss + grad_weight * _grad_loss(pred, target)
        if use_ssim:
            loss = loss + ssim_weight * _ssim_loss(pred, target)
        return loss

    _terms = ["MSE" if not (peak_weight and peak_weight > 0)
              else f"peakMSE(a={peak_weight},p={peak_pow})"]
    if use_loc:
        _terms.append(f"loc(w={loc_weight},beta={loc_beta})")
    if use_marg:
        _terms.append(f"marg(w={marg_weight})")
    if use_grad:
        _terms.append(f"grad(w={grad_weight})")
    if use_ssim:
        _terms.append(f"ssim(w={ssim_weight})")
    use_field = field_loss_weight and field_loss_weight > 0 and phase_tr is not None
    use_coh = coherence_loss_weight and coherence_loss_weight > 0
    if use_field:
        _terms.append(f"field(w={field_loss_weight},warm={field_loss_warmup})")
    if use_coh:
        _terms.append(f"coh(w={coherence_loss_weight},cut={coherence_cutoff})")
    print(f"  [loss] composite = {' + '.join(_terms)}", flush=True)
    n_train = len(Xtr)
    # Optional core-mode class balancing: oversample rare peak-psi bins so the
    # model sees each ridge location ~equally (instead of the edge-dominated
    # empirical mix). Implemented as weighted resampling with replacement.
    use_balance = sample_w is not None
    if use_balance:
        print(f"  [balance] psi-balanced sampling on "
              f"(w in [{sample_w.min():.2f},{sample_w.max():.2f}])", flush=True)
    if cache:
        Xg = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        Yg = torch.tensor(Ytr[:, None], dtype=torch.float32, device=dev)
        if use_balance:
            wt = torch.tensor(sample_w, dtype=torch.float32, device=dev)
    else:
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        Yt = torch.tensor(Ytr[:, None], dtype=torch.float32)
        if use_balance:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_w, dtype=torch.double),
                num_samples=n_train, replacement=True)
            loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size,
                                sampler=sampler)
        else:
            loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size,
                                shuffle=True)
    phase_tr_t = None
    if use_field:
        phase_tr_t = torch.tensor(phase_tr, dtype=torch.float32, device=dev if cache else "cpu")
    from spectrum_field_loss import (  # noqa: E402
        field_loss_training_grid_torch,
        field_loss_honest_phase_torch,
        coherence_loss_torch,
        eval_val_field_selection,
        field_metric_improved,
    )
    Xv = torch.tensor(Xva, dtype=torch.float32).to(dev)
    Yv_t = torch.tensor(Yva[:, None], dtype=torch.float32).to(dev)
    # psi_N coordinate along the column (dim3) axis, for the peak-location metric
    _Wv = Yva.shape[-1]
    _psi_axis = np.linspace(0.0, 1.0, _Wv)

    def _peak_psi(arr):
        # arr: (N,1,H,W) -> psi_N of the global-max pixel of each image
        a = arr.reshape(arr.shape[0], -1)
        col = np.argmax(a, axis=1) % arr.shape[-1]
        return _psi_axis[col]

    hist_path = out / f"history_{name}.jsonl"
    ckpt_path = out / f"ckpt_{name}.pt"
    last_path = out / f"ckpt_{name}_last.pt"
    best_val = float("inf"); best_state = None; no_improve = 0
    best_epoch = 0
    best_field_frac = float("inf"); best_field_p90 = float("inf")
    start_epoch = 0
    train_wall_start = time.time()
    if resume:
        rp = Path(resume)
        ck = torch.load(rp, map_location=dev)
        net.load_state_dict(ck["state_dict"])
        if ck.get("optimizer") is not None:
            try:
                opt.load_state_dict(ck["optimizer"])
            except Exception as exc:
                print(f"  [resume] could not restore optimizer state: {exc}")
        start_epoch = int(ck.get("epoch", 0))
        best_val = float(ck.get("best_val", ck.get("val_loss", float("inf"))))
        no_improve = int(ck.get("no_improve", 0))
        best_epoch = int(ck.get("best_epoch", start_epoch))
        best_field_frac = float(ck.get("best_field_frac_gt1", float("inf")))
        best_field_p90 = float(ck.get("best_field_p90", float("inf")))
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"  [resume] {rp} -> start at epoch {start_epoch+1}, "
              f"best_val={best_val:.5f} no_improve={no_improve} best_epoch={best_epoch}",
              flush=True)
    if not resume:
        hist_path.write_text("")  # truncate only for a fresh run
    yv_np = Yva[:, None]

    def _save(path, epoch, vl, vr2):
        torch.save({"state_dict": {k: v.detach().cpu().clone()
                                   for k, v in net.state_dict().items()},
                    "optimizer": opt.state_dict(), "epoch": epoch,
                    "val_loss": vl, "val_r2": vr2, "best_val": best_val,
                    "no_improve": no_improve, "best_epoch": best_epoch,
                    "best_field_frac_gt1": best_field_frac,
                    "best_field_p90": best_field_p90,
                    "select_by": select_by,
                    "model": name}, path)

    # Cosine LR annealing (manual so it resumes cleanly by absolute epoch and
    # works with --patience 0). lr(ep) goes from `lr` down to `lr_min` following
    # a half-cosine over [1, epochs]; "none" keeps lr constant.
    def _lr_at(ep: int) -> float:
        if lr_schedule == "cosine":
            prog = min(max((ep - 1) / max(1, epochs - 1), 0.0), 1.0)
            return lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * prog))
        return lr
    if lr_schedule == "cosine":
        print(f"  [lr] cosine anneal {lr:g} -> {lr_min:g} over {epochs} epochs", flush=True)
    if use_field and not cache:
        print("  [field-loss] requires GPU cache; field term disabled for this run", flush=True)
        use_field = False

    def _warm_scale(ep: int) -> float:
        if field_loss_warmup <= 0:
            return 1.0
        return min(1.0, ep / float(field_loss_warmup))

    def _aux_loss(pred, target, idx, ep: int):
        if not (use_field or use_coh):
            return pred.new_tensor(0.0)
        warm = _warm_scale(ep)
        aux = pred.new_tensor(0.0)
        if use_field:
            if is_phase and mag_pred_t is not None and phase_tr_t is not None and Y_mag_t is not None:
                mag_p = mag_pred_t[idx]
                mag_t = Y_mag_t[idx]
                ph = phase_tr_t[idx]
                aux = aux + warm * field_loss_weight * field_loss_honest_phase_torch(
                    pred, mag_p, ph, mag_t,
                    y_mean=y_mean, y_std=y_std, target_floor=target_floor)
            elif phase_tr_t is not None:
                ph = phase_tr_t[idx]
                aux = aux + warm * field_loss_weight * field_loss_training_grid_torch(
                    pred, target, ph, y_mean=y_mean, y_std=y_std, target_floor=target_floor)
        if use_coh:
            aux = aux + warm * coherence_loss_weight * coherence_loss_torch(
                pred, target, y_mean=y_mean, y_std=y_std,
                target_floor=target_floor, cutoff=coherence_cutoff)
        return aux

    run_field_eval = (
        select_by == "field"
        or (field_loss_weight > 0)
        or (coherence_loss_weight > 0)
    ) and val_field_subset is not None and val_paths is not None

    for epoch in range(start_epoch + 1, epochs + 1):
        cur_lr = _lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        net.train(); tl = 0.0
        if cache:
            if use_balance:
                perm = torch.multinomial(wt, n_train, replacement=True)
            else:
                perm = torch.randperm(n_train, device=dev)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pt = net(Xg[idx])
                loss = lossf(pt, Yg[idx], idx) + _aux_loss(pt, Yg[idx], idx, epoch)
                loss.backward(); opt.step()
                tl += loss.item() * len(idx)
        else:
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                pt = net(xb)
                if use_field or use_coh:
                    if not cache:
                        loss = lossf(pt, yb)
                    else:
                        loss = lossf(pt, yb)
                else:
                    loss = lossf(pt, yb)
                loss.backward(); opt.step()
                tl += loss.item() * len(xb)
        tl /= n_train
        net.eval()
        with torch.no_grad():
            vp = []; vcomp = 0.0
            for i in range(0, len(Xv), batch_size):
                pt = net(Xv[i:i + batch_size])
                vcomp += float(lossf(pt, Yv_t[i:i + batch_size]).item()) * pt.shape[0]
                vp.append(pt.cpu().numpy())
            vp = np.concatenate(vp)
            vl = float(np.mean((vp - yv_np) ** 2)); vr2 = r2(yv_np, vp)
            vcomp /= len(Xv)
            # peak-location error in psi_N units (0..1): how far the predicted
            # global-max sits from the true one -- the metric that actually tracks
            # core-vs-edge mode structure, which MSE/R2 are blind to.
            vdpsi = float(np.mean(np.abs(_peak_psi(vp) - _peak_psi(yv_np))))
            # RMSE in physical "dex" units (decades of |dp| below the peak): un-do
            # the global z-score by *y_std. val_rmse = overall amplitude error;
            # val_peak_rmse = error over the top-1% amplitude (peak/ridge) pixels,
            # i.e. how wrong the actual mode amplitude is (R2 hides this).
            _resid = vp - yv_np
            val_rmse = float(np.sqrt(np.mean(_resid ** 2)) * y_std)
            _gf = yv_np.reshape(yv_np.shape[0], -1)
            _thr = np.percentile(_gf, 99.0, axis=1)[:, None, None, None]
            _pk = yv_np >= _thr
            val_peak_rmse = float(np.sqrt(np.mean(_resid[_pk] ** 2)) * y_std)
        rec = {"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_r2": vr2,
               "val_comp": vcomp, "val_dpsi": vdpsi, "val_rmse": val_rmse,
               "val_peak_rmse": val_peak_rmse, "lr": cur_lr}
        field_stats = None
        if run_field_eval and (epoch % field_select_every == 0 or epoch == 1):
            field_stats = eval_val_field_selection(
                net, Xva, Yva_dex if Yva_dex is not None else Yva,
                val_field_subset, val_paths, m_grid, psi_grid, spectrum_field,
                device=str(dev), batch_size=batch_size, crf_cutoff=coherence_cutoff)
            rec.update(field_stats)
        # model selection: composite loss (peak-aware), plain MSE, or field metrics
        if select_by == "field" and field_stats is not None:
            improved = field_metric_improved(
                field_stats["val_field_frac_gt1"], field_stats["val_field_p90"],
                best_field_frac, best_field_p90)
            sel = field_stats["val_field_frac_gt1"]
        elif select_by == "field":
            improved = False
            sel = best_field_frac
        else:
            sel = vcomp if select_by == "composite" else vl
            improved = sel < best_val
        if improved:
            if select_by == "field" and field_stats is not None:
                best_field_frac = field_stats["val_field_frac_gt1"]
                best_field_p90 = field_stats["val_field_p90"]
            else:
                best_val = sel
            no_improve = 0
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            _save(ckpt_path, epoch, vl, vr2)
            rec["checkpoint"] = True
        elif select_by != "field" or field_stats is not None:
            no_improve += 1
        # Always keep a rolling "last" checkpoint (with optimizer state) so the
        # run can be resumed from exactly where it stopped, even mid-plateau.
        _save(last_path, epoch, vl, vr2)
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            _save(out / f"ckpt_{name}_ep{epoch}.pt", epoch, vl, vr2)
        with hist_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        if epoch % 5 == 0 or improved or epoch == 1:
            _loss_plot(hist_path, name, out)
        if epoch % 10 == 0 or epoch == 1:
            msg = (f"  [{name}] epoch {epoch}/{epochs} train={tl:.4f} "
                   f"val={vl:.4f} val_r2={vr2:.4f} comp={vcomp:.4f} "
                   f"dpsi={vdpsi:.4f} rmse={val_rmse:.3f} pkrmse={val_peak_rmse:.3f}")
            if field_stats is not None:
                msg += (f" fld_frac>1={field_stats['val_field_frac_gt1']:.3f}"
                        f" fld_p90={field_stats['val_field_p90']:.3f}"
                        f" fld_crf={field_stats['val_field_crf']:.3f}")
            msg += "  *best" if improved else ""
            print(msg, flush=True)
        if patience > 0 and no_improve >= patience:
            print(f"  [{name}] early stop at epoch {epoch} "
                  f"(best epoch {best_epoch}, no_improve={no_improve})", flush=True)
            with hist_path.open("a") as fh:
                fh.write(json.dumps({"epoch": epoch, "early_stop": True}) + "\n")
            break
        if time_budget_min > 0:
            elapsed_min = (time.time() - train_wall_start) / 60.0
            if elapsed_min >= time_budget_min:
                print(f"  [{name}] time budget {time_budget_min:.0f} min reached "
                      f"({elapsed_min:.1f} min elapsed); resumable {last_path}", flush=True)
                if best_state is not None:
                    net.load_state_dict(best_state)
                _loss_plot(hist_path, name, out)
                return net, n_params
    if best_state is not None:
        net.load_state_dict(best_state)
    _loss_plot(hist_path, name, out)
    return net, n_params


def _predict_net(net, X, batch_size: int) -> np.ndarray:
    import torch
    dev = next(net.parameters()).device
    net.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(net(Xt[i:i + batch_size].to(dev)).cpu().numpy())
    return np.concatenate(out).squeeze(1)  # (B, H, W)


def r2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def pattern_r2(yt: np.ndarray, yp: np.ndarray) -> float:
    """Per-image de-meaned R2 (spatial pattern fidelity, scale-offset removed)."""
    yt = yt - yt.reshape(yt.shape[0], -1).mean(1)[:, None, None]
    yp = yp - yp.reshape(yp.shape[0], -1).mean(1)[:, None, None]
    return r2(yt, yp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--spectrum-field", default="p")
    ap.add_argument("--n-cases", type=int, default=2500)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--m-lo", type=float, default=-80.0)
    ap.add_argument("--m-hi", type=float, default=20.0)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--target-norm", choices=["none", "max"], default="none",
                    help="Per-case magnitude normalization: 'max' scales each "
                         "case's spectrum so its peak is 1 (before any log).")
    ap.add_argument("--target-space", choices=["log10", "raw"], default="log10",
                    help="'log10' -> log10(mag+eps); 'raw' -> (normalized) magnitude.")
    ap.add_argument("--target-floor", type=float, default=None,
                    help="Clip the (max-norm) log10 target to N decades below the "
                         "peak (peak=0). e.g. 6 keeps 10^0..10^-6 and floors the rest "
                         "to -6, deleting the noise-floor texture. Use with "
                         "--target-norm max --target-space log10.")
    ap.add_argument("--target-smooth", type=float, default=None,
                    help="Gaussian-denoise the log target with this sigma (grid px) "
                         "to remove high-frequency speckle while keeping the ridge. "
                         "Try 1. Combine with --target-floor.")
    ap.add_argument("--target-kind", choices=["magnitude", "phase", "real", "imag"],
                    default="magnitude",
                    help="Surrogate target on the (m,psi) grid. Default: log10|δp̂|. "
                         "'phase' trains φ(m,psi) with circular loss.")
    ap.add_argument("--mag-run", default=None,
                    help="For --target-kind phase: directory of frozen magnitude run "
                         "(uses predictions_cache.npz or inference) for honest field loss.")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start FNO weights from another checkpoint (e.g. magnitude ckpt).")
    ap.add_argument("--exclude-list", default=None,
                    help="Path to a quarantine JSON (from scan_quality.py) or a text "
                         "file of case keys to EXCLUDE from the dataset (bad data).")
    ap.add_argument("--geom-channels", action="store_true",
                    help="Add geometry conditioning channels: magnetic shear s(psi_N), "
                         "flux-surface-averaged |grad psi|(psi_N), and LCFS proximity "
                         "(1-psi_N). Helps edge-localized modes.")
    ap.add_argument("--balance-psi", action="store_true",
                    help="Cross-balance core modes: oversample rare peak-psi_N bins "
                         "via weighted resampling so core-localized modes are seen as "
                         "often as edge modes (removes the edge-dominated bias).")
    ap.add_argument("--balance-bins", type=int, default=10,
                    help="Number of peak-psi_N bins used for --balance-psi.")
    ap.add_argument("--balance-strength", type=float, default=1.0,
                    help="Power on inverse bin frequency for --balance-psi: 1 = full "
                         "inverse-frequency balancing, 0 = uniform (off), 0.5 = milder.")
    ap.add_argument("--models", nargs="+", default=["fno2d", "unet"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25,
                    help="Early-stop after this many epochs with no val-loss "
                         "improvement. Use 0 to disable early stopping.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", default=None,
                    help="Continue training from a checkpoint .pt (restores "
                         "weights, optimizer, epoch, best-val; appends history). "
                         "Typically runs/<dir>/ckpt_<model>_last.pt.")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="Also save a periodic ckpt_<model>_ep<N>.pt every N epochs.")
    ap.add_argument("--peak-weight", type=float, default=0.0,
                    help="Amplitude-weighted MSE: up-weight high-|dp| (peak/ridge) "
                         "pixels by 1 + alpha*rank^pow so the peak location & amplitude "
                         "are reproduced instead of the noise floor. 0 = plain MSE. "
                         "Try 4-10.")
    ap.add_argument("--peak-pow", type=float, default=2.0,
                    help="Exponent sharpening the peak weighting (higher = focus "
                         "more tightly on the very top amplitudes).")
    ap.add_argument("--fno-modes", type=int, default=16,
                    help="FNO spectral modes per axis (128 grid -> Nyquist ~64). "
                         "16 blurs sharp peaks; try 48 to resolve the ridge.")
    ap.add_argument("--fno-hidden", type=int, default=32,
                    help="FNO hidden channel width (raise with --fno-modes, e.g. 64).")
    ap.add_argument("--loc-weight", type=float, default=0.0,
                    help="Weight of the soft-argmax peak-location loss (psi_N of the "
                         "mode peak). Directly targets core-vs-edge location. Try 0.5-5.")
    ap.add_argument("--marg-weight", type=float, default=0.0,
                    help="Weight of the psi/m marginal-profile MSE (energy-vs-psi_N "
                         "and energy-vs-m). Emphasizes 1-D structure. Try 0.5-2.")
    ap.add_argument("--loc-beta", type=float, default=8.0,
                    help="Softmax temperature for the soft-argmax peak locator "
                         "(higher = sharper toward the true argmax).")
    ap.add_argument("--grad-weight", type=float, default=0.0,
                    help="Weight of the spatial-gradient (edge) loss: sharpens the "
                         "ridge and reduces blur. Try 0.5-2.")
    ap.add_argument("--ssim-weight", type=float, default=0.0,
                    help="Weight of the (1-SSIM) structural loss: drives visually "
                         "faithful reconstructions. Try 0.5-2.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="'cosine' anneals lr from --lr down to --lr-min over "
                         "--epochs (by absolute epoch, so it resumes cleanly).")
    ap.add_argument("--lr-min", type=float, default=1e-5,
                    help="Final learning rate for the cosine schedule.")
    ap.add_argument("--select-by", choices=["mse", "composite", "field"], default="mse",
                    help="Metric used to pick the best checkpoint & early-stop: "
                         "'mse' (plain pixel MSE, R2-like), 'composite' (the full "
                         "peak-location + marginal loss), or 'field' (frac relL2>1 "
                         "then p90 on a family-stratified val subset).")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="Wall-time budget in minutes (0=disabled). Before the budget "
                         "is hit, write a resumable last checkpoint and exit cleanly.")
    ap.add_argument("--field-loss-weight", type=float, default=0.0,
                    help="Weight of differentiable IFFT-proxy field relL2 loss with "
                         "oracle true phase on the training grid (0=off).")
    ap.add_argument("--field-loss-warmup", type=int, default=20,
                    help="Linear warmup epochs for --field-loss-weight.")
    ap.add_argument("--coherence-loss-weight", type=float, default=0.0,
                    help="Weight of phase-free coherence-penalty (CRF surrogate; 0=off).")
    ap.add_argument("--coherence-cutoff", type=float, default=0.25,
                    help="Low radial-k quantile for coherence / CRF losses.")
    ap.add_argument("--field-select-n", type=int, default=64,
                    help="Family-stratified val cases for --select-by field.")
    ap.add_argument("--field-select-every", type=int, default=5,
                    help="Run field-selection eval every N epochs.")
    ap.add_argument("--no-gpu-cache", action="store_true",
                    help="Disable keeping the full train/val set resident on the GPU.")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/spectrum_image")
    ap.add_argument("--plot-only", action="store_true",
                    help="Regenerate loss curves from history_*.jsonl in --out and exit "
                         "(use to monitor a running job from the login node).")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    if args.plot_only:
        for hp in sorted(out.glob("history_*.jsonl")):
            name = hp.stem.replace("history_", "")
            _loss_plot(hp, name, out)
            rows = [json.loads(l) for l in hp.read_text().splitlines() if l.strip()]
            live = [r for r in rows if "val_r2" in r]
            if live:
                b = max(live, key=lambda r: r["val_r2"])
                print(f"{name}: {len(live)} epochs logged; best val_r2={b['val_r2']:.4f} "
                      f"@epoch {b['epoch']} -> {out/f'loss_{name}.png'}")
        return

    exclude_keys = None
    if args.exclude_list:
        raw = Path(args.exclude_list).read_text().strip()
        try:
            exclude_keys = set(json.loads(raw).keys())    # quarantine {key: reason}
        except Exception:
            exclude_keys = set(l.strip() for l in raw.splitlines() if l.strip())
        print(f"Excluding {len(exclude_keys)} quarantined cases from {args.exclude_list}")
    need_field = (
        args.field_loss_weight > 0
        or args.coherence_loss_weight > 0
        or args.select_by == "field"
        or args.target_kind == "phase"
    )
    Y_mag = None
    if need_field or args.target_kind == "phase":
        built = build_dataset(
            args.batch_dir, args.filename, args.n_cases, args.grid,
            args.m_lo, args.m_hi, args.spectrum_field, args.eps,
            target_norm=args.target_norm, target_space=args.target_space,
            target_floor=args.target_floor, target_smooth=args.target_smooth,
            target_kind=args.target_kind,
            exclude_keys=exclude_keys, geom_channels=args.geom_channels,
            return_paths=True)
        if args.target_kind == "phase":
            X, Y, chan_names, keys, paths, Y_mag = built
        else:
            X, Y, chan_names, keys, paths = built
    else:
        X, Y, chan_names, keys = build_dataset(
            args.batch_dir, args.filename, args.n_cases, args.grid,
            args.m_lo, args.m_hi, args.spectrum_field, args.eps,
            target_norm=args.target_norm, target_space=args.target_space,
            target_floor=args.target_floor, target_smooth=args.target_smooth,
            target_kind=args.target_kind,
            exclude_keys=exclude_keys, geom_channels=args.geom_channels)
        paths = None
    N = X.shape[0]
    m_grid = np.linspace(float(args.m_lo), float(args.m_hi), args.grid)
    psi_grid = np.linspace(0.0, 1.0, args.grid)

    # Persist the run configuration so `python -m surge.check_training` (and the
    # user) can see exactly what preprocessing/target this run used.
    target_desc = (f"{args.target_kind} "
                   + (("max-normalized " if args.target_norm == "max" else "")
                   + ("log10|dp|" if args.target_space == "log10" and args.target_kind == "magnitude" else "")
                   + (f", floor -{args.target_floor:g}dex" if args.target_floor else "")
                   + (f", smooth s={args.target_smooth:g}" if args.target_smooth else ""))
                   + ", global z-score")
    (out / "run_config.json").write_text(json.dumps({
        "batch_dir": args.batch_dir, "filename": args.filename,
        "spectrum_field": args.spectrum_field, "n_cases": args.n_cases,
        "grid": args.grid, "m_window": [args.m_lo, args.m_hi],
        "models": list(args.models), "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience,
        "seed": args.seed, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "target_norm": args.target_norm, "target_space": args.target_space,
        "target_floor": args.target_floor, "target_smooth": args.target_smooth,
        "target_kind": args.target_kind,
        "mag_run": args.mag_run,
        "init_from": args.init_from,
        "target": target_desc,
        "peak_weight": args.peak_weight, "peak_pow": args.peak_pow,
        "fno_modes": args.fno_modes, "fno_hidden": args.fno_hidden,
        "loc_weight": args.loc_weight, "marg_weight": args.marg_weight,
        "loc_beta": args.loc_beta,
        "grad_weight": args.grad_weight, "ssim_weight": args.ssim_weight,
        "exclude_list": args.exclude_list,
        "geom_channels": args.geom_channels,
        "balance_psi": args.balance_psi, "balance_bins": args.balance_bins,
        "balance_strength": args.balance_strength,
        "lr": args.lr, "lr_schedule": args.lr_schedule, "lr_min": args.lr_min,
        "select_by": args.select_by,
        "time_budget_min": args.time_budget_min,
        "field_loss_weight": args.field_loss_weight,
        "field_loss_warmup": args.field_loss_warmup,
        "coherence_loss_weight": args.coherence_loss_weight,
        "coherence_cutoff": args.coherence_cutoff,
        "field_select_n": args.field_select_n,
        "field_select_every": args.field_select_every,
    }, indent=2))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_test = int(args.test_frac * N)
    n_val = int(args.val_frac * N)
    te = perm[:n_test]; va = perm[n_test:n_test + n_val]; tr = perm[n_test + n_val:]
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    # Persist the exact split (case keys + row indices) so offline eval and
    # cross-model comparison use the identical held-out set (reproducibility).
    keys_arr = np.asarray(keys)
    (out / "splits.json").write_text(json.dumps({
        "seed": args.seed, "n_cases": int(N),
        "test_frac": args.test_frac, "val_frac": args.val_frac,
        "exclude_list": args.exclude_list,
        "train_idx": tr.tolist(), "val_idx": va.tolist(), "test_idx": te.tolist(),
        "train_keys": keys_arr[tr].tolist(), "val_keys": keys_arr[va].tolist(),
        "test_keys": keys_arr[te].tolist(),
    }))

    # Standardize input channels (train stats), and target (train stats, global).
    xm = X[tr].mean((0, 2, 3), keepdims=True)
    xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs
    ym = float(Y[tr].mean()); ysd = float(Y[tr].std() + 1e-8)
    Yn = (Y - ym) / ysd
    (out / "norm_stats.json").write_text(json.dumps({
        "y_mean": ym, "y_std": ysd,
        "input_mean": xm.squeeze().tolist(),
        "input_std": xs.squeeze().tolist(),
    }, indent=2))

    mag_pred_dex = None
    if args.target_kind == "phase":
        if args.mag_run:
            mag_pred_dex = load_mag_pred_dex_from_run(
                Path(args.mag_run), keys, X, tr, device="cuda", batch_size=args.batch_size)
        else:
            print("WARNING: --target-kind phase without --mag-run uses GT |δp̂| for field loss",
                  flush=True)
            mag_pred_dex = Y_mag if Y_mag is not None else None

    # Optional core-mode balancing weights (computed on the raw training targets;
    # argmax is invariant to the monotone z-score so Y vs Yn is equivalent).
    sample_w = None
    if args.balance_psi:
        sample_w, bcounts = _psi_balance_weights(
            Y[tr], n_bins=args.balance_bins, strength=args.balance_strength)
        print(f"psi-balance: peak-psi bin counts (train) = {bcounts.astype(int).tolist()}")

    phase_tr = None
    val_field_subset = None
    val_paths_list = None
    if need_field:
        from spectrum_field_loss import build_phase_grids_for_keys, stratified_subset  # noqa: E402
        print("Loading oracle phase grids for field loss / selection...", flush=True)
        phases_all = build_phase_grids_for_keys(
            args.batch_dir, args.filename, keys, args.grid,
            float(args.m_lo), float(args.m_hi), args.spectrum_field)
        phase_tr = phases_all[tr]
        keys_arr = np.asarray(keys)
        val_keys_list = keys_arr[va].tolist()
        val_paths_list = [paths[i] for i in va]
        val_field_subset = stratified_subset(
            val_keys_list, min(args.field_select_n, len(va)), args.seed)

    results: Dict[str, Dict] = {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        t0 = time.time()
        net = _build_net(name, X.shape[1], fno_modes=args.fno_modes,
                         fno_hidden=args.fno_hidden, grid=args.grid)
        if net is None:
            print(f"  unknown model {name}, skipping"); continue
        if args.init_from and not args.resume:
            import torch
            ck_init = torch.load(args.init_from, map_location="cpu", weights_only=False)
            net.load_state_dict(ck_init["state_dict"])
            print(f"  warm-started from {args.init_from} (epoch {ck_init.get('epoch')})",
                  flush=True)
        net, n_params = _train_net(
            net, name, out, Xn[tr], Yn[tr], Xn[va], Yn[va],
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            gpu_cache=not args.no_gpu_cache, resume=args.resume, ckpt_every=args.ckpt_every,
            peak_weight=args.peak_weight, peak_pow=args.peak_pow,
            loc_weight=args.loc_weight, marg_weight=args.marg_weight,
            loc_beta=args.loc_beta, lr_schedule=args.lr_schedule, lr_min=args.lr_min,
            select_by=args.select_by, y_std=ysd,
            grad_weight=args.grad_weight, ssim_weight=args.ssim_weight,
            sample_w=sample_w,
            time_budget_min=args.time_budget_min,
            field_loss_weight=args.field_loss_weight,
            field_loss_warmup=args.field_loss_warmup,
            coherence_loss_weight=args.coherence_loss_weight,
            coherence_cutoff=args.coherence_cutoff,
            phase_tr=phase_tr,
            target_floor=args.target_floor, y_mean=ym,
            field_select_n=args.field_select_n,
            field_select_every=args.field_select_every,
            val_field_subset=val_field_subset,
            val_paths=val_paths_list,
            Yva_dex=Y[va],
            m_grid=m_grid, psi_grid=psi_grid,
            spectrum_field=args.spectrum_field,
            target_kind=args.target_kind,
            Y_mag_train=Y_mag[tr] if Y_mag is not None else None,
            mag_pred_dex=mag_pred_dex[tr] if mag_pred_dex is not None else None)
        pred = _predict_net(net, Xn[te], args.batch_size)  # (n_test, H, W)
        yt = Yn[te]
        res = {"test_r2_global": r2(yt, pred),
               "test_pattern_r2": pattern_r2(yt, pred),
               "train_seconds": time.time() - t0,
               "n_params": n_params,
               "checkpoint": str(out / f"ckpt_{name}.pt"),
               "history": str(out / f"history_{name}.jsonl")}
        results[name] = res
        print(f"  {name}: test R2(global)={res['test_r2_global']:.4f} "
              f"pattern R2={res['test_pattern_r2']:.4f} "
              f"({n_params/1e6:.2f}M params, {res['train_seconds']:.0f}s)")
        _save_examples(out, name, X, Yn, te, pred, chan_names, args)

    summary = {"n_cases": N, "grid": args.grid, "channels": chan_names,
               "m_window": [args.m_lo, args.m_hi], "target": target_desc,
               "target_norm": args.target_norm, "target_space": args.target_space,
               "y_mean": ym, "y_std": ysd, "results": results,
               "per_mode_baseline_test_r2": 0.358}
    (out / "spectrum_image_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'spectrum_image_metrics.json'}")
    for k, v in results.items():
        print(f"  {k:8s} global R2={v['test_r2_global']:.3f}  pattern R2={v['test_pattern_r2']:.3f}")


def _save_examples(out: Path, name: str, X, Yn, te, pred, chan_names, args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ext = [0.0, 1.0, args.m_lo, args.m_hi]
    nshow = min(3, len(te))
    fig, axes = plt.subplots(nshow, 3, figsize=(11, 3.2 * nshow))
    if nshow == 1:
        axes = axes[None, :]
    for r in range(nshow):
        yt = Yn[te[r]]; yp = pred[r]
        vmin, vmax = np.percentile(yt, 2), np.percentile(yt, 98)
        for c, (img, title) in enumerate([(yt, "true"), (yp, "pred"),
                                          (yp - yt, "residual")]):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", aspect="auto", extent=ext,
                           cmap="magma" if c < 2 else "coolwarm",
                           vmin=(vmin if c < 2 else None), vmax=(vmax if c < 2 else None))
            ax.set_title(f"{name} {title}" if r == 0 else title)
            if c == 0:
                ax.set_ylabel("m")
            if r == nshow - 1:
                ax.set_xlabel(r"$\psi_N$")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "plots" / f"{name}_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
