#!/usr/bin/env python
"""Direct RZ surrogate: predict Re(δp)(R,Z) from equilibrium conditioning channels.

Unlike train_spectrum_image.py (log10|δp̂|(m,ψ)), this trains on pertfields/p_phi0
— the real perturbed pressure at toroidal angle φ₀ on the 201×201 RZ grid — with no
spectrum FFT and no phase oracle.

Input channels (broadcast on the RZ grid where noted):
  psin, |grad ψ| (norm), LCFS proximity, q(ψ), p(ψ), R_norm, Z_norm,
  kappa, delta, epsilon, pscale, batemanscale, ntor, q0, q95, qmin, p0, inside_mask

Target: per-case max-normalized Re(δp) (signed), optional noise floor & smooth,
then global z-score (train stats). Spatial input channels are per-case max-normalized
(peak = 1) like the spectrum recipe. Loss supports peak-weighted MSE + soft-argmax
peak location in ψ_N (loc) + R/Z marginals (marg). Model selection by z-scored MSE,
composite loss, or val relL2 (field).

Usage:
  python scripts/m3dc1/internal/train_rz_field_image.py \\
    --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \\
    --filename csdata_deltap_b_ver.h5 --n-cases 0 --grid 128 \\
    --models fno2d --out runs/rz_field_fno48_re_deltap_smooth0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[2]
if str(_SCRIPTS / "m3dc1") not in sys.path:
    sys.path.insert(0, str(_SCRIPTS / "m3dc1"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import h5py  # noqa: E402
from gauge_fix import gauge_fix_field, rel_l2_complex_batch, _sanitize_field  # noqa: E402
from dataset_complex_v2 import extract_equilibrium_scalars, find_complex_v2_files, _decode  # noqa: E402
from train_spectrum_image import (  # noqa: E402
    _build_net,
    _predict_net,
    pattern_r2,
    r2,
)


def _finite_array(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(out)
    if bad.any():
        out = out.copy()
        out[bad] = fill
    return out


def _max_norm_field(arr: np.ndarray) -> np.ndarray:
    """Per-case max-abs normalization so peak |field| = 1."""
    s = float(np.nanmax(np.abs(arr))) or 1.0
    return _finite_array(arr / s)


def _apply_target_floor(y: np.ndarray, floor_dex: Optional[float]) -> np.ndarray:
    """Zero sub-threshold pixels after max-norm (floor in dex below peak)."""
    if floor_dex is None or floor_dex <= 0:
        return y
    thr = 10.0 ** (-float(floor_dex))
    out = y.copy()
    out[np.abs(out) < thr] = 0.0
    return out


def _rz_loss_plot(hist_path: Path, name: str, out: Path) -> None:
    """Loss / realistic val relL2 curves (safe when losses are NaN or non-positive)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "train_loss" in r and "val_loss" in r]
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    tl = [r["train_loss"] for r in rows]
    vl = [r["val_loss"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, tl, label="train")
    a1.plot(ep, vl, label="val")
    finite_vl = [v for v in vl if np.isfinite(v)]
    if finite_vl:
        best = min(rows, key=lambda r: r["val_loss"] if np.isfinite(r["val_loss"]) else float("inf"))
        if np.isfinite(best["val_loss"]):
            a1.axvline(best["epoch"], color="k", ls=":", lw=1, label=f"best ep {best['epoch']}")
    pos = [v for v in tl + vl if np.isfinite(v) and v > 0]
    if pos:
        a1.set_yscale("log")
    a1.set_xlabel("epoch")
    a1.set_ylabel("MSE loss (z-scored target)")
    a1.set_title(f"{name}: loss")
    a1.legend()
    if any("val_relL2_median" in r for r in rows):
        rel = [r.get("val_relL2_median", np.nan) for r in rows]
        a2.plot(ep, rel, color="C3", label="val relL2 median")
        if any(np.isfinite(x) for x in rel):
            a2.axhline(0.49, color="g", ls="--", lw=1, label="D′ oracle ~0.49")
            a2.axhline(1.0, color="r", ls="--", lw=1, label="relL2 = 1")
        a2.set_ylabel("relL2 (max-norm field)")
        a2.set_title(f"{name}: val field relL2")
    else:
        a2.plot(ep, [r["val_r2"] for r in rows], color="C2", label="val R2")
        a2.set_ylabel("val R2")
        a2.set_title(f"{name}: val R2")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    try:
        fig.savefig(out / f"loss_{name}.png", dpi=110)
    except Exception:
        pass
    plt.close(fig)


def _save_rz_examples(
    out: Path,
    name: str,
    te_idx: np.ndarray,
    Y_phys: np.ndarray,
    pred_z: np.ndarray,
    *,
    y_mean: float,
    y_std: float,
    scales: np.ndarray,
    nshow: int = 3,
) -> None:
    """Example panels in max-normalized Re(δp) space with per-case relL2."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    nshow = min(nshow, len(te_idx))
    if nshow <= 0:
        return
    pred_norm = pred_z[:nshow] * y_std + y_mean
    gt_norm = Y_phys[te_idx[:nshow]]
    rel = _rel_l2_batch(pred_norm, gt_norm)
    fig, axes = plt.subplots(nshow, 3, figsize=(11, 3.4 * nshow))
    if nshow == 1:
        axes = axes[None, :]
    for r in range(nshow):
        yt, yp = gt_norm[r], pred_norm[r]
        vmax = float(np.percentile(np.abs(yt[np.isfinite(yt)]), 98) or 1.0)
        row_lbl = f"case {te_idx[r]}  relL2={rel[r]:.3f}  |δp|_max={scales[te_idx[r]]:.2e}"
        for c, (img, title) in enumerate([(yt, "true"), (yp, "pred"), (yp - yt, "residual")]):
            ax = axes[r, c]
            im = ax.imshow(
                img, origin="lower", aspect="auto", cmap="RdBu_r" if c < 2 else "coolwarm",
                vmin=(-vmax if c < 2 else None), vmax=(vmax if c < 2 else None),
            )
            if c == 0:
                ax.set_ylabel(row_lbl, fontsize=8)
            ax.set_title(title if r == 0 else "")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{name}: max-normalized Re(δp)(R,Z) — direct field metric")
    fig.tight_layout()
    fig.savefig(out / "plots" / f"{name}_rz_examples.png", dpi=110)
    plt.close(fig)


def _interp_profile_to_field(psin_field: np.ndarray, x_psin: Optional[np.ndarray],
                             profile: Optional[np.ndarray], fill: float = 0.0) -> np.ndarray:
    if profile is None or profile.size == 0:
        return np.full_like(psin_field, fill, dtype=np.float32)
    if x_psin is None or x_psin.size != profile.size:
        x_psin = np.linspace(0.0, 1.0, profile.size)
    order = np.argsort(x_psin)
    return np.interp(psin_field.ravel(), x_psin[order], profile[order]).reshape(psin_field.shape).astype(np.float32)


def _resize_2d(arr: np.ndarray, grid: int) -> np.ndarray:
    if arr.shape[0] == grid and arr.shape[1] == grid:
        return arr.astype(np.float32)
    try:
        from scipy.ndimage import zoom
        zr = grid / arr.shape[0]
        zc = grid / arr.shape[1]
        return zoom(arr, (zr, zc), order=1).astype(np.float32)
    except Exception:
        # nearest-neighbor fallback
        yi = (np.linspace(0, arr.shape[0] - 1, grid)).astype(int)
        xi = (np.linspace(0, arr.shape[1] - 1, grid)).astype(int)
        return arr[np.ix_(yi, xi)].astype(np.float32)


def _read_rz_case_complex(path: Path, *, time_idx: int = -1) -> Optional[Dict]:
    """Load complex δp on RZ grid (p_hat or φ₀/φ_q helical combo)."""
    try:
        with h5py.File(path, "r") as f:
            if "runs" not in f:
                return None
            rname = list(f["runs"].keys())[0]
            rg = f["runs"][rname]
            if "pertfields" not in rg:
                return None
            pf = rg["pertfields"]
            z_axis = 0.0
            if "miller" in rg:
                for zk in ("Z0", "z0", "Zaxis"):
                    if zk in rg["miller"]:
                        z_axis = float(rg["miller"][zk][()])
                        break
            if "p_hat" in pf:
                arr = np.asarray(pf["p_hat"])
                field = arr[time_idx] if arr.ndim == 3 else arr
                field = np.asarray(field, np.complex128)
            elif "p_phi0" in pf and "p_phiq" in pf:
                p0 = np.asarray(pf["p_phi0"])
                pq = np.asarray(pf["p_phiq"])
                p0 = p0[time_idx] if p0.ndim == 3 else p0
                pq = pq[time_idx] if pq.ndim == 3 else pq
                field = p0.astype(np.complex128) - 1j * pq.astype(np.complex128)
            else:
                return None
            R = Z = None
            mesh_id = rg.attrs.get("mesh_id", None)
            if mesh_id and "mesh" in f and mesh_id in f["mesh"]:
                mg = f["mesh"][mesh_id]
                if "R" in mg and "Z" in mg:
                    R = np.asarray(mg["R"], float)
                    Z = np.asarray(mg["Z"], float)
            base = _read_rz_case(path, pert_field="p_phi0", time_idx=time_idx)
            if base is None:
                return None
            base["field_complex"] = field
            base["z_axis"] = z_axis
            base["R"] = R if R is not None else base.get("R")
            base["Z"] = Z if Z is not None else base.get("Z")
            return base
    except Exception:
        return None


def _build_rz_net(name: str, in_channels: int, out_channels: int = 1, *,
                  fno_modes: int = 16, fno_hidden: int = 32, grid: int = 128):
    """RZ-local net builder (supports multi-channel output; does not modify shared _build_net)."""
    if name == "fno2d":
        from surge.model.backends.fno2d import _FNO2dNet
        return _FNO2dNet(in_channels, out_channels, hidden_channels=fno_hidden,
                         n_modes=fno_modes, n_layers=4)
    if name == "unet":
        from surge.model.backends.unet import _UNetNet
        return _UNetNet(in_channels, out_channels, base_channels=48, depth=4)
    if name == "deeponet" and out_channels == 1:
        return _build_net(name, in_channels, fno_modes=fno_modes,
                          fno_hidden=fno_hidden, grid=grid)
    return None


def _predict_rz_net(net, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict RZ fields; returns (B,H,W) or (B,2,H,W)."""
    import torch
    dev = next(net.parameters()).device
    net.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(net(Xt[i:i + batch_size].to(dev)).cpu().numpy())
    pred = np.concatenate(out)
    if pred.shape[1] == 1:
        return pred.squeeze(1)
    return pred


def _read_rz_case(path: Path, *, pert_field: str = "p_phi0", time_idx: int = -1) -> Optional[Dict]:
    """Load RZ target + conditioning from one csdata_deltap_b_ver.h5 case."""
    try:
        with h5py.File(path, "r") as f:
            if "runs" not in f:
                return None
            rname = list(f["runs"].keys())[0]
            rg = f["runs"][rname]
            if "pertfields" not in rg or "equilibrium" not in rg:
                return None
            pf = rg["pertfields"]
            key = pert_field
            if key not in pf:
                for alt in (pert_field, "p_phi0", "p", f"{pert_field}_phi0"):
                    if alt in pf:
                        key = alt
                        break
                else:
                    return None
            arr = np.asarray(pf[key])
            field = arr[time_idx] if arr.ndim == 3 else arr
            if np.iscomplexobj(field):
                field = np.real(field).astype(np.float64)
            else:
                field = np.asarray(field, np.float64)
            eq = rg["equilibrium"]
            psi = np.asarray(eq["psi"], float)
            grad = np.asarray(eq["grad_psi_mag"], float) if "grad_psi_mag" in eq else np.hypot(
                np.asarray(eq["dpsi_dR"], float), np.asarray(eq["dpsi_dZ"], float))
            psi_lcfs = float(eq["psi_lcfs"][()]) if "psi_lcfs" in eq else None
            psi_axis = float(np.nanmin(psi))
            denom = (psi_lcfs - psi_axis) if psi_lcfs is not None else None
            if denom is not None and abs(denom) > 1e-12:
                psin = np.clip((psi - psi_axis) / denom, 0.0, 1.05).astype(np.float32)
            else:
                psin = np.zeros_like(psi, dtype=np.float32)

            R = Z = None
            mesh_id = rg.attrs.get("mesh_id", None)
            if mesh_id and "mesh" in f and mesh_id in f["mesh"]:
                mg = f["mesh"][mesh_id]
                if "R" in mg and "Z" in mg:
                    R = np.asarray(mg["R"], float)
                    Z = np.asarray(mg["Z"], float)

            sh = extract_equilibrium_scalars(rg)
            n_val, pscale, bscale = 0.0, 1.0, 1.0
            if "parset" in rg:
                names = rg["parset"]["names"]
                vals = rg["parset"]["values"]
                for i, nm in enumerate(names):
                    nm = _decode(nm)
                    if i < len(vals):
                        if nm == "ntor":
                            n_val = float(vals[i])
                        elif nm == "pscale":
                            pscale = float(vals[i])
                        elif nm == "batemanscale":
                            bscale = float(vals[i])
            sh.setdefault("pscale", pscale)
            sh.setdefault("batemanscale", bscale)
            sh["ntor"] = n_val

            qprof = qpsin = pprof = ppsin = None
            if "flux_average" in rg:
                fa = rg["flux_average"]
                if "q" in fa and "profile" in fa["q"]:
                    qprof = np.asarray(fa["q"]["profile"], float).ravel()
                    qpsin = np.asarray(fa["q"]["psin"], float).ravel() if "psin" in fa["q"] else None
                if "p" in fa and "profile" in fa["p"]:
                    pprof = np.asarray(fa["p"]["profile"], float).ravel()
                    ppsin = np.asarray(fa["p"]["psin"], float).ravel() if "psin" in fa["p"] else None

            return {
                "run_id": _decode(rg.get("runID", rname)),
                "eq_id": _decode(rg.get("eqID", "eq")),
                "field": field,
                "psin": psin,
                "grad": grad.astype(np.float32),
                "psi_lcfs": psi_lcfs,
                "psi_axis": psi_axis,
                "R": R,
                "Z": Z,
                "shaping": sh,
                "qprof": qprof,
                "qpsin": qpsin,
                "pprof": pprof,
                "ppsin": ppsin,
            }
    except Exception:
        return None


def build_rz_dataset(
    batch_dir: str,
    filename: str,
    n_cases: Optional[int],
    grid: int,
    *,
    pert_field: str = "p_phi0",
    time_idx: int = -1,
    exclude_keys: Optional[set] = None,
    target_floor: Optional[float] = None,
    target_smooth: Optional[float] = None,
    gauge_fix: bool = False,
    complex_target: bool = False,
    midplane_z: str = "axis",
    gauge_ref: str = "peak",
    peak_window: int = 3,
    shaping_keys: Tuple[str, ...] = (
        "kappa", "delta", "epsilon", "pscale", "batemanscale", "ntor",
        "q0", "q95", "qmin", "p0",
    ),
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    """Return X, Y (max-norm), keys, per-case δp scales, and ψ_N grids for loc loss."""
    paths = find_complex_v2_files(batch_dir, filename=filename)
    if n_cases:
        paths = paths[:n_cases]
    mode = pert_field
    if gauge_fix or complex_target:
        mode = "complex+gauge" if gauge_fix else "complex"
    print(f"Building RZ-field dataset from {len(paths)} cases (grid={grid}x{grid}, "
          f"target={mode})")
    chan_names = [
        "psin", "grad_psi", "lcfs_prox", "q_on_rz", "p_on_rz", "R_norm", "Z_norm",
        "inside_mask", *shaping_keys,
    ]
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    scales: List[float] = []
    psin_grids: List[np.ndarray] = []
    keys: List[str] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        if gauge_fix or complex_target:
            c = _read_rz_case_complex(Path(p), time_idx=time_idx)
        else:
            c = _read_rz_case(Path(p), pert_field=pert_field, time_idx=time_idx)
        if c is None:
            continue
        key = f"{c['run_id']}_{c['eq_id']}"
        if exclude_keys and key in exclude_keys:
            continue

        if gauge_fix or complex_target:
            fc = _sanitize_field(np.asarray(c["field_complex"], np.complex128))
            scale = float(np.nanmax(np.abs(fc))) or 1.0
            fn = fc / scale
            if gauge_fix:
                gf = gauge_fix_field(
                    fn, Z=c.get("Z"), z_axis=float(c.get("z_axis", 0.0)),
                    midplane_z=midplane_z, gauge_ref=gauge_ref, peak_window=peak_window,
                )
                fn = gf.field_gf
            if complex_target:
                y = np.stack([
                    _apply_target_floor(_resize_2d(np.real(fn).astype(np.float32), grid), target_floor),
                    _apply_target_floor(_resize_2d(np.imag(fn).astype(np.float32), grid), target_floor),
                ], axis=0)
            else:
                y = _max_norm_field(np.real(fn).astype(np.float32))
                y = _resize_2d(y, grid)
                y = _apply_target_floor(y, target_floor)
            field_scale = scale
        else:
            field = c["field"]
            scale = float(np.nanmax(np.abs(field))) or 1.0
            y = _max_norm_field((field / scale).astype(np.float32))
            y = _resize_2d(y, grid)
            y = _apply_target_floor(y, target_floor)
            field_scale = scale

        if target_smooth is not None and target_smooth > 0:
            from scipy.ndimage import gaussian_filter
            if y.ndim == 3:
                y = np.stack([gaussian_filter(y[ch], sigma=float(target_smooth)).astype(np.float32)
                              for ch in range(y.shape[0])], axis=0)
            else:
                y = gaussian_filter(y, sigma=float(target_smooth)).astype(np.float32)

        psin = _resize_2d(_finite_array(c["psin"]), grid)
        psin = _max_norm_field(psin)
        grad = _max_norm_field(_resize_2d(_finite_array(c["grad"]), grid))
        lcfs_prox = np.clip(1.0 - psin, 0.0, 1.0).astype(np.float32)
        q_on = _max_norm_field(
            _resize_2d(_interp_profile_to_field(psin, c.get("qpsin"), c.get("qprof")), grid))
        p_on = _max_norm_field(
            _resize_2d(_interp_profile_to_field(psin, c.get("ppsin"), c.get("pprof")), grid))

        if c["R"] is not None and c["Z"] is not None:
            R = _resize_2d(c["R"], grid)
            Z = _resize_2d(c["Z"], grid)
            r0, r1 = float(np.nanmin(R)), float(np.nanmax(R))
            z0, z1 = float(np.nanmin(Z)), float(np.nanmax(Z))
            R_n = ((R - r0) / (r1 - r0 + 1e-8)).astype(np.float32)
            Z_n = ((Z - z0) / (z1 - z0 + 1e-8)).astype(np.float32)
        else:
            R_n = np.linspace(0, 1, grid, dtype=np.float32)[:, None] * np.ones((grid, grid), np.float32)
            Z_n = np.linspace(0, 1, grid, dtype=np.float32)[None, :] * np.ones((grid, grid), np.float32)

        inside = np.isfinite(psin)
        if c["psi_lcfs"] is not None:
            psi_r = _resize_2d(
                np.where(np.isfinite(c["psin"]), c["psin"], np.nan), grid)
            # psin already 0..1; inside = psin < 1
            inside = np.isfinite(psi_r) & (psi_r < 1.0)
        inside = inside.astype(np.float32)

        sh = c["shaping"]
        const_layers = [
            np.full((grid, grid), float(sh.get(k, 0.0) or 0.0), dtype=np.float32) for k in shaping_keys
        ]
        layers = [_finite_array(x) for x in (psin, grad, lcfs_prox, q_on, p_on, R_n, Z_n, inside, *const_layers)]
        X = np.stack(layers, axis=0)
        Xs.append(X)
        Ys.append(y)
        scales.append(field_scale)
        psin_grids.append(psin.astype(np.float32))
        keys.append(key)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    X = np.stack(Xs)
    Y = np.stack(Ys)
    Psin = np.stack(psin_grids)
    print(f"  Built X={X.shape} Y={Y.shape} in {time.time()-t0:.0f}s")
    return X, Y, chan_names, keys, np.asarray(scales, np.float32), Psin


def _rel_l2_batch(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-case max-normalized relL2 on RZ fields."""
    out = []
    for i in range(len(pred)):
        p = pred[i].ravel()
        g = gt[i].ravel()
        if mask is not None:
            m = mask[i].ravel() > 0.5
            p, g = p[m], g[m]
        denom = float(np.linalg.norm(g))
        out.append(float(np.linalg.norm(p - g) / denom) if denom > 1e-30 else np.nan)
    return np.asarray(out, float)


def _peak_psin(y: np.ndarray, psin: np.ndarray) -> np.ndarray:
    """ψ_N of the global-|field| peak (hard argmax), for logging."""
    N = y.shape[0]
    flat = np.abs(y).reshape(N, -1)
    col = np.argmax(flat, axis=1)
    return psin.reshape(N, -1)[np.arange(N), col]


def _train_rz_net(
    net,
    name: str,
    out: Path,
    Xtr, Ytr, Xva, Yva,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    gpu_cache: bool = True,
    resume: Optional[str] = None,
    peak_weight: float = 0.0,
    peak_pow: float = 1.0,
    grad_weight: float = 0.0,
    loc_weight: float = 0.0,
    marg_weight: float = 0.0,
    loc_beta: float = 8.0,
    time_budget_min: float = 0.0,
    select_by: str = "field",
    Yva_phys: Optional[np.ndarray] = None,
    psin_tr: Optional[np.ndarray] = None,
    psin_va: Optional[np.ndarray] = None,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    phase_align_eval: bool = False,
    complex_target: bool = False,
) -> Tuple[object, int]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = gpu_cache and dev.type == "cuda"
    net = net.to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    _mse = torch.nn.MSELoss()

    use_loc = loc_weight and loc_weight > 0 and psin_tr is not None
    use_marg = marg_weight and marg_weight > 0

    def _pixel_loss(pred, target):
        if peak_weight and peak_weight > 0:
            with torch.no_grad():
                s = target.abs()
                s = s / (s.amax(dim=(2, 3), keepdim=True) + 1e-8)
                w = 1.0 + peak_weight * s.pow(peak_pow)
            return (w * (pred - target) ** 2).mean()
        return _mse(pred, target)

    def _grad_loss(pred, target):
        dpx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dtx = target[:, :, :, 1:] - target[:, :, :, :-1]
        dpy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dty = target[:, :, 1:, :] - target[:, :, :-1, :]
        return _mse(dpx, dtx) + _mse(dpy, dty)

    psin_tr_t = None
    if use_loc:
        psin_tr_t = torch.tensor(psin_tr, dtype=torch.float32,
                                 device=dev if cache else "cpu")

    def _psi_softloc(z, psin_batch):
        B = z.shape[0]
        if z.shape[1] > 1:
            zf = z.abs().sum(dim=1).reshape(B, -1)
        else:
            zf = z.abs().reshape(B, -1)
        zf = zf - zf.amax(dim=1, keepdim=True)
        p = torch.softmax(loc_beta * zf, dim=1)
        pf = psin_batch.reshape(B, -1)
        return (p * pf).sum(dim=1)

    def lossf(pred, target, psin_batch=None):
        loss = _pixel_loss(pred, target)
        if use_loc and psin_batch is not None:
            lp = _psi_softloc(pred, psin_batch)
            with torch.no_grad():
                lt = _psi_softloc(target, psin_batch)
            loss = loss + loc_weight * ((lp - lt) ** 2).mean()
        if use_marg:
            loss = loss + marg_weight * (
                _mse(pred.mean(dim=2), target.mean(dim=2))
                + _mse(pred.mean(dim=3), target.mean(dim=3))
            )
        if grad_weight > 0:
            loss = loss + grad_weight * _grad_loss(pred, target)
        return loss

    _terms = ["MSE" if not (peak_weight and peak_weight > 0)
              else f"peakMSE(a={peak_weight},p={peak_pow})"]
    if use_loc:
        _terms.append(f"loc(w={loc_weight},beta={loc_beta})")
    if use_marg:
        _terms.append(f"marg(w={marg_weight})")
    if grad_weight > 0:
        _terms.append(f"grad(w={grad_weight})")
    print(f"  [loss] composite = {' + '.join(_terms)}", flush=True)
    print(f"  [select] checkpoint criterion = {select_by}", flush=True)

    n_train = len(Xtr)
    if cache:
        Xg = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        if Ytr.ndim == 4 and Ytr.shape[1] == 2:
            Yg = torch.tensor(Ytr, dtype=torch.float32, device=dev)
        else:
            Yg = torch.tensor(Ytr[:, None], dtype=torch.float32, device=dev)
    else:
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        if Ytr.ndim == 4 and Ytr.shape[1] == 2:
            Yt = torch.tensor(Ytr, dtype=torch.float32)
            loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)
        else:
            Yt = torch.tensor(Ytr[:, None], dtype=torch.float32)
            loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)

    Xv = torch.tensor(Xva, dtype=torch.float32).to(dev)
    if Yva.ndim == 4 and Yva.shape[1] == 2:
        Yv_t = torch.tensor(Yva, dtype=torch.float32).to(dev)
    else:
        Yv_t = torch.tensor(Yva[:, None], dtype=torch.float32).to(dev)
    yv_np = Yva
    psin_va_t = None
    if use_loc and psin_va is not None:
        psin_va_t = torch.tensor(psin_va, dtype=torch.float32, device=dev)

    hist_path = out / f"history_{name}.jsonl"
    best_path = out / f"ckpt_{name}.pt"
    last_path = out / f"ckpt_{name}_last.pt"
    start_epoch = 0
    best_val = float("inf")
    best_rel = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    if resume and Path(resume).is_file():
        import torch
        st = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(st["state_dict"])
        if "optimizer" in st:
            opt.load_state_dict(st["optimizer"])
        start_epoch = int(st.get("epoch", 0))
        best_val = float(st.get("best_val", best_val))
        best_rel = float(st.get("best_relL2_median", best_rel))
        best_epoch = int(st.get("best_epoch", start_epoch))
        no_improve = int(st.get("no_improve", 0))
        if "best_state_dict" in st:
            best_state = st["best_state_dict"]
        print(f"  resumed from {resume} @ epoch {start_epoch}", flush=True)

    train_wall_start = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        net.train()
        tl = 0.0
        if cache:
            perm = torch.randperm(n_train, device=dev)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                pt = net(Xg[idx])
                psin_b = psin_tr_t[idx] if use_loc else None
                loss = lossf(pt, Yg[idx], psin_b)
                loss.backward()
                opt.step()
                tl += loss.item() * len(idx)
        else:
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                pt = net(xb)
                loss = lossf(pt, yb)
                loss.backward()
                opt.step()
                tl += loss.item() * len(xb)
        tl /= n_train

        net.eval()
        with torch.no_grad():
            vp = []
            vcomp = 0.0
            for i in range(0, len(Xv), batch_size):
                sl = slice(i, i + batch_size)
                pt = net(Xv[sl])
                yb = Yv_t[sl]
                psin_b = psin_va_t[sl] if use_loc and psin_va_t is not None else None
                vcomp += float(lossf(pt, yb, psin_b).item()) * pt.shape[0]
                vp.append(pt.cpu().numpy())
            vp = np.concatenate(vp)
            if complex_target and vp.ndim == 4 and vp.shape[1] == 2:
                vl = float(np.mean((vp - yv_np) ** 2))
                vr2 = r2(yv_np.reshape(yv_np.shape[0], -1), vp.reshape(vp.shape[0], -1))
                pred_norm = vp * y_std + y_mean
            else:
                vp_sq = vp.squeeze(1) if vp.ndim == 4 else vp
                vl = float(np.mean((vp_sq - yv_np) ** 2))
                vr2 = r2(yv_np, vp_sq)
                pred_norm = vp_sq * y_std + y_mean
            vcomp /= len(Xv)
            val_rel = None
            val_rel_aligned = None
            vdpsi = None
            if Yva_phys is not None:
                if complex_target and Yva_phys.ndim == 4:
                    val_rel, val_rel_aligned = rel_l2_complex_batch(
                        pred_norm, Yva_phys, phase_align=True)
                    if not phase_align_eval:
                        val_rel_aligned = None
                else:
                    val_rel = _rel_l2_batch(pred_norm, Yva_phys)
            if psin_va is not None and Yva_phys is not None:
                phys_for_peak = (Yva_phys[:, 0] if complex_target and Yva_phys.ndim == 4
                                 else Yva_phys)
                pred_for_peak = (pred_norm[:, 0] if complex_target and pred_norm.ndim == 4
                                 else pred_norm)
                vdpsi = float(np.mean(np.abs(
                    _peak_psin(pred_for_peak, psin_va) - _peak_psin(phys_for_peak, psin_va)
                )))

        if select_by == "field" and val_rel is not None:
            sel = float(np.nanmedian(
                val_rel_aligned if phase_align_eval and val_rel_aligned is not None else val_rel
            ))
            improved = sel < best_rel
            if improved:
                best_rel = sel
        elif select_by == "composite":
            sel = vcomp
            improved = sel < best_val
            if improved:
                best_val = sel
        else:
            sel = vl
            improved = sel < best_val
            if improved:
                best_val = sel

        if improved:
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "state_dict": best_state,
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "val_loss": vl,
                "val_r2": vr2,
                "best_val": best_val,
                "best_relL2_median": best_rel,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "select_by": select_by,
                "model": name,
            }, best_path)
        else:
            no_improve += 1

        torch.save({
            "state_dict": net.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "val_loss": vl,
            "val_r2": vr2,
            "best_val": best_val,
            "best_relL2_median": best_rel,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "select_by": select_by,
            "model": name,
        }, last_path)

        hist_row = {
            "epoch": epoch, "train_loss": tl, "val_loss": vl, "val_r2": vr2,
            "val_comp": vcomp,
        }
        if vdpsi is not None:
            hist_row["val_dpsi"] = vdpsi
        if val_rel is not None:
            hist_row["val_relL2_mean"] = float(np.nanmean(val_rel))
            hist_row["val_relL2_median"] = float(np.nanmedian(val_rel))
            hist_row["val_frac_relL2_gt_1"] = float(np.mean(val_rel > 1.0))
            if val_rel_aligned is not None:
                hist_row["val_relL2_aligned_mean"] = float(np.nanmean(val_rel_aligned))
                hist_row["val_relL2_aligned_median"] = float(np.nanmedian(val_rel_aligned))
                hist_row["val_frac_relL2_aligned_gt_1"] = float(np.mean(val_rel_aligned > 1.0))
        with hist_path.open("a") as fh:
            fh.write(json.dumps(hist_row) + "\n")

        if epoch % 5 == 0 or improved or epoch == 1:
            _rz_loss_plot(hist_path, name, out)
        if epoch % 10 == 0 or epoch == 1:
            tag = "  *best" if improved else ""
            rel_msg = ""
            if val_rel is not None:
                rel_msg = (f" relL2_med={hist_row['val_relL2_median']:.3f}"
                           f" frac>1={hist_row['val_frac_relL2_gt_1']:.3f}")
            dpsi_msg = f" dpsi={vdpsi:.4f}" if vdpsi is not None else ""
            print(f"  [{name}] epoch {epoch}/{epochs} train={tl:.4f} val={vl:.4f} "
                  f"val_r2={vr2:.4f}{rel_msg}{dpsi_msg}{tag}", flush=True)

        if patience > 0 and no_improve >= patience:
            print(f"  [{name}] early stop @ epoch {epoch} (best {best_epoch})", flush=True)
            break
        if time_budget_min > 0 and (time.time() - train_wall_start) / 60.0 >= time_budget_min:
            print(f"  [{name}] time budget {time_budget_min:.0f} min reached", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    _rz_loss_plot(hist_path, name, out)
    return net, n_params


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FNO/U-Net on Re(δp)(R,Z) directly.")
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--pert-field", default="p_phi0",
                    help="pertfields dataset key (default p_phi0 = Re δp at φ₀)")
    ap.add_argument("--time-idx", type=int, default=-1)
    ap.add_argument("--n-cases", type=int, default=0)
    ap.add_argument("--grid", type=int, default=201)
    ap.add_argument("--models", nargs="+", default=["fno2d"])
    ap.add_argument("--fno-modes", type=int, default=64)
    ap.add_argument("--fno-hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--peak-weight", type=float, default=4.0,
                    help="Up-weight high-|δp| pixels (peak amplitude).")
    ap.add_argument("--peak-pow", type=float, default=1.0)
    ap.add_argument("--grad-weight", type=float, default=0.0)
    ap.add_argument("--loc-weight", type=float, default=2.0,
                    help="Soft-argmax ψ_N peak-location loss (core vs edge).")
    ap.add_argument("--marg-weight", type=float, default=1.0,
                    help="R- and Z-marginal profile MSE.")
    ap.add_argument("--loc-beta", type=float, default=8.0,
                    help="Softmax temperature for peak locator.")
    ap.add_argument("--target-floor", type=float, default=6.0,
                    help="Zero |δp|/max pixels below 10^-floor dex (noise floor).")
    ap.add_argument("--target-smooth", type=float, default=0.0,
                    help="Gaussian σ on max-norm target (0 = off, like D′ smooth0).")
    ap.add_argument("--gauge-fix", action="store_true",
                    help="Gauge-fix target by exp(−iθ_ref) at midplane peak (default OFF).")
    ap.add_argument("--complex-target", action="store_true",
                    help="Two-channel Re/Im target (default OFF; use with --gauge-fix).")
    ap.add_argument("--midplane-z", choices=["axis", "zero"], default="axis",
                    help="Midplane row: magnetic-axis Z or Z=0.")
    ap.add_argument("--gauge-ref", choices=["peak", "template"], default="peak",
                    help="Reference phase: local peak neighborhood or template overlap.")
    ap.add_argument("--peak-window", type=int, default=3,
                    help="Half-width (pixels) for amplitude-weighted θ_ref neighborhood.")
    ap.add_argument("--no-target-zscore", action="store_true",
                    help="Train in max-norm space without global Y z-score (default OFF).")
    ap.add_argument("--phase-align-eval", action="store_true",
                    help="Report relL2 after optimal global phase alignment (default OFF).")
    ap.add_argument("--select-by", choices=["mse", "composite", "field"], default="field",
                    help="Checkpoint criterion: z-scored MSE, composite loss, or val relL2.")
    ap.add_argument("--time-budget-min", type=float, default=210.0)
    ap.add_argument("--exclude-list", default="runs/quarantine/bad_cases.json")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-gpu-cache", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/rz_field_fno48_re_deltap")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    exclude_keys = None
    if args.exclude_list and Path(args.exclude_list).is_file():
        raw = Path(args.exclude_list).read_text().strip()
        try:
            exclude_keys = set(json.loads(raw).keys())
        except Exception:
            exclude_keys = {ln.strip() for ln in raw.splitlines() if ln.strip()}
        print(f"Excluding {len(exclude_keys)} quarantined cases")

    n_cases = args.n_cases if args.n_cases > 0 else None
    X, Y, chan_names, keys, scales, psin_grids = build_rz_dataset(
        args.batch_dir, args.filename, n_cases, args.grid,
        pert_field=args.pert_field, time_idx=args.time_idx, exclude_keys=exclude_keys,
        target_floor=args.target_floor, target_smooth=args.target_smooth,
        gauge_fix=args.gauge_fix, complex_target=args.complex_target,
        midplane_z=args.midplane_z, gauge_ref=args.gauge_ref,
        peak_window=args.peak_window,
    )
    N = X.shape[0]
    floor_s = f", floor -{args.target_floor:g}dex" if args.target_floor else ""
    smooth_s = f", smooth σ={args.target_smooth}" if args.target_smooth else ""
    gauge_s = ", gauge-fixed" if args.gauge_fix else ""
    complex_s = ", complex Re/Im" if args.complex_target else ""
    zscore_s = "" if args.no_target_zscore else ", global z-score"
    target_desc = (f"max-normalized RZ δp ({args.pert_field}){gauge_s}{complex_s}"
                   f"{floor_s}{smooth_s}, per-case max-norm inputs{zscore_s}")

    (out / "run_config.json").write_text(json.dumps({
        "batch_dir": args.batch_dir,
        "filename": args.filename,
        "pert_field": args.pert_field,
        "grid": args.grid,
        "models": list(args.models),
        "target": target_desc,
        "target_floor": args.target_floor,
        "target_smooth": args.target_smooth,
        "gauge_fix": args.gauge_fix,
        "complex_target": args.complex_target,
        "midplane_z": args.midplane_z,
        "gauge_ref": args.gauge_ref,
        "peak_window": args.peak_window,
        "no_target_zscore": args.no_target_zscore,
        "phase_align_eval": args.phase_align_eval,
        "fno_modes": args.fno_modes,
        "fno_hidden": args.fno_hidden,
        "peak_weight": args.peak_weight,
        "peak_pow": args.peak_pow,
        "loc_weight": args.loc_weight,
        "marg_weight": args.marg_weight,
        "loc_beta": args.loc_beta,
        "select_by": args.select_by,
        "channels": chan_names,
    }, indent=2))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_test = int(args.test_frac * N)
    n_val = int(args.val_frac * N)
    te = perm[:n_test]
    va = perm[n_test:n_test + n_val]
    tr = perm[n_test + n_val:]
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    keys_arr = np.asarray(keys)
    (out / "splits.json").write_text(json.dumps({
        "seed": args.seed,
        "train_idx": tr.tolist(), "val_idx": va.tolist(), "test_idx": te.tolist(),
        "train_keys": keys_arr[tr].tolist(),
        "val_keys": keys_arr[va].tolist(),
        "test_keys": keys_arr[te].tolist(),
    }, indent=2))

    xm = np.nanmean(X[tr], axis=(0, 2, 3), keepdims=True)
    xs = np.nanstd(X[tr], axis=(0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs
    if args.no_target_zscore:
        ym, ysd = 0.0, 1.0
        Yn = Y.astype(np.float32)
    else:
        if args.complex_target and Y.ndim == 4:
            ym = float(np.nanmean(Y[tr]))
            ysd = float(np.nanstd(Y[tr]) + 1e-8)
        else:
            ym = float(np.nanmean(Y[tr]))
            ysd = float(np.nanstd(Y[tr]) + 1e-8)
        Yn = (Y - ym) / ysd
    if not args.no_target_zscore and (not np.isfinite(ym) or not np.isfinite(ysd) or ysd < 1e-12):
        raise RuntimeError(
            f"Invalid target normalization (y_mean={ym}, y_std={ysd}). "
            "Check for empty/NaN targets in the training split."
        )
    (out / "norm_stats.json").write_text(json.dumps({
        "y_mean": ym, "y_std": ysd,
        "input_mean": xm.squeeze().tolist(),
        "input_std": xs.squeeze().tolist(),
    }, indent=2))

    results: Dict[str, Dict] = {}
    out_channels = 2 if args.complex_target else 1
    for name in args.models:
        print(f"\n=== Training {name} (RZ δp, out_ch={out_channels}) ===")
        t0 = time.time()
        net = _build_rz_net(name, X.shape[1], out_channels=out_channels,
                            fno_modes=args.fno_modes, fno_hidden=args.fno_hidden,
                            grid=args.grid)
        if net is None:
            print(f"  unknown model {name}; skip")
            continue
        resume_path = args.resume
        if resume_path and Path(resume_path).is_file():
            try:
                bad_hist = [json.loads(l) for l in (out / f"history_{name}.jsonl").read_text().splitlines() if l.strip()]
                if bad_hist and not np.isfinite(bad_hist[-1].get("train_loss", 0.0)):
                    print(f"  ignoring corrupt resume checkpoint ({resume_path})", flush=True)
                    resume_path = None
            except Exception:
                pass
        net, n_params = _train_rz_net(
            net, name, out, Xn[tr], Yn[tr], Xn[va], Yn[va],
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, gpu_cache=not args.no_gpu_cache,
            resume=resume_path, peak_weight=args.peak_weight, peak_pow=args.peak_pow,
            grad_weight=args.grad_weight, loc_weight=args.loc_weight,
            marg_weight=args.marg_weight, loc_beta=args.loc_beta,
            time_budget_min=args.time_budget_min, select_by=args.select_by,
            Yva_phys=Y[va], psin_tr=psin_grids[tr], psin_va=psin_grids[va],
            y_mean=ym, y_std=ysd,
            phase_align_eval=args.phase_align_eval,
            complex_target=args.complex_target,
        )
        pred_z = _predict_rz_net(net, Xn[te], args.batch_size)
        yt_z = Yn[te]
        if args.complex_target and pred_z.ndim == 4:
            pred_norm = pred_z * ysd + ym
            gt_norm = Y[te]
            rel_raw, rel_aligned = rel_l2_complex_batch(pred_norm, gt_norm, phase_align=True)
            res = {
                "test_r2_global": r2(yt_z.reshape(len(te), -1), pred_z.reshape(len(te), -1)),
                "test_pattern_r2": pattern_r2(yt_z[:, 0], pred_z[:, 0]),
                "test_relL2_mean": float(np.nanmean(rel_raw)),
                "test_relL2_median": float(np.nanmedian(rel_raw)),
                "test_relL2_p90": float(np.nanpercentile(rel_raw, 90)),
                "test_frac_relL2_gt_1": float(np.mean(rel_raw > 1.0)),
                "test_relL2_aligned_mean": float(np.nanmean(rel_aligned)),
                "test_relL2_aligned_median": float(np.nanmedian(rel_aligned)),
                "test_relL2_aligned_p90": float(np.nanpercentile(rel_aligned, 90)),
                "test_frac_relL2_aligned_gt_1": float(np.mean(rel_aligned > 1.0)),
                "train_seconds": time.time() - t0,
                "n_params": n_params,
                "checkpoint": str(out / f"ckpt_{name}.pt"),
            }
            print(f"  {name}: test R2={res['test_r2_global']:.4f} "
                  f"relL2 raw med={res['test_relL2_median']:.3f} "
                  f"aligned med={res['test_relL2_aligned_median']:.3f} "
                  f"frac>1 raw={res['test_frac_relL2_gt_1']:.3f} "
                  f"aligned={res['test_frac_relL2_aligned_gt_1']:.3f}")
        else:
            pred_norm = pred_z * ysd + ym
            gt_norm = Y[te]
            pred_phys = pred_norm * scales[te, None, None]
            gt_phys = gt_norm * scales[te, None, None]
            rel = _rel_l2_batch(pred_phys, gt_phys)
            res = {
                "test_r2_global": r2(yt_z, pred_z),
                "test_pattern_r2": pattern_r2(yt_z, pred_z),
                "test_relL2_mean": float(np.nanmean(rel)),
                "test_relL2_median": float(np.nanmedian(rel)),
                "test_relL2_p90": float(np.nanpercentile(rel, 90)),
                "test_frac_relL2_gt_1": float(np.mean(rel > 1.0)),
                "train_seconds": time.time() - t0,
                "n_params": n_params,
                "checkpoint": str(out / f"ckpt_{name}.pt"),
            }
            print(f"  {name}: test R2={res['test_r2_global']:.4f} patR2={res['test_pattern_r2']:.4f} "
                  f"relL2 med={res['test_relL2_median']:.3f} mean={res['test_relL2_mean']:.3f} "
                  f"frac>1={res['test_frac_relL2_gt_1']:.3f}")
        results[name] = res

        y_ex = Y if Y.ndim == 3 else Y[:, 0]
        pred_ex = pred_z if pred_z.ndim == 3 else pred_z[:, 0]
        _save_rz_examples(
            out, name, te, y_ex, pred_ex, y_mean=ym, y_std=ysd, scales=scales,
        )

    (out / "rz_field_metrics.json").write_text(json.dumps({
        "n_cases": N, "grid": args.grid, "channels": chan_names,
        "target": target_desc, "target_floor": args.target_floor,
        "target_smooth": args.target_smooth,
        "gauge_fix": args.gauge_fix,
        "complex_target": args.complex_target,
        "no_target_zscore": args.no_target_zscore,
        "phase_align_eval": args.phase_align_eval,
        "select_by": args.select_by,
        "loc_weight": args.loc_weight, "marg_weight": args.marg_weight,
        "y_mean": ym, "y_std": ysd, "results": results,
    }, indent=2))
    print(f"\nWrote {out / 'rz_field_metrics.json'}")


if __name__ == "__main__":
    main()
