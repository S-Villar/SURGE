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
from typing import Dict, List, Optional, Tuple

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


def build_dataset(
    batch_dir: str, filename: str, n_cases: Optional[int], grid: int,
    m_lo: float, m_hi: float, spectrum_field: str, eps: float,
    shaping_keys: Tuple[str, ...] = ("kappa", "delta", "epsilon", "pscale", "batemanscale"),
    target_norm: str = "none", target_space: str = "log10",
    return_paths: bool = False,
):
    """Return X (N,C,H,W), Y (N,H,W) target, channel names, case keys.

    target_norm : {"none", "max"}
        "max" divides each case's magnitude spectrum by its own peak so the max
        amplitude becomes 1 BEFORE any log -- this factors out the arbitrary
        per-case eigenmode normalization and leaves the model to learn the shape.
    target_space : {"log10", "raw"}
        "log10" -> target is log10(mag[+norm] + eps); "raw" -> target is the
        (optionally max-normalized) magnitude itself.
    """
    paths = find_complex_v2_files(batch_dir, filename=filename)
    if n_cases:
        paths = paths[:n_cases]
    print(f"Building spectrum-image dataset from {len(paths)} cases "
          f"(grid={grid}x{grid}, m in [{m_lo},{m_hi}])")
    psi_grid = np.linspace(0.0, 1.0, grid)
    m_grid = np.linspace(m_lo, m_hi, grid)
    M = np.repeat(m_grid[:, None], grid, axis=1)          # (H,W) m varies along rows
    PSI = np.repeat(psi_grid[None, :], grid, axis=0)      # (H,W) psi varies along cols
    chan_names = ["psi", "m", "q", "p", "res(m-nq)", "prox", *shaping_keys]
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    keys: List[str] = []
    kept: List[str] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        c = _read_case(Path(p), spectrum_field)
        if c is None:
            continue
        mag, m_modes, psi = c["mag"], c["m_modes"], c["psi"]
        sel = (m_modes >= m_lo) & (m_modes <= m_hi)
        if sel.sum() < 4:
            continue
        field = mag[sel, :]                                # (nmc, npsi), >= 0
        # Per-case magnitude normalization: peak -> 1 (before any log).
        if target_norm == "max":
            cmax = float(field.max())
            if cmax > 0:
                field = field / cmax
        if target_space == "log10":
            field = np.log10(field + eps)
        # else: "raw" -> keep (optionally max-normalized) magnitude
        m_vals = m_modes[sel]
        # interp along psi (cols) onto uniform psi_grid
        tmp = np.vstack([_interp_to(psi_grid, psi, row) for row in field])  # (nmc,W)
        # interp along m (rows) onto uniform m_grid
        img = np.vstack([_interp_to(m_grid, m_vals, tmp[:, j]) for j in range(grid)]).T  # (H,W)
        q_on = _interp_to(psi_grid, c["qpsin"], c["qprof"])
        p_on = _interp_to(psi_grid, c["ppsin"], c["pprof"])
        Q = np.repeat(q_on[None, :], grid, axis=0)        # (H,W) q(psi_j)
        P = np.repeat(p_on[None, :], grid, axis=0)
        n_val = c["n"]
        RES = M - n_val * Q                                # resonance detuning
        PROX = 1.0 / (1.0 + RES ** 2)                      # ridge proximity
        sh = c["shaping"]
        const = [np.full((grid, grid), float(sh.get(k, 0.0))) for k in shaping_keys]
        X = np.stack([PSI, M, Q, P, RES, PROX, *const], axis=0).astype(np.float32)  # (C,H,W)
        Xs.append(X); Ys.append(img.astype(np.float32)); keys.append(f"{c['run_id']}_{c['eq_id']}")
        kept.append(str(p))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    X = np.stack(Xs); Y = np.stack(Ys)
    print(f"  Built X={X.shape} Y={Y.shape} in {time.time()-t0:.0f}s")
    if return_paths:
        return X, Y, chan_names, keys, kept
    return X, Y, chan_names, keys


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
               lr_schedule: str = "none", lr_min: float = 0.0):
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
    net = net.to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Loss: plain MSE weights every pixel equally, so the ~90% noise-floor pixels
    # dominate and the sharp high-amplitude ridge/peak is under-fit (its location
    # and peak amplitude come out wrong). With peak_weight>0 we up-weight pixels by
    # the ground-truth amplitude (per-image min-max ranked, so the peak -> 1),
    # forcing the model to reproduce the peak/ridge accurately.
    _mse = torch.nn.MSELoss()

    # --- pixel term (plain or amplitude-weighted MSE) ---------------------- #
    def _pixel_loss(pred, target):
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
    _psi_map = None  # lazily built (H*W,) psi_N coordinate for the soft-argmax

    def _psi_softloc(z, psi_flat):
        B = z.shape[0]
        zf = z.reshape(B, -1)
        zf = zf - zf.amax(dim=1, keepdim=True)
        p = torch.softmax(loc_beta * zf, dim=1)
        return (p * psi_flat).sum(dim=1)             # (B,) expected psi_N of peak

    def lossf(pred, target):
        nonlocal _psi_map
        loss = _pixel_loss(pred, target)
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
        return loss

    _terms = ["MSE" if not (peak_weight and peak_weight > 0)
              else f"peakMSE(a={peak_weight},p={peak_pow})"]
    if use_loc:
        _terms.append(f"loc(w={loc_weight},beta={loc_beta})")
    if use_marg:
        _terms.append(f"marg(w={marg_weight})")
    print(f"  [loss] composite = {' + '.join(_terms)}", flush=True)
    cache = gpu_cache and dev.type == "cuda"
    n_train = len(Xtr)
    if cache:
        Xg = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        Yg = torch.tensor(Ytr[:, None], dtype=torch.float32, device=dev)
    else:
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        Yt = torch.tensor(Ytr[:, None], dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(dev)
    hist_path = out / f"history_{name}.jsonl"
    ckpt_path = out / f"ckpt_{name}.pt"
    last_path = out / f"ckpt_{name}_last.pt"
    best_val = float("inf"); best_state = None; no_improve = 0
    start_epoch = 0
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
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"  [resume] {rp} -> start at epoch {start_epoch+1}, "
              f"best_val={best_val:.5f}", flush=True)
    if not resume:
        hist_path.write_text("")  # truncate only for a fresh run
    yv_np = Yva[:, None]

    def _save(path, epoch, vl, vr2):
        torch.save({"state_dict": {k: v.detach().cpu().clone()
                                   for k, v in net.state_dict().items()},
                    "optimizer": opt.state_dict(), "epoch": epoch,
                    "val_loss": vl, "val_r2": vr2, "best_val": best_val,
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

    for epoch in range(start_epoch + 1, epochs + 1):
        cur_lr = _lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        net.train(); tl = 0.0
        if cache:
            perm = torch.randperm(n_train, device=dev)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad(); loss = lossf(net(Xg[idx]), Yg[idx]); loss.backward(); opt.step()
                tl += loss.item() * len(idx)
        else:
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
                tl += loss.item() * len(xb)
        tl /= n_train
        net.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xv), batch_size):
                vp.append(net(Xv[i:i + batch_size]).cpu().numpy())
            vp = np.concatenate(vp)
            vl = float(np.mean((vp - yv_np) ** 2)); vr2 = r2(yv_np, vp)
        rec = {"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_r2": vr2,
               "lr": cur_lr}
        improved = vl < best_val
        if improved:
            best_val = vl; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            _save(ckpt_path, epoch, vl, vr2)
            rec["checkpoint"] = True
        else:
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
            print(f"  [{name}] epoch {epoch}/{epochs} train={tl:.4f} "
                  f"val={vl:.4f} val_r2={vr2:.4f}{'  *best' if improved else ''}", flush=True)
        if patience > 0 and no_improve >= patience:
            print(f"  [{name}] early stop at epoch {epoch} (best val {best_val:.4f})", flush=True)
            with hist_path.open("a") as fh:
                fh.write(json.dumps({"epoch": epoch, "early_stop": True}) + "\n")
            break
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
    ap.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="'cosine' anneals lr from --lr down to --lr-min over "
                         "--epochs (by absolute epoch, so it resumes cleanly).")
    ap.add_argument("--lr-min", type=float, default=1e-5,
                    help="Final learning rate for the cosine schedule.")
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

    X, Y, chan_names, keys = build_dataset(
        args.batch_dir, args.filename, args.n_cases, args.grid,
        args.m_lo, args.m_hi, args.spectrum_field, args.eps,
        target_norm=args.target_norm, target_space=args.target_space)
    N = X.shape[0]

    # Persist the run configuration so `python -m surge.check_training` (and the
    # user) can see exactly what preprocessing/target this run used.
    target_desc = (("max-normalized " if args.target_norm == "max" else "")
                   + ("log10|dp|" if args.target_space == "log10" else "|dp|")
                   + ", global z-score")
    (out / "run_config.json").write_text(json.dumps({
        "batch_dir": args.batch_dir, "filename": args.filename,
        "spectrum_field": args.spectrum_field, "n_cases": args.n_cases,
        "grid": args.grid, "m_window": [args.m_lo, args.m_hi],
        "models": list(args.models), "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience,
        "seed": args.seed, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "target_norm": args.target_norm, "target_space": args.target_space,
        "target": target_desc,
        "peak_weight": args.peak_weight, "peak_pow": args.peak_pow,
        "fno_modes": args.fno_modes, "fno_hidden": args.fno_hidden,
        "loc_weight": args.loc_weight, "marg_weight": args.marg_weight,
        "loc_beta": args.loc_beta,
        "lr": args.lr, "lr_schedule": args.lr_schedule, "lr_min": args.lr_min,
    }, indent=2))

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_test = int(args.test_frac * N)
    n_val = int(args.val_frac * N)
    te = perm[:n_test]; va = perm[n_test:n_test + n_val]; tr = perm[n_test + n_val:]
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    # Standardize input channels (train stats), and target (train stats, global).
    xm = X[tr].mean((0, 2, 3), keepdims=True)
    xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs
    ym = float(Y[tr].mean()); ysd = float(Y[tr].std() + 1e-8)
    Yn = (Y - ym) / ysd

    results: Dict[str, Dict] = {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        t0 = time.time()
        net = _build_net(name, X.shape[1], fno_modes=args.fno_modes,
                         fno_hidden=args.fno_hidden, grid=args.grid)
        if net is None:
            print(f"  unknown model {name}, skipping"); continue
        net, n_params = _train_net(
            net, name, out, Xn[tr], Yn[tr], Xn[va], Yn[va],
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            gpu_cache=not args.no_gpu_cache, resume=args.resume, ckpt_every=args.ckpt_every,
            peak_weight=args.peak_weight, peak_pow=args.peak_pow,
            loc_weight=args.loc_weight, marg_weight=args.marg_weight,
            loc_beta=args.loc_beta, lr_schedule=args.lr_schedule, lr_min=args.lr_min)
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
