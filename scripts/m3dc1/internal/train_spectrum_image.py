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
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return X (N,C,H,W), Y (N,H,W) log10-magnitude, channel names, case keys."""
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
    t0 = time.time()
    for i, p in enumerate(paths):
        c = _read_case(Path(p), spectrum_field)
        if c is None:
            continue
        mag, m_modes, psi = c["mag"], c["m_modes"], c["psi"]
        sel = (m_modes >= m_lo) & (m_modes <= m_hi)
        if sel.sum() < 4:
            continue
        logmag = np.log10(mag[sel, :] + eps)              # (nmc, npsi)
        m_vals = m_modes[sel]
        # interp along psi (cols) onto uniform psi_grid
        tmp = np.vstack([_interp_to(psi_grid, psi, row) for row in logmag])  # (nmc,W)
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
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)")
    X = np.stack(Xs); Y = np.stack(Ys)
    print(f"  Built X={X.shape} Y={Y.shape} in {time.time()-t0:.0f}s")
    return X, Y, chan_names, keys


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
    ap.add_argument("--models", nargs="+", default=["fno2d", "unet"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/spectrum_image")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    X, Y, chan_names, keys = build_dataset(
        args.batch_dir, args.filename, args.n_cases, args.grid,
        args.m_lo, args.m_hi, args.spectrum_field, args.eps)
    N = X.shape[0]

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

    from surge.model.backends.fno2d import FNO2dModel
    from surge.model.backends.unet import UNetModel

    results: Dict[str, Dict] = {}
    for name in args.models:
        print(f"\n=== Training {name} ===")
        t0 = time.time()
        if name == "fno2d":
            model = FNO2dModel(in_channels=X.shape[1], out_channels=1, hidden_channels=48,
                               n_modes=24, n_layers=4, n_epochs=args.epochs,
                               batch_size=args.batch_size, learning_rate=1e-3, patience=20,
                               verbose=True)
        elif name == "unet":
            model = UNetModel(in_channels=X.shape[1], out_channels=1, base_channels=48,
                              depth=4, n_epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=1e-3, patience=20, verbose=True)
        else:
            print(f"  unknown model {name}, skipping"); continue
        model.fit(Xn[tr], Yn[tr][:, None])
        pred = model.predict(Xn[te])            # (n_test, H, W)
        yt = Yn[te]
        res = {"test_r2_global": r2(yt, pred),
               "test_pattern_r2": pattern_r2(yt, pred),
               "train_seconds": time.time() - t0,
               "n_params": sum(p.numel() for p in model._net.parameters())}
        results[name] = res
        print(f"  {name}: test R2(global)={res['test_r2_global']:.4f} "
              f"pattern R2={res['test_pattern_r2']:.4f} "
              f"({res['n_params']/1e6:.2f}M params, {res['train_seconds']:.0f}s)")
        _save_examples(out, name, X, Yn, te, pred, chan_names, args)

    summary = {"n_cases": N, "grid": args.grid, "channels": chan_names,
               "m_window": [args.m_lo, args.m_hi], "target": "log10|dp|, global z-score",
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
