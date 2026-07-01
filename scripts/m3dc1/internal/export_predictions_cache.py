#!/usr/bin/env python
"""Export a compact predictions cache for a trained spectrum-image FNO run.

Rebuilds the (reproducible) dataset + split used at training time, loads a
checkpoint, predicts the normalized |delta p| spectrum for the requested splits
(val/test by default), and saves everything the curation notebook needs into a
single ``predictions_cache.npz``:

    keys           (N,)          "<run_id>_<eq_id>" per case
    paths          (N,)          absolute path to each case's csdata h5
    split          (N,)          "val" / "test" (or "train")
    gt   (N,H,W) float16         ground-truth target (normalized, in target space)
    pred (N,H,W) float16         model prediction (same space, un-standardized)
    m_grid (W,), psi_grid (W,)   uniform axes
    r2_global (N,), r2_pattern (N,)   per-case R2 (standardized target space)
    gamma0, gamma1 (N,)          growth rate at the two saved time slices (sign=>stability)
    ntor (N,)                    toroidal mode number
    y_mean, y_std                scalar train-split standardization used for R2
    plus target_norm / target_space / grid / m_window metadata

Browsing the cache needs no GPU and no fpy. RZ reconstruction in the notebook
uses the case ``paths`` + m3dc1ml on demand.

Usage:
    python scripts/m3dc1/internal/export_predictions_cache.py \
        --run runs/spectrum_image_full_maxnorm_log10 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                 # for train_spectrum_image
sys.path.insert(0, str(_HERE.parents[1]))             # for dataset_complex_v2 (scripts/m3dc1)

import h5py  # noqa: E402
import train_spectrum_image as T  # noqa: E402


def _reproduce_split(n: int, seed: int, test_frac: float, val_frac: float):
    """Match train_spectrum_image.main()'s split exactly."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    te = perm[:n_test]
    va = perm[n_test:n_test + n_val]
    tr = perm[n_test + n_val:]
    return tr, va, te


def _read_gamma(path: str):
    try:
        with h5py.File(path, "r") as f:
            rg = f["runs"][list(f["runs"].keys())[0]]
            g0 = g1 = np.nan
            ntor = np.nan
            if "growth_rate" in rg:
                if "0" in rg["growth_rate"]:
                    g0 = float(rg["growth_rate"]["0"][()])
                if "1" in rg["growth_rate"]:
                    g1 = float(rg["growth_rate"]["1"][()])
            if "parset" in rg:
                names = [T._decode(x) for x in rg["parset"]["names"]]
                vals = np.asarray(rg["parset"]["values"]).ravel()
                if "ntor" in names:
                    ntor = float(vals[names.index("ntor")])
            return g0, g1, ntor
    except Exception:
        return np.nan, np.nan, np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run dir with ckpt + run_config.json")
    ap.add_argument("--ckpt", default=None, help="Checkpoint (default ckpt_<model>.pt)")
    ap.add_argument("--model", default="fno2d")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    ckpt = Path(args.ckpt) if args.ckpt else run / f"ckpt_{args.model}.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    print(f"run={run}\ncfg target={cfg.get('target')} "
          f"norm={cfg.get('target_norm')} space={cfg.get('target_space')}")

    import torch
    dev = ("cuda" if (args.device in ("cuda", "auto") and torch.cuda.is_available())
           else "cpu")

    n_cases = cfg.get("n_cases") or 0
    m_lo, m_hi = cfg.get("m_window", [-80.0, 20.0])
    X, Y, chan, keys, paths = T.build_dataset(
        cfg["batch_dir"], cfg["filename"], (n_cases or None), cfg["grid"],
        float(m_lo), float(m_hi), cfg.get("spectrum_field", "p"), 1e-12,
        target_norm=cfg.get("target_norm", "none"),
        target_space=cfg.get("target_space", "log10"), return_paths=True)
    N = X.shape[0]
    tr, va, te = _reproduce_split(N, cfg.get("seed", 42),
                                  cfg.get("test_frac", 0.2), cfg.get("val_frac", 0.1))
    idx_by = {"train": tr, "val": va, "test": te}

    # Standardize with TRAIN stats (identical to training).
    xm = X[tr].mean((0, 2, 3), keepdims=True)
    xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    ym = float(Y[tr].mean()); ysd = float(Y[tr].std() + 1e-8)

    net = T._build_net(args.model, X.shape[1])
    state = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state["state_dict"])
    net = net.to(dev).eval()
    print(f"loaded {ckpt.name} (epoch {state.get('epoch')}, "
          f"val_r2 {state.get('val_r2')}); device={dev}")

    grid = cfg["grid"]
    psi_grid = np.linspace(0.0, 1.0, grid)
    m_grid = np.linspace(float(m_lo), float(m_hi), grid)

    all_keys, all_paths, all_split = [], [], []
    gt_list, pred_list, r2g, r2p, g0s, g1s, ntors = [], [], [], [], [], [], []
    t0 = time.time()
    for sp in args.splits:
        sel = idx_by[sp]
        Xs = (X[sel] - xm) / xs
        pred_std = T._predict_net(net, Xs, 16)             # standardized space (N,H,W)
        gt_std = (Y[sel] - ym) / ysd
        pred_tgt = pred_std * ysd + ym                     # back to log10/raw space
        for j in range(len(sel)):
            all_keys.append(keys[sel[j]]); all_paths.append(paths[sel[j]])
            all_split.append(sp)
            gt_list.append(Y[sel[j]].astype(np.float16))
            pred_list.append(pred_tgt[j].astype(np.float16))
            r2g.append(T.r2(gt_std[j], pred_std[j]))
            r2p.append(T.pattern_r2(gt_std[j][None], pred_std[j][None]))
            g0, g1, nt = _read_gamma(paths[sel[j]])
            g0s.append(g0); g1s.append(g1); ntors.append(nt)
        print(f"  {sp}: {len(sel)} cases  ({time.time()-t0:.0f}s)")

    out = Path(args.out) if args.out else run / "predictions_cache.npz"
    np.savez_compressed(
        out,
        keys=np.array(all_keys), paths=np.array(all_paths),
        split=np.array(all_split),
        gt=np.stack(gt_list), pred=np.stack(pred_list),
        m_grid=m_grid.astype(np.float32), psi_grid=psi_grid.astype(np.float32),
        r2_global=np.array(r2g, np.float32), r2_pattern=np.array(r2p, np.float32),
        gamma0=np.array(g0s, np.float32), gamma1=np.array(g1s, np.float32),
        ntor=np.array(ntors, np.float32),
        y_mean=np.float32(ym), y_std=np.float32(ysd),
        target_norm=str(cfg.get("target_norm")), target_space=str(cfg.get("target_space")),
        grid=np.int32(grid), m_window=np.array([m_lo, m_hi], np.float32),
        spectrum_field=str(cfg.get("spectrum_field", "p")),
    )
    r2g = np.array(r2g)
    print(f"\nWrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(all_keys)} cases)")
    print(f"per-case R2 (standardized): median={np.median(r2g):.3f} "
          f"mean={np.mean(r2g):.3f}  [min {r2g.min():.3f}, max {r2g.max():.3f}]")


if __name__ == "__main__":
    main()
