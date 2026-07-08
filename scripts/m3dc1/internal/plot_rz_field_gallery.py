#!/usr/bin/env python
"""Worst→best RZ field gallery: truth | prediction | residual (Re δp).

Loads a trained gauge-fix RZ run, predicts on the held-out TEST split, ranks
cases by phase-aligned relL2, and saves side-by-side panels.

Usage:
  python scripts/m3dc1/internal/plot_rz_field_gallery.py \\
      --run runs/rz_field_gaugefix_complex_g201 --model fno2d --device cuda

  python scripts/m3dc1/internal/plot_rz_field_gallery.py \\
      --run runs/rz_field_gaugefix_complex_g201 --n-rows 9 --percentiles
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from gauge_fix import apply_global_phase, optimal_global_phase, rel_l2_complex_batch  # noqa: E402
import train_rz_field_image as T  # noqa: E402


def _re_field(arr: np.ndarray) -> np.ndarray:
    """Real part of (N,2,H,W) complex stack or (N,H,W) real."""
    if arr.ndim == 4 and arr.shape[1] == 2:
        return arr[:, 0]
    return arr


def _align_pred(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    theta = optimal_global_phase(pred, true)
    return apply_global_phase(pred, theta)


def _pick_indices(rel: np.ndarray, *, n_rows: int, percentiles: bool) -> List[Tuple[str, int, float]]:
    order = np.argsort(rel)  # worst first
    n = len(order)
    if percentiles:
        pct = np.linspace(0, 100, n_rows)
        idxs = [int(order[min(n - 1, round(p / 100 * (n - 1)))]) for p in pct]
        labels = [f"p{int(p):02d}" for p in pct]
    else:
        if n_rows >= n:
            idxs = order.tolist()
            labels = [f"rank{r}" for r in range(n)]
        else:
            pos = np.linspace(0, n - 1, n_rows).astype(int)
            idxs = [int(order[i]) for i in pos]
            labels = ["worst" if i == 0 else "best" if i == n_rows - 1 else f"q{i}" for i in range(n_rows)]
    return [(labels[i], idxs[i], float(rel[idxs[i]])) for i in range(len(idxs))]


def _plot_gallery(
    picks: List[Tuple[str, int, float]],
    gt: np.ndarray,
    pred: np.ndarray,
    keys: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.8 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (label, i, rel) in enumerate(picks):
        yt = gt[i]
        yp = pred[i]
        res = yp - yt
        vmax = float(np.percentile(np.abs(yt[np.isfinite(yt)]), 98) or 1.0)
        rmax = float(np.percentile(np.abs(res[np.isfinite(res)]), 98) or 1e-6)
        row_key = str(keys[i])
        for c, (img, ctitle, cmap, vmin, vmax_) in enumerate([
            (yt, "ground truth Re(δp)", "RdBu_r", -vmax, vmax),
            (yp, "prediction (phase-aligned)", "RdBu_r", -vmax, vmax),
            (res, "residual (pred − truth)", "coolwarm", -rmax, rmax),
        ]):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax_)
            if r == 0:
                ax.set_title(ctitle, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{label}\n{align_med_lbl(rel)}\n{row_key[:28]}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def align_med_lbl(rel: float) -> str:
    return f"align_relL2={rel:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="fno2d")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default=None, help="Output dir (default <run>/eval_gallery)")
    ap.add_argument("--n-rows", type=int, default=9, help="Gallery rows (worst→best)")
    ap.add_argument("--percentiles", action="store_true",
                    help="Pick evenly spaced percentiles (0%%..100%%)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    norm = json.loads((run / "norm_stats.json").read_text())
    splits = json.loads((run / "splits.json").read_text())
    out = Path(args.out) if args.out else run / "eval_gallery"
    out.mkdir(parents=True, exist_ok=True)

    test_keys = np.array(splits["test_keys"])
    test_key_allow = set(test_keys.tolist())
    ym = float(norm["y_mean"])
    ysd = float(norm["y_std"])
    xm = np.array(norm["input_mean"], float).reshape(-1, 1, 1)
    xs = np.array(norm["input_std"], float).reshape(-1, 1, 1)
    te = np.arange(len(test_keys), dtype=int)

    exclude_keys = None
    ex_path = cfg.get("exclude_list")
    if ex_path and Path(ex_path).is_file():
        raw = Path(ex_path).read_text().strip()
        try:
            exclude_keys = set(json.loads(raw).keys())
        except Exception:
            exclude_keys = {ln.strip() for ln in raw.splitlines() if ln.strip()}

    ck_tag = (f"g{cfg['grid']}_pf{cfg['pert_field']}_gf{cfg.get('gauge_fix')}"
              f"_cx{cfg.get('complex_target')}_fl{cfg.get('target_floor')}"
              f"_sm{cfg.get('target_smooth')}_ex{bool(exclude_keys)}_test{len(test_keys)}")
    ds_cache = out / f"dataset_{ck_tag}.npz"
    if ds_cache.is_file():
        print(f"Loading cached test dataset {ds_cache}", flush=True)
        z = np.load(ds_cache, mmap_mode="r")
        X, Y = z["X"], z["Y"]
    else:
        print(f"Building TEST-only dataset ({len(test_keys)} cases)...", flush=True)
        X, Y, _, keys, _, _ = T.build_rz_dataset(
            cfg["batch_dir"], cfg["filename"], 0, cfg["grid"],
            pert_field=cfg["pert_field"], exclude_keys=exclude_keys,
            target_floor=cfg.get("target_floor"), target_smooth=cfg.get("target_smooth"),
            gauge_fix=cfg.get("gauge_fix", False),
            complex_target=cfg.get("complex_target", False),
            midplane_z=cfg.get("midplane_z", "axis"),
            gauge_ref=cfg.get("gauge_ref", "peak"),
            peak_window=cfg.get("peak_window", 3),
            key_allow=test_key_allow,
        )
        np.savez(ds_cache, X=X.astype(np.float32), Y=Y.astype(np.float32))
        print(f"Cached test dataset -> {ds_cache}", flush=True)

    Xn = None  # normalized in predict loop (mmap-safe)
    if cfg.get("no_target_zscore"):
        Yn = np.asarray(Y, dtype=np.float32)
    else:
        Yn = (np.asarray(Y) - ym) / ysd

    import torch
    out_ch = 2 if cfg.get("complex_target") else 1
    net = T._build_rz_net(
        args.model, X.shape[1], out_channels=out_ch,
        fno_modes=cfg.get("fno_modes", 64),
        fno_hidden=cfg.get("fno_hidden", 32),
        grid=cfg["grid"],
    )
    ck_path = Path(args.ckpt) if args.ckpt else run / f"ckpt_{args.model}.pt"
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.to(args.device).eval()
    print(f"Loaded {ck_path} (epoch {ck.get('epoch')}, best_ep {ck.get('best_epoch')}, "
          f"best_relL2 {ck.get('best_relL2_median', float('nan')):.4f})")

    print(f"Predicting {len(test_keys)} test cases (batch={args.batch_size})...", flush=True)
    dev = next(net.parameters()).device
    net.eval()
    pred_chunks = []
    n = len(test_keys)
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            xb = (np.asarray(X[i:i + args.batch_size], dtype=np.float32) - xm) / xs
            tb = torch.tensor(xb, dtype=torch.float32, device=dev)
            pred_chunks.append(net(tb).cpu().numpy())
    pred_z = np.concatenate(pred_chunks, axis=0)
    if pred_z.shape[1] == 1:
        pred_z = pred_z.squeeze(1)
    pred_norm = pred_z * ysd + ym
    gt_norm = np.asarray(Y)

    if cfg.get("complex_target") and pred_norm.ndim == 4:
        rel_raw, rel_aligned = rel_l2_complex_batch(pred_norm, gt_norm, phase_align=True)
        pred_aligned = np.empty((len(test_keys), X.shape[2], X.shape[3]), np.float32)
        for i in range(len(test_keys)):
            pred_aligned[i] = _re_field(_align_pred(pred_norm[i:i + 1], gt_norm[i:i + 1]))[0]
        gt_re = _re_field(gt_norm)
        rank = rel_aligned
        metric_name = "aligned_relL2"
    else:
        rel_raw = T._rel_l2_batch(pred_norm, gt_norm)
        rel_aligned = rel_raw
        gt_re = gt_norm
        pred_aligned = pred_norm
        rank = rel_raw
        metric_name = "relL2"

    print(f"TEST {metric_name}: median={np.median(rank):.4f} "
          f"min={rank.min():.4f} max={rank.max():.4f}")

    picks = _pick_indices(rank, n_rows=args.n_rows, percentiles=args.percentiles)
    title = (f"{run.name} [{args.model}]  Re(δp) max-norm  "
             f"test {metric_name} med={np.median(rank):.3f}  (ckpt ep {ck.get('best_epoch', ck.get('epoch'))})")
    fpng = out / f"rz_gallery_worst_to_best_{args.model}.png"
    _plot_gallery(picks, gt_re, pred_aligned, test_keys, out_path=fpng, title=title)

    # Compact worst / median / best
    order = np.argsort(rank)
    compact = [
        ("worst", int(order[0]), float(rank[order[0]])),
        ("median", int(order[len(order) // 2]), float(rank[order[len(order) // 2]])),
        ("best", int(order[-1]), float(rank[order[-1]])),
    ]
    fpng3 = out / f"rz_worst_median_best_{args.model}.png"
    _plot_gallery(
        [(a, i, v) for a, i, v in compact],
        gt_re, pred_aligned, test_keys, out_path=fpng3,
        title=title + "  [worst / median / best]",
    )

    rows = []
    for lab, i, v in picks:
        rows.append({
            "label": lab, "test_index": int(i), "key": str(test_keys[i]),
            "aligned_relL2": float(rel_aligned[i]), "raw_relL2": float(rel_raw[i]),
        })
    summary = {
        "run": str(run), "model": args.model, "checkpoint": str(ck_path),
        "best_epoch": int(ck.get("best_epoch", ck.get("epoch", -1))),
        "metric": metric_name,
        "test_median": float(np.median(rank)),
        "test_min": float(rank.min()),
        "test_max": float(rank.max()),
        "gallery": rows,
        "figures": [str(fpng), str(fpng3)],
    }
    (out / "gallery_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "percase_test_relL2.csv").open("w") as fh:
        fh.write("key,raw_relL2,aligned_relL2\n")
        for k, rr, ra in zip(test_keys, rel_raw, rel_aligned):
            fh.write(f"{k},{rr:.5f},{ra:.5f}\n")

    print(f"\nWrote {fpng}")
    print(f"Wrote {fpng3}")
    for lab, i, v in picks:
        print(f"  {lab:8s} {test_keys[i]:28s} {metric_name}={v:.4f}")


if __name__ == "__main__":
    main()
