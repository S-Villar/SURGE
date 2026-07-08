"""Estimate exponential (log-linear) decay rates from training history JSONL.

Models (linear scale):
  L(e) ≈ A · exp(−λ e)           simple log fit on log(L)
  L(e) ≈ L_∞ + A · exp(−λ e)     offset fit on log(L − L_∞)

λ is the exponential decay rate (1/epoch). Half-life = ln(2) / λ.

Usage:
  python scripts/m3dc1/internal/fit_loss_decay.py \\
      --history runs/rz_field_gaugefix_complex_g201/history_fno2d.jsonl

  python scripts/m3dc1/internal/fit_loss_decay.py \\
      --history runs/rz_field_gaugefix_complex_g201/history_fno2d.jsonl \\
      --metrics train_loss val_comp val_relL2_aligned_median \\
      --window 30 --out runs/rz_field_gaugefix_complex_g201/loss_decay.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_history(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "train_loss" in row and "val_loss" in row:
            rows.append(row)
    return rows


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def _fit_log_decay(
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    offset: Optional[float] = None,
) -> Optional[Dict]:
    """Fit log(values - offset) = intercept + slope * epoch. slope < 0 → decay."""
    y = np.asarray(values, float)
    e = np.asarray(epochs, float)
    if offset is None:
        mask = np.isfinite(y) & (y > 1e-12)
        y_fit = y
    else:
        y_adj = y - float(offset)
        mask = np.isfinite(y_adj) & (y_adj > 1e-12)
        y_fit = y_adj
    if mask.sum() < 3:
        return None
    logy = np.log(y_fit[mask])
    ep = e[mask]
    slope, intercept = np.polyfit(ep, logy, 1)
    pred_log = intercept + slope * ep
    ss_res = float(np.sum((logy - pred_log) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    lam = -float(slope)
    half_life = math.log(2) / lam if lam > 1e-12 else float("inf")
    a = float(math.exp(intercept))
    return {
        "lambda_per_epoch": lam,
        "half_life_epochs": half_life,
        "amplitude_A": a,
        "offset_L_inf": offset,
        "log_intercept": float(intercept),
        "log_slope": float(slope),
        "r2_log_space": r2,
        "n_points": int(mask.sum()),
        "epoch_start": float(ep[0]),
        "epoch_end": float(ep[-1]),
    }


def analyze_metric(
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    window: Optional[int] = None,
    use_offset: bool = True,
    smooth: int = 1,
) -> Dict:
    y = _smooth(np.asarray(values, float), smooth)
    e = np.asarray(epochs, float)
    if window and window > 0 and len(e) > window:
        e = e[-window:]
        y = y[-window:]

    out: Dict = {"metric": None}
    simple = _fit_log_decay(e, y, offset=None)
    offset = None
    if use_offset and len(y) >= 5:
        offset = float(np.nanmin(y[-max(5, len(y) // 5):]))
        offset = max(offset * 0.95, 0.0)
    offset_fit = _fit_log_decay(e, y, offset=offset) if use_offset else None

    best = offset_fit if offset_fit and (
        simple is None or offset_fit.get("r2_log_space", -1) >= simple.get("r2_log_space", -1)
    ) else simple

    if best is None:
        return {"fit": None}

    # Project epochs to reach fraction of current value
    lam = best["lambda_per_epoch"]
    cur = float(y[-1])
    floor = best["offset_L_inf"] or 0.0
    projections = {}
    if lam > 1e-12 and cur > floor + 1e-12:
        for frac in (0.5, 0.25, 0.1):
            target = floor + frac * (cur - floor)
            if target > floor + 1e-12:
                n_ep = math.log((target - floor) / (cur - floor)) / (-lam)
                projections[f"epochs_to_{int(frac*100)}pct_of_gap"] = float(n_ep)

    return {
        "current": cur,
        "fit": best,
        "projections": projections,
        "simple_log_fit": simple,
        "offset_log_fit": offset_fit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", required=True, help="history_*.jsonl path")
    ap.add_argument("--metrics", nargs="+",
                    default=["train_loss", "val_loss", "val_comp", "val_relL2_aligned_median"],
                    help="Keys to fit (must exist in JSONL rows)")
    ap.add_argument("--window", type=int, default=0,
                    help="Fit only the last N epochs (0 = all)")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window before fit")
    ap.add_argument("--no-offset", action="store_true",
                    help="Disable L_inf offset model; use plain log(L) fit only")
    ap.add_argument("--out", default=None, help="Optional PNG with fits overlaid")
    args = ap.parse_args()

    rows = _load_history(Path(args.history))
    if not rows:
        raise SystemExit(f"No epoch rows in {args.history}")

    epochs = np.array([r["epoch"] for r in rows], float)
    print(f"History: {args.history}  epochs {int(epochs[0])}–{int(epochs[-1])}  n={len(epochs)}")
    if args.window:
        print(f"Fitting window: last {args.window} epochs")
    print()

    results: Dict[str, dict] = {}
    for key in args.metrics:
        if key not in rows[0]:
            print(f"[{key}]  —  key not in history; skip")
            continue
        vals = np.array([r.get(key, np.nan) for r in rows], float)
        res = analyze_metric(
            epochs, vals,
            window=args.window or None,
            use_offset=not args.no_offset,
            smooth=max(1, args.smooth),
        )
        results[key] = res
        fit = res.get("fit")
        if fit is None:
            print(f"[{key}]  insufficient positive points for log fit")
            continue
        lam = fit["lambda_per_epoch"]
        hl = fit["half_life_epochs"]
        r2 = fit["r2_log_space"]
        model = "offset" if fit.get("offset_L_inf") is not None else "simple"
        print(f"[{key}]")
        print(f"  model={model}  λ={lam:.5f} /epoch  t½={hl:.1f} ep  R²(log)={r2:.3f}  "
              f"current={res['current']:.5g}")
        if fit.get("offset_L_inf") is not None:
            print(f"  L_∞≈{fit['offset_L_inf']:.5g}  A≈{fit['amplitude_A']:.5g}")
        for k, v in res.get("projections", {}).items():
            print(f"  {k}: {v:.1f} epochs (linear extrapolation)")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = len([k for k in args.metrics if k in results and results[k].get("fit")])
            if n == 0:
                raise ValueError("no fits to plot")
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
            ax_i = 0
            for key in args.metrics:
                if key not in results or results[key].get("fit") is None:
                    continue
                ax = axes[0, ax_i]
                ax_i += 1
                vals = np.array([r.get(key, np.nan) for r in rows], float)
                y = _smooth(vals, max(1, args.smooth))
                ax.semilogy(epochs, y, "o-", ms=3, lw=1, alpha=0.7, label=key)
                fit = results[key]["fit"]
                lam = fit["lambda_per_epoch"]
                l_inf = fit.get("offset_L_inf") or 0.0
                a = fit["amplitude_A"]
                e_line = np.linspace(epochs[0], epochs[-1], 100)
                if fit.get("offset_L_inf") is not None:
                    pred = l_inf + a * np.exp(-lam * e_line)
                else:
                    pred = a * np.exp(-lam * e_line)
                ax.semilogy(e_line, pred, "r--", lw=1.5,
                            label=f"λ={lam:.4g}/ep  R²={fit['r2_log_space']:.2f}")
                ax.set_xlabel("epoch")
                ax.set_ylabel(key)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=120, bbox_inches="tight")
            print(f"\nWrote {out}")
        except Exception as exc:
            print(f"Plot skipped: {exc}")

    out_json = Path(args.history).with_name(Path(args.history).stem + "_decay.json")
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
