#!/usr/bin/env python3
"""Evaluate several spectrum surrogates on a COMMON reference target so models
trained on different targets (unfloored / floored / smoothed) are comparable.

All runs must share grid / m_window / seed / fracs (identical split). We build
the raw (max-norm log10, un-floored, un-smoothed) target ONCE, reproduce the
split, and for each model: derive ITS training target from the raw one
(clip/smooth as its run_config says), standardize with its own train stats,
predict, and map back to dex. Then we clip every prediction AND the raw GT to
[-clip, 0] and score on that common reference -- i.e. "how well is the top
`clip` decades of the real spectrum predicted", regardless of training target.

Spec format (repeatable):  RUNDIR:ARCH:LABEL

  python scripts/m3dc1/internal/eval_common.py --clip 6 --split valtest \
    --spec runs/spectrum_fno48_loc_p150:fno2d:fno_nofloor \
    --spec runs/spectrum_unet_loc_rt:unet:unet_nofloor \
    --spec runs/spectrum_fno48_floor6:fno2d:fno_floor6 \
    --spec runs/spectrum_unet_floor6:unet:unet_floor6 \
    --out runs/eval_common
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_spectrum_image as T           # noqa: E402
from compare_models import _percase_metrics, _panel  # noqa: E402


def _derive_target(Yraw, floor, smooth):
    """Reproduce build_dataset's post-processing (floor then smooth) on Yraw."""
    y = Yraw
    if floor:
        y = np.maximum(y, -float(floor))
    if smooth:
        from scipy.ndimage import gaussian_filter
        y = np.stack([gaussian_filter(im, sigma=float(smooth)) for im in y])
    return y


def _predict(net, Xn, bs, dev):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.tensor(Xn[i:i + bs], dtype=torch.float32).to(dev)
            out.append(net(xb).cpu().numpy()[:, 0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="RUNDIR:ARCH:LABEL (repeatable)")
    ap.add_argument("--split", choices=["val", "test", "valtest"], default="valtest")
    ap.add_argument("--clip", type=float, default=6.0,
                    help="common reference keeps 10^0..10^-clip (peak=0).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="runs/eval_common")
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); (out / "figs").mkdir(parents=True, exist_ok=True)
    specs = [s.split(":") for s in args.spec]
    ref_cfg = json.loads((Path(specs[0][0]) / "run_config.json").read_text())
    m_lo, m_hi = ref_cfg.get("m_window", [-80.0, 20.0]); grid = ref_cfg["grid"]

    print(f"[{dev}] building RAW target once ...", flush=True)
    X, Yraw, chan, keys = T.build_dataset(
        ref_cfg["batch_dir"], ref_cfg["filename"], (ref_cfg.get("n_cases") or None),
        grid, float(m_lo), float(m_hi), ref_cfg.get("spectrum_field", "p"), 1e-12,
        target_norm="max", target_space="log10")   # raw: no floor / no smooth
    N = X.shape[0]
    rng = np.random.RandomState(ref_cfg.get("seed", 42)); perm = rng.permutation(N)
    n_te = int(ref_cfg["test_frac"] * N); n_va = int(ref_cfg["val_frac"] * N)
    te = perm[:n_te]; va = perm[n_te:n_te + n_va]; tr = perm[n_te + n_va:]
    ev = {"val": va, "test": te, "valtest": np.concatenate([va, te])}[args.split]
    xm = X[tr].mean((0, 2, 3), keepdims=True); xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs
    psi = np.linspace(0.0, 1.0, grid); m = np.linspace(m_lo, m_hi, grid)

    clip = args.clip
    gt_ref = np.clip(Yraw[ev], -clip, 0.5)          # common reference GT
    keys_ev = np.array(keys)[ev]

    results = {}; preds = {}
    for rundir, arch, label in specs:
        run = Path(rundir); cfg = json.loads((run / "run_config.json").read_text())
        floor = cfg.get("target_floor"); smooth = cfg.get("target_smooth")
        Yt = _derive_target(Yraw, floor, smooth)     # this model's training target
        ym = float(Yt[tr].mean()); ysd = float(Yt[tr].std() + 1e-8)
        net = T._build_net(arch, X.shape[1], fno_modes=int(cfg.get("fno_modes", 16)),
                           fno_hidden=int(cfg.get("fno_hidden", 32)), grid=grid)
        ck = torch.load(run / f"ckpt_{arch}.pt", map_location=dev)
        net.load_state_dict(ck["state_dict"]); net = net.to(dev).eval()
        pdex = _predict(net, Xn[ev], args.batch_size, dev) * ysd + ym
        pdex = np.clip(pdex, -clip, 0.5)             # score on common reference
        met = _percase_metrics(gt_ref, pdex, psi)
        results[label] = met; preds[label] = pdex.astype(np.float16)
        print(f"loaded {label:16s} floor={floor} smooth={smooth}", flush=True)

    # summary table
    order_metrics = ["r2_global", "r2_pattern", "ssim", "rmse", "peak_rmse", "dpsi"]
    print(f"\n===== COMMON REFERENCE: top {clip:g} decades, split={args.split}, "
          f"n={len(ev)} =====")
    hdr = f"{'model':18s}" + "".join(f"{k:>11s}" for k in order_metrics)
    print(hdr); print("-" * len(hdr))
    summ = {}
    for label, met in results.items():
        summ[label] = {k: float(np.mean(met[k])) for k in order_metrics}
        print(f"{label:18s}" + "".join(f"{np.mean(met[k]):11.3f}" for k in order_metrics))
    (out / "summary.json").write_text(json.dumps(
        {"clip": clip, "split": args.split, "n": int(len(ev)), "means": summ}, indent=2))
    np.savez_compressed(out / "percase.npz", keys=keys_ev,
                        **{f"{lab}__{k}": v for lab, met in results.items()
                           for k, v in met.items()})

    # paired per-case improvement (floored vs nofloor) if both present per arch
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pairs = [("fno_nofloor", "fno_floor6", "FNO2D"), ("unet_nofloor", "unet_floor6", "U-Net")]
    have = [p for p in pairs if p[0] in results and p[1] in results]
    if have:
        fig, axs = plt.subplots(1, len(have), figsize=(6 * len(have), 5))
        if len(have) == 1:
            axs = [axs]
        for axi, (a, b, name) in zip(axs, have):
            xa = results[a]["r2_pattern"]; yb = results[b]["r2_pattern"]
            axi.scatter(xa, yb, s=6, alpha=0.3)
            lo = min(xa.min(), yb.min()); hi = 1.0
            axi.plot([lo, hi], [lo, hi], "k--", lw=1)
            win = float(np.mean(yb > xa)) * 100
            axi.set_title(f"{name}: pattern R² per case\nfloor6 better in {win:.0f}% of cases")
            axi.set_xlabel("no floor"); axi.set_ylabel("floor 1e-6"); axi.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out / "figs" / "floor_vs_nofloor_scatter.png", dpi=110)
        print("saved", out / "figs" / "floor_vs_nofloor_scatter.png")

    # representative panels for the best model (by pattern R2) among floored ones
    best_label = max(summ, key=lambda l: summ[l]["r2_pattern"])
    r2p = results[best_label]["r2_pattern"]; ordr = np.argsort(r2p)
    idx = [ordr[-1], ordr[len(ordr) // 2], ordr[0]]
    # compare the two archs' floored preds on these cases (fallback to available)
    fkey = "fno_floor6" if "fno_floor6" in preds else list(preds)[0]
    ukey = "unet_floor6" if "unet_floor6" in preds else list(preds)[-1]
    titles = [f"BEST patR²={r2p[idx[0]]:.2f}", f"MEDIAN patR²={r2p[idx[1]]:.2f}",
              f"WORST patR²={r2p[idx[2]]:.2f}"]
    _panel(gt_ref, preds[fkey].astype(np.float32), preds[ukey].astype(np.float32),
           idx, titles, psi, m, out / "figs" / "best_median_worst.png")
    print("\nDONE ->", out)


if __name__ == "__main__":
    main()
