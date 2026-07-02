#!/usr/bin/env python3
"""Head-to-head evaluation of two spectrum-image surrogates (e.g. FNO2D vs U-Net)
on the shared held-out split, with the full peak-aware metric panel.

Both runs must share grid / m_window / target / seed / fracs (same split). We
load the dataset ONCE, reproduce the split, run both checkpoints, and report per
case: global R2, pattern R2, RMSE(dex), peak-RMSE(dex, top-1% amplitude), SSIM,
and dpsi (peak-location error in psi_N). Saves a per-case table + summary and
best/worst/median spectrum comparison panels and a many-case montage.

Usage (on a GPU node):
  python scripts/m3dc1/internal/compare_models.py \
    --fno-run runs/spectrum_fno48_loc_p150 \
    --unet-run runs/spectrum_unet_loc_rt \
    --split val --out runs/compare_fno_unet
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_spectrum_image as T  # noqa: E402


def _load(run: Path, name: str, C: int, grid: int, dev):
    import torch
    cfg = json.loads((run / "run_config.json").read_text())
    net = T._build_net(name, C, fno_modes=int(cfg.get("fno_modes", 16)),
                       fno_hidden=int(cfg.get("fno_hidden", 32)), grid=grid)
    ck = torch.load(run / f"ckpt_{name}.pt", map_location=dev)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


def _predict(net, Xn, bs, dev):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.tensor(Xn[i:i + bs], dtype=torch.float32).to(dev)
            out.append(net(xb).cpu().numpy()[:, 0])
    return np.concatenate(out)


def _percase_metrics(gt, pr, psi_axis):
    """gt, pr: (N,H,W) in dex units (max-norm log10). Returns dict of (N,) arrays."""
    from skimage.metrics import structural_similarity as ssim
    N = gt.shape[0]
    g = gt.reshape(N, -1); p = pr.reshape(N, -1)
    # global R2 per case
    ss = ((g - p) ** 2).sum(1)
    tt = ((g - g.mean(1, keepdims=True)) ** 2).sum(1) + 1e-12
    r2g = 1 - ss / tt
    # pattern R2 (de-meaned per case)
    gd = g - g.mean(1, keepdims=True); pd = p - p.mean(1, keepdims=True)
    ssp = ((gd - pd) ** 2).sum(1)
    ttp = (gd ** 2).sum(1) + 1e-12
    r2p = 1 - ssp / ttp
    # RMSE (dex) and peak-RMSE (top-1% amplitude pixels)
    rmse = np.sqrt(((g - p) ** 2).mean(1))
    thr = np.percentile(g, 99.0, axis=1, keepdims=True)
    pk = g >= thr
    pkrmse = np.sqrt(((g - p) ** 2 * pk).sum(1) / pk.sum(1))
    # dpsi: peak-location error in psi_N
    W = gt.shape[-1]
    gp = psi_axis[np.argmax(g, axis=1) % W]
    pp = psi_axis[np.argmax(p, axis=1) % W]
    dpsi = np.abs(gp - pp)
    # SSIM per case (data_range from GT)
    ss_im = np.array([ssim(gt[i], pr[i], data_range=float(gt[i].max() - gt[i].min() + 1e-8))
                      for i in range(N)])
    return dict(r2_global=r2g, r2_pattern=r2p, rmse=rmse, peak_rmse=pkrmse,
                dpsi=dpsi, ssim=ss_im)


def _panel(gt, fno, unet, idx, titles, psi, m, path):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nrow = len(idx)
    fig, ax = plt.subplots(nrow, 4, figsize=(15, 3.4 * nrow))
    if nrow == 1:
        ax = ax[None, :]
    ext = [psi.min(), psi.max(), m.min(), m.max()]
    for r, (i, tt) in enumerate(zip(idx, titles)):
        vmin = np.percentile(gt[i], 2); vmax = gt[i].max()
        for c, (img, lab) in enumerate([(gt[i], "ground truth"), (fno[i], "FNO2D"),
                                        (unet[i], "U-Net")]):
            im = ax[r, c].imshow(img, origin="lower", aspect="auto", extent=ext,
                                 cmap="magma", vmin=vmin, vmax=vmax)
            ax[r, c].set_title(f"{lab}\n{tt}" if c == 0 else lab, fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel("m")
            ax[r, c].set_xlabel(r"$\psi_N$")
        # error map (unet - gt)
        d = unet[i] - gt[i]
        a = np.abs(d).max()
        im2 = ax[r, 3].imshow(d, origin="lower", aspect="auto", extent=ext,
                              cmap="RdBu_r", vmin=-a, vmax=a)
        ax[r, 3].set_title("U-Net − GT", fontsize=9); ax[r, 3].set_xlabel(r"$\psi_N$")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fno-run", required=True)
    ap.add_argument("--unet-run", required=True)
    ap.add_argument("--split", choices=["val", "test", "valtest"], default="val")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--montage", type=int, default=12, help="# cases in montage")
    ap.add_argument("--out", default="runs/compare_fno_unet")
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); (out / "figs").mkdir(parents=True, exist_ok=True)
    fno_run = Path(args.fno_run); unet_run = Path(args.unet_run)
    cfg = json.loads((unet_run / "run_config.json").read_text())
    m_lo, m_hi = cfg.get("m_window", [-80.0, 20.0])

    print(f"[{dev}] building dataset once ...", flush=True)
    X, Y, chan, keys = T.build_dataset(
        cfg["batch_dir"], cfg["filename"], (cfg.get("n_cases") or None), cfg["grid"],
        float(m_lo), float(m_hi), cfg.get("spectrum_field", "p"), 1e-12,
        target_norm=cfg.get("target_norm", "max"),
        target_space=cfg.get("target_space", "log10"),
        target_floor=cfg.get("target_floor"))
    N = X.shape[0]
    rng = np.random.RandomState(cfg.get("seed", 42)); perm = rng.permutation(N)
    n_te = int(cfg["test_frac"] * N); n_va = int(cfg["val_frac"] * N)
    te = perm[:n_te]; va = perm[n_te:n_te + n_va]; tr = perm[n_te + n_va:]
    if args.split == "val":
        ev = va
    elif args.split == "test":
        ev = te
    else:
        ev = np.concatenate([va, te])
    # standardization from train split (same for both models -> same split/seed)
    xm = X[tr].mean((0, 2, 3), keepdims=True); xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    ym = float(Y[tr].mean()); ysd = float(Y[tr].std() + 1e-8)
    Xn = (X - xm) / xs
    print(f"eval split={args.split} n={len(ev)}  (dex y_std={ysd:.3f})", flush=True)

    fno = _load(fno_run, "fno2d", X.shape[1], cfg["grid"], dev)
    unet = _load(unet_run, "unet", X.shape[1], cfg["grid"], dev)
    pf = _predict(fno, Xn[ev], args.batch_size, dev) * ysd + ym   # back to dex
    pu = _predict(unet, Xn[ev], args.batch_size, dev) * ysd + ym
    gt = Y[ev]                                                    # already dex

    psi = np.linspace(0.0, 1.0, cfg["grid"]); m = np.linspace(m_lo, m_hi, cfg["grid"])
    mf = _percase_metrics(gt, pf, psi); mu = _percase_metrics(gt, pu, psi)

    def _sm(d):
        return {k: (float(np.mean(v)), float(np.median(v))) for k, v in d.items()}
    summ = {"n": len(ev), "split": args.split, "fno": _sm(mf), "unet": _sm(mu),
            "fno_run": str(fno_run), "unet_run": str(unet_run)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    np.savez_compressed(out / "percase.npz", keys=keys[ev], **{f"fno_{k}": v for k, v in mf.items()},
                        **{f"unet_{k}": v for k, v in mu.items()})

    print("\n================ SUMMARY (mean | median) ================")
    print(f"{'metric':12s} {'FNO2D':>20s} {'U-Net':>20s}")
    for k in ["r2_global", "r2_pattern", "ssim", "rmse", "peak_rmse", "dpsi"]:
        f = _sm(mf)[k]; u = _sm(mu)[k]
        print(f"{k:12s}  {f[0]:8.3f}|{f[1]:7.3f}   {u[0]:8.3f}|{u[1]:7.3f}")

    # representative cases by U-Net pattern R2
    r2p = mu["r2_pattern"]; order = np.argsort(r2p)
    worst = order[0]; best = order[-1]; med = order[len(order) // 2]
    idx = [best, med, worst]
    titles = [f"BEST  patR²={r2p[best]:.2f} ssim={mu['ssim'][best]:.2f} dψ={mu['dpsi'][best]:.3f}",
              f"MEDIAN patR²={r2p[med]:.2f} ssim={mu['ssim'][med]:.2f} dψ={mu['dpsi'][med]:.3f}",
              f"WORST patR²={r2p[worst]:.2f} ssim={mu['ssim'][worst]:.2f} dψ={mu['dpsi'][worst]:.3f}"]
    _panel(gt, pf, pu, idx, titles, psi, m, out / "figs" / "best_median_worst.png")

    # many-case montage (evenly spaced by rank)
    sel = order[np.linspace(0, len(order) - 1, args.montage).astype(int)]
    mtitles = [f"patR²(U)={r2p[i]:.2f} dψ={mu['dpsi'][i]:.2f}" for i in sel]
    _panel(gt, pf, pu, list(sel), mtitles, psi, m, out / "figs" / "montage.png")
    print("\nDONE ->", out)


if __name__ == "__main__":
    main()
