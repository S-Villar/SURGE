#!/usr/bin/env python
"""Ground-truth vs predicted EIGENMODE FIELD reconstruction on the (R,Z) flux grid.

For the worst / median / best test cases of a trained spectrum-image model, this
reconstructs the poloidal field delta p(R,Z) from:
  - TRUE spectrum  : the case's own complex spectrum (magnitude + phase), and
  - PRED spectrum  : the model's predicted |delta p|(m, psi_N), rescaled to the
                     true peak amplitude and combined with the TRUE phase
                     (phase is an unlearnable per-eigenmode gauge), then inverse-
                     FFT'd along m and mapped to (R,Z) via m3dc1 flux coordinates.

The magnitude is what the model learns; the phase and the absolute scale are not
learnable, so we borrow them from the ground truth to isolate the *shape* the
surrogate got right/wrong. Produces a 3x3 panel (rows worst/median/best; cols
true field / predicted field / difference).

Run on a GPU compute node (FNO FFT differs on CPU):
  srun --jobid=<id> --overlap -n1 --gpus-per-node=1 \
    python scripts/m3dc1/internal/field_recon_compare.py \
      --run runs/spectrum_fno48_floor6_smooth1_qc --model fno2d \
      --ds-cache runs/compare_balance/dataset_g128_m-80.0_20.0_max_log10_fl6.0_sm1.0_exTrue.npz \
      --out runs/spectrum_fno48_floor6_smooth1_qc/field_recon
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
_REPO = _HERE.parents[3]
for p in (_REPO, _REPO / "m3dc1ml" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_spectrum_image as T  # noqa: E402


def _pc_pattern_r2(yt, yp):
    n = yt.shape[0]
    g = yt.reshape(n, -1); p = yp.reshape(n, -1)
    gd = g - g.mean(1, keepdims=True); pd = p - p.mean(1, keepdims=True)
    ss = ((gd - pd) ** 2).sum(1); tt = (gd ** 2).sum(1)
    return np.clip(np.where(tt > 1e-8, 1 - ss / np.maximum(tt, 1e-8), 0.0), -1.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="fno2d")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ds-cache", default=None, help="Prebuilt dataset .npz (X,Y,keys).")
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--m3dc1-code",
                    default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16/m3dc1_python_code")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    out = Path(args.out) if args.out else run / "field_recon"
    out.mkdir(parents=True, exist_ok=True)
    m_lo, m_hi = cfg["m_window"]; grid = cfg["grid"]

    exclude_keys = None
    if cfg.get("exclude_list"):
        raw = Path(cfg["exclude_list"]).read_text().strip()
        try:
            exclude_keys = set(json.loads(raw).keys())
        except Exception:
            exclude_keys = set(l.strip() for l in raw.splitlines() if l.strip())

    if args.ds_cache and Path(args.ds_cache).exists():
        print(f"Loading cached dataset {args.ds_cache}")
        z = np.load(args.ds_cache, allow_pickle=True)
        X, Y, keys = z["X"], z["Y"], z["keys"]
    else:
        X, Y, chan, keys = T.build_dataset(
            cfg["batch_dir"], cfg["filename"], cfg.get("n_cases", 0), grid,
            m_lo, m_hi, cfg["spectrum_field"], cfg.get("eps", 1e-12),
            target_norm=cfg["target_norm"], target_space=cfg["target_space"],
            target_floor=cfg.get("target_floor"), target_smooth=cfg.get("target_smooth"),
            exclude_keys=exclude_keys,
            geom_channels=cfg.get("geom_channels", False))
        keys = np.array(keys)
    N = X.shape[0]; keys = np.asarray(keys)

    rng = np.random.RandomState(cfg["seed"])
    perm = rng.permutation(N)
    n_test = int(cfg["test_frac"] * N); n_val = int(cfg["val_frac"] * N)
    te = perm[:n_test]; tr = perm[n_test + n_val:]

    xm = X[tr].mean((0, 2, 3), keepdims=True); xs = X[tr].std((0, 2, 3), keepdims=True) + 1e-8
    Xn = (X - xm) / xs
    ym = float(Y[tr].mean()); ysd = float(Y[tr].std() + 1e-8)

    import torch
    net = T._build_net(args.model, X.shape[1], fno_modes=cfg.get("fno_modes", 16),
                       fno_hidden=cfg.get("fno_hidden", 32), grid=grid)
    ck = torch.load(run / f"ckpt_{args.model}.pt", map_location=args.device)
    net.load_state_dict(ck["state_dict"]); net.to(args.device).eval()
    pred_std = T._predict_net(net, Xn[te], batch_size=cfg.get("batch_size", 16))
    yp_dex = pred_std * ysd + ym            # (n,H,W) log10 max-norm floored
    yt_dex = Y[te]
    pc_r2 = _pc_pattern_r2(yt_dex, yp_dex)
    order = np.argsort(pc_r2)
    picks = [("worst", int(order[0])), ("median", int(order[len(order) // 2])),
             ("best", int(order[-1]))]
    print("picks:", [(l, str(keys[te][i]), round(float(pc_r2[i]), 3)) for l, i in picks])

    # m3dc1ml reconstruction helpers
    from m3dc1ml.io.sdata import load_complex_v2_case
    from m3dc1ml.viz import explore_case as EC
    from scipy.interpolate import RegularGridInterpolator

    m_grid = np.linspace(m_lo, m_hi, grid)
    psi_grid = np.linspace(0.0, 1.0, grid)

    def pred_field_and_grid(key, img_dex):
        """Reconstruct true & predicted delta p(R,Z) for one case."""
        rid = key.split("_")[0]; eqid = key[len(rid) + 1:]
        fp = Path(args.batch_dir) / rid / eqid / cfg["filename"]
        b = load_complex_v2_case(fp, spectrum_field=cfg["spectrum_field"])
        # true reconstruction
        true_field = EC.recon_real_from_spectrum(b)
        R, Z = EC._flux_grid_for_bundle(b, m3dc1_python_code=Path(args.m3dc1_code))
        # predicted magnitude -> native grid, x true peak, x true phase
        mag_norm = np.power(10.0, img_dex)                # (H=m, W=psi), peak~1
        rgi = RegularGridInterpolator((m_grid, psi_grid), mag_norm,
                                      bounds_error=False, fill_value=0.0)
        nm = np.asarray(b["m_modes"], float); npsi = np.asarray(b["psi_norm"], float)
        MM, PP = np.meshgrid(nm, npsi, indexing="ij")
        mag_pred = rgi(np.stack([MM.ravel(), PP.ravel()], 1)).reshape(len(nm), len(npsi))
        phase = np.angle(b["spec_complex"]) if np.iscomplexobj(b["spec_complex"]) else 0.0
        b_pred = dict(b)
        b_pred["spec_complex"] = mag_pred * np.exp(1j * phase)
        pred_field = EC.recon_real_from_spectrum(b_pred)
        # Normalize BOTH fields by their own max |amplitude| so the comparison is
        # a pure shape comparison against the (max-normalized) ground-truth data
        # (absolute scale and phase are unlearnable per-eigenmode gauges).
        true_field = true_field / (np.abs(true_field).max() + 1e-30)
        pred_field = pred_field / (np.abs(pred_field).max() + 1e-30)
        return R, Z, true_field, pred_field, b.get("gamma", float("nan"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12))
    summ = {}
    for r, (label, i) in enumerate(picks):
        key = str(keys[te][i])
        try:
            R, Z, ftrue, fpred, gamma = pred_field_and_grid(key, yp_dex[i])
        except Exception as exc:
            print(f"  [{label}] {key} recon FAILED: {exc}")
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"{label}\n{key}\nrecon failed",
                                ha="center", va="center", transform=axes[r, c].transAxes)
            continue
        diff = fpred - ftrue
        vmax = 1.0                       # fields are max-normalized to unit amplitude
        rl2 = float(np.linalg.norm(diff) / (np.linalg.norm(ftrue) + 1e-30))
        summ[label] = {"key": key, "pattern_r2": float(pc_r2[i]),
                       "field_relL2": rl2, "gamma": float(gamma)}
        for c, (fld, title, vl) in enumerate([
                (ftrue, "true field", (-vmax, vmax)),
                (fpred, "pred field", (-vmax, vmax)),
                (diff, "difference", (-vmax, vmax))]):
            ax = axes[r, c]
            pc = ax.pcolormesh(R, Z, fld, shading="auto", cmap="RdBu_r",
                               vmin=vl[0], vmax=vl[1])
            ax.set_aspect("equal", adjustable="box")
            if r == 0:
                ax.set_title(title, fontsize=13)
            if c == 0:
                ax.set_ylabel(f"{label}\n{key}\npatR2={pc_r2[i]:.3f} relL2={rl2:.2f}\nZ",
                              fontsize=9)
            if r == 2:
                ax.set_xlabel("R")
            plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{run.name} [{args.model}]  eigenmode field: true vs predicted "
                 f"(each max-normalized; pred |dp| + true phase)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fpng = out / f"field_recon_{args.model}.png"
    fig.savefig(fpng, dpi=130); plt.close(fig)
    (out / "field_recon.json").write_text(json.dumps(summ, indent=2))
    print(f"\nWrote {fpng}")
    for k, v in summ.items():
        print(f"  {k:7s} {v['key']:22s} patR2={v['pattern_r2']:.3f} relL2={v['field_relL2']:.3f}")


if __name__ == "__main__":
    main()
