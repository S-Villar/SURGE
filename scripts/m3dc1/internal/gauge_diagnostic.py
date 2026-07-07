#!/usr/bin/env python3
"""Zero-GPU gauge-fix diagnostic: mean signed field before vs after gauge-fix.

Success criterion (printed explicitly):
  BEFORE gauge-fix: dataset mean signed Re(δp) is incoherent (low L2 / low ratio).
  AFTER gauge-fix: coherent structure survives in the mean (L2 ratio increases).

If criterion fails, exit code 1 — do not launch RZ training.

Usage:
  python scripts/m3dc1/internal/gauge_diagnostic.py --n-cases 500 \\
      --out-dir runs/gauge_diagnostic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "m3dc1"))

from gauge_fix import (  # noqa: E402
    _sanitize_field,
    build_template_from_gauge_fixed,
    gauge_fix_field,
    mean_signed_coherence,
)
from train_rz_field_image import _read_rz_case  # noqa: E402
from dataset_complex_v2 import find_complex_v2_files  # noqa: E402


def _load_complex_field(c: dict, time_idx: int = -1) -> np.ndarray:
    """Prefer p_hat complex; fallback p_phi0 + i·Im from p_hat unavailable."""
    # _read_rz_case only returns real; re-read complex from path if stored
    field = c.get("field_complex")
    if field is not None:
        return np.asarray(field, np.complex128)
    raise ValueError("no complex field in case dict")


def _read_complex_case(path: Path, time_idx: int = -1):
    import h5py
    from dataset_complex_v2 import _decode

    with h5py.File(path, "r") as f:
        rg = f["runs"][list(f["runs"].keys())[0]]
        pf = rg["pertfields"]
        z_axis = 0.0
        if "miller" in rg and "Z0" in rg["miller"]:
            z_axis = float(rg["miller"]["Z0"][()])
        elif "miller" in rg and "z0" in rg["miller"]:
            z_axis = float(rg["miller"]["z0"][()])
        key = "p_hat" if "p_hat" in pf else "p_phi0"
        arr = np.asarray(pf[key])
        field = arr[time_idx] if arr.ndim == 3 else arr
        if not np.iscomplexobj(field) and "p_phi0" in pf and "p_phiq" in pf:
            p0 = np.asarray(pf["p_phi0"])
            pq = np.asarray(pf["p_phiq"])
            p0 = p0[time_idx] if p0.ndim == 3 else p0
            pq = pq[time_idx] if pq.ndim == 3 else pq
            field = p0.astype(np.complex128) - 1j * pq.astype(np.complex128)
        elif not np.iscomplexobj(field):
            field = field.astype(np.complex128)
        R = Z = None
        mesh_id = rg.attrs.get("mesh_id", None)
        if mesh_id and "mesh" in f and mesh_id in f["mesh"]:
            mg = f["mesh"][mesh_id]
            if "R" in mg and "Z" in mg:
                R = np.asarray(mg["R"], float)
                Z = np.asarray(mg["Z"], float)
        return {
            "run_id": _decode(rg.get("runID", "run")),
            "eq_id": _decode(rg.get("eqID", "eq")),
            "field_complex": field,
            "R": R, "Z": Z, "z_axis": z_axis,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--n-cases", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--midplane-z", choices=["axis", "zero"], default="axis")
    ap.add_argument("--gauge-ref", choices=["peak", "template"], default="peak")
    ap.add_argument("--peak-window", type=int, default=3)
    ap.add_argument("--min-coherence-gain", type=float, default=2.0,
                    help="Require after/before coherence_ratio >= this to PASS.")
    ap.add_argument("--out-dir", default="runs/gauge_diagnostic")
    args = ap.parse_args()

    paths = find_complex_v2_files(args.batch_dir, filename=args.filename)
    rng = np.random.RandomState(args.seed)
    if args.n_cases and args.n_cases < len(paths):
        idx = rng.choice(len(paths), size=args.n_cases, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    print(f"Gauge diagnostic on {len(paths)} cases")

    raw_re: list[np.ndarray] = []
    gf_re: list[np.ndarray] = []
    gf_im: list[np.ndarray] = []
    thetas: list[float] = []
    keys: list[str] = []
    fields_for_template: list[np.ndarray] = []

    for i, p in enumerate(paths):
        try:
            c = _read_complex_case(Path(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        f = c["field_complex"]
        scale = float(np.nanmax(np.abs(f))) or 1.0
        fn = _sanitize_field(f / scale)
        raw_re.append(np.real(fn).astype(np.float32))

        if args.gauge_ref == "template" and len(fields_for_template) >= 32:
            tmpl = build_template_from_gauge_fixed(fields_for_template, max_cases=64)
            res = gauge_fix_field(
                fn, Z=c["Z"], z_axis=c["z_axis"], midplane_z=args.midplane_z,
                gauge_ref="template", peak_window=args.peak_window, template=tmpl,
            )
        else:
            res = gauge_fix_field(
                fn, Z=c["Z"], z_axis=c["z_axis"], midplane_z=args.midplane_z,
                gauge_ref="peak", peak_window=args.peak_window,
            )
        fields_for_template.append(res.field_gf)
        gf_re.append(np.real(res.field_gf).astype(np.float32))
        gf_im.append(np.imag(res.field_gf).astype(np.float32))
        thetas.append(res.theta_ref)
        keys.append(f"{c['run_id']}_{c['eq_id']}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)}")

    if len(raw_re) < 10:
        print("FAIL: too few cases loaded")
        sys.exit(1)

    raw_stack = np.stack(raw_re)
    gf_re_stack = np.stack(gf_re)
    before = mean_signed_coherence(raw_stack)
    after = mean_signed_coherence(gf_re_stack)

    gain = after["coherence_ratio"] / (before["coherence_ratio"] + 1e-30)
    passed = (
        after["l2_mean_field"] > before["l2_mean_field"] * 1.5
        and gain >= args.min_coherence_gain
        and after["peak_mean_field"] > before["peak_mean_field"] * 1.5
    )

    report = {
        "n_cases": len(raw_re),
        "midplane_z": args.midplane_z,
        "gauge_ref": args.gauge_ref,
        "peak_window": args.peak_window,
        "before": before,
        "after": after,
        "coherence_gain": float(gain),
        "theta_ref_mean": float(np.mean(thetas)),
        "theta_ref_std": float(np.std(thetas)),
        "verdict": "PASS" if passed else "FAIL",
        "success_criterion": (
            "mean signed field incoherent before (low L2 ratio); "
            "coherent structure after gauge-fix (L2 ratio gain >= "
            f"{args.min_coherence_gain})"
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gauge_diagnostic_report.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nshow = min(4, len(raw_re))
        fig, axes = plt.subplots(nshow + 1, 4, figsize=(14, 2.8 * (nshow + 1)))
        mu_raw = np.mean(raw_stack, axis=0)
        mu_gf = np.mean(gf_re_stack, axis=0)
        vmax = float(np.percentile(np.abs(mu_gf), 99) or 1.0)

        for col, (img, title) in enumerate([
            (mu_raw, "mean Re(δp) BEFORE"),
            (mu_gf, "mean Re(δp_gf) AFTER"),
            (np.mean(np.stack(gf_im), axis=0), "mean Im(δp_gf) AFTER"),
            (np.abs(mu_gf), "|mean Re(δp_gf)| AFTER"),
        ]):
            ax = axes[0, col]
            im = ax.imshow(img, origin="lower", aspect="auto", cmap="RdBu_r" if col < 3 else "viridis",
                           vmin=(-vmax if col < 3 else 0), vmax=vmax)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

        for r in range(nshow):
            vmax_c = float(np.percentile(np.abs(gf_re[r]), 99) or 1.0)
            for col, (img, title) in enumerate([
                (raw_re[r], "raw Re"),
                (gf_re[r], "gf Re"),
                (gf_im[r], "gf Im"),
                (gf_re[r] - raw_re[r], "Δ Re"),
            ]):
                ax = axes[r + 1, col]
                im = ax.imshow(img, origin="lower", aspect="auto", cmap="RdBu_r",
                               vmin=-vmax_c, vmax=vmax_c)
                if col == 0:
                    ax.set_ylabel(keys[r], fontsize=7)
                if r == 0:
                    ax.set_title(title, fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(
            f"Gauge diagnostic — {report['verdict']}  "
            f"coherence ratio {before['coherence_ratio']:.4f} → {after['coherence_ratio']:.4f} "
            f"(gain {gain:.2f}x)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "gauge_diagnostic_mean_fields.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_dir / 'gauge_diagnostic_mean_fields.png'}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\n" + "=" * 72)
    print("GAUGE DIAGNOSTIC VERDICT:", report["verdict"])
    print("=" * 72)
    print(f"  BEFORE  L2(mean)={before['l2_mean_field']:.6f}  "
          f"coherence_ratio={before['coherence_ratio']:.6f}  "
          f"peak(mean)={before['peak_mean_field']:.6f}")
    print(f"  AFTER   L2(mean)={after['l2_mean_field']:.6f}  "
          f"coherence_ratio={after['coherence_ratio']:.6f}  "
          f"peak(mean)={after['peak_mean_field']:.6f}")
    print(f"  coherence gain: {gain:.2f}x  (need >= {args.min_coherence_gain})")
    print(f"  θ_ref: mean={report['theta_ref_mean']:.3f} rad  std={report['theta_ref_std']:.3f}")
    print(f"\nSuccess criterion: {report['success_criterion']}")
    print(f"Report: {out_dir / 'gauge_diagnostic_report.json'}")

    if not passed:
        print("\nSTOP — gauge convention unstable; do not launch RZ training.")
        sys.exit(1)
    print("\nPASS — proceed to gauge-fixed RZ training.")


if __name__ == "__main__":
    main()
