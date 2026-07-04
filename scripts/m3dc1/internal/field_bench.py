#!/usr/bin/env python3
"""Batched field-reconstruction benchmark for spectrum-image FNO runs.

Ranks training experiments by field relL2 (not spectrum patR²) using cached
predictions from ``predictions_cache.npz``. Reuses the true-phase injection +
IFFT + max-normalization convention from ``field_recon_compare.py``.

Usage:
  python scripts/m3dc1/internal/field_bench.py \\
    --runs runs/spectrum_fno48_floor6_smooth1_qc \\
           runs/spectrum_fno48_floor6_smooth1_qc_peak4 \\
           runs/spectrum_fno48_floor6_smooth1_qc_mhi100 \\
    --split test --device cuda --out field_bench
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
_REPO = _HERE.parents[3]
for p in (_REPO, _REPO / "m3dc1ml" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

REGRESSION_TOL = 5e-3
REL_BINS = [0.3, 0.5, 0.7, 1.0]
REL_BIN_LABELS = ["<0.3", "<0.5", "<0.7", "<1.0", ">1.0"]


@dataclass
class CaseMeta:
    phase: np.ndarray
    m_modes: np.ndarray
    psi_norm: np.ndarray
    spec_field: str
    true_spec: np.ndarray


def parse_family(key: str) -> str:
    """Parse equilibrium family id, e.g. run21_sparc_1524 -> sparc_1524."""
    parts = key.split("_", 1)
    return parts[1] if len(parts) > 1 else key


def pattern_r2(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt.reshape(-1).astype(np.float64)
    p = pred.reshape(-1).astype(np.float64)
    gd = g - g.mean()
    pd = p - p.mean()
    ss = float(((gd - pd) ** 2).sum())
    tt = float((gd ** 2).sum())
    if tt <= 1e-8:
        return 0.0
    return float(np.clip(1.0 - ss / tt, -1.0, 1.0))


def _to_full_m_grid_complex(values_2d: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    m_vals = np.asarray(m_values, dtype=int)
    v2d = np.asarray(values_2d)
    m_min, m_max = int(m_vals.min()), int(m_vals.max())
    m_full = np.arange(m_min, m_max + 1, dtype=int)
    out = np.zeros((len(m_full), v2d.shape[1]), dtype=np.complex128)
    idx = {int(m): i for i, m in enumerate(m_full)}
    for i, m in enumerate(m_vals):
        out[idx[int(m)], :] = v2d[i, :]
    return out


def _ifft_field(spec_complex: np.ndarray, m_modes: np.ndarray) -> np.ndarray:
    spec_full = _to_full_m_grid_complex(spec_complex, m_modes)
    recon_hat = np.fft.ifft(np.fft.ifftshift(spec_full, axes=0), axis=0) * spec_full.shape[0]
    return np.real(recon_hat)


def _ifft_fields_torch(
    specs: Sequence[np.ndarray],
    m_modes_list: Sequence[np.ndarray],
    device: str,
) -> np.ndarray:
    """Batched inverse-FFT along m on GPU (padded to common shape within batch)."""
    import torch

    dev = torch.device(device)
    if not specs:
        return np.zeros((0, 1, 1), dtype=np.float64)
    spans = [int(m.max()) - int(m.min()) + 1 for m in m_modes_list]
    n_psis = [spec.shape[1] for spec in specs]
    m_pad, p_pad = max(spans), max(n_psis)
    batch = torch.zeros((len(specs), m_pad, p_pad), dtype=torch.complex64, device=dev)
    for i, (spec, m_vals) in enumerate(zip(specs, m_modes_list)):
        spec_full = _to_full_m_grid_complex(spec, m_vals)
        t = torch.from_numpy(spec_full).to(dev)
        t = torch.fft.ifft(torch.fft.ifftshift(t, dim=0), dim=0) * t.shape[0]
        batch[i, : t.shape[0], : t.shape[1]] = t
    return batch.real.cpu().numpy()


def interp_mag_to_native(
    img_dex: np.ndarray,
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
    meta: CaseMeta,
) -> np.ndarray:
    mag_norm = np.power(10.0, img_dex.astype(np.float64))
    rgi = RegularGridInterpolator(
        (m_grid, psi_grid),
        mag_norm,
        bounds_error=False,
        fill_value=0.0,
    )
    nm = meta.m_modes.astype(np.float64)
    npsi = meta.psi_norm.astype(np.float64)
    mm, pp = np.meshgrid(nm, npsi, indexing="ij")
    return rgi(np.stack([mm.ravel(), pp.ravel()], 1)).reshape(len(nm), len(npsi))


def pred_spec_from_dex(
    img_dex: np.ndarray,
    meta: CaseMeta,
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
) -> np.ndarray:
    mag_nat = interp_mag_to_native(img_dex, m_grid, psi_grid, meta)
    return mag_nat * np.exp(1j * meta.phase)


def load_case_meta(path: str, spec_field: str, cache: Dict[str, CaseMeta]) -> CaseMeta:
    if path in cache:
        return cache[path]
    from m3dc1ml.io.sdata import load_complex_v2_case

    b = load_complex_v2_case(path, spectrum_field=spec_field)
    spec = np.asarray(b["spec_complex"])
    phase = np.angle(spec) if np.iscomplexobj(spec) else np.zeros_like(spec, float)
    meta = CaseMeta(
        phase=phase,
        m_modes=np.asarray(b["m_modes"], float),
        psi_norm=np.asarray(b["psi_norm"], float),
        spec_field=spec_field,
        true_spec=spec,
    )
    cache[path] = meta
    return meta


def max_norm_field(field: np.ndarray) -> np.ndarray:
    return field / (np.abs(field).max() + 1e-30)


def rel_l2_raw(pred: np.ndarray, true: np.ndarray) -> float:
    ftrue = max_norm_field(true)
    fpred = max_norm_field(pred)
    diff = fpred - ftrue
    return float(np.linalg.norm(diff) / (np.linalg.norm(ftrue) + 1e-30))


def rel_l2_alpha(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    ftrue = max_norm_field(true)
    fpred = max_norm_field(pred)
    num = float(np.vdot(fpred.ravel(), ftrue.ravel()))
    den = float(np.vdot(fpred.ravel(), fpred.ravel())) + 1e-30
    alpha = num / den
    scaled = alpha * fpred
    diff = scaled - ftrue
    return float(np.linalg.norm(diff) / (np.linalg.norm(ftrue) + 1e-30)), alpha


def compute_crf(
    gt_dex: np.ndarray,
    pred_dex: np.ndarray,
    *,
    target_space: str = "log10",
    cutoff: float = 0.25,
) -> float:
    if target_space == "log10":
        gt = np.power(10.0, gt_dex.astype(np.float64))
        pred = np.power(10.0, pred_dex.astype(np.float64))
    else:
        gt = np.clip(gt_dex.astype(np.float64), 0.0, None)
        pred = np.clip(pred_dex.astype(np.float64), 0.0, None)
    residual = np.abs(pred) - np.abs(gt)
    power = np.abs(np.fft.fft2(residual)) ** 2
    total = float(power.sum())
    if total <= 1e-30:
        return 0.0
    h, w = residual.shape
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    ky_g, kx_g = np.meshgrid(ky, kx, indexing="ij")
    kr = np.sqrt(kx_g ** 2 + ky_g ** 2)
    thr = float(np.quantile(kr.ravel(), cutoff))
    mask = kr <= thr
    return float(power[mask].sum() / total)


def bin_counts(values: np.ndarray) -> Dict[str, int]:
    v = np.asarray(values, float)
    counts = {
        REL_BIN_LABELS[0]: int((v < REL_BINS[0]).sum()),
        REL_BIN_LABELS[1]: int(((v >= REL_BINS[0]) & (v < REL_BINS[1])).sum()),
        REL_BIN_LABELS[2]: int(((v >= REL_BINS[1]) & (v < REL_BINS[2])).sum()),
        REL_BIN_LABELS[3]: int(((v >= REL_BINS[2]) & (v < REL_BINS[3])).sum()),
        REL_BIN_LABELS[4]: int((v >= REL_BINS[3]).sum()),
    }
    return counts


def aggregate_metrics(rel: np.ndarray, rel_a: np.ndarray, crf: np.ndarray) -> Dict[str, float]:
    rel = np.asarray(rel, float)
    rel_a = np.asarray(rel_a, float)
    crf = np.asarray(crf, float)
    return {
        "n": int(rel.size),
        "mean_relL2": float(np.mean(rel)),
        "median_relL2": float(np.median(rel)),
        "p90_relL2": float(np.percentile(rel, 90)),
        "frac_relL2_gt_1": float(np.mean(rel > 1.0)),
        "mean_relL2_alpha": float(np.mean(rel_a)),
        "median_relL2_alpha": float(np.median(rel_a)),
        "p90_relL2_alpha": float(np.percentile(rel_a, 90)),
        "frac_relL2_alpha_gt_1": float(np.mean(rel_a > 1.0)),
        "mean_crf": float(np.mean(crf)),
        "relL2_bins": bin_counts(rel),
    }


def stratified_subset(keys: Sequence[str], n: int, seed: int = 42) -> List[str]:
    by_fam: Dict[str, List[str]] = defaultdict(list)
    for k in keys:
        by_fam[parse_family(k)].append(k)
    fams = sorted(by_fam)
    rng = np.random.RandomState(seed)
    per = max(1, n // max(len(fams), 1))
    picked: List[str] = []
    for fam in fams:
        pool = sorted(by_fam[fam])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) < n:
        rest = [k for k in keys if k not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def load_run_cache(run_dir: Path, split: str) -> Dict:
    cache_path = run_dir / "predictions_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"missing {cache_path}")
    z = np.load(cache_path, allow_pickle=True)
    mask = z["split"].astype(str) == split
    return {
        "run": run_dir.name,
        "run_dir": run_dir,
        "keys": z["keys"].astype(str)[mask],
        "paths": z["paths"].astype(str)[mask],
        "gt": z["gt"][mask].astype(np.float32),
        "pred": z["pred"][mask].astype(np.float32),
        "m_grid": z["m_grid"].astype(np.float64),
        "psi_grid": z["psi_grid"].astype(np.float64),
        "target_space": str(z.get("target_space", "log10")),
        "spectrum_field": str(z.get("spectrum_field", "p")),
        "r2_pattern_cache": z["r2_pattern"][mask].astype(np.float32)
        if "r2_pattern" in z else None,
    }


def maybe_reexport(run_dir: Path, device: str) -> None:
    cache_path = run_dir / "predictions_cache.npz"
    if cache_path.exists():
        return
    cmd = [
        sys.executable,
        str(_HERE.parent / "export_predictions_cache.py"),
        "--run", str(run_dir),
        "--device", device,
        "--splits", "val", "test",
    ]
    print(f"[reexport] {run_dir}")
    subprocess.check_call(cmd)


def precompute_true_fields(
    paths: Sequence[str],
    spec_field: str,
    device: str,
    batch_size: int,
    meta_cache: Dict[str, CaseMeta],
) -> np.ndarray:
    import torch

    use_torch = device == "cuda" and torch.cuda.is_available()
    fields: List[np.ndarray] = []
    n = len(paths)
    t0 = time.time()
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        metas = [load_case_meta(paths[i], spec_field, meta_cache) for i in range(b0, b1)]
        specs = [m.true_spec for m in metas]
        mlist = [m.m_modes for m in metas]
        if use_torch:
            batch = _ifft_fields_torch(specs, mlist, device)
        else:
            batch = np.stack([_ifft_field(s, m) for s, m in zip(specs, mlist)], axis=0)
        fields.append(batch)
        print(f"  true fields {b1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(fields, axis=0)


def process_model(
    keys: Sequence[str],
    gt_all: np.ndarray,
    pred_all: np.ndarray,
    true_fields: np.ndarray,
    metas: Sequence[CaseMeta],
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
    target_space: str,
    device: str,
    batch_size: int,
    crf_cutoff: float,
) -> List[Dict]:
    import torch

    use_torch = device == "cuda" and torch.cuda.is_available()
    rows: List[Dict] = []
    n = len(keys)
    t0 = time.time()
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        pred_specs = [
            pred_spec_from_dex(pred_all[i], metas[i], m_grid, psi_grid)
            for i in range(b0, b1)
        ]
        mlist = [metas[i].m_modes for i in range(b0, b1)]
        if use_torch:
            fpreds = _ifft_fields_torch(pred_specs, mlist, device)
        else:
            fpreds = np.stack(
                [_ifft_field(s, m) for s, m in zip(pred_specs, mlist)], axis=0
            )
        for j, i in enumerate(range(b0, b1)):
            ftrue = true_fields[i]
            fpred = fpreds[j]
            rel = rel_l2_raw(fpred, ftrue)
            rel_a, alpha = rel_l2_alpha(fpred, ftrue)
            crf = compute_crf(
                gt_all[i], pred_all[i], target_space=target_space, cutoff=crf_cutoff
            )
            rows.append({
                "key": keys[i],
                "family": parse_family(keys[i]),
                "patR2": pattern_r2(gt_all[i], pred_all[i]),
                "relL2": rel,
                "relL2_alpha": rel_a,
                "alpha": alpha,
                "crf": crf,
            })
        print(f"  pred fields {b1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def regression_check(run_dir: Path, device: str, crf_cutoff: float) -> None:
    ref_json = run_dir / "field_recon" / "field_recon.json"
    if not ref_json.exists():
        print(f"[regression] skip: no {ref_json}")
        return
    ref = json.loads(ref_json.read_text())
    cache = load_run_cache(run_dir, "test")
    key_to_idx = {k: i for i, k in enumerate(cache["keys"])}
    meta_cache: Dict[str, CaseMeta] = {}
    print(f"[regression] checking {run_dir.name} against {ref_json}")
    for label, rec in ref.items():
        key = rec["key"]
        if key not in key_to_idx:
            raise AssertionError(f"regression key missing from cache: {key}")
        i = key_to_idx[key]
        meta = load_case_meta(cache["paths"][i], cache["spectrum_field"], meta_cache)
        ftrue = _ifft_field(meta.true_spec, meta.m_modes)
        pred_spec = pred_spec_from_dex(
            cache["pred"][i], meta, cache["m_grid"], cache["psi_grid"]
        )
        fpred = _ifft_field(pred_spec, meta.m_modes)
        rel = rel_l2_raw(fpred, ftrue)
        ref_rel = float(rec["field_relL2"])
        if abs(rel - ref_rel) > REGRESSION_TOL:
            raise AssertionError(
                f"regression failed {label} {key}: got relL2={rel:.6f}, ref={ref_rel:.6f}"
            )
        print(f"  OK {label:6s} {key:22s} relL2={rel:.4f} (ref {ref_rel:.4f})")


def plot_rel_hist(values: np.ndarray, title: str, out: Path) -> None:
    v = np.asarray(values, float)
    counts = [bin_counts(v)[lbl] for lbl in REL_BIN_LABELS]
    colors = ["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(REL_BIN_LABELS, counts, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_ylabel("case count")
    ax.set_xlabel("field relL2")
    ax.set_title(title)
    for bar, c in zip(bars, counts):
        if c:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def rank_models(model_stats: List[Dict]) -> List[Dict]:
    return sorted(
        model_stats,
        key=lambda s: (s["frac_relL2_gt_1"], s["p90_relL2"], s["median_relL2"]),
    )


def print_leaderboard(ranked: List[Dict]) -> None:
    hdr = (
        f"{'rank':>4}  {'model':<42}  {'frac>1':>7}  {'p90':>7}  "
        f"{'median':>7}  {'mean':>7}  {'mean_a':>7}  {'CRF':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r, s in enumerate(ranked, 1):
        print(
            f"{r:4d}  {s['model']:<42}  {s['frac_relL2_gt_1']:7.3f}  "
            f"{s['p90_relL2']:7.3f}  {s['median_relL2']:7.3f}  "
            f"{s['mean_relL2']:7.3f}  {s['mean_relL2_alpha']:7.3f}  "
            f"{s['mean_crf']:6.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default="field_bench", help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Cases per GPU batch for IFFT")
    ap.add_argument("--subset", type=int, default=0,
                    help="Evaluate only N cases (0 = all)")
    ap.add_argument("--stratify-family", action="store_true",
                    help="With --subset, pick family-balanced cases")
    ap.add_argument("--crf-cutoff", type=float, default=0.25,
                    help="CRF low-k radial band quantile (default 0.25)")
    ap.add_argument("--reexport", action="store_true",
                    help="Run export_predictions_cache if cache missing")
    ap.add_argument("--regression-check", action="store_true",
                    help="Assert relL2 matches field_recon.json on known cases")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(r) for r in args.runs]
    if args.reexport:
        for rd in run_dirs:
            maybe_reexport(rd, args.device)

    if args.regression_check:
        for rd in run_dirs:
            regression_check(rd, args.device, args.crf_cutoff)

    caches = []
    for rd in run_dirs:
        split = args.split
        if split == "all":
            z = np.load(rd / "predictions_cache.npz", allow_pickle=True)
            caches.append({
                "run": rd.name,
                "run_dir": rd,
                "keys": z["keys"].astype(str),
                "paths": z["paths"].astype(str),
                "gt": z["gt"].astype(np.float32),
                "pred": z["pred"].astype(np.float32),
                "m_grid": z["m_grid"].astype(np.float64),
                "psi_grid": z["psi_grid"].astype(np.float64),
                "target_space": str(z.get("target_space", "log10")),
                "spectrum_field": str(z.get("spectrum_field", "p")),
            })
        else:
            caches.append(load_run_cache(rd, split))

    common_keys = set(caches[0]["keys"])
    for c in caches[1:]:
        common_keys &= set(c["keys"])
    keys_sorted = sorted(common_keys)
    if args.subset and args.subset < len(keys_sorted):
        if args.stratify_family:
            keys_sorted = stratified_subset(keys_sorted, args.subset, args.seed)
        else:
            rng = np.random.RandomState(args.seed)
            keys_sorted = sorted(rng.choice(keys_sorted, args.subset, replace=False).tolist())
    print(f"Benchmarking {len(keys_sorted)} cases x {len(caches)} models", flush=True)

    meta_cache: Dict[str, CaseMeta] = {}
    # Shared paths/metas — spectrum_field identical across compared runs
    ref = caches[0]
    key_to_ref = {k: i for i, k in enumerate(ref["keys"])}
    ref_paths = [ref["paths"][key_to_ref[k]] for k in keys_sorted]
    metas = [
        load_case_meta(ref_paths[i], ref["spectrum_field"], meta_cache)
        for i in range(len(keys_sorted))
    ]
    print("Precomputing true fields (once per case)...", flush=True)
    true_fields = precompute_true_fields(
        ref_paths, ref["spectrum_field"], args.device, args.batch_size, meta_cache
    )

    per_model_rows: Dict[str, List[Dict]] = {}
    rel_matrix: Dict[str, Dict[str, float]] = {k: {} for k in keys_sorted}

    for cache in caches:
        model = cache["run"]
        print(f"\n=== {model} ===", flush=True)
        key_to_idx = {k: i for i, k in enumerate(cache["keys"])}
        idxs = [key_to_idx[k] for k in keys_sorted]
        gt = cache["gt"][idxs]
        pred = cache["pred"][idxs]
        rows = process_model(
            keys_sorted, gt, pred, true_fields, metas,
            cache["m_grid"], cache["psi_grid"],
            cache["target_space"], args.device, args.batch_size, args.crf_cutoff,
        )
        per_model_rows[model] = rows
        for row in rows:
            rel_matrix[row["key"]][model] = row["relL2"]

    # per_case.csv (wide: one row per case, relL2 per model)
    model_names = [c["run"] for c in caches]
    per_case_path = out_dir / "per_case.csv"
    fieldnames = ["key", "family"] + [f"relL2_{m}" for m in model_names]
    fieldnames += [f"relL2_alpha_{m}" for m in model_names]
    fieldnames += [f"patR2_{m}" for m in model_names]
    fieldnames += [f"crf_{m}" for m in model_names]
    fieldnames += ["winner"]
    with per_case_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in keys_sorted:
            row = {"key": key, "family": parse_family(key)}
            rels = {}
            for m in model_names:
                rec = next(r for r in per_model_rows[m] if r["key"] == key)
                row[f"relL2_{m}"] = f"{rec['relL2']:.6f}"
                row[f"relL2_alpha_{m}"] = f"{rec['relL2_alpha']:.6f}"
                row[f"patR2_{m}"] = f"{rec['patR2']:.6f}"
                row[f"crf_{m}"] = f"{rec['crf']:.6f}"
                rels[m] = rec["relL2"]
            winner = min(rels, key=rels.get)
            row["winner"] = winner
            w.writerow(row)
    print(f"Wrote {per_case_path}")

    # win/loss counts
    win_counts = {m: 0 for m in model_names}
    for key in keys_sorted:
        rels = rel_matrix[key]
        win_counts[min(rels, key=rels.get)] += 1
    pairwise = {}
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            wins_a = wins_b = ties = 0
            for key in keys_sorted:
                ra, rb = rel_matrix[key][ma], rel_matrix[key][mb]
                if ra < rb - 1e-12:
                    wins_a += 1
                elif rb < ra - 1e-12:
                    wins_b += 1
                else:
                    ties += 1
            pairwise[f"{ma}_vs_{mb}"] = {
                f"{ma}_wins": wins_a,
                f"{mb}_wins": wins_b,
                "ties": ties,
            }

    # per-model aggregates + per-family
    model_stats = []
    per_family_rows = []
    for m in model_names:
        rows = per_model_rows[m]
        stats = aggregate_metrics(
            [r["relL2"] for r in rows],
            [r["relL2_alpha"] for r in rows],
            [r["crf"] for r in rows],
        )
        stats["model"] = m
        model_stats.append(stats)
        plot_rel_hist(
            np.array([r["relL2"] for r in rows]),
            f"{m}  field relL2 (n={len(rows)})",
            out_dir / f"relL2_hist_{m}.png",
        )
        by_fam: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            by_fam[r["family"]].append(r)
        for fam, fr in sorted(by_fam.items()):
            fa = aggregate_metrics(
                [x["relL2"] for x in fr],
                [x["relL2_alpha"] for x in fr],
                [x["crf"] for x in fr],
            )
            per_family_rows.append({
                "model": m,
                "family": fam,
                **{k: v for k, v in fa.items() if k != "relL2_bins"},
                "relL2_bins": fa["relL2_bins"],
            })

    ranked = rank_models(model_stats)
    leaderboard = {
        "split": args.split,
        "n_cases": len(keys_sorted),
        "models": model_stats,
        "ranking": [s["model"] for s in ranked],
        "win_counts": win_counts,
        "pairwise": pairwise,
        "crf_cutoff": args.crf_cutoff,
    }
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    per_family_path = out_dir / "per_family.csv"
    fam_fields = [
        "model", "family", "n", "mean_relL2", "median_relL2", "p90_relL2",
        "frac_relL2_gt_1", "mean_relL2_alpha", "mean_crf",
    ]
    with per_family_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fam_fields, extrasaction="ignore")
        w.writeheader()
        for row in per_family_rows:
            w.writerow(row)
    print(f"Wrote {out_dir / 'leaderboard.json'}")
    print(f"Wrote {per_family_path}")
    print_leaderboard(ranked)


if __name__ == "__main__":
    main()
