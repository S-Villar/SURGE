#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-case delta-p spectrum reconstruction on the **test** split.

For each test **case** (all inputs except poloidal mode *m*), this script:
  1. Loads the saved SURGE model + input/output scalers from a run directory.
  2. Reproduces the same train/val/test row indices as the workflow (seed, fractions).
  3. For every **test case** (same physical parameters except *m*), predicts **every** *m*
     present for that case (``--m-source case``, default) or the intersection of
     ``[m_min, m_max]`` with those modes (``--m-source range``).
  4. Reconstructs one psi-profile per case (``sum`` or ``rss`` over modes). Computes
     **pooled R^2 / RMSE** over all (case, psi) points of that reconstruction (not per-row
     per-m metrics). Also writes **per-case** R^2 for diagnostics.
  5. Writes ``per_case_recon_metrics.csv``, ``per_case_recon_summary.json``, optional plots.

**Model artifact:** ``runs/<tag>/models/<model_name>.joblib`` (from ``workflow_summary.json``).
If that file is missing, re-run the workflow or sync ``models/`` from the machine that trained.

Usage:
  # Use the same Python as SURGE training (has pandas, torch, joblib, ...):
  conda activate surge
  python scripts/m3dc1/eval_per_case_delta_p_recon.py runs/m3dc1_delta_p_per_mode \\
      --model mlp_per_mode --m-min -100 --m-max 100

  # Perlmutter (no activate): full path to env python
  /global/cfs/projectdirs/m3716/software/asvillar/envs/surge/bin/python \\
      scripts/m3dc1/eval_per_case_delta_p_recon.py runs/m3dc1_delta_p_per_mode_cfs \\
      --model mlp_per_mode --m-source case --combine rss

  # Or from repo: source scripts/m3dc1/surge_slurm_env.sh && surge_slurm_setup_python

Requires: numpy, pandas, scikit-learn, matplotlib; joblib; torch for torch.mlp models;
pyarrow for Parquet datasets; PyYAML optional (falls back to defaults if spec incomplete).

Compatible with Python 3.6+ (no postponed evaluation annotations).
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np
    import pandas as pd
except ImportError as _e:
    _env = "/global/cfs/projectdirs/m3716/software/asvillar/envs/surge/bin/python"
    sys.stderr.write(
        "Missing scientific Python stack (need at least numpy, pandas).\n"
        "Activate your SURGE conda env, e.g.:\n"
        "  conda activate surge\n"
        "or run this script with:\n"
        "  {_env} {argv}\n".format(_env=_env, argv=" ".join(sys.argv))
    )
    raise SystemExit(1) from _e
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Repo root: scripts/m3dc1 -> ../..
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from surge.io.load_compat import load_model_compat  # noqa: E402

INPUT_KEYS = [
    "eq_R0",
    "eq_a",
    "eq_kappa",
    "eq_delta",
    "q0",
    "q95",
    "qmin",
    "p0",
    "n",
    "m",
    "input_pscale",
    "input_batemanscale",
]


def _resolve_repo_path(run_dir: Path, p: Union[Path, str]) -> Path:
    path = Path(p)
    if path.is_file():
        return path.resolve()
    cand = run_dir.parent.parent / path
    if cand.is_file():
        return cand.resolve()
    cand2 = run_dir / path.name
    if cand2.is_file():
        return cand2.resolve()
    return path


def _load_spec(run_dir: Path) -> Dict[str, Any]:
    spec_p = run_dir / "spec.yaml"
    if not spec_p.is_file():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(spec_p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _raw_splits(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    *,
    test_fraction: float,
    val_fraction: float,
    random_state: int,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(df[list(input_cols)].values, dtype=np.float64)
    y = np.ascontiguousarray(df[list(output_cols)].values, dtype=np.float64)
    indices = np.asarray(df.index)
    tf = max(0.0, min(0.9, test_fraction))
    vf = max(0.0, min(0.9, val_fraction))
    if tf > 0:
        X_tv, X_te, y_tv, y_te, idx_tv, idx_te = train_test_split(
            X, y, indices, test_size=tf, random_state=random_state, shuffle=shuffle
        )
    else:
        X_tv, y_tv, idx_tv = X, y, indices
        idx_te = np.array([], dtype=indices.dtype)

    if vf > 0 and tf < 1.0:
        rel = vf / (1.0 - tf)
        rel = max(0.0, min(0.9, rel))
        _, _, _, _, idx_tr, idx_va = train_test_split(
            X_tv, y_tv, idx_tv, test_size=rel, random_state=random_state, shuffle=shuffle
        )
    else:
        idx_tr = idx_tv
        idx_va = np.array([], dtype=indices.dtype)

    return idx_tr, idx_va, idx_te


def _group_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in INPUT_KEYS if c != "m" and c in df.columns]


def _combine_modes(profiles: np.ndarray, how: str) -> np.ndarray:
    if profiles.size == 0:
        return np.array([])
    if how == "sum":
        return np.sum(profiles, axis=0)
    if how == "rss":
        return np.sqrt(np.sum(profiles**2, axis=0))
    raise ValueError(how)


def _load_saved_scalers(
    run_dir: Path, summary: Dict[str, Any]
) -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    scal = summary.get("scalers") or {}
    import joblib

    in_p = scal.get("input_scaler")
    out_p = scal.get("output_scaler")
    in_sc = joblib.load(_resolve_repo_path(run_dir, in_p)) if in_p else None
    out_sc = joblib.load(_resolve_repo_path(run_dir, out_p)) if out_p else None
    return in_sc, out_sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="SURGE run directory")
    ap.add_argument("--model", required=True, help="Model name (e.g. mlp_per_mode)")
    ap.add_argument(
        "--m-source",
        choices=("range", "case"),
        default="case",
        help="case: all m in dataset for each test case (default); range: [m_min,m_max] intersect case",
    )
    ap.add_argument("--m-min", type=int, default=-100)
    ap.add_argument("--m-max", type=int, default=100)
    ap.add_argument("--combine", choices=("sum", "rss"), default="sum")
    ap.add_argument("--max-cases", type=int, default=None, help="Limit cases (debug)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run_dir>/plots/per_case_recon)",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "workflow_summary.json"
    if not summary_path.is_file():
        print(f"Missing {summary_path}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spec = _load_spec(run_dir)

    model_entry: Optional[Dict[str, Any]] = None
    for m in summary.get("models") or []:
        if m.get("name") == args.model:
            model_entry = m
            break
    if model_entry is None:
        print(f"Model {args.model!r} not in workflow_summary models list", file=sys.stderr)
        return 1

    art = (model_entry.get("artifacts") or {}).get("model", "")
    model_path = _resolve_repo_path(run_dir, art)
    if not model_path.is_file():
        print(
            f"Model file not found: {model_path}\n"
            f"Expected saved artifact from training (e.g. models/{args.model}.joblib).",
            file=sys.stderr,
        )
        return 1

    me_load = dict(model_entry)
    if str(me_load.get("backend", "")).lower() == "torch":
        me_load["backend"] = "pytorch"

    try:
        adapter = load_model_compat(model_path, me_load)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        return 1

    dsinfo = summary.get("dataset") or {}
    fp = dsinfo.get("file_path", "")
    dataset_path = _resolve_repo_path(run_dir, fp)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    meta_path = spec.get("metadata_path") or (run_dir.parent.parent / "data/datasets/SPARC/delta_p_per_mode_metadata.yaml")
    meta_path = _resolve_repo_path(run_dir, meta_path)
    meta = _load_yaml(meta_path) if meta_path.is_file() else {}
    input_cols = [c for c in (meta.get("inputs") or []) if c]
    output_cols = [c for c in (meta.get("outputs") or []) if c]
    if not input_cols or not output_cols:
        print("Could not read input/output columns from metadata YAML.", file=sys.stderr)
        return 1

    if dataset_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(dataset_path, columns=[c for c in input_cols + output_cols if c])
    elif dataset_path.suffix.lower() in (".pkl", ".pickle"):
        df = pd.read_pickle(dataset_path)
        df = df[input_cols + output_cols]
    else:
        df = pd.read_csv(dataset_path)
        df = df[input_cols + output_cols]

    sample_rows = spec.get("sample_rows")
    seed = int(spec.get("seed", 42))
    if sample_rows is not None and int(sample_rows) > 0 and len(df) > int(sample_rows):
        df = df.sample(n=int(sample_rows), random_state=seed)

    test_fraction = float(spec.get("test_fraction", 0.2))
    val_fraction = float(spec.get("val_fraction", 0.2))
    shuffle = bool(spec.get("shuffle", True))
    std_in = bool(spec.get("standardize_inputs", True))
    std_out = bool(spec.get("standardize_outputs", True))

    _idx_train, _idx_val, idx_test = _raw_splits(
        df,
        input_cols,
        output_cols,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        random_state=seed,
        shuffle=shuffle,
    )
    test_set: Set[Any] = set(idx_test.tolist())
    if not test_set:
        print("Empty test split.", file=sys.stderr)
        return 1

    in_sc, out_sc = _load_saved_scalers(run_dir, summary)
    if std_in and in_sc is None:
        print(
            "Missing input scaler joblib (workflow_summary scalers.input_scaler). "
            "Cannot reproduce training preprocessing.",
            file=sys.stderr,
        )
        return 1
    if std_out and out_sc is None:
        print(
            "Missing output scaler joblib (workflow_summary scalers.output_scaler).",
            file=sys.stderr,
        )
        return 1

    gcols = _group_cols(df)
    if not gcols:
        print("No group columns (metadata inputs).", file=sys.stderr)
        return 1

    df["_gid"] = df.groupby(gcols, sort=False).ngroup()
    test_gids = sorted(df.loc[df.index.isin(test_set), "_gid"].unique())
    if args.max_cases is not None:
        test_gids = test_gids[: max(0, args.max_cases)]

    psi = np.linspace(0.0001, 1.0, len(output_cols))

    rows_out: List[Dict[str, Any]] = []
    recon_true_list: List[np.ndarray] = []
    recon_pred_list: List[np.ndarray] = []

    for gid in test_gids:
        block = df[df["_gid"] == gid]
        tmpl = block.iloc[0]
        ms_series = block["m"]
        if args.m_source == "case":
            ms_use = sorted(ms_series.unique().tolist())
        else:
            want = set(range(int(args.m_min), int(args.m_max) + 1))
            avail = set(ms_series.unique().tolist())
            ms_use = sorted(want.intersection(avail))
        if len(ms_use) < 1:
            continue

        sub = block[block["m"].isin(ms_use)].sort_values("m")
        ms_act = sub["m"].tolist()
        if len(ms_act) != len(set(ms_act)):
            continue

        X_list = []
        for m in ms_act:
            row = tmpl.copy()
            row["m"] = m
            X_list.append(row[input_cols].values.astype(np.float64))
        X_raw = np.vstack(X_list)
        X_scaled = in_sc.transform(X_raw) if in_sc is not None else X_raw

        y_pred = adapter.predict(X_scaled)
        if hasattr(y_pred, "numpy"):
            y_pred = y_pred.numpy()
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        if out_sc is not None:
            y_pred = out_sc.inverse_transform(y_pred)

        Y_true = sub[list(output_cols)].values.astype(np.float64)
        recon_t = _combine_modes(Y_true, args.combine)
        recon_p = _combine_modes(y_pred, args.combine)
        if recon_t.size == 0 or recon_p.size == 0:
            continue
        r2 = float(r2_score(recon_t, recon_p))
        rmse = float(np.sqrt(mean_squared_error(recon_t, recon_p)))
        rows_out.append(
            {
                "group_id": int(gid),
                "n_modes": len(ms_act),
                "n": float(tmpl["n"]) if "n" in tmpl else np.nan,
                "r2_recon": r2,
                "rmse_recon": rmse,
                "m_min": min(ms_act),
                "m_max": max(ms_act),
            }
        )
        recon_true_list.append(recon_t)
        recon_pred_list.append(recon_p)

    if not rows_out:
        print("No cases with overlapping m in range; try --m-source case or widen m range.", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (run_dir / "plots" / "per_case_recon")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_case_recon_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    r2s = np.array([r["r2_recon"] for r in rows_out], dtype=float)
    flat_true = np.concatenate(recon_true_list)
    flat_pred = np.concatenate(recon_pred_list)
    r2_pooled = float(r2_score(flat_true, flat_pred))
    rmse_pooled = float(np.sqrt(mean_squared_error(flat_true, flat_pred)))

    print(f"Cases evaluated: {len(rows_out)}")
    print(
        f"Pooled R^2 (reconstructed field, all test cases x all psi): {r2_pooled:.6f}  "
        f"RMSE={rmse_pooled:.6f}  (n_pts={len(flat_true)})"
    )
    print(
        f"Per-case R^2 (recon): mean={r2s.mean():.4f} median={np.median(r2s):.4f} "
        f"min={r2s.min():.4f} max={r2s.max():.4f}"
    )
    print(f"Wrote {csv_path}")

    summary_json = {
        "model": args.model,
        "n_cases": len(rows_out),
        "combine": args.combine,
        "m_source": args.m_source,
        "m_min": args.m_min,
        "m_max": args.m_max,
        "r2_reconstructed_field_pooled": r2_pooled,
        "rmse_reconstructed_field_pooled": rmse_pooled,
        "n_psi_points_pooled": int(len(flat_true)),
        "r2_per_case_mean": float(r2s.mean()),
        "r2_per_case_median": float(np.median(r2s)),
        "r2_per_case_min": float(r2s.min()),
        "r2_per_case_max": float(r2s.max()),
        "metrics_csv": str(csv_path),
    }
    (out_dir / "per_case_recon_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'per_case_recon_summary.json'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figures", file=sys.stderr)
        return 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r2s, bins=min(30, max(5, len(r2s) // 3)), color="steelblue", edgecolor="white")
    ax.set_xlabel(r"per-case $R^2$ (reconstructed $\delta p$ vs truth)")
    ax.set_ylabel("count")
    ax.set_title(f"{args.model} | combine={args.combine} | m_source={args.m_source}")
    fig.tight_layout()
    hist_path = out_dir / "per_case_r2_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {hist_path}")

    # Best / median / worst psi curves
    order = np.argsort(r2s)
    picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax2, (label, j) in zip(axes, picks):
        rt = recon_true_list[j]
        rp = recon_pred_list[j]
        ax2.plot(psi, rt, "k-", lw=2, label="truth recon")
        ax2.plot(psi, rp, "b--", lw=1.5, label="pred recon")
        ax2.set_xlabel(r"$\psi_N$")
        ax2.set_title(f"{label}: R^2={r2s[j]:.3f}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
    axes[0].set_ylabel(r"reconstructed $|\delta p|$")
    fig2.suptitle(f"{args.model} | {args.combine} over m")
    fig2.tight_layout()
    curve_path = out_dir / "per_case_recon_examples_best_med_worst.png"
    fig2.savefig(curve_path, dpi=150)
    plt.close(fig2)
    print(f"Wrote {curve_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
