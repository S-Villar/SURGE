#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per–test-case |delta p| maps in (m, psi_N): metrics and side-by-side plots.

For each test **group** (all inputs except poloidal mode *m*), stacks rows sorted by *m*
into matrices **Y_true**, **Y_pred** with shape *(n_modes, n_psi)*. Reports **R^2** and
**RMSE** on the flattened map (one score per scenario). Optionally saves heatmaps:
ground truth next to prediction (shared color scale), optional absolute-error panel.

Reuses the same train/val/test split and scalers as
``eval_per_case_delta_p_recon.py``.

Usage:
  conda activate surge
  python scripts/m3dc1/eval_test_delta_p_2d_maps.py runs/m3dc1_delta_p_per_mode \\
      --model mlp_per_mode --plot-mode quantiles

  python scripts/m3dc1/eval_test_delta_p_2d_maps.py runs/m3dc1_delta_p_per_mode_cfs \\
      --model mlp_per_mode --plot-mode sample --plot-sample 8 --m-source case

Requires: numpy, pandas, scikit-learn, matplotlib; joblib; torch for torch models;
pyarrow for Parquet. Python 3.6+.
"""
import argparse
import csv
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as _e:
    _env = "/global/cfs/projectdirs/m3716/software/asvillar/envs/surge/bin/python"
    sys.stderr.write(
        "Missing numpy/pandas. Try: {_env} {argv}\n".format(
            _env=_env, argv=" ".join(sys.argv)
        )
    )
    raise SystemExit(1) from _e

from sklearn.metrics import mean_squared_error, r2_score


def _r2_map_flat(y_true, y_pred):
    # type: (np.ndarray, np.ndarray) -> float
    """R^2 on flattened map; NaN when truth variation is tiny vs prediction scale (R^2 misleading)."""
    ft = np.asarray(y_true, dtype=np.float64).ravel()
    fp = np.asarray(y_pred, dtype=np.float64).ravel()
    pred_scale = max(1.0, float(np.mean(np.abs(fp))), float(np.std(fp)))
    if float(np.std(ft)) < 1e-3 * pred_scale:
        return float("nan")
    return float(r2_score(ft, fp))


def _json_num(x):
    if x is None:
        return None
    if isinstance(x, float) and x != x:
        return None
    return x


def _fill_nan(x, fallback):
    xv = float(x)
    if xv != xv:
        return float(fallback)
    return xv

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from surge.io.load_compat import load_model_compat  # noqa: E402
import joblib  # noqa: E402


def _sorted_output_p_columns(df):
    cols = [c for c in df.columns if str(c).startswith("output_p_")]

    def _key(name):
        try:
            return int(str(name).split("_")[-1])
        except ValueError:
            return name

    return sorted(cols, key=_key)


_RECON_PATH = _SCRIPTS / "eval_per_case_delta_p_recon.py"
_spec = importlib.util.spec_from_file_location("_delta_p_recon", str(_RECON_PATH))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Cannot load eval_per_case_delta_p_recon")
_recon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recon)

_load_spec = _recon._load_spec
_load_yaml = _recon._load_yaml
_resolve_repo_path = _recon._resolve_repo_path
_raw_splits = _recon._raw_splits
_group_cols = _recon._group_cols
_load_saved_scalers = _recon._load_saved_scalers


def _model_params_from_spec(spec, model_name):
    """Return params dict for a model name in spec['models']."""
    for m in spec.get("models") or []:
        if m.get("name") == model_name:
            return dict(m.get("params") or {})
    return {}


def _hidden_layers_from_state_dict(state_dict):
    """Infer hidden layer widths from PyTorchMLP state_dict keys."""
    wkeys = []
    for k in state_dict.keys():
        m = re.match(r"^network\.(\d+)\.weight$", str(k))
        if m:
            wkeys.append((int(m.group(1)), k))
    if not wkeys:
        return []
    wkeys = sorted(wkeys, key=lambda x: x[0])
    out_dims = [int(state_dict[k].shape[0]) for _, k in wkeys]
    if len(out_dims) <= 1:
        return []
    return out_dims[:-1]


def _load_epoch_checkpoint_adapter(checkpoint_path, model_params):
    """
    Build a predictor from an epoch checkpoint (.pt).

    The returned adapter expects RAW inputs and returns de-standardized outputs.
    """
    import torch
    import torch.nn as nn
    from surge.model.pytorch_impl import PyTorchMLP

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Unsupported checkpoint format (missing model_state_dict).")
    state_dict = ckpt["model_state_dict"]
    input_size = int(ckpt.get("input_size"))
    output_size = int(ckpt.get("output_size"))
    hidden_layers = model_params.get("hidden_layers")
    if not hidden_layers:
        hidden_layers = _hidden_layers_from_state_dict(state_dict)
    if not hidden_layers:
        raise ValueError("Could not infer hidden_layers from checkpoint/state_dict.")
    hidden_layers = [int(x) for x in hidden_layers]

    dropout = float(model_params.get("dropout", 0.0))
    act_name = str(model_params.get("activation_fn", "relu")).lower()
    if act_name == "tanh":
        act = nn.Tanh
    elif act_name in ("leaky_relu", "lrelu"):
        act = nn.LeakyReLU
    else:
        act = nn.ReLU

    model = PyTorchMLP(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        dropout_rate=dropout,
        activation_fn=act,
        learning_rate=float(model_params.get("learning_rate", 1e-3)),
        n_epochs=1,
        batch_size=int(model_params.get("batch_size", 64)),
    )
    model.load_state_dict(state_dict, strict=True)
    if "scaler_X" in ckpt:
        model.scaler_X = ckpt["scaler_X"]
    if "scaler_y" in ckpt:
        model.scaler_y = ckpt["scaler_y"]
    model.is_fitted = True
    model.eval()
    return model


def _fallback_load_run_scalers(run_dir):
    """Load run scalers directly from run_dir/scalers when summary is missing."""
    sdir = run_dir / "scalers"
    in_p = sdir / "inputs.joblib"
    out_p = sdir / "outputs.joblib"
    in_sc = joblib.load(in_p) if in_p.is_file() else None
    out_sc = joblib.load(out_p) if out_p.is_file() else None
    return in_sc, out_sc


def _plot_pair(
    out_path,
    psi,
    ms,
    y_true,
    y_pred,
    title_prefix,
    r2_map,
    rmse_map,
    with_error,
    log_scale=False,
    label_fontsize=12,
    title_fontsize=12,
    tick_fontsize=10,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    vmin = float(min(y_true.min(), y_pred.min()))
    vmax = float(max(y_true.max(), y_pred.max()))
    y_true_plot = np.asarray(y_true, dtype=np.float64)
    y_pred_plot = np.asarray(y_pred, dtype=np.float64)
    if log_scale:
        pos_vals = np.concatenate(
            [np.ravel(y_true[y_true > 0]), np.ravel(y_pred[y_pred > 0])]
        )
        if pos_vals.size == 0:
            norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))
        else:
            lvmin = float(np.min(pos_vals))
            lvmax = float(np.max(pos_vals))
            if lvmax <= lvmin:
                lvmax = lvmin * 10.0
            # Log scales cannot render non-positive values; clip only for plotting.
            y_true_plot = np.clip(y_true_plot, lvmin, None)
            y_pred_plot = np.clip(y_pred_plot, lvmin, None)
            norm = LogNorm(vmin=lvmin, vmax=lvmax)
    else:
        if vmin >= vmax:
            vmax = vmin + 1e-12
        norm = Normalize(vmin=vmin, vmax=vmax)
    extent = [float(psi[0]), float(psi[-1]), float(ms[0]), float(ms[-1])]

    ncols = 3 if with_error else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), squeeze=False)
    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    im0 = ax0.imshow(
        y_true_plot,
        aspect="auto",
        origin="lower",
        norm=norm,
        extent=extent,
        interpolation="nearest",
    )
    ax0.set_title("truth")
    ax0.set_xlabel(r"$\psi_N$", fontsize=label_fontsize)
    ax0.set_ylabel(r"$m$", fontsize=label_fontsize)
    ax0.tick_params(labelsize=tick_fontsize)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(
        y_pred_plot,
        aspect="auto",
        origin="lower",
        norm=norm,
        extent=extent,
        interpolation="nearest",
    )
    ax1.set_title("prediction")
    ax1.set_xlabel(r"$\psi_N$", fontsize=label_fontsize)
    ax1.tick_params(labelsize=tick_fontsize)
    ax1.set_yticklabels([])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if with_error:
        ax2 = axes[0, 2]
        err = np.abs(y_true - y_pred)
        im2 = ax2.imshow(
            err,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="magma",
        )
        ax2.set_title("|error|")
        ax2.set_xlabel(r"$\psi_N$", fontsize=label_fontsize)
        ax2.tick_params(labelsize=tick_fontsize)
        ax2.set_yticklabels([])
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    r2_title = "n/a" if (r2_map != r2_map) else "{:.4f}".format(r2_map)
    fig.suptitle(
        "{} | modes={} | R^2={} RMSE={:.4g}".format(
            title_prefix, len(ms), r2_title, rmse_map
        ),
        fontsize=title_fontsize,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="SURGE run directory")
    ap.add_argument("--model", required=True, help="Model name (e.g. mlp_per_mode)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint (.pt or .joblib). If set, use it instead of workflow_summary artifact.",
    )
    ap.add_argument(
        "--m-source",
        choices=("range", "case"),
        default="case",
        help="case: all m in dataset for each test group (default)",
    )
    ap.add_argument("--m-min", type=int, default=-100)
    ap.add_argument("--m-max", type=int, default=100)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument(
        "--plot-mode",
        choices=("none", "quantiles", "sample", "all"),
        default="quantiles",
        help="Heatmaps: none | quantiles (best/median/worst R^2) | random sample | all",
    )
    ap.add_argument(
        "--plot-sample",
        type=int,
        default=8,
        help="Number of cases when --plot-mode sample (default 8)",
    )
    ap.add_argument(
        "--plot-seed",
        type=int,
        default=42,
        help="RNG seed for --plot-mode sample",
    )
    ap.add_argument(
        "--with-error-panel",
        action="store_true",
        help="Third panel: absolute error (own color scale)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: <run_dir>/plots/delta_p_2d_test/<model>)",
    )
    ap.add_argument(
        "--plot-log-scale",
        action="store_true",
        help="Use logarithmic color scale for truth/prediction panels.",
    )
    ap.add_argument(
        "--label-fontsize",
        type=int,
        default=12,
        help="Axis label font size for saved figures (default 12).",
    )
    ap.add_argument(
        "--title-fontsize",
        type=int,
        default=12,
        help="Figure title font size for saved figures (default 12).",
    )
    ap.add_argument(
        "--tick-fontsize",
        type=int,
        default=10,
        help="Tick label font size for saved figures (default 10).",
    )
    ap.add_argument(
        "--nan-r2-value",
        type=float,
        default=0.0,
        help="Value to use when R^2 is undefined (NaN). Default 0.0.",
    )
    ap.add_argument(
        "--save-best-case-data",
        action="store_true",
        help="Save best-case arrays (truth/pred/psi/m) for post-processing.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N evaluated cases (default 20). Set 0 to disable.",
    )
    ap.add_argument(
        "--best-case-metric",
        choices=("rmse", "nrmse"),
        default="rmse",
        help="Metric used to pick best-case figure (default rmse).",
    )
    ap.add_argument(
        "--min-truth-max-for-best",
        type=float,
        default=0.0,
        help="Only consider cases with truth_max >= this for best-case plot (default 0).",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    spec = _load_spec(run_dir)
    summary_path = run_dir / "workflow_summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Model loading supports:
    #   1) final artifact from workflow_summary (default)
    #   2) explicit --checkpoint (including epoch_XXXX.pt)
    model_params = _model_params_from_spec(spec, args.model)
    backend_name = ""
    adapter = None
    checkpoint_path = (
        _resolve_repo_path(run_dir, args.checkpoint)
        if args.checkpoint is not None
        else None
    )
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            sys.stderr.write("Checkpoint file not found: {}\n".format(checkpoint_path))
            return 1
        if checkpoint_path.suffix.lower() == ".pt":
            try:
                adapter = _load_epoch_checkpoint_adapter(checkpoint_path, model_params)
                backend_name = "pytorch"
            except Exception as e:
                sys.stderr.write(
                    "Failed to load torch checkpoint {}: {}\n".format(checkpoint_path, e)
                )
                return 1
        else:
            me_load = {"backend": "pytorch"}
            try:
                adapter = load_model_compat(checkpoint_path, me_load)
                backend_name = me_load["backend"]
            except Exception as e:
                sys.stderr.write("Failed to load checkpoint {}: {}\n".format(checkpoint_path, e))
                return 1
    else:
        if not summary:
            sys.stderr.write(
                "Missing {} and no --checkpoint provided.\n".format(summary_path)
            )
            return 1
        model_entry = None
        for m in summary.get("models") or []:
            if m.get("name") == args.model:
                model_entry = m
                break
        if model_entry is None:
            sys.stderr.write(
                "Model {!r} not in workflow_summary models list\n".format(args.model)
            )
            return 1
        art = (model_entry.get("artifacts") or {}).get("model", "")
        model_path = _resolve_repo_path(run_dir, art)
        if not model_path.is_file():
            sys.stderr.write("Model file not found: {}\n".format(model_path))
            return 1
        me_load = dict(model_entry)
        backend_name = str(me_load.get("backend", "")).lower()
        if backend_name == "torch":
            me_load["backend"] = "pytorch"
            backend_name = "pytorch"
        try:
            adapter = load_model_compat(model_path, me_load)
        except Exception as e:
            sys.stderr.write("Failed to load model: {}\n".format(e))
            return 1

    fp = spec.get("dataset_path")
    if not fp:
        dsinfo = summary.get("dataset") or {}
        fp = dsinfo.get("file_path", "")
    dataset_path = _resolve_repo_path(run_dir, fp)
    if not dataset_path.is_file():
        sys.stderr.write("Dataset not found: {}\n".format(dataset_path))
        return 1

    meta_path = spec.get("metadata_path") or (
        run_dir.parent.parent / "data/datasets/SPARC/delta_p_per_mode_metadata.yaml"
    )
    meta_path = _resolve_repo_path(run_dir, meta_path)
    meta = _load_yaml(meta_path) if meta_path.is_file() else {}
    input_cols = [c for c in (meta.get("inputs") or []) if c]
    output_cols = [c for c in (meta.get("outputs") or []) if c]

    if dataset_path.suffix.lower() in (".parquet", ".pq"):
        if input_cols and output_cols:
            want = [c for c in input_cols + output_cols if c]
            df = pd.read_parquet(dataset_path, columns=want)
        else:
            df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() in (".pkl", ".pickle"):
        df = pd.read_pickle(dataset_path)
        if input_cols and output_cols:
            df = df[[c for c in input_cols + output_cols if c in df.columns]]
    else:
        df = pd.read_csv(dataset_path)
        if input_cols and output_cols:
            df = df[[c for c in input_cols + output_cols if c in df.columns]]

    if not input_cols:
        input_cols = [c for c in _recon.INPUT_KEYS if c in df.columns]
    if not output_cols:
        output_cols = _sorted_output_p_columns(df)
    if not input_cols or not output_cols:
        sys.stderr.write(
            "Could not infer input/output columns (metadata and dataframe).\n"
        )
        return 1

    drop_extra = [c for c in df.columns if c not in input_cols + output_cols]
    if drop_extra:
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
    test_set = set(idx_test.tolist())
    if not test_set:
        sys.stderr.write("Empty test split.\n")
        return 1

    in_sc = out_sc = None
    if summary:
        in_sc, out_sc = _load_saved_scalers(run_dir, summary)
    else:
        in_sc, out_sc = _fallback_load_run_scalers(run_dir)
    if std_in and in_sc is None:
        sys.stderr.write("Missing input scaler.\n")
        return 1
    if std_out and out_sc is None:
        sys.stderr.write("Missing output scaler.\n")
        return 1

    gcols = _group_cols(df)
    if not gcols:
        sys.stderr.write("No group columns.\n")
        return 1

    # Build unique cases from the TEST rows only (group by all inputs except m).
    df_test = df.loc[df.index.isin(test_set)].copy()
    if df_test.empty:
        sys.stderr.write("No rows in test split.\n")
        return 1
    df_test["_gid"] = df_test.groupby(gcols, sort=False).ngroup()
    test_gids = sorted(df_test["_gid"].unique())
    if args.max_cases is not None:
        test_gids = test_gids[: max(0, args.max_cases)]
    sys.stdout.write(
        "Evaluating {} test cases with m in [{}, {}]...\n".format(
            len(test_gids), int(args.m_min), int(args.m_max)
        )
    )
    sys.stdout.flush()

    psi = np.linspace(0.0001, 1.0, len(output_cols))

    rows_out = []
    y_true_list = []
    y_pred_list = []
    ms_rows_list = []

    pred_rows_long = []
    for idx_case, gid in enumerate(test_gids, start=1):
        block_test = df_test[df_test["_gid"] == gid]
        if block_test.empty:
            continue
        tmpl = block_test.iloc[0]

        # Build truth map from full dataframe for the same case keys (excluding m).
        mask = np.ones(len(df), dtype=bool)
        for c in gcols:
            mask &= df[c].values == tmpl[c]
        block_all = df.loc[mask]
        if args.m_source == "case":
            ms_unique = sorted(block_all["m"].unique().tolist())
        else:
            ms_unique = list(range(int(args.m_min), int(args.m_max) + 1))
        if len(ms_unique) < 1:
            continue

        y_true_rows = []
        valid_ms = []
        for m in ms_unique:
            sl = block_all.loc[block_all["m"] == m, output_cols]
            if sl.empty:
                continue
            # Duplicate (case, m) rows can exist; pick first row deterministically.
            y_true_rows.append(np.asarray(sl.iloc[0].values, dtype=np.float64))
            valid_ms.append(m)
        if not y_true_rows:
            continue
        y_true = np.vstack(y_true_rows)

        x_list = []
        for m in valid_ms:
            row = tmpl.copy()
            row["m"] = m
            x_list.append(row[input_cols].values.astype(np.float64))
        x_raw = np.vstack(x_list)
        x_scaled = in_sc.transform(x_raw) if in_sc is not None else x_raw
        y_pred = adapter.predict(x_scaled)
        if hasattr(y_pred, "numpy"):
            y_pred = y_pred.numpy()
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        if out_sc is not None:
            y_pred = out_sc.inverse_transform(y_pred)
        if y_pred.shape[1] != len(output_cols):
            if y_pred.shape[1] > len(output_cols):
                y_pred = y_pred[:, : len(output_cols)]
            else:
                continue

        if y_true.shape != y_pred.shape:
            continue

        ms_act = valid_ms

        flat_t = y_true.ravel()
        flat_p = y_pred.ravel()
        r2_m = _fill_nan(_r2_map_flat(flat_t, flat_p), args.nan_r2_value)
        rmse_m = float(np.sqrt(mean_squared_error(flat_t, flat_p)))
        denom = float(np.max(np.abs(flat_t)) + 1e-12)
        nrmse_m = float(rmse_m / denom)

        n_val = float(tmpl["n"]) if "n" in tmpl else float("nan")
        rows_out.append(
            {
                "group_id": int(gid),
                "n": n_val,
                "n_modes": len(ms_unique),
                "m_min": min(ms_unique),
                "m_max": max(ms_unique),
                "truth_min": float(np.min(y_true)),
                "truth_max": float(np.max(y_true)),
                "pred_min": float(np.min(y_pred)),
                "pred_max": float(np.max(y_pred)),
                "r2_map": r2_m,
                "rmse_map": rmse_m,
                "nrmse_map": nrmse_m,
                "figure_path": "",
            }
        )
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        ms_rows_list.append(np.asarray(ms_act, dtype=np.float64))

        # Persist prediction table rows: consecutive rows by m for each case.
        for i, mv in enumerate(ms_act):
            row_out = {
                "group_id": int(gid),
                "m": int(mv),
            }
            for j, c in enumerate(output_cols):
                row_out[c] = float(y_pred[i, j])
            pred_rows_long.append(row_out)
        if args.progress_every > 0 and (idx_case % int(args.progress_every) == 0):
            sys.stdout.write(
                "  processed {}/{} cases...\n".format(idx_case, len(test_gids))
            )
            sys.stdout.flush()

    if not rows_out:
        sys.stderr.write("No test groups evaluated.\n")
        return 1

    out_root = args.out_dir or (run_dir / "plots" / "delta_p_2d_test" / args.model)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "per_case_2d_metrics.csv"
    r2_arr = np.array([r["r2_map"] for r in rows_out], dtype=float)

    plot_indices = set()
    if args.plot_mode == "quantiles":
        fin = np.isfinite(r2_arr)
        if fin.any():
            idx_fin = np.where(fin)[0]
            vals = r2_arr[fin]
            plot_indices.add(int(idx_fin[np.argmin(vals)]))
            plot_indices.add(int(idx_fin[np.argmax(vals)]))
            plot_indices.add(int(idx_fin[np.argsort(vals)[len(vals) // 2]]))
        else:
            rmse_arr = np.array([r["rmse_map"] for r in rows_out], dtype=float)
            order = np.argsort(rmse_arr)
            plot_indices.add(int(order[0]))
            plot_indices.add(int(order[len(order) // 2]))
            plot_indices.add(int(order[-1]))
    elif args.plot_mode == "sample":
        rng = np.random.RandomState(args.plot_seed)
        idx_all = np.arange(len(rows_out))
        k = min(args.plot_sample, len(rows_out))
        pick = rng.choice(idx_all, size=k, replace=False)
        plot_indices.update(int(x) for x in pick)
    elif args.plot_mode == "all":
        plot_indices.update(range(len(rows_out)))

    for j in sorted(plot_indices):
        ms_act = ms_rows_list[j]
        r2v = rows_out[j]["r2_map"]
        r2s = "nan" if (r2v != r2v) else "{:.4f}".format(r2v)
        fn = "delta_p_2d_gid{}_n{}_r2{}.png".format(
            int(rows_out[j]["group_id"]),
            int(rows_out[j]["n"]) if not np.isnan(rows_out[j]["n"]) else 0,
            r2s,
        )
        fpng = out_root / fn
        try:
            _plot_pair(
                fpng,
                psi,
                ms_act,
                y_true_list[j],
                y_pred_list[j],
                args.model,
                rows_out[j]["r2_map"],
                rows_out[j]["rmse_map"],
                args.with_error_panel,
                log_scale=args.plot_log_scale,
                label_fontsize=args.label_fontsize,
                title_fontsize=args.title_fontsize,
                tick_fontsize=args.tick_fontsize,
            )
            rows_out[j]["figure_path"] = str(fpng)
        except Exception as ex:
            sys.stderr.write("Plot failed for group {}: {}\n".format(rows_out[j]["group_id"], ex))

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    pred_csv = out_root / "predictions_2d_rows.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["group_id", "m"] + list(output_cols)
        )
        w.writeheader()
        w.writerows(pred_rows_long)

    flat_true = np.concatenate([x.ravel() for x in y_true_list])
    flat_pred = np.concatenate([x.ravel() for x in y_pred_list])
    r2_pool = _fill_nan(_r2_map_flat(flat_true, flat_pred), args.nan_r2_value)
    rmse_pool = float(np.sqrt(mean_squared_error(flat_true, flat_pred)))
    nrmse_pool = float(
        rmse_pool / (float(np.max(np.abs(flat_true)) + 1e-12))
    )

    r2_fin = r2_arr[np.isfinite(r2_arr)]
    has_r2 = r2_fin.size > 0
    summary_json = {
        "model": args.model,
        "n_cases": len(rows_out),
        "backend": backend_name,
        "m_source": args.m_source,
        "m_eval_min": int(args.m_min),
        "m_eval_max": int(args.m_max),
        "plot_mode": args.plot_mode,
        "r2_maps_pooled": _json_num(r2_pool),
        "rmse_maps_pooled": rmse_pool,
        "nrmse_maps_pooled": nrmse_pool,
        "n_cells_pooled": int(flat_true.size),
        "r2_per_case_mean": _json_num(float(np.mean(r2_fin)) if has_r2 else float("nan")),
        "r2_per_case_median": _json_num(float(np.median(r2_fin)) if has_r2 else float("nan")),
        "r2_per_case_min": _json_num(float(np.min(r2_fin)) if has_r2 else float("nan")),
        "r2_per_case_max": _json_num(float(np.max(r2_fin)) if has_r2 else float("nan")),
        "n_per_case_r2_defined": int(np.isfinite(r2_arr).sum()),
        "metrics_csv": str(csv_path),
        "predictions_rows_csv": str(pred_csv),
        "out_dir": str(out_root),
    }

    # Best case with optional signal floor to avoid near-zero maps that look white.
    metric_key = "nrmse_map" if args.best_case_metric == "nrmse" else "rmse_map"
    cand = [
        i
        for i, r in enumerate(rows_out)
        if float(r.get("truth_max", 0.0)) >= float(args.min_truth_max_for_best)
    ]
    if not cand:
        cand = list(range(len(rows_out)))
    best_idx = int(min(cand, key=lambda i: float(rows_out[i][metric_key])))
    summary_json["best_case"] = {
        "group_id": int(rows_out[best_idx]["group_id"]),
        "metric_name": metric_key,
        "metric_value": float(rows_out[best_idx][metric_key]),
        "rmse_map": float(rows_out[best_idx]["rmse_map"]),
        "r2_map": _json_num(float(rows_out[best_idx]["r2_map"])),
        "n_modes": int(rows_out[best_idx]["n_modes"]),
        "truth_max": float(rows_out[best_idx]["truth_max"]),
        "truth_min": float(rows_out[best_idx]["truth_min"]),
    }
    (out_root / "summary_2d_maps.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    sys.stdout.write("Cases: {}\n".format(len(rows_out)))
    pool_r2s = "n/a" if (r2_pool != r2_pool) else "{:.6f}".format(r2_pool)
    sys.stdout.write(
        "Pooled R^2 (all maps, flattened): {}  RMSE={:.6g}  NRMSE={:.6g}  cells={}\n".format(
            pool_r2s, rmse_pool, nrmse_pool, flat_true.size
        )
    )
    def _fmt_stat(x):
        if x is None or (isinstance(x, float) and x != x):
            return "n/a"
        return "{:.4f}".format(x)

    sys.stdout.write(
        "Per-map R^2 (finite only): mean={} median={} min={} max={}  (defined {}/{} cases)\n".format(
            _fmt_stat(_json_num(float(np.mean(r2_fin)) if has_r2 else float("nan"))),
            _fmt_stat(_json_num(float(np.median(r2_fin)) if has_r2 else float("nan"))),
            _fmt_stat(_json_num(float(np.min(r2_fin)) if has_r2 else float("nan"))),
            _fmt_stat(_json_num(float(np.max(r2_fin)) if has_r2 else float("nan"))),
            int(np.isfinite(r2_arr).sum()),
            len(rows_out),
        )
    )
    # Always write best-case 2D truth-vs-pred figure.
    try:
        b = best_idx
        fpng = out_root / "best_case_truth_vs_pred_2d.png"
        _plot_pair(
            fpng,
            psi,
            ms_rows_list[b],
            y_true_list[b],
            y_pred_list[b],
            args.model,
            rows_out[b]["r2_map"],
            rows_out[b]["rmse_map"],
            args.with_error_panel,
            log_scale=args.plot_log_scale,
            label_fontsize=args.label_fontsize,
            title_fontsize=args.title_fontsize,
            tick_fontsize=args.tick_fontsize,
        )
        rows_out[b]["figure_path"] = str(fpng)
        sys.stdout.write("Wrote best-case figure: {}\n".format(fpng))
        if args.save_best_case_data:
            best_payload = {
                "group_id": int(rows_out[b]["group_id"]),
                "model": args.model,
                "m_values": np.asarray(ms_rows_list[b], dtype=np.float64),
                "psi_values": np.asarray(psi, dtype=np.float64),
                "y_true_map": np.asarray(y_true_list[b], dtype=np.float64),
                "y_pred_map": np.asarray(y_pred_list[b], dtype=np.float64),
                "truth_min": float(rows_out[b]["truth_min"]),
                "truth_max": float(rows_out[b]["truth_max"]),
                "pred_min": float(rows_out[b]["pred_min"]),
                "pred_max": float(rows_out[b]["pred_max"]),
                "rmse_map": float(rows_out[b]["rmse_map"]),
                "r2_map": float(rows_out[b]["r2_map"]),
                "nrmse_map": float(rows_out[b]["nrmse_map"]),
            }
            pkl_path = out_root / "best_case_data.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(best_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(out_root / "best_case_y_true.npy", best_payload["y_true_map"])
            np.save(out_root / "best_case_y_pred.npy", best_payload["y_pred_map"])
            np.save(out_root / "best_case_m.npy", best_payload["m_values"])
            np.save(out_root / "best_case_psi.npy", best_payload["psi_values"])
            sys.stdout.write("Wrote best-case arrays: {}\n".format(pkl_path))
    except Exception as ex:
        sys.stderr.write("Best-case plot failed: {}\n".format(ex))
    sys.stdout.write(
        "Wrote {}, {}, and summary_2d_maps.json under {}\n".format(
            csv_path.name, pred_csv.name, out_root
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
