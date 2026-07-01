#!/usr/bin/env python
"""Inspect the progress of a SURGE training run and plot its loss curves.

Usage
-----
    python -m surge.check_training --run <run_folder>
    python -m surge.check_training --run runs/spectrum_image_full_maxnorm_log10
    python -m surge.check_training --run runs/m3dc1_gamma_ver_multimodel --last 15

What it does
------------
* Discovers every per-model training history in ``--run`` regardless of which
  producer wrote it:
    - spectrum-image trainer  ->  ``history_<model>.jsonl``
    - SURGE workflow (final)  ->  ``training_history_<tag>.json``
    - SURGE workflow (live)   ->  ``training_progress_<tag>.jsonl``
* Prints a stats report per model: epochs logged, best val loss / R2 and the
  epoch they occurred at, the latest epoch, and whether the run is still
  improving, plateaued or worsening (trend over the last ``--last`` epochs),
  plus early-stop status and final test metrics if available.
* Plots the train/val loss curves (+ val R2 when present) for the models being
  trained into ``<run>/check_training_loss.png``.

It is read-only and safe to run against a live job from a login node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    """Load epoch records from a .jsonl (one obj/line) or .json (list) file."""
    try:
        text = path.read_text()
    except Exception:
        return []
    rows: List[Any] = []
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return []
        rows = obj if isinstance(obj, list) else obj.get("history", []) if isinstance(obj, dict) else []
    return [r for r in rows if isinstance(r, dict) and "epoch" in r]


def discover_histories(run: Path) -> Dict[str, Dict[str, Any]]:
    """Return {model_tag: {"rows": [...], "source": str, "path": Path}}.

    When several files describe the same tag, keep the one with the most epochs
    (typically the completed history over a partial live log).
    """
    found: Dict[str, Dict[str, Any]] = {}

    def _consider(tag: str, path: Path, source: str) -> None:
        rows = _load_rows(path)
        if not rows:
            return
        prev = found.get(tag)
        if prev is None or len(rows) > len(prev["rows"]):
            found[tag] = {"rows": rows, "source": source, "path": path}

    for p in sorted(run.glob("history_*.jsonl")):
        _consider(p.stem[len("history_"):], p, "spectrum-image")
    for p in sorted(run.glob("training_history_*.json")):
        _consider(p.stem[len("training_history_"):], p, "workflow")
    for p in sorted(run.glob("training_progress_*.jsonl")):
        tag = p.stem[len("training_progress_"):]
        if tag.startswith("hpo"):
            continue  # aggregate HPO logs, not a single model
        _consider(tag, p, "workflow-live")
    return found


def _best(rows: List[Dict[str, Any]], key: str, mode: str) -> Optional[Tuple[float, int]]:
    vals = [(r[key], r["epoch"]) for r in rows
            if r.get(key) is not None and isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return (min if mode == "min" else max)(vals, key=lambda t: t[0])


def _trend(rows: List[Dict[str, Any]], key: str, last: int) -> str:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if len(vals) < 4:
        return "n/a"
    k = max(2, min(last, len(vals) // 2))
    recent = sum(vals[-k:]) / k
    prev = sum(vals[-2 * k:-k]) / k
    if prev == 0:
        return "n/a"
    rel = (recent - prev) / abs(prev)
    # For a loss (lower is better): negative rel => improving.
    if rel < -0.01:
        return f"improving ({rel*100:+.1f}%)"
    if rel > 0.01:
        return f"worsening ({rel*100:+.1f}%)"
    return f"plateau ({rel*100:+.1f}%)"


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def _final_metrics(run: Path) -> Dict[str, Dict[str, Any]]:
    """Pull any available final test metrics keyed by model tag."""
    out: Dict[str, Dict[str, Any]] = {}
    si = run / "spectrum_image_metrics.json"
    if si.exists():
        try:
            res = json.loads(si.read_text()).get("results", {})
            for k, v in res.items():
                out[k] = {"test_r2": v.get("test_r2_global"),
                          "pattern_r2": v.get("test_pattern_r2")}
        except Exception:
            pass
    ws = run / "workflow_summary.json"
    if ws.exists():
        try:
            data = json.loads(ws.read_text())
            models = data.get("models", data if isinstance(data, list) else [])
            if isinstance(models, dict):
                models = list(models.values())
            for m in models if isinstance(models, list) else []:
                name = m.get("name") or m.get("model") or m.get("tag")
                met = m.get("metrics", {}) or {}
                test = met.get("test", met) if isinstance(met, dict) else {}
                if name:
                    out[str(name)] = {"test_r2": (test or {}).get("r2"),
                                      "rmse": (test or {}).get("rmse")}
        except Exception:
            pass
    return out


def report(run: Path, last: int) -> Dict[str, Dict[str, Any]]:
    hists = discover_histories(run)
    cfg_path = run / "run_config.json"
    print("=" * 78)
    print(f"SURGE training run: {run}")
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            keys = ("target", "n_cases", "grid", "models", "epochs",
                    "target_norm", "target_space", "batch_size")
            desc = "  ".join(f"{k}={cfg[k]}" for k in keys if k in cfg)
            print(f"config: {desc}")
        except Exception:
            pass
    if not hists:
        print("No per-model training histories found "
              "(looked for history_*.jsonl, training_history_*.json, "
              "training_progress_*.jsonl).")
        print("=" * 78)
        return {}

    finals = _final_metrics(run)
    print("=" * 78)
    summary: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(hists):
        rows = hists[tag]["rows"]
        src = hists[tag]["source"]
        last_row = rows[-1]
        n_ep = last_row["epoch"]
        bl = _best(rows, "val_loss", "min")
        br2 = _best(rows, "val_r2", "max")
        stopped = any(r.get("early_stop") for r in rows)
        trend = _trend(rows, "val_loss", last)
        fin = finals.get(tag, {})
        summary[tag] = {
            "epochs": n_ep, "source": src,
            "best_val_loss": bl[0] if bl else None,
            "best_val_loss_epoch": bl[1] if bl else None,
            "best_val_r2": br2[0] if br2 else None,
            "best_val_r2_epoch": br2[1] if br2 else None,
            "latest_train_loss": last_row.get("train_loss"),
            "latest_val_loss": last_row.get("val_loss"),
            "early_stopped": stopped, "trend_val_loss": trend,
            "final_test": fin,
        }
        print(f"[{tag}]  ({src})")
        print(f"   epochs logged     : {n_ep}"
              + ("   (early-stopped)" if stopped else ""))
        print(f"   latest train/val  : {_fmt(last_row.get('train_loss'))} / "
              f"{_fmt(last_row.get('val_loss'))}")
        if bl:
            print(f"   best val_loss     : {_fmt(bl[0])}  @ epoch {bl[1]}")
        if br2:
            print(f"   best val_R2       : {_fmt(br2[0])}  @ epoch {br2[1]}")
        print(f"   trend (val_loss)  : {trend}  (window={last})")
        if fin:
            ft = "  ".join(f"{k}={_fmt(v)}" for k, v in fin.items() if v is not None)
            if ft:
                print(f"   final test        : {ft}")
        print("-" * 78)
    return summary


def plot(run: Path, out: Optional[Path] = None) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"(plotting unavailable: {exc})")
        return None
    hists = discover_histories(run)
    if not hists:
        return None
    has_r2 = any(any(isinstance(r.get("val_r2"), (int, float)) for r in h["rows"])
                 for h in hists.values())
    ncol = 2 if has_r2 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.4), squeeze=False)
    ax_loss = axes[0, 0]
    for i, tag in enumerate(sorted(hists)):
        rows = hists[tag]["rows"]
        ep = [r["epoch"] for r in rows]
        tl = [r.get("train_loss", np.nan) for r in rows]
        vl = [r.get("val_loss", np.nan) for r in rows]
        c = f"C{i}"
        ax_loss.plot(ep, tl, color=c, ls="-", lw=1.3, label=f"{tag} train")
        ax_loss.plot(ep, vl, color=c, ls="--", lw=1.3, label=f"{tag} val")
        bl = _best(rows, "val_loss", "min")
        if bl:
            ax_loss.scatter([bl[1]], [bl[0]], color=c, marker="o", s=28, zorder=5)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("MSE loss")
    ax_loss.set_yscale("log"); ax_loss.set_title(f"{run.name}: loss")
    ax_loss.legend(fontsize=8); ax_loss.grid(alpha=0.3)
    if has_r2:
        ax_r2 = axes[0, 1]
        for i, tag in enumerate(sorted(hists)):
            rows = hists[tag]["rows"]
            ep = [r["epoch"] for r in rows if isinstance(r.get("val_r2"), (int, float))]
            r2 = [r["val_r2"] for r in rows if isinstance(r.get("val_r2"), (int, float))]
            if ep:
                ax_r2.plot(ep, r2, color=f"C{i}", lw=1.5, label=f"{tag} val R2")
        ax_r2.set_xlabel("epoch"); ax_r2.set_ylabel("val R2")
        ax_r2.set_title(f"{run.name}: val R2"); ax_r2.legend(fontsize=8)
        ax_r2.grid(alpha=0.3)
    fig.tight_layout()
    out = out or (run / "check_training_loss.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Path to a SURGE run folder.")
    ap.add_argument("--last", type=int, default=10,
                    help="Window (epochs) for the improving/plateau trend check.")
    ap.add_argument("--no-plot", action="store_true", help="Skip writing the loss PNG.")
    ap.add_argument("--out", default=None, help="Override the output PNG path.")
    ap.add_argument("--json", action="store_true",
                    help="Also print the machine-readable summary as JSON.")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.exists():
        raise SystemExit(f"run folder not found: {run}")

    summary = report(run, args.last)
    if not args.no_plot:
        png = plot(run, Path(args.out) if args.out else None)
        if png:
            print(f"loss curves -> {png}")
    if args.json:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
