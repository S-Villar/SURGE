#!/usr/bin/env python
"""Regenerate dataset inventory blocks in surge-datasets.canvas.tsx.

Scans data/datasets/ on disk (local PKL/CSV, benchmark NPZ caches, vision
downloads) and rewrites the ``const LOCAL`` and ``const CACHE`` arrays with
measured row counts, shapes, and cache status.

Usage::

    python scripts/refresh_datasets_canvas.py
    python scripts/refresh_datasets_canvas.py --warm-cache   # load missing loaders
    python scripts/refresh_datasets_canvas.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Filesystem inspection
# ---------------------------------------------------------------------------

def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def _datasets_root() -> pathlib.Path:
    return _repo_root() / "data" / "datasets"


def _qlknn_io_dict() -> dict:
    from surge.benchmarks.dataset_io import QLKNN_TRANSPORT_IO

    return QLKNN_TRANSPORT_IO


def _constellaration_mimo_io_dict() -> dict:
    from surge.benchmarks.dataset_io import CONSTELLARATION_MULTIOUTPUT_IO

    return CONSTELLARATION_MULTIOUTPUT_IO


def _fmt_n(n: int) -> str:
    return f"{n:,}"


def _shape_str(n_feat: int, n_tgt: int | str) -> str:
    if isinstance(n_tgt, int):
        return f"{n_feat} → {n_tgt}"
    return f"{n_feat} → {n_tgt}"


def _inspect_pkl_or_csv(path: pathlib.Path, *, target_cols: list[str] | None = None) -> dict:
    import pandas as pd

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_pickle(path)
    if target_cols is None:
        target_cols = [c for c in df.columns if "gamma" in c.lower()]
    feature_cols = [c for c in df.columns if c not in target_cols]
    return {
        "n": len(df),
        "n_feat": len(feature_cols),
        "n_tgt": len(target_cols) if target_cols else "?",
        "targets": ", ".join(target_cols) if target_cols else "—",
    }


def _inspect_npz(path: pathlib.Path) -> dict:
    import numpy as np

    d = np.load(path, allow_pickle=True)
    if "X" in d and "y" in d:
        x, y = d["X"], d["y"]
        n_feat = x.shape[1] if x.ndim > 1 else 1
        if y.ndim == 1:
            n_tgt = 1
            tgt_label = "1"
        else:
            n_tgt = y.shape[1]
            tgt_label = str(n_tgt)
        return {"n": x.shape[0], "shape": _shape_str(n_feat, tgt_label)}
    if "X" in d and "Y" in d:
        x, y = d["X"], d["Y"]
        return {
            "n": x.shape[0],
            "shape": _shape_str(x.shape[1], y.shape[1]),
        }
    return {"n": "?", "shape": "?"}


def _dir_has_files(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*"))


def _warm_missing_caches() -> list[str]:
    """Invoke leaderboard loaders for caches that are cheap to populate."""
    sys.path.insert(0, str(_repo_root()))
    from surge.benchmarks.leaderboard import (
        _load_burgers_1d,
        _load_cmod_density_limit,
        _load_lorenz63,
        _load_plasma_stability,
        _load_qlknn_transport,
    )

    loaders = [
        ("plasma.cmod_density_limit", _load_cmod_density_limit),
        ("classification.plasma_stability", _load_plasma_stability),
        ("pde.burgers_1d", _load_burgers_1d),
        ("sequence.lorenz63", _load_lorenz63),
        ("plasma.qlknn_transport", _load_qlknn_transport),
    ]
    warmed: list[str] = []
    for key, fn in loaders:
        cache_guess = {
            "plasma.cmod_density_limit": "benchmarks/plasma/cmod_density_limit.npz",
            "classification.plasma_stability": "benchmarks/classification/plasma_stability.npz",
            "pde.burgers_1d": "benchmarks/pde/burgers_1d.npz",
            "sequence.lorenz63": "benchmarks/sequence/lorenz63.npz",
            "plasma.qlknn_transport": "benchmarks/plasma/qlknn_transport.npz",
        }[key]
        if (_datasets_root() / cache_guess).exists():
            continue
        try:
            fn()
            warmed.append(key)
            print(f"  warmed {key}", file=sys.stderr)
        except Exception as exc:
            print(f"  skip {key}: {exc}", file=sys.stderr)
    return warmed


def _build_local_entries() -> list[dict]:
    root = _datasets_root()
    specs = [
        {
            "key": "nstxu.run10k_curated",
            "machine": "NSTX-U",
            "domain": "Tokamak stability",
            "file": "NSTX-U/nstxu_run10k_equil_curated.pkl",
            "format": "DataFrame (.pkl)",
            "targets_hint": ["gamma_VDE", "gamma_TOKAM"],
            "benchmark": None,
            "status": "available",
            "notes": "Primary NSTX-U growth-rate dataset. Ready for fusion.nstxu_stability benchmark.",
        },
        {
            "key": "nstxu.equil_curated",
            "machine": "NSTX-U",
            "domain": "Tokamak equilibrium",
            "file": "NSTX-U/nstxu_equil_curated.pkl",
            "format": "DataFrame (.pkl)",
            "targets_hint": ["gamma_VDE", "gamma_TOKAM", "gamma_TOKAM_zscore"],
            "benchmark": None,
            "status": "available",
            "notes": "Full 10k equilibrium set (includes VDE filter flags).",
        },
        {
            "key": "nstxu.run10k_csv",
            "machine": "NSTX-U",
            "domain": "Tokamak equilibrium",
            "file": "NSTX-U/RUN_10k/Equil_data.csv",
            "format": "CSV",
            "targets_hint": [],
            "benchmark": None,
            "status": "available",
            "notes": "Same 10k cases as .pkl but in CSV with physical units.",
        },
        {
            "key": "smart.shapes_gamma",
            "machine": "SMART",
            "domain": "Tokamak stability",
            "file": "SMART/smart_curated_shapes_gamma.pkl",
            "format": "DataFrame (.pkl)",
            "targets_hint": ["gamma", "gamma_TOKAM"],
            "benchmark": None,
            "status": "available",
            "notes": "Curated subset with shape features.",
        },
        {
            "key": "smart.10k_equil",
            "machine": "SMART",
            "domain": "Tokamak equilibrium",
            "file": "SMART/smart_curated_10k_equil_magnetics.pkl",
            "format": "DataFrame (.pkl)",
            "targets_hint": ["gamma_VDE", "gamma_TOKAM", "gamma_TOKAM_zscore"],
            "benchmark": None,
            "status": "available",
            "notes": "10k SMART equilibria with magnetics diagnostics.",
        },
        {
            "key": "hhfw.hpo_results",
            "machine": "NSTX (HHFW)",
            "domain": "RF heating HPO log",
            "file": "HHFW-NSTX/combined_hpo_results.pkl",
            "format": "DataFrame (.pkl)",
            "targets_hint": None,
            "benchmark": None,
            "status": "hpo-only",
            "notes": "Trial log only — not a training dataset.",
        },
    ]

    entries: list[dict] = []
    for spec in specs:
        path = root / spec["file"]
        entry = {**spec}
        if not path.exists():
            entry["n"] = "—"
            entry["shape"] = "—"
            entry["targets"] = "—"
            entry["status"] = "missing"
            entries.append(entry)
            continue
        if spec["status"] == "hpo-only":
            entry["n"] = 200
            entry["shape"] = "6 hyperparams → R²"
            entry["targets"] = "HPO trial score"
            entries.append(entry)
            continue
        try:
            info = _inspect_pkl_or_csv(path, target_cols=spec.get("targets_hint") or None)
            entry["n"] = info["n"]
            entry["shape"] = _shape_str(info["n_feat"], info["n_tgt"])
            entry["targets"] = info["targets"]
        except Exception as exc:
            entry["n"] = "?"
            entry["shape"] = "?"
            entry["targets"] = "?"
            entry["notes"] = f"{spec['notes']} (inspect failed: {exc})"
        entries.append(entry)
    return entries


def _build_cache_entries() -> list[dict]:
    root = _datasets_root()

    def npz_status(rel: str, *, generated: bool = False) -> tuple[str, dict | None]:
        path = root / rel
        if not path.exists():
            return "missing", None
        return ("generated" if generated else "cached-ok"), _inspect_npz(path)

    def dir_status(rel: str) -> str:
        return "cached-ok" if _dir_has_files(root / rel) else "missing"

    # Constellaration: unified path first, then legacy
    constell_paths = [
        "benchmarks/plasma/constellaration/paper_nfp3_clip0.05.npz",
        "constellaration/paper_nfp3_clip0.05.npz",
    ]
    constell_rel = next((p for p in constell_paths if (root / p).exists()), constell_paths[0])
    constell_status, constell_info = npz_status(constell_rel)

    qlknn_status, qlknn_info = npz_status("benchmarks/plasma/qlknn_transport.npz")
    cmod_status, cmod_info = npz_status("benchmarks/plasma/cmod_density_limit.npz")
    burgers_status, burgers_info = npz_status("benchmarks/pde/burgers_1d.npz", generated=True)
    lorenz_status, lorenz_info = npz_status("benchmarks/sequence/lorenz63.npz", generated=True)
    plasma_status, plasma_info = npz_status("benchmarks/classification/plasma_stability.npz")

    m3dc1_paths = [
        "benchmarks/fusion/m3dc1/m3dc1_sample.hdf5",
        "M3DC1/m3dc1_sample.hdf5",
    ]
    m3dc1_rel = next((p for p in m3dc1_paths if (root / p).exists()), m3dc1_paths[0])
    m3dc1_ok = (root / m3dc1_rel).exists()

    openml_ids: list[str] = []
    openml_dir = root / "benchmarks/tabular/sklearn_cache/openml/openml.org/api/v1/json/data"
    if openml_dir.exists():
        openml_ids = sorted(p.stem for p in openml_dir.glob("*.gz") if p.stem.isdigit())

    return [
        {
            "benchmark": "plasma.constellaration / plasma.constellaration_paper",
            "cachePath": constell_rel,
            "source": "HuggingFace proxima-fusion/constellaration",
            "n": _fmt_n(constell_info["n"]) if constell_info else "26,897",
            "shape": constell_info["shape"] if constell_info else "90 → 12",
            "firstRun": "pip install datasets → downloaded on first run",
            "status": constell_status,
        },
        {
            "benchmark": "plasma.constellaration_multioutput",
            "cachePath": constell_rel,
            "source": "Same NPZ as paper filter (joint 90→12 model)",
            "n": _fmt_n(constell_info["n"]) if constell_info else "26,897",
            "shape": "90 → 12 (joint)",
            "firstRun": "surge run -b constellaration_multi -m pytorch.residual_mlp",
            "status": constell_status,
            "io": _constellaration_mimo_io_dict(),
        },
        {
            "benchmark": "plasma.qlknn_transport",
            "cachePath": "benchmarks/plasma/qlknn_transport.npz",
            "source": "fusion_surrogates (DeepMind) + JAX inference",
            "n": _fmt_n(qlknn_info["n"]) if qlknn_info else "7,475",
            "shape": qlknn_info["shape"] if qlknn_info else "10 → 1",
            "firstRun": "pip install fusion_surrogates jax → generated on first run",
            "status": qlknn_status,
            "io": _qlknn_io_dict(),
        },
        {
            "benchmark": "plasma.cmod_density_limit",
            "cachePath": "benchmarks/plasma/cmod_density_limit.npz",
            "source": "MIT-PSFC open_density_limit_database (GitHub raw CSV)",
            "n": _fmt_n(cmod_info["n"]) if cmod_info else "~40,000 (balanced)",
            "shape": cmod_info["shape"] if cmod_info else "6 → 1",
            "firstRun": "Downloaded from GitHub on first run",
            "status": cmod_status,
        },
        {
            "benchmark": "fusion.m3dc1_sample",
            "cachePath": m3dc1_rel,
            "source": "PPPL M3DC1 team (must be supplied manually)",
            "n": "—",
            "shape": "13 → 1",
            "firstRun": "Place HDF5 file manually — no auto-download",
            "status": "cached-ok" if m3dc1_ok else "missing",
        },
        {
            "benchmark": "vision.mnist",
            "cachePath": "benchmarks/vision/MNIST/",
            "source": "torchvision auto-download",
            "n": "70,000",
            "shape": "784 → 10",
            "firstRun": "torchvision downloads on first run",
            "status": dir_status("benchmarks/vision/MNIST"),
        },
        {
            "benchmark": "vision.cifar10",
            "cachePath": "benchmarks/vision/cifar-10-batches-py/",
            "source": "torchvision auto-download",
            "n": "60,000",
            "shape": "3,072 → 10",
            "firstRun": "torchvision downloads on first run",
            "status": dir_status("benchmarks/vision/cifar-10-batches-py"),
        },
        {
            "benchmark": "tabular.* / ctr23.*",
            "cachePath": "benchmarks/tabular/sklearn_cache/",
            "source": "scikit-learn / OpenML",
            "n": f"{len(openml_ids)} OpenML IDs cached" if openml_ids else "varies",
            "shape": ", ".join(openml_ids) if openml_ids else "varies",
            "firstRun": "sklearn fetch_openml downloads on first run",
            "status": dir_status("benchmarks/tabular/sklearn_cache"),
        },
        {
            "benchmark": "pde.burgers_1d",
            "cachePath": "benchmarks/pde/burgers_1d.npz",
            "source": "Inline finite-difference solver (deterministic, seed=42)",
            "n": _fmt_n(burgers_info["n"]) if burgers_info else "1,024",
            "shape": burgers_info["shape"] if burgers_info else "64 → 64",
            "firstRun": "Generated and cached on first run",
            "status": burgers_status,
        },
        {
            "benchmark": "sequence.lorenz63",
            "cachePath": "benchmarks/sequence/lorenz63.npz",
            "source": "Inline RK4 solver (deterministic, seed=42)",
            "n": f"{_fmt_n(lorenz_info['n'])} traj." if lorenz_info else "1,200 traj.",
            "shape": "3×20 → 3×20",
            "firstRun": "Generated and cached on first run",
            "status": lorenz_status,
        },
        {
            "benchmark": "classification.plasma_stability",
            "cachePath": "benchmarks/classification/plasma_stability.npz",
            "source": "UCI repository CSV",
            "n": _fmt_n(plasma_info["n"]) if plasma_info else "10,000",
            "shape": plasma_info["shape"] if plasma_info else "12 → 1",
            "firstRun": "Downloaded from UCI on first run",
            "status": plasma_status,
        },
    ]


# ---------------------------------------------------------------------------
# TSX code generation
# ---------------------------------------------------------------------------

def _ts_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=True)


def _render_local(entries: list[dict]) -> str:
    lines = ["const LOCAL: Dataset[] = ["]
    for e in entries:
        n_val = e["n"]
        n_ts = str(n_val) if isinstance(n_val, int) else _ts_str(str(n_val))
        bench = "null" if e["benchmark"] is None else _ts_str(e["benchmark"])
        lines.append("  {")
        lines.append(f'    key: {_ts_str(e["key"])},')
        lines.append(f'    machine: {_ts_str(e["machine"])},')
        lines.append(f'    domain: {_ts_str(e["domain"])},')
        lines.append(f'    file: {_ts_str(e["file"])},')
        lines.append(f'    format: {_ts_str(e["format"])},')
        lines.append(f"    n: {n_ts},")
        lines.append(f'    shape: {_ts_str(e["shape"])},')
        lines.append(f'    targets: {_ts_str(e["targets"])},')
        lines.append(f"    benchmark: {bench},")
        lines.append(f'    status: {_ts_str(e["status"])},')
        lines.append(f'    notes: {_ts_str(e["notes"])},')
        lines.append("  },")
    lines.append("];")
    return "\n".join(lines)


def _render_io_ts(io: dict | None) -> list[str]:
    if not io:
        return []
    lines = ["    io: {"]
    lines.append("      inputs: [")
    for feat in io["inputs"]:
        lines.append(f'        {{ name: {_ts_str(feat["name"])}, desc: {_ts_str(feat["desc"])} }},')
    lines.append("      ],")
    if io.get("outputs"):
        lines.append("      outputs: [")
        for feat in io["outputs"]:
            lines.append(
                f'        {{ name: {_ts_str(feat["name"])}, desc: {_ts_str(feat["desc"])} }},'
            )
        lines.append("      ],")
    elif io.get("output"):
        out = io["output"]
        lines.append(
            f'      output: {{ name: {_ts_str(out["name"])}, desc: {_ts_str(out["desc"])} }},'
        )
    lines.append(f'      note: {_ts_str(io["note"])},')
    lines.append("    },")
    return lines


def _render_cache(entries: list[dict]) -> str:
    lines = ["const CACHE: CacheEntry[] = ["]
    for e in entries:
        lines.append("  {")
        lines.append(f'    benchmark: {_ts_str(e["benchmark"])},')
        lines.append(f'    cachePath: {_ts_str(e["cachePath"])},')
        lines.append(f'    source: {_ts_str(e["source"])},')
        lines.append(f'    n: {_ts_str(e["n"])},')
        lines.append(f'    shape: {_ts_str(e["shape"])},')
        lines.append(f'    firstRun: {_ts_str(e["firstRun"])},')
        lines.append(f'    status: {_ts_str(e["status"])},')
        lines.extend(_render_io_ts(e.get("io")))
        lines.append("  },")
    lines.append("];")
    return "\n".join(lines)


_LOCAL_START = re.compile(r"^const LOCAL: Dataset\[\] = \[")
_LOCAL_END = re.compile(r"^\];")
_CACHE_START = re.compile(r"^const CACHE: CacheEntry\[\] = \[")
_DATE_RE = re.compile(r"(updated )\d{4}-\d{2}-\d{2}")
_INSPECTED_RE = re.compile(r"(inspected )\d{4}-\d{2}-\d{2}")


def _replace_block(text: str, start_re: re.Pattern, new_block: str) -> str:
    lines = text.splitlines(keepends=True)
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and start_re.match(line.rstrip("\n")):
            start_idx = i
            continue
        if start_idx is not None and _LOCAL_END.match(line.rstrip("\n")):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not locate block for {start_re.pattern}")
    return "".join(lines[:start_idx]) + new_block + "\n\n" + "".join(lines[end_idx + 1 :])


def _patch_canvas(canvas_path: pathlib.Path, local_block: str, cache_block: str) -> None:
    text = canvas_path.read_text(encoding="utf-8")
    text = _replace_block(text, _LOCAL_START, local_block)
    text = _replace_block(text, _CACHE_START, cache_block)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = _DATE_RE.sub(rf"\g<1>{today}", text)
    text = _INSPECTED_RE.sub(rf"\g<1>{today}", text)
    canvas_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh datasets canvas from disk inventory")
    parser.add_argument(
        "--canvas",
        default=str(
            pathlib.Path.home()
            / ".cursor/projects/Users-asanche2-repos-SURGE/canvases/surge-datasets.canvas.tsx"
        ),
        help="Path to surge-datasets.canvas.tsx",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Try loading missing benchmark caches (qlknn needs fusion_surrogates)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print blocks, do not write")
    args = parser.parse_args()

    canvas_path = pathlib.Path(args.canvas)
    if args.warm_cache:
        print("Warming missing caches …", file=sys.stderr)
        _warm_missing_caches()

    local = _build_local_entries()
    cache = _build_cache_entries()
    local_block = _render_local(local)
    cache_block = _render_cache(cache)

    if args.dry_run:
        print(local_block)
        print()
        print(cache_block)
        return

    if not canvas_path.is_file():
        print(f"ERROR: canvas not found: {canvas_path}", file=sys.stderr)
        sys.exit(1)

    _patch_canvas(canvas_path, local_block, cache_block)
    cached = sum(1 for c in cache if c["status"] in ("cached-ok", "generated"))
    missing = sum(1 for c in cache if c["status"] == "missing")
    print(f"Updated {canvas_path}", file=sys.stderr)
    print(f"  local datasets: {len(local)}", file=sys.stderr)
    print(f"  cache ready: {cached}  missing: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
