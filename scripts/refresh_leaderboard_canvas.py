#!/usr/bin/env python
"""Regenerate the inline data block in surge/viz/leaderboard.canvas.tsx.

Reads every benchmark_reports/<benchmark>/<timestamp>/result.json and
aggregates per (benchmark_key, model_key) — averaging the primary metric
across all stored results (seeds / reruns) — then rewrites the
``const BENCHMARKS: Benchmark[] = [...]`` block in the canvas TSX.

Usage::

    python scripts/refresh_leaderboard_canvas.py
    python scripts/refresh_leaderboard_canvas.py --reports path/to/benchmark_reports
    python scripts/refresh_leaderboard_canvas.py --dry-run   # print generated data, don't write
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Primary metric selection
# ---------------------------------------------------------------------------

# Ordered preference: first key present in the result wins.
_METRIC_PREFERENCE = [
    # PDE / operator learning
    "test_relative_l2",
    # Sequence
    "test_nrmse",
    # ConStellaration paper aggregate
    "test_r2_mean",
    # Standard regression
    "test_r2",
    # Standard classification
    "test_accuracy",
]

_METRIC_LABEL: dict[str, str] = {
    "test_r2":          "R²",
    "test_r2_mean":     "R² (mean)",
    "test_accuracy":    "Accuracy",
    "test_relative_l2": "Relative L²",
    "test_nrmse":       "NRMSE",
}

# For accuracy / classification metrics, higher is better.
# For relative_l2 / nrmse, lower is better (flip sign in sort so best = first).
_LOWER_IS_BETTER = {"test_relative_l2", "test_nrmse"}


# ---------------------------------------------------------------------------
# Category mapping (mirroring registry.py so we don't import surge)
# ---------------------------------------------------------------------------

def _infer_category(benchmark_key: str) -> str:
    prefix = benchmark_key.split(".")[0]
    mapping = {
        "tabular":        "tabular",
        "multioutput":    "tabular",
        "sequence":       "tabular",
        "synthetic":      "tabular",
        "classification": "tabular",
        "image":          "image",
        "vision":         "image",
        "pde":            "field",
        "pdebench":       "field",
        "thewell":        "field",
        "field":          "field",
        "plasma":         "plasma",
        "fusion":         "plasma",
    }
    return mapping.get(prefix, prefix)


# ---------------------------------------------------------------------------
# Read results
# ---------------------------------------------------------------------------

def _load_results(reports_root: pathlib.Path) -> dict:
    """Return dict[benchmark_key][model_key] = list of result dicts."""
    data: dict[str, dict[str, list[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for result_json in sorted(reports_root.rglob("result.json")):
        try:
            d = json.loads(result_json.read_text(encoding="utf-8"))
            bkey = d["benchmark_key"]
            mkey = d["model_key"]
            data[bkey][mkey].append(d)
        except Exception:
            pass
    return data


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _primary_metric(metrics: dict[str, float]) -> tuple[str, float] | None:
    for k in _METRIC_PREFERENCE:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, float(metrics[k])
    return None


def _aggregate(results: list[dict]) -> dict | None:
    """Average primary metric across all stored runs for one (bench, model)."""
    values: list[float] = []
    passed_list: list[bool | None] = []
    metric_key: str | None = None

    for r in results:
        hit = _primary_metric(r.get("metrics", {}))
        if hit is None:
            continue
        k, v = hit
        if metric_key is None:
            metric_key = k
        elif k != metric_key:
            continue
        values.append(v)
        passed_list.append(r.get("passed"))

    if not values:
        return None

    avg = statistics.mean(values)
    passed = None
    if passed_list:
        non_none = [p for p in passed_list if p is not None]
        if non_none:
            passed = any(non_none)

    return {"score": avg, "n": len(values), "passed": passed, "metric_key": metric_key}


# ---------------------------------------------------------------------------
# Build TypeScript data literal
# ---------------------------------------------------------------------------

def _ts_bool(v: bool | None) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return "null"


def _build_ts_data(data: dict) -> str:
    """Return the full replacement block (comment + types + BENCHMARKS const)."""
    now = datetime.now(timezone.utc).strftime("%b %-d %Y")

    lines: list[str] = []
    lines.append(f"// ── Inline data from benchmark_reports/ (snapshot {now}) ───────────────")
    lines.append("")
    lines.append("type ModelResult = { model: string; score: number; n: number; passed: boolean | null };")
    lines.append("type Benchmark = {")
    lines.append("  key: string;")
    lines.append("  category: string;")
    lines.append("  metric: string;")
    lines.append("  results: ModelResult[];")
    lines.append("};")
    lines.append("")
    lines.append("const BENCHMARKS: Benchmark[] = [")

    # Sort benchmarks: by category order then by key
    CAT_ORDER = ["tabular", "image", "field", "plasma"]

    def _sort_key(bkey: str) -> tuple:
        cat = _infer_category(bkey)
        try:
            ci = CAT_ORDER.index(cat)
        except ValueError:
            ci = len(CAT_ORDER)
        return (ci, bkey)

    for bkey in sorted(data.keys(), key=_sort_key):
        models = data[bkey]
        # Aggregate per model
        agg_results: list[dict] = []
        metric_key: str | None = None
        for mkey, results in models.items():
            agg = _aggregate(results)
            if agg is None:
                continue
            agg["model"] = mkey
            agg_results.append(agg)
            if metric_key is None:
                metric_key = agg["metric_key"]

        if not agg_results:
            continue

        if metric_key is None:
            continue

        lower = metric_key in _LOWER_IS_BETTER
        agg_results.sort(key=lambda x: x["score"], reverse=not lower)
        metric_label = _METRIC_LABEL.get(metric_key, metric_key)
        category = _infer_category(bkey)

        lines.append(f"  // ── {category}: {bkey} {'─' * max(0, 50 - len(bkey))}")
        lines.append("  {")
        lines.append(f'    key: "{bkey}",')
        lines.append(f'    category: "{category}",')
        lines.append(f'    metric: "{metric_label}",')
        lines.append("    results: [")
        for r in agg_results:
            model_padded = f'"{r["model"]}"'.ljust(42)
            lines.append(
                f'      {{ model: {model_padded} score: {r["score"]:.4f}, '
                f'n: {r["n"]:<3}, passed: {_ts_bool(r["passed"])} }},'
            )
        lines.append("    ],")
        lines.append("  },")

    lines.append("];")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patch the canvas file
# ---------------------------------------------------------------------------

_DATA_START_RE = re.compile(r"^// ── Inline data from benchmark_reports/")
_DATA_END_RE = re.compile(r"^\];")  # end of BENCHMARKS array


def _patch_canvas(canvas_path: pathlib.Path, new_data_block: str) -> None:
    text = canvas_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Find the line range to replace:
    # from the "// ── Inline data" comment to the closing "];" of BENCHMARKS
    start_idx: int | None = None
    end_idx: int | None = None
    in_benchmarks = False

    for i, line in enumerate(lines):
        if start_idx is None and _DATA_START_RE.match(line):
            start_idx = i
        if start_idx is not None and "const BENCHMARKS" in line:
            in_benchmarks = True
        if in_benchmarks and _DATA_END_RE.match(line):
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError(
            f"Could not locate data block in {canvas_path}. "
            f"start_idx={start_idx} end_idx={end_idx}"
        )

    before = lines[:start_idx]
    after = lines[end_idx + 1 :]
    new_text = "".join(before) + new_data_block + "\n" + "".join(after)
    canvas_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Snapshot date in footer
# ---------------------------------------------------------------------------

_SNAPSHOT_RE = re.compile(r"(snapshot\s+)\w+ \d+ \d{4}")


def _patch_footer_date(canvas_path: pathlib.Path) -> None:
    now = datetime.now(timezone.utc).strftime("%b %-d %Y")
    text = canvas_path.read_text(encoding="utf-8")
    new_text = _SNAPSHOT_RE.sub(rf"\g<1>{now}", text)
    if new_text != text:
        canvas_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh leaderboard canvas data")
    parser.add_argument(
        "--reports",
        default="benchmark_reports",
        help="Path to benchmark_reports/ directory (default: benchmark_reports)",
    )
    parser.add_argument(
        "--canvas",
        default="surge/viz/leaderboard.canvas.tsx",
        help="Path to the canvas TSX file (default: surge/viz/leaderboard.canvas.tsx)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated data block instead of writing the file",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).parent.parent
    reports_root = pathlib.Path(args.reports)
    if not reports_root.is_absolute():
        reports_root = repo_root / reports_root
    canvas_path = pathlib.Path(args.canvas)
    if not canvas_path.is_absolute():
        canvas_path = repo_root / canvas_path

    if not reports_root.is_dir():
        print(f"ERROR: reports directory not found: {reports_root}", file=sys.stderr)
        sys.exit(1)
    if not canvas_path.is_file():
        print(f"ERROR: canvas file not found: {canvas_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading results from {reports_root} ...", file=sys.stderr)
    data = _load_results(reports_root)
    n_benchmarks = len(data)
    n_runs = sum(len(r) for models in data.values() for r in models.values())
    print(f"  {n_benchmarks} benchmarks, {n_runs} model-run files", file=sys.stderr)

    ts_block = _build_ts_data(data)

    if args.dry_run:
        print(ts_block)
        return

    _patch_canvas(canvas_path, ts_block)
    _patch_footer_date(canvas_path)
    print(f"Updated: {canvas_path}", file=sys.stderr)
    print(f"  Snapshot date: {datetime.now(timezone.utc).strftime('%b %-d %Y')}", file=sys.stderr)
    print(f"  {n_benchmarks} benchmarks written", file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    main()
