"""Prototype: self-contained offline HTML report generated from run artifacts.

Reads only machine-readable SURGE outputs from a runs/<tag>/ directory
(spec.yaml, metrics.json, workflow_summary.json, predictions/*.parquet,
model_card_*.json, git_rev.txt) and emits ONE portable HTML file with
inline SVG figures — no server, no JS dependencies, works over ssh/scp.

Usage (from repo root):
    python audit/prototypes/make_run_report.py \
        [--run runs/diabetes_rf] [--out audit/prototypes/output/run_report.html]
"""
from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt

from surge_style import surge_theme, fmt_metric
from make_prototype_figures import parity_figure

REPO = Path(__file__).resolve().parents[2]

CSS = """
:root { color-scheme: light dark;
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --line:#e1e0d9; --good:#006300; --critical:#d03b3b; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --page:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --line:#2c2c2a; --good:#0ca30c; } }
body { font: 14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  color: var(--ink); background: var(--page); margin: 0; }
main { max-width: 880px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 22px; margin: 0 0 2px; }
h2 { font-size: 15px; margin: 32px 0 8px; }
.sub { color: var(--ink2); font-size: 13px; margin-bottom: 24px; }
.card { background: var(--surface); border: 1px solid var(--line);
  border-radius: 8px; padding: 16px 20px; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 500;
  border-bottom: 1px solid var(--line); padding: 4px 12px 4px 0; }
td { padding: 5px 12px 5px 0; border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums; }
.k { color: var(--ink2); }
figure { margin: 8px 0 0; } figure svg { max-width: 100%; height: auto; }
code { font: 12px ui-monospace,SFMono-Regular,Menlo,monospace;
  color: var(--ink2); }
"""


def fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def metric_rows(metrics: dict) -> str:
    rows = []
    for model, m in metrics.items():
        for split in ("train", "val", "test"):
            s = m.get(split, {})
            rows.append(
                f"<tr><td>{html.escape(model)}</td><td>{split}</td>"
                f"<td>{fmt_metric(s.get('r2'))}</td>"
                f"<td>{fmt_metric(s.get('rmse'), 'rmse')}</td>"
                f"<td>{fmt_metric(s.get('mae'), 'rmse')}</td></tr>")
    return "\n".join(rows)


def build(run_dir: Path, out_path: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = json.loads((run_dir / "workflow_summary.json").read_text())
    git_rev = (run_dir / "git_rev.txt").read_text().strip() \
        if (run_dir / "git_rev.txt").exists() else "unknown"
    spec = (run_dir / "spec.yaml").read_text() \
        if (run_dir / "spec.yaml").exists() else ""

    with surge_theme("light"):
        parity_svg = fig_to_svg(parity_figure(run_dir, "light"))

    ds = summary.get("dataset", {})
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SURGE run report — {html.escape(run_dir.name)}</title>
<style>{CSS}</style></head><body><main>
<h1>SURGE run report</h1>
<div class="sub">run <code>{html.escape(run_dir.name)}</code> ·
 git <code>{html.escape(git_rev[:12])}</code> ·
 surge {html.escape(str(summary.get('surge_version', '—')))}</div>

<h2>Dataset</h2><div class="card"><table>
<tr><td class="k">path</td><td><code>{html.escape(str(ds.get('path', '—')))}</code></td></tr>
<tr><td class="k">rows</td><td>{ds.get('n_rows', '—')}</td></tr>
<tr><td class="k">inputs → outputs</td>
<td>{len(ds.get('input_columns', []) or [])} → {len(ds.get('output_columns', []) or [])}</td></tr>
</table></div>

<h2>Metrics</h2><div class="card"><table>
<tr><th>model</th><th>split</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>
{metric_rows(metrics)}
</table></div>

<h2>Test-set diagnostics</h2>
<div class="card"><figure>{parity_svg}</figure></div>

<h2>Reproduce</h2>
<div class="card"><pre><code>{html.escape(spec)}</code></pre></div>
</main></body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REPO / "runs" / "diabetes_rf"))
    ap.add_argument("--out", default=str(
        Path(__file__).parent / "output" / "run_report.html"))
    args = ap.parse_args()
    print("wrote", build(Path(args.run), Path(args.out)))
