---
name: surge-viz
description: Produce SURGE figures, reports, and dashboards with the SURGE visual system (surge.viz.theme). Use when asked for plots, publication figures, the HTML leaderboard, run reports, or when writing ANY new plotting code in this repo.
---

# SURGE visual system

All new plotting code MUST use `surge.viz.theme` — never ad-hoc styling:

```python
from surge.viz.theme import surge_theme, save_figure, fmt_metric, \
    density_cmap, sequential_cmap, diverging_cmap

with surge_theme("light") as p:        # or "dark"; yields the palette dict
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y)                       # series colors auto-applied
save_figure(fig, out_dir / "name")     # deterministic name.png/.svg/.pdf
```

Rules encoded in the theme (do not bypass):
- categorical series: fixed 8-slot order via `p["series"]`, never cycled
  past 8; status colors `p["good"]/p["critical"]` reserved for PASS/FAIL;
  text uses `p["ink"]/p["ink2"]/p["muted"]`, never series colors.
- density/parity plots: `density_cmap(mode)` = reversed plasma with the
  publication under-color; pair with `LogNorm(vmin=1)` + `cmin=1`
  (signature style: dashed identity line, R² box, panel letters (a)(b)).
- signed fields/errors: `diverging_cmap(mode)` + `CenteredNorm`.
- HPO history: solid per-trial trace, dashed running best, gold-edged
  star + labelled score box at the best trial.
- one y-axis per chart; two measures = two panels.

## Regenerating deliverables

```bash
# figure gallery (parity, training, HPO, leaderboard, classification,
# field_operator, uncertainty, characterization) x light/dark:
python examples/viz_theme_gallery.py [--only parity ...] [--modes light]
# self-contained HTML leaderboard (spiders + dataset previews + tables):
python -m surge.report.leaderboard --out examples/viz_gallery_output/surge_leaderboard.html [--mode light]
```

Outputs go to `examples/viz_gallery_output/` (git-ignored). Reports read
ONLY `benchmark_reports/**/result.json` + `surge/benchmarks/metadata.yaml`
+ cached datasets (network is socket-blocked during preview rendering) —
never hand-encode results into a report or dashboard.

Legacy modules (`surge/viz/benchmark.py` neon spider theme, `run_viz.py`,
`comparison.py`) predate the theme; when touching them, migrate them onto
`surge.viz.theme` rather than extending the old styling.
