# Figure & benchmark upgrade plan

Status: ALL SIX PHASES IMPLEMENTED (2026-07-26). Kept for design
rationale; pending follow-ups now live in DEVELOPMENT_BACKLOG.md.
Scope: six phases, ordered by effort/payoff. Everything lands via PRs to
`main` under the PR-required ruleset.

## Background — why DeepONet failed on Gray-Scott

DeepONet represents the output field as a low-rank global expansion: a
branch MLP compresses the flattened 4096-pixel input into ~p latent
coefficients, and a trunk network supplies p global basis functions.
Gray-Scott next-step dynamics are almost entirely local (diffusion
stencil + pointwise reaction) and the Turing labyrinth has fine
structure everywhere — a low-rank global basis cannot represent
"sharpen each filament where it already is", so it regresses toward a
texture-like mean. FNO-2D (spectral conv) and U-Net (local conv) have
the right spatial inductive bias, hence the 10× gap. Fixes in order of
payoff: predict the residual B(t+1) − B(t); CNN branch; larger latent
dimension; longer training. See Phase 6.

## Phase 1 — Auto light/dark README figures (small)

GitHub natively supports theme-aware images:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/dark/leaderboard.png">
  <img src="docs/assets/readme/leaderboard.png">
</picture>
```

The gallery already generates a dark variant of every figure — they are
just not tracked or referenced.

- Add `docs/assets/readme/dark/` with regenerated dark variants of every
  README figure (same filenames as the light ones).
- Wrap every README `<img>` in a `<picture>` block.
- RTD/Sphinx has no native equivalent — scope to README now, docs later.

## Phase 2 — Benchmark leaderboard figure, futuristic redesign (medium)

Rebuild the static leaderboard PNG in the style of the HTML dashboard
(`surge/report/leaderboard.py`, which stays the richer artifact):

- Dark "mission-control" canvas: near-black background, blue gradient
  bars with subtle glow, condensed tech typography for model names,
  threshold gate as a glowing vertical line with pass/fail tinting.
- Benchmark identity strip: dataset name, n samples, citation, and the
  task in math notation, e.g.
  (R/L_Ti, q, s_hat, …) ↦ q_ITG [GB]  (rendered via mathtext \mathrm),
  plus a small data-preview inset reusing
  `surge/report/dataset_previews.py`.
- Compact spider chart panel (accuracy / speed / calibration /
  robustness) in the HTML-dashboard style.
- Light variant derived from the same code so Phase 1 picks it up.

## Phase 3 — Fix the flagged figures (medium, same session as Phase 2)

**TheWell Gray-Scott figure** (`examples/thewell_grayscott_study.py`):

- Lock ALL prediction panels to the truth's color scale with one shared
  colorbar — Ridge/DeepONet currently auto-scale to their own noise
  range, which is unreadable and slightly misleading.
- Add the task notation B(t) ↦ B(t+1) to the identity strip.
- Add the DeepONet-residual variant once Phase 6 confirms it helps.

**Ensemble figure** (`examples/viz_theme_gallery.py::ensemble_figure`):

- Panel (a): show 2–3 individual member traces faintly so the concept
  is visible — "6 MLPs, different random seeds; mean = prediction,
  spread = uncertainty".
- Panel (b): readable 2D density for spread-vs-error (current beige
  under-color swamps it).
- Panel (c): coverage with a one-line takeaway annotation ("raw spread
  too narrow → multiply σ by 2.8 → honest 68/95%").
- Plus a 3-sentence README caption explaining what a deep ensemble is.

## Phase 4 — ConStellaration stellarator benchmark (large; flagship)

Proxima Fusion's ConStellaration dataset — HuggingFace
`proxima-fusion/constellaration`, ~150k QI stellarator boundaries +
VMEC ideal-MHD equilibria, MIT-licensed, a few GB. VERIFY size, schema,
and license at implementation time (knowledge-cutoff facts).

- New loader `surge/benchmarks/loaders/constellaration.py`
  (HF `datasets`).
- Task: boundary Fourier coefficients (R_mn, Z_mn) ↦ equilibrium
  figures of merit (max elongation, rotational transform ι, QI
  residual, mirror ratio).
- Register `fusion.constellaration` benchmarks with thresholds; run the
  portfolio (residual MLP, GBM, GP, KAN).
- Hero figure: plasma boundary cross-sections at several toroidal
  angles reconstructed from the Fourier coefficients — actual
  stellarator shapes, colored by prediction quality.
- TheWell `MHD_64` stays deferred (3D; needs a 3D operator backend).
- Disk note: TheWell Gray-Scott cache is 132 GB at
  `~/.surge/data/thewell/` (117 train + 15 valid; the study samples
  only ~500 fields). Trim the train split first if space is tight.

## Phase 5 — MLflow "mission control" for the QLKNN case (medium)

Dark dashboard-composite figure for one QLKNN HPO campaign (reference:
user-supplied "TERMINAL RUN" dashboard — inspiration, not a copy):

- Per-trial loss curves (fading grays, best trial in blue).
- Running-best HPO trace; hyperparameter-importance bars.
- Run-summary card (best R², epochs, params, runtime) in terminal-card
  styling.
- Generated from the JSONL training logs + Optuna study we already
  persist; plus a real MLflow UI screenshot showing nested HPO runs.
- Before drawing, scan how MLflow dashboards are presented in
  media/docs for layout ideas.

## Phase 6 — DeepONet rehabilitation experiment (small; 2 × ~8 min runs)

Rerun Gray-Scott with residual-target DeepONet (and optionally a CNN
branch) so the study reports *why* it fails and *what fixes it* —
a stronger scientific statement than a bad bar.

## Execution order

1 (small) → 2+3 (one session) → 5 → 4 (needs dataset download +
verification) → 6.

Open question for the user: any constraint on downloading
ConStellaration (~GBs) given the 132 GB TheWell footprint?
