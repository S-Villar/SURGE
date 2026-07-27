---
name: surge-wizard
description: Interactive Q&A that builds and runs a SURGE workflow for the user - asks what they want to predict, from which file, with what compute/time budget, then generates a commented spec.yaml and runs it. Use when the user is unsure how to start, says "help me set up a run", "I don't know which spec/options to use", or gives a dataset with only a vague goal.
---

# SURGE wizard — from questions to a running workflow

Turn a vague goal into a running, reproducible workflow. Ask BEFORE
generating; never make the user edit YAML by hand. The generated
spec.yaml is the deliverable — YAML stays the source of truth, the
wizard only writes it for them.

## 1. Ask (use AskUserQuestion; one round, 3–4 questions max)

Only ask what you cannot infer. ALWAYS inspect the data file first
(`surge.data_loader` schema inference / pandas peek) — infer column
count, dtypes, size, and obvious target candidates, then ask:

1. **Target** — "Which column(s) should be predicted?" (options = the
   most plausible columns from inspection, plus Other).
2. **Goal** — accuracy-first, speed-first (real-time surrogate), or
   uncertainty-needed (this decides the model slate: HPO'd residual MLP
   vs GBM vs GP/ensemble).
3. **Budget** — smoke test (~1 min, small subsample), standard (~10 min,
   HPO 10 trials), thorough (hours, HPO 40+ trials, several seeds).
4. **Tracking** — MLflow on/off (only if ambiguous).

Skip questions whose answer is obvious from the request. For field/2D
data (flat n² columns or 3D arrays) default the slate to
fno2d/unet + ridge baseline instead of tabular models.

## 2. Generate

Write `specs/<name>.yaml` following the task-shape → model table in
[surge-build-surrogate](../surge-build-surrogate/SKILL.md). Rules:

- comment every non-obvious key inline (`# 10 trials ≈ 8 min CPU`)
- always include one fast baseline (ridge / random_forest) next to the
  headline model — leaderboards need an anchor
- standardize_inputs: true unless tree-only slate
- budget=smoke → subsample + n_epochs≤10, no HPO; standard → HPO 10
  trials; thorough → HPO 40 trials + `--seeds 3`

## 3. Run + report

```bash
surge run specs/<name>.yaml --tag <name>
```

Then follow [surge-viz](../surge-viz/SKILL.md): report test metrics vs
the baseline, point at `runs/<tag>/`, and offer the parity figure. If
metrics look wrong (R² < baseline, exploding loss), diagnose before
suggesting bigger models — usually target scaling or leakage.
