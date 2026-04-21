---
skill_id: surge.m3dc1.eval_2d_delta_p
title: M3DC1 delta-p 2D test map evaluation
version: "1.0.0"
domain: [surge, m3dc1, evaluation, visualization]
academy_blueprint_hint: null
---

## Objective

Run **`eval_test_delta_p_2d_maps.py`** with a **consistent** run directory, **model key**, and **checkpoint** so 2D δ*p* test visuals match the trained **`torch_mlp_profiles_deltap`** (or configured) model.

## When to apply (triggers)

- User invokes **`scripts/m3dc1/eval_test_delta_p_2d_maps.py`** with a **`runs/...`** path.
- User sees TensorFlow/XLA **registration warnings** or GPflow notices and wonders if the run failed.

## Preconditions

- Training run exists under **`runs/<run_tag>/`** with **`checkpoints/<model_key>/epoch_XXXX.pt`**.
- Same **conda**/torch stack used for training if CUDA is required (CPU may be enough for inference—follow script behavior).

## Procedure

1. Confirm **`--model`** matches the **registry name** used in training (e.g. **`torch_mlp_profiles_deltap`**).
2. Set **`--checkpoint`** to a file under **`runs/<run_tag>/checkpoints/...`** that belongs to that model.
3. Pass **`--m-source`** and m-range flags consistent with the evaluation design (e.g. **`range --m-min -100 --m-max 100`**).
4. **`--plot-mode none`** still allows other outputs (e.g. error panels); adjust per user need.
5. If logs show **TensorFlow duplicate plugin registration** or **TF-TRT Warning**, treat as **benign** unless the process **exits non-zero**.
6. **`GPflow not available`**: ignore for **pure PyTorch δ*p* eval** unless the script path requires GPR.

## Verification

- Script exits **0**; expected outputs under **`runs/<run_tag>/plots/delta_p_2d_test/...`** (layout may vary by script version).

## Failure modes & diagnostics

| Symptom | Likely cause | Next step |
|---------|----------------|-----------|
| Checkpoint not found | Wrong **`--checkpoint`** or run dir | List **`checkpoints/`** tree |
| Shape / key errors | **`--model`** mismatch | Compare training config **`name`** under **`models`** |

## Artifacts & code references

- `scripts/m3dc1/eval_test_delta_p_2d_maps.py`
- Example run tag pattern: **`m3dc1_trial12_e500_bs128_withval_*`**
- Curated example figure (may differ): `m3dc1/figures/delta_p_2d_test_mlp_side_by_side_example.png`

## Future: Academy mapping

- **Tool:** `run_delta_p_2d_eval(run_dir: Path, model: str, checkpoint: Path, flags: list[str]) -> EvalSummary`
- Could run as a **remote tool** on a login node with filesystem access to **`runs/`**.
