# HPO vs Baseline — γ (growth rate)

Batch HPO over every usable regressor, each optimized with its built-in recipe
(Optuna TPE) on the **same** shared case-grouped split as the baseline, so the
comparison is apples-to-apples.

Reproduce:

```bash
# baseline (default hyperparameters)
python scripts/m3dc1/internal/run_workflow.py configs/internal/m3dc1_gamma_ver_allmodels.yaml
# HPO batch (recipe-based, hpo.enabled + no search_space)
python scripts/m3dc1/internal/run_workflow.py configs/internal/m3dc1_gamma_ver_batch_hpo.yaml
```

## Results (test R², higher is better)

| model | baseline | HPO | Δ |
|---|---|---|---|
| **torch_mlp** | 0.8523 | **0.8683** | +0.0161 |
| random_forest | 0.8524 | 0.8546 | +0.0023 |
| mlp_ensemble | 0.8575 | 0.8501 | −0.0075 |
| ft_transformer | 0.7820 | 0.8495 | **+0.0675** |
| residual_mlp | 0.8329 | 0.8367 | +0.0038 |
| sk_mlp | 0.8225 | 0.8238 | +0.0013 |
| gradient_boosting | 0.8035 | 0.8035 | +0.0000 |
| geom_residual_mlp | 0.3896 | 0.7427 | **+0.3531** |
| ridge | 0.5316 | 0.5315 | −0.0001 |

**Best model overall: `torch_mlp`, test R² = 0.868** (up from the baseline best
of 0.858, `mlp_ensemble`).

## Takeaways

- HPO improved **6 of 9** models. The biggest wins were for models whose
  defaults were poorly suited to this task: `geom_residual_mlp` (+0.35) and
  `ft_transformer` (+0.07).
- Well-defaulted models (`random_forest`, `mlp_ensemble`) barely moved; small
  ± changes there are within val/test noise (small dataset, ~2k test cases).
- `ridge` and `gradient_boosting` are essentially flat — little headroom from
  their (few) tunable knobs on this scalar target.

## Notes / gotchas learned

- The `sklearn.mlp` recipe originally lacked early stopping; some trials ran the
  full `max_iter` on CPU for 10–20 min each and stalled the batch. Fixed by
  adding `early_stopping` + bounded `max_iter` to the recipe. Even so, sklearn
  MLP is CPU-bound and slow here — prefer `pytorch.mlp` for MLPs.
- Recipe coverage is now complete for all usable regressors. Recipes were added
  for `gpflow.gpr`, `gpflow.multi_kernel`, `lgbm.*`, `catboost.*` (the latter
  three are untested here — optional backends not installed).
- Sampler defaults to Optuna **TPE**; set `sampler: botorch` per model in the
  workflow config to use **BoTorch** Bayesian optimization instead
  (botorch/gpytorch are installed).

The δp per-mode HPO uses the same mechanism on a capped case subset
(`configs/internal/m3dc1_deltap_real_per_mode_hpo.yaml`, SLURM).
