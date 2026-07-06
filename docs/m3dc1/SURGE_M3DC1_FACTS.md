# SURGE M3DC1 — Measured Facts Only

Machine-readable companion for Opus. No recommendations. Source: `field_bench/`, `predictions_cache.npz`, `run_config.json`, `spectrum_image_metrics.json`.

Generated from repo: `/global/u2/a/asvillar/src/SURGE`

## 1. Dataset and split

- `batch_dir`: /pscratch/sd/a/asvillar/mp288/jobs/batch_16
- `filename`: csdata_deltap_b_ver.h5
- `n_cases_total`: 9974
- `n_train`: 6983
- `n_val`: 997
- `n_test`: 1994
- `quarantined_cases`: 2
- `grid`: 128
- `m_window_default`: [-80.0, 20.0]
- `test_frac`: 0.2
- `val_frac`: 0.1
- `seed`: 42

## 2. Run identifiers

| short_id | run_dir | target_smooth | geom_channels | select_by | field_loss_weight |
|---|---|---:|---|---|---:|
| qc_peak4 | `runs/spectrum_fno48_floor6_smooth1_qc_peak4` | 1.0 | False | composite | 0 |
| qc | `runs/spectrum_fno48_floor6_smooth1_qc` | 1.0 | ? | composite | 0 |
| fieldloss_sigma1 | `runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` | 1.0 | False | field | 0.5 |
| run_B_geom | `runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom` | 1.0 | True | field | 0.5 |
| run_D_smooth05 | `runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05` | 0.5 | False | field | 0.5 |
| run_Dprime_smooth0 | `runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0` | 0.0 | False | field | 0.5 |
| qc_mhi100 | `runs/spectrum_fno48_floor6_smooth1_qc_mhi100` | 1.0 | True | composite | 0 |

## 3. Field benchmark (test split, n=1994, oracle phase, max-normalized relL2)

| short_id | model_name | frac_relL2_gt_1 | p90_relL2 | median_relL2 | mean_relL2 | mean_relL2_alpha | mean_crf | relL2_bins |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fieldloss_sigma1 | `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` | 0.100301 | 1.000982 | 0.632380 | 0.702172 | 0.630578 | 0.906868 | <0.3:5, <0.5:207, <0.7:1070, <1.0:512, >1.0:200 |
| run_B_geom | `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom` | 0.122367 | 1.060514 | 0.642282 | 0.719473 | 0.634677 | 0.928602 | <0.3:11, <0.5:164, <0.7:1077, <1.0:498, >1.0:244 |
| run_D_smooth05 | `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05` | 0.051153 | 0.819828 | 0.455423 | 0.527356 | 0.479942 | 0.711472 | <0.3:120, <0.5:1031, <0.7:492, <1.0:249, >1.0:102 |
| run_Dprime_smooth0 | `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0` | 0.048144 | 0.802933 | 0.420863 | 0.490567 | 0.448424 | 0.605373 | <0.3:433, <0.5:808, <0.7:437, <1.0:220, >1.0:96 |
| qc_mhi100 | `spectrum_fno48_floor6_smooth1_qc_mhi100` | 0.266800 | 1.350491 | 0.825349 | 0.921303 | 0.774071 | 0.788205 | <0.3:0, <0.5:19, <0.7:298, <1.0:1145, >1.0:532 |

Broken-case counts (relL2>1 from relL2_bins `>1.0`): see table above.

## 4. Spectrum pattern R² (test split, from predictions_cache.npz)

| short_id | n_test | patR2_median | patR2_mean | patR2_min | patR2_max | pct_lt_0.9 | pct_lt_0.7 | pct_lt_0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qc_peak4 | 1994 | 0.910263 | 0.894521 | -0.123021 | 0.996702 | 36.41% | 2.51% | 1.05% |
| qc | 1994 | 0.907999 | 0.891847 | -0.328849 | 0.997177 | 40.17% | 2.41% | 0.90% |
| fieldloss_sigma1 | 1994 | 0.908058 | 0.888431 | -0.644259 | 0.994431 | 39.27% | 3.16% | 1.20% |
| run_B_geom | 1994 | 0.912202 | 0.897988 | 0.183657 | 0.996420 | 38.11% | 2.36% | 0.80% |
| run_D_smooth05 | 1994 | 0.826073 | 0.816105 | -0.115711 | 0.996095 | 90.32% | 5.97% | 0.95% |
| run_Dprime_smooth0 | 1994 | 0.750145 | 0.743726 | -0.743333 | 0.991806 | 96.79% | 23.17% | 1.86% |
| qc_mhi100 | 1994 | 0.931525 | 0.906164 | -0.960747 | 0.997959 | 18.36% | 3.41% | 1.65% |

### Pattern R² quantiles (test)
- **qc_peak4**: p02=0.6623, p10=0.8589, p50=0.9103, p90=0.9393, p98=0.9873
- **qc**: p02=0.6736, p10=0.8427, p50=0.9080, p90=0.9378, p98=0.9863
- **fieldloss_sigma1**: p02=0.6186, p10=0.8494, p50=0.9081, p90=0.9360, p98=0.9789
- **run_B_geom**: p02=0.6684, p10=0.8467, p50=0.9122, p90=0.9523, p98=0.9847
- **run_D_smooth05**: p02=0.5861, p10=0.7277, p50=0.8261, p90=0.8994, p98=0.9803
- **run_Dprime_smooth0**: p02=0.5026, p10=0.6636, p50=0.7501, p90=0.8263, p98=0.9737
- **qc_mhi100**: p02=0.5574, p10=0.8664, p50=0.9315, p90=0.9547, p98=0.9835

## 5. Pairwise field relL2 wins (lower wins, same 1994 test cases)

- `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_vs_spectrum_fno48_floor6_smooth1_qc_peak4`: {'spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_wins': 1691, 'spectrum_fno48_floor6_smooth1_qc_peak4_wins': 303, 'ties': 0}
- `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_vs_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0`: {'spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_wins': 157, 'spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_wins': 1837, 'ties': 0}
- `spectrum_fno48_floor6_smooth1_qc_peak4_vs_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0`: {'spectrum_fno48_floor6_smooth1_qc_peak4_wins': 112, 'spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_wins': 1882, 'ties': 0}

### Run D′ vs Run D (merged per_case from smooth0 and smooth05 benches)
- `smooth0_lower_relL2_wins`: 1466
- `smooth05_lower_relL2_wins`: 528
- `ties`: 0
- `mean_relL2_delta_smooth0_minus_smooth05`: -0.036788822467402206
- `median_relL2_delta`: -0.03622600000000001

## 6. Per-equilibrium-family field metrics (selected families)

### sparc_1427

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 22.0 | 0.810187 | 0.725625 | 1.061639 | 0.136364 | 0.934928 |
| run_D_smooth05 | 22.0 | 0.672855 | 0.597460 | 0.871239 | 0.045455 | 0.748630 |
| run_Dprime_smooth0 | 22.0 | 0.623280 | 0.578300 | 0.849104 | 0.090909 | 0.652554 |
| run_B_geom | 22.0 | 0.856475 | 0.759376 | 1.119713 | 0.227273 | 0.948438 |

### sparc_1430

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 19.0 | 0.893680 | 0.873059 | 1.085564 | 0.263158 | 0.948258 |
| run_D_smooth05 | 19.0 | 0.744925 | 0.671116 | 1.055443 | 0.263158 | 0.855031 |
| run_Dprime_smooth0 | 19.0 | 0.844177 | 0.811902 | 1.263856 | 0.315789 | 0.804410 |
| run_B_geom | 19.0 | 0.841854 | 0.788746 | 1.124615 | 0.315789 | 0.964515 |

### sparc_1530

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 20.0 | 0.812959 | 0.771155 | 1.081498 | 0.250000 | 0.979150 |
| run_D_smooth05 | 20.0 | 0.616445 | 0.613941 | 0.908788 | 0.050000 | 0.908008 |
| run_Dprime_smooth0 | 20.0 | 0.689090 | 0.674321 | 0.886080 | 0.050000 | 0.889715 |
| run_B_geom | 20.0 | 0.677671 | 0.634579 | 0.964680 | 0.050000 | 0.972618 |

### sparc_1500

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 23.0 | 0.490049 | 0.393007 | 0.663415 | 0.086957 | 0.897126 |
| run_D_smooth05 | 23.0 | 0.414814 | 0.351356 | 0.546349 | 0.000000 | 0.899123 |
| run_Dprime_smooth0 | 23.0 | 0.415643 | 0.359507 | 0.514036 | 0.086957 | 0.873595 |
| run_B_geom | 23.0 | 0.448052 | 0.391399 | 0.531912 | 0.043478 | 0.953736 |

### sparc_1524

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 23.0 | 1.013972 | 0.860478 | 1.708153 | 0.347826 | 0.903275 |
| run_D_smooth05 | 23.0 | 0.842016 | 0.780316 | 1.261015 | 0.260870 | 0.728931 |
| run_Dprime_smooth0 | 23.0 | 0.867366 | 0.753464 | 1.178989 | 0.347826 | 0.617327 |
| run_B_geom | 23.0 | 1.019952 | 0.915084 | 1.516316 | 0.260870 | 0.942861 |

### sparc_1300

| short_id | n | mean_relL2 | median_relL2 | p90_relL2 | frac_relL2_gt_1 | mean_crf |
|---|---:|---:|---:|---:|---:|---:|
| fieldloss_sigma1 | 21.0 | 0.456987 | 0.426008 | 0.631017 | 0.047619 | 0.889990 |
| run_D_smooth05 | 21.0 | 0.479253 | 0.418296 | 0.700215 | 0.047619 | 0.891466 |
| run_Dprime_smooth0 | 21.0 | 0.454099 | 0.420092 | 0.623982 | 0.047619 | 0.853418 |
| run_B_geom | 21.0 | 0.447279 | 0.422968 | 0.665821 | 0.047619 | 0.935269 |

## 7. sparc_1530 — top 5 fieldloss regressions vs peak4 (by relL2 increase)

| case_key | relL2_peak4 | relL2_fieldloss_sigma1 | patR2_peak4 | patR2_fieldloss |
|---|---:|---:|---:|---:|
| run79_sparc_1530 | 0.496567 | 1.032458 | 0.895632 | 0.061429 |
| run43_sparc_1530 | 0.816470 | 1.298781 | 0.432810 | -0.180970 |
| run99_sparc_1530 | 0.655249 | 1.078824 | 0.404467 | -0.241258 |
| run31_sparc_1530 | 0.631002 | 1.037485 | 0.960259 | 0.117388 |
| run83_sparc_1530 | 0.793379 | 1.105557 | 0.884501 | -0.227872 |

## 8. Training completion (from history_fno2d.jsonl last epoch)

| short_id | epochs_completed | final_val_r2 | training_epochs in run_config |
|---|---:|---:|---:|
| qc_peak4 | 179 | None | 400 |
| qc | 289 | None | 400 |
| fieldloss_sigma1 | 400 | 0.9029014408588409 | 400 |
| run_B_geom | 214 | 0.903349831700325 | 400 |
| run_D_smooth05 | 400 | 0.8744910061359406 | 400 |
| run_Dprime_smooth0 | 400 | 0.8346429020166397 | 400 |
| qc_mhi100 | 237 | 0.8963351100683212 | 400 |

## 9. Field-bench ranking order (per bench file)

- **bench `field_bench/with_fieldloss`**: `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` > `spectrum_fno48_floor6_smooth1_qc_peak4` > `spectrum_fno48_floor6_smooth1_qc` > `spectrum_fno48_floor6_smooth1_qc_mhi100`
- **bench `field_bench/with_fieldloss_geom`**: `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` > `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom` > `spectrum_fno48_floor6_smooth1_qc_peak4`
- **bench `field_bench/with_fieldloss_smooth05`**: `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05` > `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` > `spectrum_fno48_floor6_smooth1_qc_peak4`
- **bench `field_bench/with_fieldloss_smooth0`**: `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0` > `spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss` > `spectrum_fno48_floor6_smooth1_qc_peak4`
- **bench `field_bench`**: `spectrum_fno48_floor6_smooth1_qc_peak4` > `spectrum_fno48_floor6_smooth1_qc` > `spectrum_fno48_floor6_smooth1_qc_mhi100`

## 10. File paths (artifacts)

### figures
- `docs/m3dc1/assets/field_bench_frac_gt1_sparc_families_smooth0.png`
- `docs/m3dc1/assets/field_bench_frac_gt1_sparc_families_smooth05.png`
- `docs/m3dc1/assets/field_bench_frac_gt1_sparc_fieldloss_smooth0.png`
- `docs/m3dc1/assets/field_bench_frac_gt1_sparc_fieldloss_smooth05.png`
- `docs/m3dc1/assets/field_bench_leaderboard_fieldloss_smooth0.json`
- `docs/m3dc1/assets/field_bench_leaderboard_fieldloss_smooth05.json`
- `docs/m3dc1/assets/field_bench_pairwise_wins_fieldloss_smooth0.png`
- `docs/m3dc1/assets/field_bench_pairwise_wins_fieldloss_smooth05.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth0.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth05.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth05_vs_baseline.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth0_vs_baseline.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_peak4_vs_fieldloss_smooth0.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_peak4_vs_fieldloss_smooth05.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_smooth0.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_smooth05.png`
- `docs/m3dc1/assets/field_bench_relL2_hist_smooth05_vs_smooth0.png`
- `docs/m3dc1/assets/field_recon_fieldloss_smooth0.png`
- `docs/m3dc1/assets/field_recon_fieldloss_smooth05.png`
- `docs/m3dc1/assets/fieldloss_smooth05_field_recon`
- `docs/m3dc1/assets/fieldloss_smooth0_field_recon`
- `docs/m3dc1/assets/loss_fieldloss_smooth05_fno2d.png`
- `docs/m3dc1/assets/loss_fieldloss_smooth0_fno2d.png`
- `docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth05_refqc_combined.png`
- `docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth05_refqc_field.png`
- `docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth0_refqc_combined.png`
- `docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth0_refqc_field.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_best_smooth05_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_median_smooth05_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_worst_smooth05_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_best_fieldloss_smooth0_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_best_smooth0_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_median_fieldloss_smooth0_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_median_smooth0_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_worst_fieldloss_smooth0_test.png`
- `docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_worst_smooth0_test.png`
- `docs/m3dc1/assets/sparc1530_diagnosis_smooth`
- `docs/m3dc1/assets/sparc1530_smooth_ablation`
- `narrative_md`: `docs/m3dc1/SURGE_M3DC1_RESEARCH_NARRATIVE.md`
- `narrative_html`: `docs/m3dc1/SURGE_M3DC1_RESEARCH_NARRATIVE.html`
- `results_report`: `docs/m3dc1/SURGE_M3DC1_RESULTS_REPORT.md`
### field_bench_dirs
- `field_bench/with_fieldloss`
- `field_bench/with_fieldloss_geom`
- `field_bench/with_fieldloss_smooth05`
- `field_bench/with_fieldloss_smooth0`
- `field_bench/leaderboard.json`

## 11. Raw JSON dump (full measured objects)

```json
{
  "runs": {
    "qc_peak4": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_peak4",
      "run_config": {
        "target_smooth": 1.0,
        "target_floor": 6.0,
        "peak_weight": 4.0,
        "geom_channels": false,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "composite",
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120
      },
      "spectrum_image_metrics": {
        "test_r2_global": 0.9142558723688126,
        "test_pattern_r2": 0.9216772764921188,
        "train_seconds": 2758.7932510375977,
        "n_params": 75506433,
        "checkpoint": "runs/spectrum_fno48_floor6_smooth1_qc_peak4/ckpt_fno2d.pt",
        "history": "runs/spectrum_fno48_floor6_smooth1_qc_peak4/history_fno2d.jsonl"
      },
      "target_description": "max-normalized log10|dp|, floor -6dex, smooth s=1, global z-score",
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.8929877579212189,
        "r2_global_mean": 0.7371152845942747,
        "r2_pattern_median": 0.9102625250816345,
        "r2_pattern_mean": 0.8945205391290315,
        "r2_pattern_min": -0.12302136421203613,
        "r2_pattern_max": 0.9967019557952881,
        "pct_pattern_lt_0": 0.15045135406218654,
        "pct_pattern_lt_0.5": 1.053159478435306,
        "pct_pattern_lt_0.7": 2.507522567703109,
        "pct_pattern_lt_0.8": 5.2156469408224675,
        "pct_pattern_lt_0.9": 36.40922768304915,
        "pct_pattern_ge_0.9": 63.59077231695085,
        "pattern_quantiles": {
          "0.02": 0.6622752857208252,
          "0.1": 0.8589203894138336,
          "0.25": 0.888090506196022,
          "0.5": 0.9102625250816345,
          "0.75": 0.9265788793563843,
          "0.9": 0.939251857995987,
          "0.98": 0.9872579431533813
        }
      },
      "predictions_cache_bytes": 132497407,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783098314.0",
      "training_epochs_completed": 179,
      "final_train_val_r2": null
    },
    "qc": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc",
      "run_config": {
        "target_smooth": 1.0,
        "target_floor": 6.0,
        "peak_weight": 0.0,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "composite",
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120
      },
      "spectrum_image_metrics": {
        "test_r2_global": 0.919909730553627,
        "test_pattern_r2": 0.9237876906991005,
        "train_seconds": 4178.0336236953735,
        "n_params": 75506433,
        "checkpoint": "runs/spectrum_fno48_floor6_smooth1_qc/ckpt_fno2d.pt",
        "history": "runs/spectrum_fno48_floor6_smooth1_qc/history_fno2d.jsonl"
      },
      "target_description": "max-normalized log10|dp|, floor -6dex, smooth s=1, global z-score",
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.8911202847957611,
        "r2_global_mean": 0.742260753271453,
        "r2_pattern_median": 0.9079992771148682,
        "r2_pattern_mean": 0.8918471397643343,
        "r2_pattern_min": -0.3288489580154419,
        "r2_pattern_max": 0.9971767067909241,
        "pct_pattern_lt_0": 0.10030090270812438,
        "pct_pattern_lt_0.5": 0.9027081243731194,
        "pct_pattern_lt_0.7": 2.4072216649949847,
        "pct_pattern_lt_0.8": 5.516549648946841,
        "pct_pattern_lt_0.9": 40.17051153460381,
        "pct_pattern_ge_0.9": 59.82948846539619,
        "pattern_quantiles": {
          "0.02": 0.6735616803169251,
          "0.1": 0.8426670789718628,
          "0.25": 0.8841686248779297,
          "0.5": 0.9079992771148682,
          "0.75": 0.9255126863718033,
          "0.9": 0.9377642393112182,
          "0.98": 0.9863397979736328
        }
      },
      "predictions_cache_bytes": 132546930,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783029132.0",
      "training_epochs_completed": 289,
      "final_train_val_r2": null
    },
    "fieldloss_sigma1": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
      "run_config": {
        "target_smooth": 1.0,
        "target_floor": 6.0,
        "peak_weight": 4.0,
        "geom_channels": false,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "field",
        "field_loss_weight": 0.5,
        "field_loss_warmup": 20,
        "field_select_n": 64,
        "field_select_every": 5,
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120,
        "time_budget_min": 210.0
      },
      "spectrum_image_metrics": {
        "test_r2_global": 0.8907209262251854,
        "test_pattern_r2": 0.896111935377121,
        "train_seconds": 6513.894383430481,
        "n_params": 75506433,
        "checkpoint": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss/ckpt_fno2d.pt",
        "history": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss/history_fno2d.jsonl"
      },
      "target_description": "max-normalized log10|dp|, floor -6dex, smooth s=1, global z-score",
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.8880326747894287,
        "r2_global_mean": 0.7233724222558671,
        "r2_pattern_median": 0.9080575406551361,
        "r2_pattern_mean": 0.8884313643994997,
        "r2_pattern_min": -0.6442592144012451,
        "r2_pattern_max": 0.9944307208061218,
        "pct_pattern_lt_0": 0.25075225677031093,
        "pct_pattern_lt_0.5": 1.2036108324974923,
        "pct_pattern_lt_0.7": 3.159478435305918,
        "pct_pattern_lt_0.8": 6.218655967903711,
        "pct_pattern_lt_0.9": 39.26780341023069,
        "pct_pattern_ge_0.9": 60.7321965897693,
        "pattern_quantiles": {
          "0.02": 0.6186316227912902,
          "0.1": 0.8494424819946289,
          "0.25": 0.8871363401412964,
          "0.5": 0.9080575406551361,
          "0.75": 0.9240408390760422,
          "0.9": 0.9360403656959534,
          "0.98": 0.9788947260379791
        }
      },
      "predictions_cache_bytes": 132319401,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783174271.0",
      "training_epochs_completed": 400,
      "final_train_val_r2": 0.9029014408588409,
      "field_bench_test": {
        "n": 1994,
        "mean_relL2": 0.7021718992882177,
        "median_relL2": 0.6323795690155662,
        "p90_relL2": 1.0009822978852987,
        "frac_relL2_gt_1": 0.10030090270812438,
        "mean_relL2_alpha": 0.6305780952093659,
        "median_relL2_alpha": 0.6110259077737857,
        "p90_relL2_alpha": 0.8123564272177984,
        "frac_relL2_alpha_gt_1": 0.0,
        "mean_crf": 0.9068675073299465,
        "relL2_bins": {
          "<0.3": 5,
          "<0.5": 207,
          "<0.7": 1070,
          "<1.0": 512,
          ">1.0": 200
        },
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss"
      },
      "field_bench_ranking": [
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "spectrum_fno48_floor6_smooth1_qc_peak4",
        "spectrum_fno48_floor6_smooth1_qc",
        "spectrum_fno48_floor6_smooth1_qc_mhi100"
      ],
      "field_bench_win_counts": {
        "spectrum_fno48_floor6_smooth1_qc_peak4": 187,
        "spectrum_fno48_floor6_smooth1_qc": 189,
        "spectrum_fno48_floor6_smooth1_qc_mhi100": 116,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": 1502
      }
    },
    "run_B_geom": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
      "run_config": {
        "target_smooth": 1.0,
        "target_floor": 6.0,
        "peak_weight": 4.0,
        "geom_channels": true,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "field",
        "field_loss_weight": 0.5,
        "field_loss_warmup": 20,
        "field_select_n": 64,
        "field_select_every": 5,
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120,
        "time_budget_min": 210.0
      },
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.8895953893661499,
        "r2_global_mean": 0.7419552157970226,
        "r2_pattern_median": 0.9122016429901123,
        "r2_pattern_mean": 0.8979877347991603,
        "r2_pattern_min": 0.18365663290023804,
        "r2_pattern_max": 0.9964202642440796,
        "pct_pattern_lt_0": 0.0,
        "pct_pattern_lt_0.5": 0.802407221664995,
        "pct_pattern_lt_0.7": 2.3570712136409226,
        "pct_pattern_lt_0.8": 5.2657973921765295,
        "pct_pattern_lt_0.9": 38.11434302908726,
        "pct_pattern_ge_0.9": 61.88565697091274,
        "pattern_quantiles": {
          "0.02": 0.668411283493042,
          "0.1": 0.8466557025909424,
          "0.25": 0.882658064365387,
          "0.5": 0.9122016429901123,
          "0.75": 0.934535413980484,
          "0.9": 0.9523445963859558,
          "0.98": 0.9847469222545624
        }
      },
      "predictions_cache_bytes": 132565628,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783198089.0",
      "training_epochs_completed": 214,
      "final_train_val_r2": 0.903349831700325,
      "field_bench_test": {
        "n": 1994,
        "mean_relL2": 0.7194733296238851,
        "median_relL2": 0.6422822483559532,
        "p90_relL2": 1.0605141111817091,
        "frac_relL2_gt_1": 0.12236710130391174,
        "mean_relL2_alpha": 0.6346770417594475,
        "median_relL2_alpha": 0.6125661068440569,
        "p90_relL2_alpha": 0.8230579417552144,
        "frac_relL2_alpha_gt_1": 0.0,
        "mean_crf": 0.9286015220419874,
        "relL2_bins": {
          "<0.3": 11,
          "<0.5": 164,
          "<0.7": 1077,
          "<1.0": 498,
          ">1.0": 244
        },
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom"
      },
      "field_bench_ranking": [
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "spectrum_fno48_floor6_smooth1_qc_peak4"
      ],
      "field_bench_win_counts": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": 1175,
        "spectrum_fno48_floor6_smooth1_qc_peak4": 176,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom": 643
      }
    },
    "run_D_smooth05": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
      "run_config": {
        "target_smooth": 0.5,
        "target_floor": 6.0,
        "peak_weight": 4.0,
        "geom_channels": false,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "field",
        "field_loss_weight": 0.5,
        "field_loss_warmup": 20,
        "field_select_n": 64,
        "field_select_every": 5,
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120,
        "time_budget_min": 210.0
      },
      "spectrum_image_metrics": {
        "test_r2_global": 0.8928801864385605,
        "test_pattern_r2": 0.8724502772092819,
        "train_seconds": 6433.361153125763,
        "n_params": 75506433,
        "checkpoint": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05/ckpt_fno2d.pt",
        "history": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05/history_fno2d.jsonl"
      },
      "target_description": "max-normalized log10|dp|, floor -6dex, smooth s=0.5, global z-score",
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.8055414855480194,
        "r2_global_mean": 0.6829127730911928,
        "r2_pattern_median": 0.8260726630687714,
        "r2_pattern_mean": 0.8161052990940174,
        "r2_pattern_min": -0.11571109294891357,
        "r2_pattern_max": 0.9960954189300537,
        "pct_pattern_lt_0": 0.05015045135406219,
        "pct_pattern_lt_0.5": 0.9528585757271816,
        "pct_pattern_lt_0.7": 5.9679037111334,
        "pct_pattern_lt_0.8": 33.55065195586761,
        "pct_pattern_lt_0.9": 90.320962888666,
        "pct_pattern_ge_0.9": 9.679037111334003,
        "pattern_quantiles": {
          "0.02": 0.5860671281814576,
          "0.1": 0.7276606857776642,
          "0.25": 0.7819378226995468,
          "0.5": 0.8260726630687714,
          "0.75": 0.8648778796195984,
          "0.9": 0.8994053363800049,
          "0.98": 0.9803176689147949
        }
      },
      "predictions_cache_bytes": 139212457,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783205804.0",
      "training_epochs_completed": 400,
      "final_train_val_r2": 0.8744910061359406,
      "field_bench_test": {
        "n": 1994,
        "mean_relL2": 0.5273555961207053,
        "median_relL2": 0.4554233902515053,
        "p90_relL2": 0.8198276543052722,
        "frac_relL2_gt_1": 0.05115346038114343,
        "mean_relL2_alpha": 0.4799417325057257,
        "median_relL2_alpha": 0.43717325388904016,
        "p90_relL2_alpha": 0.7200428062882099,
        "frac_relL2_alpha_gt_1": 0.0,
        "mean_crf": 0.7114718263741646,
        "relL2_bins": {
          "<0.3": 120,
          "<0.5": 1031,
          "<0.7": 492,
          "<1.0": 249,
          ">1.0": 102
        },
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05"
      },
      "field_bench_ranking": [
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "spectrum_fno48_floor6_smooth1_qc_peak4"
      ],
      "field_bench_win_counts": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": 127,
        "spectrum_fno48_floor6_smooth1_qc_peak4": 65,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05": 1802
      }
    },
    "run_Dprime_smooth0": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
      "run_config": {
        "target_smooth": 0.0,
        "target_floor": 6.0,
        "peak_weight": 4.0,
        "geom_channels": false,
        "m_window": [
          -80.0,
          20.0
        ],
        "select_by": "field",
        "field_loss_weight": 0.5,
        "field_loss_warmup": 20,
        "field_select_n": 64,
        "field_select_every": 5,
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120,
        "time_budget_min": 210.0
      },
      "spectrum_image_metrics": {
        "test_r2_global": 0.8542397320270538,
        "test_pattern_r2": 0.8107420355081558,
        "train_seconds": 6302.087346315384,
        "n_params": 75506433,
        "checkpoint": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0/ckpt_fno2d.pt",
        "history": "runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0/history_fno2d.jsonl"
      },
      "target_description": "max-normalized log10|dp|, floor -6dex, global z-score",
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.7242495119571686,
        "r2_global_mean": 0.6189977496473815,
        "r2_pattern_median": 0.7501447200775146,
        "r2_pattern_mean": 0.7437261722929619,
        "r2_pattern_min": -0.7433333396911621,
        "r2_pattern_max": 0.9918063282966614,
        "pct_pattern_lt_0": 0.15045135406218654,
        "pct_pattern_lt_0.5": 1.855566700100301,
        "pct_pattern_lt_0.7": 23.169508525576727,
        "pct_pattern_lt_0.8": 79.63891675025076,
        "pct_pattern_lt_0.9": 96.79037111334003,
        "pct_pattern_ge_0.9": 3.20962888665998,
        "pattern_quantiles": {
          "0.02": 0.5025584471225738,
          "0.1": 0.6635975182056427,
          "0.25": 0.705166295170784,
          "0.5": 0.7501447200775146,
          "0.75": 0.7914221882820129,
          "0.9": 0.8263462305068969,
          "0.98": 0.973706314563751
        }
      },
      "predictions_cache_bytes": 139631376,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783203529.0",
      "training_epochs_completed": 400,
      "final_train_val_r2": 0.8346429020166397,
      "field_bench_test": {
        "n": 1994,
        "mean_relL2": 0.4905667920591792,
        "median_relL2": 0.42086257796301363,
        "p90_relL2": 0.8029328011792074,
        "frac_relL2_gt_1": 0.048144433299899696,
        "mean_relL2_alpha": 0.4484243356647723,
        "median_relL2_alpha": 0.4033087521911142,
        "p90_relL2_alpha": 0.7224523903885377,
        "frac_relL2_alpha_gt_1": 0.0,
        "mean_crf": 0.6053731761737884,
        "relL2_bins": {
          "<0.3": 433,
          "<0.5": 808,
          "<0.7": 437,
          "<1.0": 220,
          ">1.0": 96
        },
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0"
      },
      "field_bench_ranking": [
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "spectrum_fno48_floor6_smooth1_qc_peak4"
      ],
      "field_bench_win_counts": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": 125,
        "spectrum_fno48_floor6_smooth1_qc_peak4": 63,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0": 1806
      }
    },
    "qc_mhi100": {
      "run_dir": "runs/spectrum_fno48_floor6_smooth1_qc_mhi100",
      "run_config": {
        "target_smooth": 1.0,
        "target_floor": 6.0,
        "peak_weight": 0.0,
        "geom_channels": true,
        "m_window": [
          -80.0,
          100.0
        ],
        "select_by": "composite",
        "fno_modes": 48,
        "fno_hidden": 32,
        "epochs": 400,
        "patience": 120
      },
      "predictions_cache": {
        "n_test": 1994,
        "r2_global_median": 0.9131181240081787,
        "r2_global_mean": 0.7062308888138835,
        "r2_pattern_median": 0.9315246641635895,
        "r2_pattern_mean": 0.9061635072336512,
        "r2_pattern_min": -0.960747480392456,
        "r2_pattern_max": 0.9979588389396667,
        "pct_pattern_lt_0": 0.3009027081243731,
        "pct_pattern_lt_0.5": 1.6549648946840523,
        "pct_pattern_lt_0.7": 3.4102306920762286,
        "pct_pattern_lt_0.8": 5.366098294884654,
        "pct_pattern_lt_0.9": 18.35506519558676,
        "pct_pattern_ge_0.9": 81.64493480441324,
        "pattern_quantiles": {
          "0.02": 0.557352727651596,
          "0.1": 0.8664063513278961,
          "0.25": 0.9106307178735733,
          "0.5": 0.9315246641635895,
          "0.75": 0.9452269673347473,
          "0.9": 0.9546804964542389,
          "0.98": 0.9835051035881041
        }
      },
      "predictions_cache_bytes": 129344475,
      "ckpt_fno2d_exists": true,
      "ckpt_fno2d_mtime": "1783112385.0",
      "training_epochs_completed": 237,
      "final_train_val_r2": 0.8963351100683212,
      "field_bench_test": {
        "n": 1994,
        "mean_relL2": 0.9213030500813933,
        "median_relL2": 0.8253486204817133,
        "p90_relL2": 1.3504908170785903,
        "frac_relL2_gt_1": 0.2668004012036108,
        "mean_relL2_alpha": 0.7740714588436963,
        "median_relL2_alpha": 0.760799772110871,
        "p90_relL2_alpha": 0.9364750364028476,
        "frac_relL2_alpha_gt_1": 0.0,
        "mean_crf": 0.7882045562903754,
        "relL2_bins": {
          "<0.3": 0,
          "<0.5": 19,
          "<0.7": 298,
          "<1.0": 1145,
          ">1.0": 532
        },
        "model": "spectrum_fno48_floor6_smooth1_qc_mhi100"
      },
      "field_bench_ranking": [
        "spectrum_fno48_floor6_smooth1_qc_peak4",
        "spectrum_fno48_floor6_smooth1_qc",
        "spectrum_fno48_floor6_smooth1_qc_mhi100"
      ],
      "field_bench_win_counts": {
        "spectrum_fno48_floor6_smooth1_qc": 722,
        "spectrum_fno48_floor6_smooth1_qc_peak4": 1042,
        "spectrum_fno48_floor6_smooth1_qc_mhi100": 230
      }
    }
  },
  "dataset": {
    "batch_dir": "/pscratch/sd/a/asvillar/mp288/jobs/batch_16",
    "filename": "csdata_deltap_b_ver.h5",
    "n_cases_total": 9974,
    "n_train": 6983,
    "n_val": 997,
    "n_test": 1994,
    "quarantined_cases": 2,
    "grid": 128,
    "m_window_default": [
      -80.0,
      20.0
    ],
    "test_frac": 0.2,
    "val_frac": 0.1,
    "seed": 42
  },
  "pairwise": {
    "from_bench_with_fieldloss_smooth0": {
      "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_vs_spectrum_fno48_floor6_smooth1_qc_peak4": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_wins": 1691,
        "spectrum_fno48_floor6_smooth1_qc_peak4_wins": 303,
        "ties": 0
      },
      "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_vs_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_wins": 157,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_wins": 1837,
        "ties": 0
      },
      "spectrum_fno48_floor6_smooth1_qc_peak4_vs_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0": {
        "spectrum_fno48_floor6_smooth1_qc_peak4_wins": 112,
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_wins": 1882,
        "ties": 0
      }
    },
    "run_Dprime_vs_run_D_direct": {
      "smooth0_lower_relL2_wins": 1466,
      "smooth05_lower_relL2_wins": 528,
      "ties": 0,
      "mean_relL2_delta_smooth0_minus_smooth05": -0.036788822467402206,
      "median_relL2_delta": -0.03622600000000001
    }
  },
  "sparc_1530_cases": {},
  "artifact_paths": {
    "figures": [
      "docs/m3dc1/assets/field_bench_frac_gt1_sparc_families_smooth0.png",
      "docs/m3dc1/assets/field_bench_frac_gt1_sparc_families_smooth05.png",
      "docs/m3dc1/assets/field_bench_frac_gt1_sparc_fieldloss_smooth0.png",
      "docs/m3dc1/assets/field_bench_frac_gt1_sparc_fieldloss_smooth05.png",
      "docs/m3dc1/assets/field_bench_leaderboard_fieldloss_smooth0.json",
      "docs/m3dc1/assets/field_bench_leaderboard_fieldloss_smooth05.json",
      "docs/m3dc1/assets/field_bench_pairwise_wins_fieldloss_smooth0.png",
      "docs/m3dc1/assets/field_bench_pairwise_wins_fieldloss_smooth05.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth0.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth05.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth05_vs_baseline.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_fieldloss_smooth0_vs_baseline.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_peak4_vs_fieldloss_smooth0.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_peak4_vs_fieldloss_smooth05.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_smooth0.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_smooth05.png",
      "docs/m3dc1/assets/field_bench_relL2_hist_smooth05_vs_smooth0.png",
      "docs/m3dc1/assets/field_recon_fieldloss_smooth0.png",
      "docs/m3dc1/assets/field_recon_fieldloss_smooth05.png",
      "docs/m3dc1/assets/fieldloss_smooth05_field_recon",
      "docs/m3dc1/assets/fieldloss_smooth0_field_recon",
      "docs/m3dc1/assets/loss_fieldloss_smooth05_fno2d.png",
      "docs/m3dc1/assets/loss_fieldloss_smooth0_fno2d.png",
      "docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth05_refqc_combined.png",
      "docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth05_refqc_field.png",
      "docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth0_refqc_combined.png",
      "docs/m3dc1/assets/metric_reality_check_qc_peak4_fieldloss_smooth0_refqc_field.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_best_smooth05_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_median_smooth05_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05_worst_smooth05_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_best_fieldloss_smooth0_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_best_smooth0_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_median_fieldloss_smooth0_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_median_smooth0_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_worst_fieldloss_smooth0_test.png",
      "docs/m3dc1/assets/rz_case_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0_worst_smooth0_test.png",
      "docs/m3dc1/assets/sparc1530_diagnosis_smooth",
      "docs/m3dc1/assets/sparc1530_smooth_ablation"
    ],
    "narrative_md": "docs/m3dc1/SURGE_M3DC1_RESEARCH_NARRATIVE.md",
    "narrative_html": "docs/m3dc1/SURGE_M3DC1_RESEARCH_NARRATIVE.html",
    "results_report": "docs/m3dc1/SURGE_M3DC1_RESULTS_REPORT.md",
    "field_bench_dirs": [
      "field_bench/with_fieldloss",
      "field_bench/with_fieldloss_geom",
      "field_bench/with_fieldloss_smooth05",
      "field_bench/with_fieldloss_smooth0",
      "field_bench/leaderboard.json"
    ]
  },
  "per_family": {
    "sparc_1427": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1427",
        "n": 22.0,
        "mean_relL2": 0.8101868585213665,
        "median_relL2": 0.7256247897072822,
        "p90_relL2": 1.0616385517709728,
        "frac_relL2_gt_1": 0.13636363636363635,
        "mean_relL2_alpha": 0.679099200245715,
        "mean_crf": 0.9349276040125905
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1427",
        "n": 22.0,
        "mean_relL2": 0.672854720637415,
        "median_relL2": 0.5974599434363973,
        "p90_relL2": 0.8712394299554768,
        "frac_relL2_gt_1": 0.045454545454545456,
        "mean_relL2_alpha": 0.576583576739308,
        "mean_crf": 0.7486302596134476
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1427",
        "n": 22.0,
        "mean_relL2": 0.6232800583215107,
        "median_relL2": 0.5783002687742445,
        "p90_relL2": 0.8491041133146441,
        "frac_relL2_gt_1": 0.09090909090909091,
        "mean_relL2_alpha": 0.5549910317444792,
        "mean_crf": 0.6525540811903435
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1427",
        "n": 22.0,
        "mean_relL2": 0.8564749949394969,
        "median_relL2": 0.7593762171515563,
        "p90_relL2": 1.1197130689720467,
        "frac_relL2_gt_1": 0.22727272727272727,
        "mean_relL2_alpha": 0.6925461295904808,
        "mean_crf": 0.9484380927614384
      }
    },
    "sparc_1430": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1430",
        "n": 19.0,
        "mean_relL2": 0.8936800618813211,
        "median_relL2": 0.8730587198825542,
        "p90_relL2": 1.0855638036740756,
        "frac_relL2_gt_1": 0.2631578947368421,
        "mean_relL2_alpha": 0.7760253142239145,
        "mean_crf": 0.9482581435022893
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1430",
        "n": 19.0,
        "mean_relL2": 0.7449249343536607,
        "median_relL2": 0.6711164479021442,
        "p90_relL2": 1.0554434045248147,
        "frac_relL2_gt_1": 0.2631578947368421,
        "mean_relL2_alpha": 0.6660167382689549,
        "mean_crf": 0.855031434439194
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1430",
        "n": 19.0,
        "mean_relL2": 0.8441770319368297,
        "median_relL2": 0.8119023197904165,
        "p90_relL2": 1.263855688856673,
        "frac_relL2_gt_1": 0.3157894736842105,
        "mean_relL2_alpha": 0.720700675461559,
        "mean_crf": 0.8044104530591627
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1430",
        "n": 19.0,
        "mean_relL2": 0.84185421547826,
        "median_relL2": 0.7887464342825177,
        "p90_relL2": 1.1246146840398548,
        "frac_relL2_gt_1": 0.3157894736842105,
        "mean_relL2_alpha": 0.7373689047555867,
        "mean_crf": 0.9645152534460699
      }
    },
    "sparc_1530": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1530",
        "n": 20.0,
        "mean_relL2": 0.8129585625640068,
        "median_relL2": 0.7711546055914787,
        "p90_relL2": 1.08149761371246,
        "frac_relL2_gt_1": 0.25,
        "mean_relL2_alpha": 0.7519246919725975,
        "mean_crf": 0.9791497755810614
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1530",
        "n": 20.0,
        "mean_relL2": 0.6164446620651364,
        "median_relL2": 0.6139414146364266,
        "p90_relL2": 0.9087876146048055,
        "frac_relL2_gt_1": 0.05,
        "mean_relL2_alpha": 0.5730721303892091,
        "mean_crf": 0.9080078884325122
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1530",
        "n": 20.0,
        "mean_relL2": 0.6890904308239743,
        "median_relL2": 0.6743214474351134,
        "p90_relL2": 0.8860798926661946,
        "frac_relL2_gt_1": 0.05,
        "mean_relL2_alpha": 0.6450378632872183,
        "mean_crf": 0.8897151235322666
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1530",
        "n": 20.0,
        "mean_relL2": 0.6776710366368681,
        "median_relL2": 0.6345793293092659,
        "p90_relL2": 0.9646802649190895,
        "frac_relL2_gt_1": 0.05,
        "mean_relL2_alpha": 0.6090177790866041,
        "mean_crf": 0.9726182452695034
      }
    },
    "sparc_1500": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1500",
        "n": 23.0,
        "mean_relL2": 0.4900494610039547,
        "median_relL2": 0.3930065963537572,
        "p90_relL2": 0.6634146342877436,
        "frac_relL2_gt_1": 0.08695652173913043,
        "mean_relL2_alpha": 0.4452568718487521,
        "mean_crf": 0.8971257244716145
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1500",
        "n": 23.0,
        "mean_relL2": 0.41481369170312016,
        "median_relL2": 0.3513558302467291,
        "p90_relL2": 0.5463489767807773,
        "frac_relL2_gt_1": 0.0,
        "mean_relL2_alpha": 0.38187769859744236,
        "mean_crf": 0.8991233314620004
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1500",
        "n": 23.0,
        "mean_relL2": 0.41564272447209166,
        "median_relL2": 0.3595070305772828,
        "p90_relL2": 0.5140357188223766,
        "frac_relL2_gt_1": 0.08695652173913043,
        "mean_relL2_alpha": 0.3717812313765967,
        "mean_crf": 0.8735952235111679
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1500",
        "n": 23.0,
        "mean_relL2": 0.44805188077182573,
        "median_relL2": 0.39139857097834274,
        "p90_relL2": 0.5319120717037594,
        "frac_relL2_gt_1": 0.043478260869565216,
        "mean_relL2_alpha": 0.4159231441291289,
        "mean_crf": 0.9537364257350083
      }
    },
    "sparc_1524": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1524",
        "n": 23.0,
        "mean_relL2": 1.0139722580061037,
        "median_relL2": 0.8604779957207713,
        "p90_relL2": 1.7081526950827977,
        "frac_relL2_gt_1": 0.34782608695652173,
        "mean_relL2_alpha": 0.7585706932954386,
        "mean_crf": 0.9032752401017237
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1524",
        "n": 23.0,
        "mean_relL2": 0.8420161334903975,
        "median_relL2": 0.780315958462692,
        "p90_relL2": 1.2610148935637615,
        "frac_relL2_gt_1": 0.2608695652173913,
        "mean_relL2_alpha": 0.6829452231091401,
        "mean_crf": 0.7289314637475186
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1524",
        "n": 23.0,
        "mean_relL2": 0.8673662725848073,
        "median_relL2": 0.7534639549738332,
        "p90_relL2": 1.1789894844585251,
        "frac_relL2_gt_1": 0.34782608695652173,
        "mean_relL2_alpha": 0.6744730573703529,
        "mean_crf": 0.617326660146407
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1524",
        "n": 23.0,
        "mean_relL2": 1.0199521560983067,
        "median_relL2": 0.9150843319714197,
        "p90_relL2": 1.5163156169537602,
        "frac_relL2_gt_1": 0.2608695652173913,
        "mean_relL2_alpha": 0.76465878900067,
        "mean_crf": 0.9428611823832662
      }
    },
    "sparc_1300": {
      "fieldloss_sigma1": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        "family": "sparc_1300",
        "n": 21.0,
        "mean_relL2": 0.4569873114005015,
        "median_relL2": 0.4260080067286333,
        "p90_relL2": 0.6310170825038877,
        "frac_relL2_gt_1": 0.047619047619047616,
        "mean_relL2_alpha": 0.44004408512595883,
        "mean_crf": 0.8899903056498069
      },
      "run_D_smooth05": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
        "family": "sparc_1300",
        "n": 21.0,
        "mean_relL2": 0.47925336249426753,
        "median_relL2": 0.4182957385606315,
        "p90_relL2": 0.7002150505692422,
        "frac_relL2_gt_1": 0.047619047619047616,
        "mean_relL2_alpha": 0.41936386070574255,
        "mean_crf": 0.891465775697907
      },
      "run_Dprime_smooth0": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
        "family": "sparc_1300",
        "n": 21.0,
        "mean_relL2": 0.4540993876521585,
        "median_relL2": 0.42009224978996657,
        "p90_relL2": 0.6239819698170562,
        "frac_relL2_gt_1": 0.047619047619047616,
        "mean_relL2_alpha": 0.41277536518076385,
        "mean_crf": 0.8534178384276927
      },
      "run_B_geom": {
        "model": "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_geom",
        "family": "sparc_1300",
        "n": 21.0,
        "mean_relL2": 0.4472793731062833,
        "median_relL2": 0.4229681734003321,
        "p90_relL2": 0.6658212300403026,
        "frac_relL2_gt_1": 0.047619047619047616,
        "mean_relL2_alpha": 0.4189601454172794,
        "mean_crf": 0.9352689495977353
      }
    }
  },
  "sparc_1530_top_fieldloss_regressions_vs_peak4": [
    {
      "key": "run79_sparc_1530",
      "family": "sparc_1530",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.496567",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "1.032458",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.895632",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "0.061429"
    },
    {
      "key": "run43_sparc_1530",
      "family": "sparc_1530",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.816470",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "1.298781",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.432810",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "-0.180970"
    },
    {
      "key": "run99_sparc_1530",
      "family": "sparc_1530",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.655249",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "1.078824",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.404467",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "-0.241258"
    },
    {
      "key": "run31_sparc_1530",
      "family": "sparc_1530",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.631002",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "1.037485",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.960259",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "0.117388"
    },
    {
      "key": "run83_sparc_1530",
      "family": "sparc_1530",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.793379",
      "relL2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "1.105557",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4": "0.884501",
      "patR2_spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss": "-0.227872"
    }
  ]
}
```