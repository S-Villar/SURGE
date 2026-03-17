# XGC A_parallel Input Variable Structure

## OLCF Hackathon Dataset (this run)

- **201 input features** (`input_0` … `input_200`)
- **2 outputs** (`output_0`, `output_1`; `output_1` = A_parallel)
- Source: `data_nprev5_set1_data.npy`, `data_nprev5_set1_target.npy`

## Timesteps and 201 = 196 + 5

The filename `data_nprev5` indicates **5 previous timesteps** (nprev=5). The 201 inputs decompose as:

| Block | Columns | Description |
|-------|---------|-------------|
| **196** | 0–195 | 14 variables × 14 slots |
| **5** | 196–200 | Extra columns, tied to nprev=5 |

**Where are the timesteps?** The source notebook does not fully document the layout. Two plausible readings:

1. **14 vars × 14 spatial points** (single time): 196 = 14 flux-surface points × 14 vars. The 5 extra (196–200) would then carry temporal info from the 5 timesteps.

2. **14 vars × 14 timesteps** (all vars at multiple times): If all variables are given at multiple timesteps, the 14 could be timesteps: 196 = 14 vars × 14 time points. The 5 extra would be additional temporal/global terms.

The prior doc (`XGC_APARALLEL_DATA.md`) assumed interpretation 1. If you observe that all vars appear at multiple timesteps, interpretation 2 may apply. To resolve this, you’d need to inspect the data-generation code or the original XGC reader that produced the `.npy` files.

## Alternative 56-variable derivation (from preprocessing slide)

A different preprocessing yields **56 inputs** from 14 physical variables:

1. **14 raw variables** from `xgc.3d.*.bp`:  
   `aparh`, `apars`, `dBphi`, `dBpsi`, `dBtheta`, `ejpar`, `ijpar`, `dpot`, `epara`, `epara2`, `epsi`, `etheta`, `eden`, `iden`

2. For each variable V:
   - Extract n=0 mode: `V_n=0`
   - Extract non-n=0: `V_n!=0`
   - Flux-surface average: `F_n=0`, `F_n!=0`
   - Normalize: `W_n=0 = V_n=0 / F_n=0`, `W_n!=0 = V_n!=0 / F_n!=0`

3. **4 values per variable**: `W_n=0`, `W_n!=0`, `log(F_n=0)`, `log(F_n!=0)`  
   → 14 × 4 = **56 inputs**

## Relation 201 vs 56

The 201 inputs in the OLCF hackathon data likely correspond to a different expansion (e.g. 14 vars × multiple spatial points). The 56-variable scheme is a reduced representation. SHAP analysis on the 201 inputs shows which of those 201 features matter most for accuracy.
