# Spectrum-image surrogate — training runbook

How to (re)train the whole-spectrum FNO2D surrogate for the normalized
`|δp̂|(m, ψ_N)` eigenmode magnitude, resume from checkpoints, and monitor
progress. All commands are run from the SURGE repo root
(`/global/homes/a/asvillar/src/SURGE`).

---

## 0. The idea (normalization)

Eigenmode overall amplitude is arbitrary (M3D-C1 normalizes each solution
differently; across cases the peak `|δp|` ranges from ~1e-7 to ~1e4). Only the
*shape* / relative structure carries information. So each case's spectrum is
divided by its own maximum (**peak → 1**) *before* any log, and that normalized
field is the ground truth. This is `--target-norm max`. Adding it moved FNO2D
from a **~0.60 → ~0.84 val R²** plateau on the full 9,976-case dataset.

Target space:
- `--target-space log10` → target is `log10(mag_normalized + eps)` (**best: ~0.84**)
- `--target-space raw`   → target is the normalized magnitude itself (~0.79)

---

## 1. Easiest: one command (interactive GPU node)

```bash
# fresh full-dataset log10 run (default)
bash scripts/m3dc1/internal/spectrum_train.sh

# fresh raw-magnitude run
bash scripts/m3dc1/internal/spectrum_train.sh raw

# CONTINUE a previous run from its last checkpoint (see §3)
bash scripts/m3dc1/internal/spectrum_train.sh log10 resume

# override total epochs (3rd arg)
bash scripts/m3dc1/internal/spectrum_train.sh log10 fresh 400
```

This grabs an interactive A100 node (`salloc`), sets up the conda env, and runs
the training **on the compute node** via `srun`. Output lands in
`runs/spectrum_image_full_maxnorm_<space>/`.

> ⚠️ Why `srun` matters: `salloc <command>` runs the command on the **login
> node**, whose GPU is shared and usually full (→ instant CUDA OOM). The
> training must be launched with `srun` so it runs on the allocated GPU. The
> launcher handles this for you.

---

## 2. Unattended: submit to the batch queue (survives logout)

```bash
sbatch --time=04:00:00 -J si_log10 \
  --export=ALL,SI_NCASES=0,SI_OUT=runs/spectrum_image_full_maxnorm_log10,\
SI_MODELS=fno2d,SI_EPOCHS=300,SI_TNORM=max,SI_TSPACE=log10,SI_PATIENCE=40,SI_CKPT_EVERY=25 \
  scripts/m3dc1/internal/train_spectrum_image.slurm
```

Env knobs (all optional): `SI_NCASES` (0 = whole dataset), `SI_OUT`,
`SI_EPOCHS`, `SI_TNORM` (none|max), `SI_TSPACE` (log10|raw), `SI_PATIENCE`,
`SI_CKPT_EVERY`, `SI_RESUME` (checkpoint path), `SI_MODELS`, `SI_BATCH`,
`SI_GRID`. In an `sbatch` job the script already runs on the compute node, so no
`srun` wrapper is needed. (Note: the `regular` GPU queue can be deep — thousands
of pending jobs — so interactive §1 is usually faster for iteration.)

---

## 3. Resuming / continuing training from a checkpoint

Every epoch now writes:
- `ckpt_fno2d.pt`        — **best** val-loss checkpoint (weights + optimizer)
- `ckpt_fno2d_last.pt`   — **latest** epoch (weights + optimizer) → resume point
- `ckpt_fno2d_ep<N>.pt`  — periodic snapshots if `--ckpt-every N` is set

Each checkpoint stores `state_dict`, `optimizer` (Adam moments), `epoch`,
`val_loss`, `val_r2`, `best_val`. Resuming restores all of that and **appends**
to `history_fno2d.jsonl` (it does not truncate).

```bash
# via the launcher (auto-picks ckpt_fno2d_last.pt, else ckpt_fno2d.pt)
bash scripts/m3dc1/internal/spectrum_train.sh log10 resume 400

# or directly (inside a GPU allocation)
python scripts/m3dc1/internal/train_spectrum_image.py \
  --batch-dir /pscratch/sd/a/asvillar/mp288/jobs/batch_16 \
  --filename csdata_deltap_b_ver.h5 --n-cases 0 --grid 128 --m-lo -80 --m-hi 20 \
  --models fno2d --target-norm max --target-space log10 \
  --epochs 400 --patience 40 --batch-size 16 \
  --resume runs/spectrum_image_full_maxnorm_log10/ckpt_fno2d_last.pt \
  --out runs/spectrum_image_full_maxnorm_log10
```

> The split is reproducible (`--seed 42` default), so a resume trains on exactly
> the same train/val/test partition. Checkpoints saved by older runs contain
> weights but no optimizer state — resuming still works, it just restarts Adam's
> moments (a harmless brief transient).

---

## 4. Why did the earlier runs "early stop"?

They were **not killed** — they hit **early stopping**. The trainer keeps the
best val-loss epoch and stops after `--patience` epochs with no val-loss
improvement (default 25). The full log10 run's best val R² was at epoch 49; by
epoch 74 there had been 25 epochs with no improvement, so it stopped and kept the
epoch-49 weights. This is the intended behaviour (avoids overfitting), but if you
want to push further:

- raise `--patience` (e.g. 40–60) to tolerate longer plateaus,
- raise `--epochs`,
- or `--resume` the `..._last.pt` checkpoint and keep going (§3).

To disable early stopping entirely, pass `--patience 0`.

---

## 5. Monitoring a run (live, read-only)

```bash
# stats report + loss/R2 curves PNG (works while the job is running)
python -m surge.check_training --run runs/spectrum_image_full_maxnorm_log10

# quick peek at raw history without importing surge
tail -f runs/spectrum_image_full_maxnorm_log10/history_fno2d.jsonl
```

`check_training` writes `check_training_loss.png` in the run folder and prints
best/latest val R², the improving/plateau trend, and early-stop status.

---

## 6. Where things live

| Artifact | Path |
|---|---|
| Trainer | `scripts/m3dc1/internal/train_spectrum_image.py` |
| Interactive launcher | `scripts/m3dc1/internal/spectrum_train.sh` |
| Batch script | `scripts/m3dc1/internal/train_spectrum_image.slurm` |
| Monitor | `surge/check_training.py` (`python -m surge.check_training`) |
| Dataset (raw) | `/pscratch/sd/a/asvillar/mp288/jobs/batch_16/**/csdata_deltap_b_ver.h5` |
| Runs (log10 / raw) | `runs/spectrum_image_full_maxnorm_log10/`, `..._raw/` |
