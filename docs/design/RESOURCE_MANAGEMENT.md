# Resource management & accelerated training — design plan (R15–R18)

Status: proposed (2026-07-26). Continues the numbering of
`ARCHITECTURE_RECOMMENDATIONS.md` (R1–R14).

Motivation from real use: the TheWell Gray-Scott study trained FNO-2D and
U-Net for ~45 min *each* on CPU while this workstation's Apple-Silicon GPU
sat idle — all 19 torch backends resolve `device or ("cuda" if
torch.cuda.is_available() else "cpu")` and never consider MPS. Model
training within a workflow is also strictly sequential, and HPO trials run
one at a time.

## R15 — Unified device resolution (small, do first)

One helper, used by every torch backend:

```python
# surge/utils/device.py
def resolve_device(requested: str | None = None) -> torch.device:
    """auto -> SURGE_DEVICE env -> cuda:N -> mps -> cpu."""
```

- Order: explicit arg > `SURGE_DEVICE` env > `cuda` (if available) >
  `mps` (if available) > `cpu`.
- Spec-level override: `device: mps` at workflow top level, inherited by
  every model unless the model block overrides it.
- MPS caveats to encode in the helper + docs:
  - float64 is unsupported → backends already use float32 (OK).
  - `torch.fft.rfft2` (FNO path) has incomplete MPS coverage on some torch
    versions → set `PYTORCH_ENABLE_MPS_FALLBACK=1` when device is mps, and
    benchmark FNO on mps vs cpu before making it the default for spectral
    models.
  - Reproducibility: mps kernels are not bit-deterministic; run artifacts
    must record the resolved device (extend `env.txt`).
- Migration: replace the 19 copies of the cuda-or-cpu one-liner with the
  helper; keyword `device="auto"` becomes the documented default.

## R16 — Parallel model training within a workflow (medium)

A workflow spec with N models currently trains them sequentially.

- `parallel_models: <int>` in the spec → process pool
  (`spawn` context; torch + fork is unsafe) training independent models
  concurrently, each with `torch.set_num_threads(cpu_count //
  n_parallel)` so they don't thrash.
- Artifacts are already per-model (predictions, cards, logs) → no write
  conflicts; workflow_summary merges at the end.
- HPO: Optuna supports concurrent trials against a shared storage
  (sqlite journal); `hpo.n_jobs: <int>` per model block. Per-epoch
  training logs already carry the trial index.

## R17 — Resource caps & simple scheduling (medium)

- The benchmark registry already records
  `resource_expectation: {device, memory_tier}` per benchmark — enforce
  it: a run declares its budget (`max_parallel`, `max_memory_tier`) and
  the scheduler holds jobs that would exceed it.
- Round-robin GPU allocator for multi-GPU boxes: trial/model i →
  `cuda:{i % n_gpus}`; expose `CUDA_VISIBLE_DEVICES` slicing for HPC
  batch systems.
- `surge run --dry-run` prints the planned placement (model → device,
  threads) before training.

## R18 — Multi-GPU single-model training (large, last)

- DDP via `torchrun` for the large operator models (FNO/U-Net on ≥128²
  grids); a `distributed: {backend: nccl, nproc: N}` spec block that
  wraps the backend's fit loop.
- Only worth it after R15–R17: most SURGE models are minutes-scale; the
  wins come from parallelism *across* models/trials first.

## Sequencing

1. R15 (one file + 19 mechanical call-site edits + smoke test per device)
2. R16 model-parallel + HPO n_jobs
3. R17 caps/scheduler
4. R18 DDP

R15 alone likely cuts the Gray-Scott study wall-time severalfold on this
workstation once FNO/U-Net train on MPS.
