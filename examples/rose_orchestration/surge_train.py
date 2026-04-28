# -*- coding: utf-8 -*-
"""Run a SURGE workflow (in-process ROSE tasks, subprocess demos, Example 3 MLP search)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_surge_on_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root


_WORKFLOW_FILES: dict[str, str] = {
    "rf": "workflow_rf.yaml",
    "mlp": "workflow_mlp.yaml",
    "m3dc1_rf": "workflow_m3dc1_rf.yaml",
    "m3dc1_mlp": "workflow_m3dc1_mlp.yaml",
}


def run_one_surge(
    workflow: str,
    iteration: int,
    *,
    verbose: bool = True,
) -> dict:
    ex_dir = Path(__file__).resolve().parent
    os.chdir(ex_dir)

    _ensure_surge_on_path()
    from dataset_utils import write_json_atomic  # noqa: WPS433
    import yaml  # noqa: WPS433
    import surge  # noqa: F401, WPS433
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    wf_key = workflow
    if wf_key not in _WORKFLOW_FILES:
        raise KeyError(f"Unknown workflow {workflow!r}; expected one of {sorted(_WORKFLOW_FILES)}")
    spec_path = ex_dir / _WORKFLOW_FILES[wf_key]
    with spec_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    run_tag = f"rose_{wf_key}_iter_{iteration}"
    payload["run_tag"] = run_tag
    payload["overwrite_existing_run"] = True

    if verbose:
        print("=" * 72, flush=True)
        print(f"[SURGE] Loading spec {spec_path.name}", flush=True)
        print(f"[SURGE] run_tag={run_tag}  cwd={ex_dir}", flush=True)
        print("=" * 72, flush=True)

    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec)

    model0 = summary["models"][0]
    val = model0["metrics"]["val"]
    rmse = float(val["rmse"])
    mse = rmse**2
    r2 = float(val.get("r2", float("nan")))
    out = {
        "run_tag": run_tag,
        "workflow": wf_key,
        "iteration": iteration,
        "val_rmse": rmse,
        "val_mse": mse,
        "val_r2": r2,
        "artifacts_root": summary["artifacts"]["root"],
    }
    w = ex_dir / "workspace"
    w.mkdir(parents=True, exist_ok=True)
    metrics_path = w / "last_surge_metrics.json"
    write_json_atomic(metrics_path, out)

    if verbose:
        print(f"[SURGE] Wrote metrics for ROSE: {metrics_path}", flush=True)
        print(f"[SURGE] val_rmse={rmse:.6f}  val_mse={mse:.6f}  val_r2={r2:.6f}", flush=True)

    return out


def run_mlp_surge_random_arch(
    iteration: int,
    hidden_layer_sizes: list[int],
    *,
    dataset: str = "synthetic",
    sklearn_mlp_max_iter: int = 800,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    ex_dir = Path(__file__).resolve().parent
    os.chdir(ex_dir)
    _ensure_surge_on_path()
    from dataset_utils import write_json_atomic  # noqa: WPS433
    import yaml  # noqa: WPS433
    import surge  # noqa: F401, WPS433
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    template = "m3dc1_mlp" if dataset == "m3dc1" else "mlp"
    spec_path = ex_dir / _WORKFLOW_FILES[template]
    with spec_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    arch = list(hidden_layer_sizes)
    htag = "_".join(str(x) for x in arch)
    run_tag = f"rose_mlp_rand_i{iteration}_{htag}"
    payload["run_tag"] = run_tag
    payload["overwrite_existing_run"] = True
    payload["models"][0]["params"]["hidden_layer_sizes"] = arch
    payload["models"][0]["params"]["max_iter"] = int(sklearn_mlp_max_iter)
    payload["models"][0]["params"]["random_state"] = int(random_state)

    if verbose:
        print("=" * 72, flush=True)
        print(f"[SURGE] MLP spec from {spec_path.name}", flush=True)
        print(f"[SURGE] hidden_layer_sizes={arch}  dataset={dataset}  run_tag={run_tag}", flush=True)
        print("=" * 72, flush=True)

    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec)
    model0 = summary["models"][0]
    val = model0["metrics"]["val"]
    rmse = float(val["rmse"])
    mse = rmse**2
    r2 = float(val.get("r2", float("nan")))

    out = {
        "run_tag": run_tag,
        "workflow": template,
        "iteration": iteration,
        "val_rmse": rmse,
        "val_mse": mse,
        "val_r2": r2,
        "hidden_layer_sizes": arch,
        "sklearn_mlp_max_iter": int(sklearn_mlp_max_iter),
        "artifacts_root": summary["artifacts"]["root"],
    }
    w = ex_dir / "workspace"
    w.mkdir(parents=True, exist_ok=True)
    write_json_atomic(w / "last_surge_metrics.json", out)

    if verbose:
        print(f"[SURGE] val_rmse={rmse:.6f}  val_r2={r2:.6f}", flush=True)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train SURGE surrogate for one ROSE iteration.")
    ap.add_argument("--workflow", choices=tuple(sorted(_WORKFLOW_FILES.keys())), default="rf")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = run_one_surge(args.workflow, args.iteration, verbose=args.verbose)
    print(out["val_mse"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
