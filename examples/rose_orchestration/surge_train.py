# -*- coding: utf-8 -*-
"""Run a SURGE workflow (in-process ROSE tasks, subprocess demos, Example 3 MLP search)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _ensure_surge_on_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root


_WORKFLOW_FILES: dict[str, str] = {
    "rf": "workflow_rf.yaml",
    "mlp": "workflow_mlp.yaml",
    "gpr": "workflow_gpr.yaml",
    "gpflow_gpr": "workflow_gpflow_gpr.yaml",
    "m3dc1_rf": "workflow_m3dc1_rf.yaml",
    "m3dc1_mlp": "workflow_m3dc1_mlp.yaml",
    "m3dc1_gpr": "workflow_m3dc1_gpr.yaml",
    "m3dc1_gpflow_gpr": "workflow_m3dc1_gpflow_gpr.yaml",
}


def _load_workflow_payload(ex_dir: Path, workflow_key: str) -> dict:
    import yaml  # noqa: WPS433

    spec_path = ex_dir / _WORKFLOW_FILES[workflow_key]
    with spec_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    dataset_path = Path(payload["dataset_path"])
    if not dataset_path.is_absolute():
        payload["dataset_path"] = str((ex_dir / dataset_path).resolve())
    metadata_path = payload.get("metadata_path")
    if metadata_path:
        mp = Path(metadata_path)
        if not mp.is_absolute():
            payload["metadata_path"] = str((ex_dir / mp).resolve())
    output_dir = Path(payload.get("output_dir", "."))
    if not output_dir.is_absolute():
        payload["output_dir"] = str(ex_dir.resolve())
    return payload


def _override_payload_paths(
    payload: dict,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    if dataset_path is not None:
        payload["dataset_path"] = str(Path(dataset_path).resolve())
    if output_dir is not None:
        payload["output_dir"] = str(Path(output_dir).resolve())


def _metrics_payload(*, iteration: int, workflow: str, run_tag: str, summary: dict, extra: dict | None = None) -> dict:
    model0 = summary["models"][0]
    val = model0["metrics"]["val"]
    rmse = float(val["rmse"])
    mse = rmse**2
    r2 = float(val.get("r2", float("nan")))
    payload = {
        "run_tag": run_tag,
        "workflow": workflow,
        "iteration": iteration,
        "val_rmse": rmse,
        "val_mse": mse,
        "val_r2": r2,
        "artifacts_root": summary["artifacts"]["root"],
        "splits": dict(summary.get("splits", {})),
        "split_fractions": {
            "train": 1.0 - float(summary["dataset"].get("test_fraction", 0.0)) - float(summary["dataset"].get("val_fraction", 0.0)),
            "val": float(summary["dataset"].get("val_fraction", 0.0)),
            "test": float(summary["dataset"].get("test_fraction", 0.0)),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _append_surge_run_log(summary: dict, log_path: str | Path | None) -> None:
    if log_path is None:
        return
    run_log = summary.get("artifacts", {}).get("run_log")
    if not run_log:
        return
    src = Path(run_log)
    if not src.is_file():
        return
    dst = Path(log_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as out:
        out.write("\n")
        out.write("=" * 80 + "\n")
        out.write(f"SURGE run log: {src}\n")
        out.write("=" * 80 + "\n")
        out.write(src.read_text(encoding="utf-8", errors="replace"))
        out.write("\n")


def run_one_surge(
    workflow: str,
    iteration: int,
    *,
    namespace: str | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    run_tag_prefix: str = "rose",
    verbose: bool = True,
) -> dict:
    ex_dir = Path(__file__).resolve().parent

    _ensure_surge_on_path()
    from dataset_utils import write_iteration_state  # noqa: WPS433
    import surge  # noqa: F401, WPS433
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    wf_key = workflow
    if wf_key not in _WORKFLOW_FILES:
        raise KeyError(f"Unknown workflow {workflow!r}; expected one of {sorted(_WORKFLOW_FILES)}")
    payload = _load_workflow_payload(ex_dir, wf_key)
    _override_payload_paths(payload, dataset_path=dataset_path, output_dir=output_dir)
    spec_path = ex_dir / _WORKFLOW_FILES[wf_key]

    run_tag = f"{run_tag_prefix}_{wf_key}_iter_{iteration}"
    payload["run_tag"] = run_tag
    payload["overwrite_existing_run"] = True

    if verbose:
        print("=" * 72, flush=True)
        print(f"[SURGE] Loading spec {spec_path.name}", flush=True)
        print(f"[SURGE] run_tag={run_tag}  cwd={ex_dir}", flush=True)
        print("=" * 72, flush=True)

    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec, console_output=log_path is None)
    _append_surge_run_log(summary, log_path)
    out = _metrics_payload(iteration=iteration, workflow=wf_key, run_tag=run_tag, summary=summary)
    metrics_path = write_iteration_state(iteration, "surge_metrics", out, namespace=namespace)

    if verbose:
        print(f"[SURGE] Wrote metrics for ROSE: {metrics_path}", flush=True)
        print(
            "[SURGE] splits "
            f"train={out['splits'].get('train', '?')} "
            f"val={out['splits'].get('val', '?')} "
            f"test={out['splits'].get('test', '?')}",
            flush=True,
        )
        print(
            f"[SURGE] val_rmse={out['val_rmse']:.6f}  "
            f"val_mse={out['val_mse']:.6f}  val_r2={out['val_r2']:.6f}",
            flush=True,
        )

    return out


def run_mlp_surge_random_arch(
    iteration: int,
    hidden_layer_sizes: list[int],
    *,
    dataset: str = "synthetic",
    sklearn_mlp_max_iter: int = 800,
    random_state: int = 42,
    namespace: str | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    run_tag_prefix: str = "rose_mlp_rand",
    verbose: bool = True,
) -> dict:
    ex_dir = Path(__file__).resolve().parent
    _ensure_surge_on_path()
    from dataset_utils import write_iteration_state  # noqa: WPS433
    import surge  # noqa: F401, WPS433
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    template = "m3dc1_mlp" if dataset == "m3dc1" else "mlp"
    spec_path = ex_dir / _WORKFLOW_FILES[template]
    payload = _load_workflow_payload(ex_dir, template)
    _override_payload_paths(payload, dataset_path=dataset_path, output_dir=output_dir)

    arch = list(hidden_layer_sizes)
    htag = "_".join(str(x) for x in arch)
    run_tag = f"{run_tag_prefix}_i{iteration}_{htag}"
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
    summary = run_surrogate_workflow(spec, console_output=log_path is None)
    _append_surge_run_log(summary, log_path)
    out = _metrics_payload(
        iteration=iteration,
        workflow=template,
        run_tag=run_tag,
        summary=summary,
        extra={
            "hidden_layer_sizes": arch,
            "sklearn_mlp_max_iter": int(sklearn_mlp_max_iter),
        },
    )
    write_iteration_state(iteration, "surge_metrics", out, namespace=namespace)

    if verbose:
        print(
            "[SURGE] splits "
            f"train={out['splits'].get('train', '?')} "
            f"val={out['splits'].get('val', '?')} "
            f"test={out['splits'].get('test', '?')}",
            flush=True,
        )
        print(f"[SURGE] val_rmse={out['val_rmse']:.6f}  val_r2={out['val_r2']:.6f}", flush=True)

    return out


def _flatten_numeric(values: object) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, list):
        out: list[float] = []
        for item in values:
            out.extend(_flatten_numeric(item))
        return out
    return []


def run_mlp_ensemble_uq(
    iteration: int,
    *,
    dataset: str = "m3dc1",
    namespace: str | None = None,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_path: str | Path | None = None,
    hidden_layer_sizes: list[int] | None = None,
    ensemble_n: int = 5,
    max_iter: int = 300,
    random_state: int = 42,
    run_tag_prefix: str = "rose_uq",
    verbose: bool = True,
) -> dict:
    ex_dir = Path(__file__).resolve().parent
    _ensure_surge_on_path()
    from dataset_utils import write_iteration_state  # noqa: WPS433
    import surge  # noqa: F401, WPS433
    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    template = "m3dc1_mlp" if dataset == "m3dc1" else "mlp"
    spec_path = ex_dir / _WORKFLOW_FILES[template]
    payload = _load_workflow_payload(ex_dir, template)
    _override_payload_paths(payload, dataset_path=dataset_path, output_dir=output_dir)

    arch = list(hidden_layer_sizes or [32, 32])
    run_tag = f"{run_tag_prefix}_iter_{iteration}_ens{int(ensemble_n)}"
    payload["run_tag"] = run_tag
    payload["overwrite_existing_run"] = True
    payload["models"][0]["name"] = "mlp_ensemble_uq"
    payload["models"][0]["request_uncertainty"] = True
    payload["models"][0]["params"]["hidden_layer_sizes"] = arch
    payload["models"][0]["params"]["ensemble_n"] = int(ensemble_n)
    payload["models"][0]["params"]["max_iter"] = int(max_iter)
    payload["models"][0]["params"]["random_state"] = int(random_state)

    if verbose:
        print("=" * 72, flush=True)
        print(f"[SURGE-UQ] MLP ensemble spec from {spec_path.name}", flush=True)
        print(
            f"[SURGE-UQ] arch={arch} ensemble_n={ensemble_n} dataset={dataset} run_tag={run_tag}",
            flush=True,
        )
        print("=" * 72, flush=True)

    spec = SurrogateWorkflowSpec.from_dict(payload)
    summary = run_surrogate_workflow(spec, console_output=log_path is None)
    _append_surge_run_log(summary, log_path)
    out = _metrics_payload(
        iteration=iteration,
        workflow=template,
        run_tag=run_tag,
        summary=summary,
        extra={
            "hidden_layer_sizes": arch,
            "ensemble_n": int(ensemble_n),
            "sklearn_mlp_max_iter": int(max_iter),
        },
    )

    model0 = summary["models"][0]
    uq_path = model0.get("artifacts", {}).get("predictions", {}).get("val_uq")
    if uq_path:
        blob = json.loads(Path(uq_path).read_text(encoding="utf-8"))
        std_values = _flatten_numeric(blob.get("std"))
        if std_values:
            out["uq_val_std_mean"] = sum(std_values) / len(std_values)
            out["uq_val_std_max"] = max(std_values)
            out["uq_val_std_cells"] = len(std_values)
        out["uq_artifact"] = uq_path

    write_iteration_state(iteration, "surge_metrics", out, namespace=namespace)

    if verbose:
        print(
            "[SURGE-UQ] splits "
            f"train={out['splits'].get('train', '?')} "
            f"val={out['splits'].get('val', '?')} "
            f"test={out['splits'].get('test', '?')}",
            flush=True,
        )
        print(
            f"[SURGE-UQ] val_r2={out['val_r2']:.6f} "
            f"mean_val_std={out.get('uq_val_std_mean', float('nan')):.6f}",
            flush=True,
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train SURGE surrogate for one ROSE iteration.")
    ap.add_argument("--workflow", choices=tuple(sorted(_WORKFLOW_FILES.keys())), default="rf")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--run-tag-prefix", default="rose")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = run_one_surge(
        args.workflow,
        args.iteration,
        namespace=args.namespace,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        log_path=args.log_file,
        run_tag_prefix=args.run_tag_prefix,
        verbose=args.verbose,
    )
    print(out["val_mse"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
