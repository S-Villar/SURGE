#!/usr/bin/env python3
"""
XGC Workflow Orchestrator – runs steps sequentially, launching each only after the previous completes.

Polls for workflow completion (workflow_summary.json + models present) before starting the next step.

Usage:
  python scripts/xgc_workflow_orchestrator.py
  python scripts/xgc_workflow_orchestrator.py --python /path/to/python

Environment:
  Set SURGE_PYTHON or pass --python for the interpreter to use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "/global/cfs/projectdirs/m499/olcf_ai_hackathon_2025"


def run_cmd(cmd: list[str], desc: str, timeout: int | None = None) -> bool:
    """Run command; return True on success."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 60)
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            timeout=timeout,
        )
        ok = result.returncode == 0
        if ok:
            print(f"  OK: {desc}\n")
        else:
            print(f"  FAILED (exit {result.returncode}): {desc}\n")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {desc}\n")
        return False
    except Exception as e:
        print(f"  ERROR: {e}\n")
        return False


def run_ready(run_dir: Path, min_models: int = 1, require_summary: bool = True) -> bool:
    """True if run_dir has at least min_models and (optionally) workflow_summary.json."""
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return False
    model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pt"))
    if len(model_files) < min_models:
        return False
    if require_summary and not (run_dir / "workflow_summary.json").exists():
        return False
    return True


def wait_for_run(run_dir: Path, min_models: int = 1, poll_interval: int = 30, timeout: int | None = None) -> bool:
    """
    Wait until run_dir has workflow_summary.json and at least min_models in models/.
    Returns True when ready, False on timeout.
    """
    run_dir = Path(run_dir)
    start = time.time()
    while True:
        if run_ready(run_dir, min_models):
            return True
        if timeout and (time.time() - start) > timeout:
            return False
        models_dir = run_dir / "models"
        n = len(list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pt"))) if models_dir.exists() else 0
        summary = (run_dir / "workflow_summary.json").exists()
        print(f"  Waiting for {run_dir.name}... (models={n}/{min_models}, summary={summary})")
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="XGC workflow orchestrator – sequential execution with completion checks")
    parser.add_argument("--python", type=str, default=None, help="Python interpreter (default: sys.executable)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between completion checks")
    parser.add_argument("--skip-model1", action="store_true", help="Skip Model1 (assume already done)")
    parser.add_argument("--skip-model12", action="store_true", help="Skip Model12")
    parser.add_argument("--skip-models", action="store_true", help="Skip ModelS")
    parser.add_argument("--quick", action="store_true", help="Use quick configs (small sample, no HPO)")
    parser.add_argument("--model1-already-running", action="store_true",
        help="Model1 was started elsewhere; poll until it completes, then continue")
    args = parser.parse_args()

    py = args.python or sys.executable

    steps_ok = True

    # Step 1: Model1 (61 cols)
    run_dir = PROJECT_ROOT / "runs" / "xgc_model1_61cols"
    if args.model1_already_running:
        print("\nModel1 already running elsewhere. Polling until completion...")
        if not wait_for_run(run_dir, min_models=1, poll_interval=args.poll_interval):
            print("Model1 did not complete. Exiting.")
            return 1
        print("Model1 completed. Continuing.\n")
    elif not args.skip_model1:
        config = "configs/xgc_aparallel_set1_61cols_quick.yaml" if args.quick else "configs/xgc_aparallel_set1_61cols.yaml"
        ok = run_cmd(
            [py, "-m", "surge.cli", "run", config, "--run-tag", "xgc_model1_61cols"],
            "1. Train Model1 (61 cols) on set1",
        )
        if not ok:
            print("Model1 failed. Stopping.")
            return 1
        steps_ok = steps_ok and ok
    else:
        if not run_ready(run_dir, min_models=1):
            print(f"Model1 run dir missing or incomplete (need workflow_summary.json + models): {run_dir}")
            return 1

    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_model1_61cols", "--datastreamset-eval"],
        "2a. Eval Model1 on set1 datastreamsets",
    )
    steps_ok = steps_ok and ok

    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_model1_61cols", "--datastreamset-eval", "--datastreamset-eval-set", "set2_beta0p5"],
        "2b. Eval Model1 on set2 datastreamsets",
    )
    steps_ok = steps_ok and ok

    # Step 3: Fine-tune Model1 → Model12
    if not args.skip_model12:
        ok = run_cmd(
            [py, "-m", "surge.cli", "run", "configs/xgc_model12_finetune.yaml"],
            "3. Fine-tune Model1 with set2 → Model12",
        )
        if not ok:
            print("Model12 failed. Stopping.")
            return 1

    # Step 4: Eval Model12 on set2 and set1
    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_model12_finetune", "--datastreamset-eval"],
        "4a. Eval Model12 on set2 datastreamsets",
    )
    steps_ok = steps_ok and ok

    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_model12_finetune", "--datastreamset-eval", "--datastreamset-eval-set", "set1"],
        "4b. Eval Model12 on set1 datastreamsets",
    )
    steps_ok = steps_ok and ok

    # Step 5: Loss-curve continuation
    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_model12_finetune", "--loss-curve-continuation"],
        "5. Loss-curve continuation viz",
    )
    steps_ok = steps_ok and ok

    # Step 6: Train ModelS (SHAP top 20)
    if not args.skip_models:
        config = "configs/xgc_aparallel_set1_shap20_quick.yaml" if args.quick else "configs/xgc_aparallel_set1_shap20.yaml"
        ok = run_cmd(
            [py, "-m", "surge.cli", "run", config, "--run-tag", "xgc_modelS_shap20"],
            "6. Train ModelS (SHAP top 20) on set1",
        )
        if not ok:
            print("ModelS failed.")

    # Step 7: Eval ModelS
    ok = run_cmd(
        [py, "-m", "surge.cli", "viz", "runs/xgc_modelS_shap20", "--datastreamset-eval"],
        "7. Eval ModelS on set1 datastreamsets",
    )
    steps_ok = steps_ok and ok

    # Step 7b: Fine-tune ModelS with set2
    if not args.skip_models:
        ok = run_cmd(
            [py, "-m", "surge.cli", "run", "configs/xgc_modelS_finetune.yaml"],
            "7b. Fine-tune ModelS with set2 → ModelS finetune",
        )
        if not ok:
            print("ModelS finetune failed.")
        steps_ok = steps_ok and ok

    # Step 7c: Eval ModelS finetune on set2 and set1 (only if finetune completed)
    models_finetune_dir = PROJECT_ROOT / "runs" / "xgc_modelS_finetune"
    if run_ready(models_finetune_dir, min_models=1):
        ok = run_cmd(
            [py, "-m", "surge.cli", "viz", "runs/xgc_modelS_finetune", "--datastreamset-eval"],
            "7c. Eval ModelS finetune on set2 datastreamsets",
        )
        steps_ok = steps_ok and ok

        ok = run_cmd(
            [py, "-m", "surge.cli", "viz", "runs/xgc_modelS_finetune", "--datastreamset-eval", "--datastreamset-eval-set", "set1"],
            "7d. Eval ModelS finetune on set1 datastreamsets",
        )
        steps_ok = steps_ok and ok
    else:
        print("  Skipping 7c/7d: ModelS finetune run not found or incomplete.")

    # Step 8: Animation
    ok = run_cmd(
        [py, "scripts/xgc_predictions_animation.py", "--run-dir", "runs/xgc_aparallel_set1_v3", "--data-dir", DATA_DIR, "--format", "gif"],
        "8. XGC predictions animation",
    )
    steps_ok = steps_ok and ok

    # Step 9: Export ONNX (Model1 and Model12)
    ok = run_cmd(
        [py, "scripts/xgc_export_onnx.py", "--run-dir", "runs/xgc_model1_61cols"],
        "9a. Export Model1 to ONNX",
    )
    steps_ok = steps_ok and ok

    ok = run_cmd(
        [py, "scripts/xgc_export_onnx.py", "--run-dir", "runs/xgc_model12_finetune", "--model", "xgc_mlp_aparallel"],
        "9b. Export Model12 to ONNX",
    )
    steps_ok = steps_ok and ok

    print("\n" + "=" * 60)
    print("  Workflow orchestrator finished.")
    print("=" * 60)
    return 0 if steps_ok else 1


if __name__ == "__main__":
    sys.exit(main())
