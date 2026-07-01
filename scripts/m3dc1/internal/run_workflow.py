#!/usr/bin/env python3
"""Run a SURGE surrogate workflow spec (YAML).

On the model-bench line, `surge.cli` only dispatches to the benchmark runner, so
this thin wrapper drives `run_surrogate_workflow` directly. It preserves the
workflow's shared train/val/test split (including case-grouped `group_columns`)
and all artifacts (splits.json, metrics.json, predictions, ...).

Usage:
    python scripts/m3dc1/internal/run_workflow.py <spec.yaml>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    spec_path = sys.argv[1]
    payload = yaml.safe_load(Path(spec_path).read_text())
    spec = SurrogateWorkflowSpec.from_dict(payload)
    run_surrogate_workflow(spec, invocation={"spec_path": spec_path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
