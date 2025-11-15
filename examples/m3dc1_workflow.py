"""CLI wrapper for running the M3D-C1 surrogate workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to load workflow specs.") from exc

import surge  # noqa: F401 - ensures package initialization registers adapters
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SURGE M3D-C1 workflow demo.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/m3dc1_demo.yaml"),
        help="Path to the workflow spec YAML file.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run tag override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.spec.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    spec = SurrogateWorkflowSpec.from_dict(payload)
    if args.run_tag:
        spec.run_tag = args.run_tag
    if args.output_dir:
        spec.output_dir = str(args.output_dir)
    summary = run_surrogate_workflow(spec)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()





