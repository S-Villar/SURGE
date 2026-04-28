#!/usr/bin/env bash
# Run ROSE+SURGE examples 1–3 with --dataset m3dc1 (requires PKL). Use from this directory
# after activating your env and setting PYTHONPATH to the SURGE root (see M3DC1_EXAMPLES.md).
set -euo pipefail
_EX="$(cd "$(dirname "$0")" && pwd)"
# repo root: examples/rose_orchestration -> ../..
SURGE_ROOT="${SURGE_ROOT:-$(cd "$_EX/../.." && pwd)}"
PKL="$SURGE_ROOT/data/datasets/M3DC1/sparc-m3dc1-D1.pkl"

if [[ ! -f "$PKL" ]]; then
  echo "Missing M3DC1 pickle: $PKL" >&2
  echo "Copy sparc-m3dc1-D1.pkl into data/datasets/M3DC1/ (see M3DC1_EXAMPLES.md)." >&2
  exit 1
fi

export PYTHONPATH="${SURGE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$_EX"

echo "== Example 1 (m3dc1, max-iter 2) ==" >&2
python example_01_rose_inprocess_verbose.py --dataset m3dc1 --max-iter 2 --quiet

echo "== Example 2 (m3dc1, max-iter 2) ==" >&2
python example_02_rose_subprocess_shell.py --dataset m3dc1 --max-iter 2 --quiet

echo "== Example 3 (m3dc1, max-iter 4, r2 >= 0.85) ==" >&2
python example_03_rose_mlp_random_r2_stop.py --dataset m3dc1 --max-iter 4 --r2-threshold 0.85 --quiet

echo "Done. M3DC1 smokes completed." >&2
