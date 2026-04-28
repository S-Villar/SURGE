#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/run_demo_interactive.sh" demos/demo_01_better_surrogate_selection.sh "$@"
