#!/bin/bash
# Wrapper script to run test_eigenmode_extraction.py with proper environment

# Load fusion-io module if available
if command -v module &> /dev/null; then
    # Try to load fusion-io (adjust module name as needed for your system)
    module load fusion-io 2>/dev/null || \
    module load fusion-io/stable 2>/dev/null || \
    echo "Note: Could not load fusion-io module (may not be needed)"
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null
    conda activate surge 2>/dev/null || \
    conda activate surge-devel 2>/dev/null || \
    echo "Note: Could not activate conda environment (using system Python)"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script
python3 "$SCRIPT_DIR/test_eigenmode_extraction.py" "$@"

