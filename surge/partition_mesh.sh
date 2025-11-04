#!/bin/bash
# Mesh partitioning script for M3DC1 simulations
# Creates part.smb from original mesh file using split_smb with 96 partitions

#SBATCH --partition=debug
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=partition_mesh
#SBATCH -o partition_mesh.log.%j

set -euo pipefail

# Get batch directory from argument or use current directory
BATCH_DIR="${1:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# Setup environment
export M3DC1_DIR=${M3DC1_DIR:-${M3DC1_PYTHON_PATH}/..}
export PATH=$M3DC1_DIR/unstructured/_perlmutter_cpu/bin:$PATH
export M3DC1_MPIRUN=${M3DC1_MPIRUN:-srun}

# Source mesh directory (from batch_0)
SOURCE_MESH_DIR="${SPARC_DATASETS:-/global/cfs/projectdirs/mp288/asvillar/proj/mlsurrogate/datasets/SPARC}/batch_0/run1/mesh"

# Target mesh directory (in batch_X)
MESH_DIR="${BATCH_DIR}/mesh"

echo "============================================================"
echo "Mesh Partitioning Job"
echo "============================================================"
echo "Batch directory: $BATCH_DIR"
echo "Source mesh: $SOURCE_MESH_DIR"
echo "Target mesh: $MESH_DIR"
echo ""

# Create mesh directory
mkdir -p "$MESH_DIR"

# Copy all part*.smb partition files if they exist
if [ -d "$SOURCE_MESH_DIR" ]; then
    if [ -n "$(ls -A "$SOURCE_MESH_DIR"/part*.smb 2>/dev/null)" ]; then
        echo "Copying existing partition files..."
        cp "$SOURCE_MESH_DIR"/part*.smb "$MESH_DIR/"
        echo "Copied $(ls -1 "$MESH_DIR"/part*.smb 2>/dev/null | wc -l) partition files"
    fi
fi

# Check if part.smb already exists
if [ -f "$MESH_DIR/part.smb" ]; then
    echo "part.smb already exists in $MESH_DIR"
    echo "Skipping mesh partitioning."
    exit 0
fi

# Check if part.smb exists in source
if [ -f "$SOURCE_MESH_DIR/part.smb" ]; then
    echo "Copying part.smb from source..."
    cp "$SOURCE_MESH_DIR/part.smb" "$MESH_DIR/"
    echo "Copied part.smb successfully"
    exit 0
fi

# Find original mesh file (not a partition)
ORIG_SMB=$(find "$SOURCE_MESH_DIR" -name "*.smb" ! -name "part*.smb" | head -1)

if [ -z "$ORIG_SMB" ] || [ ! -f "$ORIG_SMB" ]; then
    echo "ERROR: No original mesh file found in $SOURCE_MESH_DIR" >&2
    echo "Expected a .smb file (not part*.smb) to create partitioned mesh" >&2
    exit 1
fi

echo "Found original mesh file: $ORIG_SMB"

# Copy original mesh to mesh directory
cd "$MESH_DIR"
cp "$ORIG_SMB" .
ORIG_BASENAME=$(basename "$ORIG_SMB")

# Mesh partitioning parameters
PARTS=96
NODES=8
NTASKS_PER_NODE=12

echo ""
echo "============================================================"
echo "Running mesh partitioning"
echo "============================================================"
echo "Original mesh: $ORIG_BASENAME"
echo "Output: part.smb"
echo "Partitions: $PARTS"
echo "Nodes: $NODES"
echo "Tasks per node: $NTASKS_PER_NODE"
echo "Total tasks: $((NODES * NTASKS_PER_NODE))"
echo ""
echo "Command: $M3DC1_MPIRUN -N $NODES --ntasks-per-node=$NTASKS_PER_NODE split_smb $ORIG_BASENAME part.smb $PARTS"
echo ""

# Run mesh partitioning
$M3DC1_MPIRUN -N $NODES --ntasks-per-node=$NTASKS_PER_NODE split_smb "$ORIG_BASENAME" part.smb $PARTS

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Mesh partitioning completed successfully!"
    echo "============================================================"
    echo "Created: $MESH_DIR/part.smb"
    
    # Clean up original mesh copy
    rm -f "$ORIG_BASENAME"
    echo "Cleaned up temporary files"
else
    echo ""
    echo "ERROR: Mesh partitioning failed" >&2
    exit 1
fi
