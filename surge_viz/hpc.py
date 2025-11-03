"""
HPC integration for SURGE visualizer (optional).

Provides async SSH and SLURM job submission for Frontier/Perlmutter.
This is a stub implementation that can be extended.
"""

from pathlib import Path
from typing import Dict, Optional

try:
    import asyncssh
    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False


class HPCClient:
    """
    Client for HPC cluster interaction (Frontier/Perlmutter).
    
    This is a stub implementation. Extend with actual SSH/SLURM functionality.
    """
    
    def __init__(self, host: str, user: str, key_path: Optional[Path] = None):
        """
        Initialize HPC client.
        
        Parameters
        ----------
        host : str
            HPC hostname.
        user : str
            Username.
        key_path : Path, optional
            Path to SSH key.
        """
        self.host = host
        self.user = user
        self.key_path = key_path
        self.connected = False
    
    async def connect(self):
        """Connect to HPC cluster."""
        if not ASYNCSSH_AVAILABLE:
            raise ImportError("asyncssh not available. Install with: pip install asyncssh")
        
        # Stub implementation
        # TODO: Implement actual SSH connection
        self.connected = True
        return self
    
    async def submit_slurm(
        self,
        script: str,
        workdir: str,
        env: Optional[Dict] = None
    ) -> str:
        """
        Submit a SLURM job.
        
        Parameters
        ----------
        script : str
            SLURM script content.
        workdir : str
            Working directory on cluster.
        env : dict, optional
            Environment variables.
        
        Returns
        -------
        str
            Job ID.
        """
        # Stub implementation
        # TODO: Implement actual SLURM submission
        return "mock_job_12345"
    
    async def job_status(self, job_id: str) -> Dict:
        """
        Get job status.
        
        Parameters
        ----------
        job_id : str
            SLURM job ID.
        
        Returns
        -------
        dict
            Job status with keys: state, runtime, logs_tail
        """
        # Stub implementation
        # TODO: Implement actual status check
        return {
            'state': 'RUNNING',
            'runtime': 0,
            'logs_tail': 'No logs available',
        }


# SLURM script templates
FRONTIER_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=surge_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=normal

# Load modules
module load python
module load conda

# Activate environment
conda activate surge-viz

# Run training
python train_surge.py "$@"
"""

PERLMUTTER_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=surge_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --constraint=cpu

# Load modules
module load python
module load conda

# Activate environment
conda activate surge-viz

# Run training
python train_surge.py "$@"
"""

