"""
M3DC1 HDF5 Dataset Loader

This module provides utilities for loading M3DC1 simulation data from HDF5 files
and converting it to pandas DataFrame format ready for ML training.

The module handles the sdata.h5 structure with groups:
- /equilibrium/sparc_*/ - equilibrium data from geqdsk files
- /mesh/ - mesh information (R, Z coordinates, nelements)
- /run_*/input/ - input parameters (ntor, pscale, batemanscale, runid, eqid)
- /run_*/output/ - output data (gamma, eignemode)

Example usage:
    >>> import sys
    >>> from pathlib import Path
    >>> sys.path.insert(0, str(Path(__file__).parent))
    >>> from loader import load_m3dc1_hdf5, convert_to_dataframe
    >>> df = load_m3dc1_hdf5('sdata.h5')
    >>> # Or use the converter directly
    >>> df = convert_to_dataframe('sdata.h5')
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def load_m3dc1_hdf5(
    file_path: Union[str, Path],
    include_equilibrium: bool = True,
    include_mesh: bool = False,
    include_profiles: bool = False,
) -> Dict[str, Any]:
    """
    Load M3DC1 HDF5 dataset and return structured data dictionary.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the sdata.h5 file
    include_equilibrium : bool, default=True
        Whether to include equilibrium data
    include_mesh : bool, default=False
        Whether to include mesh data (usually constant across runs)
    include_profiles : bool, default=False
        Whether to include profile data (q_profile, p_profile) - requires analysis.pro
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'runs': List of run data dictionaries
        - 'equilibrium': Dictionary of equilibrium data (if include_equilibrium=True)
        - 'mesh': Dictionary of mesh data (if include_mesh=True)
        
    Raises
    ------
    ImportError
        If h5py is not available
    FileNotFoundError
        If file does not exist
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 support. Install with: pip install h5py"
        )
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"M3DC1 HDF5 file not found: {file_path}")
    
    print(f"📂 Loading M3DC1 HDF5 dataset from: {file_path}")
    
    data = {
        'runs': [],
        'equilibrium': {} if include_equilibrium else None,
        'mesh': {} if include_mesh else None,
    }
    
    with h5py.File(file_path, 'r') as f:
        # List root groups
        root_keys = list(f.keys())
        print(f"📋 Root groups: {root_keys}")
        
        # Load equilibrium data
        if include_equilibrium and 'equilibrium' in root_keys:
            print("\n🔍 Loading equilibrium data...")
            eq_group = f['equilibrium']
            for sparc_name in eq_group.keys():
                sparc_group = eq_group[sparc_name]
                eq_data = {}
                for key in sparc_group.keys():
                    dataset = sparc_group[key]
                    if dataset.shape == ():  # Scalar
                        eq_data[key] = dataset[()]
                    else:  # Array
                        eq_data[key] = dataset[:]
                data['equilibrium'][sparc_name] = eq_data
            print(f"   ✅ Loaded {len(data['equilibrium'])} equilibrium cases")
        
        # Load mesh data
        if include_mesh and 'mesh' in root_keys:
            print("\n🔍 Loading mesh data...")
            mesh_group = f['mesh']
            for key in mesh_group.keys():
                dataset = mesh_group[key]
                if dataset.shape == ():  # Scalar
                    data['mesh'][key] = dataset[()]
                else:  # Array
                    data['mesh'][key] = dataset[:]
            print(f"   ✅ Loaded mesh data: {list(data['mesh'].keys())}")
        
        # Load run data
        print("\n🔍 Loading run data...")
        run_keys = [k for k in root_keys if k.startswith('run_')]
        run_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        for run_key in run_keys:
            run_group = f[run_key]
            run_data = {'run_id': run_key}
            
            # Load input data
            if 'input' in run_group.keys():
                input_group = run_group['input']
                run_data['input'] = {}
                for key in input_group.keys():
                    dataset = input_group[key]
                    if dataset.shape == ():  # Scalar
                        run_data['input'][key] = dataset[()]
                    else:  # Array
                        run_data['input'][key] = dataset[:]
            
            # Load output data
            if 'output' in run_group.keys():
                output_group = run_group['output']
                run_data['output'] = {}
                for key in output_group.keys():
                    dataset = output_group[key]
                    if dataset.shape == ():  # Scalar
                        run_data['output'][key] = dataset[()]
                    else:  # Array
                        run_data['output'][key] = dataset[:]
            
            data['runs'].append(run_data)
        
        print(f"   ✅ Loaded {len(data['runs'])} runs")
    
    return data


def convert_to_dataframe(
    file_path: Union[str, Path],
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None,
    include_equilibrium_features: bool = True,
    include_mesh_features: bool = False,
) -> pd.DataFrame:
    """
    Convert M3DC1 HDF5 dataset to pandas DataFrame ready for ML training.
    
    This function extracts input parameters (from run_*/input/) and output data
    (from run_*/output/) and creates a flat DataFrame where each row represents
    one simulation run.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the sdata.h5 file
    input_features : list of str, optional
        Specific input features to include. If None, includes all available inputs.
        Common inputs: ['ntor', 'pscale', 'batemanscale', 'runid', 'eqid']
    output_features : list of str, optional
        Specific output features to include. If None, includes all available outputs.
        Common outputs: ['gamma', 'eignemode']
    include_equilibrium_features : bool, default=True
        Whether to include equilibrium-derived features (a, R0, kappa, delta, etc.)
    include_mesh_features : bool, default=False
        Whether to include mesh features (usually constant, not useful for ML)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per simulation run. Columns include:
        - Input features: ntor, pscale, batemanscale, runid, eqid, and optionally
          equilibrium features (a, R0, kappa, delta, etc.)
        - Output features: gamma, eignemode (flattened if multi-dimensional)
        
    Examples
    --------
    >>> df = convert_to_dataframe('sdata.h5')
    >>> print(df.head())
    >>> print(f"Input columns: {[c for c in df.columns if c.startswith('input_')]}")
    >>> print(f"Output columns: {[c for c in df.columns if c.startswith('output_')]}")
    """
    # Load the HDF5 data
    data = load_m3dc1_hdf5(
        file_path,
        include_equilibrium=include_equilibrium_features,
        include_mesh=include_mesh_features,
    )
    
    rows = []
    
    for run_data in data['runs']:
        row = {}
        
        # Extract run ID
        run_id = run_data.get('run_id', '')
        row['run_id'] = run_id
        
        # Extract input features
        if 'input' in run_data:
            input_data = run_data['input']
            
            # Use specified features or all available
            input_keys = input_features if input_features else list(input_data.keys())
            
            for key in input_keys:
                if key in input_data:
                    value = input_data[key]
                    # Handle arrays by flattening or taking first element
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:  # Scalar array
                            row[f'input_{key}'] = value.item()
                        elif value.ndim == 1 and len(value) == 1:  # Single-element array
                            row[f'input_{key}'] = value[0]
                        else:
                            # For multi-element arrays, create separate columns
                            for i, val in enumerate(value):
                                row[f'input_{key}_{i}'] = val
                    else:
                        row[f'input_{key}'] = value
        
        # Extract equilibrium features if available
        if include_equilibrium_features and data['equilibrium']:
            eq_id = input_data.get('eqid', None) if 'input' in run_data else None
            if eq_id is not None:
                # Find matching equilibrium data
                for sparc_name, eq_data in data['equilibrium'].items():
                    if eq_data.get('id', None) == eq_id:
                        # Add key equilibrium features (shape parameters)
                        for key in ['a', 'R0', 'kappa', 'delta', 'simag', 'sibry', 'current']:
                            if key in eq_data:
                                value = eq_data[key]
                                if isinstance(value, np.ndarray) and value.ndim == 0:
                                    row[f'eq_{key}'] = value.item()
                                elif isinstance(value, np.ndarray) and value.size == 1:
                                    row[f'eq_{key}'] = value[0]
                                else:
                                    row[f'eq_{key}'] = value
                        
                        # Extract profile parameters from qpsi and pres profiles
                        # q0: safety factor at axis (first value of qpsi)
                        if 'qpsi' in eq_data:
                            qpsi = eq_data['qpsi']
                            if isinstance(qpsi, np.ndarray) and len(qpsi) > 0:
                                row['q0'] = qpsi[0] if qpsi.ndim == 1 else qpsi.flatten()[0]
                                # q95: safety factor at 95% flux (approximately index 95% of length)
                                if len(qpsi) > 1:
                                    idx_95 = int(0.95 * (len(qpsi) - 1))
                                    row['q95'] = qpsi[idx_95] if qpsi.ndim == 1 else qpsi.flatten()[idx_95]
                        
                        # p0: pressure at axis (first value of pres)
                        if 'pres' in eq_data:
                            pres = eq_data['pres']
                            if isinstance(pres, np.ndarray) and len(pres) > 0:
                                row['p0'] = pres[0] if pres.ndim == 1 else pres.flatten()[0]
                        
                        break
        
        # Extract output features
        if 'output' in run_data:
            output_data = run_data['output']
            
            # Use specified features or all available
            output_keys = output_features if output_features else list(output_data.keys())
            
            for key in output_keys:
                if key in output_data:
                    value = output_data[key]
                    # Handle arrays by flattening
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:  # Scalar array
                            row[f'output_{key}'] = value.item()
                        elif value.ndim == 1:
                            if len(value) == 1:
                                row[f'output_{key}'] = value[0]
                            else:
                                # Flatten 1D arrays into separate columns
                                for i, val in enumerate(value):
                                    row[f'output_{key}_{i}'] = val
                        else:
                            # Flatten multi-dimensional arrays
                            flat_value = value.flatten()
                            for i, val in enumerate(flat_value):
                                row[f'output_{key}_{i}'] = val
                    else:
                        row[f'output_{key}'] = value
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    print(f"\n✅ Converted to DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Print column summary
    input_cols = [c for c in df.columns if c.startswith('input_')]
    output_cols = [c for c in df.columns if c.startswith('output_')]
    eq_cols = [c for c in df.columns if c.startswith('eq_')]
    
    print(f"   📥 Input columns: {len(input_cols)}")
    print(f"   📤 Output columns: {len(output_cols)}")
    if eq_cols:
        print(f"   🔄 Equilibrium columns: {len(eq_cols)}")
    
    return df


def read_m3dc1_hdf5_structure(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and print the structure of an M3DC1 HDF5 file without loading all data.
    
    Useful for exploring the file structure before loading.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the sdata.h5 file
        
    Returns
    -------
    dict
        Dictionary describing the file structure
    """
    if not H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 support. Install with: pip install h5py"
        )
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"M3DC1 HDF5 file not found: {file_path}")
    
    structure = {
        'root_groups': [],
        'equilibrium': {},
        'mesh': {},
        'runs': {},
    }
    
    with h5py.File(file_path, 'r') as f:
        root_keys = list(f.keys())
        structure['root_groups'] = root_keys
        
        print("=" * 70)
        print("📂 M3DC1 HDF5 File Structure")
        print("=" * 70)
        print(f"\nRoot groups: {root_keys}\n")
        
        # Explore equilibrium group
        if 'equilibrium' in root_keys:
            print("─" * 70)
            print("EQUILIBRIUM GROUP")
            print("─" * 70)
            eq_group = f['equilibrium']
            eq_keys = list(eq_group.keys())
            print(f"Equilibrium cases: {len(eq_keys)}")
            
            if eq_keys:
                # Show first case structure
                first_case = eq_keys[0]
                print(f"\nExample case: {first_case}")
                case_group = eq_group[first_case]
                case_keys = list(case_group.keys())
                structure['equilibrium'][first_case] = {
                    'keys': case_keys,
                    'shapes': {k: case_group[k].shape for k in case_keys[:10]}
                }
                print(f"  Keys: {case_keys[:10]}{'...' if len(case_keys) > 10 else ''}")
                for key in case_keys[:5]:
                    dataset = case_group[key]
                    shape_str = f"shape={dataset.shape}" if dataset.shape else "scalar"
                    dtype_str = f"dtype={dataset.dtype}"
                    print(f"    {key}: {shape_str}, {dtype_str}")
        
        # Explore mesh group
        if 'mesh' in root_keys:
            print("\n" + "─" * 70)
            print("MESH GROUP")
            print("─" * 70)
            mesh_group = f['mesh']
            mesh_keys = list(mesh_group.keys())
            structure['mesh'] = {k: {'shape': mesh_group[k].shape, 'dtype': str(mesh_group[k].dtype)} 
                                 for k in mesh_keys}
            print(f"Mesh keys: {mesh_keys}")
            for key in mesh_keys:
                dataset = mesh_group[key]
                shape_str = f"shape={dataset.shape}" if dataset.shape else "scalar"
                print(f"  {key}: {shape_str}")
        
        # Explore run groups
        run_keys = [k for k in root_keys if k.startswith('run_')]
        run_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        if run_keys:
            print("\n" + "─" * 70)
            print("RUN GROUPS")
            print("─" * 70)
            print(f"Number of runs: {len(run_keys)}")
            
            # Show first run structure
            first_run = run_keys[0]
            print(f"\nExample run: {first_run}")
            run_group = f[first_run]
            run_subgroups = list(run_group.keys())
            print(f"  Subgroups: {run_subgroups}")
            
            structure['runs'][first_run] = {}
            
            if 'input' in run_subgroups:
                input_group = run_group['input']
                input_keys = list(input_group.keys())
                structure['runs'][first_run]['input'] = {
                    'keys': input_keys,
                    'shapes': {k: input_group[k].shape for k in input_keys}
                }
                print(f"    Input keys: {input_keys}")
            
            if 'output' in run_subgroups:
                output_group = run_group['output']
                output_keys = list(output_group.keys())
                structure['runs'][first_run]['output'] = {
                    'keys': output_keys,
                    'shapes': {k: output_group[k].shape for k in output_keys}
                }
                print(f"    Output keys: {output_keys}")
        
        print("\n" + "=" * 70)
    
    return structure


__all__ = [
    'load_m3dc1_hdf5',
    'convert_to_dataframe',
    'read_m3dc1_hdf5_structure',
    'H5PY_AVAILABLE',
]

