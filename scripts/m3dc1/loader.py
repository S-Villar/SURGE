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
    *,
    format: str = "auto",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convert M3DC1 HDF5 dataset to pandas DataFrame ready for ML training.
    
    Supports both sdata.h5 and sdata_complex_v2.h5 (delta p spectra). Use
    format="auto" to auto-detect from filename, or pass format="sdata_complex_v2"
    explicitly for delta p spectra.
    
    Parameters
    ----------
    file_path : str or Path
        Path to sdata.h5 or sdata_complex_v2.h5
    input_features : list of str, optional
        Specific input features to include.
    output_features : list of str, optional
        For sdata format: output features (e.g. gamma, eignemode). Ignored for
        sdata_complex_v2 (delta p spectra are always the output).
    include_equilibrium_features : bool, default=True
        Whether to include equilibrium-derived features.
    include_mesh_features : bool, default=False
        Whether to include mesh features (sdata only).
    format : str, default="auto"
        "auto" (detect from filename), "sdata", or "sdata_complex_v2"
    **kwargs
        For sdata_complex_v2: delta_p_key, use_magnitude (see
        convert_sdata_complex_v2_to_dataframe).
        
    Returns
    -------
    pd.DataFrame
    """
    path = Path(file_path)
    fmt = (format or "auto").lower()
    if fmt == "auto":
        name = path.name.lower()
        if "complex" in name and "v2" in name:
            fmt = "sdata_complex_v2"
        else:
            fmt = "sdata"
    
    if fmt == "sdata_complex_v2":
        return convert_sdata_complex_v2_to_dataframe(
            path,
            input_features=input_features,
            include_equilibrium_features=include_equilibrium_features,
            **kwargs,
        )
    
    return _convert_sdata_to_dataframe_impl(
        path,
        input_features=input_features,
        output_features=output_features,
        include_equilibrium_features=include_equilibrium_features,
        include_mesh_features=include_mesh_features,
    )


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


# ---------------------------------------------------------------------------
# sdata_complex_v2.h5 support (delta p spectra)
# ---------------------------------------------------------------------------

# Common key names for delta p spectra in sdata_complex_v2.h5 output groups
DELTA_P_SPECTRUM_KEYS = ('p', 'delta_p', 'p_spectrum', 'eigenmode_p', 'eignemode')


def read_sdata_complex_v2_structure(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect the structure of an sdata_complex_v2.h5 file.
    
    Use this to discover the exact keys and shapes before loading.
    Run from the SURGE repo root:
        python -c "
        import sys; sys.path.insert(0, 'scripts/m3dc1')
        from loader import read_sdata_complex_v2_structure
        read_sdata_complex_v2_structure('path/to/sdata_complex_v2.h5')
        "
    
    Parameters
    ----------
    file_path : str or Path
        Path to sdata_complex_v2.h5
        
    Returns
    -------
    dict
        Structure summary with root groups, run layout, and output keys/shapes
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    structure: Dict[str, Any] = {'root_groups': [], 'runs': {}}
    
    with h5py.File(file_path, 'r') as f:
        structure['root_groups'] = list(f.keys())
        run_keys = [k for k in f.keys() if re.match(r'run_\d+', k)]
        run_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        
        if run_keys:
            first_run = f[run_keys[0]]
            structure['runs']['example'] = run_keys[0]
            if 'input' in first_run:
                structure['runs']['input_keys'] = list(first_run['input'].keys())
                structure['runs']['input_shapes'] = {
                    k: first_run['input'][k].shape for k in first_run['input'].keys()
                }
            if 'output' in first_run:
                out_grp = first_run['output']
                structure['runs']['output_keys'] = list(out_grp.keys())
                structure['runs']['output_shapes'] = {
                    k: (out_grp[k].shape, str(out_grp[k].dtype))
                    for k in out_grp.keys()
                }
    
    # Print summary for convenience
    print("=" * 60)
    print("sdata_complex_v2.h5 structure")
    print("=" * 60)
    print(f"Root groups: {structure['root_groups']}")
    if structure.get('runs'):
        print(f"Example run: {structure['runs'].get('example', 'N/A')}")
        print(f"Input keys: {structure['runs'].get('input_keys', [])}")
        print(f"Output keys: {structure['runs'].get('output_keys', [])}")
        if 'output_shapes' in structure['runs']:
            for k, (shp, dt) in structure['runs']['output_shapes'].items():
                print(f"  {k}: shape={shp}, dtype={dt}")
    print("=" * 60)
    return structure


def convert_sdata_complex_v2_to_dataframe(
    file_path: Union[str, Path],
    *,
    delta_p_key: Optional[str] = None,
    use_magnitude: bool = True,
    input_features: Optional[List[str]] = None,
    include_equilibrium_features: bool = True,
) -> pd.DataFrame:
    """
    Convert sdata_complex_v2.h5 to a pandas DataFrame for surrogate training.
    
    Extracts equilibrium/input parameters as features and delta p spectra as
    the output target. Complex spectra are converted to magnitude by default.
    
    Parameters
    ----------
    file_path : str or Path
        Path to sdata_complex_v2.h5
    delta_p_key : str, optional
        Key name for delta p spectra in run_*/output. If None, auto-detect from
        DELTA_P_SPECTRUM_KEYS (p, delta_p, p_spectrum, eigenmode_p, eignemode).
    use_magnitude : bool, default=True
        If True and spectrum is complex, use |z| for each component.
        If False, use real and imag as separate columns (output_p_0_real, etc.).
    input_features : list of str, optional
        Input columns to include. If None, uses equilibrium + input params.
    include_equilibrium_features : bool, default=True
        Whether to include eq_R0, eq_a, eq_kappa, eq_delta, q0, q95, qmin, p0.
        
    Returns
    -------
    pd.DataFrame
        One row per run. Columns: input_*, eq_*, output_p_0, output_p_1, ...
        
    Examples
    --------
    >>> df = convert_sdata_complex_v2_to_dataframe('sdata_complex_v2.h5')
    >>> # Or with explicit key if auto-detect fails:
    >>> df = convert_sdata_complex_v2_to_dataframe('sdata_complex_v2.h5', delta_p_key='p')
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"sdata_complex_v2 file not found: {file_path}")
    
    # Load base structure (same as sdata.h5)
    data = load_m3dc1_hdf5(
        file_path,
        include_equilibrium=include_equilibrium_features,
        include_mesh=False,
    )
    
    rows = []
    for run_data in data['runs']:
        row = {}
        row['run_id'] = run_data.get('run_id', '')
        input_data = run_data.get('input', {})
        
        # Input features
        input_keys = input_features or list(input_data.keys())
        for key in input_keys:
            if key in input_data:
                val = input_data[key]
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        row[f'input_{key}'] = val.item()
                    elif val.size == 1:
                        row[f'input_{key}'] = val.flat[0]
                    else:
                        for i, v in enumerate(val.flat):
                            row[f'input_{key}_{i}'] = v
                else:
                    row[f'input_{key}'] = val
        
        # Equilibrium features
        if include_equilibrium_features and data['equilibrium']:
            eq_id = input_data.get('eqid', None)
            for sparc_name, eq_data in data['equilibrium'].items():
                if eq_data.get('id', None) == eq_id:
                    for key in ['a', 'R0', 'kappa', 'delta', 'simag', 'sibry', 'current']:
                        if key in eq_data:
                            v = eq_data[key]
                            if isinstance(v, np.ndarray) and v.ndim == 0:
                                row[f'eq_{key}'] = v.item()
                            elif isinstance(v, np.ndarray) and v.size == 1:
                                row[f'eq_{key}'] = v.flat[0]
                            else:
                                row[f'eq_{key}'] = v
                    if 'qpsi' in eq_data:
                        qpsi = np.asarray(eq_data['qpsi']).flatten()
                        if len(qpsi) > 0:
                            row['q0'] = float(qpsi[0])
                            if len(qpsi) > 1:
                                idx95 = int(0.95 * (len(qpsi) - 1))
                                row['q95'] = float(qpsi[idx95])
                            row['qmin'] = float(np.min(qpsi))
                    if 'pres' in eq_data:
                        pres = np.asarray(eq_data['pres']).flatten()
                        if len(pres) > 0:
                            row['p0'] = float(pres[0])
                    break
        
        # Delta p spectra from output
        output_data = run_data.get('output', {})
        spectrum = None
        if delta_p_key and delta_p_key in output_data:
            spectrum = output_data[delta_p_key]
        else:
            for key in DELTA_P_SPECTRUM_KEYS:
                if key in output_data:
                    spectrum = output_data[key]
                    break
        
        if spectrum is not None:
            arr = np.asarray(spectrum)
            flat = arr.flatten()
            if np.iscomplexobj(flat) and use_magnitude:
                flat = np.abs(flat)
            for i, v in enumerate(flat):
                row[f'output_p_{i}'] = float(v)
        else:
            # Log for first run only
            if len(rows) == 0:
                print(
                    f"⚠ No delta p spectrum found in output. Keys: {list(output_data.keys())}. "
                    f"Try delta_p_key= with one of these, or run read_sdata_complex_v2_structure() to inspect."
                )
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    n_out = len([c for c in df.columns if c.startswith('output_p_')])
    print(f"\n✅ sdata_complex_v2 → DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"   📤 Delta p spectrum: {n_out} components (output_p_0 .. output_p_{n_out-1})")
    return df


def _convert_sdata_to_dataframe_impl(
    file_path: Path,
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None,
    include_equilibrium_features: bool = True,
    include_mesh_features: bool = False,
) -> pd.DataFrame:
    """Original sdata.h5 conversion logic (extracted for dispatch)."""
    data = load_m3dc1_hdf5(
        file_path,
        include_equilibrium=include_equilibrium_features,
        include_mesh=include_mesh_features,
    )
    
    rows = []
    for run_data in data['runs']:
        row = {}
        row['run_id'] = run_data.get('run_id', '')
        input_data = run_data.get('input', {})
        input_keys = input_features or list(input_data.keys())
        
        for key in input_keys:
            if key in input_data:
                value = input_data[key]
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        row[f'input_{key}'] = value.item()
                    elif value.ndim == 1 and len(value) == 1:
                        row[f'input_{key}'] = value[0]
                    else:
                        for i, val in enumerate(value.flat):
                            row[f'input_{key}_{i}'] = val
                else:
                    row[f'input_{key}'] = value
        
        if include_equilibrium_features and data['equilibrium']:
            eq_id = input_data.get('eqid', None)
            for sparc_name, eq_data in data['equilibrium'].items():
                if eq_data.get('id', None) == eq_id:
                    for key in ['a', 'R0', 'kappa', 'delta', 'simag', 'sibry', 'current']:
                        if key in eq_data:
                            value = eq_data[key]
                            if isinstance(value, np.ndarray) and value.ndim == 0:
                                row[f'eq_{key}'] = value.item()
                            elif isinstance(value, np.ndarray) and value.size == 1:
                                row[f'eq_{key}'] = value[0]
                            else:
                                row[f'eq_{key}'] = value
                    if 'qpsi' in eq_data:
                        qpsi = eq_data['qpsi']
                        if isinstance(qpsi, np.ndarray) and len(qpsi) > 0:
                            qflat = qpsi.flatten()
                            row['q0'] = qflat[0]
                            if len(qflat) > 1:
                                idx_95 = int(0.95 * (len(qflat) - 1))
                                row['q95'] = qflat[idx_95]
                    if 'pres' in eq_data:
                        pres = eq_data['pres']
                        if isinstance(pres, np.ndarray) and len(pres) > 0:
                            row['p0'] = pres.flatten()[0]
                    break
        
        output_data = run_data.get('output', {})
        output_keys = output_features or list(output_data.keys())
        for key in output_keys:
            if key in output_data:
                value = output_data[key]
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        row[f'output_{key}'] = value.item()
                    elif value.ndim == 1 and len(value) == 1:
                        row[f'output_{key}'] = value[0]
                    else:
                        flat_value = value.flatten()
                        for i, val in enumerate(flat_value):
                            row[f'output_{key}_{i}'] = val
                else:
                    row[f'output_{key}'] = value
        rows.append(row)
    
    df = pd.DataFrame(rows)
    input_cols = [c for c in df.columns if c.startswith('input_')]
    output_cols = [c for c in df.columns if c.startswith('output_')]
    eq_cols = [c for c in df.columns if c.startswith('eq_')]
    print(f"\n✅ Converted to DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"   📥 Input columns: {len(input_cols)}")
    print(f"   📤 Output columns: {len(output_cols)}")
    if eq_cols:
        print(f"   🔄 Equilibrium columns: {len(eq_cols)}")
    return df


# Re-export per-run batch loader (same directory)
try:
    from dataset_complex_v2 import (
        build_dataframe_from_batch,
        find_complex_v2_files,
        load_complex_v2_for_surge,
        load_single_complex_v2,
    )
    COMPLEX_V2_BATCH_AVAILABLE = True
except ImportError:
    build_dataframe_from_batch = None
    find_complex_v2_files = None
    load_complex_v2_for_surge = None
    load_single_complex_v2 = None
    COMPLEX_V2_BATCH_AVAILABLE = False

__all__ = [
    'load_m3dc1_hdf5',
    'convert_to_dataframe',
    'convert_sdata_complex_v2_to_dataframe',
    'read_m3dc1_hdf5_structure',
    'read_sdata_complex_v2_structure',
    'DELTA_P_SPECTRUM_KEYS',
    'H5PY_AVAILABLE',
]
if COMPLEX_V2_BATCH_AVAILABLE:
    __all__.extend([
        'build_dataframe_from_batch',
        'find_complex_v2_files',
        'load_complex_v2_for_surge',
        'load_single_complex_v2',
    ])

