#!/usr/bin/env python3
"""
Collect M3DC1 data from batch folder without requiring analysis.pro output files.

This script scans a batch folder containing runXX directories and extracts:
- Growth rate (gamma) from C1.h5 files
- q profiles (q0, q95, qmin) from C1.h5 files  
- p profiles (p0) from C1.h5 files
- Equilibrium parameters from equilibrium.h5/geqdsk files
- Input parameters (ntor, pscale, batemanscale) from C1input files
- GS error from gs_error files
- Walltime from started/finished files

Uses m3dc1_python_code utilities when available, or h5py for direct HDF5 reading.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Optional, Dict, Any, Tuple

# Try to import h5py (required)
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)

# Try to import M3DC1 modules (optional - will use fallbacks if not available)
try:
    import fpy
    import m3dc1 as m1
    M3DC1_AVAILABLE = True
except ImportError:
    M3DC1_AVAILABLE = False
    print("Note: m3dc1/fpy modules not available. Will use h5py direct reading.")

# Try to import m3dc1_python_code utilities if available (works without fusion-io)
M3DC1_PYTHON_CODE_AVAILABLE = False
M3DC1_READ_H5 = None
try:
    # Try importing read_h5 functions that work with h5py directly
    from m3dc1.read_h5 import readC1File, readParameter
    M3DC1_PYTHON_CODE_AVAILABLE = True
    M3DC1_READ_H5 = {'readC1File': readC1File, 'readParameter': readParameter}
except ImportError:
    # Will try to add m3dc1_python_code to path dynamically if found in batch folder
    pass


def extract_runid_eqid_from_path(run_dir_path: Path, sparc_dir_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Extract runid and eqid from directory paths."""
    run_match = re.search(r'run(\d+)', str(run_dir_path))
    runid = int(run_match.group(1)) if run_match else None
    
    sparc_match = re.search(r'sparc_(\d+)', str(sparc_dir_path))
    eqid = int(sparc_match.group(1)) if sparc_match else None
    
    return runid, eqid


def read_growth_rate_from_c1h5(c1_h5_file: Path, use_m3dc1: bool = True) -> Optional[float]:
    """
    Read growth rate (gamma) from C1.h5 file.
    
    Uses m3dc1/fpy if available, otherwise reads directly from HDF5.
    """
    if not c1_h5_file.exists():
        return None
    
    gamma = None
    
    # Method 1: Use m3dc1/fpy if available (most accurate)
    if use_m3dc1 and M3DC1_AVAILABLE:
        try:
            cwd = os.getcwd()
            os.chdir(c1_h5_file.parent)
            mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
            gamma_result = m1.growth_rate(sim=mysim, slurm=False)
            if gamma_result and len(gamma_result) > 0:
                gamma = 0.5 * gamma_result[0]  # Convert to growth rate
            os.chdir(cwd)
            return gamma
        except Exception as e:
            os.chdir(cwd)
            # Fall through to h5py method
    
    # Method 2: Read directly from HDF5 using h5py
    if H5PY_AVAILABLE:
        try:
            with h5py.File(c1_h5_file, 'r') as f:
                # Try to find gamma/growth_rate in various possible locations
                # Check scalars group first (common in M3DC1)
                if 'scalars' in f:
                    scalars = f['scalars']
                    for key in ['gamma', 'growth_rate', 'growth', 'gamma_rate']:
                        if key in scalars:
                            data = scalars[key]
                            if hasattr(data, 'shape') and data.shape == ():
                                gamma = float(data[()])
                                return gamma
                            elif hasattr(data, '__len__') and len(data) > 0:
                                gamma = float(data[-1])
                                return gamma
                
                # Check time_traces group OR scalars group (M3DC1 stores time traces in scalars)
                traces = None
                if 'time_traces' in f:
                    traces = f['time_traces']
                elif 'scalars' in f:
                    # In M3DC1, time traces are often stored in scalars group
                    traces = f['scalars']
                
                if traces is not None:
                    # Look for gamma/growth_rate directly in traces
                    for key in ['gamma', 'growth_rate', 'growth', 'gamma_rate']:
                        if key in traces:
                            data = traces[key]
                            if hasattr(data, 'shape') and data.shape == ():
                                gamma = float(data[()])
                                return gamma
                            elif hasattr(data, '__len__') and len(data) > 0:
                                gamma = float(data[-1])
                                return gamma
                    
                    # Compute from kinetic energy trace (for linear simulations)
                    # M3DC1 stores kinetic energy as E_KP, E_KT, E_K3 components
                    ke_components = []
                    ke_names = ['E_KP', 'E_KT', 'E_K3']
                    
                    for name in ke_names:
                        if name in traces:
                            ke_components.append(np.array(traces[name][:]))
                    
                    # Also try single KE trace names
                    if len(ke_components) == 0:
                        for name in ['ke', 'kinetic_energy', 'E_K']:
                            if name in traces:
                                ke_components.append(np.array(traces[name][:]))
                                break
                    
                    if len(ke_components) > 0:
                        # Sum all KE components
                        if len(ke_components) > 1:
                            ke_total = np.sum(ke_components, axis=0)
                        else:
                            ke_total = ke_components[0]
                        
                        # Get time array
                        if 'time' in traces:
                            times = np.array(traces['time'][:])
                        else:
                            times = np.arange(len(ke_total))
                        
                        # Filter out zeros and negative values, need at least 10 points
                        if len(ke_total) > 10:
                            mask = (ke_total > 1e-20) & np.isfinite(ke_total)
                            if np.sum(mask) > 10:
                                ke_filtered = ke_total[mask]
                                times_filtered = times[mask]
                                
                                # Use last portion of data (where growth is established)
                                # Use 70% onwards to avoid initial transients
                                start_idx = max(0, int(0.7 * len(ke_filtered)))
                                if len(ke_filtered) - start_idx > 5:
                                    log_ke = np.log(ke_filtered[start_idx:])
                                    times_subset = times_filtered[start_idx:]
                                    
                                    # Linear fit to log values: log(KE) = gamma*t + const
                                    if len(times_subset) > 1 and np.all(np.isfinite(log_ke)):
                                        coeffs = np.polyfit(times_subset, log_ke, 1)
                                        gamma = coeffs[0]  # Growth rate (slope)
                                        return gamma
                
                # Check other possible top-level locations
                possible_paths = [
                    'output/gamma',
                    'output/growth_rate',
                    'gamma',
                    'growth_rate',
                    'eigenvalue/gamma',
                ]
                
                for path in possible_paths:
                    try:
                        # Handle nested paths
                        if '/' in path:
                            parts = path.split('/')
                            obj = f
                            found = True
                            for part in parts:
                                if part in obj:
                                    obj = obj[part]
                                else:
                                    found = False
                                    break
                            
                            if found and isinstance(obj, h5py.Dataset):
                                if obj.shape == ():
                                    gamma = float(obj[()])
                                    return gamma
                                elif len(obj) > 0:
                                    gamma = float(obj[-1])
                                    return gamma
                        elif path in f:
                            data = f[path]
                            if hasattr(data, 'shape') and data.shape == ():
                                gamma = float(data[()])
                                return gamma
                            elif hasattr(data, '__len__') and len(data) > 0:
                                gamma = float(data[-1])
                                return gamma
                    except:
                        continue
                        
        except Exception as e:
            pass
    
    return gamma


def read_q_profile_from_h5(h5_file: Path, use_m3dc1: bool = True, prefer_equilibrium: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read q profile from HDF5 file (equilibrium.h5 or C1.h5).
    
    Returns (nflux, q_profile) where nflux is normalized flux and q_profile is safety factor.
    
    Parameters:
    -----------
    h5_file : Path
        Path to HDF5 file (equilibrium.h5 or C1.h5)
    use_m3dc1 : bool
        Whether to use m3dc1 flux_average if available
    prefer_equilibrium : bool
        If True, prefer reading from equilibrium data rather than computing flux averages
    """
    if not h5_file.exists():
        return None, None
    
    nflux = None
    q_profile = None
    
    # Method 1: Read directly from equilibrium.h5 (fastest, most reliable)
    if H5PY_AVAILABLE:
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try various possible locations for q profile in equilibrium.h5
                possible_q_paths = [
                    'qpsi',  # Most common in equilibrium.h5
                    'q_profile',
                    'q',
                    'safety_factor',
                    'equilibrium/qpsi',
                ]
                
                possible_psin_paths = [
                    'psin',  # Most common normalized flux coordinate
                    'psi_norm',
                    'normalized_flux',
                    'nflux',
                ]
                
                # Find q profile
                for path in possible_q_paths:
                    if path in f:
                        q_data = f[path]
                        q_profile = np.array(q_data[:])
                        break
                
                # Find corresponding flux coordinate
                if q_profile is not None:
                    for path in possible_psin_paths:
                        if path in f:
                            nflux = np.array(f[path][:])
                            break
                    
                    # If no psin found, create normalized flux array
                    if nflux is None:
                        nflux = np.linspace(0, 1, len(q_profile))
                    
                    return nflux, q_profile
        except Exception as e:
            pass
    
    # Method 2: Use m3dc1 flux_average if available (for C1.h5)
    if use_m3dc1 and M3DC1_AVAILABLE and not prefer_equilibrium:
        try:
            # Import flux_average from m3dc1_python_code if available
            batch_folder = h5_file.parent.parent.parent
            m3dc1_code_path = batch_folder / 'm3dc1_python_code'
            if m3dc1_code_path.exists():
                sys.path.insert(0, str(m3dc1_code_path))
                try:
                    from m3dc1.flux_average import flux_average
                    cwd = os.getcwd()
                    os.chdir(h5_file.parent)
                    nflux, q_profile = flux_average('q', slice=-1, file=str(h5_file), points=400, nflux=None)
                    os.chdir(cwd)
                    return nflux, q_profile
                except Exception:
                    pass
                finally:
                    if str(m3dc1_code_path) in sys.path:
                        sys.path.remove(str(m3dc1_code_path))
        except Exception:
            pass
    
    # Method 3: Try reading from C1.h5 structure (if not equilibrium.h5)
    if H5PY_AVAILABLE and 'equilibrium' not in str(h5_file):
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try various possible locations for q profile
                possible_paths = [
                    'q_profile',
                    'flux_coordinates/q',
                    'equilibrium/qpsi',
                    'qpsi',
                ]
                
                for path in possible_paths:
                    if path in f:
                        q_data = f[path]
                        q_profile = np.array(q_data[:])
                        # Try to find corresponding flux coordinate
                        if 'psin' in f:
                            nflux = np.array(f['psin'][:])
                        elif 'flux_coordinates/psi_norm' in f:
                            nflux = np.array(f['flux_coordinates/psi_norm'][:])
                        else:
                            # Create normalized flux array
                            nflux = np.linspace(0, 1, len(q_profile))
                        break
        except Exception as e:
            pass
    
    return nflux, q_profile


def read_p_profile_from_h5(h5_file: Path, use_m3dc1: bool = True, prefer_equilibrium: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read pressure profile from HDF5 file (equilibrium.h5 or C1.h5).
    
    Returns (nflux, p_profile) where nflux is normalized flux and p_profile is pressure.
    
    Parameters:
    -----------
    h5_file : Path
        Path to HDF5 file (equilibrium.h5 or C1.h5)
    use_m3dc1 : bool
        Whether to use m3dc1 flux_average if available
    prefer_equilibrium : bool
        If True, prefer reading from equilibrium data rather than computing flux averages
    """
    if not h5_file.exists():
        return None, None
    
    nflux = None
    p_profile = None
    
    # Method 1: Read directly from equilibrium.h5 (fastest, most reliable)
    if H5PY_AVAILABLE:
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try various possible locations for pressure profile in equilibrium.h5
                possible_p_paths = [
                    'pres',  # Most common in equilibrium.h5
                    'pressure',
                    'p_profile',
                    'pressure_profile',
                    'equilibrium/pres',
                ]
                
                possible_psin_paths = [
                    'psin',  # Most common normalized flux coordinate
                    'psi_norm',
                    'normalized_flux',
                    'nflux',
                ]
                
                # Find pressure profile
                for path in possible_p_paths:
                    if path in f:
                        p_data = f[path]
                        p_profile = np.array(p_data[:])
                        break
                
                # Find corresponding flux coordinate
                if p_profile is not None:
                    for path in possible_psin_paths:
                        if path in f:
                            nflux = np.array(f[path][:])
                            break
                    
                    # If no psin found, create normalized flux array
                    if nflux is None:
                        nflux = np.linspace(0, 1, len(p_profile))
                    
                    return nflux, p_profile
        except Exception as e:
            pass
    
    # Method 2: Use m3dc1 flux_average if available (for C1.h5)
    if use_m3dc1 and M3DC1_AVAILABLE and not prefer_equilibrium:
        try:
            batch_folder = h5_file.parent.parent.parent
            m3dc1_code_path = batch_folder / 'm3dc1_python_code'
            if m3dc1_code_path.exists():
                sys.path.insert(0, str(m3dc1_code_path))
                try:
                    from m3dc1.flux_average import flux_average
                    cwd = os.getcwd()
                    os.chdir(h5_file.parent)
                    nflux, p_profile = flux_average('p', slice=-1, file=str(h5_file), points=400, nflux=None)
                    os.chdir(cwd)
                    return nflux, p_profile
                except Exception:
                    pass
                finally:
                    if str(m3dc1_code_path) in sys.path:
                        sys.path.remove(str(m3dc1_code_path))
        except Exception:
            pass
    
    # Method 3: Try reading from C1.h5 structure (if not equilibrium.h5)
    if H5PY_AVAILABLE and 'equilibrium' not in str(h5_file):
        try:
            with h5py.File(h5_file, 'r') as f:
                # Try various possible locations for pressure profile
                possible_paths = [
                    'p_profile',
                    'pressure_profile',
                    'equilibrium/pres',
                    'pres',
                    'pressure',
                ]
                
                for path in possible_paths:
                    if path in f:
                        p_data = f[path]
                        p_profile = np.array(p_data[:])
                        # Try to find corresponding flux coordinate
                        if 'psin' in f:
                            nflux = np.array(f['psin'][:])
                        elif 'flux_coordinates/psi_norm' in f:
                            nflux = np.array(f['flux_coordinates/psi_norm'][:])
                        else:
                            # Create normalized flux array
                            nflux = np.linspace(0, 1, len(p_profile))
                        break
        except Exception as e:
            pass
    
    return nflux, p_profile


def extract_q0_q95_qmin(nflux: Optional[np.ndarray], q_profile: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract q0, q95, qmin from q profile."""
    if q_profile is None or len(q_profile) == 0:
        return None, None, None
    
    q0 = float(q_profile[0])
    
    # q95: interpolate at normalized flux = 0.95
    if nflux is not None and len(nflux) > 1:
        psin_min = float(np.min(nflux))
        psin_max = float(np.max(nflux))
        print(f"psinmax =:{psin_max} and psimin is =:{psin_min}")
        q_interp = interp1d(nflux, q_profile, kind='cubic', fill_value='extrapolate')
        q95 = float(q_interp(0.95))
    else:
        # Approximate: use 95% index
        idx_95 = int(0.95 * (len(q_profile) - 1))
        q95 = float(q_profile[idx_95])
    
    qmin = float(np.min(q_profile))
    
    return q0, q95, qmin


def extract_p0(nflux: Optional[np.ndarray], p_profile: Optional[np.ndarray]) -> Optional[float]:
    """Extract p0 (pressure at axis) from pressure profile."""
    if p_profile is None or len(p_profile) == 0:
        return None
    
    return float(p_profile[0])


def read_equilibrium_parameters(sparc_dir: Path, eqid: int, batch_folder: Optional[Path] = None) -> Dict[str, Any]:
    """Read equilibrium parameters from sdata.h5, equilibrium.h5, or geqdsk files."""
    eq_params = {}
    
    # Method 1: Try to read from sdata.h5 first (most reliable, contains all equilibrium data)
    if batch_folder and H5PY_AVAILABLE:
        sdata_file = batch_folder / "sdata.h5"
        if sdata_file.exists():
            try:
                with h5py.File(sdata_file, 'r') as f:
                    sparc_key = f'sparc_{eqid}'
                    if 'equilibrium' in f and sparc_key in f['equilibrium']:
                        sparc_data = f[f'equilibrium/{sparc_key}']
                        # Read common equilibrium parameters
                        for key in ['R0', 'a', 'delta', 'kappa', 'qpsi', 'psin', 'pres', 'simag', 'sibry', 'current']:
                            if key in sparc_data:
                                data = sparc_data[key]
                                if data.shape == ():
                                    eq_params[key] = float(data[()])
                                else:
                                    eq_params[key] = np.array(data[:])
                        # If we got the data, return early
                        if eq_params:
                            return eq_params
            except Exception as e:
                pass
    
    # Method 2: Try to read from equilibrium.h5
    eq_h5_file = sparc_dir / "equilibrium.h5"
    if eq_h5_file.exists() and H5PY_AVAILABLE:
        try:
            with h5py.File(eq_h5_file, 'r') as f:
                # Read common equilibrium parameters
                for key in ['R0', 'a', 'delta', 'kappa', 'qpsi', 'psin', 'pres', 'simag', 'sibry', 'current']:
                    if key in f:
                        data = f[key]
                        if data.shape == ():
                            eq_params[key] = float(data[()])
                        else:
                            eq_params[key] = np.array(data[:])
        except Exception as e:
            pass
    
    # Method 3: Fallback: Try to read from geqdsk using m3dc1 if available
    geqdsk_file = sparc_dir / "geqdsk"
    if geqdsk_file.exists() and M3DC1_AVAILABLE and not eq_params:
        try:
            cwd = os.getcwd()
            os.chdir(sparc_dir)
            gfdat = m1.read_gfile('geqdsk')
            eq_params['R0'] = float(gfdat.rcentr)
            eq_params['a'] = float(gfdat.rdim)
            eq_params['delta'] = float(gfdat.delta) if hasattr(gfdat, 'delta') else None
            eq_params['kappa'] = float(gfdat.kappa) if hasattr(gfdat, 'kappa') else None
            eq_params['qpsi'] = gfdat.qpsi if hasattr(gfdat, 'qpsi') else None
            eq_params['psin'] = gfdat.psin if hasattr(gfdat, 'psin') else None
            eq_params['pres'] = gfdat.pres if hasattr(gfdat, 'pres') else None
            os.chdir(cwd)
        except Exception as e:
            os.chdir(cwd)
    
    return eq_params


def read_input_parameters(sparc_dir: Path) -> Dict[str, Optional[float]]:
    """Read input parameters from C1input file."""
    input_params = {
        'ntor': None,
        'pscale': None,
        'batemanscale': None
    }
    
    c1input_file = sparc_dir / "C1input"
    if c1input_file.exists():
        try:
            with open(c1input_file, 'r') as f:
                for line in f:
                    # Skip comment lines
                    if line.strip().startswith('!'):
                        continue
                    
                    line_lower = line.lower()
                    # Handle format: "parameter = value" or "parameter value"
                    if '=' in line:
                        parts = [p.strip() for p in line.split('=')]
                        if len(parts) == 2:
                            param_name = parts[0].strip().lower()
                            param_value = parts[1].strip()
                            
                            if param_name == 'ntor':
                                try:
                                    input_params['ntor'] = int(float(param_value))
                                except:
                                    pass
                            elif param_name == 'pscale':
                                try:
                                    input_params['pscale'] = float(param_value)
                                except:
                                    pass
                            elif param_name == 'batemanscale':
                                try:
                                    input_params['batemanscale'] = float(param_value)
                                except:
                                    pass
                    else:
                        # Try space-separated format
                        parts = line.split()
                        for i, part in enumerate(parts):
                            part_lower = part.lower()
                            if part_lower == 'ntor' and i + 1 < len(parts):
                                try:
                                    input_params['ntor'] = int(float(parts[i+1]))
                                except:
                                    pass
                            elif part_lower == 'pscale' and i + 1 < len(parts):
                                try:
                                    input_params['pscale'] = float(parts[i+1])
                                except:
                                    pass
                            elif part_lower == 'batemanscale' and i + 1 < len(parts):
                                try:
                                    input_params['batemanscale'] = float(parts[i+1])
                                except:
                                    pass
        except Exception as e:
            pass
    
    return input_params


def read_gs_error(sparc_dir: Path) -> Optional[float]:
    """Read GS error from gs_error file."""
    gs_error_file = sparc_dir / "gs_error"
    if gs_error_file.exists():
        try:
            with open(gs_error_file, 'r') as f:
                content = f.read().strip()
                return float(content)
        except (ValueError, IOError):
            pass
    return None


def read_walltime(sparc_dir: Path) -> Optional[float]:
    """Read walltime from started/finished files."""
    started_file = sparc_dir / "started"
    finished_file = sparc_dir / "finished"
    
    if started_file.exists() and finished_file.exists():
        try:
            start_time = os.path.getmtime(started_file)
            finish_time = os.path.getmtime(finished_file)
            return finish_time - start_time
        except Exception:
            pass
    return None


def collect_data_from_batch_folder(
    batch_folder: Path,
    output_file: Optional[Path] = None,
    use_m3dc1: bool = True
) -> pd.DataFrame:
    """
    Collect data from all run directories in a batch folder.
    
    Parameters:
    -----------
    batch_folder : Path
        Path to the batch folder containing runXX directories
    output_file : Path, optional
        Path to save the output DataFrame (CSV and pickle)
    use_m3dc1 : bool, default=True
        Whether to use m3dc1/fpy when available (more accurate)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: runid, eqid, R0, a, delta, kappa, q0, q95, qmin, p0,
                                gamma, ntor, pscale, batemanscale, gs_error, walltime
    """
    batch_folder = Path(batch_folder)
    if not batch_folder.exists():
        raise ValueError(f"Batch folder does not exist: {batch_folder}")
    
    # Try to add m3dc1_python_code to path if it exists in batch folder
    global M3DC1_PYTHON_CODE_AVAILABLE, M3DC1_READ_H5
    m3dc1_python_code_path = batch_folder / 'm3dc1_python_code'
    if m3dc1_python_code_path.exists() and not M3DC1_PYTHON_CODE_AVAILABLE:
        try:
            # Add parent directory to path so we can import m3dc1
            if str(m3dc1_python_code_path.parent) not in sys.path:
                sys.path.insert(0, str(m3dc1_python_code_path.parent))
            # Try importing read_h5 functions (these work without fusion-io)
            from m3dc1.read_h5 import readC1File, readParameter
            M3DC1_PYTHON_CODE_AVAILABLE = True
            M3DC1_READ_H5 = {'readC1File': readC1File, 'readParameter': readParameter}
            print(f"Note: Found m3dc1_python_code in batch folder. Using read_h5 functions (no fusion-io required).")
        except ImportError as e:
            # Remove from path if import failed
            if str(m3dc1_python_code_path.parent) in sys.path:
                sys.path.remove(str(m3dc1_python_code_path.parent))
            # Continue without m3dc1_python_code
    
    print(f"Scanning batch folder: {batch_folder}")
    print("=" * 80)
    
    data_list = []
    
    # Find all run directories
    run_dirs = sorted([d for d in batch_folder.iterdir() 
                      if d.is_dir() and d.name.startswith('run') 
                      and d.name[3:].isdigit()])
    
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        print(f"\nProcessing {run_dir.name}...")
        
        # Find all sparc subdirectories
        sparc_dirs = sorted([d for d in run_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('sparc_')])
        
        print(f"  Found {len(sparc_dirs)} sparc directories")
        
        for sparc_dir in sparc_dirs:
            runid, eqid = extract_runid_eqid_from_path(run_dir, sparc_dir)
            
            if runid is None or eqid is None:
                print(f"  Warning: Could not extract runid/eqid from {sparc_dir}")
                continue
            
            print(f"  Processing sparc_{eqid}...", end=" ")
            
            # Check if case converged (C1.h5 exists)
            c1_h5_file = sparc_dir / "C1.h5"
            converged = c1_h5_file.exists()
            
            # Initialize data dictionary
            case_data = {
                'runid': runid,
                'eqid': eqid,
                'converged': converged
            }
            
            # Read equilibrium parameters
            eq_params = read_equilibrium_parameters(sparc_dir, eqid, batch_folder)
            case_data['R0'] = eq_params.get('R0', None)
            case_data['a'] = eq_params.get('a', None)
            case_data['delta'] = eq_params.get('delta', None)
            case_data['kappa'] = eq_params.get('kappa', None)
            
            # Read q and p profiles - prefer sdata.h5, then equilibrium.h5, fall back to C1.h5
            # First try to get from equilibrium parameters (which reads from sdata.h5)
            nflux_q, q_profile = None, None
            nflux_p, p_profile = None, None
            
            if 'qpsi' in eq_params and 'psin' in eq_params:
                q_profile = eq_params['qpsi']
                nflux_q = eq_params['psin']
            
            if 'pres' in eq_params and 'psin' in eq_params:
                p_profile = eq_params['pres']
                nflux_p = eq_params['psin']
            
            # If not found in equilibrium params, try equilibrium.h5
            eq_h5_file = sparc_dir / "equilibrium.h5"
            if (q_profile is None or p_profile is None) and eq_h5_file.exists():
                if q_profile is None:
                    nflux_q, q_profile = read_q_profile_from_h5(eq_h5_file, use_m3dc1=use_m3dc1, prefer_equilibrium=True)
                if p_profile is None:
                    nflux_p, p_profile = read_p_profile_from_h5(eq_h5_file, use_m3dc1=use_m3dc1, prefer_equilibrium=True)
            
            # If still not found and C1.h5 exists, try C1.h5
            if (q_profile is None or p_profile is None) and converged:
                if q_profile is None:
                    nflux_q, q_profile = read_q_profile_from_h5(c1_h5_file, use_m3dc1=use_m3dc1, prefer_equilibrium=False)
                if p_profile is None:
                    nflux_p, p_profile = read_p_profile_from_h5(c1_h5_file, use_m3dc1=use_m3dc1, prefer_equilibrium=False)
            
            # Extract q0, q95, qmin from q profile
            if q_profile is not None:
                q0, q95, qmin = extract_q0_q95_qmin(nflux_q, q_profile)
                case_data['q0'] = q0
                case_data['q95'] = q95
                case_data['qmin'] = qmin
            else:
                case_data['q0'] = None
                case_data['q95'] = None
                case_data['qmin'] = None
            
            # Extract p0 from pressure profile
            if p_profile is not None:
                case_data['p0'] = extract_p0(nflux_p, p_profile)
            else:
                case_data['p0'] = None
            
            # Read growth rate from C1.h5 (only available if converged)
            if converged:
                gamma = read_growth_rate_from_c1h5(c1_h5_file, use_m3dc1=use_m3dc1)
                case_data['gamma'] = gamma
            else:
                case_data['gamma'] = None
            
            # Read GS error
            gs_error = read_gs_error(sparc_dir)
            case_data['gs_error'] = gs_error
            
            # Read walltime
            walltime = read_walltime(sparc_dir)
            case_data['walltime'] = walltime
            
            # Read input parameters
            input_params = read_input_parameters(sparc_dir)
            case_data['ntor'] = input_params['ntor']
            case_data['pscale'] = input_params['pscale']
            case_data['batemanscale'] = input_params['batemanscale']
            
            data_list.append(case_data)
            
            # Print status - checkmark if we got at least q0 (gamma may not be available without m3dc1)
            status = "✓" if case_data.get('q0') is not None else "⚠"
            print(f"{status}")
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    print("\n" + "=" * 80)
    print(f"Collected {len(df)} cases")
    
    # Print statistics
    if 'converged' in df.columns:
        converged_count = df['converged'].sum()
        print(f"\nConvergence statistics:")
        print(f"   Converged cases: {converged_count}/{len(df)} ({100*converged_count/len(df):.1f}%)")
    
    print(f"\nData completeness:")
    for col in ['q0', 'q95', 'qmin', 'p0', 'gamma', 'gs_error', 'walltime']:
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"   Cases with {col}: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    
    # Save to file if specified
    if output_file:
        output_file = Path(output_file)
        # Save as CSV
        csv_file = output_file.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nSaved DataFrame to: {csv_file}")
        
        # Save as pickle
        pkl_file = output_file.with_suffix('.pkl')
        df.to_pickle(pkl_file)
        print(f"Saved DataFrame to: {pkl_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect M3DC1 data from batch folder without requiring analysis.pro"
    )
    parser.add_argument(
        'batch_folder',
        type=str,
        help='Path to batch folder containing runXX directories'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: m3dc1_data.csv/pkl in batch folder)'
    )
    parser.add_argument(
        '--no-m3dc1',
        action='store_true',
        help='Do not use m3dc1/fpy even if available (use h5py only)'
    )
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        batch_path = Path(args.batch_folder)
        args.output = batch_path / "m3dc1_data"
    
    # Collect data
    df = collect_data_from_batch_folder(
        Path(args.batch_folder),
        Path(args.output),
        use_m3dc1=not args.no_m3dc1
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    if len(df) > 0:
        print(df.describe())
        print("\nFirst few rows:")
        print(df.head(10))
    else:
        print("No data collected. Check that the batch folder contains runXX directories.")

