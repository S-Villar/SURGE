#!/usr/bin/env python3
"""
Collect M3DC1 data from run directories and create a DataFrame.

This script scans a batch folder containing runXX directories, extracts:
- Equilibrium parameters (R0, a, delta, kappa, etc.)
- q0, q95, p0 (from profile files or computed from arrays)
- Growth rate (gamma) from C1.h5 files
- Input parameters (runid, eqid, ntor, pscale, batemanscale)

Output: pandas DataFrame saved as CSV and pickle.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

# Try to import M3DC1 modules (optional - will use fallbacks if not available)
try:
    import m3dc1 as m1
    import fpy
    M3DC1_AVAILABLE = True
except ImportError:
    M3DC1_AVAILABLE = False
    print("Warning: m3dc1 or fpy modules not available. Will use alternative methods.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Error: h5py not available. Please install: pip install h5py")
    sys.exit(1)


def extract_runid_eqid_from_path(run_dir_path, sparc_dir_path):
    """Extract runid and eqid from directory paths."""
    run_match = re.search(r'run(\d+)', str(run_dir_path))
    runid = int(run_match.group(1)) if run_match else None
    
    sparc_match = re.search(r'sparc_(\d+)', str(sparc_dir_path))
    eqid = int(sparc_match.group(1)) if sparc_match else None
    
    return runid, eqid


def read_equilibrium_parameters(sparc_dir, eqid, batch_folder=None):
    """Read equilibrium parameters from equilibrium.h5 and geqdsk files."""
    eq_params = {}
    
    # First priority: Read from equilibrium.h5 (check if it exists)
    eq_h5_file = sparc_dir / "equilibrium.h5"
    if not eq_h5_file.exists():
        # If equilibrium.h5 doesn't exist, return empty dict
        return eq_params
    
    # Read from geqdsk file (contains R0, a, delta, kappa, qpsi, psin, pres)
    geqdsk_file = sparc_dir / "geqdsk"
    if geqdsk_file.exists():
        if M3DC1_AVAILABLE:
            try:
                cwd = os.getcwd()
                os.chdir(sparc_dir)
                gfdat = m1.read_gfile('geqdsk')
                
                # Extract key parameters from geqdsk
                eq_params['R0'] = float(gfdat.rcentr)  # Major radius
                eq_params['a'] = float(gfdat.rdim)  # Minor radius (approximate)
                eq_params['delta'] = float(gfdat.delta) if hasattr(gfdat, 'delta') else None
                eq_params['kappa'] = float(gfdat.kappa) if hasattr(gfdat, 'kappa') else None
                eq_params['qpsi'] = gfdat.qpsi if hasattr(gfdat, 'qpsi') else None
                eq_params['psin'] = gfdat.psin if hasattr(gfdat, 'psin') else None
                eq_params['pres'] = gfdat.pres if hasattr(gfdat, 'pres') else None
                
                os.chdir(cwd)
            except Exception as e:
                os.chdir(cwd)
        else:
            # Fallback: try to read from sdata.h5 if m3dc1 not available
            if batch_folder and H5PY_AVAILABLE:
                sdata_file = Path(batch_folder) / "sdata.h5"
                if sdata_file.exists():
                    try:
                        with h5py.File(sdata_file, 'r') as f:
                            sparc_key = f'sparc_{eqid}'
                            if 'equilibrium' in f and sparc_key in f['equilibrium']:
                                sparc_data = f[f'equilibrium/{sparc_key}']
                                # Read common equilibrium parameters
                                for key in ['R0', 'a', 'delta', 'kappa', 'qpsi', 'psin', 'pres']:
                                    if key in sparc_data:
                                        data = sparc_data[key]
                                        if data.shape == ():
                                            eq_params[key] = float(data[()])
                                        else:
                                            eq_params[key] = data[:]
                    except Exception as e:
                        pass
    
    return eq_params


def extract_q0_q95_p0(sparc_dir, eq_params):
    """Extract q0, q95, p0 from profile files or compute from arrays."""
    q0, q95, p0 = None, None, None
    
    # Try to read from profile files first
    q_profile_file = sparc_dir / "q_profile"
    p_profile_file = sparc_dir / "p_profile"
    
    if q_profile_file.exists():
        try:
            q_data = np.loadtxt(q_profile_file)
            q_nflux = q_data[:, 0]
            q_profile = q_data[:, 1]
            q0 = float(q_profile[0])
            q95 = float(np.interp(0.95, q_nflux, q_profile))
        except Exception as e:
            print(f"  Warning: Could not read q_profile: {e}")
    
    if p_profile_file.exists():
        try:
            p_data = np.loadtxt(p_profile_file)
            p_profile = p_data[:, 1]
            p0 = float(p_profile[0])
        except Exception as e:
            print(f"  Warning: Could not read p_profile: {e}")
    
    # Fallback: compute from arrays if profiles not available
    if q0 is None and 'qpsi' in eq_params and eq_params['qpsi'] is not None:
        qpsi = eq_params['qpsi']
        if 'psin' in eq_params and eq_params['psin'] is not None:
            psin = eq_params['psin']
        else:
            psin = np.linspace(0, 1, len(qpsi))
        
        q0 = float(qpsi[0])
        q_interp = interp1d(psin, qpsi, kind='linear', fill_value='extrapolate')
        q95 = float(q_interp(0.95))
    
    if p0 is None and 'pres' in eq_params and eq_params['pres'] is not None:
        pres = eq_params['pres']
        p0 = float(pres[0])
    
    return q0, q95, p0


def read_growth_rate(sparc_dir, runid, eqid, batch_folder=None):
    """Read growth rate (gamma) from growth_rate file, sdata.h5, or C1.h5."""
    gamma = None
    
    # First priority: Read from growth_rate text file
    growth_rate_file = sparc_dir / "growth_rate"
    if growth_rate_file.exists():
        try:
            with open(growth_rate_file, 'r') as f:
                content = f.read().strip()
                # Try to parse as float
                gamma = float(content)
                return gamma
        except (ValueError, IOError) as e:
            pass
    
    # Second priority: Try to read from sdata.h5 (if available)
    if batch_folder and H5PY_AVAILABLE:
        sdata_file = Path(batch_folder) / "sdata.h5"
        if sdata_file.exists():
            try:
                with h5py.File(sdata_file, 'r') as f:
                    # Find run with matching runid and eqid
                    for run_key in f.keys():
                        if run_key.startswith('run_'):
                            try:
                                run_runid = int(f[f'{run_key}/input/runid'][()])
                                run_eqid = int(f[f'{run_key}/input/eqid'][()])
                                if run_runid == runid and run_eqid == eqid:
                                    if 'output' in f[run_key] and 'gamma' in f[f'{run_key}/output']:
                                        gamma = float(f[f'{run_key}/output/gamma'][()])
                                        return gamma
                            except:
                                continue
            except Exception as e:
                pass
    
    # Third priority: Try to read from C1.h5 file (requires m3dc1)
    c1_h5_file = sparc_dir / "C1.h5"
    if c1_h5_file.exists() and M3DC1_AVAILABLE:
        try:
            cwd = os.getcwd()
            os.chdir(sparc_dir)
            mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
            gamma = 0.5 * m1.growth_rate(sim=mysim, slurm=False)[0]
            os.chdir(cwd)
        except Exception as e:
            os.chdir(cwd)
    
    return gamma


def read_input_parameters(sparc_dir):
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
                    if 'ntor' in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            input_params['ntor'] = int(parts[2])
                    elif 'pscale' in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            input_params['pscale'] = float(parts[2])
                    elif 'batemanscale' in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            input_params['batemanscale'] = float(parts[2])
        except Exception as e:
            print(f"  Warning: Could not read C1input: {e}")
    
    return input_params


def read_gs_error(sparc_dir):
    """Read GS error from gs_error file."""
    gs_error = None
    
    gs_error_file = sparc_dir / "gs_error"
    if gs_error_file.exists():
        try:
            with open(gs_error_file, 'r') as f:
                content = f.read().strip()
                # Try to parse as float (scientific notation supported)
                gs_error = float(content)
        except (ValueError, IOError) as e:
            pass
    
    return gs_error


def collect_data_from_batch_folder(batch_folder, output_file=None):
    """
    Collect data from all run directories in a batch folder.
    
    Parameters:
    -----------
    batch_folder : str or Path
        Path to the batch folder containing runXX directories
    output_file : str or Path, optional
        Path to save the output DataFrame (CSV and pickle)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: runid, eqid, R0, a, delta, kappa, q0, q95, p0, 
                                gamma, ntor, pscale, batemanscale
    """
    batch_folder = Path(batch_folder)
    if not batch_folder.exists():
        raise ValueError(f"Batch folder does not exist: {batch_folder}")
    
    print(f"Scanning batch folder: {batch_folder}")
    print("=" * 80)
    
    data_list = []
    
    # Find all run directories
    run_dirs = sorted([d for d in batch_folder.iterdir() 
                      if d.is_dir() and d.name.startswith('run') 
                      and not d.name.endswith('.nate')])
    
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
            
            # Check if equilibrium.h5 exists (required)
            eq_h5_file = sparc_dir / "equilibrium.h5"
            if not eq_h5_file.exists():
                print(f"  ⚠ (skipping - no equilibrium.h5)", end=" ")
                continue
            
            # Initialize data dictionary
            case_data = {
                'runid': runid,
                'eqid': eqid,
                'converged': converged
            }
            
            # Read equilibrium parameters (from equilibrium.h5/geqdsk)
            eq_params = read_equilibrium_parameters(sparc_dir, eqid, batch_folder)
            
            # Extract key equilibrium parameters
            case_data['R0'] = eq_params.get('R0', None)
            case_data['a'] = eq_params.get('a', None)
            case_data['delta'] = eq_params.get('delta', None)
            case_data['kappa'] = eq_params.get('kappa', None)
            
            # Extract q0, q95, p0
            q0, q95, p0 = extract_q0_q95_p0(sparc_dir, eq_params)
            case_data['q0'] = q0
            case_data['q95'] = q95
            case_data['p0'] = p0
            
            # Read growth rate (only if converged)
            if converged:
                gamma = read_growth_rate(sparc_dir, runid, eqid, batch_folder)
                case_data['gamma'] = gamma
            else:
                case_data['gamma'] = None
            
            # Read GS error
            gs_error = read_gs_error(sparc_dir)
            case_data['gs_error'] = gs_error
            
            # Read input parameters
            input_params = read_input_parameters(sparc_dir)
            case_data['ntor'] = input_params['ntor']
            case_data['pscale'] = input_params['pscale']
            case_data['batemanscale'] = input_params['batemanscale']
            
            data_list.append(case_data)
            
            # Print status
            status = "✓" if (q0 is not None and gamma is not None) else "⚠"
            print(f"{status}")
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    print("\n" + "=" * 80)
    print(f"Collected {len(df)} cases")
    
    # Print convergence statistics
    if 'converged' in df.columns:
        converged_count = df['converged'].sum()
        print(f"\nConvergence statistics:")
        print(f"   Converged cases (C1.h5 exists): {converged_count}/{len(df)} ({100*converged_count/len(df):.1f}%)")
        print(f"   Non-converged cases: {len(df) - converged_count}/{len(df)} ({100*(len(df)-converged_count)/len(df):.1f}%)")
    
    print(f"\nData completeness:")
    print(f"   Cases with q0: {df['q0'].notna().sum()}/{len(df)}")
    print(f"   Cases with q95: {df['q95'].notna().sum()}/{len(df)}")
    print(f"   Cases with p0: {df['p0'].notna().sum()}/{len(df)}")
    print(f"   Cases with gamma: {df['gamma'].notna().sum()}/{len(df)}")
    print(f"   Cases with gs_error: {df['gs_error'].notna().sum()}/{len(df)}")
    
    # Save to file if specified
    if output_file:
        output_file = Path(output_file)
        # Save as CSV
        csv_file = output_file.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        print(f"\nSaved DataFrame to: {csv_file}")
        
        # Save as HDF5 using h5py directly (more compatible)
        h5_file = output_file.with_suffix('.h5')
        if H5PY_AVAILABLE:
            try:
                with h5py.File(h5_file, 'w') as f:
                    # Create a group for the data
                    grp = f.create_group('m3dc1_data')
                    
                    # Save each column as a dataset
                    for col in df.columns:
                        data = df[col].values
                        if df[col].dtype == 'object':
                            # Handle string/object columns
                            grp.create_dataset(col, data=[str(x).encode('utf-8') if pd.notna(x) else b'' for x in data])
                        else:
                            # Handle numeric columns
                            grp.create_dataset(col, data=data, dtype=df[col].dtype)
                    
                    # Store column names and index as attributes
                    grp.attrs['columns'] = [col.encode('utf-8') for col in df.columns]
                    grp.attrs['nrows'] = len(df)
                    grp.attrs['description'] = 'M3DC1 collected data: runid, eqid, equilibrium parameters, q0, q95, p0, gamma, and input parameters'
                
                print(f"Saved DataFrame to: {h5_file}")
            except Exception as e:
                print(f"Warning: Could not save to HDF5: {e}")
        else:
            print(f"Warning: h5py not available, skipping HDF5 save")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect M3DC1 data from run directories into a DataFrame"
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
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        batch_path = Path(args.batch_folder)
        args.output = batch_path / "m3dc1_data"
    
    # Collect data
    df = collect_data_from_batch_folder(args.batch_folder, args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(df.describe())
    print("\nFirst few rows:")
    print(df.head(10))

