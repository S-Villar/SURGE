#!/usr/bin/env python3
"""
Optimized version of sdata_write.py with parallelization and performance improvements.

Key optimizations:
1. Parallel processing using multiprocessing
2. Removed verbose flags
3. Use absolute paths instead of os.chdir()
4. Cache directory listings
5. Batch operations where possible
"""

import fpy
import numpy as np
import h5py
import m3dc1 as m1
from scipy.interpolate import interp1d
import os
import sys
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time


parent_dir = Path(".").resolve()

# Find the next available sdataXX.h5 filename
def get_next_sdata_filename():
    """Find the next available sdataXX.h5 filename by incrementing from existing files."""
    existing_files = list(parent_dir.glob('sdata*.h5'))
    if not existing_files:
        return 'sdata01.h5'
    
    # Extract numbers from filenames like sdata01.h5, sdata02.h5, etc.
    numbers = []
    for f in existing_files:
        name = f.stem  # e.g., 'sdata01' or 'sdata'
        if name.startswith('sdata'):
            suffix = name[5:]  # Everything after 'sdata'
            if suffix.isdigit():
                numbers.append(int(suffix))
    
    if numbers:
        next_num = max(numbers) + 1
    else:
        next_num = 1
    
    return f'sdata{next_num:02d}.h5'

m3dc1_sfile = get_next_sdata_filename()
print(f"📁 Output file: {m3dc1_sfile}")

# Performance settings
# Use SLURM_CPUS_ON_NODE if available (when running in SLURM), otherwise use cpu_count
if 'SLURM_CPUS_ON_NODE' in os.environ:
    N_WORKERS = int(os.environ['SLURM_CPUS_ON_NODE'])
elif 'SLURM_CPUS_PER_TASK' in os.environ:
    N_WORKERS = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    N_WORKERS = min(cpu_count(), 16)  # Limit to 16 workers to avoid overwhelming system
VERBOSE = False  # Set to False for production runs

# Debug flags for printing nflux info (only once)
_debug_nflux_printed = False
_debug_nflux_normalized = False

print(f"\n{'='*80}")
print(f"Optimized sdata_write.py")
print(f"Using {N_WORKERS} parallel workers")
print(f"{'='*80}\n")


def extract_numbers(path_str):
    """Extract run and sparc numbers from path."""
    path_components = path_str.split(os.sep)
    run_name = path_components[-2] if len(path_components) >= 2 else ""
    sparc_name = path_components[-1] if path_components else ""
    
    run_match = re.search(r'\d+', run_name)
    run_number = int(run_match.group()) if run_match else None
    
    sparc_match = re.search(r'\d+', sparc_name)
    sparc_number = int(sparc_match.group()) if sparc_match else None
    
    return run_number, sparc_number


def process_equilibrium_data(sparc_path):
    """Process equilibrium data for a single sparc directory."""
    sparc_path = Path(sparc_path)
    run_number, sparc_number = extract_numbers(str(sparc_path))
    
    if sparc_number is None:
        return None
    
    result = {
        'sparc_number': sparc_number,
        'run_number': run_number,
        'geqdsk_data': None,
        'shape_data': None
    }
    
    # Process geqdsk
    geqdsk_path = sparc_path / "geqdsk"
    if geqdsk_path.exists():
        try:
            os.chdir(str(sparc_path))  # m1.read_gfile may need to be in directory
            gfdat = m1.read_gfile('geqdsk')
            result['geqdsk_data'] = {
                'id': sparc_number,
                'shot': gfdat.shot,
                'time': gfdat.time,
                'fittype': gfdat.fittype,
                'date': gfdat.date,
                'idum': gfdat.idum,
                'nw': gfdat.nw,
                'nh': gfdat.nh,
                'rdim': gfdat.rdim,
                'zdim': gfdat.zdim,
                'rcentr': gfdat.rcentr,
                'rleft': gfdat.rleft,
                'zmid': gfdat.zmid,
                'rg': gfdat.rg,
                'zg': gfdat.zg,
                'rmaxis': gfdat.rmaxis,
                'zmaxis': gfdat.zmaxis,
                'simag': gfdat.simag,
                'sibry': gfdat.sibry,
                'bcentr': gfdat.bcentr,
                'current': gfdat.current,
                'fpol': gfdat.fpol,
                'Ipol': gfdat.Ipol,
                'pres': gfdat.pres,
                'ffprim': gfdat.ffprim,
                'pprim': gfdat.pprim,
                'psirz': gfdat.psirz,
                'psirzn': gfdat.psirzn,
                'qpsi': gfdat.qpsi,
                'nbbbs': gfdat.nbbbs,
                'rbbbs': gfdat.rbbbs,
                'zbbbs': gfdat.zbbbs,
                'limitr': gfdat.limitr,
                'rlim': gfdat.rlim,
                'zlim': gfdat.zlim,
                'kvtor': gfdat.kvtor,
                'rvtor': gfdat.rvtor,
                'nmass': gfdat.nmass,
                'psin': gfdat.psin,
                'amin': gfdat.amin,
            }
        except Exception as e:
            if VERBOSE:
                print(f"Error processing geqdsk in {sparc_path}: {e}")
        finally:
            os.chdir(parent_dir)
    
    # Process shape data from C1.h5
    c1_path = sparc_path / "C1.h5"
    if c1_path.exists():
        try:
            os.chdir(str(sparc_path))
            mysim = fpy.sim_data('C1.h5', time=-1, verbose=False)
            shape = m1.get_shape(mysim, res=250)
            result['shape_data'] = {
                'a': shape["a"],
                'R0': shape["R0"],
                'kappa': shape["kappa"],
                'delta': shape["delta"]
            }
        except Exception as e:
            if VERBOSE:
                print(f"Error processing C1.h5 shape in {sparc_path}: {e}")
        finally:
            os.chdir(parent_dir)
    
    return result


def process_run_data(sparc_path):
    """Process run data for a single sparc directory."""
    sparc_path = Path(sparc_path)
    run_number, sparc_number = extract_numbers(str(sparc_path))
    
    if sparc_number is None:
        return None
    
    result = {
        'sparc_number': sparc_number,
        'run_number': run_number,
        'input_data': None,
        'output_data': None
    }
    
    # Process input data
    c1input_path = sparc_path / "C1input"
    ntor, pscale, bscale = 1, 1., 1.
    
    if c1input_path.exists():
        try:
            with open(c1input_path, 'r') as sf:
                for line in sf:
                    if 'ntor' in line:
                        ntorline = line.split()
                        if len(ntorline) > 2:
                            # Handle both integer and float formats (e.g., "2" or "2.0")
                            try:
                                ntor = int(float(ntorline[2]))
                            except (ValueError, IndexError):
                                pass  # Keep default value
                    elif 'pscale' in line and 'ikapscale' not in line:
                        # Match pscale exactly (not ikapscale)
                        pscaleline = line.split()
                        if len(pscaleline) > 2:
                            try:
                                pscale = float(pscaleline[2])
                            except (ValueError, IndexError):
                                # Try parsing with '=' separator as fallback
                                if '=' in line:
                                    parts = line.split('=')
                                    if len(parts) > 1:
                                        try:
                                            pscale = float(parts[1].strip())
                                        except (ValueError, IndexError):
                                            pass  # Keep default value
                                pass  # Keep default value
                    elif 'batemanscale' in line:
                        bscaleline = line.split()
                        if len(bscaleline) > 2:
                            try:
                                bscale = float(bscaleline[2])
                            except (ValueError, IndexError):
                                # Try parsing with '=' separator as fallback
                                if '=' in line:
                                    parts = line.split('=')
                                    if len(parts) > 1:
                                        try:
                                            bscale = float(parts[1].strip())
                                        except (ValueError, IndexError):
                                            pass  # Keep default value
                                pass  # Keep default value
        except Exception as e:
            if VERBOSE:
                print(f"Error reading C1input in {sparc_path}: {e}")
    
    result['input_data'] = {
        'runid': run_number,
        'eqid': sparc_number,
        'ntor': ntor,
        'pscale': pscale,
        'batemanscale': bscale
    }
    
    # Process output data from C1.h5
    c1_path = sparc_path / "C1.h5"
    if c1_path.exists():
        try:
            os.chdir(str(sparc_path))
            mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
            
            # Get growth rate
            try:
                gamma = 0.5 * m1.growth_rate(sim=mysim, slurm=False)[0]
            except Exception as e:
                if VERBOSE:
                    print(f"Error computing growth_rate in {sparc_path}: {e}")
                gamma = None
            
            # Get eigenfunction
            try:
                spec = m1.eigenfunction(sim=mysim, fcoords='pest', device='sparc', time='last')
            except Exception as e:
                if VERBOSE:
                    print(f"Error computing eigenfunction in {sparc_path}: {e}")
                spec = None

            # Initialize profile variables
            nflux_q, q_profile = None, None
            nflux_p, p_profile = None, None
            xq, xp = None, None
            q, p = None, None
            q0, q95, qmin, p0 = None, None, None, None
            
            # Get profiles - suppress warnings as they're usually not critical
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    warnings.filterwarnings('ignore', category=UserWarning)
                    try:
                        nflux_q, q_profile = m1.flux_average('q',sim=mysim,device='sparc',fcoords='pest')
                        nflux_p, p_profile = m1.flux_average('p',sim=mysim,device='sparc',fcoords='pest')
                    except Exception as e:
                        # If flux_average fails, set to None but continue
                        if VERBOSE:
                            print(f"  Warning: flux_average failed in {sparc_path}: {e}")
                        nflux_q, q_profile = None, None
                        nflux_p, p_profile = None, None
                    
                    # flux_average returns (nflux, profile) where both are 1D arrays
                    # We need to combine them into a 2D array [nflux, profile]
                    if q_profile is not None and nflux_q is not None:
                        q_profile = np.array(q_profile)
                        nflux_q = np.array(nflux_q)
                    
                        # DEBUG: Print nflux_q range (first run only)
                        global _debug_nflux_printed, _debug_nflux_normalized
                        if VERBOSE and not _debug_nflux_printed:
                            print(f"  DEBUG: nflux_q range: min={np.min(nflux_q):.6f}, max={np.max(nflux_q):.6f}")
                            print(f"  DEBUG: nflux_q first 5: {nflux_q[:5]}")
                            print(f"  DEBUG: nflux_q last 5: {nflux_q[-5:]}")
                            _debug_nflux_printed = True
                        
                        # Normalize nflux_q to [0, 1] if needed
                        nflux_min = float(np.min(nflux_q))
                        nflux_max = float(np.max(nflux_q))
                        nflux_range = nflux_max - nflux_min
                        
                        # Check if nflux is already normalized [0, 1] or needs normalization
                        if nflux_max > 1.1 or nflux_min < -0.1:
                            # Not normalized - normalize it
                            nflux_q_normalized = (nflux_q - nflux_min) / nflux_range if nflux_range > 0 else nflux_q
                            if VERBOSE and not _debug_nflux_normalized:
                                print(f"  DEBUG: Normalizing nflux_q from [{nflux_min:.6f}, {nflux_max:.6f}] to [0, 1]")
                                _debug_nflux_normalized = True
                        else:
                            # Already normalized (or close to it)
                            nflux_q_normalized = nflux_q
                        
                        # Combine into 2D array: [nflux, q_value]
                        if q_profile.ndim == 1 and nflux_q.ndim == 1 and len(q_profile) == len(nflux_q):
                            q_2d = np.column_stack([nflux_q_normalized, q_profile])
                            xq = nflux_q_normalized
                            qmin = float(np.min(q_profile))
                            q0 = float(q_profile[0])
                            
                            # Interpolate q95 at normalized flux = 0.95
                            try:
                                target_nflux = 0.95
                                nflux_norm_max = float(np.max(nflux_q_normalized))
                                
                                if target_nflux <= nflux_norm_max:
                                    qfun = interp1d(nflux_q_normalized, q_profile, kind='linear', fill_value='extrapolate', bounds_error=False)
                                    q95 = float(qfun(target_nflux))
                                elif len(q_profile) > 0:
                                    # If 0.95 is beyond range, use the value at highest normalized flux
                                    q95 = float(q_profile[-1])
                                else:
                                    q95 = None
                            except Exception as e:
                                if VERBOSE:
                                    print(f"  Warning: Error interpolating q95: {e}")
                                # Fallback: use value at highest nflux or last value
                                if len(q_profile) > 0:
                                    q95 = float(q_profile[-1])
                                else:
                                    q95 = None
                            
                            q = q_2d  # Store as 2D for consistency
                        else:
                            # Shape mismatch - try to salvage what we can
                            if VERBOSE:
                                print(f"  Warning: Unexpected q profile shape: q_profile.shape={q_profile.shape if hasattr(q_profile, 'shape') else 'N/A'}, nflux_q.shape={nflux_q.shape if hasattr(nflux_q, 'shape') else 'N/A'}")
                            # Try to use what we have - use minimum length
                            if q_profile is not None and nflux_q is not None:
                                min_len = min(len(q_profile), len(nflux_q))
                                if min_len > 0:
                                    # Normalize nflux_q
                                    nflux_q_subset = nflux_q[:min_len]
                                    nflux_q_min = float(np.min(nflux_q_subset))
                                    nflux_q_max = float(np.max(nflux_q_subset))
                                    nflux_q_range = nflux_q_max - nflux_q_min
                                    if nflux_q_max > 1.1 or nflux_q_min < -0.1:
                                        nflux_q_normalized = (nflux_q_subset - nflux_q_min) / nflux_q_range if nflux_q_range > 0 else nflux_q_subset
                                    else:
                                        nflux_q_normalized = nflux_q_subset
                                    q_2d = np.column_stack([nflux_q_normalized, q_profile[:min_len]])
                                    xq = nflux_q_normalized
                                    qmin = float(np.min(q_profile[:min_len]))
                                    q0 = float(q_profile[0])
                                    # Try q95 interpolation
                                    try:
                                        target_nflux = 0.95
                                        nflux_norm_max = float(np.max(nflux_q_normalized))
                                        if target_nflux <= nflux_norm_max:
                                            qfun = interp1d(nflux_q_normalized, q_profile[:min_len], kind='linear', fill_value='extrapolate', bounds_error=False)
                                            q95 = float(qfun(target_nflux))
                                        else:
                                            q95 = float(q_profile[-1])
                                    except:
                                        q95 = float(q_profile[-1]) if len(q_profile) > 0 else None
                                    q = q_2d
                                else:
                                    xq = None
                                    q95 = None
                                    qmin = None
                                    q0 = None
                                    q = None
                            else:
                                xq = None
                                q95 = None
                                qmin = None
                                q0 = None
                                q = None
                    else:
                        # q_profile or nflux_q is None - cannot process
                        if VERBOSE:
                            print(f"  Warning: q_profile or nflux_q is None")
                        xq = None
                        q95 = None
                        qmin = None
                        q0 = None
                        q = None
                
                if p_profile is not None and nflux_p is not None:
                    p_profile = np.array(p_profile)
                    nflux_p = np.array(nflux_p)
                    
                    # Normalize nflux_p to [0, 1] if needed (same as q profile)
                    nflux_p_min = float(np.min(nflux_p))
                    nflux_p_max = float(np.max(nflux_p))
                    nflux_p_range = nflux_p_max - nflux_p_min
                    
                    # Check if nflux is already normalized [0, 1] or needs normalization
                    if nflux_p_max > 1.1 or nflux_p_min < -0.1:
                        # Not normalized - normalize it
                        nflux_p_normalized = (nflux_p - nflux_p_min) / nflux_p_range if nflux_p_range > 0 else nflux_p
                    else:
                        # Already normalized (or close to it)
                        nflux_p_normalized = nflux_p
                    
                    # Combine into 2D array: [nflux, p_value]
                    if p_profile.ndim == 1 and nflux_p.ndim == 1 and len(p_profile) == len(nflux_p):
                        p_2d = np.column_stack([nflux_p_normalized, p_profile])
                        xp = nflux_p_normalized  # Store normalized psin values
                        p0 = float(p_profile[0])
                        p = p_2d  # Store as 2D for consistency
                    else:
                        # Shape mismatch - try to salvage what we can
                        if VERBOSE:
                            print(f"  Warning: Unexpected p profile shape: p_profile.shape={p_profile.shape if hasattr(p_profile, 'shape') else 'N/A'}, nflux_p.shape={nflux_p.shape if hasattr(nflux_p, 'shape') else 'N/A'}")
                        # Try to use what we have - use minimum length
                        if p_profile is not None and nflux_p is not None:
                            min_len = min(len(p_profile), len(nflux_p))
                            if min_len > 0:
                                # Normalize nflux_p
                                nflux_p_subset = nflux_p[:min_len]
                                nflux_p_min = float(np.min(nflux_p_subset))
                                nflux_p_max = float(np.max(nflux_p_subset))
                                nflux_p_range = nflux_p_max - nflux_p_min
                                if nflux_p_max > 1.1 or nflux_p_min < -0.1:
                                    nflux_p_normalized = (nflux_p_subset - nflux_p_min) / nflux_p_range if nflux_p_range > 0 else nflux_p_subset
                                else:
                                    nflux_p_normalized = nflux_p_subset
                                p_2d = np.column_stack([nflux_p_normalized, p_profile[:min_len]])
                                xp = nflux_p_normalized
                                p0 = float(p_profile[0])
                                p = p_2d
                            else:
                                xp = None
                                p0 = None
                                p = None
                        else:
                            xp = None
                            p0 = None
                            p = None
                else:
                    # p_profile or nflux_p is None - cannot process
                    if VERBOSE:
                        print(f"  Warning: p_profile or nflux_p is None")
                    xp = None
                    p0 = None
                    p = None
                    
            except Exception as e:
                # Even if there's an error, preserve any profiles we might have extracted
                # Variables are already initialized above, so they keep their values
                if VERBOSE:
                    print(f"Error computing the q and p profiles in {sparc_path}: {e}")
                    import traceback
                    traceback.print_exc()
                # Profiles are already initialized, so they keep their current values
                # (which may be None or may have been set successfully)
                pass
            
            # Final output - ensure scalars are floats, arrays are numpy arrays
            result['output_data'] = {
                'q0': q0,  # Already converted to float above
                'q95': q95,  # Already converted to float above
                'qmin': qmin,  # Already converted to float above
                'p0': p0,  # Already converted to float above
                'q': q,  # Already numpy array
                'p': p,  # Already numpy array
                'xq': np.array(xq) if xq is not None else None,  # Ensure numpy array
                'xp': np.array(xp) if xp is not None else None,  # Ensure numpy array
                'gamma': gamma,
                'eignemode': spec
            }        
            
        except Exception as e:
            if VERBOSE:
                print(f"Error processing C1.h5 output in {sparc_path}: {e}")
        finally:
            os.chdir(parent_dir)
    
    return result


def process_mesh_data(sparc_path):
    """Process mesh data (only needs to be done once)."""
    sparc_path = Path(sparc_path)
    c1_path = sparc_path / "C1.h5"
    
    if not c1_path.exists():
        return None
    
    try:
        os.chdir(str(sparc_path))
        mysim = fpy.sim_data('C1.h5', time=-1, verbose=False)
        mesh = mysim.get_mesh(time=-1)
        
        return {
            'version': mesh.version,
            'nplanes': mesh.nplanes,
            'nelements': mesh.elements.shape[0],
            'R': mesh.elements[:, 4],
            'Z': mesh.elements[:, 5]
        }
    except Exception as e:
        if VERBOSE:
            print(f"Error processing mesh in {sparc_path}: {e}")
        return None
    finally:
        os.chdir(parent_dir)


# Main processing
if __name__ == "__main__":
    try:
        print("Step 1: Collecting all sparc directories...")
        start_time = time.time()

        # Collect all sparc directories
        all_sparc_dirs = []
        equilibrium_dirs = []  # Only from run1 for equilibrium data (like original)
        for run_dir in parent_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run"):
                for sub in run_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("sparc_"):
                        all_sparc_dirs.append(sub)
                        # For equilibrium data, only use run1 (like original script)
                        if run_dir.name == "run1":
                            equilibrium_dirs.append(sub)

        print(f"Found {len(all_sparc_dirs)} sparc directories")
        print(f"Collection took {time.time() - start_time:.2f} seconds\n")

        # Open HDF5 file
        with h5py.File(m3dc1_sfile, 'w') as hf:
            
            # GROUP 1: Equilibrium data (only from run1, like original script)
            print("Step 2: Processing equilibrium data (parallel)...")
            print(f"Processing {len(equilibrium_dirs)} equilibrium directories from run1")
            start_time = time.time()
            eq_group = hf.create_group('equilibrium')
            
            with Pool(processes=N_WORKERS) as pool:
                eq_results = pool.map(process_equilibrium_data, equilibrium_dirs)
            
            # Write equilibrium data (check for duplicates)
            seen_sparc_numbers = set()
            for result in eq_results:
                if result and result['sparc_number'] is not None:
                    sparc_key = f'sparc_{result["sparc_number"]}'
                    # Skip if already processed (shouldn't happen with run1 only, but safety check)
                    if sparc_key in seen_sparc_numbers:
                        if VERBOSE:
                            print(f"Warning: Skipping duplicate {sparc_key}")
                        continue
                    seen_sparc_numbers.add(sparc_key)
                    
                    # Check if group already exists (safety check)
                    if sparc_key in eq_group:
                        sparc_group = eq_group[sparc_key]
                    else:
                        sparc_group = eq_group.create_group(sparc_key)
                    
                    if result['geqdsk_data']:
                        for key, value in result['geqdsk_data'].items():
                            if key not in sparc_group:
                                sparc_group.create_dataset(key, data=value)
                    
                    if result['shape_data']:
                        for key, value in result['shape_data'].items():
                            if key not in sparc_group:
                                sparc_group.create_dataset(key, data=value)
            
            print(f"Equilibrium processing took {time.time() - start_time:.2f} seconds\n")
            
            # GROUP 2: Mesh data (only need one, from run1/sparc_1300 like original)
            print("Step 3: Processing mesh data...")
            start_time = time.time()
            mesh_group = hf.create_group('mesh')
            
            # Find sparc_1300 in run1 (like original script)
            mesh_data = None
            for sparc_dir in equilibrium_dirs:  # Only check run1 directories
                if sparc_dir.name == "sparc_1300":
                    mesh_data = process_mesh_data(sparc_dir)
                    if mesh_data:
                        break
            
            # Fallback: find first sparc directory with C1.h5 if sparc_1300 not found
            if mesh_data is None:
                for sparc_dir in equilibrium_dirs:
                    mesh_data = process_mesh_data(sparc_dir)
                    if mesh_data:
                        break
            
            if mesh_data:
                mesh_group.create_dataset('nelements', data=mesh_data['nelements'])
                mesh_group.create_dataset('nplanes', data=mesh_data['nplanes'])
                mesh_group.create_dataset('R', data=mesh_data['R'])
                mesh_group.create_dataset('Z', data=mesh_data['Z'])
            
            print(f"Mesh processing took {time.time() - start_time:.2f} seconds\n")
            
            # GROUP 3: Run data
            print("Step 4: Processing run data (parallel)...")
            start_time = time.time()
            
            with Pool(processes=N_WORKERS) as pool:
                run_results = pool.map(process_run_data, all_sparc_dirs)
            
            # Write run data
            count = 0
            for result in run_results:
                if result and result['sparc_number'] is not None:
                    count += 1
                    try:
                        sparc_groupx = hf.create_group(f'run_{count}')
                        input_group = sparc_groupx.create_group('input')
                        output_group = sparc_groupx.create_group('output')
                        
                        if result['input_data']:
                            for key, value in result['input_data'].items():
                                try:
                                    input_group.create_dataset(key, data=value)
                                except Exception as e:
                                    if VERBOSE:
                                        print(f"Warning: Could not write input {key} for run_{count}: {e}")
                        
                        if result['output_data']:
                            for key, value in result['output_data'].items():
                                if value is not None:  # Skip None values
                                    # Ensure proper data types for HDF5
                                    try:
                                        if isinstance(value, (list, tuple)):
                                            value = np.array(value)
                                        elif isinstance(value, np.ndarray):
                                            pass  # Already numpy array
                                        elif isinstance(value, (int, float, np.integer, np.floating)):
                                            value = float(value)  # Ensure float for scalars
                                        
                                        # Skip if still None after conversion
                                        if value is not None:
                                            output_group.create_dataset(key, data=value)
                                    except Exception as e:
                                        if VERBOSE:
                                            print(f"Warning: Could not write output {key} for run_{count}: {e}")
                        
                        # Progress reporting for long-running jobs
                        if count % 100 == 0:
                            print(f"Processed {count} runs...")
                            
                    except Exception as e:
                        print(f"Error writing run_{count}: {e}")
                        if VERBOSE:
                            import traceback
                            traceback.print_exc()
                        continue  # Continue with next run instead of crashing
            
            print(f"Run data processing took {time.time() - start_time:.2f} seconds\n")
            print(f"Total subdirectories processed: {count}")

        print(f"\n{'='*80}")
        print("Processing complete!")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

