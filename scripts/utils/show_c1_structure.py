#!/usr/bin/env python3
"""
Show the structure of C1.h5 file to understand timestep organization.
Uses the same import pattern as collect_m3dc1_from_batch.py
"""

import sys
from pathlib import Path

# Try to import h5py (same as collect_m3dc1_from_batch.py)
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)

def examine_c1_structure(c1_file_path):
    """Examine C1.h5 structure and identify timestep data."""
    c1_path = Path(c1_file_path)
    
    if not c1_path.exists():
        print(f"ERROR: File not found: {c1_path}")
        return
    
    print("="*80)
    print(f"EXAMINING: {c1_path.name}")
    print("="*80)
    
    with h5py.File(c1_path, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}\n")
        
        # Check scalars group (common for time-dependent data)
        if 'scalars' in f:
            print("[GROUP] scalars/")
            scalars = f['scalars']
            print(f"  Keys: {list(scalars.keys())}\n")
            
            for key in scalars.keys():
                obj = scalars[key]
                if isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    dtype = obj.dtype
                    print(f"  [DATASET] {key}:")
                    print(f"    shape: {shape}")
                    print(f"    dtype: {dtype}")
                    
                    # Check if it looks like time series
                    if len(shape) == 1 and shape[0] > 1:
                        print(f"    -> 1D array with {shape[0]} elements (likely timesteps)")
                        try:
                            sample = obj[:min(5, shape[0])]
                            print(f"    -> First few values: {sample}")
                        except:
                            pass
                    elif len(shape) > 1:
                        print(f"    -> Multi-dimensional: {shape}")
                        if shape[-1] > 1:
                            print(f"    -> Last dimension has {shape[-1]} elements (might be timesteps)")
                    print()
        
        # Check time_traces group
        if 'time_traces' in f:
            print("[GROUP] time_traces/")
            traces = f['time_traces']
            print(f"  Keys: {list(traces.keys())}\n")
            
            for key in traces.keys():
                obj = traces[key]
                if isinstance(obj, h5py.Dataset):
                    print(f"  [DATASET] {key}: shape={obj.shape}, dtype={obj.dtype}")
                    if len(obj.shape) > 0:
                        print(f"    -> Has {obj.shape[0]} timesteps!")
                    print()
        
        # Check for other groups that might contain timestep data
        print("\n[OTHER GROUPS]:")
        for key in f.keys():
            if key not in ['scalars', 'time_traces']:
                obj = f[key]
                if isinstance(obj, h5py.Group):
                    print(f"  [GROUP] {key}/")
                    print(f"    Keys: {list(obj.keys())[:10]}{'...' if len(obj.keys()) > 10 else ''}")
                    
                    # Check if any datasets in this group have time dimension
                    for subkey in list(obj.keys())[:5]:
                        subobj = obj[subkey]
                        if isinstance(subobj, h5py.Dataset):
                            if len(subobj.shape) > 1 and subobj.shape[-1] > 1:
                                print(f"      {subkey}: shape={subobj.shape} (possible timesteps)")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  [DATASET] {key}: shape={obj.shape}, dtype={obj.dtype}")
                    if len(obj.shape) > 1:
                        print(f"    -> Multi-dimensional, last dim: {obj.shape[-1]}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        has_timesteps = False
        timestep_count = None
        
        if 'scalars' in f:
            scalars = f['scalars']
            for key in scalars.keys():
                obj = scalars[key]
                if isinstance(obj, h5py.Dataset) and len(obj.shape) > 0:
                    if len(obj.shape) == 1 and obj.shape[0] > 1:
                        has_timesteps = True
                        if timestep_count is None:
                            timestep_count = obj.shape[0]
                        elif timestep_count != obj.shape[0]:
                            print(f"  WARNING: Different timestep counts found!")
                    elif len(obj.shape) > 1 and obj.shape[-1] > 1:
                        has_timesteps = True
                        if timestep_count is None:
                            timestep_count = obj.shape[-1]
        
        if has_timesteps:
            print(f"\n✓ C1.h5 CONTAINS MULTIPLE TIMESTEPS")
            print(f"  Estimated number of timesteps: {timestep_count}")
            print(f"  Data is stored in 'scalars' group")
        else:
            print(f"\n? Could not definitively determine timestep structure")
            print(f"  Check the groups above for time-dependent data")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_c1_structure.py <path_to_C1.h5>")
        print("\nExample:")
        print("  python show_c1_structure.py /path/to/C1.h5")
        sys.exit(1)
    
    examine_c1_structure(sys.argv[1])

