#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examine the structure of C1.h5 to see if it contains multiple timesteps.
"""

import sys
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:
    print("ERROR: h5py not available. Trying alternative methods...")
    sys.exit(1)

def print_structure(file_path, max_depth=4, current_depth=0, prefix=""):
    """Recursively print HDF5 structure."""
    try:
        with h5py.File(file_path, 'r') as f:
            if current_depth == 0:
                print(f"\n{'='*80}")
                print(f"STRUCTURE OF: {file_path.name}")
                print(f"{'='*80}\n")
            
            # Print groups and datasets at current level
            items = []
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Group):
                    items.append(('GROUP', key, obj))
                elif isinstance(obj, h5py.Dataset):
                    items.append(('DATASET', key, obj))
            
            # Sort: groups first, then datasets
            items.sort(key=lambda x: (x[0] != 'GROUP', x[1]))
            
            for item_type, key, obj in items:
                indent = "  " * current_depth
                
                if item_type == 'GROUP':
                    print(f"{prefix}{indent}[GROUP] {key}/")
                    # Print group attributes
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            try:
                                print(f"{prefix}{indent}  [ATTR] {attr_name}: {attr_value}")
                            except:
                                print(f"{prefix}{indent}  [ATTR] {attr_name}: <unprintable>")
                    
                    # Recurse if not too deep
                    if current_depth < max_depth:
                        print_structure(obj, max_depth, current_depth + 1, prefix)
                
                elif item_type == 'DATASET':
                    shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
                    dtype_str = f"dtype={obj.dtype}"
                    size_str = f"size={obj.size}"
                    
                    print(f"{prefix}{indent}[DATASET] {key}: {shape_str}, {dtype_str}, {size_str}")
                    
                    # Print dataset attributes
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            try:
                                print(f"{prefix}{indent}    [ATTR] {attr_name}: {attr_value}")
                            except:
                                print(f"{prefix}{indent}    [ATTR] {attr_name}: <unprintable>")
                    
                    # Show sample values for small arrays or first few values
                    if obj.size > 0:
                        try:
                            if obj.size <= 10:
                                sample = obj[:]
                                print(f"{prefix}{indent}    [VALUES] {sample}")
                            elif len(obj.shape) == 1 and obj.shape[0] <= 20:
                                sample = obj[:]
                                print(f"{prefix}{indent}    [VALUES] {sample}")
                            elif len(obj.shape) == 1:
                                sample = obj[:min(5, obj.shape[0])]
                                print(f"{prefix}{indent}    [VALUES] (first {len(sample)}): {sample}")
                            elif len(obj.shape) > 1:
                                # For multi-dimensional, show shape info
                                print(f"{prefix}{indent}    [INFO] Multi-dimensional array")
                                if 'time' in key.lower() or 'timestep' in key.lower():
                                    print(f"{prefix}{indent}    [INFO] This appears to be time-dependent data!")
                        except Exception as e:
                            print(f"{prefix}{indent}    [ERROR] Could not read values: {e}")
            
            # Print file-level attributes
            if current_depth == 0 and f.attrs:
                print(f"\n{prefix}[FILE ATTRIBUTES]:")
                for attr_name, attr_value in f.attrs.items():
                    try:
                        print(f"{prefix}  {attr_name}: {attr_value}")
                    except:
                        print(f"{prefix}  {attr_name}: <unprintable>")
                        
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        import traceback
        traceback.print_exc()

def check_for_timesteps(file_path):
    """Specifically check for timestep-related structures."""
    print(f"\n{'='*80}")
    print("CHECKING FOR TIMESTEP DATA")
    print(f"{'='*80}\n")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Look for common timestep indicators
            timestep_indicators = []
            
            def check_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Check shape - if last dimension varies, might be timesteps
                    if len(obj.shape) > 1:
                        shape_str = str(obj.shape)
                        if 'time' in name.lower() or 'timestep' in name.lower():
                            timestep_indicators.append((name, obj.shape, 'name_contains_time'))
                        elif obj.shape[-1] > 1:
                            timestep_indicators.append((name, obj.shape, 'last_dimension_>1'))
            
            f.visititems(check_item)
            
            # Check scalars group (common location for time traces)
            if 'scalars' in f:
                scalars = f['scalars']
                print("[FOUND] 'scalars' group - common location for time-dependent data")
                print(f"  Keys in scalars: {list(scalars.keys())}")
                for key in scalars.keys():
                    obj = scalars[key]
                    if isinstance(obj, h5py.Dataset):
                        print(f"    {key}: shape={obj.shape}, dtype={obj.dtype}")
                        if len(obj.shape) > 0 and obj.shape[0] > 1:
                            print(f"      -> This appears to have {obj.shape[0]} timesteps!")
            
            # Check time_traces group
            if 'time_traces' in f:
                traces = f['time_traces']
                print("\n[FOUND] 'time_traces' group")
                print(f"  Keys in time_traces: {list(traces.keys())}")
            
            # Check for time dimension in datasets
            if timestep_indicators:
                print(f"\n[FOUND] {len(timestep_indicators)} potential timestep datasets:")
                for name, shape, reason in timestep_indicators:
                    print(f"  - {name}: shape={shape} ({reason})")
            else:
                print("\n[INFO] No obvious timestep indicators found in dataset names")
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_c1_structure.py <path_to_C1.h5>")
        print("\nExample:")
        print("  python examine_c1_structure.py /path/to/C1.h5")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Print full structure
    print_structure(file_path, max_depth=3)
    
    # Specifically check for timesteps
    check_for_timesteps(file_path)

