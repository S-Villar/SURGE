#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examine the structure of time_000.h5, time_001.h5, and C1.h5 files
to understand what they contain and whether multiple timesteps are saved.
"""

import sys
from pathlib import Path
import h5py
import numpy as np

def print_h5_structure(file_path, max_depth=3, indent=0):
    """Recursively print HDF5 file structure."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n{'  ' * indent}[FILE] {file_path.name}")
            print(f"{'  ' * indent}{'=' * 60}")
            
            def print_group(name, obj, depth=0):
                if depth > max_depth:
                    return
                
                indent_str = '  ' * (indent + depth + 1)
                
                if isinstance(obj, h5py.Group):
                    print(f"{indent_str}[GROUP] {name}/")
                    # Print attributes if any
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            try:
                                print(f"{indent_str}  [ATTR] {attr_name}: {attr_value}")
                            except:
                                print(f"{indent_str}  [ATTR] {attr_name}: <unprintable>")
                elif isinstance(obj, h5py.Dataset):
                    shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
                    dtype_str = f"dtype={obj.dtype}"
                    print(f"{indent_str}[DATASET] {name}: {shape_str}, {dtype_str}")
                    
                    # Print attributes if any
                    if obj.attrs:
                        for attr_name, attr_value in obj.attrs.items():
                            try:
                                print(f"{indent_str}    [ATTR] {attr_name}: {attr_value}")
                            except:
                                print(f"{indent_str}    [ATTR] {attr_name}: <unprintable>")
                    
                    # Show sample values for small arrays
                    if obj.size > 0 and obj.size <= 20:
                        try:
                            sample = obj[:]
                            print(f"{indent_str}    [SAMPLE] {sample}")
                        except:
                            pass
                    elif obj.size > 20:
                        try:
                            sample = obj[:min(5, obj.size)]
                            print(f"{indent_str}    [SAMPLE] (first {len(sample)}): {sample}")
                        except:
                            pass
            
            f.visititems(print_group)
            
            # Print file-level attributes
            if f.attrs:
                print(f"\n{'  ' * indent}[FILE ATTRS]:")
                for attr_name, attr_value in f.attrs.items():
                    try:
                        print(f"{'  ' * indent}  {attr_name}: {attr_value}")
                    except:
                        print(f"{'  ' * indent}  {attr_name}: <unprintable>")
                        
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] Error reading {file_path}: {e}")
        import traceback
        traceback.print_exc()

def examine_timestep_files(base_dir):
    """Examine time_*.h5 and C1.h5 files in a directory."""
    base_path = Path(base_dir)
    
    print("=" * 80)
    print("EXAMINING HDF5 FILES")
    print("=" * 80)
    
    # Look for time_*.h5 files
    time_files = sorted(base_path.glob("time_*.h5"))
    if time_files:
        print(f"\n[INFO] Found {len(time_files)} time_*.h5 files:")
        for tf in time_files[:5]:  # Show first 5
            print(f"   - {tf.name}")
        if len(time_files) > 5:
            print(f"   ... and {len(time_files) - 5} more")
        
        # Examine first few time files
        for tf in time_files[:3]:
            print_h5_structure(tf, max_depth=2)
    else:
        print("\n[WARN] No time_*.h5 files found")
    
    # Look for C1.h5
    c1_file = base_path / "C1.h5"
    if c1_file.exists():
        print("\n" + "=" * 80)
        print("EXAMINING C1.h5")
        print("=" * 80)
        print_h5_structure(c1_file, max_depth=3)
    else:
        print("\n[WARN] C1.h5 not found")
    
    # Check for equilibrium.h5
    eq_file = base_path / "equilibrium.h5"
    if eq_file.exists():
        print("\n" + "=" * 80)
        print("EXAMINING equilibrium.h5")
        print("=" * 80)
        print_h5_structure(eq_file, max_depth=3)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examine_h5_files.py <directory_path>")
        print("\nExample:")
        print("  python examine_h5_files.py /path/to/batch_16/run_1/sparc_1")
        sys.exit(1)
    
    directory = sys.argv[1]
    examine_timestep_files(directory)

