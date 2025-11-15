#!/usr/bin/env python3
"""
Fix all C1input files to ensure ntor values are written as integers (e.g., 2 instead of 2.0).
"""

import os
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

def fix_ntor_in_file(filepath):
    """Fix ntor value in a single C1input file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return False
    
    try:
        # Read the file
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern to match ntor assignments: ntor = 2.0 or ntor = 2
        # Match various formats: ntor = 2.0, ntor=2.0, ntor = 2, etc.
        pattern = r'(ntor\s*=\s*)(\d+\.0+)(\s|$|!|\t)'
        
        def replace_ntor(match):
            prefix = match.group(1)
            value = float(match.group(2))
            suffix = match.group(3)
            # Convert to integer
            return f"{prefix}{int(value)}{suffix}"
        
        new_content = re.sub(pattern, replace_ntor, content)
        
        # Only write if content changed
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def find_all_c1input_files(root_dir):
    """Find all C1input files in the directory tree."""
    root = Path(root_dir)
    c1input_files = []
    for run_dir in root.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run"):
            for sparc_dir in run_dir.iterdir():
                if sparc_dir.is_dir() and sparc_dir.name.startswith("sparc_"):
                    c1input_path = sparc_dir / "C1input"
                    if c1input_path.exists():
                        c1input_files.append(c1input_path)
    return c1input_files

if __name__ == "__main__":
    root_dir = Path(".").resolve()
    print(f"Finding all C1input files in {root_dir}...")
    
    c1input_files = find_all_c1input_files(root_dir)
    print(f"Found {len(c1input_files)} C1input files")
    
    # Use multiprocessing to fix files in parallel
    n_workers = min(cpu_count(), 32)
    print(f"Using {n_workers} workers to fix files...")
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(fix_ntor_in_file, c1input_files)
    
    fixed_count = sum(results)
    print(f"\nFixed {fixed_count} files out of {len(c1input_files)}")
    print("Done!")

