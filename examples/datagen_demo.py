"""Example demonstrating the DataGenerator usage (dry-run).

Run this file directly to see how cases would be generated without calling
the external HotPlasmaAI scripts.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from surge.datagen import DataGenerator


def main():
    templates_dir = os.path.join(os.path.dirname(__file__), 'datagen', 'templates')

    # ---------- M3DC1 batch using C1input ----------
    gen1 = DataGenerator(dry_run=False, use_python_replacement=True)
    res1 = gen1.generate(
        inpnames=['pscale', 'batemanscale', 'ntor'],
        inputfilename='C1input',
        ranges=[[0.7, 1.1], [0.95, 1.2], [1, 15]],
        integer_mask=[False, False, True],
        n_samples=10,
        method='lhs',
        base_case_dir=templates_dir,
        seed=123,
        batch_root=os.path.join(os.path.dirname(__file__), 'datagen'),
        confirm_dirs=False,
        save_plots=True,
    )

    print('\nM3DC1 batch: generated cases:')
    for r in res1:
        print(r)

    # Plots saved by generator (save_plots=True)

    # ---------- Petra-M batch using global_ns.py ----------
    # Assumed ranges (adjust as needed):
    # nemax [1e18, 5e18], temax [5, 100], freq [3e7, 8e7], dner [1e19, 8e19], dter [100, 800]
    gen2 = DataGenerator(dry_run=False, use_python_replacement=True)
    res2 = gen2.generate(
        inpnames=['nemax', 'temax', 'freq', 'dner', 'dter'],
        inputfilename='global_ns.py',
        ranges=[[1e18, 5e18], [5, 100], [3e7, 8e7], [1e19, 8e19], [100, 800]],
        integer_mask=[False, False, False, False, False],
        n_samples=10,
        method='lhs',
        base_case_dir=templates_dir,
        seed=321,
        batch_root=os.path.join(os.path.dirname(__file__), 'datagen'),
        confirm_dirs=False,
        save_plots=True,
    )

    print('\nPetra-M batch: generated cases:')
    for r in res2:
        print(r)

    # Plots saved by generator (save_plots=True)


if __name__ == '__main__':
    main()
