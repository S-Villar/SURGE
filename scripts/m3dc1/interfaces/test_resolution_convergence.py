#!/usr/bin/env python3
"""
Test script to check if delta B is well resolved with 200 points or needs more resolution.

This script:
1. Runs flux_average with different resolutions (100, 200, 400, 800 points)
2. Interpolates all to the same psi_N grid for comparison
3. Compares the results to check convergence
4. Plots all resolutions together
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# Fix scipy compatibility
import scipy.integrate
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = np.trapz
if not hasattr(scipy.integrate, 'cumtrapz'):
    def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=0):
        return np.cumsum(np.trapz(y, x=x, dx=dx, axis=axis)) + initial
    scipy.integrate.cumtrapz = cumtrapz

# Add fusion-io to path
fusion_io_lib = Path("/global/homes/a/asvillar/.local/fusion-io/lib")
if fusion_io_lib.exists():
    sys.path.insert(0, str(fusion_io_lib))

# Import M3DC1 modules
try:
    import fpy
    print("✅ Successfully imported fpy")
except ImportError as e:
    print(f"❌ Error importing fpy: {e}")
    sys.exit(1)

# Try to import m3dc1
M1_AVAILABLE = False
m1 = None
m3dc1_src = Path("/global/homes/a/asvillar/src/M3DC1/unstructured/python")
if m3dc1_src.exists():
    sys.path.insert(0, str(m3dc1_src))
    try:
        import m3dc1 as m1
        M1_AVAILABLE = True
        print(f"✅ Successfully imported m3dc1 from {m3dc1_src}")
    except ImportError:
        pass

if not M1_AVAILABLE:
    print("❌ Error: m3dc1 module is required")
    sys.exit(1)


def compute_flux_average_at_resolution(sparc_dir, field_name, resolution, time='last'):
    """
    Compute flux average at a specific resolution.
    
    Parameters:
    -----------
    sparc_dir : Path
        Path to sparc_* directory
    field_name : str
        Field name ('B' or 'p')
    resolution : int
        Number of points for flux average (points parameter)
    time : str or int
        Time slice
    
    Returns:
    --------
    tuple: (psi_N, field_values) both as numpy arrays
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        
        # Use flux_average with points parameter to control resolution
        nflux, field_data = m1.flux_average(
            field_name, 
            sim=mysim, 
            device='sparc', 
            fcoords='pest',
            points=resolution  # Control resolution here
        )
        
        nflux = np.array(nflux)
        field_data = np.array(field_data)
        
        # Normalize flux coordinate
        nflux_max = np.max(nflux)
        if nflux_max > 1.1:
            psi_N = nflux / nflux_max
        else:
            psi_N = nflux
        
        os.chdir(cwd)
        return psi_N, field_data
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error computing flux average: {e}")


def test_multiple_resolutions(sparc_dir, field_name='B', resolutions=[100, 200, 400, 800]):
    """
    Test flux_average at multiple resolutions using the points parameter.
    """
    print(f"\nTesting {field_name} field at different resolutions...")
    print(f"Resolutions to test: {resolutions}")
    print(f"Note: Using flux_average 'points' parameter to control resolution")
    
    results = {}
    
    # Compute at each resolution directly using flux_average
    for res in resolutions:
        print(f"\n  Computing at {res} points resolution...")
        try:
            psi_N, field_data = compute_flux_average_at_resolution(
                sparc_dir, field_name, resolution=res, time='last'
            )
            
            # Filter to psi_N <= 0.995 as requested
            mask = psi_N <= 0.995
            psi_N_filtered = psi_N[mask]
            field_filtered = field_data[mask]
            
            # Check if field is complex
            is_complex = np.iscomplexobj(field_filtered)
            if is_complex:
                field_mag = np.abs(field_filtered)
                field_phase = np.angle(field_filtered)
                results[res] = {
                    'psi_N': psi_N_filtered,
                    'field': field_filtered,  # Keep complex
                    'field_mag': field_mag,   # Magnitude
                    'field_phase': field_phase, # Phase
                    'n_points_raw': len(psi_N),
                    'is_complex': True
                }
                print(f"    ✅ Got {len(psi_N_filtered)} points (filtered to psi_N <= 0.995)")
                print(f"    ⚠️  Field is COMPLEX")
                print(f"    psi_N range: [{np.min(psi_N_filtered):.6f}, {np.max(psi_N_filtered):.6f}]")
                print(f"    Magnitude range: [{np.min(field_mag):.6e}, {np.max(field_mag):.6e}]")
                print(f"    Phase range: [{np.min(field_phase):.6f}, {np.max(field_phase):.6f}] radians")
            else:
                results[res] = {
                    'psi_N': psi_N_filtered,
                    'field': field_filtered,
                    'n_points_raw': len(psi_N),
                    'is_complex': False
                }
                print(f"    ✅ Got {len(psi_N_filtered)} points (filtered to psi_N <= 0.995)")
                print(f"    Field is REAL")
                print(f"    psi_N range: [{np.min(psi_N_filtered):.6f}, {np.max(psi_N_filtered):.6f}]")
                print(f"    Field range: [{np.min(field_filtered):.6e}, {np.max(field_filtered):.6e}]")
            
        except Exception as e:
            print(f"    ⚠️  Error at {res} points: {e}")
            continue
    
    # Use the highest resolution as reference
    if results:
        max_res = max(results.keys())
        psi_N_ref = results[max_res]['psi_N']
        field_ref = results[max_res]['field']
        return results, psi_N_ref, field_ref
    else:
        raise RuntimeError("No successful resolutions computed")


def compare_resolutions(results, psi_N_target):
    """
    Compare results at different resolutions.
    
    Parameters:
    -----------
    results : dict
        Dictionary with resolution as key, containing psi_N and field arrays
    psi_N_target : array
        Target grid for comparison (finest resolution)
    
    Returns:
    --------
    dict: Comparison metrics
    """
    if len(results) < 2:
        return {}
    
    # Interpolate all to the finest target grid for comparison
    comparisons = {}
    resolutions = sorted(results.keys())
    finest_res = max(resolutions)
    
    print(f"\n{'='*80}")
    print("Resolution Comparison")
    print(f"{'='*80}")
    
    # Use finest resolution as reference
    ref_res = finest_res
    ref_data = results[ref_res]['field']
    ref_psi = results[ref_res]['psi_N']
    
    # Interpolate reference to target grid
    interp_ref = interp1d(ref_psi, ref_data, kind='linear',
                         fill_value='extrapolate', bounds_error=False)
    ref_on_target = interp_ref(psi_N_target)
    
    comparisons[ref_res] = {
        'rmse_vs_ref': 0.0,
        'max_diff_vs_ref': 0.0,
        'mean_diff_vs_ref': 0.0
    }
    
    print(f"\nReference: {ref_res} points")
    
    # Compare each resolution to reference
    for res in resolutions:
        if res == ref_res:
            continue
        
        data = results[res]['field']
        psi = results[res]['psi_N']
        
        # Interpolate to target grid
        # Handle complex fields
        if np.iscomplexobj(data):
            # Interpolate real and imaginary parts separately
            interp_real = interp1d(psi, np.real(data), kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
            interp_imag = interp1d(psi, np.imag(data), kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
            data_on_target = interp_real(psi_N_target) + 1j * interp_imag(psi_N_target)
        else:
            interp_func = interp1d(psi, data, kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
            data_on_target = interp_func(psi_N_target)
        
        # Compute differences (use magnitude for complex)
        if np.iscomplexobj(data_on_target) or np.iscomplexobj(ref_on_target):
            diff = np.abs(data_on_target) - np.abs(ref_on_target)
        else:
            diff = data_on_target - ref_on_target
        rmse = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        
        # Relative error (normalized by max absolute value of reference)
        ref_max_abs = np.max(np.abs(ref_on_target))
        rel_rmse = rmse / ref_max_abs if ref_max_abs > 0 else 0
        rel_max_diff = max_diff / ref_max_abs if ref_max_abs > 0 else 0
        
        comparisons[res] = {
            'rmse_vs_ref': rmse,
            'max_diff_vs_ref': max_diff,
            'mean_diff_vs_ref': mean_diff,
            'rel_rmse': rel_rmse,
            'rel_max_diff': rel_max_diff
        }
        
        print(f"\n{res} points vs {ref_res} points (reference):")
        print(f"  RMSE: {rmse:.6e} (relative: {rel_rmse*100:.4f}%)")
        print(f"  Max difference: {max_diff:.6e} (relative: {rel_max_diff*100:.4f}%)")
        print(f"  Mean |difference|: {mean_diff:.6e}")
    
    return comparisons


def plot_resolution_comparison(results, comparisons, field_name, output_file=None):
    """
    Plot comparison of different resolutions.
    Handles both real and complex fields.
    """
    # Check if any field is complex
    is_complex = any(data.get('is_complex', False) for data in results.values())
    
    if is_complex:
        # Plot magnitude and phase separately for complex fields
        fig = plt.figure(figsize=(16, 12))
        n_plots = 6
    else:
        fig = plt.figure(figsize=(14, 10))
        n_plots = 4
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    if is_complex:
        # Plot 1: Magnitude at all resolutions
        ax1 = plt.subplot(3, 2, 1)
        for i, (res, data) in enumerate(sorted(results.items())):
            field_to_plot = data.get('field_mag', np.abs(data['field']))
            ax1.plot(data['psi_N'], field_to_plot, 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i])
        ax1.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax1.set_ylabel(f'|δ{field_name}| (magnitude)', fontsize=12)
        ax1.set_title(f'Perturbed {field_name} Magnitude at Different Resolutions', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.995)
        
        # Plot 2: Phase at all resolutions
        ax2 = plt.subplot(3, 2, 2)
        for i, (res, data) in enumerate(sorted(results.items())):
            phase_to_plot = data.get('field_phase', np.angle(data['field']))
            ax2.plot(data['psi_N'], phase_to_plot, 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i])
        ax2.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax2.set_ylabel(f'arg(δ{field_name}) (phase)', fontsize=12)
        ax2.set_title(f'Perturbed {field_name} Phase at Different Resolutions', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 0.995)
        
        # Plot 3: Real part
        ax3 = plt.subplot(3, 2, 3)
        for i, (res, data) in enumerate(sorted(results.items())):
            real_part = np.real(data['field'])
            ax3.plot(data['psi_N'], real_part, 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i])
        ax3.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax3.set_ylabel(f'Re(δ{field_name})', fontsize=12)
        ax3.set_title(f'Real Part of δ{field_name}', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 0.995)
        
        # Plot 4: Imaginary part
        ax4 = plt.subplot(3, 2, 4)
        for i, (res, data) in enumerate(sorted(results.items())):
            imag_part = np.imag(data['field'])
            ax4.plot(data['psi_N'], imag_part, 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i])
        ax4.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax4.set_ylabel(f'Im(δ{field_name})', fontsize=12)
        ax4.set_title(f'Imaginary Part of δ{field_name}', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 0.995)
        
        # Plot 5: Magnitude differences from finest
        ax5 = plt.subplot(3, 2, 5)
        ref_res = max(results.keys())
        ref_data = results[ref_res]
        ref_mag = ref_data.get('field_mag', np.abs(ref_data['field']))
        for res, data in sorted(results.items()):
            if res == ref_res:
                continue
            mag = data.get('field_mag', np.abs(data['field']))
            interp_func = interp1d(data['psi_N'], mag, kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
            mag_interp = interp_func(ref_data['psi_N'])
            diff = mag_interp - ref_mag
            ax5.plot(ref_data['psi_N'], diff, 
                    label=f'{res} - {ref_res}', linewidth=2, alpha=0.7)
        ax5.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax5.set_ylabel(f'|δ{field_name}| difference', fontsize=12)
        ax5.set_title(f'Magnitude Difference from {ref_res} points', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlim(0, 0.995)
        
        # Plot 6: Error metrics
        ax6 = plt.subplot(3, 2, 6)
    else:
        # Plot 1: All resolutions overlaid
        ax1 = plt.subplot(2, 2, 1)
        for i, (res, data) in enumerate(sorted(results.items())):
            ax1.plot(data['psi_N'], data['field'], 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i])
        ax1.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax1.set_ylabel(f'δ{field_name}', fontsize=12)
        ax1.set_title(f'Perturbed {field_name} at Different Resolutions', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.995)
        
        # Plot 2: Differences from finest resolution
        ax2 = plt.subplot(2, 2, 2)
        ref_res = max(results.keys())
        ref_data = results[ref_res]
        for res, data in sorted(results.items()):
            if res == ref_res:
                continue
            interp_func = interp1d(data['psi_N'], data['field'], kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
            data_interp = interp_func(ref_data['psi_N'])
            diff = data_interp - ref_data['field']
            ax2.plot(ref_data['psi_N'], diff, 
                    label=f'{res} - {ref_res}', linewidth=2, alpha=0.7)
        ax2.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax2.set_ylabel(f'Difference from {ref_res} points', fontsize=12)
        ax2.set_title('Difference from Finest Resolution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 0.995)
        
        # Plot 3: Zoom on a region
        ax3 = plt.subplot(2, 2, 3)
        for i, (res, data) in enumerate(sorted(results.items())):
            mask = (data['psi_N'] >= 0.3) & (data['psi_N'] <= 0.7)
            ax3.plot(data['psi_N'][mask], data['field'][mask], 
                    label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i], marker='o', markersize=3)
        ax3.set_xlabel('ψ_N (normalized flux)', fontsize=12)
        ax3.set_ylabel(f'δ{field_name}', fontsize=12)
        ax3.set_title('Zoom: ψ_N = 0.3 to 0.7', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error metrics
        ax4 = plt.subplot(2, 2, 4)
        ax6 = ax4  # For consistency with complex case
    
    # Error metrics plot (works for both real and complex)
    ref_res = max(results.keys())
    resolutions = sorted([r for r in comparisons.keys() if r != ref_res])
    rmse_vals = [comparisons[r]['rmse_vs_ref'] for r in resolutions]
    max_diff_vals = [comparisons[r]['max_diff_vs_ref'] for r in resolutions]
    
    x = np.arange(len(resolutions))
    width = 0.35
    ax6.bar(x - width/2, rmse_vals, width, label='RMSE', alpha=0.7)
    ax6.bar(x + width/2, max_diff_vals, width, label='Max |Diff|', alpha=0.7)
    ax6.set_xlabel('Resolution (points)', fontsize=12)
    ax6.set_ylabel('Error vs Reference', fontsize=12)
    ax6.set_title(f'Error Metrics (vs {ref_res} points)', fontsize=14)
    ax6.set_xticks(x)
    ax6.set_xticklabels(resolutions)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # Plot 2: Differences from finest resolution
    ax2 = plt.subplot(2, 2, 2)
    ref_res = max(results.keys())
    ref_data = results[ref_res]
    for res, data in sorted(results.items()):
        if res == ref_res:
            continue
        # Interpolate to reference grid
        interp_func = interp1d(data['psi_N'], data['field'], kind='linear',
                              fill_value='extrapolate', bounds_error=False)
        data_interp = interp_func(ref_data['psi_N'])
        diff = data_interp - ref_data['field']
        ax2.plot(ref_data['psi_N'], diff, 
                label=f'{res} - {ref_res}', linewidth=2, alpha=0.7)
    ax2.set_xlabel('ψ_N (normalized flux)', fontsize=12)
    ax2.set_ylabel(f'Difference from {ref_res} points', fontsize=12)
    ax2.set_title('Difference from Finest Resolution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlim(0, 0.995)
    
    # Plot 3: Zoom on a region (middle section)
    ax3 = plt.subplot(2, 2, 3)
    for i, (res, data) in enumerate(sorted(results.items())):
        mask = (data['psi_N'] >= 0.3) & (data['psi_N'] <= 0.7)
        ax3.plot(data['psi_N'][mask], data['field'][mask], 
                label=f'{res} points', linewidth=2, alpha=0.7, color=colors[i], marker='o', markersize=3)
    ax3.set_xlabel('ψ_N (normalized flux)', fontsize=12)
    ax3.set_ylabel(f'δ{field_name}', fontsize=12)
    ax3.set_title('Zoom: ψ_N = 0.3 to 0.7', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error metrics
    ax4 = plt.subplot(2, 2, 4)
    resolutions = sorted([r for r in comparisons.keys() if r != ref_res])
    rmse_vals = [comparisons[r]['rmse_vs_ref'] for r in resolutions]
    max_diff_vals = [comparisons[r]['max_diff_vs_ref'] for r in resolutions]
    
    x = np.arange(len(resolutions))
    width = 0.35
    ax4.bar(x - width/2, rmse_vals, width, label='RMSE', alpha=0.7)
    ax4.bar(x + width/2, max_diff_vals, width, label='Max |Diff|', alpha=0.7)
    ax4.set_xlabel('Resolution (points)', fontsize=12)
    ax4.set_ylabel('Error vs Reference', fontsize=12)
    ax4.set_title(f'Error Metrics (vs {ref_res} points)', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(resolutions)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def check_complex_field(sparc_dir, field_name):
    """Check if a field is complex."""
    print(f"\n{'='*80}")
    print(f"Checking if {field_name} field is complex")
    print(f"{'='*80}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
        
        # Get field at high resolution to check
        nflux, field_data = m1.flux_average(
            field_name, 
            sim=mysim, 
            device='sparc', 
            fcoords='pest',
            points=200
        )
        
        field_data = np.array(field_data)
        
        is_complex = np.iscomplexobj(field_data)
        has_imag = np.any(np.abs(np.imag(field_data)) > 1e-15) if is_complex else False
        
        print(f"\nField: {field_name}")
        print(f"  Data type: {field_data.dtype}")
        print(f"  Is complex: {is_complex}")
        print(f"  Has non-zero imaginary part: {has_imag}")
        
        if is_complex:
            print(f"  Real part range: [{np.min(np.real(field_data)):.6e}, {np.max(np.real(field_data)):.6e}]")
            print(f"  Imaginary part range: [{np.min(np.imag(field_data)):.6e}, {np.max(np.imag(field_data)):.6e}]")
            print(f"  Magnitude range: [{np.min(np.abs(field_data)):.6e}, {np.max(np.abs(field_data)):.6e}]")
            print(f"  Phase range: [{np.min(np.angle(field_data)):.6f}, {np.max(np.angle(field_data)):.6f}] radians")
        else:
            print(f"  Value range: [{np.min(field_data):.6e}, {np.max(field_data):.6e}]")
        
        os.chdir(cwd)
        return is_complex, field_data
        
    except Exception as e:
        os.chdir(cwd)
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function."""
    test_case = Path("/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run12/sparc_1429")
    
    if not test_case.exists():
        print(f"Error: Test case not found: {test_case}")
        sys.exit(1)
    
    # First check if fields are complex
    print("=" * 80)
    print("Checking Field Types (Complex vs Real)")
    print("=" * 80)
    
    is_B_complex, B_sample = check_complex_field(test_case, 'B')
    is_p_complex, p_sample = check_complex_field(test_case, 'p')
    
    # Test resolutions
    resolutions = [100, 200, 400, 800]
    
    # Test delta B
    print(f"\n{'='*80}")
    print("Resolution Convergence Test for Delta B")
    print("=" * 80)
    print(f"\nTest case: {test_case}")
    
    try:
        results_B, psi_N_ref_B, field_ref_B = test_multiple_resolutions(
            test_case, field_name='B', resolutions=resolutions
        )
    except Exception as e:
        print(f"\nError testing B: {e}")
        import traceback
        traceback.print_exc()
        results_B = {}
    
    # Test delta p
    print(f"\n{'='*80}")
    print("Resolution Convergence Test for Delta P")
    print("=" * 80)
    
    try:
        results_p, psi_N_ref_p, field_ref_p = test_multiple_resolutions(
            test_case, field_name='p', resolutions=resolutions
        )
    except Exception as e:
        print(f"\nError testing p: {e}")
        import traceback
        traceback.print_exc()
        results_p = {}
    
    # Create target grids for comparison
    if results_B:
        psi_N_target_B = np.linspace(0, 0.995, max(resolutions))
        comparisons_B = compare_resolutions(results_B, psi_N_target_B)
        output_plot_B = Path("test_resolution_convergence_B.png")
        plot_resolution_comparison(results_B, comparisons_B, 'B', output_file=output_plot_B)
    
    if results_p:
        psi_N_target_p = np.linspace(0, 0.995, max(resolutions))
        comparisons_p = compare_resolutions(results_p, psi_N_target_p)
        output_plot_p = Path("test_resolution_convergence_p.png")
        plot_resolution_comparison(results_p, comparisons_p, 'p', output_file=output_plot_p)
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    print(f"\nField Type Check:")
    print(f"  Delta B is complex: {is_B_complex}")
    print(f"  Delta p is complex: {is_p_complex}")
    
    if results_B:
        print(f"\nDelta B Convergence:")
        print(f"  Tested resolutions: {sorted(results_B.keys())}")
        if comparisons_B:
            for res in sorted(comparisons_B.keys()):
                if res == max(results_B.keys()):
                    continue
                comp = comparisons_B[res]
                print(f"    {res} points: RMSE = {comp['rmse_vs_ref']:.6e} "
                      f"({comp['rel_rmse']*100:.4f}% relative)")
    
    if results_p:
        print(f"\nDelta p Convergence:")
        print(f"  Tested resolutions: {sorted(results_p.keys())}")
        if comparisons_p:
            for res in sorted(comparisons_p.keys()):
                if res == max(results_p.keys()):
                    continue
                comp = comparisons_p[res]
                print(f"    {res} points: RMSE = {comp['rmse_vs_ref']:.6e} "
                      f"({comp['rel_rmse']*100:.4f}% relative)")
    
    print(f"\n{'='*80}")
    print("Test complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

