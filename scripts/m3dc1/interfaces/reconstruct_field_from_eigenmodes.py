#!/usr/bin/env python3
"""
Reconstruct total field from eigenmode amplitudes and structures.

This script:
1. Extracts eigenmode data using eigenfunction
2. Reconstructs the total field from eigenmode amplitudes
3. Compares with direct field access from HDF5
4. Plots the reconstructed field vs direct field
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

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


def get_eigenmode_amplitudes(sparc_dir, field_name='p', time='last', points=400):
    """
    Get eigenmode amplitudes from eigenfunction.
    
    Returns:
    --------
    dict with:
        - 'amplitudes': (n_modes, n_flux) array of mode amplitudes
        - 'psi_N': normalized flux coordinates
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        
        eigen_data = m1.eigenfunction(
            field=field_name,
            sim=mysim,
            fcoords='pest',
            device='sparc',
            time=time,
            points=points,
            makeplot=False,
            quiet=True,
            fourier=True
        )
        
        eigen_array = np.array(eigen_data)
        
        # Create flux coordinates
        n_flux = eigen_array.shape[1]
        psi_N = np.linspace(0, 1, n_flux)
        mask = psi_N <= 0.995
        psi_N_filtered = psi_N[mask]
        eigen_filtered = eigen_array[:, mask]
        
        os.chdir(cwd)
        
        return {
            'amplitudes': eigen_filtered,
            'psi_N': psi_N_filtered,
            'n_modes': eigen_array.shape[0],
            'n_flux': len(psi_N_filtered)
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error getting eigenmode amplitudes: {e}")


def get_direct_field_from_hdf5(sparc_dir, field_name='p', time_slice=1):
    """
    Get direct field data from HDF5 file.
    
    For pressure: uses 'P' and 'P_i' (real and imaginary parts)
    For magnetic field: would need to check field names
    
    Returns:
    --------
    dict with:
        - 'field_complex': complex field at mesh points
        - 'field_real': real part
        - 'field_imag': imaginary part
        - 'mesh_points': number of mesh points
        - 'n_modes': number of Fourier modes
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    time_key = f'time_{time_slice:03d}'
    
    with h5py.File(c1_h5_file, 'r') as f:
        if time_key not in f or 'fields' not in f[time_key]:
            raise ValueError(f"Time slice {time_key} not found in HDF5")
        
        fields = f[time_key]['fields']
        
        # Map field names
        field_map = {
            'p': ('P', 'P_i'),
            'P': ('P', 'P_i'),
            'pressure': ('P', 'P_i'),
        }
        
        if field_name not in field_map:
            raise ValueError(f"Field {field_name} not in field map. Available: {list(fields.keys())}")
        
        real_name, imag_name = field_map[field_name]
        
        if real_name not in fields or imag_name not in fields:
            raise ValueError(f"Fields {real_name} or {imag_name} not found")
        
        field_real = fields[real_name][:]
        field_imag = fields[imag_name][:]
        
        # Construct complex field
        field_complex = field_real + 1j * field_imag
        
        return {
            'field_complex': field_complex,
            'field_real': field_real,
            'field_imag': field_imag,
            'mesh_points': field_complex.shape[0],
            'n_modes': field_complex.shape[1]
        }


def reconstruct_flux_averaged_field(eigenmode_data):
    """
    Reconstruct flux-averaged field from eigenmode amplitudes.
    
    The eigenfunction gives mode amplitudes as a function of flux.
    To get the total field, we sum over modes (taking magnitude or real part).
    
    Parameters:
    -----------
    eigenmode_data : dict
        Output from get_eigenmode_amplitudes
    
    Returns:
    --------
    dict with reconstructed field profiles
    """
    amplitudes = eigenmode_data['amplitudes']
    psi_N = eigenmode_data['psi_N']
    
    # The eigenfunction amplitudes are mode coefficients
    # Sum over modes to get total field amplitude
    # For each flux surface, sum the mode contributions
    field_total = np.sum(np.abs(amplitudes), axis=0)  # Sum over modes
    
    # Or use RMS: sqrt(sum of squares)
    field_rms = np.sqrt(np.sum(amplitudes**2, axis=0))
    
    # Or use maximum mode (dominant mode)
    dominant_mode_idx = np.argmax(np.max(np.abs(amplitudes), axis=1))
    field_dominant = amplitudes[dominant_mode_idx, :]
    
    return {
        'psi_N': psi_N,
        'field_total': field_total,
        'field_rms': field_rms,
        'field_dominant': field_dominant,
        'dominant_mode': dominant_mode_idx
    }


def plot_reconstruction_comparison(eigenmode_data, hdf5_data, reconstructed, field_name, output_file=None):
    """
    Plot comparison of reconstructed field vs direct field access.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Eigenmode analysis
    amplitudes = eigenmode_data['amplitudes']
    psi_N = eigenmode_data['psi_N']
    
    # Plot 1: Mode amplitudes heatmap
    ax = axes[0, 0]
    im = ax.imshow(np.abs(amplitudes), aspect='auto', origin='lower', 
                   cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='|Amplitude|')
    ax.set_xlabel('Flux surface index', fontsize=12)
    ax.set_ylabel('Mode number (m)', fontsize=12)
    ax.set_title(f'{field_name}: Eigenmode Amplitudes', fontsize=14, fontweight='bold')
    
    # Plot 2: Mode spectrum (amplitude vs mode number)
    ax = axes[0, 1]
    mode_amplitudes = np.max(np.abs(amplitudes), axis=1)  # Max over flux
    ax.plot(mode_amplitudes, 'o-', linewidth=2, markersize=4)
    ax.set_xlabel('Mode number (m)', fontsize=12)
    ax.set_ylabel('Max amplitude', fontsize=12)
    ax.set_title('Mode Spectrum', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reconstructed flux-averaged profile
    ax = axes[0, 2]
    ax.plot(reconstructed['psi_N'], reconstructed['field_rms'], 'b-', 
           linewidth=2, label='RMS (sqrt(sum squares))')
    ax.plot(reconstructed['psi_N'], reconstructed['field_total'], 'r--', 
           linewidth=2, label='Sum of |amplitudes|')
    ax.plot(reconstructed['psi_N'], np.abs(reconstructed['field_dominant']), 'g:', 
           linewidth=2, label=f'Dominant mode (m={reconstructed["dominant_mode"]})')
    ax.set_xlabel('ψ_N (normalized flux)', fontsize=12)
    ax.set_ylabel('Field amplitude', fontsize=12)
    ax.set_title('Reconstructed Flux-Averaged Field', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.995)
    
    # Row 2: Direct field from HDF5
    field_complex = hdf5_data['field_complex']
    field_real = hdf5_data['field_real']
    field_imag = hdf5_data['field_imag']
    
    # Plot 4: Real part distribution
    ax = axes[1, 0]
    # Sum over modes to get total field at each mesh point
    field_total_real = np.sum(field_real, axis=1)
    ax.hist(field_total_real, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Field value (real part)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Direct Field: Real Part Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Imaginary part distribution
    ax = axes[1, 1]
    field_total_imag = np.sum(field_imag, axis=1)
    ax.hist(field_total_imag, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('Field value (imaginary part)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Direct Field: Imaginary Part Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Magnitude distribution
    ax = axes[1, 2]
    field_total_mag = np.abs(np.sum(field_complex, axis=1))
    ax.hist(field_total_mag, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('|Field| (magnitude)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Direct Field: Magnitude Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function."""
    test_case = Path("/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run12/sparc_1429")
    
    if not test_case.exists():
        print(f"Error: Test case not found: {test_case}")
        sys.exit(1)
    
    print("=" * 80)
    print("Reconstructing Field from Eigenmode Amplitudes")
    print("=" * 80)
    print(f"\nTest case: {test_case}")
    
    # Get eigenmode amplitudes
    print("\n" + "=" * 80)
    print("Step 1: Extracting eigenmode amplitudes")
    print("=" * 80)
    try:
        eigenmode_data = get_eigenmode_amplitudes(test_case, field_name='p', time='last', points=400)
        print(f"✅ Got eigenmode amplitudes")
        print(f"   Shape: {eigenmode_data['amplitudes'].shape}")
        print(f"   Number of modes: {eigenmode_data['n_modes']}")
        print(f"   Number of flux surfaces: {eigenmode_data['n_flux']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get direct field from HDF5
    print("\n" + "=" * 80)
    print("Step 2: Getting direct field from HDF5")
    print("=" * 80)
    try:
        hdf5_data = get_direct_field_from_hdf5(test_case, field_name='p', time_slice=1)
        print(f"✅ Got direct field from HDF5")
        print(f"   Mesh points: {hdf5_data['mesh_points']}")
        print(f"   Number of modes: {hdf5_data['n_modes']}")
        print(f"   Field magnitude range: [{np.min(np.abs(hdf5_data['field_complex'])):.6e}, {np.max(np.abs(hdf5_data['field_complex'])):.6e}]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        hdf5_data = None
    
    # Reconstruct field
    print("\n" + "=" * 80)
    print("Step 3: Reconstructing flux-averaged field")
    print("=" * 80)
    try:
        reconstructed = reconstruct_flux_averaged_field(eigenmode_data)
        print(f"✅ Reconstructed field")
        print(f"   Dominant mode: {reconstructed['dominant_mode']}")
        print(f"   RMS field range: [{np.min(reconstructed['field_rms']):.6e}, {np.max(reconstructed['field_rms']):.6e}]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        reconstructed = None
    
    # Plot comparison
    if eigenmode_data and hdf5_data and reconstructed:
        output_plot = Path("field_reconstruction_plot.png")
        plot_reconstruction_comparison(eigenmode_data, hdf5_data, reconstructed, 
                                     'δp', output_file=output_plot)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Eigenmode amplitudes: {eigenmode_data['amplitudes'].shape}")
    print(f"  - Direct field modes: {hdf5_data['n_modes'] if hdf5_data else 'N/A'}")
    print(f"  - Reconstructed field: {'Success' if reconstructed else 'Failed'}")
    print(f"\nNote: The eigenfunction gives mode amplitudes as a function of flux.")
    print(f"      To reconstruct the full spatial field, we need to:")
    print(f"      1. Use the mode amplitudes from eigenfunction")
    print(f"      2. Apply the appropriate basis functions (Fourier modes)")
    print(f"      3. Sum over modes to get total field")


if __name__ == "__main__":
    main()

