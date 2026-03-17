#!/usr/bin/env python3
"""
Test script to extract and plot eigenmodes from C1.h5.

This script:
1. Extracts eigenmode data using m3dc1.eigenfunction
2. Plots the eigenmode structure
3. Shows flux-averaged profiles
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


def extract_eigenmode(sparc_dir, field_name='p', time='last', points=400):
    """
    Extract eigenmode data using m3dc1.eigenfunction.
    
    Parameters:
    -----------
    sparc_dir : Path
        Path to sparc_* directory
    field_name : str
        Field name ('p' or 'B')
    time : str or int
        Time slice
    points : int
        Number of flux surfaces
    
    Returns:
    --------
    dict with eigenmode data
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        
        print(f"  Extracting eigenfunction for field '{field_name}'...")
        
        # Get eigenfunction
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
        print(f"    Eigenfunction shape: {eigen_array.shape}")
        print(f"    Dtype: {eigen_array.dtype}")
        print(f"    Is complex: {np.iscomplexobj(eigen_array)}")
        print(f"    Min/Max: {np.min(eigen_array):.6e} / {np.max(eigen_array):.6e}")
        
        # The shape is typically (n_modes, n_flux_surfaces)
        # For Fourier modes, we might need to reconstruct the complex field
        # or the array might represent |amplitude|^2 or similar
        
        # Also get flux-averaged profile
        print(f"  Computing flux-averaged profile...")
        nflux, field_flux = m1.flux_average(
            field_name,
            sim=mysim,
            device='sparc',
            fcoords='pest',
            points=points
        )
        nflux = np.array(nflux)
        field_flux = np.array(field_flux)
        
        # Normalize flux coordinate
        nflux_max = np.max(nflux)
        if nflux_max > 1.1:
            psi_N = nflux / nflux_max
        else:
            psi_N = nflux
        
        # Filter to psi_N <= 0.995
        mask = psi_N <= 0.995
        psi_N_filtered = psi_N[mask]
        field_flux_filtered = field_flux[mask]
        
        os.chdir(cwd)
        
        return {
            'eigenmode': eigen_array,
            'psi_N': psi_N_filtered,
            'flux_averaged': field_flux_filtered,
            'n_modes': eigen_array.shape[0] if eigen_array.ndim >= 1 else 1,
            'n_flux': eigen_array.shape[1] if eigen_array.ndim >= 2 else len(psi_N_filtered)
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error extracting eigenmode: {e}")


def plot_eigenmode(eigenmode_data_p, eigenmode_data_B, output_file=None):
    """
    Plot eigenmode data.
    
    Creates plots showing:
    1. Eigenmode structure (modes vs flux surfaces)
    2. Flux-averaged profiles
    3. Mode amplitudes
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Delta p eigenmode
    if eigenmode_data_p is not None:
        eigen_p = eigenmode_data_p['eigenmode']
        psi_N_p = eigenmode_data_p['psi_N']
        flux_p = eigenmode_data_p['flux_averaged']
        
        # Plot 1: Eigenmode structure (modes vs flux)
        ax1 = plt.subplot(2, 3, 1)
        if eigen_p.ndim == 2:
            im = ax1.imshow(eigen_p, aspect='auto', origin='lower', 
                           cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax1, label='Eigenmode amplitude')
            ax1.set_xlabel('Flux surface index', fontsize=12)
            ax1.set_ylabel('Mode number (m)', fontsize=12)
            ax1.set_title('δp: Eigenmode Structure', fontsize=14, fontweight='bold')
        else:
            ax1.plot(eigen_p)
            ax1.set_xlabel('Index', fontsize=12)
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.set_title('δp: Eigenmode', fontsize=14, fontweight='bold')
        
        # Plot 2: Dominant mode profile
        ax2 = plt.subplot(2, 3, 2)
        if eigen_p.ndim == 2:
            # Find dominant mode (mode with maximum amplitude)
            mode_amplitudes = np.max(np.abs(eigen_p), axis=1)
            dominant_mode_idx = np.argmax(mode_amplitudes)
            dominant_mode = eigen_p[dominant_mode_idx, :]
            
            # Create flux coordinate for eigenmode
            n_flux = eigen_p.shape[1]
            psi_N_eigen = np.linspace(0, 0.995, n_flux)
            
            ax2.plot(psi_N_eigen, dominant_mode, 'b-', linewidth=2, 
                    label=f'Mode {dominant_mode_idx}')
            ax2.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax2.set_ylabel('Amplitude', fontsize=12)
            ax2.set_title(f'δp: Dominant Mode (m={dominant_mode_idx})', 
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '1D eigenmode data', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Flux-averaged profile
        ax3 = plt.subplot(2, 3, 3)
        if psi_N_p is not None and flux_p is not None:
            ax3.plot(psi_N_p, flux_p, 'b-', linewidth=2)
            ax3.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax3.set_ylabel('|δp| (amplitude)', fontsize=12)
            ax3.set_title('δp: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 0.995)
        else:
            ax3.text(0.5, 0.5, 'No flux-averaged data', 
                    ha='center', va='center', transform=ax3.transAxes)
    
    # Row 2: Delta B eigenmode
    if eigenmode_data_B is not None:
        eigen_B = eigenmode_data_B['eigenmode']
        psi_N_B = eigenmode_data_B['psi_N']
        flux_B = eigenmode_data_B['flux_averaged']
        
        # Plot 4: Eigenmode structure
        ax4 = plt.subplot(2, 3, 4)
        if eigen_B.ndim == 2:
            im = ax4.imshow(eigen_B, aspect='auto', origin='lower', 
                           cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax4, label='Eigenmode amplitude')
            ax4.set_xlabel('Flux surface index', fontsize=12)
            ax4.set_ylabel('Mode number (m)', fontsize=12)
            ax4.set_title('δB: Eigenmode Structure', fontsize=14, fontweight='bold')
        else:
            ax4.plot(eigen_B)
            ax4.set_xlabel('Index', fontsize=12)
            ax4.set_ylabel('Amplitude', fontsize=12)
            ax4.set_title('δB: Eigenmode', fontsize=14, fontweight='bold')
        
        # Plot 5: Dominant mode profile
        ax5 = plt.subplot(2, 3, 5)
        if eigen_B.ndim == 2:
            # Find dominant mode
            mode_amplitudes = np.max(np.abs(eigen_B), axis=1)
            dominant_mode_idx = np.argmax(mode_amplitudes)
            dominant_mode = eigen_B[dominant_mode_idx, :]
            
            # Create flux coordinate for eigenmode
            n_flux = eigen_B.shape[1]
            psi_N_eigen = np.linspace(0, 0.995, n_flux)
            
            ax5.plot(psi_N_eigen, dominant_mode, 'r-', linewidth=2, 
                    label=f'Mode {dominant_mode_idx}')
            ax5.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax5.set_ylabel('Amplitude', fontsize=12)
            ax5.set_title(f'δB: Dominant Mode (m={dominant_mode_idx})', 
                          fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, '1D eigenmode data', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # Plot 6: Flux-averaged profile
        ax6 = plt.subplot(2, 3, 6)
        if psi_N_B is not None and flux_B is not None:
            ax6.plot(psi_N_B, flux_B, 'r-', linewidth=2)
            ax6.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax6.set_ylabel('|δB| (amplitude)', fontsize=12)
            ax6.set_title('δB: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(0, 0.995)
        else:
            ax6.text(0.5, 0.5, 'No flux-averaged data', 
                    ha='center', va='center', transform=ax6.transAxes)
    
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
    print("Testing Eigenmode Extraction from C1.h5")
    print("=" * 80)
    print(f"\nTest case: {test_case}")
    
    # Extract eigenmodes
    print("\n" + "=" * 80)
    print("Extracting Delta p Eigenmode")
    print("=" * 80)
    try:
        eigenmode_p = extract_eigenmode(test_case, field_name='p', time='last', points=400)
        print(f"✅ Successfully extracted delta p eigenmode")
        print(f"   Shape: {eigenmode_p['eigenmode'].shape}")
        print(f"   Number of modes: {eigenmode_p['n_modes']}")
        print(f"   Number of flux surfaces: {eigenmode_p['n_flux']}")
    except Exception as e:
        print(f"❌ Error extracting delta p eigenmode: {e}")
        import traceback
        traceback.print_exc()
        eigenmode_p = None
    
    print("\n" + "=" * 80)
    print("Extracting Delta B Eigenmode")
    print("=" * 80)
    try:
        eigenmode_B = extract_eigenmode(test_case, field_name='B', time='last', points=400)
        print(f"✅ Successfully extracted delta B eigenmode")
        print(f"   Shape: {eigenmode_B['eigenmode'].shape}")
        print(f"   Number of modes: {eigenmode_B['n_modes']}")
        print(f"   Number of flux surfaces: {eigenmode_B['n_flux']}")
    except Exception as e:
        print(f"❌ Error extracting delta B eigenmode: {e}")
        import traceback
        traceback.print_exc()
        eigenmode_B = None
    
    # Plot results
    output_plot = Path("eigenmode_test_plot.png")
    if eigenmode_p is not None or eigenmode_B is not None:
        plot_eigenmode(eigenmode_p, eigenmode_B, output_file=output_plot)
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
