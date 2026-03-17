#!/usr/bin/env python3
"""
Plot complex eigenmode fields: real, imaginary, and flux-averaged amplitude profiles.

Creates a 2x3 subplot layout:
Row 1: delta p - Real part, Imaginary part, Amplitude profile vs psi_N
Row 2: delta B - Real part, Imaginary part, Amplitude profile vs psi_N
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


def get_complex_field_data(sparc_dir, field_name, time='last', resolution=400):
    """
    Get complex field data using m3dc1 functions.
    
    Parameters:
    -----------
    sparc_dir : Path
        Path to sparc_* directory
    field_name : str
        Field name ('p' or 'B')
    time : str or int
        Time slice
    resolution : int
        Resolution for flux averaging
    
    Returns:
    --------
    dict with keys:
        - 'field_real': 2D array of real part (R, Z)
        - 'field_imag': 2D array of imaginary part (R, Z)
        - 'psi_N': normalized flux coordinates
        - 'amplitude': flux-averaged amplitude profile
        - 'is_complex': bool
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        
        # Get field object from fpy
        print(f"  Getting {field_name} field data...")
        
        # Get field object directly from fpy
        p_field = mysim.get_field(field_name, time=time)
        
        # Evaluate field on a 2D grid (R, Z plane at phi=0)
        # Get mesh bounds
        try:
            # Try to get mesh information
            mesh_data = mysim.get_mesh()
            if hasattr(mesh_data, 'R'):
                R_min, R_max = np.min(mesh_data.R), np.max(mesh_data.R)
                Z_min, Z_max = np.min(mesh_data.Z), np.max(mesh_data.Z)
            else:
                # Default bounds
                R_min, R_max = 1.5, 2.2
                Z_min, Z_max = -0.5, 0.5
        except:
            # Default bounds
            R_min, R_max = 1.5, 2.2
            Z_min, Z_max = -0.5, 0.5
        
        # Create evaluation grid
        nR, nZ = 200, 200
        R_grid = np.linspace(R_min, R_max, nR)
        Z_grid = np.linspace(Z_min, Z_max, nZ)
        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)
        phi_val = 0.0  # Toroidal angle
        
        # Evaluate field on grid
        print(f"    Evaluating field on {nR}x{nZ} grid...")
        field_values = np.zeros((nZ, nR), dtype=complex)
        
        for i in range(nZ):
            for j in range(nR):
                try:
                    val = p_field.evaluate((R_mesh[i, j], phi_val, Z_mesh[i, j]))
                    if isinstance(val, (tuple, list)):
                        val = val[0]
                    field_values[i, j] = val
                except:
                    field_values[i, j] = np.nan
        
        # Check if complex
        is_complex = np.iscomplexobj(field_values) and np.any(np.abs(np.imag(field_values)) > 1e-15)
        
        print(f"    Field data shape: {field_values.shape}")
        print(f"    Is complex: {is_complex}")
        
        if is_complex:
            field_real = np.real(field_values)
            field_imag = np.imag(field_values)
        else:
            field_real = np.real(field_values)  # Will be same as field_values if real
            field_imag = np.zeros_like(field_real)
        
        # Get flux-averaged amplitude profile
        print(f"  Computing flux-averaged amplitude profile...")
        try:
            nflux, field_flux = m1.flux_average(
                field_name,
                sim=mysim,
                device='sparc',
                fcoords='pest',
                points=resolution
            )
            nflux = np.array(nflux)
            field_flux = np.array(field_flux)
            
            # Normalize flux coordinate
            nflux_max = np.max(nflux)
            if nflux_max > 1.1:
                psi_N = nflux / nflux_max
            else:
                psi_N = nflux
            
            # Get amplitude (magnitude if complex, or absolute value if real)
            if np.iscomplexobj(field_flux):
                amplitude = np.abs(field_flux)
            else:
                amplitude = np.abs(field_flux)
            
            # Filter to psi_N <= 0.995
            mask = psi_N <= 0.995
            psi_N_filtered = psi_N[mask]
            amplitude_filtered = amplitude[mask]
            
        except Exception as e:
            print(f"    Error computing flux average: {e}")
            psi_N_filtered = None
            amplitude_filtered = None
        
        os.chdir(cwd)
        
        return {
            'field_real': field_real,
            'field_imag': field_imag,
            'psi_N': psi_N_filtered,
            'amplitude': amplitude_filtered,
            'is_complex': is_complex
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error getting field data: {e}")


def plot_complex_eigenmode(sparc_dir, output_file=None, time='last', resolution=400):
    """
    Plot complex eigenmode fields.
    
    Creates 2x3 subplot:
    Row 1: delta p - Real, Imaginary, Amplitude profile
    Row 2: delta B - Real, Imaginary, Amplitude profile
    """
    print("=" * 80)
    print("Plotting Complex Eigenmode Fields")
    print("=" * 80)
    print(f"\nCase: {sparc_dir}")
    
    # Get delta p data
    print("\nExtracting delta p field...")
    try:
        p_data = get_complex_field_data(sparc_dir, 'p', time=time, resolution=resolution)
    except Exception as e:
        print(f"Error extracting p field: {e}")
        import traceback
        traceback.print_exc()
        p_data = None
    
    # Get delta B data
    print("\nExtracting delta B field...")
    try:
        B_data = get_complex_field_data(sparc_dir, 'B', time=time, resolution=resolution)
    except Exception as e:
        print(f"Error extracting B field: {e}")
        import traceback
        traceback.print_exc()
        B_data = None
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Delta p
    if p_data is not None:
        # Real part
        ax = axes[0, 0]
        if p_data['field_real'].ndim == 2:
            im = ax.imshow(p_data['field_real'], aspect='auto', origin='lower', cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='Re(δp)')
        else:
            ax.plot(p_data['field_real'])
        ax.set_title('δp: Real Part', fontsize=14, fontweight='bold')
        ax.set_xlabel('R index' if p_data['field_real'].ndim == 2 else 'Index')
        ax.set_ylabel('Z index' if p_data['field_real'].ndim == 2 else 'Value')
        
        # Imaginary part
        ax = axes[0, 1]
        if p_data['field_imag'].ndim == 2:
            im = ax.imshow(p_data['field_imag'], aspect='auto', origin='lower', cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='Im(δp)')
        else:
            ax.plot(p_data['field_imag'])
        ax.set_title('δp: Imaginary Part', fontsize=14, fontweight='bold')
        ax.set_xlabel('R index' if p_data['field_imag'].ndim == 2 else 'Index')
        ax.set_ylabel('Z index' if p_data['field_imag'].ndim == 2 else 'Value')
        
        # Amplitude profile
        ax = axes[0, 2]
        if p_data['psi_N'] is not None and p_data['amplitude'] is not None:
            ax.plot(p_data['psi_N'], p_data['amplitude'], 'b-', linewidth=2)
            ax.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax.set_ylabel('|δp| (amplitude)', fontsize=12)
            ax.set_title('δp: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 0.995)
        else:
            ax.text(0.5, 0.5, 'No flux-averaged data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('δp: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
    else:
        for ax in axes[0, :]:
            ax.text(0.5, 0.5, 'No delta p data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Row 2: Delta B
    if B_data is not None:
        # Real part
        ax = axes[1, 0]
        if B_data['field_real'].ndim == 2:
            im = ax.imshow(B_data['field_real'], aspect='auto', origin='lower', cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='Re(δB)')
        else:
            ax.plot(B_data['field_real'])
        ax.set_title('δB: Real Part', fontsize=14, fontweight='bold')
        ax.set_xlabel('R index' if B_data['field_real'].ndim == 2 else 'Index')
        ax.set_ylabel('Z index' if B_data['field_real'].ndim == 2 else 'Value')
        
        # Imaginary part
        ax = axes[1, 1]
        if B_data['field_imag'].ndim == 2:
            im = ax.imshow(B_data['field_imag'], aspect='auto', origin='lower', cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='Im(δB)')
        else:
            ax.plot(B_data['field_imag'])
        ax.set_title('δB: Imaginary Part', fontsize=14, fontweight='bold')
        ax.set_xlabel('R index' if B_data['field_imag'].ndim == 2 else 'Index')
        ax.set_ylabel('Z index' if B_data['field_imag'].ndim == 2 else 'Value')
        
        # Amplitude profile
        ax = axes[1, 2]
        if B_data['psi_N'] is not None and B_data['amplitude'] is not None:
            ax.plot(B_data['psi_N'], B_data['amplitude'], 'r-', linewidth=2)
            ax.set_xlabel('ψ_N (normalized flux)', fontsize=12)
            ax.set_ylabel('|δB| (amplitude)', fontsize=12)
            ax.set_title('δB: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 0.995)
        else:
            ax.text(0.5, 0.5, 'No flux-averaged data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('δB: Flux-Averaged Amplitude', fontsize=14, fontweight='bold')
    else:
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, 'No delta B data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, axes


def main():
    """Main function."""
    test_case = Path("/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run12/sparc_1429")
    
    if not test_case.exists():
        print(f"Error: Test case not found: {test_case}")
        sys.exit(1)
    
    output_plot = Path("eigenmode_complex_plot.png")
    
    try:
        plot_complex_eigenmode(test_case, output_file=output_plot, time='last', resolution=400)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Plot complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

