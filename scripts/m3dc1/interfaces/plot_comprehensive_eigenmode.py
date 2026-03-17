#!/usr/bin/env python3
"""
Comprehensive eigenmode visualization script.

Creates a single figure with 8 plots (2 rows x 4 columns):
Row 1 (δp): Dominant mode, Spectrum, Flux-averaged amplitude, 2D field
Row 2 (δB): Dominant mode, Spectrum, Flux-averaged amplitude, 2D field
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


def extract_eigenmode_data(sparc_dir, field_name='p', time='last', resolution=400, max_modes=50):
    """
    Extract eigenmode data using m3dc1.eigenfunction.
    
    Returns:
    --------
    dict with:
        - 'eigenmode_array': 2D array (n_modes, n_flux_surfaces)
        - 'psi_N': normalized flux array
        - 'dominant_mode_idx': index of dominant mode
        - 'dominant_mode_amplitude': amplitude profile of dominant mode
        - 'flux_avg_amplitude': flux-averaged amplitude profile
    """
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        
        # Get eigenfunction
        eigen = m1.eigenfunction(
            field=field_name,
            sim=mysim,
            fcoords='pest',
            device='sparc',
            time=time,
            points=resolution,
            makeplot=False,
            quiet=True,
            fourier=True
        )
        
        eigen_array = np.array(eigen)
        
        # Truncate to max_modes
        if eigen_array.shape[0] > max_modes:
            eigen_array = eigen_array[:max_modes, :]
        
        # Get psi_N from flux average
        # We need to get psi_N for the flux surfaces
        # The eigenfunction returns amplitudes as function of flux surface index
        # We need to map this to psi_N
        flux_avg = m1.flux_average(
            field=field_name,
            sim=mysim,
            fcoords='pest',
            device='sparc',
            time=time,
            points=resolution
        )
        
        # flux_avg should give us psi_N and the profile
        if isinstance(flux_avg, (tuple, list)) and len(flux_avg) >= 2:
            psi_N = flux_avg[0]
            profile = flux_avg[1]
        else:
            # If not, we need to construct psi_N from 0 to 0.995
            psi_N = np.linspace(0, 0.995, eigen_array.shape[1])
            profile = None
        
        # Find dominant mode (mode with maximum integrated amplitude)
        mode_integrals = np.sum(np.abs(eigen_array), axis=1)
        dominant_mode_idx = np.argmax(mode_integrals)
        dominant_mode_amplitude = np.abs(eigen_array[dominant_mode_idx, :])
        
        # Compute flux-averaged amplitude: sqrt(sum over modes of |amplitude|^2)
        flux_avg_amplitude = np.sqrt(np.sum(np.abs(eigen_array)**2, axis=0))
        
        os.chdir(cwd)
        
        return {
            'eigenmode_array': eigen_array,
            'psi_N': psi_N,
            'dominant_mode_idx': dominant_mode_idx,
            'dominant_mode_amplitude': dominant_mode_amplitude,
            'flux_avg_amplitude': flux_avg_amplitude,
            'n_modes': eigen_array.shape[0],
            'n_flux_surfaces': eigen_array.shape[1]
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error extracting eigenmode data: {e}") from e


def get_2d_field_eigenmode(sparc_dir, field_name='p', time='last', nR=200, nZ=200):
    """
    Get 2D eigenmode field (R-Z plane) by evaluating field on a grid.
    The field.evaluate() method already sums over all Fourier modes,
    giving us the total eigenmode structure in 2D space.
    
    Returns:
    --------
    dict with:
        - 'R': R mesh (2D array)
        - 'Z': Z mesh (2D array)
        - 'field_mag': field magnitude (2D array)
    """
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        mysim = fpy.sim_data('C1.h5', time=time, verbose=False)
        field = mysim.get_field(field_name, time=time)
        
        # Find valid R bounds by testing evaluation
        R_test = np.linspace(1.0, 3.0, 30)
        Z_test = np.linspace(-1.3, 1.3, 30)  # Fixed Z range as requested
        
        valid_R = []
        for R_val in R_test:
            # Test at Z=0 (magnetic axis)
            try:
                val_tuple = field.evaluate((R_val, 0.0, 0.0))
                if val_tuple is not None:
                    if isinstance(val_tuple, (tuple, list)):
                        if len(val_tuple) == 1:
                            val = val_tuple[0]
                        elif len(val_tuple) == 3:
                            val = np.sqrt(val_tuple[0]**2 + val_tuple[1]**2 + val_tuple[2]**2)
                        else:
                            val = val_tuple[0] if len(val_tuple) > 0 else None
                    else:
                        val = val_tuple
                    
                    if val is not None and isinstance(val, (int, float, np.number)):
                        if not (np.isnan(val) or np.isinf(val)):
                            valid_R.append(R_val)
            except:
                continue
        
        # Set fixed Z range as requested: -1.3 to 1.3 m
        Z_min, Z_max = -1.3, 1.3
        
        if len(valid_R) == 0:
            R_min, R_max = 1.5, 2.2
        else:
            R_min, R_max = min(valid_R), max(valid_R)
            R_margin = (R_max - R_min) * 0.05
            R_min -= R_margin
            R_max += R_margin
        
        # Create grid
        R_grid = np.linspace(R_min, R_max, nR)
        Z_grid = np.linspace(Z_min, Z_max, nZ)
        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)
        
        # Evaluate field on grid (this sums over all modes automatically)
        print(f"    Evaluating field on {nR}x{nZ} grid...", end="", flush=True)
        field_mag = np.zeros((nZ, nR))
        for i in range(nZ):
            if i % 30 == 0:
                print(".", end="", flush=True)
            for j in range(nR):
                try:
                    val_tuple = field.evaluate((R_mesh[i, j], 0.0, Z_mesh[i, j]))
                    if isinstance(val_tuple, (tuple, list)):
                        if len(val_tuple) == 1:
                            val = val_tuple[0]
                        elif len(val_tuple) == 3:
                            # Vector field - compute magnitude
                            val = np.sqrt(val_tuple[0]**2 + val_tuple[1]**2 + val_tuple[2]**2)
                        else:
                            val = val_tuple[0] if len(val_tuple) > 0 else np.nan
                    else:
                        val = val_tuple
                    
                    if isinstance(val, (int, float, np.number)) and not (np.isnan(val) or np.isinf(val)):
                        field_mag[i, j] = float(val)
                    else:
                        field_mag[i, j] = np.nan
                except:
                    field_mag[i, j] = np.nan
        print(" done!")
        
        os.chdir(cwd)
        
        return {
            'R': R_mesh,
            'Z': Z_mesh,
            'field_mag': field_mag
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error getting 2D eigenmode field: {e}") from e


def plot_comprehensive_eigenmode(sparc_dir, output_file=None, max_modes=50):
    """
    Create comprehensive eigenmode visualization.
    
    Creates a 2x4 grid:
    Row 1 (δp): Dominant mode, Spectrum, Flux-averaged, 2D field
    Row 2 (δB): Dominant mode, Spectrum, Flux-averaged, 2D field
    """
    print(f"\n{'='*80}")
    print(f"Creating comprehensive eigenmode plot for: {sparc_dir}")
    print(f"{'='*80}")
    
    # Extract eigenmode data
    print("\nExtracting δp eigenmode data...")
    try:
        eigen_p = extract_eigenmode_data(sparc_dir, field_name='p', max_modes=max_modes)
        print(f"  ✅ Got {eigen_p['n_modes']} modes, {eigen_p['n_flux_surfaces']} flux surfaces")
        print(f"  Dominant mode: m={eigen_p['dominant_mode_idx']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None
    
    print("\nExtracting δB eigenmode data...")
    try:
        eigen_B = extract_eigenmode_data(sparc_dir, field_name='B', max_modes=max_modes)
        print(f"  ✅ Got {eigen_B['n_modes']} modes, {eigen_B['n_flux_surfaces']} flux surfaces")
        print(f"  Dominant mode: m={eigen_B['dominant_mode_idx']}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None
    
    # Get 2D eigenmode fields (evaluating field on grid sums over all modes)
    print("\nExtracting 2D δp eigenmode field...")
    try:
        field_2d_p = get_2d_field_eigenmode(sparc_dir, field_name='p', nR=150, nZ=150)
        print(f"  ✅ Got 2D field: {field_2d_p['field_mag'].shape}")
        valid_mask = ~np.isnan(field_2d_p['field_mag'])
        if np.any(valid_mask):
            valid_vals = field_2d_p['field_mag'][valid_mask]
            print(f"  Field range: [{np.min(valid_vals):.6e}, {np.max(valid_vals):.6e}]")
            print(f"  Valid points: {np.sum(valid_mask)}/{field_2d_p['field_mag'].size}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\nExtracting 2D δB eigenmode field...")
    try:
        field_2d_B = get_2d_field_eigenmode(sparc_dir, field_name='B', nR=150, nZ=150)
        print(f"  ✅ Got 2D field: {field_2d_B['field_mag'].shape}")
        valid_mask = ~np.isnan(field_2d_B['field_mag'])
        if np.any(valid_mask):
            valid_vals = field_2d_B['field_mag'][valid_mask]
            print(f"  Field range: [{np.min(valid_vals):.6e}, {np.max(valid_vals):.6e}]")
            print(f"  Valid points: {np.sum(valid_mask)}/{field_2d_B['field_mag'].size}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # ========== ROW 1: δp ==========
    
    # Plot 1: Dominant mode
    ax = axes[0, 0]
    ax.plot(eigen_p['psi_N'], eigen_p['dominant_mode_amplitude'], 'b-', linewidth=2, label=f'Mode {eigen_p["dominant_mode_idx"]}')
    ax.set_xlabel('Ψ_N (normalized flux)', fontsize=11)
    ax.set_ylabel('Eigenmode amplitude', fontsize=11)
    ax.set_title(f'δp: Dominant Mode (m={eigen_p["dominant_mode_idx"]})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 0.995)
    
    # Plot 2: Spectrum
    ax = axes[0, 1]
    eigen_abs = np.abs(eigen_p['eigenmode_array'])
    im = ax.imshow(eigen_abs, aspect='auto', origin='lower', cmap='viridis', 
                   extent=[0, eigen_p['n_flux_surfaces']-1, 0, eigen_p['n_modes']-1],
                   interpolation='nearest')
    ax.set_xlabel('Flux surface index', fontsize=11)
    ax.set_ylabel('Mode number (m)', fontsize=11)
    ax.set_title('δp: Eigenmode Spectrum', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Eigenmode amplitude')
    
    # Plot 3: Flux-averaged amplitude
    ax = axes[0, 2]
    ax.plot(eigen_p['psi_N'], eigen_p['flux_avg_amplitude'], 'b-', linewidth=2)
    ax.set_xlabel('Ψ_N (normalized flux)', fontsize=11)
    ax.set_ylabel('|δp| (amplitude)', fontsize=11)
    ax.set_title('δp: Flux-Averaged Amplitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.995)
    
    # Plot 4: 2D field
    ax = axes[0, 3]
    im = ax.contourf(field_2d_p['R'], field_2d_p['Z'], field_2d_p['field_mag'], 
                     levels=50, cmap='viridis', extend='both')
    ax.set_xlabel('R [m]', fontsize=11)
    ax.set_ylabel('Z [m]', fontsize=11)
    ax.set_title('δp: 2D Field (sum over modes)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_ylim(-1.3, 1.3)  # Fixed Z range
    plt.colorbar(im, ax=ax, label='|δp|')
    
    # ========== ROW 2: δB ==========
    
    # Plot 5: Dominant mode
    ax = axes[1, 0]
    ax.plot(eigen_B['psi_N'], eigen_B['dominant_mode_amplitude'], 'r-', linewidth=2, label=f'Mode {eigen_B["dominant_mode_idx"]}')
    ax.set_xlabel('Ψ_N (normalized flux)', fontsize=11)
    ax.set_ylabel('Eigenmode amplitude', fontsize=11)
    ax.set_title(f'δB: Dominant Mode (m={eigen_B["dominant_mode_idx"]})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 0.995)
    
    # Plot 6: Spectrum
    ax = axes[1, 1]
    eigen_abs = np.abs(eigen_B['eigenmode_array'])
    im = ax.imshow(eigen_abs, aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, eigen_B['n_flux_surfaces']-1, 0, eigen_B['n_modes']-1],
                   interpolation='nearest')
    ax.set_xlabel('Flux surface index', fontsize=11)
    ax.set_ylabel('Mode number (m)', fontsize=11)
    ax.set_title('δB: Eigenmode Spectrum', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Eigenmode amplitude')
    
    # Plot 7: Flux-averaged amplitude
    ax = axes[1, 2]
    ax.plot(eigen_B['psi_N'], eigen_B['flux_avg_amplitude'], 'r-', linewidth=2)
    ax.set_xlabel('Ψ_N (normalized flux)', fontsize=11)
    ax.set_ylabel('|δB| (amplitude)', fontsize=11)
    ax.set_title('δB: Flux-Averaged Amplitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.995)
    
    # Plot 8: 2D field
    ax = axes[1, 3]
    im = ax.contourf(field_2d_B['R'], field_2d_B['Z'], field_2d_B['field_mag'],
                     levels=50, cmap='viridis', extend='both')
    ax.set_xlabel('R [m]', fontsize=11)
    ax.set_ylabel('Z [m]', fontsize=11)
    ax.set_title('δB: 2D Field (sum over modes)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_ylim(-1.3, 1.3)  # Fixed Z range
    plt.colorbar(im, ax=ax, label='|δB|')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive eigenmode visualization')
    parser.add_argument('sparc_dir', type=str, help='Path to sparc_* directory')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--max-modes', type=int, default=50, help='Maximum number of modes to plot (default: 50)')
    
    args = parser.parse_args()
    
    sparc_dir = Path(args.sparc_dir)
    if not sparc_dir.exists():
        print(f"Error: Directory not found: {sparc_dir}")
        sys.exit(1)
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("comprehensive_eigenmode_plot.png")
    
    plot_comprehensive_eigenmode(sparc_dir, output_file=output_file, max_modes=args.max_modes)


if __name__ == "__main__":
    main()

