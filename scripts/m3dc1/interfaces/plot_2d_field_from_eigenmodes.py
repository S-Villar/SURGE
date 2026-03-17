#!/usr/bin/env python3
"""
Reconstruct and plot 2D field (R-Z plane) by summing all poloidal modes.

This script:
1. Extracts eigenmode amplitudes using eigenfunction
2. Reconstructs the 2D spatial field by summing over all modes
3. Plots the field in the R-Z plane
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


def get_2d_field_from_hdf5(sparc_dir, field_name='p', time_slice=1, phi=0.0):
    """
    Get 2D field (R-Z plane) from HDF5 by summing over all Fourier modes.
    
    Parameters:
    -----------
    sparc_dir : Path
        Path to sparc_* directory
    field_name : str
        Field name ('p' for pressure)
    time_slice : int
        Time slice number
    phi : float
        Toroidal angle (default 0.0)
    
    Returns:
    --------
    dict with:
        - 'R': R coordinates
        - 'Z': Z coordinates  
        - 'field_real': Real part of field
        - 'field_imag': Imaginary part of field
        - 'field_mag': Magnitude of field
    """
    c1_h5_file = sparc_dir / "C1.h5"
    if not c1_h5_file.exists():
        raise FileNotFoundError(f"C1.h5 not found in {sparc_dir}")
    
    cwd = os.getcwd()
    os.chdir(str(sparc_dir))
    
    try:
        # Use fpy to get field and evaluate on a regular grid
        mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
        p_field = mysim.get_field(field_name, time='last')
        
        # Create a regular R-Z grid for evaluation
        # First, get approximate bounds by evaluating at many points
        R_test = np.linspace(1.0, 3.0, 30)
        Z_test = np.linspace(-1.0, 1.0, 30)
        
        # Find valid region by testing all points
        valid_R = []
        valid_Z = []
        for R_val in R_test:
            for Z_val in Z_test:
                try:
                    val_tuple = p_field.evaluate((R_val, 0.0, Z_val))
                    # evaluate() returns a tuple, extract first element
                    if isinstance(val_tuple, (tuple, list)) and len(val_tuple) > 0:
                        val = val_tuple[0]
                    else:
                        val = val_tuple
                    
                    if val is not None:
                        if isinstance(val, (int, float, complex, np.number)):
                            if not (np.isnan(val) or np.isinf(val)):
                                valid_R.append(R_val)
                                valid_Z.append(Z_val)
                        else:
                            valid_R.append(R_val)
                            valid_Z.append(Z_val)
                except:
                    continue
        
        if len(valid_R) == 0:
            # Use default bounds for SPARC
            R_min, R_max = 1.5, 2.2
            Z_min, Z_max = -0.5, 0.5
            print(f"    Warning: No valid points found, using default bounds")
        else:
            R_min, R_max = min(valid_R), max(valid_R)
            Z_min, Z_max = min(valid_Z), max(valid_Z)
            # Add small margin
            R_margin = (R_max - R_min) * 0.05
            Z_margin = (Z_max - Z_min) * 0.05
            R_min -= R_margin
            R_max += R_margin
            Z_min -= Z_margin
            Z_max += Z_margin
        
        # Create evaluation grid
        nR, nZ = 200, 200
        R_grid = np.linspace(R_min, R_max, nR)
        Z_grid = np.linspace(Z_min, Z_max, nZ)
        R_mesh, Z_mesh = np.meshgrid(R_grid, Z_grid)
        
        print(f"    Evaluating field on {nR}x{nZ} grid...")
        print(f"    R range: [{R_min:.4f}, {R_max:.4f}] m")
        print(f"    Z range: [{Z_min:.4f}, {Z_max:.4f}] m")
        
        # Evaluate field on grid
        field_real_2d = np.zeros((nZ, nR))
        field_imag_2d = np.zeros((nZ, nR))
        
        print(f"    Progress: ", end="", flush=True)
        for i in range(nZ):
            if i % 20 == 0:
                print(f"{i}/{nZ}...", end="", flush=True)
            for j in range(nR):
                try:
                    val_tuple = p_field.evaluate((R_mesh[i, j], 0.0, Z_mesh[i, j]))
                    # evaluate() returns a tuple
                    # For scalar fields (like p): tuple with 1 element
                    # For vector fields (like B): tuple with 3 elements (B_R, B_phi, B_Z)
                    if isinstance(val_tuple, (tuple, list)):
                        if len(val_tuple) == 1:
                            # Scalar field
                            val = val_tuple[0]
                            # Check if valid number
                            if isinstance(val, (int, float, np.number)):
                                if not (np.isnan(val) or np.isinf(val)):
                                    if np.iscomplexobj(val):
                                        field_real_2d[i, j] = np.real(val)
                                        field_imag_2d[i, j] = np.imag(val)
                                    else:
                                        field_real_2d[i, j] = float(val)
                                        field_imag_2d[i, j] = 0.0
                                else:
                                    field_real_2d[i, j] = np.nan
                                    field_imag_2d[i, j] = np.nan
                            else:
                                field_real_2d[i, j] = np.nan
                                field_imag_2d[i, j] = np.nan
                        elif len(val_tuple) == 3:
                            # Vector field - compute magnitude
                            B_R, B_phi, B_Z = val_tuple[0], val_tuple[1], val_tuple[2]
                            # Compute magnitude: |B| = sqrt(B_R^2 + B_phi^2 + B_Z^2)
                            if all(isinstance(v, (int, float, np.number)) and not (np.isnan(v) or np.isinf(v)) 
                                   for v in [B_R, B_phi, B_Z]):
                                B_mag = np.sqrt(B_R**2 + B_phi**2 + B_Z**2)
                                field_real_2d[i, j] = B_mag
                                field_imag_2d[i, j] = 0.0
                            else:
                                field_real_2d[i, j] = np.nan
                                field_imag_2d[i, j] = np.nan
                        else:
                            field_real_2d[i, j] = np.nan
                            field_imag_2d[i, j] = np.nan
                    else:
                        # Not a tuple, treat as scalar
                        val = val_tuple
                        if isinstance(val, (int, float, np.number)):
                            if not (np.isnan(val) or np.isinf(val)):
                                if np.iscomplexobj(val):
                                    field_real_2d[i, j] = np.real(val)
                                    field_imag_2d[i, j] = np.imag(val)
                                else:
                                    field_real_2d[i, j] = float(val)
                                    field_imag_2d[i, j] = 0.0
                            else:
                                field_real_2d[i, j] = np.nan
                                field_imag_2d[i, j] = np.nan
                        else:
                            field_real_2d[i, j] = np.nan
                            field_imag_2d[i, j] = np.nan
                except Exception:
                    field_real_2d[i, j] = np.nan
                    field_imag_2d[i, j] = np.nan
        print(" done!")
        
        field_mag_2d = np.sqrt(field_real_2d**2 + field_imag_2d**2)
        
        # Check how many valid points we have
        valid_mask = ~np.isnan(field_real_2d)
        n_valid = np.sum(valid_mask)
        print(f"    Valid points: {n_valid}/{nR*nZ} ({100*n_valid/(nR*nZ):.1f}%)")
        if n_valid > 0:
            valid_vals = field_mag_2d[valid_mask]
            print(f"    Field magnitude range: [{np.min(valid_vals):.6e}, {np.max(valid_vals):.6e}]")
        else:
            print(f"    Warning: No valid field values found!")
        
        os.chdir(cwd)
        
        return {
            'R': R_mesh,
            'Z': Z_mesh,
            'field_real': field_real_2d,
            'field_imag': field_imag_2d,
            'field_mag': field_mag_2d,
            'field_complex': field_real_2d + 1j * field_imag_2d
        }
        
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Error getting 2D field: {e}")


def plot_2d_field(field_data, field_name='p', output_file=None):
    """
    Plot 2D field in R-Z plane.
    
    Creates plots showing:
    1. Real part
    2. Imaginary part
    3. Magnitude
    """
    R_mesh = field_data['R']
    Z_mesh = field_data['Z']
    field_real_2d = field_data['field_real']
    field_imag_2d = field_data['field_imag']
    field_mag_2d = field_data['field_mag']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Real part
    ax = axes[0]
    im = ax.contourf(R_mesh, Z_mesh, field_real_2d, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Re(δ' + field_name + ')')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δ' + field_name + ': Real Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 2: Imaginary part
    ax = axes[1]
    im = ax.contourf(R_mesh, Z_mesh, field_imag_2d, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Im(δ' + field_name + ')')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δ' + field_name + ': Imaginary Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 3: Magnitude
    ax = axes[2]
    im = ax.contourf(R_mesh, Z_mesh, field_mag_2d, levels=50, cmap='viridis', extend='both')
    plt.colorbar(im, ax=ax, label='|δ' + field_name + '|')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δ' + field_name + ': Magnitude (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def plot_2d_fields_combined(field_data_p, field_data_B, output_file=None):
    """
    Plot both pressure and magnetic field in R-Z plane.
    
    Creates a 2x3 grid:
    Top row: δp (real, imag, magnitude)
    Bottom row: δB (real, imag, magnitude)
    """
    R_mesh_p = field_data_p['R']
    Z_mesh_p = field_data_p['Z']
    p_real = field_data_p['field_real']
    p_imag = field_data_p['field_imag']
    p_mag = field_data_p['field_mag']
    
    R_mesh_B = field_data_B['R']
    Z_mesh_B = field_data_B['Z']
    B_real = field_data_B['field_real']
    B_imag = field_data_B['field_imag']
    B_mag = field_data_B['field_mag']
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Pressure
    # Plot 1: Real part
    ax = axes[0, 0]
    im = ax.contourf(R_mesh_p, Z_mesh_p, p_real, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Re(δp)')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δp: Real Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 2: Imaginary part
    ax = axes[0, 1]
    im = ax.contourf(R_mesh_p, Z_mesh_p, p_imag, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Im(δp)')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δp: Imaginary Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 3: Magnitude
    ax = axes[0, 2]
    im = ax.contourf(R_mesh_p, Z_mesh_p, p_mag, levels=50, cmap='viridis', extend='both')
    plt.colorbar(im, ax=ax, label='|δp|')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δp: Magnitude (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Bottom row: Magnetic field
    # Plot 4: Real part
    ax = axes[1, 0]
    im = ax.contourf(R_mesh_B, Z_mesh_B, B_real, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Re(δB)')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δB: Real Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 5: Imaginary part
    ax = axes[1, 1]
    im = ax.contourf(R_mesh_B, Z_mesh_B, B_imag, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(im, ax=ax, label='Im(δB)')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δB: Imaginary Part (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Plot 6: Magnitude
    ax = axes[1, 2]
    im = ax.contourf(R_mesh_B, Z_mesh_B, B_mag, levels=50, cmap='viridis', extend='both')
    plt.colorbar(im, ax=ax, label='|δB|')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title('δB: Magnitude (sum over modes)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Combined plot saved to: {output_file}")
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
    print("Reconstructing 2D Field by Summing All Poloidal Modes")
    print("=" * 80)
    print(f"\nTest case: {test_case}")
    
    # Get 2D field from HDF5 (summing over modes)
    print("\n" + "=" * 80)
    print("Extracting 2D field from HDF5 (summing over all modes)")
    print("=" * 80)
    
    # Test with pressure
    print("\nField: δp (pressure)")
    field_data_p = None
    try:
        field_data_p = get_2d_field_from_hdf5(test_case, field_name='p', time_slice=1, phi=0.0)
        print(f"✅ Got 2D pressure field data")
        print(f"   Grid size: {field_data_p['R'].shape}")
        print(f"   R range: [{np.min(field_data_p['R']):.4f}, {np.max(field_data_p['R']):.4f}] m")
        print(f"   Z range: [{np.min(field_data_p['Z']):.4f}, {np.max(field_data_p['Z']):.4f}] m")
        valid_mask = ~np.isnan(field_data_p['field_mag'])
        if np.any(valid_mask):
            valid_vals = field_data_p['field_mag'][valid_mask]
            print(f"   Field magnitude range: [{np.min(valid_vals):.6e}, {np.max(valid_vals):.6e}]")
            print(f"   Valid points: {np.sum(valid_mask)}/{field_data_p['field_mag'].size}")
        else:
            print(f"   Warning: All field values are NaN!")
        
    except Exception as e:
        print(f"❌ Error getting pressure field: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with magnetic field
    print("\nField: δB (magnetic field)")
    field_data_B = None
    try:
        field_data_B = get_2d_field_from_hdf5(test_case, field_name='B', time_slice=1, phi=0.0)
        print(f"✅ Got 2D magnetic field data")
        print(f"   Grid size: {field_data_B['R'].shape}")
        print(f"   R range: [{np.min(field_data_B['R']):.4f}, {np.max(field_data_B['R']):.4f}] m")
        print(f"   Z range: [{np.min(field_data_B['Z']):.4f}, {np.max(field_data_B['Z']):.4f}] m")
        valid_mask = ~np.isnan(field_data_B['field_mag'])
        if np.any(valid_mask):
            valid_vals = field_data_B['field_mag'][valid_mask]
            print(f"   Field magnitude range: [{np.min(valid_vals):.6e}, {np.max(valid_vals):.6e}]")
            print(f"   Valid points: {np.sum(valid_mask)}/{field_data_B['field_mag'].size}")
        else:
            print(f"   Warning: All field values are NaN!")
        
    except Exception as e:
        print(f"❌ Error getting magnetic field: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot combined
    if field_data_p is not None and field_data_B is not None:
        output_plot_combined = Path("2d_field_pressure_and_B_summed_modes.png")
        plot_2d_fields_combined(field_data_p, field_data_B, output_file=output_plot_combined)
    elif field_data_p is not None:
        output_plot_p = Path("2d_field_pressure_summed_modes.png")
        plot_2d_field(field_data_p, field_name='p', output_file=output_plot_p)
    elif field_data_B is not None:
        output_plot_B = Path("2d_field_B_summed_modes.png")
        plot_2d_field(field_data_B, field_name='B', output_file=output_plot_B)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print("\nThe 2D field is reconstructed by:")
    print("  1. Reading field data from HDF5 (P and P_i for each Fourier mode)")
    print("  2. Summing over all Fourier modes: field_total = sum_m field[m]")
    print("  3. Interpolating to regular R-Z grid for visualization")
    print("  4. Plotting real part, imaginary part, and magnitude")


if __name__ == "__main__":
    main()

