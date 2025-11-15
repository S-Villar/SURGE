# Analysis of HDF5 Files: time_000.h5, time_001.h5, and C1.h5

## File Overview

Based on examination of `/pscratch/sd/a/asvillar/mp288/jobs/batch_16/run12/sparc_1429/`:

| File | Size | Purpose |
|------|------|---------|
| `time_000.h5` | 72 MB | Full 3D field data at timestep 0 |
| `time_001.h5` | 72 MB | Full 3D field data at timestep 1 |
| `C1.h5` | 628 KB | Summary/aggregated data with multiple timesteps |
| `equilibrium.h5` | 81 MB | Equilibrium data (profiles, geometry) |

## Key Findings

### 1. **C1.h5 - Multiple Timesteps Stored**

Yes, **C1.h5 contains multiple timesteps**. Evidence:

- Code uses `slice=-1` when reading profiles (gets last timestep):
  ```python
  nflux, q_profile = flux_average('q', slice=-1, file=str(h5_file), ...)
  ```

- Code uses `time='last'` when reading gamma:
  ```python
  mysim = fpy.sim_data('C1.h5', time='last', verbose=False)
  ```

- Code checks for time-dependent data in `scalars` and `time_traces` groups:
  ```python
  if 'scalars' in f:
      # In M3DC1, time traces are often stored in scalars group
      traces = f['scalars']
  ```

### 2. **time_000.h5 and time_001.h5 - Individual Timestep Snapshots**

These files are **much larger (72 MB each)** compared to C1.h5 (628 KB), suggesting they contain:
- **Full 3D field data** (magnetic fields, velocity, pressure, etc.)
- **Spatial mesh data** for that specific timestep
- **Complete state** needed to restart/continue simulation

**Note**: The current codebase (`scripts/m3dc1/collect_from_batch.py`) does **NOT** directly read from `time_*.h5` files. It only reads from:
- `C1.h5` (for gamma, profiles)
- `equilibrium.h5` (for equilibrium parameters)
- `sdata.h5` (for aggregated dataset)

### 3. **C1.h5 Structure (from code analysis)**

Based on the code, C1.h5 likely contains:

**Groups:**
- `scalars/` - Time-dependent scalar quantities (gamma, growth_rate, etc.)
- `time_traces/` - Time series data (alternative location)
- `q_profile` or `flux_coordinates/q` - Safety factor profile
- `psin` or `flux_coordinates/psi_norm` - Normalized flux coordinate
- `equilibrium/qpsi` - Equilibrium q profile
- `equilibrium/pres` - Pressure profile

**Data Access Pattern:**
- Profiles are read with `slice=-1` (last timestep)
- Gamma is read from last timestep: `data[-1]` or `time='last'`
- Multiple timesteps are stored but only the last one is typically used

### 4. **What Gets Saved?**

**Currently saved to sdata.h5:**
- ✅ Last timestep profiles (q95, q0, p0) from C1.h5
- ✅ Last timestep gamma from C1.h5
- ✅ Equilibrium parameters from equilibrium.h5
- ❌ NOT saving: time_*.h5 data (full 3D fields)

**Recommendation:**
- If you need time evolution, you could read multiple timesteps from C1.h5
- If you need full 3D fields, you'd need to read from time_*.h5 files
- Current approach (last timestep only) is appropriate for steady-state/linear stability analysis

## Questions to Answer

1. **Do we need to save multiple timesteps?**
   - For linear stability: No (only final gamma matters)
   - For nonlinear evolution: Yes (need time series)

2. **Should we read from time_*.h5 files?**
   - Only if you need full 3D field data
   - Current approach (C1.h5) is sufficient for profile extraction

3. **Is C1.h5 storing all timesteps or just summaries?**
   - Based on size (628 KB vs 72 MB per timestep), C1.h5 likely stores:
     - Time-averaged or time-series summaries
     - Profiles at multiple timesteps (but not full 3D fields)
   - Full 3D data is in time_*.h5 files

## Next Steps

To fully understand the structure, you could:
1. Use Python with h5py to examine the actual structure
2. Check M3DC1 documentation for file format
3. Read a few timesteps from C1.h5 to see what's available
4. Compare data in time_000.h5 vs C1.h5 to understand differences

