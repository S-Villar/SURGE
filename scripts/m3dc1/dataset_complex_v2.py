"""
Dataset generator and loader for per-run M3DC1 HDF5 files with runs/*/spectrum/p.

Default leaf name is ``sdata_pertfields_grid_complex_v2.h5`` (pertfields grid).
On CFS bulk trees, use ``filename="sdata_complex_v2.h5"``: same ``runs`` layout
and ``spectrum/p`` (float or complex); δp is not mirror-symmetrized in |m|.

Each run directory (run1/sparc_1300, run2/sparc_1305, etc.) contains its own
sdata_pertfields_grid_complex_v2.h5 (or sdata_complex_v2.h5) with:
- spectrum/p: delta p modes spectrum (m_modes x psi_norm, complex)
- miller: R0, a, kappa, delta (equilibrium inputs)
- parset: ntor, pscale, batemanscale (run inputs)
- flux_average: q, p profiles for q0, q95, qmin, p0
- growth_rate: gamma

This module finds all such files, extracts the right fields, and builds
a DataFrame for SURGE surrogate training.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
import pandas as pd

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# Default filename for per-run complex v2 HDF5 (pertfields grid)
COMPLEX_V2_FILENAME = "sdata_pertfields_grid_complex_v2.h5"
# Canonical bulk tree (e.g. amsc007 CFS): per-case nonsymmetric δp spectrum
SDATA_COMPLEX_V2_FILENAME = "sdata_complex_v2.h5"


def _decode(value: Any) -> Any:
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return _decode(value[()])
        except (TypeError, IndexError):
            pass
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def find_complex_v2_files(
    batch_dir: Union[str, Path],
    *,
    run_pattern: str = "run*",
    sparc_pattern: str = "sparc_*",
    filename: str = COMPLEX_V2_FILENAME,
) -> List[Path]:
    """
    Find all sdata_pertfields_grid_complex_v2.h5 files under batch_dir.

    Scans run*/sparc_*/ for the file. Returns sorted list of paths.

    Parameters
    ----------
    batch_dir : str or Path
        Root batch directory containing `run*/sparc_*/` subtrees
        (e.g. `/path/to/m3dc1/batch_16`).
    run_pattern : str
        Glob for run directories (default: run*)
    sparc_pattern : str
        Glob for sparc directories (default: sparc_*)
    filename : str
        HDF5 filename (default: sdata_pertfields_grid_complex_v2.h5)

    Returns
    -------
    list of Path
    """
    batch_dir = Path(batch_dir)
    paths = []
    for run_dir in sorted(batch_dir.glob(run_pattern)):
        if not run_dir.is_dir():
            continue
        for sparc_dir in sorted(run_dir.glob(sparc_pattern)):
            if not sparc_dir.is_dir():
                continue
            f = sparc_dir / filename
            if f.exists():
                paths.append(f)
    return paths


def load_single_complex_v2(
    file_path: Union[str, Path],
    *,
    spectrum_field: str = "p",
    spectrum_time_idx: int = -1,
    use_magnitude: bool = True,
    input_cols: Optional[List[str]] = None,
    mode_step: Optional[int] = None,
    psi_step: Optional[int] = None,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Load one sdata_pertfields_grid_complex_v2.h5 and extract inputs + spectrum.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file
    spectrum_field : str
        Spectrum group name (default: "p" for delta p)
    spectrum_time_idx : int
        Time slice index for spectrum (default: -1 = last)
    use_magnitude : bool
        If True, use |spec| for complex spectrum
    input_cols : list of str, optional
        Input columns to include. If None, uses default set.
    mode_step, psi_step : int, optional
        Downsample spectrum: take every Nth mode/psi to reduce output size.
        E.g. mode_step=4, psi_step=4 yields 50x50=2500 outputs instead of 200x200.
    target_shape : (n_modes, n_psi), optional
        If set, resample spectrum to this fixed shape so all cases have the same
        resolution. Overrides mode_step/psi_step. Requires scipy.

    Returns
    -------
    dict
        Keys: run_id, eq_id, input_*, output_p_0, output_p_1, ...
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py required. pip install h5py")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    row: Dict[str, Any] = {}
    with h5py.File(file_path, "r") as f:
        if "runs" not in f:
            raise ValueError(f"No 'runs' group in {file_path}")
        run_names = list(f["runs"].keys())
        if not run_names:
            raise ValueError(f"Empty runs in {file_path}")
        rname = run_names[0]
        rg = f["runs"][rname]

        run_id = _decode(rg.get("runID", rname))
        eq_id = _decode(rg.get("eqID", "eq"))
        row["run_id"] = run_id
        row["eq_id"] = eq_id

        # Miller (equilibrium)
        if "miller" in rg:
            for k in ["R0", "a", "kappa", "delta"]:
                if k in rg["miller"]:
                    row[f"eq_{k}"] = float(rg["miller"][k][()])

        # Parset (ntor, pscale, batemanscale)
        if "parset" in rg:
            names = rg["parset"]["names"]
            values = rg["parset"]["values"]
            for i, n in enumerate(names):
                name = _decode(n) if hasattr(n, "decode") else str(n)
                if i < len(values):
                    row[f"input_{name}"] = float(values[i])

        # Flux-averaged profiles for q0, q95, qmin, p0
        if "flux_average" in rg:
            fa = rg["flux_average"]
            if "q" in fa and "profile" in fa["q"] and "psin" in fa["q"]:
                psin = np.asarray(fa["q"]["psin"])
                qprof = np.asarray(fa["q"]["profile"])
                if len(qprof) > 0:
                    row["q0"] = float(qprof[0])
                    row["qmin"] = float(np.min(qprof))
                    if len(psin) > 1:
                        idx95 = int(0.95 * (len(psin) - 1))
                        row["q95"] = float(qprof[min(idx95, len(qprof) - 1)])
            if "p" in fa and "profile" in fa["p"]:
                pprof = np.asarray(fa["p"]["profile"])
                if len(pprof) > 0:
                    row["p0"] = float(pprof[0])

        # q95 scalar if present
        if "q95" in rg:
            row["q95"] = float(rg["q95"][()])

        # Growth rate
        if "growth_rate" in rg:
            gr0 = rg["growth_rate"].get("0")
            if gr0 is not None:
                row["gamma"] = float(gr0[()])

        # Spectrum (modes)
        if "spectrum" not in rg or spectrum_field not in rg["spectrum"]:
            raise ValueError(f"No spectrum/{spectrum_field} in {file_path}")

        sp = rg["spectrum"][spectrum_field]
        spec = np.asarray(sp["spec"])
        if np.iscomplexobj(spec) and use_magnitude:
            spec = np.abs(spec)
        if spec.ndim == 3:
            spec = spec[spectrum_time_idx]
        if target_shape is not None:
            if not SCIPY_AVAILABLE:
                raise ImportError("target_shape requires scipy. pip install scipy")
            n_m, n_p = target_shape
            if spec.shape != (n_m, n_p):
                zoom_factors = (n_m / spec.shape[0], n_p / spec.shape[1])
                spec = zoom(spec.astype(np.float64), zoom_factors, order=1)
        elif mode_step is not None or psi_step is not None:
            sl_m = slice(None, None, mode_step or 1)
            sl_p = slice(None, None, psi_step or 1)
            spec = spec[sl_m, sl_p]
        flat = spec.flatten()
        for i, v in enumerate(flat):
            row[f"output_p_{i}"] = float(v)

    return row


def load_single_complex_v2_per_mode(
    file_path: Union[str, Path],
    *,
    spectrum_field: str = "p",
    spectrum_time_idx: int = -1,
    use_magnitude: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load one file and return one row per (n, m) mode. Native resolution (no downsampling).

    Each row: inputs (eq_*, input_*, q0, q95, ..., n, m) + outputs (200 profile values).
    So the model learns: given (equilibrium, n, m), predict δp_n,m(ψ_N) - the amplitude
    profile vs psi for that mode.

    Returns
    -------
    list of dict
        One dict per m-mode. Keys: eq_*, input_*, n, m, output_p_0..output_p_199
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py required. pip install h5py")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    rows: List[Dict[str, Any]] = []
    with h5py.File(file_path, "r") as f:
        if "runs" not in f:
            raise ValueError(f"No 'runs' group in {file_path}")
        run_names = list(f["runs"].keys())
        if not run_names:
            raise ValueError(f"Empty runs in {file_path}")
        rname = run_names[0]
        rg = f["runs"][rname]

        run_id = _decode(rg.get("runID", rname))
        eq_id = _decode(rg.get("eqID", "eq"))

        base: Dict[str, Any] = {"run_id": run_id, "eq_id": eq_id}
        if "miller" in rg:
            for k in ["R0", "a", "kappa", "delta"]:
                if k in rg["miller"]:
                    base[f"eq_{k}"] = float(rg["miller"][k][()])
        if "parset" in rg:
            names = rg["parset"]["names"]
            values = rg["parset"]["values"]
            for i, n in enumerate(names):
                name = _decode(n) if hasattr(n, "decode") else str(n)
                if i < len(values):
                    base[f"input_{name}"] = float(values[i])
        if "flux_average" in rg:
            fa = rg["flux_average"]
            if "q" in fa and "profile" in fa["q"] and "psin" in fa["q"]:
                qprof = np.asarray(fa["q"]["profile"])
                psin = np.asarray(fa["q"]["psin"])
                if len(qprof) > 0:
                    base["q0"] = float(qprof[0])
                    base["qmin"] = float(np.min(qprof))
                    if len(psin) > 1:
                        idx95 = int(0.95 * (len(psin) - 1))
                        base["q95"] = float(qprof[min(idx95, len(qprof) - 1)])
            if "p" in fa and "profile" in fa["p"]:
                pprof = np.asarray(fa["p"]["profile"])
                if len(pprof) > 0:
                    base["p0"] = float(pprof[0])
        if "q95" in rg:
            base["q95"] = float(rg["q95"][()])
        if "growth_rate" in rg:
            gr0 = rg["growth_rate"].get("0")
            if gr0 is not None:
                base["gamma"] = float(gr0[()])

        sp = rg["spectrum"][spectrum_field]
        spec = np.asarray(sp["spec"])
        m_modes = np.asarray(sp["m_modes"])
        if np.iscomplexobj(spec) and use_magnitude:
            spec = np.abs(spec)
        if spec.ndim == 3:
            spec = spec[spectrum_time_idx]
        n_m, n_psi = spec.shape

        n_val = float(base.get("input_ntor", base.get("input_n", 0)))
        for m_idx in range(n_m):
            row = dict(base)
            row["n"] = n_val
            row["m"] = int(m_modes[m_idx])
            profile = spec[m_idx, :]
            for i, v in enumerate(profile):
                row[f"output_p_{i}"] = float(v)
            rows.append(row)

    return rows


def build_dataframe_per_mode(
    batch_dir: Union[str, Path],
    *,
    run_pattern: str = "run*",
    sparc_pattern: str = "sparc_*",
    filename: str = COMPLEX_V2_FILENAME,
    spectrum_field: str = "p",
    spectrum_time_idx: int = -1,
    use_magnitude: bool = True,
    max_cases: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build DataFrame in per-mode format: one row per (case, m).

    Each row: inputs (eq_*, input_*, n, m) + outputs (200 profile values).
    Native resolution (200 m-modes × 200 psi). Model learns: given (inputs, n, m),
    predict δp_n,m(ψ_N) - the amplitude profile for that mode.
    """
    paths = find_complex_v2_files(
        batch_dir, run_pattern=run_pattern, sparc_pattern=sparc_pattern, filename=filename
    )
    if not paths:
        raise FileNotFoundError(
            f"No {filename} found under {batch_dir}/{run_pattern}/{sparc_pattern}"
        )
    if max_cases is not None:
        paths = paths[:max_cases]
    if verbose:
        print(f"Loading {len(paths)} cases (per-mode, native resolution) from {batch_dir}")

    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows = load_single_complex_v2_per_mode(
                p,
                spectrum_field=spectrum_field,
                spectrum_time_idx=spectrum_time_idx,
                use_magnitude=use_magnitude,
            )
            all_rows.extend(rows)
        except Exception as e:
            if verbose:
                print(f"  Skip {p}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    if verbose:
        n_out = len([c for c in df.columns if c.startswith("output_p_")])
        print(f"  Loaded {len(df)} rows ({len(paths)} cases × 200 m-modes), {n_out} profile points")
    return df


# 12 inputs for per-mode: R0, a, kappa, delta, q0, q95, qmin, p0, n, m, pscale, batemanscale
PER_MODE_INPUT_COLS = [
    "eq_R0", "eq_a", "eq_kappa", "eq_delta",
    "q0", "q95", "qmin", "p0",
    "n", "m",
    "input_pscale", "input_batemanscale",
]


def load_per_mode_for_surge(
    batch_dir: Union[str, Path],
    **kwargs: Any,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load per-mode format for SURGE. 12 inputs, 200 profile outputs.
    """
    df = build_dataframe_per_mode(batch_dir, **kwargs)
    input_cols = [c for c in PER_MODE_INPUT_COLS if c in df.columns]
    output_cols = [c for c in df.columns if c.startswith("output_p_")]
    return df, input_cols, output_cols


def build_dataframe_from_batch(
    batch_dir: Union[str, Path],
    *,
    run_pattern: str = "run*",
    sparc_pattern: str = "sparc_*",
    filename: str = COMPLEX_V2_FILENAME,
    spectrum_field: str = "p",
    spectrum_time_idx: int = -1,
    use_magnitude: bool = True,
    mode_step: Optional[int] = None,
    psi_step: Optional[int] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    max_cases: Optional[int] = None,
    include_eigenmodes: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame from all per-run sdata_pertfields_grid_complex_v2.h5 files.

    Parameters
    ----------
    batch_dir : str or Path
        Root batch directory
    run_pattern, sparc_pattern, filename
        As in find_complex_v2_files
    spectrum_field : str
        Spectrum group (default: "p")
    spectrum_time_idx : int
        Time slice (default: -1)
    use_magnitude : bool
        Use magnitude for complex spectrum
    mode_step, psi_step : int, optional
        Downsample spectrum (e.g. 4 yields 50x50 instead of 200x200)
    target_shape : (n_modes, n_psi), optional
        If set, resample all cases to this fixed shape (same resolution for all)
    max_cases : int, optional
        Limit number of cases (for testing)
    include_eigenmodes : bool
        If True, extract eigenmode_amp_* from C1.h5 when m3dc1/fpy available
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        One row per case. Columns: run_id, eq_id, eq_*, input_*, q0, q95, qmin, p0, gamma, output_p_0, ...
    """
    paths = find_complex_v2_files(
        batch_dir, run_pattern=run_pattern, sparc_pattern=sparc_pattern, filename=filename
    )
    if not paths:
        raise FileNotFoundError(
            f"No {filename} found under {batch_dir}/{run_pattern}/{sparc_pattern}"
        )
    if max_cases is not None:
        paths = paths[:max_cases]
    if verbose:
        print(f"Loading {len(paths)} cases from {batch_dir}")

    rows = []
    for i, p in enumerate(paths):
        try:
            row = load_single_complex_v2(
                p,
                spectrum_field=spectrum_field,
                spectrum_time_idx=spectrum_time_idx,
                use_magnitude=use_magnitude,
                mode_step=mode_step,
                psi_step=psi_step,
                target_shape=target_shape,
            )
            if include_eigenmodes:
                try:
                    _scripts_dir = Path(__file__).resolve().parent
                    if str(_scripts_dir) not in sys.path:
                        sys.path.insert(0, str(_scripts_dir))
                    from eigenmode_features import get_eigenmode_amplitudes_safe
                    sparc_dir = p.parent
                    extra = get_eigenmode_amplitudes_safe(sparc_dir)
                    row.update(extra)
                except Exception:
                    pass
            rows.append(row)
        except Exception as e:
            if verbose:
                print(f"  Skip {p}: {e}")
            continue
    df = pd.DataFrame(rows)
    if verbose:
        n_out = len([c for c in df.columns if c.startswith("output_p_")])
        print(f"  Loaded {len(df)} cases, {len(df.columns)} columns ({n_out} spectrum components)")
    return df


def load_complex_v2_for_surge(
    batch_dir: Union[str, Path],
    **kwargs: Any,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load complex_v2 batch and return (DataFrame, input_columns, output_columns)
    ready for SurrogateDataset.load_from_dataframe().

    Parameters
    ----------
    batch_dir : str or Path
        Batch root directory
    **kwargs
        Passed to build_dataframe_from_batch

    Returns
    -------
    df : pd.DataFrame
    input_columns : list of str
    output_columns : list of str
    """
    df = build_dataframe_from_batch(batch_dir, **kwargs)
    input_cols = [
        c for c in df.columns
        if (c.startswith("eq_") and c not in ("eq_id",))
        or c.startswith("input_")
        or c.startswith("eigenmode_amp_")
        or c in ("q0", "q95", "qmin", "p0")
    ]
    output_cols = [c for c in df.columns if c.startswith("output_p_")]
    return df, input_cols, output_cols


__all__ = [
    "find_complex_v2_files",
    "load_single_complex_v2",
    "load_single_complex_v2_per_mode",
    "build_dataframe_per_mode",
    "load_per_mode_for_surge",
    "build_dataframe_from_batch",
    "load_complex_v2_for_surge",
    "COMPLEX_V2_FILENAME",
    "SDATA_COMPLEX_V2_FILENAME",
]
