"""
Dataset generator and loader for per-run sdata_pertfields_grid_complex_v2.h5 files.

Each run directory (run1/sparc_1300, run2/sparc_1305, etc.) contains its own
sdata_pertfields_grid_complex_v2.h5 with:
- spectrum/p: delta p modes spectrum (m_modes x psi_norm, complex)
- miller: R0, a, kappa, delta (equilibrium inputs)
- parset: ntor, pscale, batemanscale (run inputs)
- flux_average: q, p profiles for q0, q95, qmin, p0
- growth_rate: gamma

This module finds all such files, extracts the right fields, and builds
a DataFrame for SURGE surrogate training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# Default filename for per-run complex v2 HDF5
COMPLEX_V2_FILENAME = "sdata_pertfields_grid_complex_v2.h5"


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
        Root batch directory (e.g. /pscratch/.../batch_16)
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
        if mode_step is not None or psi_step is not None:
            sl_m = slice(None, None, mode_step or 1)
            sl_p = slice(None, None, psi_step or 1)
            spec = spec[sl_m, sl_p]
        flat = spec.flatten()
        for i, v in enumerate(flat):
            row[f"output_p_{i}"] = float(v)

    return row


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
    max_cases: Optional[int] = None,
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
    max_cases : int, optional
        Limit number of cases (for testing)
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
            )
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
        if c.startswith("eq_") or c.startswith("input_") or c in ("q0", "q95", "qmin", "p0")
    ]
    output_cols = [c for c in df.columns if c.startswith("output_p_")]
    return df, input_cols, output_cols


__all__ = [
    "find_complex_v2_files",
    "load_single_complex_v2",
    "build_dataframe_from_batch",
    "load_complex_v2_for_surge",
    "COMPLEX_V2_FILENAME",
]
