"""
Plotting helpers for M3DC1 perturbed fields and spectra.

Adapted from /pscratch/sd/a/asvillar/mp288/jobs/batch_16/post/plot_sdata_pertfields.py
for use in Jupyter notebooks. Functions can save to file or return figures for inline display.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _decode(value):
    if isinstance(value, h5py.Dataset):
        return _decode(value[()])
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value[()])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_mesh_coords(
    h5: h5py.File, run_group: h5py.Group
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get R, Z mesh coordinates for a run from the HDF5 file."""
    mesh_id = _decode(run_group.get("meshID", None))
    if not mesh_id or "mesh" not in h5:
        return None, None
    mesh_group = h5["mesh"]
    if mesh_id not in mesh_group:
        return None, None
    mg = mesh_group[mesh_id]
    if "R" not in mg or "Z" not in mg:
        return None, None
    return np.asarray(mg["R"]), np.asarray(mg["Z"])


def _select_time_positions(
    time_index: np.ndarray,
    count: int,
    explicit: Optional[Iterable[int]],
) -> List[int]:
    """Select which time positions to plot."""
    if time_index.size == 0:
        return []
    if explicit:
        positions = []
        lookup = {int(val): idx for idx, val in enumerate(time_index.tolist())}
        for val in explicit:
            if int(val) in lookup:
                positions.append(lookup[int(val)])
        return positions
    if time_index.size <= count:
        return list(range(time_index.size))
    return np.linspace(0, time_index.size - 1, count, dtype=int).tolist()


def _time_label(time_idx: int, time_val: Optional[float]) -> str:
    """Format time label for plot titles."""
    if time_val is None or np.isnan(time_val):
        return f"t{time_idx:03d}"
    return f"t{time_idx:03d}_{time_val:.3e}s"


def plot_2d_fields(
    h5: h5py.File,
    run_group: h5py.Group,
    tag: str,
    time_index: np.ndarray,
    time_values: np.ndarray,
    time_positions: List[int],
    out_dir: Optional[Union[str, Path]] = None,
    fields: Optional[List[str]] = None,
) -> Optional[List[plt.Figure]]:
    """
    Plot 2D perturbed fields (p, BR, BZ, BPHI) on R-Z mesh or index grid.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file
    run_group : h5py.Group
        Run group (e.g. h5['runs']['run_0001'])
    tag : str
        Tag for titles
    time_index, time_values, time_positions
        Time dimension info (use empty arrays if no time dim)
    out_dir : Path, optional
        If set, save figures to this directory
    fields : list of str, optional
        Field names to plot (default: ["p", "BR", "BZ", "BPHI"])

    Returns
    -------
    list of Figure or None
        If out_dir is None, returns list of figures for inline display.
    """
    if "pertfields" not in run_group:
        print("WARNING: pertfields not found")
        return None
    R, Z = _get_mesh_coords(h5, run_group)
    use_mesh = R is not None and Z is not None
    pf = run_group["pertfields"]
    fields = fields or ["p", "BR", "BZ", "BPHI"]
    figures = []

    for field in fields:
        field_name = field
        if field_name not in pf and f"{field}_phi0" in pf:
            field_name = f"{field}_phi0"
        if field_name not in pf:
            continue
        data = np.asarray(pf[field_name])
        if np.iscomplexobj(data):
            data = np.real(data)
        has_time = time_index.size > 0 and data.ndim >= 2 and data.shape[0] == time_index.size
        tpos = time_positions if time_positions else ([0] if has_time else [])

        def _make_plot(data_t, t_idx=None, t_val=None):
            fig, ax = plt.subplots()
            if use_mesh:
                if R.ndim == 1 and Z.ndim == 1 and data_t.ndim == 2:
                    RR, ZZ = np.meshgrid(R, Z)
                else:
                    RR, ZZ = R, Z
                if RR.shape == data_t.shape and ZZ.shape == data_t.shape:
                    pcm = ax.pcolormesh(RR, ZZ, data_t, shading="auto")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("R")
                    ax.set_ylabel("Z")
                else:
                    pcm = ax.imshow(data_t, origin="lower", aspect="auto")
                    ax.set_xlabel("index")
                    ax.set_ylabel("index")
            else:
                pcm = ax.imshow(data_t, origin="lower", aspect="auto")
                ax.set_xlabel("index")
                ax.set_ylabel("index")
            tl = _time_label(t_idx, t_val) if t_idx is not None else ""
            ax.set_title(f"{field} ({tag}" + (f", {tl})" if tl else ")"))
            fig.colorbar(pcm, ax=ax, label=field)
            fig.tight_layout()
            return fig

        if has_time:
            for pos in tpos:
                t_idx = int(time_index[pos])
                t_val = time_values[pos] if time_values.size > pos else None
                fig = _make_plot(data[pos], t_idx, t_val)
                if out_dir:
                    fig.savefig(Path(out_dir) / f"{field}_2d_{tag}_t{t_idx:03d}.png", dpi=200)
                    plt.close(fig)
                else:
                    figures.append(fig)
        else:
            fig = _make_plot(data)
            if out_dir:
                fig.savefig(Path(out_dir) / f"{field}_2d_{tag}.png", dpi=200)
                plt.close(fig)
            else:
                figures.append(fig)

    return figures if figures else None


def plot_pertfield_complex(
    h5: h5py.File,
    run_group: h5py.Group,
    tag: str,
    field: str,
    mode: str = "hat-mag",
    time_index: Optional[np.ndarray] = None,
    time_values: Optional[np.ndarray] = None,
    time_positions: Optional[List[int]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Plot complex perturbed field (real, imag, magnitude, or phase).

    Parameters
    ----------
    h5, run_group, tag : as in plot_2d_fields
    field : str
        Field name (e.g. "p", "BPHI")
    mode : str
        "phi0", "phiq", "hat-real", "hat-imag", "hat-mag", "hat-phase"
    time_index, time_values, time_positions
        Time dimension (use empty/None if no time)
    out_dir : Path, optional
        Save path

    Returns
    -------
    Figure or None
    """
    if "pertfields" not in run_group:
        print("WARNING: pertfields not found")
        return None
    pf = run_group["pertfields"]
    R, Z = _get_mesh_coords(h5, run_group)
    use_mesh = R is not None and Z is not None
    time_index = time_index if time_index is not None else np.array([], dtype=int)
    time_values = time_values if time_values is not None else np.array([], dtype=float)
    time_positions = time_positions or []

    key = field
    if mode in {"phi0", "phiq"}:
        key = f"{field}_{mode}"
    elif mode.startswith("hat-"):
        key = f"{field}_hat"
    if key not in pf and field in pf:
        key = field
    if key not in pf:
        print(f"WARNING: Missing pertfield {key} (tried {field})")
        return None
    data = np.asarray(pf[key])
    if mode == "hat-real":
        data = np.real(data)
    elif mode == "hat-imag":
        data = np.imag(data)
    elif mode == "hat-mag":
        data = np.abs(data)
    elif mode == "hat-phase":
        data = np.angle(data)
    elif key == field and np.iscomplexobj(data):
        data = np.abs(data)

    has_time = time_index.size > 0 and data.ndim >= 2 and data.shape[0] == time_index.size
    if has_time and time_positions:
        pos = time_positions[0]
        data = data[pos]
        t_idx = int(time_index[pos])
        t_val = time_values[pos] if time_values.size > pos else None
        title = f"{key} ({tag}, {_time_label(t_idx, t_val)})"
    else:
        data = data.squeeze()
        title = f"{key} ({tag})"

    fig, ax = plt.subplots()
    if use_mesh and data.shape == R.shape:
        pcm = ax.pcolormesh(R, Z, data, shading="auto")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
    else:
        pcm = ax.imshow(data, origin="lower", aspect="auto")
        ax.set_xlabel("index")
        ax.set_ylabel("index")
    fig.colorbar(pcm, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / f"{key}_{tag}.png", dpi=200)
    return fig


def plot_spectrum(
    run_group: h5py.Group,
    tag: str,
    field_name: str = "p",
    time_index: Optional[np.ndarray] = None,
    time_values: Optional[np.ndarray] = None,
    time_positions: Optional[List[int]] = None,
    scale: str = "linear",
    log_floor: float = 1e-12,
    out_dir: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Plot spectrum (m-modes vs psi_N) for delta p or similar.

    Parameters
    ----------
    run_group : h5py.Group
    tag : str
    field_name : str
        "p" or "bphi" (or key in spectrum group)
    scale : "linear" or "log"
    log_floor : float
        Floor for log scale
    """
    if "spectrum" not in run_group:
        print("WARNING: spectrum not found")
        return None
    spec_group = run_group["spectrum"]
    groups = {k: spec_group[k] for k in spec_group.keys()
              if isinstance(spec_group.get(k), h5py.Group)}
    if groups and field_name in groups:
        group = groups[field_name]
    else:
        group = spec_group
    if "m_modes" not in group or "psi_norm" not in group or "spec" not in group:
        print(f"WARNING: incomplete spectrum for {field_name}")
        return None
    m_modes = np.asarray(group["m_modes"])
    psi_norm = np.asarray(group["psi_norm"])
    spec = np.asarray(group["spec"])
    if np.iscomplexobj(spec):
        spec = np.abs(spec)
    spec_label = "|F_m(psi)|"
    if scale == "log":
        vmax = np.nanmax(spec)
        floor = max(vmax * log_floor, 1e-30)
        spec = np.log10(np.maximum(spec, floor))
        spec_label = "log10(|F_m(psi)|)"

    time_index = time_index if time_index is not None else np.array([], dtype=int)
    time_values = time_values if time_values is not None else np.array([], dtype=float)
    time_positions = time_positions or []
    has_time = time_index.size > 0 and spec.ndim == 3 and spec.shape[0] == time_index.size
    if has_time and not time_positions:
        time_positions = [0]

    if has_time and time_positions:
        pos = time_positions[0]
        spec_t = spec[pos]
        t_idx = int(time_index[pos])
        t_val = time_values[pos] if time_values.size > pos else None
        title = f"Spectrum {field_name} ({tag}, {_time_label(t_idx, t_val)})"
    else:
        spec_t = spec
        title = f"Spectrum {field_name} ({tag})"

    extent = [psi_norm.min(), psi_norm.max(), m_modes.min(), m_modes.max()]
    fig, ax = plt.subplots()
    im = ax.imshow(spec_t, origin="lower", aspect="auto", extent=extent)
    fig.colorbar(im, ax=ax, label=spec_label)
    ax.set_xlabel("psi_N")
    ax.set_ylabel("m")
    ax.set_title(title)
    fig.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / f"spectrum_{field_name}_{tag}.png", dpi=200)
    return fig


def run_tag(run_group: h5py.Group) -> str:
    """Get run tag from run group."""
    run_id = _decode(run_group.get("runID", "run"))
    eq_id = _decode(run_group.get("eqID", "eq"))
    tag = f"{run_id}_{eq_id}"
    return tag.replace("/", "_")
