"""
Optional eigenmode feature extraction for M3DC1 dataset pipeline.

Extracts eigenmode amplitudes from C1.h5 in sparc_* dirs. Requires m3dc1 and fpy
(M3DC1 environment). Returns empty dict when unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Lazy imports - m3dc1 and fpy are only in M3DC1 env
M1_AVAILABLE = False
FPY_AVAILABLE = False

try:
    import fpy  # noqa: F401
    FPY_AVAILABLE = True
except ImportError:
    pass

_m1_src = Path(__file__).resolve().parent.parent.parent / "M3DC1" / "unstructured" / "python"
# Optional fallback: users on shared HPC filesystems can point M3DC1_PYTHON_PATH
# at their own unstructured/python checkout. There is no longer a hardcoded
# default — the public repo is portable.
if not _m1_src.exists():
    import os as _os
    _env = _os.environ.get("M3DC1_PYTHON_PATH")
    if _env:
        _m1_src = Path(_env)
if _m1_src.exists():
    import sys
    if str(_m1_src) not in sys.path:
        sys.path.insert(0, str(_m1_src))
    try:
        import m3dc1  # noqa: F401
        M1_AVAILABLE = True
    except ImportError:
        pass


def get_eigenmode_amplitudes_safe(
    sparc_dir: Path,
    *,
    field_name: str = "p",
    time: str = "last",
    points: int = 400,
    max_modes: int = 20,
) -> Dict[str, float]:
    """
    Extract eigenmode amplitude features from C1.h5. Safe to call when m3dc1/fpy unavailable.

    Returns dict with eigenmode_amp_0, eigenmode_amp_1, ... (max_modes entries).
    Each is max amplitude over flux surfaces for that mode.
    Returns empty dict if m3dc1/fpy not available or extraction fails.
    """
    result: Dict[str, float] = {}
    if not M1_AVAILABLE or not FPY_AVAILABLE:
        return result

    sparc_dir = Path(sparc_dir)
    c1_path = sparc_dir / "C1.h5"
    if not c1_path.exists():
        return result

    cwd = os.getcwd()
    try:
        os.chdir(str(sparc_dir))
        import numpy as np

        mysim = fpy.sim_data("C1.h5", time=time, verbose=False)
        eigen_data = m3dc1.eigenfunction(
            field=field_name,
            sim=mysim,
            fcoords="pest",
            device="sparc",
            time=time,
            points=points,
            makeplot=False,
            quiet=True,
            fourier=True,
        )
        eigen_array = np.asarray(eigen_data)
        if eigen_array.ndim < 2:
            return result
        # Shape (n_modes, n_flux); take max over flux per mode
        amp_per_mode = np.max(np.abs(eigen_array), axis=1)
        n = min(len(amp_per_mode), max_modes)
        for i in range(n):
            result[f"eigenmode_amp_{i}"] = float(amp_per_mode[i])
    except Exception:
        pass
    finally:
        os.chdir(cwd)
    return result


__all__ = ["get_eigenmode_amplitudes_safe", "M1_AVAILABLE", "FPY_AVAILABLE"]
