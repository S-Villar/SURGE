"""Field-loss helpers for spectrum-image FNO training (additive module).

True-phase oracle idealization: at train time we inject ground-truth phase from
``csdata_deltap_b_ver.h5`` (complex ``spectrum/p/spec``) to define a differentiable
IFFT-proxy loss on the uniform training grid. This optimizes field quality *given*
oracle phase — the same post-hoc convention as ``field_recon_compare.py``.

Native-grid relL2 (matching ``field_recon_compare.py`` / ``field_bench.py``) is used
for checkpoint selection (--select-by field) and regression tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[1]))

from dataset_complex_v2 import find_complex_v2_files, _decode  # noqa: E402
import train_spectrum_image as T  # noqa: E402


def parse_family(key: str) -> str:
    parts = key.split("_", 1)
    return parts[1] if len(parts) > 1 else key


def stratified_subset(keys: Sequence[str], n: int, seed: int = 42) -> np.ndarray:
    """Return indices into ``keys`` for a family-balanced subset."""
    keys = list(keys)
    by_fam: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        by_fam.setdefault(parse_family(k), []).append(i)
    fams = sorted(by_fam)
    rng = np.random.RandomState(seed)
    per = max(1, n // max(len(fams), 1))
    picked: List[int] = []
    for fam in fams:
        pool = list(by_fam[fam])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) < n:
        rest = [i for i in range(len(keys)) if i not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return np.array(sorted(picked[:n]), dtype=int)


def _read_case_phase(path: Path, spectrum_field: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "runs" not in f:
                return None
            rname = list(f["runs"].keys())[0]
            rg = f["runs"][rname]
            sp = rg["spectrum"][spectrum_field]
            spec = np.asarray(sp["spec"])
            if spec.ndim == 3:
                spec = spec[-1]
            if not np.iscomplexobj(spec):
                return None
            m_modes = np.asarray(sp["m_modes"], float).ravel()
            psi = (
                np.asarray(sp["psi_norm"], float).ravel()
                if "psi_norm" in sp
                else np.linspace(1e-4, 1.0, spec.shape[1])
            )
            phase = np.angle(spec)
            return phase, m_modes, psi
    except Exception:
        return None


def build_phase_grids_for_keys(
    batch_dir: str,
    filename: str,
    keys: Sequence[str],
    grid: int,
    m_lo: float,
    m_hi: float,
    spectrum_field: str,
) -> Tuple[np.ndarray, List[str]]:
    """Build (N,H,W) phase grids aligned with ``keys`` order (missing -> zeros)."""
    key_to_path = {}
    for p in find_complex_v2_files(batch_dir, filename=filename):
        parts = Path(p).parts
        eq_id = parts[-2]
        run_id = parts[-3]
        key_to_path[f"{run_id}_{eq_id}"] = p

    psi_grid = np.linspace(0.0, 1.0, grid)
    m_grid = np.linspace(m_lo, m_hi, grid)
    phases = np.zeros((len(keys), grid, grid), dtype=np.float32)
    for i, key in enumerate(keys):
        p = key_to_path.get(key)
        if p is None:
            continue
        got = _read_case_phase(Path(p), spectrum_field)
        if got is None:
            continue
        phase, m_modes, psi = got
        sel = (m_modes >= m_lo) & (m_modes <= m_hi)
        if sel.sum() < 4:
            continue
        ph = phase[sel, :]
        m_vals = m_modes[sel]
        tmp = np.vstack([T._interp_to(psi_grid, psi, row) for row in ph])
        img = np.vstack([T._interp_to(m_grid, m_vals, tmp[:, j]) for j in range(grid)]).T
        phases[i] = img.astype(np.float32)
    return phases


def _to_full_m_grid_complex(values_2d: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    m_vals = np.asarray(m_values, dtype=int)
    v2d = np.asarray(values_2d)
    m_min, m_max = int(m_vals.min()), int(m_vals.max())
    m_full = np.arange(m_min, m_max + 1, dtype=int)
    out = np.zeros((len(m_full), v2d.shape[1]), dtype=np.complex128)
    idx = {int(m): i for i, m in enumerate(m_full)}
    for i, m in enumerate(m_vals):
        out[idx[int(m)], :] = v2d[i, :]
    return out


def _ifft_field_numpy(spec_complex: np.ndarray, m_modes: np.ndarray) -> np.ndarray:
    spec_full = _to_full_m_grid_complex(spec_complex, m_modes)
    recon = np.fft.ifft(np.fft.ifftshift(spec_full, axes=0), axis=0) * spec_full.shape[0]
    return np.real(recon)


def field_rel_l2_native_numpy(
    pred_dex: np.ndarray,
    path: str,
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
    spectrum_field: str,
    *,
    true_field: Optional[np.ndarray] = None,
    meta_cache: Optional[dict] = None,
) -> float:
    """Native-grid relL2 matching ``field_recon_compare.py``."""
    from scipy.interpolate import RegularGridInterpolator
    from m3dc1ml.io.sdata import load_complex_v2_case

    if meta_cache is not None and path in meta_cache:
        b = meta_cache[path]
    else:
        b = load_complex_v2_case(path, spectrum_field=spectrum_field)
        if meta_cache is not None:
            meta_cache[path] = b

    if true_field is None:
        ftrue = _ifft_field_numpy(np.asarray(b["spec_complex"]), np.asarray(b["m_modes"], float))
    else:
        ftrue = true_field

    mag_norm = np.power(10.0, pred_dex.astype(np.float64))
    rgi = RegularGridInterpolator(
        (m_grid, psi_grid), mag_norm, bounds_error=False, fill_value=0.0
    )
    nm = np.asarray(b["m_modes"], float)
    npsi = np.asarray(b["psi_norm"], float)
    mm, pp = np.meshgrid(nm, npsi, indexing="ij")
    mag_pred = rgi(np.stack([mm.ravel(), pp.ravel()], 1)).reshape(len(nm), len(npsi))
    phase = np.angle(b["spec_complex"]) if np.iscomplexobj(b["spec_complex"]) else 0.0
    pred_spec = mag_pred * np.exp(1j * phase)
    fpred = _ifft_field_numpy(pred_spec, nm)

    ftrue_n = ftrue / (np.abs(ftrue).max() + 1e-30)
    fpred_n = fpred / (np.abs(fpred).max() + 1e-30)
    diff = fpred_n - ftrue_n
    return float(np.linalg.norm(diff) / (np.linalg.norm(ftrue_n) + 1e-30))


def compute_crf_numpy(gt_dex: np.ndarray, pred_dex: np.ndarray, cutoff: float = 0.25) -> float:
    gt = np.power(10.0, gt_dex.astype(np.float64))
    pred = np.power(10.0, pred_dex.astype(np.float64))
    residual = np.abs(pred) - np.abs(gt)
    power = np.abs(np.fft.fft2(residual)) ** 2
    total = float(power.sum())
    if total <= 1e-30:
        return 0.0
    h, w = residual.shape
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    ky_g, kx_g = np.meshgrid(ky, kx, indexing="ij")
    kr = np.sqrt(kx_g ** 2 + ky_g ** 2)
    thr = float(np.quantile(kr.ravel(), cutoff))
    return float(power[kr <= thr].sum() / total)


def _dex_to_linear_torch(dex, target_floor: Optional[float]):
    import torch

    if target_floor is not None and target_floor > 0:
        dex = torch.clamp(dex, min=-float(target_floor))
    return torch.pow(10.0, dex)


def field_loss_honest_phase_torch(
    pred_phase_std,
    mag_pred_dex_batch,
    true_phase_batch,
    true_mag_dex_batch,
    *,
    y_mean: float,
    y_std: float,
    target_floor: Optional[float],
):
    """Field relL2: |δp̂|_pred × φ_pred  vs  |δp̂|_true × φ_true (training grid)."""
    import torch

    pred_phase = pred_phase_std.squeeze(1) * y_std + y_mean
    true_phase = true_phase_batch.to(device=pred_phase.device, dtype=pred_phase.dtype)
    mag_pred_lin = _dex_to_linear_torch(mag_pred_dex_batch, target_floor)
    mag_true_lin = _dex_to_linear_torch(true_mag_dex_batch, target_floor)
    pred_c = mag_pred_lin * torch.exp(1j * pred_phase)
    true_c = mag_true_lin * torch.exp(1j * true_phase)
    m_dim = -2
    pred_f = torch.real(
        torch.fft.ifft(torch.fft.ifftshift(pred_c, dim=m_dim), dim=m_dim) * pred_c.shape[m_dim]
    )
    true_f = torch.real(
        torch.fft.ifft(torch.fft.ifftshift(true_c, dim=m_dim), dim=m_dim) * true_c.shape[m_dim]
    )
    pred_n = pred_f / (pred_f.abs().amax(dim=(-2, -1), keepdim=True) + 1e-30)
    true_n = true_f / (true_f.abs().amax(dim=(-2, -1), keepdim=True) + 1e-30)
    diff = pred_n - true_n
    num = diff.flatten(1).norm(dim=1)
    den = true_n.flatten(1).norm(dim=1) + 1e-30
    return (num / den).mean()


def field_loss_training_grid_torch(
    pred_std,
    target_std,
    phase_batch,
    *,
    y_mean: float,
    y_std: float,
    target_floor: Optional[float],
):
    """Differentiable IFFT-proxy relL2 on the uniform training grid."""
    import torch

    pred_dex = pred_std * y_std + y_mean
    true_dex = target_std * y_std + y_mean
    pred_lin = _dex_to_linear_torch(pred_dex.squeeze(1), target_floor)
    true_lin = _dex_to_linear_torch(true_dex.squeeze(1), target_floor)
    phase = phase_batch.to(device=pred_lin.device, dtype=pred_lin.dtype)
    pred_c = pred_lin * torch.exp(1j * phase)
    true_c = true_lin * torch.exp(1j * phase)
    m_dim = -2
    pred_f = torch.real(
        torch.fft.ifft(torch.fft.ifftshift(pred_c, dim=m_dim), dim=m_dim) * pred_c.shape[m_dim]
    )
    true_f = torch.real(
        torch.fft.ifft(torch.fft.ifftshift(true_c, dim=m_dim), dim=m_dim) * true_c.shape[m_dim]
    )
    pred_n = pred_f / (pred_f.abs().amax(dim=(-2, -1), keepdim=True) + 1e-30)
    true_n = true_f / (true_f.abs().amax(dim=(-2, -1), keepdim=True) + 1e-30)
    diff = pred_n - true_n
    num = diff.flatten(1).norm(dim=1)
    den = true_n.flatten(1).norm(dim=1) + 1e-30
    return (num / den).mean()


def coherence_loss_torch(
    pred_std,
    target_std,
    *,
    y_mean: float,
    y_std: float,
    target_floor: Optional[float],
    cutoff: float = 0.25,
):
    """Differentiable CRF surrogate: fraction of |pred|-|true| residual in low-k band."""
    import torch

    pred_dex = pred_std * y_std + y_mean
    true_dex = target_std * y_std + y_mean
    pred = _dex_to_linear_torch(pred_dex.squeeze(1), target_floor)
    true = _dex_to_linear_torch(true_dex.squeeze(1), target_floor)
    residual = pred.abs() - true.abs()
    power = torch.abs(torch.fft.fft2(residual)) ** 2
    total = power.sum() + 1e-30
    h, w = residual.shape[-2], residual.shape[-1]
    ky = torch.fft.fftfreq(h, device=residual.device, dtype=residual.dtype)
    kx = torch.fft.fftfreq(w, device=residual.device, dtype=residual.dtype)
    ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
    kr = torch.sqrt(kx_g ** 2 + ky_g ** 2)
    thr = torch.quantile(kr.reshape(-1), cutoff)
    mask = kr <= thr
    low = power[:, mask].sum()
    return (low / total).mean()


def field_metric_improved(
    frac_gt1: float,
    p90: float,
    best_frac: float,
    best_p90: float,
) -> bool:
    if frac_gt1 < best_frac - 1e-9:
        return True
    if abs(frac_gt1 - best_frac) <= 1e-9 and p90 < best_p90 - 1e-9:
        return True
    return False


def eval_val_field_selection(
    net,
    Xva: np.ndarray,
    Yva_dex: np.ndarray,
    subset_idx: np.ndarray,
    paths: Sequence[str],
    m_grid: np.ndarray,
    psi_grid: np.ndarray,
    spectrum_field: str,
    *,
    device: str,
    batch_size: int,
    crf_cutoff: float = 0.25,
) -> Dict[str, float]:
    """Native-grid field metrics on a val subset (no grad)."""
    import torch

    net.eval()
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    meta_cache: dict = {}
    true_cache: dict = {}
    rels: List[float] = []
    crfs: List[float] = []

    with torch.no_grad():
        preds = []
        Xt = torch.tensor(Xva, dtype=torch.float32)
        for i in range(0, len(Xva), batch_size):
            preds.append(net(Xt[i : i + batch_size].to(dev)).cpu().numpy())
        pred_all = np.concatenate(preds).squeeze(1)

    for j in subset_idx:
        path = paths[j]
        if path not in true_cache:
            from m3dc1ml.io.sdata import load_complex_v2_case

            b = load_complex_v2_case(path, spectrum_field=spectrum_field)
            true_cache[path] = _ifft_field_numpy(
                np.asarray(b["spec_complex"]), np.asarray(b["m_modes"], float)
            )
        rel = field_rel_l2_native_numpy(
            pred_all[j],
            path,
            m_grid,
            psi_grid,
            spectrum_field,
            true_field=true_cache[path],
            meta_cache=meta_cache,
        )
        crfs.append(compute_crf_numpy(Yva_dex[j], pred_all[j], crf_cutoff))
        rels.append(rel)

    rels_arr = np.asarray(rels, float)
    return {
        "val_field_frac_gt1": float(np.mean(rels_arr > 1.0)),
        "val_field_p90": float(np.percentile(rels_arr, 90)),
        "val_field_median": float(np.median(rels_arr)),
        "val_field_mean": float(np.mean(rels_arr)),
        "val_field_crf": float(np.mean(crfs)),
    }
