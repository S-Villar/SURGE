"""Gauge-fix M3DC1 linear eigenmode fields (additive helper module).

M3D-C1 eigenmodes are defined up to a global complex phase α = A·exp(iθ₀).
Max-normalization removes A but not θ₀. This module fixes the gauge by rotating
each case so the reference point has real-positive amplitude.

Convention (colleague): θ_ref = arg(δp) at the midplane peak; δp_gf = exp(−iθ_ref)·δp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

MidplaneZ = Literal["axis", "zero"]
GaugeRef = Literal["peak", "template"]


@dataclass
class GaugeFixResult:
    field_gf: np.ndarray
    theta_ref: float
    ref_row: int
    ref_col: int
    midplane_row: int
    ref_mode: str


def _midplane_row(Z: Optional[np.ndarray], mode: MidplaneZ, z_axis: float) -> int:
    """Row index of the poloidal midplane (Z closest to axis or zero)."""
    if Z is None:
        return 0
    z1d = np.nanmean(Z, axis=1) if Z.ndim == 2 else np.asarray(Z, float).ravel()
    target = float(z_axis) if mode == "axis" else 0.0
    return int(np.argmin(np.abs(z1d - target)))


def _sanitize_field(f: np.ndarray) -> np.ndarray:
    """Replace non-finite values with zero for gauge operations."""
    out = np.asarray(f, np.complex128).copy()
    bad = ~np.isfinite(out.real) | ~np.isfinite(out.imag)
    out[bad] = 0.0
    return out


def _weighted_phase(field: np.ndarray, weights: np.ndarray) -> float:
    """Amplitude-weighted mean phase (robust circular average)."""
    w = np.asarray(weights, float).ravel()
    f = np.asarray(field).ravel()
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-30:
        return 0.0
    z = np.sum(w * f) / (w.sum() + 1e-30)
    return float(np.angle(z))


def compute_theta_ref(
    field: np.ndarray,
    *,
    Z: Optional[np.ndarray] = None,
    z_axis: float = 0.0,
    midplane_z: MidplaneZ = "axis",
    gauge_ref: GaugeRef = "peak",
    peak_window: int = 3,
    template: Optional[np.ndarray] = None,
) -> Tuple[float, int, int, int]:
    """Return (theta_ref, ref_row, ref_col, midplane_row)."""
    f = _sanitize_field(field)
    H, W = f.shape[-2], f.shape[-1]
    mp_row = _midplane_row(Z, midplane_z, z_axis)
    mp_row = int(np.clip(mp_row, 0, H - 1))

    if gauge_ref == "template" and template is not None:
        tmpl = np.asarray(template, dtype=np.complex128)
        if tmpl.shape != f.shape:
            raise ValueError(f"template shape {tmpl.shape} != field shape {f.shape}")
        w = np.abs(f) * np.abs(tmpl)
        theta = _weighted_phase(f * np.conj(tmpl), w)
        # report peak on midplane for logging
        strip = np.abs(f[mp_row])
        ref_col = int(np.argmax(strip))
        return theta, mp_row, ref_col, mp_row

    strip = np.abs(f[mp_row])
    ref_col = int(np.argmax(strip))
    hw = max(0, int(peak_window))
    r0 = max(0, mp_row - hw)
    r1 = min(H, mp_row + hw + 1)
    c0 = max(0, ref_col - hw)
    c1 = min(W, ref_col + hw + 1)
    patch = f[r0:r1, c0:c1]
    w = np.abs(patch)
    theta = _weighted_phase(patch, w)
    return theta, mp_row, ref_col, mp_row


def gauge_fix_field(
    field: np.ndarray,
    *,
    Z: Optional[np.ndarray] = None,
    z_axis: float = 0.0,
    midplane_z: MidplaneZ = "axis",
    gauge_ref: GaugeRef = "peak",
    peak_window: int = 3,
    template: Optional[np.ndarray] = None,
) -> GaugeFixResult:
    """Rotate complex δp by exp(−iθ_ref) so the reference is real-positive."""
    f = _sanitize_field(field)
    if not np.iscomplexobj(f):
        f = f.astype(np.complex128)

    theta, ref_r, ref_c, mp_row = compute_theta_ref(
        f, Z=Z, z_axis=z_axis, midplane_z=midplane_z,
        gauge_ref=gauge_ref, peak_window=peak_window, template=template,
    )
    rot = np.exp(-1j * theta)
    gf = f * rot
    # Enforce real-positive at reference (fix numerical drift)
    ref_val = gf[ref_r, ref_c]
    if np.abs(ref_val.imag) > 1e-6 * (np.abs(ref_val.real) + 1e-12):
        gf = gf * np.exp(-1j * np.angle(ref_val))
        theta += float(np.angle(ref_val))

    return GaugeFixResult(
        field_gf=gf,
        theta_ref=float(theta),
        ref_row=ref_r,
        ref_col=ref_c,
        midplane_row=mp_row,
        ref_mode=gauge_ref,
    )


def build_template_from_gauge_fixed(
    fields_gf: list[np.ndarray],
    *,
    max_cases: Optional[int] = None,
) -> np.ndarray:
    """Mean complex field over gauge-fixed cases (for template reference mode)."""
    stack = fields_gf[:max_cases] if max_cases else fields_gf
    if not stack:
        raise ValueError("empty field list for template")
    arr = np.stack([np.asarray(f, np.complex128) for f in stack], axis=0)
    tmpl = np.mean(arr, axis=0)
    peak = np.abs(tmpl).max() or 1.0
    return (tmpl / peak).astype(np.complex128)


def mean_signed_coherence(field_real: np.ndarray) -> dict:
    """Metrics for dataset-mean signed field coherence (NaN-safe)."""
    m = np.asarray(field_real, float)
    n = m.shape[0]
    mu = np.nanmean(m, axis=0)
    l2_mean = float(np.sqrt(np.nansum(mu ** 2)))
    l2_per_case = float(np.nanmean([np.sqrt(np.nansum(m[i] ** 2)) for i in range(n)]))
    ratio = l2_mean / (l2_per_case + 1e-30)
    peak_mean = float(np.nanmax(np.abs(mu)))
    peak_case = float(np.nanmean([np.nanmax(np.abs(m[i])) for i in range(n)]))
    return {
        "l2_mean_field": l2_mean,
        "l2_mean_per_case": l2_per_case,
        "coherence_ratio": ratio,
        "peak_mean_field": peak_mean,
        "peak_mean_per_case": peak_case,
    }


def optimal_global_phase(pred: np.ndarray, true: np.ndarray) -> float:
    """θ_opt = arg⟨pred, true⟩ for complex or (Re,Im) stacked arrays."""
    if pred.ndim == true.ndim and pred.shape[-3] == 2:
        pc = pred[..., 0, :, :] + 1j * pred[..., 1, :, :]
        tc = true[..., 0, :, :] + 1j * true[..., 1, :, :]
    else:
        pc = np.asarray(pred, np.complex128)
        tc = np.asarray(true, np.complex128)
    inner = np.sum(pc * np.conj(tc))
    return float(np.angle(inner)) if np.abs(inner) > 1e-30 else 0.0


def apply_global_phase(field: np.ndarray, theta: float) -> np.ndarray:
    """Multiply complex field by exp(i·theta); preserves (N,2,H,W) layout."""
    if field.ndim >= 3 and field.shape[-3] == 2:
        c = field[..., 0, :, :] + 1j * field[..., 1, :, :]
        c = c * np.exp(1j * theta)
        out = np.stack([np.real(c), np.imag(c)], axis=-3)
        return out.astype(np.float32)
    c = np.asarray(field, np.complex128) * np.exp(1j * theta)
    return c


def rel_l2_complex_batch(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    phase_align: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-case relL2 on complex fields. Returns (raw, aligned) if phase_align."""
    n = pred.shape[0]
    raw = np.empty(n, float)
    aligned = np.empty(n, float) if phase_align else None
    for i in range(n):
        if pred.ndim == 4 and pred.shape[1] == 2:
            p = pred[i, 0] + 1j * pred[i, 1]
            t = true[i, 0] + 1j * true[i, 1]
        else:
            p = np.asarray(pred[i], np.complex128)
            t = np.asarray(true[i], np.complex128)
        denom = float(np.linalg.norm(t.ravel()))
        raw[i] = float(np.linalg.norm((p - t).ravel()) / denom) if denom > 1e-30 else np.nan
        if phase_align:
            th = float(np.angle(np.sum(p * np.conj(t)))) if np.abs(np.sum(p * np.conj(t))) > 1e-30 else 0.0
            pa = p * np.exp(-1j * th)
            aligned[i] = float(np.linalg.norm((pa - t).ravel()) / denom) if denom > 1e-30 else np.nan
    if phase_align:
        return raw, aligned  # type: ignore[return-value]
    return raw, raw
