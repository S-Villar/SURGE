#!/usr/bin/env python3
"""Regression: native field relL2 matches field_recon_compare.json within tolerance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from spectrum_field_loss import field_rel_l2_native_numpy  # noqa: E402

TOL = 5e-3
RUN = Path("runs/spectrum_fno48_floor6_smooth1_qc_peak4")
REF = RUN / "field_recon" / "field_recon.json"
CACHE = RUN / "predictions_cache.npz"


def main() -> None:
    if not REF.exists() or not CACHE.exists():
        print(f"SKIP: need {REF} and {CACHE}")
        return
    ref = json.loads(REF.read_text())
    z = np.load(CACHE, allow_pickle=True)
    keys = z["keys"].astype(str)
    te = z["split"].astype(str) == "test"
    key_to_i = {k: i for i, k in enumerate(keys[te])}
    m_grid = z["m_grid"].astype(float)
    psi_grid = z["psi_grid"].astype(float)
    spec_field = str(z.get("spectrum_field", "p"))
    paths = z["paths"].astype(str)
    pred = z["pred"][te]
    test_keys = keys[te]
    path_map = {k: paths[te][i] for i, k in enumerate(test_keys)}

    for label, rec in ref.items():
        key = rec["key"]
        i = key_to_i[key]
        rel = field_rel_l2_native_numpy(
            pred[i], path_map[key], m_grid, psi_grid, spec_field
        )
        ref_rel = float(rec["field_relL2"])
        assert abs(rel - ref_rel) <= TOL, (
            f"{label} {key}: got {rel:.6f} ref {ref_rel:.6f}"
        )
        print(f"OK {label:6s} {key:22s} relL2={rel:.4f} (ref {ref_rel:.4f})")
    print("field_loss regression: PASS")


if __name__ == "__main__":
    main()
