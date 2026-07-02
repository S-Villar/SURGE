#!/usr/bin/env python3
"""Quantify how much of the |delta p|(m, psi_N) spectral energy falls inside the
m-window used for training ([-80, 20] by default) vs outside, and where the
outside energy sits. Answers: 'are all the modes really inside -80..20?'

  python scripts/m3dc1/internal/scan_mwindow.py --n 1500 --m-lo -80 --m-hi 20
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_spectrum_image as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--m-lo", type=float, default=-80.0)
    ap.add_argument("--m-hi", type=float, default=20.0)
    ap.add_argument("--field", default="p")
    args = ap.parse_args()

    paths = T.find_complex_v2_files(args.batch_dir, filename=args.filename)[:args.n]
    frac_in = []; peak_in = []; peak_m = []; frac_hi = []; frac_lo = []
    # fraction of energy captured if we also included a wider window, to see if
    # the 'missing' energy is coherent (would be recovered by more m) or floor.
    for p in paths:
        c = T._read_case(Path(p), args.field)
        if c is None:
            continue
        mag = c["mag"]; mm = c["m_modes"]                 # (nm, npsi), (nm,)
        E = (mag ** 2).sum(axis=1)                        # energy per m row
        tot = E.sum()
        if tot <= 0:
            continue
        sel = (mm >= args.m_lo) & (mm <= args.m_hi)
        frac_in.append(E[sel].sum() / tot)
        frac_hi.append(E[mm > args.m_hi].sum() / tot)     # positive-m tail
        frac_lo.append(E[mm < args.m_lo].sum() / tot)     # negative-m tail
        mpk = mm[np.argmax(E)]
        peak_m.append(float(mpk))
        peak_in.append(bool(args.m_lo <= mpk <= args.m_hi))

    frac_in = np.array(frac_in); peak_m = np.array(peak_m)
    frac_hi = np.array(frac_hi); frac_lo = np.array(frac_lo)
    print(f"cases scanned: {len(frac_in)}   window [{args.m_lo:g}, {args.m_hi:g}]")
    print(f"energy fraction INSIDE window : mean={frac_in.mean():.3f} "
          f"median={np.median(frac_in):.3f}  p10={np.percentile(frac_in,10):.3f}")
    print(f"energy in POSITIVE-m tail (>{args.m_hi:g}): mean={frac_hi.mean():.3f} "
          f"median={np.median(frac_hi):.3f}  p90={np.percentile(frac_hi,90):.3f}")
    print(f"energy in NEGATIVE-m tail (<{args.m_lo:g}): mean={frac_lo.mean():.3f} "
          f"median={np.median(frac_lo):.3f}")
    print(f"peak-m INSIDE window: {100*np.mean(peak_in):.1f}% of cases")
    print(f"peak-m distribution: min={peak_m.min():.0f} p05={np.percentile(peak_m,5):.0f} "
          f"median={np.median(peak_m):.0f} p95={np.percentile(peak_m,95):.0f} "
          f"max={peak_m.max():.0f}")
    # how many cases lose >5% / >10% of energy to clipping
    lost = 1 - frac_in
    print(f"cases losing >5%% energy: {100*np.mean(lost>0.05):.1f}%%   "
          f">10%%: {100*np.mean(lost>0.10):.1f}%%   >25%%: {100*np.mean(lost>0.25):.1f}%%")


if __name__ == "__main__":
    main()
