#!/usr/bin/env python3
"""Scan the whole dataset for bad-data / degenerate cases and write a quarantine
list that the trainer can exclude (--exclude-list).

Flags a case as BAD if ANY of:
  * growth rate missing/NaN/inf         -> run didn't converge (no eigenvalue)
  * spectrum has NaN/inf                 -> corrupt
  * spectrum max <= 0                    -> empty
  * dynamic range < --min-dyn decades    -> flat noise, no localized mode
                                            (median amplitude within <N decades
                                             of the peak => no real ridge)

Writes:
  runs/quarantine/bad_cases.json   {key: reason}
  runs/quarantine/quality.csv      per-case features
  runs/quarantine/quality.png      distributions with the bad cut

  python scripts/m3dc1/internal/scan_quality.py --min-dyn 0.8
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_spectrum_image as T  # noqa: E402


def _read_gamma(path):
    try:
        with h5py.File(path, "r") as f:
            rg = f["runs"][list(f["runs"].keys())[0]]
            g = [np.nan, np.nan]
            if "growth_rate" in rg:
                for k in ("0", "1"):
                    if k in rg["growth_rate"]:
                        g[int(k)] = float(rg["growth_rate"][k][()])
            return g[0], g[1]
    except Exception:
        return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="/pscratch/sd/a/asvillar/mp288/jobs/batch_16")
    ap.add_argument("--filename", default="csdata_deltap_b_ver.h5")
    ap.add_argument("--field", default="p")
    ap.add_argument("--m-lo", type=float, default=-80.0)
    ap.add_argument("--m-hi", type=float, default=20.0)
    ap.add_argument("--min-dyn", type=float, default=0.8,
                    help="min dynamic range (decades from peak to median) to be a "
                         "real localized mode; below this = flat noise = bad.")
    ap.add_argument("--out", default="runs/quarantine")
    args = ap.parse_args()

    paths = T.find_complex_v2_files(args.batch_dir, filename=args.filename)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []; bad = {}
    t0 = time.time()
    for i, p in enumerate(paths):
        c = T._read_case(Path(p), args.field)
        if c is None:
            # unreadable / missing spectrum -> quarantine by path stem
            bad[Path(p).parent.name] = "unreadable"
            continue
        key = f"{c['run_id']}_{c['eq_id']}"
        mag = c["mag"]; mm = c["m_modes"]
        g0, g1 = _read_gamma(p)
        finite = bool(np.all(np.isfinite(mag)))
        mmax = float(np.nanmax(mag)) if mag.size else 0.0
        sel = (mm >= args.m_lo) & (mm <= args.m_hi)
        dyn = np.nan; peak_psi = np.nan
        if finite and mmax > 0 and sel.sum() >= 4:
            fld = mag[sel, :] / mmax
            lg = np.log10(fld + 1e-12)
            dyn = float(-np.median(lg))            # decades from peak(0) to median
            a = np.argmax(fld); peak_psi = float(c["psi"][a % fld.shape[1]]) \
                if c["psi"].size == fld.shape[1] else np.nan
        gamma_ok = np.isfinite(g0) or np.isfinite(g1)
        reason = None
        if not finite:
            reason = "nan_inf_spectrum"
        elif mmax <= 0:
            reason = "empty_spectrum"
        elif not gamma_ok:
            reason = "no_growth_rate"
        elif np.isfinite(dyn) and dyn < args.min_dyn:
            reason = "flat_no_mode"
        if reason:
            bad[key] = reason
        rows.append((key, mmax, dyn, g0, g1, peak_psi, reason or "ok"))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s) bad so far={len(bad)}",
                  flush=True)

    # write outputs
    (out / "bad_cases.json").write_text(json.dumps(bad, indent=2))
    import csv
    with (out / "quality.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "mag_max", "dyn_decades", "g0", "g1",
                                        "peak_psi", "reason"])
        w.writerows(rows)
    # summary
    from collections import Counter
    reasons = Counter(v for v in bad.values())
    print(f"\nscanned {len(rows)} readable cases, total flagged BAD = {len(bad)} "
          f"({100*len(bad)/max(1,len(paths)):.1f}%)")
    for r, n in reasons.most_common():
        print(f"  {r:20s} {n}")
    print(f"quarantine -> {out/'bad_cases.json'}")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dyn = np.array([r[2] for r in rows], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dyn[np.isfinite(dyn)], bins=80, color="#4C72B0")
    ax[0].axvline(args.min_dyn, color="r", ls="--", label=f"bad cut {args.min_dyn}")
    ax[0].set_xlabel("dynamic range (decades peak->median)")
    ax[0].set_title("flat/no-mode cases are the left spike"); ax[0].legend()
    ax[1].bar(range(len(reasons)), [n for _, n in reasons.most_common()])
    ax[1].set_xticks(range(len(reasons)))
    ax[1].set_xticklabels([r for r, _ in reasons.most_common()], rotation=30, ha="right")
    ax[1].set_title(f"{len(bad)} quarantined cases by reason")
    plt.tight_layout(); plt.savefig(out / "quality.png", dpi=110)
    print("saved", out / "quality.png")


if __name__ == "__main__":
    main()
