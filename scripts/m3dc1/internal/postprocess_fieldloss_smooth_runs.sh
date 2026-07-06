#!/usr/bin/env bash
# Postprocess Run D (smooth05) and D' (smooth0): metric galleries, case panels,
# field_recon_compare, field_bench summary figures.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO"
source scripts/m3dc1/surge_slurm_env.sh
surge_slurm_setup_python

ASSETS=docs/m3dc1/assets
REF_CACHE=runs/spectrum_fno48_floor6_smooth1_qc/predictions_cache.npz
mkdir -p "$ASSETS" "$ASSETS/sparc1530_diagnosis_smooth"

declare -A RUNS=(
  [smooth05]=runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05
  [smooth0]=runs/spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0
)

DS_CACHE=runs/compare_balance/dataset_g128_m-80.0_20.0_max_log10_fl6.0_sm1.0_exTrue.npz
SPARC_CASES="1904 1327 2729"  # run79, run43, run99

for tag in smooth05 smooth0; do
  RUN="${RUNS[$tag]}"
  echo ""
  echo "========== $RUN ($tag) =========="

  python scripts/m3dc1/internal/metric_gallery.py \
    --cache "$RUN/predictions_cache.npz" \
    --ref-cache "$REF_CACHE" \
    --split test --metric r2_pattern \
    --out "$ASSETS/metric_reality_check_qc_peak4_fieldloss_${tag}_refqc_combined.png"

  python scripts/m3dc1/internal/metric_gallery.py \
    --cache "$RUN/predictions_cache.npz" \
    --ref-cache "$REF_CACHE" \
    --split test --metric r2_pattern --field \
    --out "$ASSETS/metric_reality_check_qc_peak4_fieldloss_${tag}_refqc_field.png"

  python scripts/m3dc1/internal/plot_case_field_recon.py \
    --run "$RUN" --split test --out-dir "$ASSETS" --tag "${tag}_test"

  for ci in $SPARC_CASES; do
    python scripts/m3dc1/internal/plot_case_field_recon.py \
      --run "$RUN" --split test --out-dir "$ASSETS/sparc1530_diagnosis_smooth" \
      --tag sparc1530_diagnosis --cases "$ci"
  done

  python scripts/m3dc1/internal/field_recon_compare.py \
    --run "$RUN" --model fno2d --device cuda \
    --ds-cache "$DS_CACHE" \
    --out "$RUN/field_recon"

  cp -f "$RUN/field_recon/field_recon_fno2d.png" \
    "$ASSETS/field_recon_fieldloss_${tag}.png"
  cp -f "$RUN/loss_fno2d.png" "$ASSETS/loss_fieldloss_${tag}_fno2d.png"
done

# Field-bench comparison figures
python3 << 'PY'
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/global/homes/a/asvillar/src/SURGE")
ASSETS = REPO / "docs/m3dc1/assets"

def load_per_case(bench_dir, model):
    col = f"relL2_{model}"
    vals = []
    with (bench_dir / "per_case.csv").open() as f:
        for r in csv.DictReader(f):
            if col in r:
                vals.append(float(r[col]))
    if not vals:
        raise KeyError(f"column {col} not in {bench_dir}/per_case.csv")
    return np.array(vals)

def overlay_hist(bench_dir, new_model, new_label, out_name):
    peak4 = load_per_case(bench_dir, "spectrum_fno48_floor6_smooth1_qc_peak4")
    base = load_per_case(bench_dir, "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss")
    new = load_per_case(bench_dir, new_model)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 2.5, 50)
    ax.hist(peak4, bins=bins, alpha=0.45, label=f"peak4 (n={len(peak4)})", color="#ff7f0e")
    ax.hist(base, bins=bins, alpha=0.45, label=f"fieldloss smooth1 (n={len(base)})", color="#4c72b0")
    ax.hist(new, bins=bins, alpha=0.55, label=f"{new_label} (n={len(new)})", color="#2ca02c")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("field relL2"); ax.set_ylabel("count")
    ax.set_title(f"Full test relL2 — {new_label} vs baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = ASSETS / out_name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

def pairwise_bar(bench_dir, new_model, out_name):
    lb = json.loads((bench_dir / "leaderboard.json").read_text())
    pw = lb["pairwise"]
    key = (
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_vs_"
        + new_model
    )
    d = pw[key]
    wins_new = d[new_model + "_wins"]
    wins_base = d["spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_wins"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["fieldloss\nsmooth1", Path(new_model).name.split("_")[-1]],
           [wins_base, wins_new], color=["#4c72b0", "#2ca02c"], edgecolor="k")
    ax.set_ylabel("cases won (lower relL2)")
    ax.set_title("Pairwise wins vs fieldloss (n=1994)")
    for i, v in enumerate([wins_base, wins_new]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    p = ASSETS / out_name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

def sparc_frac_bar(bench_dir, new_model, tag):
    families = ["sparc_1427", "sparc_1430", "sparc_1530"]
    models = [
        "spectrum_fno48_floor6_smooth1_qc_peak4",
        "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss",
        new_model,
    ]
    labels = ["peak4", "fieldloss", tag]
    data = {m: {} for m in models}
    with (bench_dir / "per_family.csv").open() as f:
        for r in csv.DictReader(f):
            if r["family"] in families and r["model"] in models:
                data[r["model"]][r["family"]] = float(r["frac_relL2_gt_1"])
    x = np.arange(len(families))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, lab) in enumerate(zip(models, labels)):
        vals = [data[m].get(fam, np.nan) for fam in families]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(families)
    ax.set_ylabel("frac(relL2 > 1)")
    ax.set_title(f"Sparc families — {tag} vs baselines")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.35, ax.get_ylim()[1]))
    fig.tight_layout()
    p = ASSETS / f"field_bench_frac_gt1_sparc_families_{tag}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

configs = [
    ("with_fieldloss_smooth05", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05",
     "smooth05", "field_bench_relL2_hist_peak4_vs_fieldloss_smooth05.png",
     "field_bench_pairwise_wins_fieldloss_smooth05.png"),
    ("with_fieldloss_smooth0", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0",
     "smooth0", "field_bench_relL2_hist_peak4_vs_fieldloss_smooth0.png",
     "field_bench_pairwise_wins_fieldloss_smooth0.png"),
]
for bench, model, tag, hist_out, pw_out in configs:
    bdir = REPO / "field_bench" / bench
    overlay_hist(bdir, model, f"fieldloss {tag}", hist_out)
    pairwise_bar(bdir, model, pw_out)
    sparc_frac_bar(bdir, model, tag)

# Copy field_bench per-model hists
for tag, model in [("smooth05", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05"),
                   ("smooth0", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0")]:
    src = REPO / "field_bench" / f"with_fieldloss_{tag}"
    for f in src.glob(f"relL2_hist_{model}.png"):
        dst = ASSETS / f"field_bench_relL2_hist_{tag}.png"
        import shutil
        shutil.copy2(f, dst)
        print(f"Copied {f.name} -> {dst}")

# Combined D vs D' distribution
def load_rel(bench, model):
    col = f"relL2_{model}"
    with (REPO / "field_bench" / bench / "per_case.csv").open() as f:
        return np.array([float(r[col]) for r in csv.DictReader(f)])
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2.5, 50)
for bench, model, label, color in [
    ("with_fieldloss_smooth05", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss", "fieldloss smooth1", "#4c72b0"),
    ("with_fieldloss_smooth05", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth05", "smooth05 (D)", "#9467bd"),
    ("with_fieldloss_smooth0", "spectrum_fno48_floor6_smooth1_qc_peak4_fieldloss_smooth0", "smooth0 (D')", "#2ca02c"),
]:
    v = load_rel(bench, model)
    ax.hist(v, bins=bins, alpha=0.5, label=f"{label} (n={len(v)})", color=color)
ax.axvline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("field relL2"); ax.set_ylabel("count")
ax.set_title("Run D / D' — full test relL2 distribution")
ax.legend(fontsize=8)
fig.tight_layout()
p = ASSETS / "field_bench_relL2_hist_smooth05_vs_smooth0.png"
fig.savefig(p, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {p}")
PY

echo ""
echo "=== postprocess complete $(date) ==="
