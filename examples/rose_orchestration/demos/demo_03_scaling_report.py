import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_summary(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_summaries(root):
    return sorted(root.glob("**/summary.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Demo 3 scaling summaries into CSV/Markdown.")
    ap.add_argument("--input-root", default="demos/output/demo_03_scaling")
    ap.add_argument("--output-prefix", default="demos/output/demo_03_scaling/scaling_report")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary_path in _find_summaries(input_root):
        blob = _load_summary(summary_path)
        rows.append(
            {
                "label": summary_path.parent.name,
                "summary_path": str(summary_path),
                "available_cpus": blob.get("available_cpus"),
                "cpus_per_trial": blob.get("cpus_per_trial"),
                "parallel_trials": blob.get("parallel_trials"),
                "completed_trials": blob.get("completed_trials"),
                "wall_time_seconds": blob.get("wall_time_seconds"),
                "trials_per_hour": blob.get("trials_per_hour"),
                "best_val_r2": blob.get("best_val_r2"),
                "best_val_rmse": blob.get("best_val_rmse"),
                "best_family": blob.get("best_family"),
                "best_detail": blob.get("best_detail"),
                "total_rows": blob.get("total_rows"),
            }
        )

    rows.sort(key=lambda row: (int(row.get("available_cpus") or 0), int(row.get("parallel_trials") or 0)))

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "label",
        "available_cpus",
        "cpus_per_trial",
        "parallel_trials",
        "completed_trials",
        "wall_time_seconds",
        "trials_per_hour",
        "best_val_r2",
        "best_val_rmse",
        "best_family",
        "best_detail",
        "total_rows",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Demo 3 scaling report",
        "",
        "| label | alloc cpus | cpus/trial | parallel trials | completed | wall s | trials/hour | best R2 | best RMSE | best family | detail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['available_cpus']} | {row['cpus_per_trial']} | "
            f"{row['parallel_trials']} | {row['completed_trials']} | "
            f"{float(row['wall_time_seconds'] or 0):.1f} | {float(row['trials_per_hour'] or 0):.2f} | "
            f"{float(row['best_val_r2'] or 0):.5f} | {float(row['best_val_rmse'] or 0):.6f} | "
            f"{row['best_family']} | {row['best_detail']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
