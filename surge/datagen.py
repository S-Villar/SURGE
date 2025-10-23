"""SURGE datagenerator utilities.

Provides a simple DataGenerator class that can create parameter samples
using random uniform sampling or Latin Hypercube Sampling (LHS) and
write/replace values into input files by calling external shell scripts
from the HotPlasmaAI/bin folder (replace_var.sh).

API (high level):
  generator = DataGenerator(bin_dir=...)
  generator.generate(
      inpnames=[...],
      inputfilename='C1input',
      ranges=[[min,max],...],
      integer_mask=[True,False,...],
      n_samples=10,
      method='lhs',
      out_dir='/path/to/param_cases',
      base_case_dir='/path/to/case_default'
  )

The class will create case folders under out_dir named case_1, case_2, ...
and copy the base_case contents into each one before modifying the input
file by calling the replace_var.sh script.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Sequence, Optional
import json
from datetime import datetime

import numpy as np
from scipy.stats import qmc


class DataGenerator:
    """Generate parameter cases and modify input files using helper scripts.

    Args:
        bin_dir: directory where HotPlasmaAI/bin scripts live (replace_var.sh)
        replace_script: script name to call for replacing variables (default: replace_var.sh)
    """

    def __init__(
        self,
        bin_dir: Optional[str] = None,
        replace_script: str = "replace_var.sh",
        dry_run: bool = False,
        use_python_replacement: bool = False,
    ):
        self.bin_dir = bin_dir or os.path.join(os.environ.get("HOME", "."), "HotPlasmaAI", "bin")
        self.replace_script = replace_script
        # If dry_run is True, the generator will not execute external scripts and
        # will only print the replacement commands. Useful for testing on systems
        # that don't have the HotPlasmaAI scripts available.
        self.dry_run = bool(dry_run)
        # If True, perform in-Python replacement (no external sed/perl) and
        # avoid creating editor backup files like file.py~.
        self.use_python_replacement = bool(use_python_replacement)
        # Tracks the most recent batch directory created by generate()
        self.last_batch_dir = None

    def _call_replace(self, varname: str, value, filepath: str):
        script_path = os.path.join(self.bin_dir, self.replace_script)
        # Convert numpy types to python native
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()

        cmd = [script_path, str(varname), str(value), str(filepath)]

        if self.dry_run:
            # Just print the command and return
            print("[dry-run] would run:", " ".join(cmd))
            return

        if self.use_python_replacement:
            return self._replace_in_file(varname, value, filepath)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Replace script not found: {script_path}")

        # Use subprocess.run to execute the shell script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to run replace script: {e}")

    def _replace_in_file(self, varname: str, value, filepath: str):
        """Replace all occurrences of a variable assignment in a file.

        Matching rule: replace lines where the variable appears on the LHS of
        an assignment and the variable name is not part of a longer token. A
        valid boundary character before/after the name is start/end of string
        or one of: space, dot (.), or equals sign (=).
        """
        import re
        import tempfile

        # Convert numpy types
        if isinstance(value, (np.floating, np.integer)):
            value_to_write = repr(value.item())
        else:
            value_to_write = repr(value)

        # Pattern to find the variable as a standalone-ish token in a LHS
        # part. We will split lines at the first '=' and inspect the LHS.
        # The pattern asserts that the char before (if any) is start, space,
        # dot or '=' and similarly for char after.
        var_pat = re.compile(rf"(?:(?<=^)|(?<=[\s.=])){re.escape(varname)}(?:(?=$)|(?=[\s.=]))")

        replaced_count = 0
        dirpath = os.path.dirname(filepath) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_replace_", dir=dirpath)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf, open(filepath, "r", encoding="utf-8") as srcf:
                for line in srcf:
                    if "=" in line:
                        lhs, rhs = line.split("=", 1)
                        if var_pat.search(lhs):
                            # reconstruct line with new RHS
                            tmpf.write(f"{lhs.strip()} = {value_to_write}\n")
                            replaced_count += 1
                            continue
                    tmpf.write(line)

            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

        if replaced_count == 0:
            raise RuntimeError(f"Variable '{varname}' not found (per matching rules) in {filepath}")

    def _make_case(self, base_case_dir: str, out_dir: str, case_idx: int):
        """Create a case directory and copy only the template file to it.

        The template file name is provided by the caller (see generate).
        """
        wdir = os.path.join(out_dir, f"case{case_idx}")
        os.makedirs(wdir, exist_ok=False)

        if not os.path.isdir(base_case_dir):
            raise FileNotFoundError(f"Base case dir not found: {base_case_dir}")

        # Copy only the template file that will be modified (input file)
        # The caller passes the template filename and will place the file in
        # the correct place after creating the case directory.
        return wdir

    def generate(
        self,
        inpnames: Sequence[str],
        inputfilename: str,
        ranges: Sequence[Sequence[float]],
        integer_mask: Optional[Sequence[bool]] = None,
        n_samples: int = 10,
        method: str = "random",
        out_dir: Optional[str] = None,
        base_case_dir: Optional[str] = None,
        seed: Optional[int] = None,
        batch_root: Optional[str] = None,
        confirm_dirs: bool = True,
        save_plots: bool = False,
    ) -> List[dict]:
        """Generate parameter files.

        Returns a list of dicts with generated parameter values and case directory.
        """
        if integer_mask is None:
            integer_mask = [False] * len(inpnames)

        if len(inpnames) != len(ranges) or len(inpnames) != len(integer_mask):
            raise ValueError("Lengths of inpnames, ranges and integer_mask must match")

        out_dir = out_dir or os.path.join(os.environ.get("HOME", "."), "Simulations", "param_cases")
        base_case_dir = base_case_dir or os.path.join(out_dir, f"case_default")
        os.makedirs(out_dir, exist_ok=True)

        dim = len(inpnames)

        rng = np.random.default_rng(seed)

        if method.lower() == "lhs":
            # LatinHypercube in scipy may accept a `random_state` or not depending
            # on the scipy version. Use a sampler and then scale.
            # Note: older/newer scipy versions differ in LatinHypercube constructor
            # arguments. Keep the simplest compatible usage.
            sampler = qmc.LatinHypercube(d=dim)
            sample = sampler.random(n=n_samples)
            # scale to ranges using qmc.scale for each column
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                # sample[:, i] is in [0,1); scale manually to [lo, hi]
                sample[:, i] = sample[:, i] * (hi - lo) + lo
        elif method.lower() in ("random", "uniform"):
            sample = np.empty((n_samples, dim), dtype=float)
            for i in range(dim):
                lo, hi = ranges[i][0], ranges[i][1]
                sample[:, i] = rng.uniform(lo, hi, size=n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Batch management: prompt for directory if not provided and create batch folder
        results = []
        samples = np.zeros((n_samples, dim), dtype=float)

        # Determine batch root (non-interactive optional)
        if batch_root is None:
            surge_root = os.path.dirname(os.path.dirname(__file__))
            batch_root = os.path.join(surge_root, "examples", "datagen")

        # Create a new numbered batch directory under batch_root
        if not self.dry_run:
            if confirm_dirs:
                try:
                    perm = input(f"Allow creating batch directory and subdirectories at {batch_root}? [Y/n]: ").strip()
                except Exception:
                    perm = "y"
                if perm and perm.lower() not in ("y", "yes", ""):
                    raise PermissionError("User declined creation of batch directories")

        os.makedirs(batch_root, exist_ok=True)
        existing = [d for d in os.listdir(batch_root) if os.path.isdir(os.path.join(batch_root, d)) and d.startswith("batch")]
        idxs = [int(d.replace("batch", "")) for d in existing if d.replace("batch", "").isdigit()]
        next_idx = max(idxs) + 1 if idxs else 1
        batch_dir = os.path.join(batch_root, f"batch{next_idx}")
        if self.dry_run:
            print(f"[dry-run] would create batch directory: {batch_dir}")
        else:
            os.makedirs(batch_dir, exist_ok=False)
        # record last created batch dir
        self.last_batch_dir = batch_dir

        for idx in range(sample.shape[0]):
            caseidx = idx + 1
            # Create the case directory inside the batch_dir (case names: case1, case2,...)
            target_parent = batch_dir or out_dir
            if self.dry_run:
                wdir = os.path.join(target_parent, f"case{caseidx}")
                print(f"[dry-run] would create case directory: {wdir}")
            else:
                wdir = self._make_case(base_case_dir, target_parent, caseidx)

            # Copy the template file from base_case_dir into the new case dir
            src_template = os.path.join(base_case_dir, inputfilename)
            dst_template = os.path.join(wdir, inputfilename)
            if os.path.exists(src_template):
                if self.dry_run:
                    print(f"[dry-run] would copy template {src_template} -> {dst_template}")
                else:
                    shutil.copy2(src_template, dst_template)
            else:
                # fallback: try global_ns.py or model.py in base
                for alt in ("global_ns.py", "model.py"):
                    alt_src = os.path.join(base_case_dir, alt)
                    if os.path.exists(alt_src):
                        if self.dry_run:
                            print(f"[dry-run] would copy template {alt_src} -> {dst_template}")
                        else:
                            shutil.copy2(alt_src, dst_template)
                        break
                else:
                    # If template not found, create an empty file so replacement fails clearly
                    if self.dry_run:
                        print(f"[dry-run] template not found in base case: would create empty {dst_template}")
                    else:
                        open(dst_template, "w", encoding="utf-8").close()
            # Determine target file path to modify
            target_file = os.path.join(wdir, inputfilename)
            if not os.path.exists(target_file):
                # allow modifying global_ns.py or model.py if inputfilename not present
                # prefer global_ns.py in case directory
                if os.path.exists(os.path.join(wdir, "global_ns.py")):
                    target_file = os.path.join(wdir, "global_ns.py")
                elif os.path.exists(os.path.join(wdir, "model.py")):
                    target_file = os.path.join(wdir, "model.py")

            assigned = {}
            for j, name in enumerate(inpnames):
                val = sample[idx, j]
                if integer_mask[j]:
                    val = int(round(val))
                samples[idx, j] = val
                # call replace to change value in target file
                try:
                    self._call_replace(name, val, target_file)
                except Exception as e:
                    # continue but record error
                    assigned[name] = {"value": val, "error": str(e)}
                else:
                    assigned[name] = {"value": val}

            results.append({"case": f"case{caseidx}", "dir": wdir, "params": assigned})

        # If batch_dir provided, save samples and README
        if batch_dir is not None and not self.dry_run:
            meta = {
                "inpnames": list(inpnames),
                "ranges": list([list(r) for r in ranges]),
                "integer_mask": list(integer_mask),
                "n_samples": n_samples,
                "method": method,
                "seed": seed,
                "base_case_dir": base_case_dir,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            npz_path = os.path.join(batch_dir, "samples.npz")
            np.savez_compressed(npz_path, X=samples, names=np.array(inpnames), integer_mask=np.array(integer_mask), meta=json.dumps(meta))

            # Write README with Markdown tables
            readme_path = os.path.join(batch_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                bname = os.path.basename(batch_dir)
                f.write(f"# Batch {bname}\n\n")
                f.write(f"Created: {meta['created_at']}\n\n")

                # Parameters table
                f.write("## Parameters\n\n")
                f.write("| Variable | Range Min | Range Max | Integer |\n")
                f.write("|:--|--:|--:|:--:|\n")
                def _fmt(v, as_int=False):
                    try:
                        if as_int:
                            return str(int(round(float(v))))
                        # scientific notation with 4 significant figures
                        return f"{float(v):.3e}"
                    except Exception:
                        return str(v)

                for name, rng_, is_int in zip(inpnames, ranges, integer_mask):
                    f.write(
                        f"| {name} | {_fmt(rng_[0], as_int=is_int)} | {_fmt(rng_[1], as_int=is_int)} | {'Yes' if is_int else 'No'} |\n"
                    )

                # Sampling config table
                f.write("\n## Sampling\n\n")
                f.write("| Key | Value |\n")
                f.write("|:--|:--|\n")
                f.write(f"| method | {method} |\n")
                f.write(f"| n_samples | {n_samples} |\n")
                f.write(f"| seed | {seed} |\n")
                f.write(f"| base_case_dir | {base_case_dir} |\n")

                # Samples preview table (first 10)
                f.write("\n## Samples (first 10)\n\n")
                # header
                f.write("| " + " | ".join(inpnames) + " |\n")
                f.write("|" + "|".join([":-:" for _ in inpnames]) + "|\n")
                preview_rows = min(10, samples.shape[0])
                for i in range(preview_rows):
                    row = [
                        _fmt(samples[i, j], as_int=bool(integer_mask[j]))
                        for j in range(samples.shape[1])
                    ]
                    f.write("| " + " | ".join(row) + " |\n")

                # Summary statistics table
                f.write("\n## Summary stats (all samples)\n\n")
                f.write("| Variable | Min | Mean | Max |\n")
                f.write("|:--|--:|--:|--:|\n")
                for j, name in enumerate(inpnames):
                    col = samples[:, j]
                    f.write(
                        f"| {name} | {_fmt(col.min(), as_int=bool(integer_mask[j]))} | {_fmt(col.mean(), as_int=False)} | {_fmt(col.max(), as_int=bool(integer_mask[j]))} |\n"
                    )

            # Optional: save sampling plots
            if save_plots:
                try:
                    save_sampling_plots(samples, list(inpnames), batch_dir, title=f"{os.path.basename(batch_dir)}")
                except Exception as e:
                    print(f"[warn] failed to save plots: {e}")

        return results


def example_usage():
    """Simple example for local testing."""
    gen = DataGenerator()
    res = gen.generate(
        inpnames=["nemax", "temax"],
        inputfilename="global_ns.py",
        ranges=[[1e18, 5e18], [5, 100]],
        integer_mask=[False, False],
        n_samples=3,
        method="lhs",
        out_dir=os.path.join(os.environ.get("HOME", "."), "Simulations", "Petra-HOT", "1D_TOMATOR", "param_cases"),
        base_case_dir=os.path.join(os.environ.get("HOME", "."), "Simulations", "Petra-HOT", "1D_TOMATOR", "case_default_neLHS_2"),
        seed=42,
    )
    for r in res:
        print(r)


# -------- Visualization helpers --------
def save_sampling_plots(X: np.ndarray, names: Sequence[str], out_dir: str, title: Optional[str] = None):
    """Save simple visualizations of the sampled dataset.

    Produces:
      - sampling_hist.png: 1D histograms for each variable
      - sampling_scatter.png: pairwise scatter matrix (up to 4 variables to avoid clutter)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    n, d = X.shape
    title = title or "Sampling"

    # 1) Histograms
    cols = min(4, d)
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols + 2, 3.2 * rows), squeeze=False)
    for j in range(d):
        ax = axes[j // cols][j % cols]
        ax.hist(X[:, j], bins=12, color="#4472c4", alpha=0.85)
        ax.set_title(names[j])
    # hide empty subplots
    for k in range(d, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle(f"{title} – Histograms (n={n})", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sampling_hist.png"), dpi=150)
    plt.close(fig)

    # 2) Pairwise scatter (first up to 4 vars)
    m = min(4, d)
    if m >= 2:
        fig2, axes2 = plt.subplots(m, m, figsize=(3.2 * m + 2, 3.2 * m), squeeze=False)
        for i in range(m):
            for j in range(m):
                ax = axes2[i][j]
                if i == j:
                    ax.hist(X[:, j], bins=12, color="#70ad47", alpha=0.85)
                else:
                    ax.scatter(X[:, j], X[:, i], s=14, alpha=0.7, color="#2f5597")
                if i == m - 1:
                    ax.set_xlabel(names[j])
                if j == 0:
                    ax.set_ylabel(names[i])
        fig2.suptitle(f"{title} – Pairwise (first {m} vars)", y=0.98)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "sampling_scatter.png"), dpi=150)
        plt.close(fig2)
