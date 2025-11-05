#!/usr/bin/env python3
"""Batch dataset generator for SURGE.

This script orchestrates dataset batch creation using surge.datagen.DataGenerator.
Configure via a YAML file and/or CLI overrides.

Three modes are supported:
  1) No equilibria set: creates batchN/case1..caseN and modifies a copied input file
  2) Equilibria set (set, fixed, or per_case): 
     - "set": creates batch_N/run1..runN, each run contains all equilibria with same params
     - "fixed": creates batch_N/run1..runN, all runs use the same single equilibrium
     - "per_case": creates batch_N/<equilibrium>/run1..runN, independent params per equilibrium

Example usage:
  python surge_batch_setup.py --config examples/batch_setup.yml

YAML schema (keys):
  out_root: <dir where new batch_N is created>
  inpfile: <full path to reference input file to copy and edit>
           IMPORTANT: Always provide this to ensure correct Fortran namelist structure.
           The template file ensures parameters are in the right namelist blocks.
  params: [name1, name2, ...]
  ranges: [[lo1, hi1], [lo2, hi2], ...]
  integer_mask: [true/false, ...]
  nsamples: 5
  spl: lhs | random
  seed: 42
  equilibria: null | set | fixed | per_case
  eqsetpath: <path to a batch or to a run folder containing sparc_* directories>
  use_python_replacement: true
  confirm_dirs: false
  save_plots: true

Requirements:
  This script requires numpy, scipy, and optionally yaml.
  If using a conda environment, activate it first:
    conda activate surge
  Or ensure the required packages are installed in your current Python environment.
"""

from __future__ import annotations

import sys
import os

# Check for required dependencies and provide helpful error messages
_missing_deps = []
try:
	import numpy  # type: ignore
except ImportError:
	_missing_deps.append("numpy")

try:
	import scipy  # type: ignore
except ImportError:
	_missing_deps.append("scipy")

if _missing_deps:
	_surge_env_hint = ""
	# Check if conda is available and suggest activating surge environment
	try:
		import subprocess
		result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, timeout=2)
		if "surge" in result.stdout:
			_surge_env_hint = "\n  💡 Try activating the surge conda environment:\n     conda activate surge\n"
	except Exception:
		pass
	
	print("❌ Missing required dependencies:", ", ".join(_missing_deps), file=sys.stderr)
	print(f"\n  Please install the missing packages:", file=sys.stderr)
	print(f"     pip install {' '.join(_missing_deps)}", file=sys.stderr)
	print(_surge_env_hint, file=sys.stderr)
	sys.exit(1)

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
	import yaml  # type: ignore
except Exception:
	yaml = None  # handled later if --config is used

# Import DataGenerator directly using importlib to bypass __init__.py (which imports torch)
import importlib.util
spec = importlib.util.spec_from_file_location("datagen", os.path.join(os.path.dirname(__file__), "surge", "datagen.py"))
datagen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datagen_module)
DataGenerator = datagen_module.DataGenerator


def _load_config(path: Optional[str]) -> Dict[str, Any]:
	if not path:
		return {}
	if yaml is None:
		raise RuntimeError("PyYAML is required to read the config file. Please `pip install pyyaml`. ")
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError("Configuration root must be a mapping/dictionary")
	return data


def _next_batch_dir(out_root: str, prefer_underscore: bool = True) -> str:
	os.makedirs(out_root, exist_ok=True)
	entries = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
	nums: List[int] = []
	# Support both patterns: batch_# and batch#
	for d in entries:
		if d.startswith("batch_"):
			sfx = d[len("batch_") :]
			if sfx.isdigit():
				nums.append(int(sfx))
		elif d.startswith("batch"):
			sfx = d[len("batch") :]
			if sfx.isdigit():
				nums.append(int(sfx))
	nxt = (max(nums) + 1) if nums else 1
	name = f"batch_{nxt}" if prefer_underscore else f"batch{nxt}"
	return os.path.join(out_root, name)


def _coerce_bool_list(x: Sequence[Any]) -> List[bool]:
	out: List[bool] = []
	for v in x:
		if isinstance(v, bool):
			out.append(v)
		elif isinstance(v, (int, float)):
			out.append(bool(int(v)))
		elif isinstance(v, str):
			out.append(v.strip().lower() in {"true", "1", "yes", "y"})
		else:
			raise ValueError(f"Cannot coerce {v!r} to bool")
	return out


def _derive_source_run_dir(eqsetpath: Optional[str], inpfile: Optional[str]) -> Optional[str]:
	"""Pick the directory that contains sparc_* case folders.

	Preference order:
	  - dirname(inpfile) if it exists and contains sparc_* subfolders
	  - eqsetpath if it exists and contains sparc_* subfolders
	  - eqsetpath/run1 if that exists and contains sparc_* subfolders
	  - None (not found)
	"""
	def has_sparc_children(p: str) -> bool:
		try:
			return any(
				os.path.isdir(os.path.join(p, d)) and d.startswith("sparc_") for d in os.listdir(p)
			)
		except Exception:
			return False

	cand: List[str] = []
	if inpfile:
		dr = os.path.dirname(inpfile)
		if os.path.isdir(dr):
			cand.append(dr)
	if eqsetpath and os.path.isdir(eqsetpath):
		cand.append(eqsetpath)
		run1 = os.path.join(eqsetpath, "run1")
		if os.path.isdir(run1):
			cand.append(run1)
	for p in cand:
		if has_sparc_children(p):
			return p
	return None


def run_from_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> str:
	# Merge overrides (CLI wins)
	cfg = {**config, **{k: v for k, v in overrides.items() if v is not None}}

	params: List[str] = cfg.get("params") or cfg.get("inpnames")
	ranges: List[List[float]] = cfg.get("ranges")
	integer_mask: List[Any] = cfg.get("integer_mask")
	log_space: Optional[List[Any]] = cfg.get("log_space")
	nsamples: int = int(cfg.get("nsamples", 5))
	spl: str = str(cfg.get("spl", cfg.get("method", "lhs"))).lower()
	seed: Optional[int] = cfg.get("seed")
	equilibria: Optional[str] = cfg.get("equilibria")
	eqsetpath: Optional[str] = cfg.get("eqsetpath")
	out_root: Optional[str] = cfg.get("out_root")
	inpfile: Optional[str] = cfg.get("inpfile")
	# scratch: CLI override takes precedence, then config, default to True
	if "scratch" in overrides and overrides["scratch"] is not None:
		scratch = bool(overrides["scratch"])
	elif "scratch" in cfg and cfg["scratch"] is not None:
		scratch = bool(cfg["scratch"])
	else:
		scratch = True  # Default to True
	confirm_dirs: bool = bool(cfg.get("confirm_dirs", False))
	save_plots: bool = bool(cfg.get("save_plots", True))
	use_python_replacement: bool = bool(cfg.get("use_python_replacement", True))
	dry_run: bool = bool(cfg.get("dry_run", False))

	if not params or not ranges or not integer_mask:
		raise ValueError("params, ranges, and integer_mask are required in config or CLI")
	if len(params) != len(ranges) or len(params) != len(integer_mask):
		raise ValueError("Length mismatch among params, ranges, and integer_mask")
	integer_mask_b = _coerce_bool_list(integer_mask)
	if log_space is not None:
		if len(log_space) != len(params):
			raise ValueError("Length mismatch: log_space must match params length")
		log_space_b = _coerce_bool_list(log_space)
	else:
		log_space_b = [False] * len(params)
	method = "lhs" if spl in ("lhs", "latin", "latin_hypercube") else ("random" if spl in ("random", "uniform") else spl)

	if not out_root:
		# Default to $SCRATCH/mp288/jobs if scratch=True, otherwise use SURGE/examples/datagen
		if scratch:
			scratch_dir = os.environ.get("SCRATCH")
			if scratch_dir:
				out_root = os.path.join(scratch_dir, "mp288", "jobs")
				if not dry_run:
					print(f"Using scratch directory: {out_root}")
			else:
				print("[warn] SCRATCH environment variable not set, falling back to default location")
				surge_root = os.path.dirname(os.path.abspath(__file__))
				out_root = os.path.join(surge_root, "examples", "datagen")
		else:
			# default to SURGE/examples/datagen
			surge_root = os.path.dirname(os.path.abspath(__file__))
			out_root = os.path.join(surge_root, "examples", "datagen")
	os.makedirs(out_root, exist_ok=True)

	if not inpfile or not os.path.isfile(inpfile):
		raise FileNotFoundError(f"Reference input file not found: {inpfile}")

	inputfilename = os.path.basename(inpfile)
	base_case_dir = os.path.dirname(inpfile)

	gen = DataGenerator(dry_run=dry_run, use_python_replacement=use_python_replacement)

	# Equilibria mode
	if equilibria:
		equilibria = str(equilibria).lower()
		if equilibria not in ("fixed", "set", "per_case"):
			raise ValueError("equilibria must be one of: fixed | set | per_case")

		# Decide the source_run_dir that contains sparc_* folders
		source_run_dir = _derive_source_run_dir(eqsetpath, inpfile)
		if source_run_dir is None:
			raise FileNotFoundError(
				"Could not determine equilibria source_run_dir. Provide eqsetpath (to a run folder) or set inpfile within a run folder."
			)

		# Create a new batch directory to hold the runs
		batch_dir = _next_batch_dir(out_root, prefer_underscore=True)
		if dry_run:
			print(f"[dry-run] would create {batch_dir}")
		else:
			os.makedirs(batch_dir, exist_ok=False)

		# Copy the reference input file into the batch root for record-keeping
		try:
			if not dry_run:
				dst_ref = os.path.join(batch_dir, inputfilename)
				if os.path.abspath(inpfile) != os.path.abspath(dst_ref):
					import shutil

					shutil.copy2(inpfile, dst_ref)
		except Exception as e:
			print(f"[warn] failed to copy reference input file: {e}")

		created = gen.generate_runs_from_equilibria(
			inpnames=params,
			inputfilename=inputfilename,
			source_run_dir=source_run_dir,
			ranges=ranges,
			integer_mask=integer_mask_b,
			n_runs=nsamples,
			method=method,
			out_root=batch_dir,
			equilibria_mode=equilibria,
			seed=seed,
			template_inpfile=inpfile,
			log_space=log_space_b,
		)

		# Save a top-level metadata file
		if not dry_run:
			meta_path = os.path.join(batch_dir, "meta.json")
			with open(meta_path, "w", encoding="utf-8") as f:
				json.dump(
					{
						"mode": "equilibria",
						"equilibria_mode": equilibria,  # "set", "fixed", or "per_case"
						"source_run_dir": source_run_dir,
						"params": params,
						"ranges": ranges,
						"integer_mask": integer_mask_b,
						"n_runs": nsamples,
						"method": method,
						"seed": seed,
					},
					f,
					indent=2,
				)

		return batch_dir

	# Simple cases mode (no equilibria)
	results = gen.generate(
		inpnames=params,
		inputfilename=inputfilename,
		ranges=ranges,
		integer_mask=integer_mask_b,
		n_samples=nsamples,
		method=method,
		base_case_dir=base_case_dir,
		batch_root=out_root,
		seed=seed,
		confirm_dirs=confirm_dirs,
		save_plots=save_plots,
	)

	batch_dir = gen.last_batch_dir or out_root

	# Copy the reference input file into the created batch dir for provenance
	try:
		if not dry_run and gen.last_batch_dir:
			import shutil

			dst_ref = os.path.join(gen.last_batch_dir, inputfilename)
			if os.path.abspath(inpfile) != os.path.abspath(dst_ref):
				shutil.copy2(inpfile, dst_ref)
	except Exception as e:
		print(f"[warn] failed to copy reference input file: {e}")

	return batch_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Create dataset batches using SURGE DataGenerator")
	p.add_argument("--config", type=str, help="Path to YAML configuration", default=None)

	# Common overrides
	p.add_argument("--out_root", type=str, default=None, help="Directory where batch_N is created")
	p.add_argument("--inpfile", type=str, default=None, help="Full path to the reference input file")
	p.add_argument("--nsamples", type=int, default=None, help="Number of samples (runs/cases)")
	p.add_argument("--spl", type=str, default=None, help="Sampling method: lhs | random")
	p.add_argument("--seed", type=int, default=None, help="Random seed")
	p.add_argument("--equilibria", type=str, default=None, help="fixed | per_case | (omit for none)")
	p.add_argument("--eqsetpath", type=str, default=None, help="Path to batch_0 or a run folder containing sparc_* cases")
	p.add_argument("--scratch", type=lambda x: x.lower() in ("true", "yes", "1"), default=None, help="Use $SCRATCH/mp288/jobs as out_root (default: true)")
	p.add_argument("--no-scratch", action="store_true", help="Disable scratch mode (use default SURGE/examples/datagen)")
	p.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
	p.add_argument("--use-python-replacement", action="store_true", help="Use in-Python replacement (no external scripts)")

	return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = parse_args(argv)
	cfg = _load_config(args.config)

	# Handle scratch flag (--no-scratch takes precedence over --scratch)
	scratch_override = None
	if args.no_scratch:
		scratch_override = False
	elif args.scratch is not None:
		scratch_override = args.scratch

	overrides = {
		"out_root": args.out_root,
		"inpfile": args.inpfile,
		"nsamples": args.nsamples,
		"spl": args.spl,
		"seed": args.seed,
		"equilibria": args.equilibria,
		"eqsetpath": args.eqsetpath,
		"scratch": scratch_override,
		"dry_run": args.dry_run,
		"use_python_replacement": args.use_python_replacement or cfg.get("use_python_replacement", True),
	}

	try:
		batch_dir = run_from_config(cfg, overrides)
	except Exception as e:
		print(f"[error] {e}", file=sys.stderr)
		return 1

	print(f"Created batch at: {batch_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

