# Public open-source release — single source of truth

This file consolidates planning that was discussed for a **DOE-visible, citable** public repository: what to **strip from history**, what is **already done in the working tree**, **Opus-aligned** steps (BSD-3, `pyproject`, HHFW untracked, two remotes, DOE CODE), and **follow-on product work** (DDStore, resources API, etc.). Use it as the checklist before you give **publications** a GitHub URL.

For code-level cleanup (duplicate registries, `pytorch.py` vs `pytorch_impl.py` naming, resources/workers policy, dataset ladder, CI regression), see [REFACTORING_PLAN.md](REFACTORING_PLAN.md).

---

## A. Why you might keep a private repo (optional, human)

Publishing is a **policy and visibility** choice, not a moral requirement. It is normal to:

- Keep a **private** clone for aggressive experiments, paths, and half-finished ideas.
- Ship a **curated public** history that is **useful and redistributable**, not a byte-for-byte mirror of every local trial.

If staying private feels better for a season, that is compatible with **continued research**; you can still prepare a public drop when the lab and your comfort level align. This document stays valid either way.

---

## B. Non-negotiables for a public URL (summary)

| Item | Status (in repo) | Your action |
|------|------------------|-------------|
| License **BSD-3-Clause** | `LICENSE` updated | Counsel / OTT sign-off |
| Funding + disclaimer | `NOTICE` | Replace disclaimer with **lab-approved** text if required |
| Installable package | `pyproject.toml` (`surge-ml` dist name) | `pip install -e .` locally; set real **Homepage** URL |
| Root README | `README.md` | Set **public GitHub URL** for publications |
| HHFW binaries in **git** | Removed from index; `.gitignore` | **History** still needs cleanup — see §E |
| Legacy `scripts/ml_utils.py` | **Removed** from tree | None |
| XGC **dev-only** trees | **Removed** from tree — see §D | None for this commit; **history** still needs §E |

---

## C. Opus-aligned Git strategy (reminder)

- **One working copy**, **two remotes** is fine: e.g. `origin` = private, `public` = GitHub.
- **Publish only** the branch you **rewrote** (after `filter-repo`); do not push the uncleared history to the public remote.
- **`git filter-repo`** removes named paths from **all commits**; you **keep commit count and narrative**, with **new commit hashes** (history rewritten). **Orphan squash** = one commit — **not** the same as preserving history.
- **DOE CODE / OSTI** uses the **public** repo URL you register — not Zenodo as primary, unless policy changes.

---

## D. Paths removed from the *working tree* for public hygiene (this commit)

These are **application / lab-specific** or **legacy**; they should **not** appear in the public **release** tree. (Git **history** may still contain them until you run §E.)

| Path | Reason |
|------|--------|
| `scripts/ml_utils.py` | Legacy monolith; public API is `SurrogateEngine` + workflow. |
| `scripts/xgc_*.py` | Personal / application scripts (inference, animations, comparisons), not the core library. |
| `xgc_cpp/` | C++ inference template / tests tied to a specific workflow. |
| `docs/xgc/` | Proprietary or collaboration-specific documentation; not for general redistribution. |

**Kept (for now):** `surge/datasets/xgc.py` and `format=xgc` in `SurrogateDataset` — a **generic stacked-`.npy` directory** loader. Docstrings in code were **genericized** (no institution names). A later refactor can **rename** `XGCDataset` / helpers to neutral names (tracked in [ROADMAP.md](ROADMAP.md)).

---

## E. `git filter-repo` — make old blobs **unreachable** (you run on a *clone*)

**Do not** run this on your only copy. **Clone** → **filter** → **verify** → **push to public remote**.

Install: `pip install git-filter-repo` (or your HPC module).

**Example** — adjust paths after reviewing `git log -- path`:

```bash
git clone /path/to/SURGE.git SURGE_public_prep
cd SURGE_public_prep
# Exclude paths from ALL of history (see git-filter-repo docs: --path … --invert-paths):
git filter-repo --force \
  --path data/datasets/HHFW-NSTX/PwE_.pkl \
  --path data/datasets/HHFW-NSTX/PwIF_.pkl \
  --path scripts/ml_utils.py \
  --path scripts/xgc_datastreamset_comparison.py \
  --path scripts/xgc_inference.py \
  --path scripts/xgc_spatial_aparallel_animation.py \
  --path scripts/xgc_predictions_animation.py \
  --path docs/xgc \
  --path xgc_cpp \
  --invert-paths
```

*Notes:*

- With `--invert-paths`, each `--path` is **excluded** from the rewritten history. Re-run the same command after editing the list. Use `git filter-repo -h` and test on a **throwaway** clone. For glob patterns, see `--path-glob` in the tool’s help.
- Add any other **large** or **sensitive** paths (e.g. old notebooks, `docs/agents` if you exclude them) in consultation with your lab.
- After rewrite: `git reflog expire --all && git gc --prune=now` (optional) on the public clone.

---

## F. Implementation already recorded elsewhere (by topic)

| Topic | Where |
|--------|--------|
| Product roadmap (CV, big data, viz) | [ROADMAP.md](ROADMAP.md) |
| Prerelease “what works / what doesn’t” | [PRERELEASE.md](PRERELEASE.md) |
| Scale, DDStore, ezpz, resource validation | [dev/EXTENDED_ROADMAP.md](dev/EXTENDED_ROADMAP.md) |
| `pyproject` / README / license files | repository root |

**Next engineering passes (not blocking a minimal public tag, but planned):**  
`ResourceRequest` in workflow, sklearn vs multi-GPU **warnings**, **DDStore** backend, sharded `Dataset`, DDP **torch** path — see **EXTENDED** and **ROADMAP**.

---

## G. Checklist before sending URL to publications

- [ ] Lab sign-off on license + NOTICE wording.
- [ ] `filter-repo` (or lab-approved process) on a **clone**; verify sensitive paths **gone** with `git log -- path`.
- [ ] `git remote add public …` and `git push public main --tags` (or your branch name).
- [ ] Set GitHub “About” + **public** URL in `pyproject.toml` and `README.md`.
- [ ] File **DOE CODE** with publications using that URL + `v0.1.0` (or agreed tag).
- [ ] Optional: [CITATION.cff](https://citation-file-format.github.io/) after DOI known.

---

*This document is the **single** place that ties **history hygiene**, **tree exclusions**, and **roadmap** for the public release; other docs point here to avoid duplicate narratives.*
