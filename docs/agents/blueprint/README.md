# Blueprint mapping: SURGE on HPC ↔ Academy Agents

This file connects **local SURGE workflows** to the **pattern language** used by [Academy Agents](https://github.com/academy-agents) and the [Agentic Blueprint Catalog](https://github.com/academy-agents/agentic_blueprint_catalog). It is **not** an implementation—there is no `academy-py` dependency in SURGE today.

---

## Reference architecture (external)

| Resource | Role |
|----------|------|
| [Academy](https://github.com/academy-agents/academy) | Stateful agents: `@action`, `@loop`, managers, exchanges; federated execution. |
| [Agentic Blueprint Catalog](https://github.com/academy-agents/agentic_blueprint_catalog) | Reusable **federated**, **HPC hierarchical**, and **remote agent** patterns (Globus Compute, Parsl, etc.). |
| [docs.academy-agents.org](https://docs.academy-agents.org/latest/get-started) | Getting started for Academy. |

---

## Local “blueprints” (this repo)

These are **conceptual** IDs used in [`skills-manifest.yaml`](../skills-manifest.yaml) as `implements_blueprint`. They name SURGE-specific patterns an Academy port could wrap.

### `local-hpc-batch-bootstrap`

- **Idea:** A batch job must inherit the **correct filesystem context** (repo root, module paths, dataset paths).
- **SURGE manifestation:** `SLURM_SUBMIT_DIR`, `cd "$SURGE_ROOT"`, launcher scripts that `cd` before `sbatch`.
- **Academy analogy:** Pre-execution **executor** setup or a lightweight **director** that validates environment before launching training **tools**.
- **Catalog cousin:** *HPC hierarchical* jobs where each node assumes a consistent working directory and module stack—see catalog’s [`hpc_hierarchical/`](https://github.com/academy-agents/agentic_blueprint_catalog/tree/main/agentic_blueprint_catalog/hpc_hierarchical) narrative.

### `local-hpc-python-environment-parity`

- **Idea:** **Login node** and **compute node** must use the **same** intended Python stack (conda prefix, pyarrow, optional GPflow).
- **SURGE manifestation:** `surge_slurm_env.sh`, `module load conda`, `conda activate <prefix>`, `import pyarrow` check.
- **Academy analogy:** A **tool** precondition or a **setup action** run once per batch step; could be a **remote environment provision** step before model training actions.
- **Catalog cousin:** Federated / remote agents that assume a **prepared execution environment** on the endpoint—see [`federated/`](https://github.com/academy-agents/agentic_blueprint_catalog/tree/main/agentic_blueprint_catalog/federated).

---

## Suggested porting path (future work)

1. **Tools:** Map each skill in [`skills/`](../skills/) to a **single-responsibility tool**: e.g. `verify_surge_root`, `ensure_pyarrow`, `resolve_results_path`.
2. **Actions:** Thin **Academy** `@action` methods that call those tools with typed arguments (run_tag, cluster).
3. **Loops / directors:** For **trial suites**, a director could mirror *hierarchical* blueprint patterns: partition work, monitor `sacct`, harvest metrics.
4. **Federation:** Training on Perlmutter could become a **Globus Compute** endpoint task; SURGE stays the **computational unit** inside the remote function.

---

## Honest gap analysis

| Gap | Notes |
|-----|--------|
| No `academy-py` | Skills are Markdown + YAML manifest only. |
| No remote executor | All procedures assume SSH/Slurm usage documented in skills. |
| No typed tool API | Procedures are natural-language numbered steps; wrap when porting. |

This gap list is intentional: the catalog documents **intent** and **procedure** first; a framework bind is a second phase.
