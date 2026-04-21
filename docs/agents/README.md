# SURGE agent skills and blueprints

This directory is a **local skill catalog** for AI agents (and humans) working on **SURGE** and **M3DC1** surrogates. It is intentionally shaped like an **agentic blueprint catalog**: explicit triggers, preconditions, procedures, and failure modes—similar in spirit to the [Agentic Blueprint Catalog](https://github.com/academy-agents/agentic_blueprint_catalog) from [Academy Agents](https://github.com/academy-agents).

> **Relation to Academy:** [Academy](https://github.com/academy-agents/academy) (`academy-py`) is a framework for **stateful, federated** agents (`@action`, `@loop`, managers, exchanges). This repo does **not** embed Academy today; these docs are **portable skill specs** you can later wrap as Academy tools/actions or map to HPC blueprints (Globus Compute, Parsl). See [`blueprint/README.md`](blueprint/README.md) for that mapping.

**External references**

- [Academy Agents organization](https://github.com/academy-agents)
- [Academy core (`academy-py`)](https://github.com/academy-agents/academy) — [docs](https://docs.academy-agents.org/latest/get-started)
- [Agentic Blueprint Catalog](https://github.com/academy-agents/agentic_blueprint_catalog) — HPC / federated patterns

---

## Catalog layout

```
docs/agents/
├── README.md                 ← You are here (catalog index)
├── skills-manifest.yaml      ← Machine-readable index of skills
├── SKILL_SCHEMA.md           ← How skill documents are structured
├── CONTEXT.md                ← Repo + workflow context (γ vs δp, paths, git policy)
├── blueprint/
│   └── README.md             ← SURGE-on-HPC patterns vs Academy blueprints
└── skills/
    ├── surge-slurm-bootstrap.md
    ├── surge-hpc-conda-parquet.md
    ├── surge-results-locator.md
    └── surge-m3dc1-eval-2d-delta-p.md
```

---

## Registered skills

| ID | Title | Spec |
|----|--------|------|
| `surge.slurm.bootstrap` | Slurm job uses correct repo root | [`skills/surge-slurm-bootstrap.md`](skills/surge-slurm-bootstrap.md) |
| `surge.hpc.conda_parquet` | Conda + pyarrow parity on compute nodes | [`skills/surge-hpc-conda-parquet.md`](skills/surge-hpc-conda-parquet.md) |
| `surge.results.locator` | Find γ vs δ*p* metrics, plots, run dirs | [`skills/surge-results-locator.md`](skills/surge-results-locator.md) |
| `surge.m3dc1.eval_2d_delta_p` | δ*p* 2D map evaluation CLI | [`skills/surge-m3dc1-eval-2d-delta-p.md`](skills/surge-m3dc1-eval-2d-delta-p.md) |

Load the manifest in automation: [`skills-manifest.yaml`](skills-manifest.yaml).

---

## Design notes (honest scope)

- **Started as session notes**; this refactor makes them **agent-consumable** (schema + manifest) but **not** a full Academy deployment.
- **Next steps** if you adopt Academy: implement each skill as a **tool** or **`@action`**, add **state** (run IDs, cluster), and reuse **blueprint** patterns from the catalog (e.g. batch directors, remote executors).

---

## Related SURGE docs

- [`m3dc1/results.md`](../../m3dc1/results.md) — metrics tables and figure paths
- [`scripts/m3dc1/surge_slurm_env.sh`](../../scripts/m3dc1/surge_slurm_env.sh) — shared Slurm Python setup
- [`docs/m3dc1/`](../m3dc1/) — M3DC1 guides
