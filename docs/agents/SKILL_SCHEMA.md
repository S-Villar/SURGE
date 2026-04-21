# Skill document schema (`docs/agents/skills/`)

Skills are **Markdown files** with a **YAML frontmatter** block so humans and tools can parse the same source.

## Required frontmatter keys

| Key | Type | Meaning |
|-----|------|---------|
| `skill_id` | string | Stable ID, e.g. `surge.slurm.bootstrap` (matches `skills-manifest.yaml`) |
| `title` | string | Short human title |
| `version` | string | Semver or date string for the spec |
| `domain` | list | Tags: `hpc`, `slurm`, `conda`, `surge`, `m3dc1`, … |
| `academy_blueprint_hint` | string or null | Optional link to a pattern in [Agentic Blueprint Catalog](https://github.com/academy-agents/agentic_blueprint_catalog) or local `blueprint/README.md` anchor |

## Required body sections

Use these level-2 headings in order (Academy / blueprint style):

1. **Objective** — one paragraph.
2. **When to apply (triggers)** — bullet list; mirror manifest `triggers`.
3. **Preconditions** — environment, paths, cluster assumptions.
4. **Procedure** — numbered steps an agent should execute.
5. **Verification** — how to know the skill succeeded.
6. **Failure modes & diagnostics** — symptoms → likely cause → next step.
7. **Artifacts & code references** — repo paths, logs, configs.
8. **Future: Academy mapping** — optional: suggested `@action` / tool decomposition.

## Optional

- **Tools** — shell commands, CLI entry points.
- **Safety** — e.g. do not commit multi-GB Parquet.

## Machine index

Regenerate or extend [`skills-manifest.yaml`](skills-manifest.yaml) when adding a skill so discovery stays programmatic.
