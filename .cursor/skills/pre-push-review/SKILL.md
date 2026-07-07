---
name: pre-push-review
description: >-
  Analyze un-pushed changes on the current git branch (commits ahead of upstream
  plus uncommitted/untracked work), produce a categorized change report, and
  run a public-repo rigor checklist before push. Use when the user asks for a
  pre-push review, push readiness check, or invokes pre-push-review. Do NOT
  compare against main/master — use the companion branch-review skill for that.
disable-model-invocation: true
---

# Pre-Push Review

## Usage

Invoke explicitly before pushing to a public remote:

- In chat: `@pre-push-review` or "run pre-push-review"
- Or ask: "review my un-pushed changes before I push"

**Scope:** only work on the **current branch** that has **not yet been pushed** to its upstream. Do **not** diff against `main`/`master` — that belongs to a separate **branch-review** skill (same output format; compare against the PR base branch instead of upstream).

**Requirements:** run from the repository root. Execute the git commands below every time — never assume prior conversation context.

---

## Workflow

Copy this checklist and mark steps as you go:

```
Pre-push review progress:
- [ ] Step 1: Gather changes (upstream scope)
- [ ] Step 2: Summarize by conceptual change
- [ ] Step 3: Map files to each conceptual change
- [ ] Step 4: Categorize each change
- [ ] Step 5: Rigor review (verify against actual diff)
- [ ] Step 6: Produce final report
```

---

## Step 1 — Gather changes

Run these commands **in parallel** from the repo root:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse --abbrev-ref @{upstream} 2>/dev/null || echo "NO_UPSTREAM"
git status --porcelain=v1
git log --oneline @{upstream}..HEAD 2>/dev/null || true
git diff --stat @{upstream}...HEAD 2>/dev/null || true
git diff @{upstream}...HEAD 2>/dev/null || true
git diff HEAD
git diff --cached
git ls-files --others --exclude-standard
```

### Scope rules

| Upstream | Analyze |
|----------|---------|
| **Exists** | Union of: (a) `git diff @{upstream}...HEAD` — local commits not on remote, and (b) uncommitted changes vs `HEAD` (`git diff HEAD`, `git diff --cached`) plus untracked files |
| **Missing** (`NO_UPSTREAM`) | Uncommitted vs `HEAD` + untracked only. Note in the report: *no upstream — uncommitted only* |

**Do not** run `git diff main...HEAD` or compare to any base branch.

### Classify each touched file

From `git status --porcelain` and diffs:

| Prefix | Status |
|--------|--------|
| ` M`, `M `, `MM` | modified |
| `A `, `??` | added |
| `D ` | deleted |
| `R ` | renamed (read rename target from status) |

Read full diffs for any file you will describe. For untracked files, read their contents.

---

## Step 2 — Summarize by conceptual change

Group changes into **conceptual changes** — logical units of work, not one section per file.

Examples of good groupings:

- "Add canonical ConStellaration split and wire paper benchmark runner"
- "Normalize artifact paths to forward slashes for Windows portability"
- "Fix MLP ensemble training log streaming via ProgressList"

Examples of bad groupings:

- "Changes to leaderboard.py" (file-centric, not conceptual)
- "Misc updates" (too vague)

Each conceptual change gets **one line** summarizing the intent.

---

## Step 3 — Map files to each conceptual change

Under each conceptual change, list affected files:

```markdown
Affected files:
- `path/to/file` — brief note on what changed *for this concept*
```

A file may appear under multiple conceptual changes if it spans concerns.

---

## Step 4 — Categorize each change

Tag every conceptual change with exactly one primary category:

| Label | Use when |
|-------|----------|
| `[bug]` | Fixes incorrect behavior |
| `[feature]` | New capability or user-visible behavior |
| `[refactor]` | Restructure without intended behavior change |
| `[docs]` | Documentation only |
| `[test]` | Tests only |
| `[chore]` | Tooling, config, housekeeping |
| `[perf]` | Performance improvement |
| `[style]` | Formatting, naming, lint-only |
| `[security]` | Security hardening or vulnerability fix |
| `[breaking-change]` | Intentional API/contract break |

Use machine-parseable `[category]` prefix in the heading. Add a secondary tag only if truly needed (e.g. `[feature][breaking-change]`).

---

## Step 5 — Rigor review (public-repo readiness)

Verify each check against the **actual diff and file contents**. Search explicitly when needed:

```bash
# Examples — adapt patterns to the repo
git diff @{upstream}...HEAD 2>/dev/null; git diff HEAD
rg -i '(api[_-]?key|secret|password|token|credential|BEGIN (RSA|OPENSSH)|aws_access)' --glob '!*.lock' .
rg '(TODO|FIXME|XXX|HACK|console\.log|debugger|pdb\.set_trace|breakpoint\(\)|print\()' 
```

### Required checks

| Check | How to verify |
|-------|---------------|
| Secrets/credentials exposed | Scan diff for keys, tokens, `.env`, PEM blocks, connection strings |
| PII / internal-only references | Hostnames, employee names, internal URLs, private bucket paths |
| Debug / dead code | `console.log`, `print(`, `debugger`, `pdb`, large commented-out blocks |
| Tests | New logic without tests? Do existing tests need updates? Run tests if feasible |
| Breaking changes | Public API, config schema, CLI flags, defaults changed? |
| Documentation | README, docstrings, changelog updated for user-visible changes? |
| Dependencies | New packages intentional? License acceptable? |
| Commit hygiene | Changes coherent? Conventional-commit style suggested for each concept? |
| Error handling | Edge cases, fail modes, validation present? |

### Status values

| Status | Meaning |
|--------|---------|
| `PASS` | Verified clean or adequately addressed |
| `FAIL` | Definite blocker — must fix before push |
| `⚠️` | Needs attention — judgment call or could not fully verify |

Every row must include a **short justification** citing specific files/lines when relevant — not just the question restated.

---

## Step 6 — Output format

Produce Markdown **exactly** in this structure (fill in all sections):

```markdown
## Change Analysis
_Scope: un-pushed changes on branch `<branch>` (vs `<upstream>` | no upstream — uncommitted only)_

### [category] One-line summary of conceptual change 1
Affected files:
- `path/to/file` — what changed here
- `path/to/other` — what changed here

### [category] One-line summary of conceptual change 2
Affected files:
- `path/to/file` — what changed here

## Pre-Push Rigor Review
| Check | Status | Notes |
|-------|--------|-------|
| Secrets/credentials exposed | PASS/FAIL/⚠️ | ... |
| PII / internal-only references | PASS/FAIL/⚠️ | ... |
| Debug statements / dead code | PASS/FAIL/⚠️ | ... |
| Tests adequate and passing | PASS/FAIL/⚠️ | ... |
| Breaking changes documented | PASS/FAIL/⚠️ | ... |
| Documentation updated | PASS/FAIL/⚠️ | ... |
| Dependencies intentional and licensed | PASS/FAIL/⚠️ | ... |
| Commit scope coherent (conventional commits) | PASS/FAIL/⚠️ | ... |
| Error handling and edge cases | PASS/FAIL/⚠️ | ... |

## Verdict
Ready to push: YES / NO
Blocking issues:
- <item> or "none"
```

### Verdict rules

- **YES** only if zero `FAIL` items and no unresolved secrets/security issues.
- **NO** if any `FAIL`, or secrets/PII found, or critical tests failing.
- `⚠️` items alone do not block — list them but allow YES with caveats.

---

## Guidance

- Prefer **fewer, broader conceptual changes** over many tiny ones.
- When the diff is large, read by conceptual area — do not skip files.
- If there are unpushed commits **and** uncommitted changes, analyze both and note which files are only local/uncommitted.
- Suggest **conventional-commit messages** per conceptual change when commits are not yet made or are messy.
- Do not run `git push` unless the user explicitly asks after a YES verdict.
- If the repo has no commits yet or is not a git repo, report that and stop.

---

## Companion skill

A **branch-review** skill (diff vs. the PR base branch, e.g. `main`) is intended as a separate companion. It should reuse this same output format for consistency. This skill (`pre-push-review`) is strictly for **un-pushed work on the current branch vs. its upstream**.
