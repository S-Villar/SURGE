# Open questions — human decisions needed

This file is the checkpoint queue for the release sprint (see
`docs/RELEASE_SPRINT.md`). Whenever Composer hits a blocker it **cannot
safely decide**, it appends a dated entry here and stops until the
question is answered in-line (edit the file or reply in chat).

Format per entry:

```
## YYYY-MM-DD HHMM — <short title>
**Context:** <what task was being attempted>
**Blocker:** <what the agent does not know>
**Options:**
  1. <option A, with consequence>
  2. <option B, with consequence>
**Default if no answer in 12 h:** <safe fallback, or "halt">
**Resolution:** <filled by human>
```

---

## 2026-04-21 — **BLOCKER: NERSC home quota exceeded (109.3%)**

**Context:** Starting T1.1 of Night 1. `git status` fails with
`fatal: sha1 file '.../.git/index.lock' write error: Disk quota exceeded`.
`myquota` confirms `home 43.73 GiB / 40.00 GiB quota (109.3%)`. This halts
the entire release sprint — no commits, no reliable writes.

**The SURGE repo itself is fine** (4.0 GB total, .git is 158 MB). Something
else under `${HOME}/` is over budget.

**Options, ranked:**

1. **Move the SURGE repo to `$PSCRATCH`** (19.9% of 20 TiB used — massive
   headroom). Safest, fastest. Command pattern:
   ```
   cp -a ${SURGE_HOME} ${SURGE_SCRATCH}/src/SURGE
   # verify: diff -rq ${SURGE_HOME} /pscratch/.../SURGE
   # Then point work + `git worktree` at the scratch copy.
   ```
   Caveat: `$PSCRATCH` is **purged** periodically (typically 8-week
   untouched retention). Keep the `git` remote on GitHub as durable
   backup.

2. **Clean `$HOME` first** (`du -sh ~/* 2>/dev/null | sort -h | tail -20`)
   to get back under 40 GiB, then keep working in place. Takes whatever
   time the cleanup takes.

3. **Request a quota increase** via NERSC. Slow (~business day).

**Recommendation:** Option 1 (`$PSCRATCH`) for the sprint, Option 2 in
parallel when you have time. Do **not** do Option 3 alone — the sprint
waits.

**Resolution:** _(awaiting human)_

---

## 2026-04-21 — **Q5 auto-resolved by partial T1.1 audit**

History-rewrite decision was deferred to Night 3 (see Q5 below). The
partial audit found these sensitive items still present in past commits:

- `scripts/xgc_*.py` (7 files)
- `docs/xgc/*.md` (8+ files: `AI_OCLF_ARCHITECTURE.md`,
  `DRIFT_DETECTION_POLICY.md`, `XGC_INPUT_STRUCTURE.md`, `XGC_TRAINING.md`,
  `XGC_CAPABILITIES.md`, `XGC_SUMMARY.md`, `XGC_APARALLEL_DATA.md`,
  `XGC_FEATURES_2025-03.md`, `COMMITS_TO_PUSH.md`, `CHANGES_XGC_VIZ_2025-03.md`)
- `xgc_cpp/` directory (`CMakeLists.txt`, `main_test.cpp`,
  `xgc_surrogate_infer.cpp`/`.h`, `README.md`)
- `data/datasets/HHFW-NSTX/PwE_.pkl`, `PwIF_.pkl`
- `scripts/ml_utils.py`

**Decision (auto):** Night 3 **will** run `git filter-repo` per
`docs/PUBLIC_OPEN_SOURCE_PLAN.md`. The repo **cannot** be silently flipped
public in its current history state.

---

## 2026-04-21 — Queued at plan-drafting time (pre-sprint)

The following decisions must be answered before Night 1 starts. None of
them are blocking the plan doc itself, but Composer will stop on them.

### Q1 — `SECURITY.md` contact email
**Context:** Night 1 T1.5 creates `SECURITY.md` for the public repo.
**Resolution (2026-04-21):** `asvillar@princeton.edu`.

### Q2 — UCI Energy Efficiency: in-repo copy or runtime fetch?
**Context:** Night 2 T2.1 ships rung-4 (8 in → 2 out) regression dataset.
**Resolution (2026-04-21):** Runtime fetch, cache to `~/.cache/surge/`
(or `$SURGE_CACHE_DIR`), keep on disk after first run, do **not** auto-delete.
Implement as `surge.datasets.uci_energy.load()` that downloads the raw
xlsx/csv, verifies a checksum, converts once, and hits the cache on repeat
calls. Tests that require it are skipped when offline so CI on a blocked
runner does not fail.

### Q3 — Author / affiliation block in `CITATION.cff`
**Context:** Night 1 T1.5 writes `CITATION.cff`.
**Resolution (2026-04-21):**
  - name: Álvaro Sánchez Villar
  - ORCID: `0000-0003-3727-9319`
  - affiliation: left blank for now (can add later).

### Q4 — Public repo org / name
**Context:** Night 3 T3.3 replaces `REPLACE_ORG` everywhere.
**Resolution (2026-04-21):** Night 3 target is **personal GitHub account**
`S-Villar` (confirmed via `git remote -v` on the NERSC checkout:
`origin = git@github.com:S-Villar/SURGE.git`). All five URL sites
(`pyproject.toml`, `README.md`, `CITATION.cff`, `CONTRIBUTING.md`, and
`.github/ISSUE_TEMPLATE/config.yml`) were updated early during T1.6 so the
local repo already matches the intended push target. Longer-term target
org remains `PrincetonUniversity` — plan the move as a v0.1.1 follow-up
via GitHub's "Transfer ownership" flow (keeps stars/issues/PRs intact);
when it happens, sweep those same five files.

### Q5 — History rewrite decision
**Context:** Night 3 T3.2. Decided **by the Night 1 audit output**, not
up-front.
**Resolution (2026-04-21):** Deferred to Night 3 — will be decided
automatically based on `docs/AUDIT_NIGHT1.md` findings (clean history →
no rewrite; dirty history → run `git filter-repo` per
`docs/PUBLIC_OPEN_SOURCE_PLAN.md`).
