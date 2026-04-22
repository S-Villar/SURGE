# SURGE open-source release sprint (3 nights)

Goal: the GitHub repo looks like a **finished DOE open-source drop** — CI,
tests, one real example, clean docs, license/notice/citation — and is *safe
to flip public without announcing*, so that when publications pulls the URL
nothing disclosure-sensitive is there.

Companion documents:

- `docs/PUBLIC_OPEN_SOURCE_PLAN.md` — git-history rewrite + DOE CODE process
  (happens on **Night 3**).
- `docs/RELEASE_DEMO_PLAN.md` — the end-to-end user tour the README points to.
- `docs/REFACTORING_PLAN.md` — code-shape cleanup, `ResourceSpec`, dataset
  ladder, which this sprint draws from.
- `docs/ROADMAP.md` + `docs/dev/EXTENDED_ROADMAP.md` — post-v0.1.0 work.

Everything below is scoped for a **Composer-2.0-assisted** sprint. Tasks that
are *"stop-and-ask"* tasks are explicitly flagged; Composer must not guess
through them.

---

## Stop-and-query protocol (read this first)

Composer cannot send email. The mechanism is instead:

1. On any **STOP** condition listed below, Composer
   - appends a timestamped entry to `docs/OPEN_QUESTIONS.md` describing
     the blocker, its location, and the options,
   - stops all further writes for the current task,
   - reports the blocker in chat.
2. On any **policy-sensitive edit** (license text, NOTICE wording, DOE CODE
   metadata, anything touching author/affiliation strings, anything that
   changes git history), Composer proposes the diff and waits for a human
   "go" before applying.
3. Composer commits in **small chunks** (one logical task per commit) and
   never force-pushes. `git revert <hash>` is always the escape hatch.

### STOP conditions

- Any command surface inspection (`rg`, `grep`) finds **proprietary
  tokens** (`XGC`, `HHFW`, `C1input`, lab/project internal names that you
  have not whitelisted) outside the already-known locations.
- `git log --all -- <path>` shows a file in **history** that should not be
  in the public repo (history rewrite is a Night 3 human-run step).
- A rename / deletion would break a downstream consumer the agent cannot
  see (e.g. a config you keep on `$SCRATCH`, another NERSC project).
- A dependency bump would change the install story (e.g. pinning `torch>=2.2`
  breaks Python 3.9 wheels).
- Any test goes from green to red and the fix is not mechanical.
- The Energy Efficiency or any other external dataset would require a
  **network fetch at install or test time** (see Night 2).

When any of these fires, Composer writes to `docs/OPEN_QUESTIONS.md` and
halts. You resolve, respond in chat or edit the doc, Composer resumes.

---

## What "safe to flip public without announcing" means (the disclosure bar)

This is the bar for Night 1. If any of these fails, the repo must stay
private until fixed:

1. **Working tree is clean of identifiable proprietary content.**
   - No `XGC*`, `HHFW*`, `C1input*`, internal project codenames in any
     tracked file under `surge/`, `tests/`, `examples/`, `configs/`,
     `docs/`, `scripts/`, `data/datasets/`, top-level files.
   - No absolute NERSC-user paths (`/global/homes/...`, `/pscratch/...`)
     in tracked files except where they are obviously placeholders
     documented as such.
   - No `.env`, `.pem`, `.key`, `.pkl` binaries under `data/` that we did
     not intend to publish.
2. **Every tracked `README` and doc reflects the current state** — no stale
   references to deleted modules, to `ml_utils.py`, to XGC pipelines, or to
   features that don't exist.
3. **LICENSE + NOTICE + CITATION.cff + `pyproject.toml`** present and
   consistent (author, year, BSD-3, DOE acknowledgement).
4. **CI is green on `main`** for both unit tests and the E2E regression
   (this we already landed).
5. **`git log` is acceptable for *silent* publication** — i.e. even if
   history has things you'd eventually rewrite, nothing in history
   *names* a collaborator, project, or code they did not authorize.

Point (5) is the one that decides whether you can flip the GitHub visibility
without running `git filter-repo` first. On Night 1 we audit. If the audit
fails, the repo stays private until Night 3.

---

## Night 1 (tonight) — safety + three high-leverage items + a presentable face

Target: at end of the night, you could privately toggle the repo to
**public** and nothing bad happens. No announcement yet, but it would not
disclose anything.

### T1.1 — History & working-tree audit (read-only, 30 min)
- `rg -i 'xgc|hhfw|c1input|olcf' -- ':(exclude)docs/PUBLIC_OPEN_SOURCE_PLAN.md' ':(exclude)docs/REFACTORING_PLAN.md' ':(exclude)docs/RELEASE_SPRINT.md'`
- `rg -i '/global/homes|/pscratch|asvillar' -- ':(exclude)docs/*'`
- `git log --all --source --remotes --oneline --name-only -- 'scripts/ml_utils.py' 'docs/xgc/' 'xgc_cpp/' 'data/datasets/HHFW-NSTX/*.pkl'`
- Write the findings to `docs/AUDIT_NIGHT1.md` as a checklist.
- **STOP** if any surprising hit appears.

### T1.2 — Fix the `from loader` install-time bug (45 min)
- Move the five public functions from `scripts/m3dc1/loader.py` into
  `surge/datasets/m3dc1.py` (if not already there).
- Remove the `sys.path.insert(...)` block in `surge/__init__.py`.
- Import directly from `surge.datasets.m3dc1` with a clean
  `try/except ImportError` only for optional heavy deps.
- Update the `M3DC1_LOADER_AVAILABLE` flag semantics.
- Add one test in `tests/test_core.py` that asserts the public symbols are
  importable after `pip install`.

### T1.3 — `ResourceSpec` + `resources:` field + `[surge.fit] …` log (3 h)
Follow `docs/REFACTORING_PLAN.md` §2 verbatim. Minimum viable:
- `surge/hpc/policy.py` with `ResourceSpec`.
- Field on `SurrogateWorkflowSpec` and on `EngineRunConfig`.
- Base-class helper `_log_fit_banner()` on `BaseModelAdapter`.
- One-line policy in each adapter (RF/MLP sklearn: CPU; torch: CPU or CUDA
  with `num_workers` plumbing; GPflow: single device).
- Non-strict mode (warn + adjust) is the **only** mode implemented tonight;
  strict mode is day-2.
- Test: one test per adapter that a bad combo emits the warning.

### T1.4 — `model_size_bytes` + `inference_ms_per_sample` in metrics (2 h)
- `surge/metrics.py::measure_inference(adapter, X, n_warmup=3, n=50)` →
  returns `{"inference_ms_per_sample": ..., "throughput_sps": ...}`.
- Called from `SurrogateEngine._train_single_model` after fit.
- `_measure_model_size(adapter) -> int` helper: joblib dump for sklearn,
  `state_dict` for torch, GPflow native checkpoint size.
- Both fields land in `ModelRunResult.extra` and in `metrics.json`.
- Test: fields are present and positive on the sample CSV.

### T1.5 — Presentable repo face (2 h)
- `README.md`: add a CI badge, link to `docs/RELEASE_DEMO_PLAN.md`, install
  one-liner, `pytest -q` one-liner, links to `LICENSE` / `NOTICE` /
  `CITATION.cff` / `docs/`. Verify every link resolves in-repo.
- `CHANGELOG.md`: initial file with an `Unreleased` and a `0.1.0 (pending)`
  section populated from the work already done.
- `CITATION.cff`: BSD-3 citation block with a `# TODO: DOE CODE ID` line.
  Real ID goes in on Night 3.
- `CONTRIBUTING.md`: one-page, points to `docs/REFACTORING_PLAN.md`
  phasing, explains how to run CI locally.
- `CODE_OF_CONDUCT.md`: Contributor Covenant 2.1, standard text.
- `SECURITY.md`: "Report via GitHub security advisories"; point of contact
  (**STOP** — ask for the email you want listed).
- `docs/OPEN_QUESTIONS.md`: create empty scaffold.
- Add `.github/ISSUE_TEMPLATE/bug_report.md` and
  `.github/ISSUE_TEMPLATE/feature_request.md` (standard templates).
- Add `.github/pull_request_template.md` with a short checklist.

### T1.6 — Documentation freshness pass (1 h)
- Confirm every `docs/*.md` top-of-file status line matches reality (no
  "see ml_utils.py", no "XGC export", no dead cross-refs).
- `docs/README.md` (or equivalent entry point): keep as the docs index,
  list current docs in order of reader-journey (install → quick start →
  demo plan → workflow spec → roadmap → refactor plan).
- Run `rg -n 'TODO|FIXME|TBD'` across `docs/` and triage — resolve, move
  to `docs/ROADMAP.md`, or leave as-is with justification in the sweep
  commit message.

### Night 1 commit plan (one commit per item)
```
fix(core): drop sys.path hack for m3dc1 loader; import via surge.datasets
feat(hpc): ResourceSpec, resources: field, adapter fit-banner log
feat(metrics): record model_size_bytes and inference_ms_per_sample
docs(release): CHANGELOG, CITATION.cff, CONTRIBUTING, CoC, SECURITY, PR/issue templates
docs(sweep): align READMEs + docs index with current module layout
```

Exit criterion (go / no-go): all five commits merged to `main`, CI green
(both `test` and `e2e-regression` jobs), `docs/AUDIT_NIGHT1.md` shows no
outstanding disclosure items. If audit fails → repo stays private, Night 2
rescopes.

---

## Night 2 (tomorrow) — regression test suite over real datasets + `workflow run` best-config

Target: three progressively-complex public regression datasets pass through
`surge workflow run` with a tiny HPO / multi-model sweep and surface a
"best configuration." These become the canonical examples on the README
and run in CI (fast variants).

### T2.1 — Dataset loaders (no install-time network fetch, 2 h)
Design rule: **no hidden downloads**. Every dataset is either
- synthetic and generated in-process (rung 1), or
- shipped as a small CSV under `data/datasets/<NAME>/` with a
  `SOURCE.md` that includes URL, license, checksum, and is verified to be
  redistributable.

Rungs for Night 2:

| Rung | Shape    | Dataset                        | Loader                                | CI?  |
| ---- | -------- | ------------------------------ | ------------------------------------- | ---- |
| 1    | 1→1      | synthetic y = f(x)+ε           | `surge.datasets.synthetic.make_1d`    | yes  |
| 4    | 8→2      | UCI Energy Efficiency          | `surge.datasets.uci_energy.load()`    | yes  |
| 5    | 6→32     | M3DC1 synthetic sample (inrepo)| `surge.datasets.m3dc1.load_sample()`  | yes  |

UCI Energy Efficiency: **STOP** and confirm — I want to ship a trimmed
(<50 kB) copy directly in-repo with a `SOURCE.md` attributing UCI + the
paper (Tsanas & Xifara, 2012) and the CC-BY license of the UCI mirror, vs.
fetching at runtime with a `torchvision.datasets`-style cache. The
in-repo-copy route is the one this sprint assumes.

### T2.2 — `surge workflow run` best-config demo (3 h)
Three configs under `configs/`:
- `configs/demo_release_1d.yaml`
- `configs/demo_release_uci_energy.yaml`
- `configs/demo_release_m3dc1_sample.yaml`

Each runs the same shape: four models (sklearn.rf, sklearn.mlp, pytorch.mlp,
and — if available — gpflow.gpr) with minimal HPO (Optuna, 10 trials) and
writes the usual artifact tree. A new driver
`examples/release_best_config.py` collapses a run folder into a
`best_config.json` + a leaderboard Markdown table (model, RMSE, MAE,
`model_size_bytes`, `inference_ms_per_sample`).

### T2.3 — Regression-test suite (2 h)
`tests/test_regression_datasets.py` with parametrized tests:
- Rung 1 synthetic: assert R² > 0.9 within 30 s on CPU.
- Rung 4 UCI energy: assert RMSE below a loose upper bound, within 60 s.
- Rung 5 M3DC1 sample: assert RMSE below a loose upper bound, within 90 s.
All three use a reduced-HPO variant (3 trials) to stay in the CI budget.
The full-HPO versions are marked `@pytest.mark.slow` and skipped by default.

CI update: the `e2e-regression` job gets a new step that runs
`pytest -q -m "not slow" tests/test_regression_datasets.py`.

### T2.4 — README headline: the 3-rung story (45 min)
Rewrite the "Run the demo" section around the three rungs so a
first-time visitor sees, in order:
1. one-command install,
2. one-command rung-1 synthetic run,
3. one-command rung-4 UCI energy run,
4. link to rung-5 for the real-sample story.

### T2.5 — Optional: Refactor pass, file-count reduction (if time remains)
Collapse `pytorch.py` + `pytorch_impl.py` into one `surge/model/torch_mlp.py`
(and the same for gpflow). Keep deprecation shims. Only if Night 1 finished
early; otherwise skip to Night 3.

### Night 2 commit plan
```
feat(datasets): synthetic, UCI energy (in-repo), m3dc1 loaders
feat(workflow): release_best_config driver + 3 demo configs
test(regression): parametrized dataset regression + CI wiring
docs(readme): 3-rung headline story
```

Exit criterion: three demo configs run end-to-end locally in under 3 min
total, `best_config.json` + leaderboard table are produced, all CI jobs
green.

---

## Night 3 — publication (the only "cannot be delegated" night)

### T3.1 — Final pre-public audit (you, not Composer)
- Re-run the Night 1 audit script on current HEAD.
- Read the latest three commits' diffs one more time.

### T3.2 — History decision
- If Night 1 audit passed on history, **skip filter-repo** and publish as-is.
- Else, run `git filter-repo` per `docs/PUBLIC_OPEN_SOURCE_PLAN.md`.
- **Either path is a human-in-the-loop decision.**

### T3.3 — Create public repo + push
- New public GitHub repo.
- `git remote set-url public <url>` (or add as second remote).
- `git push public main --tags`.
- Replace `REPLACE_ORG` placeholders in `pyproject.toml`, `README.md`,
  `CITATION.cff`.

### T3.4 — DOE CODE submission
- Submit repo URL + required metadata.
- On receipt of the DOE CODE ID, update `CITATION.cff`.

### T3.5 — Tag `v0.1.0`
- `git tag -a v0.1.0 -m "SURGE v0.1.0 — DOE open-source release"`
- `git push public v0.1.0`

---

## Explicitly **not** in this sprint

- Read-the-Docs / Sphinx build. The `docs/` tree is readable on GitHub;
  a hosted docs site comes later.
- Multi-GPU / DDP / DDStore.
- k-fold cross-validation implementation.
- MLflow integration (hooks already exist; wiring is post-release).
- Registry consolidation into a single `surge/registry.py`.
- `py.typed` and mypy-in-CI.
- Any change to the model training algorithms themselves.

These are in `docs/ROADMAP.md`, `docs/dev/EXTENDED_ROADMAP.md`, and
`docs/REFACTORING_PLAN.md`. They stay post-v0.1.0.

---

## Rollback

Every commit from this sprint touches at most one concern. If anything
looks wrong after a night:

```
git log --oneline <night-1-start>..HEAD
git revert <hash>
```

If the repo has already been flipped public and something slipped
through, do not try to rewrite — open a `SECURITY` advisory, tag the
affected file(s) in a `v0.1.0.post1` commit that removes them, and email
publications that the prior window was invalid. There is no clean
"un-publish" on GitHub.
