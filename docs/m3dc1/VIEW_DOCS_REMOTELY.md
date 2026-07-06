# View M3DC1 docs in a browser (remote / Perlmutter)

Docs live under `docs/m3dc1/` in the SURGE repo.

| File | Purpose |
|------|---------|
| [`SURGE_M3DC1_FACTS.md`](SURGE_M3DC1_FACTS.md) | Measured numbers only (Opus input) |
| [`SURGE_M3DC1_FACTS.html`](SURGE_M3DC1_FACTS.html) | Same, browser |
| [`SURGE_M3DC1_FACTS.json`](SURGE_M3DC1_FACTS.json) | Same, JSON |
| [`SURGE_M3DC1_RESEARCH_NARRATIVE.md`](SURGE_M3DC1_RESEARCH_NARRATIVE.md) | Interpretation / story |
| [`SURGE_M3DC1_RESEARCH_NARRATIVE.html`](SURGE_M3DC1_RESEARCH_NARRATIVE.html) | Same, browser (figures load from `assets/`) |
| [`SURGE_M3DC1_RESULTS_REPORT.md`](SURGE_M3DC1_RESULTS_REPORT.md) | Full lab notebook |
| [`SURGE_M3DC1_RESULTS_REPORT.html`](SURGE_M3DC1_RESULTS_REPORT.html) | Full lab notebook, browser |

## Option A — SSH port forward + local browser (recommended)

**On Perlmutter login node** (leave running):

```bash
cd $HOME/src/SURGE/docs/m3dc1
python3 -m http.server 8765 --bind 127.0.0.1
```

**On your laptop** (new terminal, adjust user/host):

```bash
ssh -L 8765:127.0.0.1:8765 asvillar@perlmutter.nersc.gov
```

Then open locally:

- Facts: http://localhost:8765/SURGE_M3DC1_FACTS.html
- Full results report: http://localhost:8765/SURGE_M3DC1_RESULTS_REPORT.html
- Narrative (with figures): http://localhost:8765/SURGE_M3DC1_RESEARCH_NARRATIVE.html

Stop server: `Ctrl+C` in the Perlmutter terminal.

## Option B — One SSH command (forward + server)

From laptop:

```bash
ssh -L 8765:localhost:8765 asvillar@perlmutter.nersc.gov \
  'cd $HOME/src/SURGE/docs/m3dc1 && python3 -m http.server 8765 --bind 127.0.0.1'
```

Open http://localhost:8765/SURGE_M3DC1_FACTS.html

## Option C — Copy HTML to laptop

```bash
scp asvillar@perlmutter.nersc.gov:$HOME/src/SURGE/docs/m3dc1/SURGE_M3DC1_FACTS.html .
scp -r asvillar@perlmutter.nersc.gov:$HOME/src/SURGE/docs/m3dc1/assets ./assets
```

Open the local `.html` file (narrative needs the `assets/` folder beside it).

## Option D — Upload to Opus

Upload any of:

- `SURGE_M3DC1_FACTS.json` (smallest, all numbers)
- `SURGE_M3DC1_FACTS.md`
- `SURGE_M3DC1_RESEARCH_NARRATIVE.md`

Paths on Perlmutter:

```
/global/homes/a/asvillar/src/SURGE/docs/m3dc1/SURGE_M3DC1_FACTS.json
/global/homes/a/asvillar/src/SURGE/docs/m3dc1/SURGE_M3DC1_FACTS.md
/global/homes/a/asvillar/src/SURGE/docs/m3dc1/SURGE_M3DC1_RESEARCH_NARRATIVE.md
```
