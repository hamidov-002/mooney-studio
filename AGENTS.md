# Working on Mooney Studio with a coding agent

This file is for Claude, Codex or any other coding agent (and for people) who change this code. Read it
before the first change. `CLAUDE.md` points here.

## What the tool is

A local Gradio app that turns face photos into two-tone Mooney images, runs vision systems on each photo and
its Mooney, and compares what they return; the same experiment runs on rendered 3D faces with known depth.
Start: `./setup.sh` once, then `./run.sh` (or `python app.py`), open http://127.0.0.1:7860.

## Where things are

| File | Holds |
|---|---|
| `app.py` | the whole interface: menu, pages, handlers, wiring. No computation of its own: every number comes from the modules below |
| `config.py` | every default setting with its meaning; `python config.py` lists them. Users override in `config.json` (git-ignored) |
| `collect_faces.py` | photo collection from Wikimedia Commons and the quality rules |
| `mooney.py` | the four-phase generator (background separation, grayscale, blur, threshold to a white share of the face) |
| `batch.py` | one Mooney per photo with one set of values; writes `params.csv` |
| `vision.py` | the vision systems, one `run_*` function each; each writes `ResultN/` and `analysis/resultN_*.csv` plus a protocol JSON |
| `results.py` | summary table, figure, per-photo panels |
| `analysis.py` | analysis by photo properties, saved runs (`save_run`, `list_runs`, `compare_runs`) |
| `runqueue.py` | the queue of runs: a list in `project/queue.json` and a thread that runs it; knows nothing about what a run does |
| `face3d.py` | 3D faces reconstructed from photos (MediaPipe mesh), renders, true depth |
| `bfm.py`, `bfm_fit.py` | Basel Face Model: invented faces and faces fitted to photos; renderer with cast shadows |
| `analysis3d.py` | systems against the true depth, comparison of the stimulus sets |
| `docs/` | `MANUAL.md`, `VISION_SYSTEMS.md`, `REFERENCES.md`, two README screenshots |

User data is never in this folder. It lives in the working folder (default `~/MooneyStudio`, set in
`config.json` as `WORK_DIR`): `project/` (photos, Mooneys, results), `runs/` (saved runs), `notes/` (private
Markdown, shown in Docs → Notes).

## The interface

A menu on the left with six sections; each section is a `gr.Column` whose visibility the menu buttons switch
(`go()` in the wiring), and its pages are `gr.Tab`s inside it.

| Section | Pages |
|---|---|
| Home | status, next step, working folder |
| Dataset | Photos · Subsets · 3D datasets |
| Mooney | Batch · By hand |
| Experiments | Vision systems · One-button run · Queue · Saved runs |
| Analysis | Results · By photo properties · Compare runs · 3D comparison |
| Docs | Manual · Vision systems · References · Notes |

`pages(k, state, items)` wires a section: a page is refreshed when its tab is chosen and when the menu opens
the section while that page is the open one. Do not build the menu from `gr.Tabs` turned vertical: Gradio's
overflow logic hides tabs that do not fit the narrow column.

## Rules that must not break

1. **Project folder layout.** `Original/`, `Mooney/`, `Subset/{Original,Mooney}/`, `Result1..N/{Original,Mooney}/`,
   `analysis/`, `figures/`, `params.csv`, `settings.json`, `dataset.csv`. Users hand this layout in as is.
2. **Same input, same output.** A change to the interface must not change any Mooney or any number. Check it:
   run batch and systems 1, 3, 4 on a small scratch project before and after; files must be byte-identical.
3. **Every value is traceable.** A system stores the model's raw output and writes its settings to a protocol
   JSON in `analysis/`. New settings go into that protocol too. Document new values in `docs/VISION_SYSTEMS.md`
   (model output / our settings / calculated by us).
4. **Nothing is written without the user pressing a button.** No auto-save. A change on one photo never
   touches another photo.
   **One long job at a time, and every long job can be stopped.** Long work goes through `run_logged`, which
   claims the busy flag and hands the job a `log` that raises `Stopped` once Stop was pressed; wrap progress
   callbacks in `stoppable()`. `Stopped` is a `BaseException`, so `except Exception` in the work does not
   swallow it. Never catch it and go on.
5. **Licences.**
   - The Basel Face Model `.h5` files are licensed to one person for non-commercial research. Never commit,
     upload, copy elsewhere or share them, and never put them where they could be published.
   - Photos keep their own licences (recorded in `dataset.csv`); they are never part of the repository.
   - No pictures of real people and no Basel renders in the repository.
6. **No race or ethnicity** is asked, stored or inferred anywhere.
7. **Public functions keep their names and arguments.** Add optional arguments only.

## How to make a change

1. Work on a copy of a project, never on the user's own working folder:
   `MOONEY_WORK=/tmp/test-work MOONEY_PORT=7870 python app.py` (copy a few photos into `/tmp/test-work/project/Original/`).
2. Check the code: `python -m pyflakes *.py`.
3. Check the interface in a real browser, not only by reading code. Playwright is enough:
   - Gradio keeps only the visible page in the DOM: open a section with
     `page.getByRole('button', {name, exact: true})`, a page with `page.getByRole('tab', {name, exact: true})`.
   - Labels repeat between pages; prefer `getByRole('checkbox' | 'button', {name})` over `getByLabel`.
   - Look at light and dark (`?__theme=dark`), at 1440 px and about 1280 px wide; no page errors, no
     tracebacks in the log.
4. Check rule 2 if anything near generation or the systems was touched.
5. Update `docs/MANUAL.md` (how to use), `docs/VISION_SYSTEMS.md` (what a value means), README if setup changes.
6. Commit with a message that says what changed and why.

## Recipes

- **A new setting:** add it to `SETTINGS` in `config.py` with its meaning; read it as `cfg.NAME`.
- **A new vision system:** a `run_<name>(project, source="Original", log=print, progress=None)` in `vision.py`
  that writes `Result<N>/Original`, `Result<N>/Mooney`, `analysis/result<N>_<name>.csv` and
  `analysis/result<N>_<name>_protocol.json`; add its rows to `results.summary`; a button in the vision page;
  a section in `docs/VISION_SYSTEMS.md`.
- **A new 3D source:** a module with `run_render_set(project, ...)` writing `project/<folder>/` in the layout
  of `3D/` (renders in `Original/`, true depth in `truth/`, `truth/renders.csv`); register the folder where the
  3D sources are listed in `app.py` and in `analysis3d.SETS`.
- **A new page:** a `with gr.Tab(...)` block inside its section in `app.py`; its handlers above the interface
  section; wiring at the end, with an entry in that section's `pages(...)` call. Long work goes through
  `run_logged(job, n_outputs, final, what=...)` so it streams a readable log, is refused while another job
  works, and can be stopped; put a Stop button next to its start button.
- **A new step of the one-button run:** add it to `pipeline_steps` and a key to `pipeline_config`; the queue
  then runs it too, because both call `pipeline_job`.

## Style

- Plain, short wording in the interface and the docs; say what a button does and where results are written.
- Logs: one line per step, clock time in front, errors shown in the log and never swallowed.
- Colours and type are set once in `THEME`/`CSS` in `app.py`; reuse the CSS variables.
