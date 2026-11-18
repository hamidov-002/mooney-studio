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
| `analysis.py` | analysis by photo properties, saved runs (`save_run`, `list_runs`, `compare_runs`), the scores of a run in a fixed format (`run_scores`, `scores.json`, the index of all runs) |
| `runqueue.py` | the queue of runs: a list in `project/queue.json` (runs in groups, with stage states) and a thread that runs it; knows nothing about what a run does |
| `publish.py` | copies the numbers of a project or saved run to `data/<name>/` for the repository: tables only, with a guard against pictures |
| `runboard.py` | the pictures of the Runs page as HTML (stage cards, board of groups, history) with their CSS and the click script; no data access |
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
| Experiments | Vision systems · Runs (new run as stage cards · groups and queue · history) |
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
   - No pictures of real people and no Basel renders in the repository. `data/` holds published results as
     tables only (scores, settings, per-photo CSVs, two charts without faces); it is written by `publish.py`,
     and `publish.check(folder)` must return nothing for anything put there.
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
   - Look at 1440 px and about 1280 px wide; no page errors, no tracebacks in the log. The app is light only:
     with the browser set to dark, or `?__theme=dark` in the address, the page must still be the light one.
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
- **A new stage or step of a run:** a run is a plain dict made by `run_config`; `run_steps` turns it into
  `(title, weight, fn(log, prog))` steps and `run_stage_titles` names their chips (keep the two in step);
  `queue_run_one` runs them and reports stage states to the queue. Add the field to `run_config`, `r_fill` and
  the form, and a summary line in `r_summaries`. `pipeline_steps` is only there for queue items of the old shape.
- **The Runs page pictures** are custom HTML components: `gr.HTML(value, js_on_load=rb.JS)`. `runboard.py`
  builds the HTML; anything clickable carries `data-act` (and `data-id`, `data-group`, `data-stage`), the script
  turns a click into a Gradio click event, and the handler reads `evt._data`. The timer redraws a picture only
  when its HTML changed, so open logs and the scroll position stay.
- **A new score:** add it to `run_scores` in `analysis.py` (read it from a result table, never compute again),
  to `score_row` if it belongs in the index, and to the table in the manual; keep `schema` or raise its number
  if a field changes meaning.

## Style

- Plain, short wording in the interface and the docs; say what a button does and where results are written.
- Logs: one line per step, clock time in front, errors shown in the log and never swallowed.
- The look is a white page set like a paper, light only (no dark variant, on purpose): serif (Source Serif 4,
  Georgia) for the title, headings and document text; IBM Plex Sans for controls and tables; mono only for logs
  and file names; one accent, a deep ink blue; soft status colours (done, running, waiting, failed).
- Colours and type are set once in `THEME`/`CSS` in `app.py` (and `runboard.CSS`); reuse the CSS variables.
  Tables carry a rule above and below and light lines between rows, no boxes. Charts use the three series
  colours in `results.COLORS` / `analysis.BLUE, ORANGE, GREEN` and always carry labels.
