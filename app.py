"""
Mooney Studio - a workbench for two-tone (Mooney) face images and how vision systems handle them.

Workflow (one tab per step):
    Dataset         collect openly licensed portraits, review them, remove bad ones
    Batch           make a Mooney for every photo with one set of parameters
    Mooney          look at single photos, adjust parameters by hand, mark favourites
    Vision systems  run each system on every original and every Mooney
    Results         summary table, figure, side-by-side viewer, export of figures
    Analysis          results by properties of the photos, comparison of saved runs
    3D faces        the same experiment on rendered 3D faces with known light and depth

Run:  python app.py [--work FOLDER] [--port N] [--share]        (opens http://127.0.0.1:7860)

The heavy lifting lives in the other modules; this file only builds the interface:
    collect_faces.py  dataset collection and quality rules      mooney.py   the four-phase generator
    batch.py          batch generation and quality score        vision.py   the vision systems
    results.py        summary, figure, panels                   face3d.py   3D faces reconstructed from photos
    analysis.py       per-photo analysis: groups, agreement, ranking, figures
    analysis3d.py     3D faces against the truth, comparison of sets    bfm.py      invented 3D faces (Basel Face Model)
    bfm_fit.py        the Basel Face Model fitted to photo landmarks
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback


def _command_line():
    """Options given on the command line become environment variables, which config.py reads on import."""
    ap = argparse.ArgumentParser(description="Mooney Studio", allow_abbrev=False)
    ap.add_argument("--work", metavar="FOLDER", help="working folder for this start (holds project/, runs/ and notes/)")
    ap.add_argument("--port", type=int, help="port to listen on (default: the first free port from 7860)")
    ap.add_argument("--share", action="store_true", help="also create a temporary public gradio.live link")
    args, _ = ap.parse_known_args()
    if args.work:
        os.environ["MOONEY_WORK"] = args.work
    if args.port:
        os.environ["MOONEY_PORT"] = str(args.port)
    if args.share:
        os.environ["MOONEY_SHARE"] = "1"


if os.path.basename(sys.argv[0]) == "app.py":       # only when started as a program, not when imported
    _command_line()

import gradio as gr                                  # noqa: E402  (the imports below read the settings set above)
import requests
from PIL import Image

import analysis as an
import analysis3d as a3
import batch as bt
import bfm
import bfm_fit
import collect_faces as cf
import config as cfg
import face3d as f3
import publish as pub
import results as rs
import runboard as rb
import runqueue as rq
import vision as vs
from mooney import make_mooney, subject_mask, resize_max, face_brightness

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = cfg.WORK_DIR                           # the working folder: config.py, config.json, MOONEY_WORK or --work
PROJECT_ROOT = cfg.PROJECT_DIR                     # <working folder>/project unless set otherwise
RUNS_ROOT = cfg.RUNS_DIR                           # <working folder>/runs unless set otherwise
NOTES_ROOT = cfg.NOTES_DIR                         # <working folder>/notes: the user's own notes, never part of the code
DOCS_ROOT = os.path.join(HERE, "docs")
DATA_ROOT = os.path.join(HERE, "data")            # published results: tables only, next to the code so they can be committed
NAMES_FILE = os.path.join(PROJECT_ROOT, "identities.txt")   # the list of names last used in Dataset
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TARGET_PHOTOS, TARGET_SUBSET = cfg.TARGET_PHOTOS, cfg.TARGET_SUBSET

# the folders of a project: photos, Mooneys, the chosen subset and one result folder per vision system,
# plus the working folders the app uses
FOLDERS = ["Original", "Mooney", "Subset/Original", "Subset/Mooney",
           "Result1/Original", "Result1/Mooney", "Result2/Original", "Result2/Mooney",
           "Result3/Original", "Result3/Mooney", "Result4/Original", "Result4/Mooney"]
WORKING = ["Raw", "analysis", "figures"]


# ============================================================================ project state
def P(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def init_project():
    for rel in FOLDERS + WORKING:
        os.makedirs(P(rel), exist_ok=True)


def images_in(rel):
    """Sorted image file names in a project folder (hidden files and mesh previews excluded)."""
    p = P(rel)
    if not os.path.isdir(p):
        return []
    return sorted(f for f in os.listdir(p) if f.lower().endswith(IMG_EXT) and not f.startswith(".") and "__mesh" not in f)


def stems_in(rel):
    p = P(rel)
    return {os.path.splitext(f)[0] for f in os.listdir(p) if not f.startswith(".")} if os.path.isdir(p) else set()


def gallery_of(rel):
    return [(P(rel, f), f) for f in images_in(rel)]


def read_csv(name):
    p = P(name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def state():
    """Everything the status strip and the Overview page need, counted from the project folder."""
    orig = stems_in("Original") & {os.path.splitext(f)[0] for f in images_in("Original")}
    moon = stems_in("Mooney")
    params = read_csv("params.csv")
    people = {s.rsplit("_", 1)[0] for s in orig}
    pairs = sum(1 for p in people if sum(1 for s in orig if s.rsplit("_", 1)[0] == p) >= 2)
    return {
        "photos": len(orig), "people": len(people), "pairs": pairs,
        "mooneys": len(moon & orig), "manual": sum(1 for r in params if r.get("mode") == "manual"),
        "favs": len(load_favs()), "subset": len(stems_in("Subset/Mooney") & stems_in("Subset/Original")),
        "r": {n: (len(stems_in(f"Result{n}/Original") & orig), len(stems_in(f"Result{n}/Mooney") & orig)) for n in (1, 2, 3, 4, 5)},
        "figures": len([f for f in images_in("figures") if f.startswith("panel_")]),      # panels exist only after an export
        # system 2 is heavy, so it may be run on the subset only; its target is then the subset, not all photos
        "vlm_target": len(orig) if photo_set_stems(vlm_scope_label()) is None else len(set(photo_set_stems(vlm_scope_label())) & orig),
    }


def status_html():
    """The strip of counters under the title, shown on every tab."""
    s = state()

    def chip(label, value, done=None):
        cls = "" if done is None else (" ok" if done else " part")
        return f"<span class='chip{cls}'>{label} <b>{value}</b></span>"

    n = s["photos"]
    chips = [chip("Photos", f"{n} / {TARGET_PHOTOS}", n >= TARGET_PHOTOS if n else None),
             chip("People", s["people"]),
             chip("Mooneys", f"{s['mooneys']} / {n}", (s["mooneys"] == n) if n and s["mooneys"] else None)]
    if s["manual"]:
        chips.append(chip("Adjusted by hand", s["manual"]))
    for k, name in ((1, "Face model"), (2, "VLM"), (3, "Depth"), (4, "Shading")):
        o, m = s["r"][k]
        t = s["vlm_target"] if k == 2 else n
        done = min(o, m, t) if k == 2 else min(o, m)          # older results outside the subset are not counted
        chips.append(chip(f"{k} {name}", f"{done} / {t}", (done >= t) if t and (o or m) else None))
    chips.append(chip("Subset", f"{s['subset']} / {subset_target()}", (s["subset"] >= subset_target()) if s["subset"] else None))
    return "<div class='chips'>" + "".join(chips) + "</div>"


def overview_md():
    """The Overview page: what comes next, what is done, current settings, folder check."""
    s, st = state(), load_settings()
    n = s["photos"]
    word = lambda done, part=False: "done" if done else ("in progress" if part else "not yet")
    r = {k: min(*s["r"][k]) for k in (1, 2, 3, 4)}
    tgt = {1: n, 2: s["vlm_target"], 3: n, 4: n}
    r[2] = min(r[2], tgt[2])

    if n == 0:
        nxt = ("**Start here.** This project is empty.\n\n"
               "1. Open **Dataset → Photos** in the menu on the left, enter a few names and press *Collect*. "
               "Or copy your own square face photos into the `Original/` folder shown below and reload the page.\n"
               "2. Make the Mooneys in **Mooney → Batch**.\n"
               "3. Run the vision systems in **Experiments → Vision systems** and read the outcome in **Analysis → Results**.\n\n"
               "**Experiments → Runs** does steps 2 and 3 in one go, and keeps the outcome under a name.")
    elif s["mooneys"] < n:
        nxt = f"Make the Mooneys in **Mooney → Batch** ({n - s['mooneys']} of {n} photos have none yet)."
    elif any(r[k] < tgt[k] for k in tgt):
        k = next(k for k in (1, 2, 3, 4) if r[k] < tgt[k])
        nxt = f"Run system {k} in **Experiments → Vision systems** ({r[k]} of {tgt[k]} photos done)."
    elif s["subset"] < subset_target():
        nxt = (f"Mark your best Mooneys with the star in **Mooney → By hand**, then choose {subset_target()} of them in "
               "**Dataset → Subsets** and save.")
    elif not s["figures"]:
        nxt = "Press *Export figures* in **Analysis → Results**."
    else:
        nxt = ("Every step is done. Save the state as a run in **Experiments → Runs** to keep it, or go on with a 3D dataset in "
               "**Dataset → 3D datasets**.")
    if 0 < n < TARGET_PHOTOS:
        nxt += f"  \nThe dataset has {n} of the {TARGET_PHOTOS} photos set as target; collect more in **Dataset → Photos** if you want the full set."

    rule = ("the same for every photo" if st["white_rule"] == "fixed"
            else f"by face brightness ({st['bright_white']:g} % for faces brighter than {st['bright_border']:g})")
    lines = [
        "### Next step", nxt, "",
        "### Progress",
        "| Step | Status | Detail |", "|---|---|---|",
        f"| Photos | {word(n >= TARGET_PHOTOS, n > 0)} | {n} photos of {s['people']} people; {s['pairs']} people have two or more photos |",
        f"| Mooneys (Batch) | {word(n > 0 and s['mooneys'] == n, s['mooneys'] > 0)} | {s['mooneys']} Mooneys for {n} photos |",
        f"| Mooneys by hand | optional | {s['manual']} adjusted by hand, {s['favs']} favourites |",
        f"| Vision systems | {word(n > 0 and all(r[k] >= tgt[k] for k in tgt), max(r.values()) > 0)} | "
        f"face model {r[1]} / {n} · language model {r[2]} / {tgt[2]}{'' if vlm_scope_label() == SET_ALL else ' (' + vlm_scope_label() + ')'} · "
        f"learned depth {r[3]} / {n} · depth from shading {r[4]} / {n} |",
        f"| Figures exported | {word(s['figures'] > 0)} | {s['figures']} per-photo panels exported to figures/ |",
        f"| Subset of {subset_target()} | {word(s['subset'] >= subset_target(), s['subset'] > 0)} | {s['subset']} pairs in Subset/ |",
        f"| 3D faces from photos | {d3_word()} | {d3_line()} |",
        f"| 3D faces: Basel model fitted to photos | {d3_word(bfm_fit.SUB)} | {d3_line(bfm_fit.SUB)} |",
        f"| 3D faces from the Basel Face Model | {d3_word(bfm.SUB)} | {d3_line(bfm.SUB)} |",
        "",
        "### Current generation settings",
        f"Background **{st['bg_mode']}** · phase-1 gray **{st['bg_level']}** · blur σ **{st['sigma']:g} px** · "
        f"white **{st['white']:g} %** of the face · white share **{rule}** · preprocessing **{st['preprocess']}**.  ",
        "<span class='hint'>Change them in Mooney → Batch; they are stored in the project's settings.json.</span>",
        "",
        "### Project folders",
        "| Folder | Images |", "|---|---:|",
    ]
    for rel in FOLDERS:
        lines.append(f"| `{rel}/` | {len(stems_in(rel))} |")
    return "\n".join(lines)


def folders_md():
    """Where everything is kept, for the Overview page."""
    notes = len(note_files())
    rows = [("Working folder", WORK_ROOT, "everything of yours: project, saved runs, notes"),
            ("Project", PROJECT_ROOT, "photos, Mooneys, results, settings"),
            ("Saved runs", RUNS_ROOT, "frozen copies made in Experiments; listed in Experiments → Runs"),
            ("Notes", NOTES_ROOT, f"{notes} Markdown file{'s' if notes != 1 else ''}, shown in Docs → Notes" if notes
             else "optional: Markdown files put here appear in Docs → Notes after a restart"),
            ("Model files", os.path.join(HERE, "models"), "downloaded on first use, next to the code")]
    out = ["| | Folder | Holds |", "|---|---|---|"]
    out += [f"| {name} | <span class='mono'>{path}</span> | {what} |" for name, path, what in rows]
    if os.environ.get("MOONEY_PROJECT") or os.environ.get("MOONEY_RUNS") or os.environ.get("MOONEY_WORK"):
        out += ["", "<span class='hint'>At least one of these folders was set for this start on the command line or by an "
                    "environment variable; that wins over the saved working folder.</span>"]
    return "\n".join(out)


def change_work_dir(folder):
    """Save a new working folder for the next start. The running app keeps using the current one."""
    try:
        path = cfg.set_work_dir(folder)
    except ValueError as e:
        return f"**Not changed.** {e}"
    if path == WORK_ROOT and PROJECT_ROOT == os.path.join(path, "project") and RUNS_ROOT == os.path.join(path, "runs"):
        return f"`{path}` is already the working folder."
    has = os.path.isdir(os.path.join(path, "project"))
    return (f"**Saved.** The working folder is now `{path}` ({'it already holds a project' if has else 'a new, empty project will be created there'}). "
            "**Restart the app to switch:** stop it in the terminal with Ctrl+C and start it again with `./run.sh`. "
            "Until then this window keeps working in the folders listed above. Nothing was moved or copied.")


# ============================================================================ settings, favourites, params
DEFAULTS = {"sigma": cfg.MOONEY_SIGMA, "white": cfg.MOONEY_WHITE, "bg_level": cfg.MOONEY_BG_LEVEL, "bg_mode": cfg.MOONEY_BACKGROUND,
            "separate": cfg.MOONEY_SEPARATE, "white_rule": cfg.MOONEY_WHITE_RULE, "bright_white": cfg.MOONEY_BRIGHT_WHITE,
            "bright_border": cfg.MOONEY_BRIGHT_BORDER, "preprocess": cfg.MOONEY_PREPROCESS, "vlm_scope": cfg.VLM_SCOPE}
RULES = {"Same for every photo": "fixed", "By face brightness": "brightness"}
RULE_LABEL = {v: k for k, v in RULES.items()}


def load_settings():
    try:
        return {**DEFAULTS, **json.load(open(P("settings.json")))}
    except Exception:
        return dict(DEFAULTS)


def save_settings(**changes):
    init_project()
    d = {**load_settings(), **changes}
    json.dump(d, open(P("settings.json"), "w"), indent=2)
    return d


def load_favs():
    try:
        return list(json.load(open(P("favorites.json"))))
    except Exception:
        return []


def save_favs(favs):
    init_project()
    json.dump(sorted(set(favs)), open(P("favorites.json"), "w"), indent=2)


def toggle_favourite(stem):
    favs = load_favs()
    favs = [f for f in favs if f != stem] if stem in favs else favs + [stem]
    save_favs(favs)
    return favs


def fav_gallery():
    out = []
    for stem in load_favs():
        f = next((f for f in images_in("Mooney") if os.path.splitext(f)[0] == stem), None)
        if f:
            out.append((P("Mooney", f), stem))
    return out


def subset_stems():
    """The photos system 2 runs on in subset mode: the pairs in Subset/, or the favourites while Subset/ is empty."""
    sub = stems_in("Subset/Original") & stems_in("Subset/Mooney")
    return sub or set(load_favs())


def fav_text():
    favs = load_favs()
    return f"{len(favs)}: " + ", ".join(favs) if favs else "none yet"


# ---------------------------------------------------------------------------- named subsets
# subsets.json: {"active": name or None, "sets": {name: {"size": n, "stems": [...]}}}. Every saved subset is a pair
# of folders Subsets/<name>/Original and Subsets/<name>/Mooney; the active one is also written to Subset/, the
# folder the rest of the app (system 2, the counter at the top) reads.
def subset_slug(name):
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9]+", "-", str(name or "").strip())).strip("-")[:40]


def load_subsets():
    try:
        d = json.load(open(P("subsets.json"), encoding="utf-8"))
        d.setdefault("active", None); d.setdefault("sets", {})
    except Exception:
        d = {"active": None, "sets": {}}
    if not d["sets"]:                                    # a project from before named subsets: take what is in Subset/
        old = sorted(stems_in("Subset/Original") & stems_in("Subset/Mooney"))
        if old:
            d = {"active": "main", "sets": {"main": {"size": max(len(old), TARGET_SUBSET), "stems": old}}}
    return d


def subset_target():
    d = load_subsets()
    return int(d["sets"][d["active"]]["size"]) if d["active"] in d["sets"] else TARGET_SUBSET


# ---------------------------------------------------------------------------- photo sets a step can run on
SET_ALL, SET_FAVS, SET_SUGGEST = "All photos", "Favourites", "Suggested: one photo per person, favourites first"
SET_PREFIX = "Subset: "


def photo_set_choices(suggest=False):
    """The sets offered wherever a step can run on part of the photos."""
    out = [SET_ALL, SET_FAVS] + [SET_PREFIX + n for n in sorted(load_subsets()["sets"])]
    return ([SET_SUGGEST] + out) if suggest else out


def photo_set_stems(choice):
    """Photo stems of a set, in its own order, or None for all photos."""
    have = {os.path.splitext(f)[0] for f in images_in("Original")}
    if choice in (None, "", SET_ALL, "all"):
        return None
    if choice == SET_FAVS:
        return [s for s in load_favs() if s in have]
    if choice == SET_SUGGEST:
        return [os.path.splitext(f)[0] for f in a3.default_photos(PROJECT_ROOT)]
    if choice == "subset":                               # older settings: the project's subset
        return sorted(subset_stems() & have)
    if str(choice).startswith(SET_PREFIX):
        e = load_subsets()["sets"].get(str(choice)[len(SET_PREFIX):])
        return [s for s in (e or {}).get("stems", []) if s in have]
    return None


def photo_set_files(choice):
    """File names in Original/ for a set (all files for 'All photos')."""
    files = images_in("Original")
    stems = photo_set_stems(choice)
    if stems is None:
        return files
    by = {os.path.splitext(f)[0]: f for f in files}
    return [by[s] for s in stems if s in by]


def vlm_scope_label():
    """The saved choice for system 2 as one of the offered sets."""
    sc = load_settings().get("vlm_scope", "subset")
    if sc == "all":
        return SET_ALL
    if sc == "subset":
        d = load_subsets()
        return SET_PREFIX + d["active"] if d["active"] in d["sets"] else SET_FAVS
    return sc if sc in photo_set_choices() else SET_FAVS


def write_pairs(rel, stems):
    """Empty <rel>/Original and <rel>/Mooney and copy the pairs of these photos into them. Returns how many were copied."""
    n = 0
    for part in ("Original", "Mooney"):
        os.makedirs(P(rel, part), exist_ok=True)
        for old in os.listdir(P(rel, part)):
            if os.path.isfile(P(rel, part, old)):
                os.remove(P(rel, part, old))
    for stem in stems:
        fo = next((f for f in images_in("Original") if os.path.splitext(f)[0] == stem), None)
        if fo and os.path.exists(P("Mooney", stem + ".png")):
            shutil.copy(P("Original", fo), P(rel, "Original")); shutil.copy(P("Mooney", stem + ".png"), P(rel, "Mooney")); n += 1
    return n


def upsert_params(row):
    """Insert or replace the row for row['file'] in params.csv."""
    init_project()
    rows = [r for r in read_csv("params.csv") if r.get("file") != row["file"]] + [row]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(P("params.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)


def dataset_table():
    cols = ["identity", "file", "license", "author", "date", "yaw", "sharpness", "commons_url"]
    return [[r.get(c, "") for c in cols] for r in read_csv("dataset.csv")] or None


PARAM_COLS = ["file", "mode", "sigma", "white_target", "white_fraction", "threshold", "face_brightness", "bg_mooney", "bg_level",
              "white_rule", "preprocess", "quality"]


def params_table():
    return [[r.get(c, "") for c in PARAM_COLS] for r in read_csv("params.csv")] or None


# ============================================================================ long jobs with a readable log
_busy = {"what": None}                               # the long job that is working now, if any
_busy_lock = threading.Lock()
_stop = {"on": False, "was": False}                  # on: stop was asked for; was: the last job ended because of it
Stopped = rq.StopNow                                 # raised inside a job at its next log line or progress step


def busy_now():
    """Name of the long job that is working (a run, Batch, the queue …), or None."""
    return _busy["what"] or ("a run (Experiments → Runs)" if rq.is_running() else None)


def claim(what):
    """Reserve the machine for one long job. False if another one is working."""
    with _busy_lock:
        if busy_now():
            return False
        _busy["what"] = what
        _stop["on"] = _stop["was"] = False
        return True


def release():
    _busy["what"] = None
    _stop["on"] = False


def busy_text():
    return (f"Not started: {busy_now() or 'another job'} is still working. One job runs at a time; "
            "wait until it has finished (its log shows where it is), or stop it.")


def check_stop():
    """Called at every log line and progress step of a job: ends the job there if Stop was pressed."""
    if _stop["on"]:
        raise Stopped()


def stoppable(progress):
    """A progress callback that also lets the job be stopped at this point."""
    def prog(p, *a, **k):
        check_stop()
        return progress(p, *a, **k)
    return prog


def was_stopped():
    return _stop["was"]


def stop_job():
    """The Stop button next to a long job. The job ends after the image or step it is working on."""
    if rq.is_running():
        gr.Info("A run is working. Stop it in Experiments → Runs.")
    elif _busy["what"]:
        _stop["on"] = True
        gr.Info(f"Stopping {_busy['what']}: it ends after the image it is working on.")
    else:
        gr.Info("Nothing is running.")


STOP_NOTE = ("Files written so far stay as they are, so this step is only partly done "
             "(for example Mooneys or results up to that photo). Running the step again completes it.")


def run_logged(job, n_wait, final, what="another job"):
    """
    Run job(log) in a thread and stream its log to a textbox.
    Yields (log_text, n_wait x no-change) while it runs, then (log_text, *final()) once.
    Every log line gets the clock time in front; errors are shown in the log, not swallowed.
    Only one such job works at a time: a second one is refused with a line in its log.
    The job can be stopped: after Stop is pressed, its next log line or progress step ends it.
    """
    if not claim(what):
        yield (busy_text(),) + (gr.update(),) * n_wait
        return
    lines, err = [], {}

    def emit(msg=""):
        msg = str(msg)
        lines.append(f"{time.strftime('%H:%M:%S')}  {msg}" if msg.strip() else "")

    def log(msg=""):
        check_stop()
        emit(msg)

    def work():
        try:
            job(log)
        except Stopped:
            err["stopped"] = True
        except Exception:
            err["tb"] = traceback.format_exc()
        finally:
            _stop["was"] = bool(err.get("stopped"))
            release()

    t = threading.Thread(target=work, daemon=True); t.start()
    while t.is_alive():
        time.sleep(0.5)
        yield ("\n".join(lines[-400:]),) + (gr.update(),) * n_wait
    if err.get("stopped"):
        last = next((l for l in reversed(lines) if l.strip()), "")
        m = re.search(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]", last) or re.search(r"\s(\d+)/(\d+)\s", last)
        emit(""); emit("STOPPED BY YOU" + (f" after {m.group(1)} of {m.group(2)}" if m else "") + ". " + STOP_NOTE)
    elif err:
        emit(""); emit("STOPPED WITH AN ERROR"); lines.extend(err["tb"].strip().splitlines()[-12:])
    yield ("\n".join(lines[-400:]),) + tuple(final())


# ============================================================================ Dataset
def names_default():
    return open(NAMES_FILE, encoding="utf-8").read() if os.path.exists(NAMES_FILE) else ""


def collect(names_text, per_id, n_cand, min_face, progress=gr.Progress()):
    init_project()
    names = [l.strip() for l in names_text.splitlines() if l.strip() and not l.startswith("#")]
    if not names:
        yield "Enter at least one name, one per line.", gr.update(), gr.update(), gr.update(); return
    open(NAMES_FILE, "w", encoding="utf-8").write("\n".join(names) + "\n")            # the box is remembered
    job = lambda log: cf.collect(PROJECT_ROOT, names, per_identity=int(per_id), candidates=int(n_cand),
                                 min_face_frac=float(min_face), log=log, progress=stoppable(lambda p: progress(p)))
    yield from run_logged(job, 3, lambda: (gallery_of("Original"), dataset_table(), status_html()), what="the photo collection")


def pick_photo(evt: gr.SelectData):
    items = images_in("Original")
    if evt.index is None or evt.index >= len(items):
        return "", gr.update()
    f = items[evt.index]
    row = next((r for r in read_csv("dataset.csv") if r.get("file") == f), {})
    info = f"**{f}** · {row.get('identity', '')} · licence: {row.get('license') or '—'}"
    if row.get("commons_url"):
        info += f" · [source page]({row['commons_url']})"
    return f, info


def remove_photo(f, lab_source):
    before = images_in("Original")
    if not f or f not in before:
        return ("", "Click a photo in the gallery first.") + (gr.update(),) * 7
    k = before.index(f)
    r = cf.remove_image(PROJECT_ROOT, f)
    after = images_in("Original")
    nxt = after[min(k, len(after) - 1)] if after else ""                              # the preview moves to the next photo
    msg = (f"Removed **{f}**: moved to `_removed/`, {r['deleted']} derived file(s) deleted. "
           + ("It will not be collected again. " if r["blocked"] else "It had no source record, so it could be collected again. ")
           + (f"Now selected: **{nxt}**." if nxt else ""))
    return (nxt, msg, gr.update(value=gallery_of("Original"), selected_index=(min(k, len(after) - 1) if after else None)),
            dataset_table(), status_html(), gallery_of("Mooney"), params_table(), fav_gallery(), lab_files(lab_source))


# ============================================================================ Batch
def run_batch(source, sigma, white, bg_mode, bg_level, separate, rule, bright_white, bright_border, preprocess,
              measure, keep_manual, sweep, sweep_sigmas, sweep_whites, progress=gr.Progress()):
    init_project()
    if sweep:
        sigmas = tuple(float(x) for x in sweep_sigmas) or (float(sigma),)
        whites = tuple(float(x) / 100 for x in sweep_whites) or (float(white) / 100,)
    else:
        sigmas, whites = (float(sigma),), (float(white) / 100,)
    job = lambda log: bt.run_batch(
        PROJECT_ROOT, source=source, sigmas=sigmas, whites=whites, bg_level=int(bg_level), separate=bool(separate),
        bg_mooney=bg_mode, stats_on="face" if measure.startswith("Face") else "subject", white_rule=RULES[rule],
        bright_white=float(bright_white) / 100, bright_border=float(bright_border), preprocess=preprocess,
        keep_manual=bool(keep_manual), log=log, progress=stoppable(lambda p: progress(p)))
    yield from run_logged(job, 3, lambda: (gallery_of("Mooney"), params_table(), status_html()), what="Batch")


def save_defaults(sigma, white, bg_mode, bg_level, separate, rule, bright_white, bright_border, preprocess):
    save_settings(sigma=float(sigma), white=float(white), bg_mode=bg_mode, bg_level=int(bg_level), separate=bool(separate),
                  white_rule=RULES[rule], bright_white=float(bright_white), bright_border=float(bright_border), preprocess=preprocess)
    return "Saved as defaults (settings.json). The Mooney tab starts from these values after a reload."


def star_in_gallery(evt: gr.SelectData):
    items = images_in("Mooney")
    if evt.index is not None and evt.index < len(items):
        toggle_favourite(os.path.splitext(items[evt.index])[0])
    return fav_text(), fav_gallery()


# ============================================================================ Subset (named, of any size)
SUBSET_SHOW = ["Favourites only", "All Mooneys"]


def subset_names():
    return sorted(load_subsets()["sets"])


def subset_candidates(show):
    """Stems offered for choosing, in the order shown: the favourites, or every photo that has a Mooney."""
    have = [os.path.splitext(f)[0] for f in images_in("Mooney")]
    favs = load_favs()
    return [s for s in have if s in favs] if show == SUBSET_SHOW[0] else have


def subset_view(name, size, show, chosen, msg=""):
    """Everything the Subset tab shows for the current choice: line of text, the two galleries, originals, message."""
    chosen = [s for s in chosen if os.path.exists(P("Mooney", s + ".png"))]
    cand = subset_candidates(show)
    size = int(size or TARGET_SUBSET)
    d = load_subsets()
    saved = d["sets"].get(str(name or "").strip(), {}).get("stems")
    state = ("not saved yet" if saved is None else "saved" if sorted(saved) == sorted(chosen) else "**changed since it was saved**")
    line = (f"**{len(chosen)} of {size}** chosen · {state}"
            + (" · this is the project's subset (also in `Subset/`)" if d["active"] == str(name or "").strip() and saved is not None else "")
            + f" · {len(cand)} Mooneys offered" + (" (mark more with the star in **Mooney → By hand**)" if show == SUBSET_SHOW[0] else ""))
    orig = {os.path.splitext(f)[0]: f for f in images_in("Original")}
    return (line,
            [(P("Mooney", s + ".png"), ("✓ " if s in chosen else "") + s) for s in cand],
            [(P("Mooney", s + ".png"), s) for s in chosen],
            [(P("Original", orig[s]), s) for s in chosen if s in orig],
            msg, chosen)


def subset_open(name, show):
    """Another subset was chosen or a new name typed: load its photos and size (a new name starts empty)."""
    d = load_subsets()
    e = d["sets"].get(str(name or "").strip())
    size = int(e["size"]) if e else TARGET_SUBSET
    return (gr.update(value=size),) + subset_view(name, size, show, list(e["stems"]) if e else [])


def subset_toggle(name, size, show, chosen, evt: gr.SelectData):
    cand = subset_candidates(show)
    msg = ""
    if evt.index is not None and evt.index < len(cand):
        s = cand[evt.index]
        if s in chosen:
            chosen = [c for c in chosen if c != s]
        elif len(chosen) >= int(size or TARGET_SUBSET):
            msg = f"The subset is full ({len(chosen)} of {int(size)}). Remove one first, or raise the size."
        else:
            chosen = chosen + [s]
    return subset_view(name, size, show, chosen, msg)


def subset_remove(name, size, show, chosen, evt: gr.SelectData):
    if evt.index is not None and evt.index < len(chosen):
        chosen = [c for k, c in enumerate(chosen) if k != evt.index]
    return subset_view(name, size, show, chosen)


def subset_fill(name, size, show, chosen):
    """Add favourites, in their order, until the size is reached."""
    room = int(size or TARGET_SUBSET) - len(chosen)
    add = [s for s in subset_candidates(SUBSET_SHOW[0]) if s not in chosen][:max(room, 0)]
    msg = f"Added {len(add)} favourites." if add else "Nothing added: the subset is full or there are no more favourites."
    return subset_view(name, size, show, chosen + add, msg)


def subset_save(name, size, show, chosen, make_active):
    name = str(name or "").strip()
    slug = subset_slug(name)
    if not slug:
        return subset_view(name, size, show, chosen, "Give the subset a name first.") + (gr.update(), status_html())
    if not chosen:
        return subset_view(name, size, show, chosen, "Nothing chosen yet: click Mooneys on the left.") + (gr.update(), status_html())
    init_project()
    d = load_subsets()
    d["sets"][name] = {"size": int(size or TARGET_SUBSET), "stems": list(chosen)}
    n = write_pairs(os.path.join("Subsets", slug), chosen)
    where = f"`Subsets/{slug}/`"
    if make_active or d["active"] in (None, name):
        d["active"] = name
        write_pairs("Subset", chosen)
        where += " and `Subset/`"
    json.dump(d, open(P("subsets.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    msg = f"Saved “{name}”: {n} pairs in {where} (Original and Mooney)."
    if n < int(size or TARGET_SUBSET):
        msg += f" The size is {int(size)}; {int(size) - n} more can be added."
    return subset_view(name, size, show, chosen, msg) + (gr.update(choices=subset_names(), value=name), status_html())


def person_of(stem, who=None):
    """The person a photo shows: the identity in dataset.csv, else the file name without its _n ending."""
    return ((who or {}).get(stem) or stem.rsplit("_", 1)[0]).strip().lower()


def draw_subsets(pool, size, count, seed, one_per_person=True, disjoint=True, who=None):
    """
    Draw `count` random subsets of `size` photos from `pool` (stems). Never a photo twice in one subset.
    one_per_person: no person twice inside a subset; disjoint: the subsets share no photo.
    The same pool and seed give the same subsets. Only full subsets are returned, so the list can be shorter than count.
    """
    rnd = random.Random(int(seed))
    pool, used, out = sorted(set(pool)), set(), []
    for _ in range(int(count)):
        cand = [s for s in pool if not (disjoint and s in used)]
        rnd.shuffle(cand)
        pick, people = [], set()
        for s in cand:
            if len(pick) >= int(size):
                break
            p = person_of(s, who)
            if one_per_person and p in people:
                continue
            pick.append(s); people.add(p)
        if len(pick) < int(size):
            break
        out.append(pick); used.update(pick)
    return out


def subset_random(pool_label, size, count, stem, seed, one_per_person, disjoint, show):
    """Create random subsets and save each like Save subset does. The project's subset is not changed."""
    size, count, seed = int(size or TARGET_SUBSET), int(count or 1), int(seed or 0)
    stem = str(stem or "").strip() or "random"
    pool = subset_candidates(pool_label)
    keep = (gr.update(), gr.update())

    def refuse(msg):
        return keep + (gr.update(),) * 4 + (msg, gr.update()) + (gr.update(), status_html())
    if not subset_slug(stem):
        return refuse("Give the random subsets a name stem with letters or digits.")
    if size < 1 or count < 1:
        return refuse("Size and number of subsets must be at least 1.")
    if not pool:
        return refuse(f"Nothing to draw from: “{pool_label}” holds no Mooneys.")
    who = {os.path.splitext(r.get("file", ""))[0]: r.get("identity", "") for r in read_csv("dataset.csv")}
    drawn = draw_subsets(pool, size, count, seed, bool(one_per_person), bool(disjoint), who)
    if not drawn:
        people = len({person_of(s, who) for s in pool})
        return refuse(f"No subset of {size} could be drawn: the pool has {len(pool)} photos of {people} people"
                      + (" and only one photo per person is allowed." if one_per_person else "."))
    init_project()
    d = load_subsets()
    taken = set(d["sets"]) | {subset_slug(n) for n in d["sets"]}
    names, k = [], 1
    for pick in drawn:
        while f"{stem} {k}" in taken or subset_slug(f"{stem} {k}") in taken:
            k += 1
        name = f"{stem} {k}"
        taken.add(name)
        d["sets"][name] = {"size": size, "stems": list(pick),
                           "random": {"pool": pool_label, "pool_size": len(pool), "seed": seed, "one_per_person": bool(one_per_person),
                                      "disjoint": bool(disjoint), "drawn_together": len(drawn)}}
        write_pairs(os.path.join("Subsets", subset_slug(name)), pick)
        names.append(name)
    json.dump(d, open(P("subsets.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    short = count - len(drawn)
    msg = (f"Created {len(names)} random subset{'s' if len(names) != 1 else ''} of {size}: {', '.join('“' + n + '”' for n in names)} "
           f"· pool {pool_label} ({len(pool)} photos) · seed {seed}"
           + (" · one photo per person" if one_per_person else "") + (" · no photo shared between them" if disjoint and len(names) > 1 else "") + ".")
    if short:
        msg += f" {short} more could not be made: the pool is too small for that with these rules."
    msg += " The project's subset was not changed."
    return (names[0], gr.update(value=size)) + subset_view(names[0], size, show, list(drawn[0]), msg) + (
        gr.update(choices=subset_names(), value=names[0]), status_html())


def subset_delete(name, show):
    name = str(name or "").strip()
    d = load_subsets()
    if name not in d["sets"]:
        return ((gr.update(), gr.update()) + subset_view(name, TARGET_SUBSET, show, [], "This subset was never saved; nothing to delete.")
                + (gr.update(), status_html()))
    del d["sets"][name]
    shutil.rmtree(P("Subsets", subset_slug(name)), ignore_errors=True)
    note = ""
    if d["active"] == name:
        d["active"] = None
        write_pairs("Subset", [])
        note = " It was the project's subset, so `Subset/` is empty now."
    json.dump(d, open(P("subsets.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    names = subset_names()
    nxt = names[0] if names else "main"
    out = subset_open(nxt, show)                          # size, line, candidates, chosen Mooneys, originals, message, state
    return ((nxt,) + out[:5] + (f"Deleted “{name}”.{note}", out[6])
            + (gr.update(choices=names, value=nxt if names else None), status_html()))


# ============================================================================ Mooney (single photo, by hand)
_mask_cache = {"key": None, "mask": None}
_face_cache = {"key": None, "region": None, "finder": None}
_last = {"result": None, "stem": None, "params": None, "head": ""}
LAB_SOURCES = {"Original": "Original", "Raw (uncropped)": "Raw", "Upload": None}


def _key(image):
    return image.size, image.tobytes()[:4096]


def get_mask(image):
    if _mask_cache["key"] != _key(image):
        _mask_cache["key"], _mask_cache["mask"] = _key(image), subject_mask(resize_max(image, 1024))
    return _mask_cache["mask"]


def get_face_region(image):
    """Face box of the image, cached per image: the region the white share is counted on, as in Batch."""
    if _face_cache["key"] != _key(image):
        try:
            if _face_cache["finder"] is None:
                _face_cache["finder"] = cf.FaceFinder()
            region = bt.face_region(resize_max(image, 1024), _face_cache["finder"])
        except Exception:
            region = None
        _face_cache["key"], _face_cache["region"] = _key(image), region
    return _face_cache["region"]


def run_lab(image, separate, bg_level, bg_mode, sigma, white_pct, preprocess, manual, thr, source, fname):
    if image is None:
        return None, None, None, None, "Choose a photo from the project or upload one."
    mask = get_mask(image) if separate else None
    region = get_face_region(image)
    r = make_mooney(image, sigma=sigma, white_fraction=white_pct / 100, threshold=thr if manual else None,
                    separate=bool(separate) and mask is not None, bg_level=int(bg_level), mask=mask, bg_mooney=bg_mode,
                    stat_region=region, preprocess=preprocess)
    fb = face_brightness(resize_max(image, 1024), mask, region)
    where = "of the face" if region is not None else "of the person (no face found)" if r.mask is not None else "of the image"
    info = (f"white **{r.white_fraction * 100:.1f} %** {where} · threshold **{r.threshold:.0f}** · blur σ {r.sigma:g} px"
            + (f" · face brightness **{fb:.0f}**" if fb is not None else "")
            + f" · {r.mooney.shape[1]} × {r.mooney.shape[0]} px")
    stem = None if LAB_SOURCES.get(source) is None or not fname else os.path.splitext(fname)[0]
    _last.update(result=r, stem=stem, params={
        "sigma": float(sigma), "white_target": round(white_pct / 100, 3), "white_fraction": f"{r.white_fraction:.4f}",
        "threshold": f"{r.threshold:.2f}", "bg_level": int(bg_level), "bg_mooney": bg_mode,
        "separated": int(r.mask is not None), "stats_on": "face" if region is not None else "subject",
        "face_brightness": "" if fb is None else f"{fb:.0f}", "preprocess": preprocess,
        "manual_threshold": int(bool(manual)), "mode": "manual"})
    _last["head"] = info
    info += "  \n" + lab_saved_note(stem, _last["params"])
    return Image.fromarray(r.separated), Image.fromarray(r.gray), Image.fromarray(r.blurred), Image.fromarray(r.mooney), info


def lab_row(stem):
    """The row of params.csv for this photo's saved Mooney, or None."""
    return next((r for r in read_csv("params.csv") if r.get("file") == f"{stem}.png"), None) if stem else None


def _same(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def lab_saved_note(stem, now):
    """One line saying what is saved for this photo and whether the sliders show something else."""
    if not stem:
        return "An uploaded picture cannot be saved to the project."
    row = lab_row(stem)
    if row is None or not os.path.exists(P("Mooney", stem + ".png")):
        return "No Mooney is saved for this photo yet. **Save this Mooney** keeps the one on screen."
    how = "adjusted by hand" if row.get("mode") == "manual" else "made by Batch"
    by_hand = str(row.get("manual_threshold", "0")) == "1"
    same = (_same(row.get("sigma"), now["sigma"], 0.01) and row.get("bg_mooney") == now["bg_mooney"]
            and _same(row.get("bg_level"), now["bg_level"], 0.5) and (row.get("preprocess") or "none") == now["preprocess"]
            and by_hand == bool(now["manual_threshold"])
            and (_same(row.get("threshold"), now["threshold"], 0.6) if by_hand else _same(row.get("white_target"), now["white_target"], 0.0006)))
    if same:
        return f"Saved Mooney of this photo: {how}. The sliders show its saved values."
    return (f"Saved Mooney of this photo: {how}. **What you see is not saved.** *Save this Mooney* keeps it for this photo only; "
            "no other photo changes.")


def lab_values(source, f):
    """
    The control values for a photo: those its saved Mooney was made with, or the defaults if it has none.
    Called whenever another photo is chosen, so an adjustment made on one photo is never carried to the next.
    """
    st = load_settings()
    row = lab_row(os.path.splitext(f)[0]) if f and LAB_SOURCES.get(source) == "Original" else None
    num = lambda key, default: float(row[key]) if row and _same(row.get(key), row.get(key), 0) else default
    by_hand = bool(row) and str(row.get("manual_threshold", "0")) == "1"
    mode = row.get("bg_mooney") if row and row.get("bg_mooney") in ("black", "white", "auto") else st["bg_mode"]
    pre = row.get("preprocess") if row and row.get("preprocess") in ("none", "clahe") else st["preprocess"]
    return (st["separate"], int(num("bg_level", st["bg_level"])), mode, num("sigma", st["sigma"]),
            round(100 * num("white_target", st["white"] / 100), 1), pre, by_hand,
            gr.update(value=round(num("threshold", 128)) if by_hand else 128, visible=by_hand))


def save_current():
    """Write the Mooney on screen to Mooney/<stem>.png and its parameters to params.csv (mode = manual)."""
    if _last["result"] is None or not _last["stem"]:
        return "Nothing to save: choose a photo from the project, not an upload."
    init_project()
    out = _last["stem"] + ".png"
    Image.fromarray(_last["result"].mooney).save(P("Mooney", out))
    upsert_params({"file": out, "source": _last["stem"], **_last["params"]})
    return f"Saved Mooney/{out} with these values. Batch will leave it untouched."


def star_current():
    if not _last["stem"]:
        return "Choose a photo from the project first.", gr.update(), gr.update(), gr.update()
    favs = toggle_favourite(_last["stem"])
    on = _last["stem"] in favs
    return (f"{'Added to' if on else 'Removed from'} favourites: {_last['stem']} ({len(favs)} marked).",
            fav_gallery(), gr.update(value="★ Favourite" if on else "☆ Favourite"), fav_text())


def lab_list(source):
    rel = LAB_SOURCES.get(source)
    return images_in(rel) if rel else []


def lab_files(source):
    fs = lab_list(source)
    return gr.update(choices=fs, value=fs[0] if fs else None, visible=LAB_SOURCES.get(source) is not None)


def lab_step(source, f, d):
    """Previous / Next through the files of the chosen folder; wraps around at the ends."""
    fs = lab_list(source)
    if not fs:
        return gr.update()
    i = fs.index(f) if f in fs else (-1 if d > 0 else 0)
    return gr.update(choices=fs, value=fs[(i + d) % len(fs)])


def lab_position(source, f):
    fs = lab_list(source)
    return f"<div class='pos'>{fs.index(f) + 1} / {len(fs)}</div>" if f in fs else ""


def lab_load(source, f):
    rel = LAB_SOURCES.get(source)
    if not rel or not f:
        return gr.update()
    return Image.open(P(rel, f)).convert("RGB")


def lab_star_label(source, f):
    stem = os.path.splitext(f)[0] if f and LAB_SOURCES.get(source) else None
    return gr.update(value="★ Favourite" if stem in load_favs() else "☆ Favourite")


def lab_reset():
    st = load_settings()
    return st["separate"], st["bg_level"], st["bg_mode"], st["sigma"], st["white"], st["preprocess"], False


# ============================================================================ Vision systems
def vlm_args(source, identify, model, prompt, scope):
    """Arguments for system 2, or None (= refuse to start) when no model is chosen or the chosen set is empty."""
    if not model:
        return None
    save_settings(vlm_scope="all" if scope == SET_ALL else scope)
    only = photo_set_stems(scope)
    if only is not None and not only:
        return None
    return {"source": source, "identify": bool(identify), "model": model, "prompt": prompt,
            "only": None if only is None else set(only)}


def pick_model(models):
    """The configured Ollama model if it is installed, otherwise the first installed one."""
    return cfg.OLLAMA_MODEL if cfg.OLLAMA_MODEL in models else (models[0] if models else None)


def vision_status():
    st = vs.status()
    return (st["face"], st["vlm"], st["depth"], st["shading"], st["mesh"],
            gr.update(choices=st["vlm_models"], value=pick_model(st["vlm_models"])))


# ============================================================================ Results
SUMMARY_COLS = ["system", "measure", "Original", "Mooney", "Mooney, re-blurred", "n", "note"]


def results_load():
    rows = rs.summary(PROJECT_ROOT)
    table = [[r.get(c, "") for c in SUMMARY_COLS] for r in rows] or None
    fig = rs.figure(PROJECT_ROOT)
    stems = rs.stems(PROJECT_ROOT)
    msg = "" if rows else "No results yet. Run a system in Experiments → Vision systems, then press Refresh."
    if rows and fig is None:
        msg = "Table only: the figure needs matplotlib (`pip install matplotlib`)."
    return table, fig, gr.update(choices=stems, value=(stems[0] if stems else None)), msg


def results_show(stem):
    return (rs.panel(PROJECT_ROOT, stem), rs.detail(PROJECT_ROOT, stem)) if stem else (None, "")


def results_step(stem, d):
    stems = rs.stems(PROJECT_ROOT)
    if not stems:
        return gr.update()
    return gr.update(value=stems[(stems.index(stem) + d) % len(stems)] if stem in stems else stems[0])


def results_export():
    favs = load_favs()
    made = rs.export(PROJECT_ROOT, favs or rs.stems(PROJECT_ROOT)[:10])
    which = "your favourites" if favs else "the first 10 photos (no favourites marked)"
    return f"Wrote {len(made)} files to figures/: the summary table and figure, and one panel each for {which}."


def run_choices():
    return [an.CURRENT] + an.list_runs(RUNS_ROOT)


RUN_LABELS = {v[0]: k for k, v in an.RUN_GROUPS.items()}          # label shown in the app -> key


def runs_compare(names):
    head, rows, shead, srows = an.compare_runs(PROJECT_ROOT, RUNS_ROOT, names)
    if not rows and not srows:
        return gr.update(value=None), gr.update(value=None), "Choose two or more runs. Save a run in Experiments first."
    note = "" if len(names or []) > 1 else "Choose at least one more run to compare."
    return gr.update(value=rows or None, headers=head), gr.update(value=srows, headers=shead), note


# ============================================================================ 3D faces
D3_SOURCES = {"Faces reconstructed from your photos": f3.SUB, "Basel Face Model fitted to your photos": bfm_fit.SUB,
              "Invented faces from the Basel Face Model": bfm.SUB}
D3_LEAD = {
    f3.SUB: "For a few photos a 3D face mesh is estimated and drawn again under light directions we choose.",
    bfm_fit.SUB: "The fitted heads are drawn under the lights chosen below.",
    bfm.SUB: "The faces are drawn in the head poses and under the light directions chosen below.",
}
FIT_POSES = list(bfm_fit.POSES)


def fit_default_set():
    """The photos the fitted source starts with: the project's subset, else the favourites."""
    d = load_subsets()
    return SET_PREFIX + d["active"] if d["active"] in d["sets"] else SET_FAVS
BASEL_MODELS = {"Whole head": cfg.BASEL_MODEL, "Face only": cfg.BASEL_MODEL_FACE}


def d3_sub(source):
    """Folder of the chosen source of 3D faces (3D, 3D_basel_fit or 3D_basel)."""
    return D3_SOURCES.get(source, f3.SUB)


def basel_note(which="Whole head"):
    """Whether the Basel model can be used, as one plain line; says how to get it if it is missing."""
    path = BASEL_MODELS.get(which, cfg.BASEL_MODEL)
    problem = bfm.check(path)
    short = os.path.relpath(path, HERE)
    return f"**Not available yet.** {problem.replace(path, short)}" if problem else f"Model file found: `{short}`."


def d3_gallery(rel, sub=f3.SUB):
    d = P(sub, rel)
    if not os.path.isdir(d):
        return []
    return [(os.path.join(d, f), os.path.splitext(f)[0]) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]


def d3_line(sub=f3.SUB):
    """Where a 3D set stands, in one line."""
    s = a3.status(PROJECT_ROOT, sub)
    if not s["renders"]:
        return "no renders yet"
    line = f"{s['faces']} faces under {s['lights']} lights"
    if len(s["poses"]) > 1:
        line += f" in {len(s['poses'])} head poses"
    line += f" = {s['renders']} renders"
    if s["model"]:
        line += f" ({s['model']}, cast shadows {'on' if s['shadows'] == 'yes' else 'off'})"
    line += f" · {s['mooneys']} Mooneys"
    line += " · systems " + ", ".join(str(n) for n in s["systems"]) + " run" if s["systems"] else " · no system run yet"
    return line + (" · compared with the truth" if s["scored"] else "")


def d3_word(sub=f3.SUB):
    s = a3.status(PROJECT_ROOT, sub)
    return "done" if s["scored"] and s["mooneys"] >= s["renders"] > 0 else ("in progress" if s["renders"] else "optional")


def d3_sheet(source, stem):
    return a3.person_sheet(PROJECT_ROOT, stem, sub=d3_sub(source)) if stem else None


def d3_switch(source, which):
    """The source of 3D faces was changed: show its controls, its explanation and what its folder holds."""
    sub = d3_sub(source)
    basel, fitted = sub == bfm.SUB, sub == bfm_fit.SUB
    photo_set = fit_default_set() if fitted else SET_SUGGEST
    return (gr.update(visible=not basel), gr.update(visible=basel or fitted), D3_LEAD[sub] + " " + a3.caveat(sub)
            + (" Strongly turned heads are skipped, because this simple mesh breaks for them." if sub == f3.SUB else ""),
            basel_note(which),
            gr.update(visible=basel), gr.update(visible=basel), gr.update(visible=fitted),
            gr.update(choices=photo_set_choices(suggest=True), value=photo_set),
            gr.update(choices=images_in("Original"), value=photo_set_files(photo_set))) + d3_data_view(source)


def d3_suggest(choice=SET_SUGGEST):
    return gr.update(choices=images_in("Original"), value=photo_set_files(choice))


def d3_vlm_only(sub, vlm_set, log=None):
    """
    The renders system 2 runs on in a 3D dataset (it is heavy, so usually not all of them), as a set of file stems or
    None for all. Faces made from photos follow the chosen photo set; invented faces take the first VLM_3D_FACES.
    """
    rows = a3.renders(PROJECT_ROOT, sub)
    faces = list(dict.fromkeys(os.path.splitext(r["source"])[0] for r in rows))
    if a3.has_photos(sub):
        want = photo_set_stems(vlm_set or vlm_scope_label())
        chosen = faces if want is None else [f for f in faces if f in set(want)]
        if not chosen and faces:                         # the set shares no photo with this 3D dataset
            chosen = faces[:max(int(cfg.VLM_3D_FACES), 1)] if int(cfg.VLM_3D_FACES) else faces
            if log: log(f"The set “{vlm_set}” has no photo in this 3D dataset; taking the first {len(chosen)} faces.")
    else:
        chosen = faces[:int(cfg.VLM_3D_FACES)] if int(cfg.VLM_3D_FACES) else faces
    if len(chosen) == len(faces):
        return None
    keep = set(chosen)
    return {os.path.splitext(r["file"])[0] for r in rows if os.path.splitext(r["source"])[0] in keep}


def d3_job(steps, sub, files, lights, basel, progress, systems=None, vlm_set=None):
    """The chosen steps of the 3D chain as one job for run_logged. Each step checks that the one before it was done.
    systems: which of 1, 2, 3 and 4 the step "systems" runs (None = 1, 3 and 4). System 2 is compared with the light
    and pose we set; vlm_set is the photo set it follows on faces made from photos."""
    def job(log):
        root = P(sub)
        for k, step in enumerate(steps):
            if len(steps) > 1:
                if k:
                    log("")
                log(f"Step {k + 1} of {len(steps)}: {step}")
            prog = stoppable(lambda p, k=k, step=step: progress((k + p) / len(steps), desc=step))
            if step == "render" and sub == bfm_fit.SUB:
                if not files or not lights:
                    log("Choose at least one photo and one light direction."); return
                who = {os.path.splitext(r.get("file", ""))[0]: r.get("identity", "") for r in read_csv("dataset.csv")}
                made = bfm_fit.run_fit_set(PROJECT_ROOT, list(files), lights, pose=basel.get("fit_pose", "frontal"),
                                           model_path=BASEL_MODELS.get(basel["model"], cfg.BASEL_MODEL), shadows=bool(basel["shadows"]),
                                           identities=who, log=log, progress=prog)
                if not made:
                    return                                          # the reason is in the log
            elif step == "render" and sub == bfm.SUB:
                if not lights or not basel["poses"]:
                    log("Choose at least one light direction and one head pose."); return
                made = bfm.run_render_set(PROJECT_ROOT, range(1, int(basel["faces"]) + 1), lights, basel["poses"],
                                          model_path=BASEL_MODELS.get(basel["model"], cfg.BASEL_MODEL), scale=cfg.BASEL_SCALE,
                                          shadows=bool(basel["shadows"]), log=log, progress=prog)
                if not made:
                    return                                          # the reason is in the log; nothing was changed
            elif step == "render":
                if not files or not lights:
                    log("Choose at least one photo and one light direction."); return
                who = {os.path.splitext(r.get("file", ""))[0]: r.get("identity", "") for r in read_csv("dataset.csv")}
                f3.run_render_set(PROJECT_ROOT, list(files), [l for l in f3.LIGHTS if l in lights], identities=who, log=log, progress=prog)
            elif step == "Mooneys":
                if not d3_gallery("Original", sub):
                    log("No renders yet - press Render first."); return
                st = load_settings()
                a3.clear_results(PROJECT_ROOT, sub)                 # results of earlier Mooneys would no longer match
                log(f"Settings as saved in Mooney → Batch: blur {st['sigma']:g} · white {st['white']:g} % · background {st['bg_mode']}.")
                bt.run_batch(root, source="Original", sigmas=(float(st["sigma"]),), whites=(float(st["white"]) / 100,),
                             bg_level=int(st["bg_level"]), separate=bool(st["separate"]), bg_mooney=st["bg_mode"],
                             white_rule=st["white_rule"], bright_white=float(st["bright_white"]) / 100,
                             bright_border=float(st["bright_border"]), preprocess=st["preprocess"], keep_manual=False,
                             log=log, progress=prog)
            else:
                if not d3_gallery("Mooney", sub):
                    log("No Mooneys of the renders yet - make them in Dataset → 3D datasets first."); return
                which = [n for n in (1, 3, 4) if systems is None or n in systems]
                vlm = systems is not None and 2 in systems
                if not which and not vlm:
                    log("Tick at least one system."); return
                span = 0.5 if vlm and which else 1.0             # system 2 gets the second half of the bar when both run
                parts = {1: (vs.run_face, 0.0, 0.2), 3: (vs.run_depth, 0.2, 0.4), 4: (vs.run_shading, 0.6, 0.3)}
                for i, n in enumerate(which):
                    fn, start, share = parts[n]
                    if i:
                        log("")
                    fn(root, "Original", log=log, progress=lambda p, start=start, share=share: prog(span * (start + share * p)))
                if which:
                    log(""); a3.score_truth(PROJECT_ROOT, log=log, progress=lambda p: prog(span * (0.9 + 0.1 * p)), sub=sub)
                if vlm:
                    if which:
                        log("")
                    model = pick_model(vs.status()["vlm_models"])
                    if not model:
                        log("System 2 skipped: Ollama is not running or has no vision model (Experiments → Vision systems → settings).")
                    else:
                        only = d3_vlm_only(sub, vlm_set, log)
                        log("System 2 on " + ("every render" if only is None else f"{len(only)} renders") + " and their Mooneys.")
                        vs.run_vlm(root, model, source="Original", log=log, only=only,
                                   progress=lambda p: prog((span if which else 0.0) + (1 - (span if which else 0.0)) * 0.98 * p))
                        log(""); log("System 2 against the light and pose we set:")
                        a3.score_vlm(PROJECT_ROOT, log=log, sub=sub)
    return job


def d3_run(steps):
    def handler(source, files, lights, n_faces, model, poses, shadows, fit_pose="frontal", progress=gr.Progress()):
        init_project()
        basel = {"faces": n_faces or cfg.BASEL_FACES, "model": model, "poses": poses or [], "shadows": shadows, "fit_pose": fit_pose}
        yield from run_logged(d3_job(steps, d3_sub(source), files, lights, basel, progress), 3, lambda: d3_data_view(source),
                              what="the 3D step")
    return handler


def d3_export(source):
    sub = d3_sub(source)
    made = a3.export(PROJECT_ROOT, sub)
    return f"Wrote {len(made)} files to {sub}/figures/: " + ", ".join(made) if made else "Nothing to export yet."


def d3_compare():
    """The comparison of the stimulus sets: table by measure, table by light, chart, and a line if there is nothing yet."""
    head, rows = a3.compare_sets(PROJECT_ROOT)
    lhead, lrows = a3.compare_lights(PROJECT_ROOT)
    msg = "" if rows else ("Nothing to compare yet. Run the vision systems on the photos and on a 3D dataset "
                           "in Experiments → Vision systems.")
    return (gr.update(value=rows or None, headers=head), gr.update(value=lrows or None, headers=lhead),
            a3.fig_compare(PROJECT_ROOT) if rows else None, msg)


def d3_compare_export():
    made = a3.export_compare(PROJECT_ROOT)
    return f"Wrote {len(made)} files to figures/: " + ", ".join(made) if made else "Nothing to export yet."


# ============================================================================ Runs: one run = one source and its stages
PIPE_SYSTEMS = ["1 Face model", "2 Vision-language model (chosen set only, needs Ollama)", "3 Learned depth", "4 Depth from shading"]
PIPE_KEYS = ["name", "note", "keep", "do_batch", "sigma", "white", "bg_mode", "bg_level", "systems", "d3", "n_faces", "do_export",
             "do_save", "vlm_set", "d3_set", "d3_fit_set"]
R_PHOTOS = "photos"
R_SOURCES = {R_PHOTOS: "Photos", f3.SUB: "3D · reconstructed", bfm_fit.SUB: "3D · Basel fitted", bfm.SUB: "3D · Basel invented"}
R_SOURCE_KEY = {v: k for k, v in R_SOURCES.items()}
R_SYS = {1: "1 Face model", 2: "2 Vision-language model (needs Ollama)", 3: "3 Learned depth", 4: "4 Depth from shading"}
R_SYS_KEY = {v: k for k, v in R_SYS.items()}
R_SYS_SHORT = {1: "1 Face", 2: "2 VLM", 3: "3 Depth", 4: "4 Shading"}
R_ON = {"mooneys": False, "systems": True, "export": True, "save": True}       # which stages a new run starts with
R_NO_SET = "–"
QUEUE_NOTE = ("Every run is frozen in its own saved copy. The project itself is one folder, so after a group has run it holds "
              "the Mooneys and results of the **last** run; put the run you want to keep working on last.")


def pipeline_config(name, note, keep, do_batch, sigma, white, bg_mode, bg_level, systems, d3, n_faces, do_export, do_save,
                    vlm_set=None, d3_set=SET_SUGGEST, d3_fit_set=None):
    """The form of the one-button run as a plain dict (it can be stored in the queue file)."""
    return {"name": str(name or "").strip(), "note": str(note or "").strip(), "keep": list(keep or []), "do_batch": bool(do_batch),
            "sigma": float(sigma), "white": float(white), "bg_mode": bg_mode, "bg_level": int(bg_level),
            "systems": list(systems or []), "d3": list(d3 or []), "n_faces": int(n_faces or cfg.BASEL_FACES),
            "do_export": bool(do_export), "do_save": bool(do_save), "vlm_set": vlm_set, "d3_set": d3_set or SET_SUGGEST,
            "d3_fit_set": d3_fit_set}


def pipeline_refusal(c):
    """Why a run cannot start, or None."""
    if c["do_save"] and not an.run_slug(c["name"]):
        return "Not started: give the run a name first, or untick *Save the run*."
    if not images_in("Original"):
        return "Not started: there are no photos yet. Collect some in **Dataset → Photos** first."
    if not c["do_batch"] and not images_in("Mooney"):
        return "Not started: there are no Mooneys yet. Tick *Make the Mooneys*."
    if not (c["do_batch"] or c["systems"] or c["d3"] or c["do_export"] or c["do_save"]):
        return "Not started: nothing is ticked."
    return None


def pipeline_steps(c, saved):
    """The steps of one run as (title, weight, fn(log, prog)). Every step calls the same function as its own page."""
    name, note, keep, systems, d3 = c["name"], c["note"], c["keep"], c["systems"], c["d3"]
    sigma, white, bg_mode, bg_level, vlm_set = c["sigma"], c["white"], c["bg_mode"], c["bg_level"], c["vlm_set"]
    steps = []
    if c["do_batch"]:
        def step_batch(log, prog):
            st = save_settings(sigma=float(sigma), white=float(white), bg_mode=bg_mode, bg_level=int(bg_level))
            log(f"Values: blur {st['sigma']:g} · white {st['white']:g} % · background {st['bg_mode']} · phase-1 gray {st['bg_level']}. "
                "Saved as the project's defaults. Mooneys adjusted by hand are kept.")
            bt.run_batch(PROJECT_ROOT, source="Original", sigmas=(float(sigma),), whites=(float(white) / 100,), bg_level=int(bg_level),
                         separate=bool(st["separate"]), bg_mooney=bg_mode, stats_on="face", white_rule=st["white_rule"],
                         bright_white=float(st["bright_white"]) / 100, bright_border=float(st["bright_border"]),
                         preprocess=st["preprocess"], keep_manual=True, log=log, progress=prog)
        steps.append(("Mooneys", 3, step_batch))
    for label, fn in ((PIPE_SYSTEMS[0], vs.run_face), (PIPE_SYSTEMS[2], vs.run_depth), (PIPE_SYSTEMS[3], vs.run_shading)):
        if label in systems:
            steps.append((f"System {label}", 3, lambda log, prog, fn=fn: fn(PROJECT_ROOT, "Original", log=log, progress=prog)))
    if PIPE_SYSTEMS[1] in systems:
        def step_vlm(log, prog):
            model, only = pick_model(vs.status()["vlm_models"]), photo_set_stems(vlm_set or vlm_scope_label())
            if not model:
                log("Skipped: Ollama is not running or has no vision model (start it with `ollama serve`)."); return
            if only is not None and not only:
                log(f"Skipped: the set “{vlm_set}” holds no photos."); return
            save_settings(vlm_scope="all" if vlm_set == SET_ALL else (vlm_set or vlm_scope_label()))
            log(f"On: {vlm_set or vlm_scope_label()}" + ("" if only is None else f" ({len(only)} photos)"))
            vs.run_vlm(PROJECT_ROOT, model, source="Original", log=log, progress=prog, only=None if only is None else set(only))
        steps.append(("System 2 Vision-language model", 3, step_vlm))
    for label in d3:
        def step_3d(log, prog, label=label):
            basel = {"faces": int(c["n_faces"] or cfg.BASEL_FACES), "model": "Whole head", "poses": ["frontal"], "shadows": cfg.BASEL_SHADOWS}
            which = (c["d3_fit_set"] or fit_default_set()) if D3_SOURCES[label] == bfm_fit.SUB else c["d3_set"]
            if D3_SOURCES[label] != bfm.SUB:
                log(f"Photos: {which} ({len(photo_set_files(which))})")
            d3_job(["render", "Mooneys", "systems"], D3_SOURCES[label], photo_set_files(which),
                   [l for l in cfg.LIGHTS_3D if l in f3.LIGHTS], basel, lambda v, desc=None: prog(v))(log)
        steps.append((f"3D faces: {label}", 4, step_3d))
    if c["do_export"]:
        def step_export(log, prog):
            favs = load_favs()
            made = rs.export(PROJECT_ROOT, favs or rs.stems(PROJECT_ROOT)[:10])
            log(f"figures/: {len(made)} files (summary table, figure, one panel each for "
                + ("your favourites" if favs else "the first 10 photos") + ").")
            for s in (f3.SUB, bfm_fit.SUB, bfm.SUB):
                if d3_gallery("Mooney", s):
                    log(f"{s}/figures/: {len(a3.export(PROJECT_ROOT, s))} files.")
            made = a3.export_compare(PROJECT_ROOT)
            if made:
                log(f"figures/: comparison of the stimulus sets, {len(made)} files.")
        steps.append(("Export figures", 1, step_export))
    if c["do_save"]:
        def step_save(log, prog):
            saved["folder"], saved["msg"] = an.save_run(PROJECT_ROOT, RUNS_ROOT, name, include=[RUN_LABELS[x] for x in (keep or [])], note=note)
            log(saved["msg"])
        steps.append(("Save the run", 1, step_save))
    return steps


def r_sources():
    """Sources a run can take: the photos, and every live 3D dataset that has renders."""
    return [R_SOURCES[R_PHOTOS]] + [R_SOURCES[s] for s in (f3.SUB, bfm_fit.SUB, bfm.SUB) if d3_gallery("Original", s)]


def r_groups():
    seen = rq.groups(PROJECT_ROOT)
    for j in r_saved_runs():
        if j["group"] not in seen and j["group"] != R_NO_GROUP:
            seen.append(j["group"])
    return seen or [rq.DEFAULT_GROUP]


def run_config(name, group, note, source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, on):
    """The form of a new run as a plain dict (it is stored in the queue file and, in short, with the saved run)."""
    on = {**R_ON, **(on or {})}
    src = R_SOURCE_KEY.get(source, R_PHOTOS)
    saved_set = None if saved_set in (None, "", R_NO_SET) else saved_set
    if saved_set:                                           # a saved 3D dataset decides the source
        j = next((x for x in d3_set_list() if x["folder"] == saved_set), None)
        src = j.get("sub", src) if j else src
    nums = sorted(R_SYS_KEY[s] for s in (systems or []) if s in R_SYS_KEY)
    if src != R_PHOTOS:
        nums = [n for n in nums if n != 2]
    return {"v": 2, "name": str(name or "").strip(), "group": str(group or "").strip() or rq.DEFAULT_GROUP, "note": str(note or "").strip(),
            "source": src, "saved_set": saved_set if src != R_PHOTOS else None, "vlm_set": vlm_set,
            "mooneys": bool(on["mooneys"]), "sigma": float(sigma), "white": float(white), "bg_mode": bg_mode, "bg_level": int(bg_level),
            "systems": nums if on["systems"] else [], "export": bool(on["export"]), "save": bool(on["save"]), "keep": list(keep or [])}


def run_is_old(c):
    return c.get("v") != 2


def run_refusal(c):
    """Why a run cannot start, or None."""
    if run_is_old(c):
        return pipeline_refusal({**{k: None for k in PIPE_KEYS}, **c})
    if c["save"] and not an.run_slug(c["name"]):
        return "Not started: give the run a name first, or switch the stage *Save* off."
    if not (c["mooneys"] or c["systems"] or c["export"] or c["save"]):
        return "Not started: every stage is switched off."
    if c["source"] == R_PHOTOS:
        if not images_in("Original"):
            return "Not started: there are no photos yet. Collect some in **Dataset → Photos** first."
        if not c["mooneys"] and not images_in("Mooney"):
            return "Not started: there are no Mooneys yet. Switch the stage *Mooneys* on."
        return None
    sub = c["source"]
    if c.get("saved_set"):
        if not os.path.isdir(P(D3_SETS, c["saved_set"], "data")):
            return f"Not started: the saved 3D dataset “{c['saved_set']}” is not there any more."
        return None
    if not d3_gallery("Original", sub):
        return f"Not started: `{sub}/` holds no renders. Make the 3D dataset in **Dataset → 3D datasets** first."
    if c["systems"] and not c["mooneys"] and not d3_gallery("Mooney", sub):
        return f"Not started: the renders in `{sub}/` have no Mooneys yet. Switch the stage *Mooneys* on."
    return None


def run_stage_titles(c):
    """The chips of a run, in the order its steps run."""
    if run_is_old(c):
        return [t for t, _w, _f in pipeline_steps({**{k: None for k in PIPE_KEYS}, **c}, {})]
    out = (["Load 3D dataset"] if c.get("saved_set") else []) + (["Mooneys"] if c["mooneys"] else [])
    if c["source"] == R_PHOTOS:
        out += [R_SYS_SHORT[n] for n in (1, 3, 4, 2) if n in c["systems"]]
    elif c["systems"]:
        out += ["Systems " + ", ".join(str(n) for n in c["systems"]) + " and the truth"]
    return out + (["Export"] if c["export"] else []) + (["Save"] if c["save"] else [])


def run_settings_line(c):
    """A few words on what a run does to the Mooneys, for the board."""
    if run_is_old(c):
        return "set up in an earlier version"
    if c["mooneys"]:
        return f"blur {c['sigma']:g} · {c['white']:g} % · {c['bg_mode']}"
    return "Mooneys as they are"


def run_source_line(c):
    if run_is_old(c):
        return "photos and 3D in one run"
    if c["source"] == R_PHOTOS:
        return f"Photos · {len(images_in('Original'))}"
    if c.get("saved_set"):
        return f"{R_SOURCES.get(c['source'], c['source'])} · saved “{c['saved_set']}”"
    return f"{R_SOURCES.get(c['source'], c['source'])} · {len(d3_gallery('Original', c['source']))}"


def d3_set_put_live(folder, log=print):
    """Copy a saved 3D dataset over the live folder of its source. Returns the sub-folder, or None if it cannot be read."""
    j = next((x for x in d3_set_list() if x["folder"] == folder), None)
    src = P(D3_SETS, folder, "data")
    if not j or j.get("sub") not in D3_SOURCES.values() or not os.path.isdir(src):
        log(f"The saved 3D dataset “{folder}” cannot be read."); return None
    sub = j["sub"]
    log(f"Loading “{j.get('name', folder)}” as {sub}/. It replaces: {d3_line(sub)}. The saved copy stays as it is.")
    shutil.rmtree(P(sub), ignore_errors=True)
    shutil.copytree(src, P(sub))
    return sub


def run_steps(c, saved):
    """The steps of one run as (title, weight, fn(log, prog)). Every step calls the same function as its own page."""
    if run_is_old(c):
        return pipeline_steps({**{k: None for k in PIPE_KEYS}, **c}, saved)
    titles, steps = run_stage_titles(c), []
    sigma, white, bg_mode, bg_level, sub = c["sigma"], c["white"], c["bg_mode"], c["bg_level"], c["source"]
    photos = sub == R_PHOTOS

    def values(log, kept):
        st = save_settings(sigma=float(sigma), white=float(white), bg_mode=bg_mode, bg_level=int(bg_level))
        log(f"Values: blur {st['sigma']:g} · white {st['white']:g} % · background {st['bg_mode']} · phase-1 gray {st['bg_level']}. "
            "Saved as the project's defaults." + kept)
        return st

    if c.get("saved_set"):
        def step_load(log, prog):
            if d3_set_put_live(c["saved_set"], log) is None:
                raise RuntimeError("the saved 3D dataset could not be loaded")
        steps.append((1, step_load))
    if c["mooneys"] and photos:
        def step_batch(log, prog):
            st = values(log, " Mooneys adjusted by hand are kept.")
            bt.run_batch(PROJECT_ROOT, source="Original", sigmas=(float(sigma),), whites=(float(white) / 100,), bg_level=int(bg_level),
                         separate=bool(st["separate"]), bg_mooney=bg_mode, stats_on="face", white_rule=st["white_rule"],
                         bright_white=float(st["bright_white"]) / 100, bright_border=float(st["bright_border"]),
                         preprocess=st["preprocess"], keep_manual=True, log=log, progress=prog)
        steps.append((3, step_batch))
    elif c["mooneys"]:
        def step_batch3d(log, prog):
            values(log, "")
            d3_job(["Mooneys"], sub, [], [], {}, lambda v, desc=None: prog(v))(log)
        steps.append((3, step_batch3d))
    if photos:
        for n, fn in ((1, vs.run_face), (3, vs.run_depth), (4, vs.run_shading)):
            if n in c["systems"]:
                steps.append((3, lambda log, prog, fn=fn: fn(PROJECT_ROOT, "Original", log=log, progress=prog)))
        if 2 in c["systems"]:
            def step_vlm(log, prog):
                vlm_set = c.get("vlm_set") or vlm_scope_label()
                model, only = pick_model(vs.status()["vlm_models"]), photo_set_stems(vlm_set)
                if not model:
                    log("Skipped: Ollama is not running or has no vision model (see Experiments → Vision systems → settings)."); return SKIPPED
                if only is not None and not only:
                    log(f"Skipped: the set “{vlm_set}” holds no photos."); return SKIPPED
                save_settings(vlm_scope="all" if vlm_set == SET_ALL else vlm_set)
                log(f"On: {vlm_set}" + ("" if only is None else f" ({len(only)} photos)"))
                vs.run_vlm(PROJECT_ROOT, model, source="Original", log=log, progress=prog, only=None if only is None else set(only))
            steps.append((3, step_vlm))
    elif c["systems"]:
        steps.append((6, lambda log, prog: d3_job(["systems"], sub, [], [], {}, lambda v, desc=None: prog(v), systems=c["systems"],
                                                  vlm_set=c.get("vlm_set"))(log)))
    if c["export"]:
        def step_export(log, prog):
            if photos:
                favs = load_favs()
                made = rs.export(PROJECT_ROOT, favs or rs.stems(PROJECT_ROOT)[:10])
                log(f"figures/: {len(made)} files (summary table, figure, one panel each for "
                    + ("your favourites" if favs else "the first 10 photos") + ").")
            elif d3_gallery("Mooney", sub):
                log(f"{sub}/figures/: {len(a3.export(PROJECT_ROOT, sub))} files.")
            made = a3.export_compare(PROJECT_ROOT)
            if made:
                log(f"figures/: comparison of the stimulus sets, {len(made)} files.")
        steps.append((1, step_export))
    if c["save"]:
        def step_save(log, prog):
            facts = {k: c[k] for k in ("source", "saved_set", "mooneys", "systems", "export", "vlm_set")}
            facts["source_name"] = R_SOURCES.get(sub, sub)
            saved["folder"], saved["msg"] = an.save_run(PROJECT_ROOT, RUNS_ROOT, c["name"], include=[RUN_LABELS[x] for x in c["keep"] if x in RUN_LABELS],
                                                        note=c["note"], group=c["group"], run=facts)
            log(saved["msg"])
            if not saved["folder"]:
                raise RuntimeError(saved["msg"])
        steps.append((1, step_save))
    return [(t, w, fn) for t, (w, fn) in zip(titles, steps)]


SKIPPED = "skipped"
_PROGRESS_LINE = re.compile(r"^\s*\[?\s*(\d+)\s*/\s*(\d+)\s*\]?(\s|$)")


def queue_run_one(c, log, report=None):
    """What the queue does with one run. Tells the queue where the run is; returns its line and its saved folder."""
    init_project()
    report = report or (lambda *a, **k: None)
    refuse = run_refusal(c)
    if refuse:
        raise RuntimeError(refuse.replace("**", "").replace("*", "").replace("`", ""))
    saved = {}
    steps = run_steps(c, saved)
    stages = [{"title": t, "state": rq.WAITING, "detail": ""} for t, _w, _f in steps]
    total, done, t0 = sum(w for _t, w, _f in steps) or 1, 0, time.time()
    try:
        check_stop()
        log(f"Run “{c.get('name') or 'unnamed'}” · {run_source_line(c)} · {len(steps)} stage{'s' if len(steps) != 1 else ''}.")
        for k, (title, w, fn) in enumerate(steps):
            cur = stages[k]
            cur["state"], seen = rq.RUNNING, {"count": False}
            report(stages, done / total)

            def slog(msg="", cur=cur, seen=seen):
                check_stop()
                m = _PROGRESS_LINE.match(str(msg))
                if m and int(m.group(2)) > 0:
                    cur["detail"], seen["count"] = f"{m.group(1)} / {m.group(2)}", True
                log(msg)

            def prog(p, *a, cur=cur, seen=seen, done=done, w=w, **kw):
                check_stop()
                p = min(max(float(p), 0.0), 1.0)
                if not seen["count"]:
                    cur["detail"] = f"{p:.0%}"
                report(stages, (done + w * p) / total)
            log(""); log(f"── Stage {k + 1} of {len(steps)}: {title}")
            r = fn(slog, prog)
            cur["state"], cur["detail"] = (SKIPPED if r == SKIPPED else rq.DONE), ""
            done += w
            report(stages, done / total)
        log(""); log(f"Finished in {(time.time() - t0) / 60:.1f} min.")
    except BaseException:
        report(stages, done / total)
        raise
    finally:
        _stop["on"] = False
    return {"message": saved.get("msg") or ("finished; not saved (the stage Save was off)" if not c.get("save", c.get("do_save"))
                                            else "finished, but not saved; see the log"), "folder": saved.get("folder") or ""}


# ---------------------------------------------------------------------------- the board of groups and runs
def r_board(open_logs=()):
    items = rq.snapshot(PROJECT_ROOT)
    facts = {it["id"]: (run_settings_line(it["config"]), run_source_line(it["config"]), run_is_old(it["config"])) for it in items}
    return rb.board(items, open_logs or (), rq.is_running(), _busy["what"], facts)


def r_state_line():
    items = rq.snapshot(PROJECT_ROOT)
    waiting = sum(1 for it in items if it["state"] == rq.WAITING)
    cur = next((it for it in items if it["state"] == rq.RUNNING), None)
    if rq.is_running() and cur:
        return f"**Running:** “{cur['name']}” in group “{cur['group']}” · {waiting} waiting."
    if _busy["what"]:
        return f"{_busy['what'].capitalize()} is working on another page, so no run can start now. {waiting} waiting."
    return f"Nothing is running. {waiting} run{'s' if waiting != 1 else ''} waiting." if items else "Nothing is running."


def r_view(open_logs, msg=None):
    """Board, state line, its signature (so the timer can skip a refresh when nothing changed)."""
    html_ = r_board(open_logs)
    line = r_state_line()
    out = (html_, line, hash((html_, line)))
    return out + ((msg,) if msg is not None else ())


def r_tick(open_logs, sig, hsig, picked, armed, opened):
    """The timer: redraw the board and the history only when something changed."""
    html_, line, new = r_view(open_logs)
    hist = r_history(picked, armed, opened)
    hnew = hash(hist)
    return (gr.update() if new == sig else html_, gr.update() if new == sig else line, new,
            gr.update() if hnew == hsig else hist, hnew, gr.update() if hnew == hsig else status_html())


def r_start(open_logs, group=None, only=None):
    if rq.is_running():
        return r_view(open_logs, "A run is working already. Runs go one after another.")
    if busy_now():
        return r_view(open_logs, busy_text())
    items = rq.snapshot(PROJECT_ROOT)
    todo = [it for it in items if it["state"] == rq.WAITING and (group is None or it["group"] == group) and (only is None or it["id"] == only)]
    if not todo:
        return r_view(open_logs, "Nothing is waiting. Build a run above and press *Add to group*.")
    _stop["on"] = False
    rq.start(PROJECT_ROOT, queue_run_one, group=group, only=only)
    what = f"“{todo[0]['name']}”" if only else (f"group “{group}” ({len(todo)} run{'s' if len(todo) != 1 else ''})" if group
                                                else f"all groups ({len(todo)} runs)")
    return r_view(open_logs, f"Started {what}. You can close the browser window, but the app (the terminal window) must stay "
                             "open and the computer awake.")


def r_stop_after(open_logs):
    if not rq.is_running():
        return r_view(open_logs, "Nothing is running.")
    rq.stop_after_current()
    return r_view(open_logs, "The run that is working now will finish; the ones after it stay waiting.")


def r_stop_now(open_logs):
    if not rq.is_running():
        return r_view(open_logs, "Nothing is running.")
    _stop["on"] = True
    return r_view(open_logs, "Stopping now: the run ends after the image it is working on and is marked *stopped*; it is not saved. "
                             "The runs after it stay waiting. " + STOP_NOTE)


def r_clear(open_logs):
    rq.clear_finished(PROJECT_ROOT)
    return r_view(open_logs, "Finished, failed and stopped runs were taken off the board. Their saved copies are in History.")


# ---------------------------------------------------------------------------- the form of a new run
def r_summaries(source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, on):
    """The one-line summary on each stage card."""
    on = {**R_ON, **(on or {})}
    src = R_SOURCE_KEY.get(source, R_PHOTOS)
    saved_set = None if saved_set in (None, "", R_NO_SET) else saved_set
    if saved_set:
        j = next((x for x in d3_set_list() if x["folder"] == saved_set), None)
        s_source = f"saved 3D dataset<br><b>{rb.esc(j.get('name', saved_set) if j else saved_set)}</b>"
        src = j.get("sub", src) if j else src
    elif src == R_PHOTOS:
        s_source = f"Photos<br>{len(images_in('Original'))} photos, {len(images_in('Mooney'))} Mooneys"
    else:
        s_source = f"{rb.esc(R_SOURCES[src])}<br>{len(d3_gallery('Original', src))} renders, {len(d3_gallery('Mooney', src))} Mooneys"
    nums = [R_SYS_KEY[s] for s in (systems or []) if s in R_SYS_KEY and (src == R_PHOTOS or R_SYS_KEY[s] != 2)]
    s_sys = rb.chips([R_SYS_SHORT[n] for n in sorted(nums)]) or "no system ticked"
    if 2 in nums:
        s_sys += f"<br>system 2 on <b>{rb.esc(vlm_set or vlm_scope_label())}</b>"
    if src != R_PHOTOS and nums:
        s_sys += "<br>then the scores against the true depth"
    kept = [k for k in (keep or [])]
    return {
        "source": s_source,
        "mooneys": (f"make them: blur <b>{float(sigma):g}</b> · white <b>{float(white):g} %</b><br>{rb.esc(bg_mode)} background · gray {int(bg_level)}"
                    if on["mooneys"] else "keep the Mooneys as they are"),
        "systems": s_sys if on["systems"] else "no systems in this run",
        "export": ("summary table, figure" + (", panels" if src == R_PHOTOS else " by light") + "<br>and the comparison of the sets"
                   if on["export"] else "nothing is exported"),
        "save": ((f"a frozen copy with {len(kept)} part{'s' if len(kept) != 1 else ''}<br>besides the Mooneys" if kept
                  else "a frozen copy of the Mooneys and their values") if on["save"] else "the run is not saved"),
    }


def r_cards(sel, source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, on):
    return rb.cards(sel, {**R_ON, **(on or {})}, r_summaries(source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, on))


def r_panels(sel):
    return tuple(gr.update(visible=sel == k) for k in rb.STAGES)


def r_card_click(sel, on, source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, evt: gr.EventData):
    d = evt._data or {}
    on = {**R_ON, **(on or {})}
    stage = d.get("stage") if d.get("stage") in rb.STAGES else sel
    if d.get("act") == "toggle" and stage != "source":
        on[stage] = not on[stage]
    sel = stage
    return (sel, on, r_cards(sel, source, saved_set, vlm_set, sigma, white, bg_mode, bg_level, systems, keep, on)) + r_panels(sel)


def r_source_changed(source, saved_set, systems):
    """Another source: the systems that fit it, and whether the set for system 2 is asked for."""
    saved_set = None if saved_set in (None, "", R_NO_SET) else saved_set
    src = R_SOURCE_KEY.get(source, R_PHOTOS)
    if saved_set:
        j = next((x for x in d3_set_list() if x["folder"] == saved_set), None)
        src = j.get("sub", src) if j else src
    photos = src == R_PHOTOS
    choices = [R_SYS[n] for n in (1, 2, 3, 4)]
    value = [s for s in (systems or []) if s in choices and (photos or R_SYS_KEY[s] != 2)] or [R_SYS[n] for n in (1, 3, 4)]
    return gr.update(choices=choices, value=value), gr.update(visible=True)


R_FORM_N = 20                # how many values r_fill returns: 15 fields and pictures, 5 panels


def r_fill(c, sel="source"):
    """Values for every field of the form from a run's description (for edit, copy and 'duplicate last run')."""
    on = {"mooneys": bool(c["mooneys"]), "systems": bool(c["systems"]), "export": bool(c["export"]), "save": bool(c["save"])}
    source = R_SOURCES.get(c["source"], R_SOURCES[R_PHOTOS])
    photos = c["source"] == R_PHOTOS
    sys_choices = [R_SYS[n] for n in (1, 2, 3, 4)]
    systems = [R_SYS[n] for n in c["systems"] if R_SYS[n] in sys_choices] or [R_SYS[n] for n in (1, 3, 4)]
    saved_set = c.get("saved_set") or R_NO_SET
    vlm_set = c.get("vlm_set") if c.get("vlm_set") in photo_set_choices() else vlm_scope_label()
    keep = [k for k in c.get("keep", []) if k in RUN_LABELS]
    return (c["name"], c["group"], c["note"], gr.update(choices=r_sources(), value=source if source in r_sources() else R_SOURCES[R_PHOTOS]),
            gr.update(choices=[R_NO_SET] + [f for _l, f in d3_set_choices()], value=saved_set),
            gr.update(choices=photo_set_choices(), value=vlm_set, visible=photos), c["sigma"], c["white"], c["bg_mode"], c["bg_level"],
            gr.update(choices=sys_choices, value=systems), keep, on, sel,
            r_cards(sel, source, saved_set, vlm_set, c["sigma"], c["white"], c["bg_mode"], c["bg_level"], systems, keep, on)) + r_panels(sel)


def r_add(open_logs, edit_id, *form):
    """Add to group, or (while a waiting run is being edited) change that run in place."""
    c = run_config(*form)
    if not an.run_slug(c["name"]):
        return r_view(open_logs, "Not added: give the run a name first. Each run is saved under its name.") + (edit_id, r_edit_note(edit_id))
    if not c["save"]:
        return r_view(open_logs, "Not added: switch the stage *Save* on. A run in a group that is not saved would be overwritten "
                                 "by the next one. To run without saving, use *Run now*.") + (edit_id, r_edit_note(edit_id))
    refuse = run_refusal(c)
    if refuse:
        return r_view(open_logs, refuse.replace("Not started", "Not added")) + (edit_id, r_edit_note(edit_id))
    if edit_id and rq.replace_config(PROJECT_ROOT, edit_id, c["name"], c, c["group"], run_stage_titles(c)):
        return r_view(open_logs, f"Changed “{c['name']}” in group “{c['group']}”. It keeps its place.") + ("", r_edit_note(""))
    rq.add(PROJECT_ROOT, c["name"], c, c["group"], run_stage_titles(c))
    return r_view(open_logs, f"Added “{c['name']}” to group “{c['group']}”. It keeps the values of the form as they are now. "
                             "Press *Run group* on the group, or *Run all groups*.") + ("", r_edit_note(""))


def r_run_now(open_logs, *form):
    """Add the run and start it at once (only this run)."""
    c = run_config(*form)
    if not c["name"]:
        c["name"] = f"run {time.strftime('%H:%M')}"
    refuse = run_refusal(c)
    if refuse:
        return r_view(open_logs, refuse)
    if busy_now():
        return r_view(open_logs, busy_text())
    item = rq.add(PROJECT_ROOT, c["name"], c, c["group"], run_stage_titles(c))
    return r_start(open_logs, only=item)[:3] + (f"Started “{c['name']}”. Its stages fill in below as it runs.",)


def r_edit_note(edit_id):
    it = rq.get(PROJECT_ROOT, edit_id) if edit_id else None
    return (f"Editing “{it['name']}”: *Add to group* now changes that run in place. "
            "Press *Cancel edit* to add a new run instead.") if it else ""


def r_board_click(open_logs, edit_id, evt: gr.EventData):
    """A click on the board. Returns board, state line, signature, message, open logs, id being edited, edit note,
    and - for edit and copy - the whole form; everything else leaves the form alone."""
    d = evt._data or {}
    what, item_id, group = d.get("act"), d.get("id") or "", d.get("group") or ""
    open_logs = list(open_logs or [])
    form = (gr.update(),) * R_FORM_N
    msg, want = "", ""
    it = rq.get(PROJECT_ROOT, item_id) if item_id else None
    if what == "run-group":
        return r_start(open_logs, group=group) + (open_logs, edit_id, r_edit_note(edit_id), want) + form
    if what == "run-one" and it:
        return r_start(open_logs, only=item_id) + (open_logs, edit_id, r_edit_note(edit_id), want) + form
    if what == "stop-now":
        return r_stop_now(open_logs) + (open_logs, edit_id, r_edit_note(edit_id), want) + form
    if what == "results" and it and it.get("folder"):
        want, msg = it["folder"], "The saved copy of this run is open in the history below."
    if what in ("up", "down"):
        msg = "" if rq.move(PROJECT_ROOT, item_id, -1 if what == "up" else 1) else "It is already at the end of the waiting runs of its group."
    elif what == "remove":
        msg = "Removed from the board." if rq.remove(PROJECT_ROOT, item_id) else "A run that is working cannot be removed; stop it first."
        edit_id = "" if edit_id == item_id else edit_id
    elif what == "again":
        msg = ("Set back to waiting; it runs again with the same description." if rq.requeue(PROJECT_ROOT, item_id)
               else "Only a finished, failed or stopped run can be set back.")
    elif what == "log":
        open_logs = [x for x in open_logs if x != item_id] if item_id in open_logs else open_logs + [item_id]
    elif what in ("edit", "dup") and it and not run_is_old(it["config"]):
        c = dict(it["config"])
        if what == "dup":
            c["name"] = f"{c['name']} copy"
            edit_id, msg = "", f"The form holds a copy of “{it['name']}”. Change what you like and press *Add to group*."
        else:
            edit_id, msg = item_id, f"The form holds “{it['name']}”."
        form = r_fill(c)
    return r_view(open_logs, msg) + (open_logs, edit_id, r_edit_note(edit_id), want) + form


# ---------------------------------------------------------------------------- history of saved runs
R_NO_GROUP = "no group"
_kpi_cache = {}


def r_show(folder):
    """Facts, summary table and figure of a saved run for the history; the table and figure are hidden when there is nothing to show."""
    if not folder:
        return "", gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    facts, table, fig = run_show(folder)
    return facts, gr.update(value=table, visible=bool(table)), gr.update(value=fig, visible=fig is not None)


def r_compare(picked):
    if len(picked or []) < 2:
        return (gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                "Tick *compare* on one more run to see them side by side." if picked else "")
    head, rows, shead, srows = an.compare_runs(PROJECT_ROOT, RUNS_ROOT, picked)
    return (gr.update(value=rows or None, headers=head, visible=bool(rows)), gr.update(value=srows or None, headers=shead, visible=bool(srows)),
            "" if rows else "These runs hold no result tables to compare.")


def run_kpis(folder):
    """A few key numbers of a saved run, read from its scores.json (written when the run is saved, or filled in later)."""
    if folder in _kpi_cache:
        return _kpi_cache[folder]
    path, out = os.path.join(RUNS_ROOT, folder), []
    try:
        sc = an.read_scores(path) or an.run_scores(path)
        src, pct = sc["run"].get("source"), lambda k, n: f"{100 * k / n:.0f} %"
        if src in sc.get("sets_3d", {}):
            b = sc["sets_3d"][src]
            dv, fd = b["depth_vs_truth"], b["face_detected"]
            for label, v in (("depth 3 vs truth, render", dv["3_learned_depth"]["render"]), ("Mooney", dv["3_learned_depth"]["mooney"]),
                             ("depth 4 vs truth, render", dv["4_depth_from_shading"]["render"]), ("Mooney", dv["4_depth_from_shading"]["mooney"])):
                if v is not None:
                    out.append((label, f"{v:.2f}"))
            if fd.get("mooney_reblurred") and fd["mooney_reblurred"]["n"]:
                out.append(("face after re-blur", pct(fd["mooney_reblurred"]["yes"], fd["mooney_reblurred"]["n"])))
        else:
            s_ = sc.get("systems", {})
            fd = (s_.get("1_face_model") or {}).get("face_detected")
            if fd and fd.get("mooney_n"):
                out += [("face in Mooney", pct(fd["mooney"], fd["mooney_n"])), ("after re-blur", pct(fd["mooney_reblurred"], fd["mooney_n"]))]
            for key, label in (("3_learned_depth", "depth 3"), ("4_depth_from_shading", "depth 4")):
                m = ((s_.get(key) or {}).get("mooney_vs_original") or {}).get("mean")
                if m is not None:
                    out.append((label, f"{m:.2f}"))
            v = s_.get("2_vision_language_model")
            if v and v.get("n_mooney"):
                out.append(("VLM sees a face", pct(v["face_present_yes"]["mooney"], v["n_mooney"])))
    except Exception:                                       # a run saved by hand or with other tables: no key numbers
        out = []
    _kpi_cache[folder] = out
    return out


def r_saved_runs():
    """Saved runs, newest first, as the history shows them."""
    out = []
    for d in reversed(an.list_runs(RUNS_ROOT)):
        try:
            j = json.load(open(os.path.join(RUNS_ROOT, d, "run.json"), encoding="utf-8"))
        except Exception:
            j = {}
        u, run = j.get("used") or {}, j.get("run") or {}
        settings = " · ".join(x for x in (f"blur {u['Blur σ (px)']}" if u.get("Blur σ (px)") else "", u.get("White target", ""),
                                          u.get("Background", ""), f"{u['Adjusted by hand']} by hand" if u.get("Adjusted by hand") else "") if x)
        sub = run.get("source")
        if sub and sub != R_PHOTOS:
            folder3d = os.path.join(RUNS_ROOT, d, sub, "Original")
            n = len([f for f in os.listdir(folder3d) if f.lower().endswith(".png")]) if os.path.isdir(folder3d) else ""
        else:
            n = (j.get("counts") or {}).get("Mooney", "")
        out.append({"folder": d, "name": j.get("name", d), "group": j.get("group") or R_NO_GROUP, "saved": j.get("saved", ""),
                    "settings": settings, "source": (run.get("source_name") or "Photos") + (f" · {n}" if n else ""),
                    "note": j.get("note", "")})
    return out


def r_history(picked=(), armed=None, opened=None):
    runs = r_saved_runs()
    for r in runs:
        r["kpis"] = run_kpis(r["folder"])
    return rb.history(runs, picked or (), armed, opened)


def r_history_click(picked, armed, opened, evt: gr.EventData):
    """A click in the history: open, compare, delete. Returns history, picked, armed, opened, facts, table, figure,
    comparison table, settings table, comparison note, the runs offered in Analysis → Compare runs, message."""
    d = evt._data or {}
    what, folder = d.get("act"), d.get("id") or ""
    picked, msg = [p for p in (picked or []) if os.path.isdir(os.path.join(RUNS_ROOT, p))], ""
    facts = table = fig = gr.update()
    if what == "open":
        opened = None if opened == folder else folder
        facts, table, fig = r_show(opened)
        armed = None
    elif what == "pick":
        picked = [p for p in picked if p != folder] if folder in picked else picked + [folder]
    elif what == "publish" and folder and os.path.isdir(os.path.join(RUNS_ROOT, folder)):
        try:
            j = json.load(open(os.path.join(RUNS_ROOT, folder, "run.json"), encoding="utf-8"))
        except Exception:
            j = {}
        _out, _n, msg = pub.publish(os.path.join(RUNS_ROOT, folder), DATA_ROOT, j.get("name") or folder, j.get("note", ""))
        msg += " Nothing was committed or pushed; do that yourself when you want it on GitHub."
    elif what == "delete":
        armed = folder
    elif what == "delete-no":
        armed = None
    elif what == "delete-yes" and folder and os.path.isdir(os.path.join(RUNS_ROOT, folder)):
        trash = os.path.join(RUNS_ROOT, "_trash")
        os.makedirs(trash, exist_ok=True)
        dst, k = os.path.join(trash, folder), 2
        while os.path.exists(dst):
            dst = os.path.join(trash, f"{folder}-{k}"); k += 1
        shutil.move(os.path.join(RUNS_ROOT, folder), dst)
        _kpi_cache.pop(folder, None)
        an.write_scores_index(RUNS_ROOT)
        picked, armed = [p for p in picked if p != folder], None
        if opened == folder:
            opened, (facts, table, fig) = None, r_show(None)
        msg = f"Moved “{folder}” to `runs/_trash/`. Nothing was erased; delete that folder yourself to free the space."
    cmp_table, cmp_settings, cmp_note = r_compare(picked)
    hist = r_history(picked, armed, opened)
    return (hist, picked, armed, opened, facts, table, fig, cmp_table, cmp_settings, cmp_note,
            gr.update(choices=run_choices()), msg, hash(hist))


def r_open_result(folder, picked, armed, opened):
    """'results' on a finished run of the board: open its saved copy in the history."""
    if not folder or not os.path.isdir(os.path.join(RUNS_ROOT, folder)):
        return (gr.update(),) * 6
    hist = r_history(picked, armed, folder)
    return (hist, hash(hist), folder) + r_show(folder)


def r_duplicate():
    """'Duplicate last run': the form takes the description of the run added last."""
    items = [it for it in rq.snapshot(PROJECT_ROOT) if not run_is_old(it["config"])]
    if not items:
        return (gr.update(),) * R_FORM_N + ("There is no earlier run on the board to copy from.",)
    last = max(enumerate(items), key=lambda p: (p[1].get("added", ""), p[0]))[1]
    c = dict(last["config"])
    c["name"] = f"{c['name']} copy"
    return r_fill(c) + (f"The form holds a copy of “{last['name']}”. Change what you like and press *Add to group*.",)


def r_publish_state(name):
    """'Publish the current project state': the tables of the project as it is now, into data/<name>/ next to the code."""
    if not str(name or "").strip():
        return "Give the published results a name first."
    _out, _n, msg = pub.publish(PROJECT_ROOT, DATA_ROOT, name, "the project as it was when published")
    return msg + (" Nothing was committed or pushed; do that yourself when you want it on GitHub." if _out else "")


def r_page(open_logs, picked, armed, opened, edit_id):
    """The Runs page is opened: board, history, and the choices of the form that depend on the project."""
    for _f in an.backfill_scores(RUNS_ROOT):                # older runs get their scores.json; nothing else in them changes
        _kpi_cache.pop(_f, None)
    html_, line, sig = r_view(open_logs)
    hist = r_history(picked, armed, opened)
    return (html_, line, sig, hist, hash(hist), gr.update(choices=r_sources()),
            gr.update(choices=[R_NO_SET] + [f for _l, f in d3_set_choices()]), gr.update(choices=photo_set_choices()),
            gr.update(choices=r_groups()), r_edit_note(edit_id))


# ---------------------------------------------------------------------------- reading one saved run
def run_folder_choices():
    return list(reversed(an.list_runs(RUNS_ROOT)))


def run_show(folder):
    """Facts, summary table and figure of one saved run, read from its folder. Nothing in the run is changed."""
    if not folder or not os.path.isdir(os.path.join(RUNS_ROOT, folder)):
        return "Choose a saved run.", None, None
    path = os.path.join(RUNS_ROOT, folder)
    try:
        j = json.load(open(os.path.join(path, "run.json"), encoding="utf-8"))
    except Exception:
        j = {}
    lines = [f"### {j.get('name', folder)}", f"Saved **{j.get('saved', '?')}** · folder `runs/{folder}/`"
             + (f" · note: {j['note']}" if j.get("note") else ""), ""]
    used = j.get("used") or {}
    if used:
        lines += ["| Made with | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in used.items()] + [""]
    counts = {k: v for k, v in (j.get("counts") or {}).items() if v}
    if counts:
        lines.append("Holds: " + " · ".join(f"`{k}/` {v}" for k, v in counts.items()) + ".")
    rows = rs.summary(path)
    table = [[r.get(c, "") for c in SUMMARY_COLS] for r in rows] or None
    fig = None
    if rows:                                              # drawn into a temporary file, so the frozen run stays as it is
        fig = rs.figure(path, path=os.path.join(tempfile.gettempdir(), f"mooney_run_{an.run_slug(folder)}.png"))
    if not rows:
        lines.append("\nThis run holds no result tables.")
    return "\n".join(lines), table, fig



# ============================================================================ Vision systems on any dataset
V_PHOTOS = "Photos"
V_SYSTEMS = ["1 Face model", "2 Vision-language model", "3 Learned depth", "4 Depth from shading", "5 3D face mesh (optional)"]
V_3D = [V_SYSTEMS[0], V_SYSTEMS[1], V_SYSTEMS[2], V_SYSTEMS[3]]      # what can run on a 3D dataset
V_3D_DEFAULT = [V_SYSTEMS[0], V_SYSTEMS[2], V_SYSTEMS[3]]            # ticked at first (system 2 is slow)


def v_datasets():
    """Photos, and every 3D source that has Mooneys to run on."""
    return [V_PHOTOS] + [label for label, sub in D3_SOURCES.items() if d3_gallery("Mooney", sub)]


def v_results(dataset):
    """What the Vision systems page shows below the log for the chosen dataset: line, table, figure."""
    if dataset in D3_SOURCES:
        sub = D3_SOURCES[dataset]
        head, rows = a3.light_table(PROJECT_ROOT, sub)
        vh, vr = a3.vlm_table(PROJECT_ROOT, sub)
        vlm = ("\n\n**System 2 against the light and pose we set** (`" + sub + "/analysis/vlm_truth.csv`)\n\n| " + " | ".join(vh) + " |\n|"
               + "---|" * len(vh) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in vr)) if vr else ""
        return (f"`{sub}/`: {d3_line(sub)}. The table is by light direction, against the true depth." + vlm,
                gr.update(value=rows or None, headers=head, label="One row per light direction"),
                a3.fig_truth(PROJECT_ROOT, sub) if rows else None)
    rows = rs.summary(PROJECT_ROOT)
    table = [[r.get(c, "") for c in SUMMARY_COLS] for r in rows] or None
    return ("" if rows else "No results for the photos yet.",
            gr.update(value=table, headers=SUMMARY_COLS, label="Summary   counts are images, other values are means"),
            rs.figure(PROJECT_ROOT) if rows else None)


def v_switch(dataset, ticks):
    """Another dataset was chosen: offer the systems that can run on it and show its results."""
    if dataset in D3_SOURCES:
        choices, value = V_3D, [t for t in (ticks or []) if t in V_3D] or V_3D_DEFAULT
    else:
        choices, value = V_SYSTEMS, [t for t in (ticks or []) if t in V_SYSTEMS] or V_3D_DEFAULT
    return (gr.update(choices=choices, value=value),) + v_results(dataset)


def v_page(dataset, ticks):
    """The page is opened: current datasets, the saved choice for system 2, results."""
    ds = v_datasets()
    dataset = dataset if dataset in ds else V_PHOTOS
    return (gr.update(choices=ds, value=dataset), gr.update(choices=photo_set_choices(), value=vlm_scope_label())) + v_switch(dataset, ticks)


def vision_run(dataset, ticks, source, identify, model, prompt, scope, progress=gr.Progress()):
    """Run the ticked systems on the chosen dataset, one after another, with one log."""
    init_project()
    ticks = [t for t in V_SYSTEMS if t in (ticks or [])]
    final = lambda: (status_html(),) + v_results(dataset)
    if not ticks:
        yield ("Tick at least one system.",) + (gr.update(),) * 4
        return
    if dataset in D3_SOURCES:
        nums = [int(t[0]) for t in ticks if t in V_3D]
        job = d3_job(["systems"], D3_SOURCES[dataset], None, None, {}, progress, systems=nums, vlm_set=scope)
        yield from run_logged(job, 4, final, what="the vision systems")
        return
    if not images_in("Mooney"):
        yield ("There are no Mooneys yet. Make them in Mooney → Batch first.",) + (gr.update(),) * 4
        return
    steps = []
    for t in ticks:
        n = int(t[0])
        if n == 1:
            steps.append((t, lambda log, prog: vs.run_face(project=PROJECT_ROOT, log=log, progress=prog, source=source, identify=bool(identify))))
        elif n == 2:
            def step_vlm(log, prog):
                extra = vlm_args(source, identify, model, prompt, scope)
                if extra is None:
                    log("Skipped: either the chosen set of photos is empty or no Ollama model is selected "
                        "(start Ollama with `ollama serve`, pull a vision model, then press the check button in the settings below).")
                    return
                log(f"On: {scope}" + ("" if extra["only"] is None else f" ({len(extra['only'])} photos)"))
                vs.run_vlm(project=PROJECT_ROOT, log=log, progress=prog, **extra)
            steps.append((t, step_vlm))
        elif n == 3:
            steps.append((t, lambda log, prog: vs.run_depth(project=PROJECT_ROOT, log=log, progress=prog, source=source)))
        elif n == 4:
            steps.append((t, lambda log, prog: vs.run_shading(project=PROJECT_ROOT, log=log, progress=prog, source=source)))
        else:
            steps.append((t, lambda log, prog: vs.run_mesh(project=PROJECT_ROOT, log=log, progress=prog, source=source)))

    def job(log):
        for k, (title, fn) in enumerate(steps):
            if k:
                log("")
            if len(steps) > 1:
                log(f"── {k + 1} of {len(steps)}: system {title}")
            fn(log, stoppable(lambda p, k=k, title=title: progress((k + min(max(float(p), 0), 1)) / len(steps), desc=title)))
    yield from run_logged(job, 4, final, what="the vision systems")


# ---------------------------------------------------------------------------- Ollama (the server behind system 2)
# Three fixed actions only: look, start the server, download a model through the server's own API.
# There is no command box anywhere in the app on purpose: it would hand a shell to anyone who can open the page.
OLLAMA_LOG = os.path.join(WORK_ROOT, "ollama_server.log")
MODEL_NAME = re.compile(r"^[A-Za-z0-9._/-]+(:[A-Za-z0-9._-]+)?$")


def ollama_status():
    """One line: is the program installed, does the server answer, which models are there."""
    exe, models = shutil.which("ollama"), vs.ollama_models()
    prog = f"program installed (`{exe}`)" if exe else "program **not installed** ([get it here](https://ollama.com/download))"
    if models is None:
        srv = f"server **not answering** at `{vs.OLLAMA}`"
    else:
        srv = f"server answering at `{vs.OLLAMA}` · " + ("models: " + ", ".join(f"`{m}`" for m in models) if models
                                                           else "**no model downloaded yet**")
    return f"Ollama: {prog} · {srv}"


def ollama_refresh(msg=""):
    models = vs.ollama_models() or []
    return ollama_status() + (f"  \n{msg}" if msg else ""), gr.update(choices=models, value=pick_model(models)), vs.status()["vlm"]


def ollama_start():
    """Start `ollama serve` in the background if the program is installed and the server does not answer."""
    if vs.ollama_models() is not None:
        return ollama_refresh("The server is already running; nothing was started.")
    exe = shutil.which("ollama")
    if not exe:
        return ollama_refresh("Not started: the Ollama program is not installed on this computer. Install it from "
                              "https://ollama.com/download, then press *Start Ollama* again.")
    try:
        os.makedirs(WORK_ROOT, exist_ok=True)
        out = open(OLLAMA_LOG, "ab")
        subprocess.Popen([exe, "serve"], stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        return ollama_refresh(f"Not started: {e}")
    for _ in range(12):                                   # the server needs a moment
        time.sleep(0.5)
        if vs.ollama_models() is not None:
            return ollama_refresh(f"Started. It keeps running after this app is closed; its output goes to `{OLLAMA_LOG}`.")
    return ollama_refresh(f"Started, but the server does not answer yet. Press *Check* in a few seconds; its output is in `{OLLAMA_LOG}`.")


def ollama_pull(name, progress=gr.Progress()):
    """Download a model through the server's own API (POST /api/pull), with the progress in the page's log."""
    name = str(name or "").strip()
    keep = (gr.update(),) * 3
    if not MODEL_NAME.match(name):
        yield ("Not started: a model name looks like `llava:7b` (letters, digits, . _ / - and an optional :tag).",) + keep
        return
    if vs.ollama_models() is None:
        yield ("Not started: the Ollama server is not answering. Press Start Ollama first (settings of the systems).",) + keep
        return

    def job(log):
        log(f"Downloading {name} through {vs.OLLAMA}/api/pull. This can be several GB.")
        last = {"line": "", "t": 0.0}
        with requests.post(vs.OLLAMA + "/api/pull", json={"model": name, "name": name, "stream": True}, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                check_stop()
                if not raw:
                    continue
                d = json.loads(raw)
                if d.get("error"):                         # for example a model name the registry does not know
                    log(f"Not downloaded. The Ollama server answered: {d['error']}"); return
                done, total = d.get("completed"), d.get("total")
                line = d.get("status", "")
                if total:
                    progress(done / total if done else 0, desc=line)
                    line += f"  {100 * (done or 0) / total:.0f} % of {total / 1e9:.2f} GB"
                if line != last["line"] and (time.time() - last["t"] > 2 or not total):   # not one line per chunk
                    log(line); last.update(line=line, t=time.time())
        log(f"Finished. {name} is installed." if name in (vs.ollama_models() or [])
            else "The download ended, but the model is not in the server's list yet; press Check.")
    yield from run_logged(job, 3, lambda: ollama_refresh(f"`{name}`: see the log."), what="a model download")


def v_save_run(name, keep, note):
    folder, msg = an.save_run(PROJECT_ROOT, RUNS_ROOT, name, include=[RUN_LABELS[x] for x in (keep or [])], note=note)
    return msg, gr.update(choices=run_choices())


# ============================================================================ 3D datasets: named copies of a 3D source
D3_SETS = "3D_sets"
D3_SET_COLS = ["Name", "Source", "Saved", "Faces", "Lights", "Renders", "Mooneys", "Systems run", "Mooneys made with"]


def d3_set_list():
    """Saved 3D datasets, newest first: the content of each <project>/3D_sets/<name>/set.json."""
    out, base = [], P(D3_SETS)
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        try:
            j = json.load(open(os.path.join(base, d, "set.json"), encoding="utf-8"))
        except Exception:
            continue
        j["folder"] = d
        out.append(j)
    return sorted(out, key=lambda j: j.get("saved", ""), reverse=True)


def d3_set_table():
    rows = []
    for j in d3_set_list():
        st = j.get("status", {})
        rows.append([j.get("name", j["folder"]), j.get("source", ""), j.get("saved", ""), st.get("faces", ""), st.get("lights", ""),
                     st.get("renders", ""), st.get("mooneys", ""), ", ".join(str(n) for n in st.get("systems", [])) or "–",
                     j.get("mooney_settings", "")])
    return rows or None


def d3_set_choices():
    return [(f"{j.get('name', j['folder'])}  ·  {j.get('source', '')}", j["folder"]) for j in d3_set_list()]


def d3_sets_view(msg=None):
    out = (gr.update(value=d3_set_table(), headers=D3_SET_COLS), gr.update(choices=d3_set_choices()))
    return out + ((msg,) if msg is not None else ())


def d3_data_view(source):
    """State line and the two galleries of the live 3D set of a source."""
    sub = d3_sub(source)
    return (f"`{sub}/`: {d3_line(sub)}",
            gr.update(value=d3_gallery("Original", sub), label=f"{sub}/Original/   the renders"),
            gr.update(value=d3_gallery("Mooney", sub), label=f"{sub}/Mooney/   their Mooneys"))


def d3_set_save(source, name):
    """Copy the live folder of a source to 3D_sets/<name>/data and note what it is. The live folder is not changed."""
    sub, slug = d3_sub(source), subset_slug(name)
    if busy_now():
        return d3_sets_view(busy_text())
    if not slug:
        return d3_sets_view("Give the 3D dataset a name first.")
    if not d3_gallery("Original", sub):
        return d3_sets_view(f"Nothing to save: `{sub}/` holds no renders yet.")
    dst = P(D3_SETS, slug)
    if os.path.exists(dst):
        return d3_sets_view(f"A 3D dataset called “{slug}” exists already. Choose another name, or delete that one first.")
    shutil.copytree(P(sub), os.path.join(dst, "data"))
    used = an.used_settings(P(sub)) if os.path.exists(P(sub, "params.csv")) else {}
    info = {"name": str(name).strip(), "source": source, "sub": sub, "saved": time.strftime("%Y-%m-%d %H:%M"),
            "status": a3.status(PROJECT_ROOT, sub),
            "mooney_settings": (f"blur {used.get('Blur σ (px)', '–')} · white {used.get('White target', '–')} · "
                                f"{used.get('Background', '–')}") if used else "no Mooneys"}
    json.dump(info, open(os.path.join(dst, "set.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return d3_sets_view(f"Saved “{info['name']}” to `{D3_SETS}/{slug}/`: {d3_line(sub)}.")


def d3_set_load(folder):
    """Put a saved 3D dataset back as the live folder of its source (what was there is replaced)."""
    if busy_now():
        return (gr.update(),) * 4 + d3_sets_view(busy_text())
    j = next((x for x in d3_set_list() if x["folder"] == folder), None)
    if not j:
        return (gr.update(),) * 4 + d3_sets_view("Choose a saved 3D dataset first.")
    sub = j.get("sub")
    source = next((label for label, s in D3_SOURCES.items() if s == sub), None)
    src = P(D3_SETS, folder, "data")
    if source is None or not os.path.isdir(src):
        return (gr.update(),) * 4 + d3_sets_view("This saved 3D dataset cannot be read.")
    before = d3_line(sub)
    shutil.rmtree(P(sub), ignore_errors=True)
    shutil.copytree(src, P(sub))
    msg = (f"Loaded “{j.get('name', folder)}” as `{sub}/` ({source}). It replaced: {before}. "
           "The saved copy stays as it is.")
    return (gr.update(value=source),) + d3_data_view(source) + d3_sets_view(msg)


def d3_set_delete(folder):
    if not folder or not os.path.isdir(P(D3_SETS, folder)):
        return d3_sets_view("Choose a saved 3D dataset first.")
    shutil.rmtree(P(D3_SETS, folder), ignore_errors=True)
    return d3_sets_view(f"Deleted the saved 3D dataset “{folder}”. The live folders were not touched.")


def d3_truth_view(source, person=None):
    """Analysis → 3D comparison, upper part: state, table by light, chart, face chooser and its sheet."""
    sub = d3_sub(source)
    head, rows = a3.light_table(PROJECT_ROOT, sub)
    people = a3.sources(PROJECT_ROOT, sub)
    person = person if person in people else (people[0] if people else None)
    line = f"`{sub}/`: {d3_line(sub)}. " + a3.caveat(sub)
    if not rows:
        line += " **No scores yet:** run systems on this 3D set in **Experiments → Vision systems**."
    return (line, gr.update(value=rows or None, headers=head), a3.fig_truth(PROJECT_ROOT, sub) if rows else None,
            gr.update(choices=people, value=person), a3.person_sheet(PROJECT_ROOT, person, sub=sub) if person else None,
            gr.update(value=f"Export to {sub}/figures/"))


# ============================================================================ References
def references_md():
    """docs/REFERENCES.md with the photo credits of this project put under its Photos section."""
    text = read_doc("REFERENCES.md", "_docs/REFERENCES.md is missing._")
    clean = lambda v: re.sub(r"<[^>]+>", "", str(v or "")).replace("|", "/").replace("\n", " ").strip()
    rows = []
    for r in read_csv("dataset.csv"):
        link = f"[source page]({r['commons_url']})" if r.get("commons_url") else "–"
        rows.append((clean(r.get("identity")), clean(r.get("file")), clean(r.get("author")) or "–", clean(r.get("license")) or "–", link))
    for r in read_csv("credits.csv"):
        src = clean(r.get("source"))
        link = f"[source]({src})" if src.startswith("http") else (src or "–")
        rows.append(("", clean(r.get("file")), clean(r.get("author")) or "–", clean(r.get("licence") or r.get("license")) or "–", link))
    if rows:
        table = [f"**{len(rows)} photos in this project.**", "", "| Person | File | Author | Licence | Source |", "|---|---|---|---|---|"]
        table += [f"| {a} | `{b}` | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    else:
        table = ["_This project holds no photos yet._"]
    mark = "## Use of AI"
    block = "\n".join(table) + "\n\n"
    return text.replace(mark, block + mark) if mark in text else text + "\n\n" + block


# ============================================================================ documents shown in the app
def read_doc(name, missing):
    """A document of the app (docs/ in the code folder), or any Markdown file given by its full path."""
    p = name if os.path.isabs(name) else os.path.join(DOCS_ROOT, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else missing


def doc_segments(name):
    """Split a Markdown file into text and image segments, so its figures can be shown as images.
    Image paths are read relative to the folder of the file."""
    p = name if os.path.isabs(name) else os.path.join(DOCS_ROOT, name)
    base = os.path.dirname(p)
    segs, buf = [], []
    for line in read_doc(p, f"_{os.path.basename(p)} is missing._").splitlines():
        s = line.strip()
        if s.startswith("![") and "](" in s and s.endswith(")"):
            if buf:
                segs.append(("text", "\n".join(buf))); buf = []
            segs.append(("image", (s[2:s.index("](")], os.path.join(base, s[s.index("](") + 2:-1]))))
        else:
            buf.append(line)
    if buf:
        segs.append(("text", "\n".join(buf)))
    return segs


def note_files():
    """Markdown files in <working folder>/notes, the user's own notes. Empty if there is no such folder."""
    if not os.path.isdir(NOTES_ROOT):
        return []
    return sorted(f for f in os.listdir(NOTES_ROOT) if f.lower().endswith(".md") and not f.startswith("."))


def note_title(f):
    """The first heading of a note, or its file name."""
    try:
        with open(os.path.join(NOTES_ROOT, f), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return os.path.splitext(f)[0]


# ============================================================================ look: a white page, set like a paper
# Light only. The app has one look; a dark system setting or ?__theme=dark does not change it (see HEAD and the loop below).
ACCENT = "#1c3d6e"
SERIF = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md, spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#ffffff", body_text_color="#15181d", body_text_color_subdued="#5c6672",
    background_fill_primary="#ffffff", background_fill_secondary="#f7f8f9",
    border_color_primary="#dfe3e8", border_color_accent=ACCENT, color_accent=ACCENT, color_accent_soft="#e8eef7",
    link_text_color=ACCENT, link_text_color_hover="#27518f", link_text_color_visited=ACCENT, link_text_color_active=ACCENT,
    block_background_fill="#ffffff", block_border_width="1px", block_border_color="#dfe3e8", block_shadow="none",
    block_label_background_fill="#ffffff", block_label_border_width="0px", block_label_text_color="#5c6672", block_label_text_size="12px",
    block_title_text_color="#5c6672", block_title_text_size="12px", block_title_text_weight="500", block_info_text_color="#5c6672",
    panel_background_fill="#ffffff", panel_border_color="#dfe3e8",
    input_background_fill="#ffffff", input_background_fill_focus="#ffffff", input_border_color="#cfd5dc", input_border_color_focus=ACCENT,
    input_shadow="none", input_shadow_focus="0 0 0 3px rgba(28,61,110,.12)", input_placeholder_color="#8d96a1",
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover="#27518f", button_primary_text_color="#ffffff",
    button_primary_border_color=ACCENT, button_primary_shadow="none", button_primary_shadow_hover="none",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_hover="#f3f5f8", button_secondary_border_color="#cfd5dc",
    button_secondary_text_color="#15181d", button_secondary_shadow="none", button_secondary_shadow_hover="none",
    button_cancel_background_fill="#ffffff", button_cancel_background_fill_hover="#fbecea", button_cancel_text_color="#a5352c",
    button_cancel_border_color="#e9c2bd", button_cancel_shadow="none",
    button_large_radius="6px", button_small_radius="6px", button_medium_radius="6px",
    slider_color=ACCENT, checkbox_background_color_selected=ACCENT, checkbox_border_color_selected=ACCENT,
    checkbox_border_color="#b8c0ca", checkbox_label_background_fill="#ffffff", checkbox_label_background_fill_hover="#f3f5f8",
    checkbox_label_background_fill_selected="#eef2f8", checkbox_label_border_color="#dfe3e8", checkbox_label_shadow="none",
    table_even_background_fill="#ffffff", table_odd_background_fill="#ffffff", table_border_color="#e6e9ed", table_row_focus="#eef2f8",
    table_text_color="#15181d", code_background_fill="#f3f5f7",
)
for _k in [k for k in dir(THEME) if k.endswith("_dark") and not k.startswith("_")]:   # the dark variant is the light one
    if hasattr(THEME, _k[:-5]):
        setattr(THEME, _k, getattr(THEME, _k[:-5]))
# Loaded before the page draws: the serif for headings and documents, and a guard that keeps the page light whatever
# the system or the address says.
HEAD = """
<meta name="color-scheme" content="light only">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<script>
(function () {
  try {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') && u.searchParams.get('__theme') !== 'light') {
      u.searchParams.set('__theme', 'light'); window.history.replaceState(null, '', u.toString());
    }
  } catch (e) {}
  function light() {
    document.documentElement.classList.remove('dark');
    if (document.body) { document.body.classList.remove('dark'); }
  }
  light();
  document.addEventListener('DOMContentLoaded', function () {
    light();
    new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
  });
  new MutationObserver(light).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
})();
</script>
"""
CSS = """
:root, .gradio-container { color-scheme: light;
  --ink:#15181d; --ink-2:#2c323a; --muted:#5c6672; --muted-2:#8d96a1; --line:#dfe3e8; --line-2:#eceef1; --card:#ffffff; --paper-2:#f7f8f9;
  --accent:#1c3d6e; --accent-2:#27518f; --accent-soft:#9db3d1; --accent-wash:rgba(28,61,110,.07);
  --ok:#1a7350; --ok-bg:#e8f4ee; --ok-line:#bcdccb; --run-bg:#e8eef7; --run-line:#bccde4; --wait-bg:#eef0f3;
  --bad:#a5352c; --bad-bg:#fbecea; --bad-line:#e9c2bd; --part:#865d00; --part-bg:#fbf3dc; --part-line:#e6d29a;
  --log-bg:#f6f7f8; --log-ink:#2a3038;
  --serif: 'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace; }
body, .gradio-container { background: #ffffff !important; }
.gradio-container { max-width: 1400px !important; padding: 22px 34px 60px !important; }
footer { display: none !important; }

/* title block */
#head { display: flex; align-items: center; gap: 14px; padding: 2px 0 6px; }
#head .mark { width: 34px; height: 34px; border-radius: 50%; flex: none; border: 1.5px solid var(--ink);
  background: linear-gradient(115deg, var(--ink) 0 52%, var(--card) 52% 100%); }
#head h1 { font-family: var(--serif); font-size: 27px; font-weight: 600; letter-spacing: -0.01em; margin: 0; color: var(--ink); line-height: 1.1; }
#head .sub { font-family: var(--serif); font-style: italic; font-size: 14.5px; color: var(--muted); margin-top: 3px; }
.chips { margin: 4px 0 2px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
.chip { display: inline-block; border: 1px solid var(--line); background: var(--card); color: var(--muted);
  padding: 2px 11px; margin: 0 7px 6px 0; border-radius: 12px; font-size: 12.5px; white-space: nowrap; }
.chip b { color: var(--ink); font-weight: 600; margin-left: 4px; font-variant-numeric: tabular-nums; }
.chip.ok { border-color: var(--ok-line); background: var(--ok-bg); } .chip.ok b { color: var(--ok); }
.chip.part { border-color: var(--part-line); background: var(--part-bg); } .chip.part b { color: var(--part); }

/* the menu on the left and the pages */
#shell { align-items: flex-start; gap: 34px; flex-wrap: nowrap; margin-top: 10px; }
#navcol { flex: 0 0 164px !important; min-width: 164px !important; position: sticky; top: 14px; align-self: flex-start; gap: 2px !important; }
#navcol button { justify-content: flex-start; text-align: left; border: none !important; border-left: 2px solid transparent !important;
  border-radius: 0 6px 6px 0 !important; background: transparent !important; box-shadow: none !important; font-family: var(--serif);
  font-weight: 400; font-size: 16.5px; padding: 8px 14px; color: var(--muted); min-width: 0; }
#navcol button:hover { color: var(--ink); background: var(--paper-2) !important; }
#navcol button.primary { border-left-color: var(--accent) !important; background: var(--accent-wash) !important; color: var(--accent); font-weight: 600; }
#main { min-width: 0; flex: 1 1 0 !important; }
@media (max-width: 900px) {
  .gradio-container { padding: 14px 14px 40px !important; }
  #shell { flex-wrap: wrap; gap: 8px; }
  #navcol { flex: 1 1 100% !important; flex-direction: row !important; flex-wrap: wrap; position: static; border-bottom: 1px solid var(--line); padding: 0 0 6px; }
  #navcol button { flex: 0 0 auto; width: auto; border-left: none !important; border-bottom: 2px solid transparent !important; border-radius: 6px 6px 0 0 !important; }
  #navcol button.primary { border-bottom-color: var(--accent) !important; }
}
.tab-wrapper, .tabs > .tab-nav { margin-bottom: 12px; }
.tabs > .tab-nav button, .tab-wrapper button { font-size: 14px; color: var(--muted); padding: 8px 14px !important; }
.tabs > .tab-nav button.selected, .tab-wrapper button.selected { color: var(--accent) !important; font-weight: 600; }
.tab-wrapper button.selected::after { background: var(--accent) !important; }

/* text */
.hint, .hint p { font-size: 13px; color: var(--muted); line-height: 1.5; max-width: 96ch; }
.mono, .mono * { font-family: var(--mono); }
.lead p { font-size: 15px; line-height: 1.55; color: var(--ink-2); max-width: 82ch; margin-bottom: 4px; }
.block.prose { max-width: 78ch; }
.block.prose p, .block.prose li { font-family: var(--serif); font-size: 16px; line-height: 1.62; color: var(--ink-2); }
.block.prose h1, .block.prose h2, .block.prose h3, .block.prose h4 { font-family: var(--serif); color: var(--ink); letter-spacing: -0.005em; }
.block.prose h1 { font-size: 28px; font-weight: 600; margin: 4px 0 10px; }
.block.prose h2 { font-size: 21px; font-weight: 600; margin-top: 30px; padding-bottom: 5px; border-bottom: 1px solid var(--line); }
.block.prose h3 { font-size: 17px; font-weight: 600; margin-top: 22px; }
.block.prose a { color: var(--accent); text-decoration: underline; text-decoration-color: var(--accent-soft); text-underline-offset: 2px; }
.block.prose code, .lead code, .hint code { font-family: var(--mono); font-size: .86em; background: var(--paper-2); border: 1px solid var(--line-2);
  border-radius: 4px; padding: 0 4px; }
.block.prose.wide, .block.prose.ov, .block.prose.folders { max-width: 112ch; }
h2.zone { font-family: var(--serif); font-size: 21px; font-weight: 600; color: var(--ink); margin: 26px 0 4px; padding-bottom: 6px;
  border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 10px; }
h2.zone span { font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: 12px; font-weight: 600; color: var(--accent); background: var(--run-bg);
  width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; }
.caption, .caption p { font-family: var(--serif); font-style: italic; font-size: 14px; color: var(--muted); max-width: 86ch; margin-top: -2px; }

/* tables in the manner of a paper: a rule above and below, light lines between rows, no boxes */
.block.prose table { border-collapse: collapse; border: none !important; border-top: 1.5px solid var(--ink) !important;
  border-bottom: 1.5px solid var(--ink) !important; margin: 10px 0 16px;
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block.prose table td, .block.prose table th { border: none !important; border-bottom: 1px solid var(--line-2) !important; padding: 6px 14px 6px 0 !important;
  vertical-align: top; font-size: 13.5px; line-height: 1.45; font-variant-numeric: tabular-nums; font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.block.prose table th { border-bottom: 1px solid var(--ink) !important; color: var(--ink); font-weight: 600; text-align: left; background: none; white-space: nowrap; }
.block.prose table tr:last-child td { border-bottom: none !important; }
.block.prose table td, .block.prose table th { word-break: normal !important; overflow-wrap: normal; }
.block.prose table td code { white-space: nowrap; }
.block.prose table tr, .block.prose table thead, .block.prose table tbody { border-left: none !important; border-right: none !important; }
.block.ov table td:first-child, .ov table td:nth-child(2), .folders table td:first-child { white-space: nowrap; }
.folders table td .mono { word-break: break-all; font-size: 12px; }
.gradio-container .table-wrap { border: none !important; border-top: 1.5px solid var(--ink) !important; border-bottom: 1.5px solid var(--ink) !important;
  border-radius: 0 !important; }
.gradio-container .table-wrap, .gradio-container .table-wrap * { font-family: 'IBM Plex Sans', system-ui, sans-serif !important; }
.gradio-container .table-wrap td, .gradio-container .table-wrap th { font-size: 13px !important; }
.gradio-container .table-wrap thead th { border-bottom: 1px solid var(--ink) !important; background: #ffffff !important; font-weight: 600; }
.gradio-container .table-wrap td, .gradio-container .table-wrap th { border-left: none !important; border-right: none !important;
  font-variant-numeric: tabular-nums; }
.gradio-container .table-wrap tbody td { border-bottom: 1px solid var(--line-2) !important; }

/* controls */
.block label > span, .block .label-wrap span { font-size: 12.5px; color: var(--muted); }
button.lg, button.md, button.sm { font-weight: 600; }
button.secondary { border: 1px solid #cfd5dc !important; }
#navcol button.secondary { border: none !important; border-left: 2px solid transparent !important; }
button.stop { border: 1px solid var(--bad-line) !important; color: var(--bad) !important; background: #ffffff !important; }
button.stop:hover { background: var(--bad-bg) !important; }
.log textarea { font-family: var(--mono) !important; font-size: 12.5px !important; line-height: 1.55 !important; white-space: pre !important;
  overflow-x: auto !important; background: var(--log-bg) !important; color: var(--log-ink) !important; }
.system { border-top: 1px solid var(--line); padding-top: 10px; margin-top: 4px; }
.system h4, .system strong { color: var(--ink); }
.pos { text-align: center; font-size: 12.5px; color: var(--muted); padding-top: 6px; font-variant-numeric: tabular-nums; }
.gallery-item, .thumbnail-item { border-radius: 4px !important; }
.stage-panel { border: 1px solid var(--accent-soft) !important; border-radius: 8px !important; background: #ffffff !important; margin-top: 2px; }
.stage-panel > .styler { background: #ffffff !important; }
.stage-panel .hint { padding: 2px 14px 10px; }
.gradio-container .html-container { padding: 0 !important; }
#r-line p { font-family: 'IBM Plex Sans', system-ui, sans-serif; font-size: 14px; color: var(--ink-2); }
""" + rb.CSS


# ============================================================================ interface
init_project()
try:
    cf.reconcile(PROJECT_ROOT, NAMES_FILE)                      # repair dataset.csv after an interrupted collection
except Exception as _e:                                         # noqa
    print("reconcile:", _e)
_st, _vs = load_settings(), vs.status()
NAV = ["Home", "Dataset", "Mooney", "Experiments", "Analysis", "Docs"]

with gr.Blocks(title="Mooney Studio", analytics_enabled=False) as demo:
    gr.HTML("<div id='head'><div class='mark'></div><div><h1>Mooney Studio</h1><div class='sub'>Two-tone face images, "
            "and what vision systems make of them</div></div></div>")
    status = gr.HTML(status_html())

    with gr.Row(elem_id="shell"):
        # the menu on the left: one button per section; the selected one is drawn as 'primary'
        with gr.Column(elem_id="navcol", scale=0, min_width=158):
            nav_btns = [gr.Button(n, variant="primary" if i == 0 else "secondary", size="md") for i, n in enumerate(NAV)]
        with gr.Column(elem_id="main", scale=1):

            # ================================================================== Home
            with gr.Column(visible=True) as sec_home:
                overview = gr.Markdown(overview_md(), elem_classes=["prose", "ov"])
                ov_refresh = gr.Button("Refresh", size="sm", scale=0)
                gr.Markdown("### Where your data is kept", elem_classes="prose")
                ov_folders = gr.Markdown(folders_md(), elem_classes=["prose", "folders"])
                with gr.Row():
                    ov_work = gr.Textbox(WORK_ROOT, label="Working folder", scale=4, max_lines=1,
                                         info="Any folder outside the code folder. It is created if it does not exist; ~ is your home folder.")
                    ov_work_btn = gr.Button("Use this folder", size="md", scale=1, min_width=150)
                ov_work_msg = gr.Markdown("The change takes effect after a restart of the app. To use another folder for one start "
                                          "only: `./run.sh --work FOLDER`.", elem_classes="hint")

            # ================================================================== Dataset
            with gr.Column(visible=False) as sec_dataset:
                pg_dataset = gr.State("Photos")
                with gr.Tabs():
                    # -------------------------------------------------------------- Photos
                    with gr.Tab("Photos") as tab_photos:
                        gr.Markdown("Collect openly licensed portraits from Wikimedia Commons, look through them, and remove the ones you do not want. "
                                    "Each photo is cropped from hair to collar and saved at 512 × 512 px in `Original/`; the uncropped photo goes to `Raw/` "
                                    "and its licence, author and source page to `dataset.csv`.", elem_classes="lead")
                        with gr.Row():
                            names_box = gr.Textbox(names_default(), label="People, one per line (as on Wikimedia Commons)", lines=9, scale=2)
                            with gr.Column(scale=3):
                                with gr.Row():
                                    collect_btn = gr.Button("Collect", variant="primary", size="md", scale=4)
                                    ds_stop = gr.Button("Stop", variant="stop", size="md", scale=1, min_width=80)
                                ds_log = gr.Textbox(label="Log   + added   = already complete   - skipped   ! Commons error", lines=8, max_lines=8,
                                                    interactive=False, autoscroll=True, elem_classes="log")
                        with gr.Accordion("Collection settings (the defaults are recommended)", open=False):
                            with gr.Row():
                                per_id = gr.Slider(1, 4, cfg.COLLECT_PER_PERSON, step=1, label="Photos per person")
                                n_cand = gr.Slider(20, 160, cfg.COLLECT_CANDIDATES, step=10, label="Candidates looked at per person")
                                min_face = gr.Slider(0.02, 0.3, cfg.COLLECT_MIN_FACE, step=0.01, label="Smallest face (share of the photo)")
                            gr.Markdown("A photo is kept only if all of these hold: open licence (CC0, CC BY, CC BY-SA, public domain) · the file title "
                                        "contains the person's first name and surname · it is a photograph, not a painting or statue · exactly one face · "
                                        "the crop lies inside the photo and is at least 480 px wide · the face is sharp and neither too dark nor too bright · "
                                        "the photos of one person come from different occasions · no photo is used for two people. "
                                        "People who already have all their photos are skipped.", elem_classes="hint")
                        ds_gallery = gr.Gallery(gallery_of("Original"), label="Original/   click a photo to select it", columns=10, height=420, object_fit="cover")
                        with gr.Row():
                            ds_sel = gr.Textbox("", label="Selected photo", interactive=False, scale=2)
                            ds_del = gr.Button("Remove selected photo", variant="stop", scale=1)
                        ds_msg = gr.Markdown("Removing moves the photo to `_removed/` (it is not deleted), deletes its Mooney and result files, and stops "
                                             "that photo from being collected again. Press Collect afterwards to fetch a replacement.", elem_classes="hint")
                        with gr.Accordion("dataset.csv", open=False):
                            ds_table = gr.Dataframe(dataset_table(), headers=["identity", "file", "licence", "author", "date", "yaw", "sharpness", "source page"],
                                                    wrap=True, interactive=False)

                    # -------------------------------------------------------------- Subsets
                    with gr.Tab("Subsets") as tab_subset:
                        gr.Markdown("Choose the pairs that go into a subset. Click a Mooney on the left to add it, click it again or click it on the "
                                    "right to take it out, then **Save**. A subset has a name and a size; you can keep several.", elem_classes="lead")
                        _sub0 = load_subsets()
                        _sub_name = _sub0["active"] or (sorted(_sub0["sets"])[0] if _sub0["sets"] else "main")
                        with gr.Row():
                            s_pick = gr.Dropdown(choices=sorted(_sub0["sets"]), value=_sub_name if _sub_name in _sub0["sets"] else None, scale=2,
                                                 label="Open a saved subset")
                            s_name = gr.Textbox(_sub_name, label="Name (a new name saves a new subset)", max_lines=1, scale=2)
                            s_size = gr.Number(_sub0["sets"].get(_sub_name, {}).get("size", TARGET_SUBSET), label="Size", precision=0, minimum=1,
                                               maximum=500, scale=1, min_width=90)
                            s_show = gr.Radio(SUBSET_SHOW, value=SUBSET_SHOW[0], label="Show", scale=2)
                        s_line = gr.Markdown(elem_classes="prose")
                        s_chosen = gr.State([])
                        with gr.Row():
                            with gr.Column(scale=3):
                                s_cand = gr.Gallery(label="Click to add or remove   ✓ = in the subset", columns=6, height=560, object_fit="contain",
                                                    allow_preview=False)
                            with gr.Column(scale=2):
                                s_in = gr.Gallery(label="In this subset: Mooneys (click to remove)", columns=4, height=270, object_fit="contain",
                                                  allow_preview=False)
                                s_orig = gr.Gallery(label="In this subset: originals", columns=4, height=270, object_fit="contain", allow_preview=False)
                        with gr.Row():
                            s_save = gr.Button("Save subset", variant="primary", scale=2)
                            s_active = gr.Checkbox(True, label="Make it the project's subset (written to Subset/ as well)", scale=2)
                            s_fill = gr.Button("Fill up from favourites", scale=1)
                            s_del = gr.Button("Delete this subset", scale=1)
                        s_msg = gr.Markdown(elem_classes="hint")
                        with gr.Accordion("Create random subsets", open=False):
                            with gr.Row():
                                sx_pool = gr.Radio(SUBSET_SHOW, value=SUBSET_SHOW[0], label="Draw from", scale=2)
                                sx_size = gr.Number(_sub0["sets"].get(_sub_name, {}).get("size", TARGET_SUBSET), label="Size of each subset",
                                                    precision=0, minimum=1, maximum=500, scale=1, min_width=110)
                                sx_count = gr.Number(1, label="How many subsets", precision=0, minimum=1, maximum=100, scale=1, min_width=110)
                                sx_stem = gr.Textbox("random", label="Name stem (random 1, random 2, …)", max_lines=1, scale=2)
                                sx_seed = gr.Number(0, label="Seed", precision=0, scale=1, min_width=90)
                            with gr.Row():
                                sx_person = gr.Checkbox(True, label="One photo per person (no person twice inside a subset)", scale=2)
                                sx_disjoint = gr.Checkbox(True, label="Subsets do not share photos", scale=2)
                                sx_go = gr.Button("Create", variant="primary", scale=1)
                            gr.Markdown("A photo never appears twice in one subset. The same seed and the same pool give the same subsets, so a "
                                        "draw can be repeated; pool, seed and both options are stored with each subset in `subsets.json`. "
                                        "The project's subset is not changed.", elem_classes="hint")
                        gr.Markdown("Each saved subset is a folder `Subsets/<name>/` with `Original/` and `Mooney/`. The project's subset is "
                                    "also in `Subset/`: the counter at the top shows it. Subsets can be chosen wherever a step runs on part "
                                    "of the photos (system 2, 3D datasets).", elem_classes="hint")

                    # -------------------------------------------------------------- 3D datasets
                    with gr.Tab("3D datasets") as tab_3d:
                        gr.Markdown("Make a dataset of 3D faces: pictures of faces drawn under light directions we choose, and their Mooneys. "
                                    "For these pictures the light and the depth of every pixel are known, so the depth systems can be checked "
                                    "against the truth. This page only makes the dataset; the systems run on it in **Experiments → Vision "
                                    "systems**, and the comparison with the truth is in **Analysis → 3D comparison**.", elem_classes="lead")
                        d_source = gr.Radio(list(D3_SOURCES), value=list(D3_SOURCES)[0], label="Source of the 3D faces",
                                            info="Reconstructed faces go to 3D/, fitted Basel faces to 3D_basel_fit/, invented faces to 3D_basel/. "
                                                 "Each source has one live folder; the sets are kept side by side.")
                        d_caveat = gr.Markdown(D3_LEAD[f3.SUB] + " " + a3.caveat(f3.SUB) + " Strongly turned heads are skipped, because this simple "
                                               "mesh breaks for them.", elem_classes="hint")
                        with gr.Column(visible=True) as d_box_photos:
                            with gr.Row():
                                d_set = gr.Dropdown(photo_set_choices(suggest=True), value=SET_SUGGEST, label="Take the photos from", scale=2)
                                d_suggest = gr.Button("Fill in again", size="sm", scale=0, min_width=120)
                            d_files = gr.Dropdown(choices=images_in("Original"), value=a3.default_photos(PROJECT_ROOT), multiselect=True,
                                                  label="Photos to turn into 3D faces (you can still add or remove single photos)")
                        with gr.Column(visible=False) as d_box_basel:
                            d_model_note = gr.Markdown(basel_note(), elem_classes="hint")
                            d_fit_pose = gr.Radio(FIT_POSES, value=FIT_POSES[0], label="Head pose of the fitted head", visible=False,
                                                  info="Frontal: comparable with the invented faces. As in the photo: the pose found by the fit.")
                            with gr.Row():
                                d_n = gr.Slider(1, 50, value=cfg.BASEL_FACES, step=1, label="Number of invented faces",
                                                info="Seeds 1, 2, 3, …: the same number always gives the same face.")
                                d_model = gr.Radio(list(BASEL_MODELS), value="Whole head", label="Model file",
                                                   info="Whole head: with ears, neck and skull. Face only: the face region.")
                            with gr.Row():
                                d_poses = gr.CheckboxGroup(list(bfm.POSES), value=["frontal"], label="Head pose",
                                                           info="Left and right as seen by the viewer. Every chosen pose is drawn for every face.")
                                d_shadows = gr.Checkbox(value=cfg.BASEL_SHADOWS, label="Cast shadows",
                                                        info="Parts of the face shade other parts, for example the nose the cheek.")
                        d_lights = gr.CheckboxGroup(list(f3.LIGHTS), value=[l for l in cfg.LIGHTS_3D if l in f3.LIGHTS], label="Light directions")
                        with gr.Row():
                            d_b1 = gr.Button("Render", size="md")
                            d_b2 = gr.Button("Make Mooneys", size="md")
                            d_all = gr.Button("Render and make Mooneys", variant="primary", size="md")
                            d_stop = gr.Button("Stop", variant="stop", size="md", scale=0, min_width=80)
                        gr.Markdown("The buttons work on the source chosen above. *Render* replaces the earlier renders of that source together "
                                    "with their Mooneys and results, so save the dataset below first if you want to keep it. *Make Mooneys* uses "
                                    "the values saved as defaults in Mooney → Batch and removes earlier system results of this 3D set.",
                                    elem_classes="hint")
                        d_state = gr.Markdown(elem_classes="hint")
                        d_log = gr.Textbox(label="Log", lines=9, max_lines=9, interactive=False, autoscroll=True, elem_classes="log")
                        d_gal_o = gr.Gallery(label="3D/Original/   the renders", columns=10, height=250, object_fit="contain", allow_preview=False)
                        d_gal_m = gr.Gallery(label="3D/Mooney/   their Mooneys", columns=10, height=250, object_fit="contain", allow_preview=False)
                        gr.Markdown("### Saved 3D datasets\n<span class='hint'>A saved 3D dataset is a named copy of the live folder of a source: "
                                    "renders, true depth, Mooneys and any system results it has. *Load* puts a saved one back as the live folder of "
                                    "its source, so the systems and the analysis work on it.</span>", elem_classes="prose")
                        with gr.Row():
                            d_set_name = gr.Textbox("", label="Save the 3D dataset of the source chosen above as…", max_lines=1, scale=3,
                                                    placeholder="for example: basel 50 faces")
                            d_set_save = gr.Button("Save 3D dataset", scale=1)
                        d_set_table = gr.Dataframe(headers=D3_SET_COLS, interactive=False, wrap=True)
                        with gr.Row():
                            d_set_pick = gr.Dropdown(choices=d3_set_choices(), label="A saved 3D dataset", scale=3)
                            d_set_load = gr.Button("Load", scale=1)
                            d_set_del = gr.Button("Delete", variant="stop", scale=1)
                        d_set_msg = gr.Markdown(elem_classes="hint")

            # ================================================================== Mooney
            with gr.Column(visible=False) as sec_mooney:
                pg_mooney = gr.State("Batch")
                with gr.Tabs():
                    # -------------------------------------------------------------- Batch
                    with gr.Tab("Batch") as tab_batch:
                        gr.Markdown("Make a Mooney for every photo with one set of values. The values below are the project defaults; "
                                    "afterwards you can adjust single photos in **By hand**, and Batch will leave those alone.", elem_classes="lead")
                        with gr.Row():
                            b_sigma = gr.Slider(0, 20, value=_st["sigma"], step=0.5, label="Blur σ (pixels, at 512 px)")
                            b_white = gr.Slider(1, 99, value=_st["white"], step=1, label="White (% of the face)")
                            b_bgm = gr.Radio(["black", "white", "auto"], value=_st["bg_mode"], label="Background of the Mooney")
                            b_bgl = gr.Slider(0, 255, value=_st["bg_level"], step=1, label="Phase-1 gray (changes only the outline of the person)")
                        with gr.Accordion("Optional settings (off or at their default unless you change them)", open=False):
                            with gr.Row():
                                b_rule = gr.Radio(list(RULES), value=RULE_LABEL[_st["white_rule"]], label="White share")
                                b_bw = gr.Slider(1, 99, value=_st["bright_white"], step=1, label="White for bright faces (%)")
                                b_bb = gr.Slider(60, 200, value=_st["bright_border"], step=1, label="A face is bright from brightness")
                            gr.Markdown("*By face brightness* gives bright, evenly lit faces more white, because at a small white share they tend "
                                        "to break into fragments. It depends partly on skin tone, so say so if you use it.", elem_classes="hint")
                            with gr.Row():
                                b_pre = gr.Radio(["none", "clahe"], value=_st["preprocess"], label="Preprocessing before the blur")
                                b_measure = gr.Radio(["Face", "Whole person"], value="Face", label="White share is counted on")
                                b_sep = gr.Checkbox(_st["separate"], label="Separate the background (U²-Net)")
                                b_keep = gr.Checkbox(True, label="Keep Mooneys adjusted by hand")
                                b_src = gr.Radio(["Original", "Raw"], value="Original", label="Source folder")
                            with gr.Row():
                                b_sweep = gr.Checkbox(False, label="Sweep: try several values per photo and keep the best-scoring one")
                                b_sig = gr.CheckboxGroup(["3", "4", "6", "8"], value=["4", "6"], label="Blur values to try")
                                b_wht = gr.CheckboxGroup(["20", "22", "25", "28", "30", "35"], value=["20", "25"], label="White shares to try (%)")
                        with gr.Row():
                            b_run = gr.Button("Generate all Mooneys", variant="primary", size="md", scale=2)
                            b_stop = gr.Button("Stop", variant="stop", size="md", scale=0, min_width=80)
                            b_save_def = gr.Button("Save these values as defaults", size="md", scale=1)
                        b_log = gr.Textbox(label="Log", lines=9, max_lines=9, interactive=False, autoscroll=True, elem_classes="log")
                        b_favs = gr.Textbox(fav_text(), label="Favourites (click a Mooney below to add or remove it; choose a subset from them in Dataset → Subsets)",
                                            interactive=False)
                        b_gal = gr.Gallery(gallery_of("Mooney"), label="Mooney/", columns=10, height=420, object_fit="contain", allow_preview=False)
                        with gr.Accordion("params.csv (every value used, per photo)", open=False):
                            b_tab = gr.Dataframe(params_table(), headers=PARAM_COLS, interactive=False, wrap=True)

                    # -------------------------------------------------------------- By hand
                    with gr.Tab("By hand") as tab_lab:
                        gr.Markdown("Look at one photo at a time and see the four phases. Move the sliders to adjust it by hand, then **Save this Mooney**. "
                                    "A change affects this photo only, and only when you save it; another photo always opens with its own saved "
                                    "values. Mark the best ones with the star; choose a subset from them in **Dataset → Subsets**.", elem_classes="lead")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=330):
                                lab_src = gr.Radio(list(LAB_SOURCES), value="Original", label="Photos from")
                                lab_file = gr.Dropdown(choices=[], label="Photo", interactive=True)
                                with gr.Row():
                                    lab_prev = gr.Button("← Previous", size="sm", min_width=80)
                                    lab_pos = gr.HTML("", min_width=60)
                                    lab_next = gr.Button("Next →", size="sm", min_width=80)
                                inp = gr.Image(type="pil", label="Photo", height=230)
                                sigma = gr.Slider(0, 20, value=_st["sigma"], step=0.1, label="Blur σ (pixels)")
                                white = gr.Slider(1, 99, value=_st["white"], step=0.5, label="White (% of the face)")
                                bg_mode = gr.Radio(["black", "white", "auto"], value=_st["bg_mode"], label="Background of the Mooney")
                                with gr.Accordion("More settings", open=False):
                                    bg_level = gr.Slider(0, 255, value=_st["bg_level"], step=1, label="Phase-1 gray")
                                    separate = gr.Checkbox(_st["separate"], label="Separate the background (U²-Net)")
                                    preprocess = gr.Radio(["none", "clahe"], value=_st["preprocess"], label="Preprocessing before the blur")
                                    manual = gr.Checkbox(False, label="Set the threshold by hand (ignores the white share)")
                                    thr = gr.Slider(0, 255, value=128, step=1, label="Threshold", visible=False)
                                with gr.Row():
                                    lab_save = gr.Button("Save this Mooney", variant="primary", size="sm", min_width=120)
                                    lab_fav = gr.Button("☆ Favourite", size="sm", min_width=90)
                                    lab_default = gr.Button("Back to defaults", size="sm", min_width=110)
                                info = gr.Markdown(elem_classes="hint")
                                lab_msg = gr.Markdown(elem_classes="hint")
                            with gr.Column(scale=3):
                                with gr.Row():
                                    out_sep = gr.Image(label="Phase 1   background separated")
                                    out_gray = gr.Image(label="Phase 2   grayscale")
                                with gr.Row():
                                    out_blur = gr.Image(label="Phase 3   Gaussian blur")
                                    out_moon = gr.Image(label="Phase 4   Mooney", format="png")
                                with gr.Accordion("Favourites", open=False):
                                    fav_gal = gr.Gallery(fav_gallery(), columns=10, height=150, object_fit="contain", allow_preview=False)

            # ================================================================== Experiments
            with gr.Column(visible=False) as sec_exp:
                pg_exp = gr.State("Vision systems")
                with gr.Tabs():
                    # -------------------------------------------------------------- Vision systems
                    with gr.Tab("Vision systems") as tab_vision:
                        gr.Markdown("Run vision systems on a dataset. Each system takes every original (or render) and its Mooney and returns values "
                                    "for both: one result file per image in `ResultN/Original` and `ResultN/Mooney`, and a table per system in "
                                    "`analysis/`. A 3D dataset appears here once it has Mooneys.", elem_classes="lead")
                        with gr.Row():
                            v_data = gr.Radio(v_datasets(), value=V_PHOTOS, label="Run on", scale=2)
                            v_ticks = gr.CheckboxGroup(V_SYSTEMS, value=V_3D_DEFAULT, label="Systems to run", scale=3)
                        gr.Markdown("**1** YuNet detector and SFace embedding: is a face found, and how similar is the Mooney to its original? · "
                                    "**2** a local vision-language model through Ollama: face present, head pose, light direction, description · "
                                    "**3** Depth Anything V2: a learned depth map · **4** shape from shading (Tsai & Shah 1994): depth from a "
                                    "formula · **5** MediaPipe face mesh. On a 3D dataset the depth maps of 3 and 4 are compared with the true "
                                    "depth, and the answers of system 2 with the light and pose we set (it runs on part of the faces; see its setting). Every value is explained in **Docs → Vision systems**.", elem_classes="hint")
                        with gr.Row():
                            v_go = gr.Button("Run the ticked systems", variant="primary", scale=6)
                            v_stop = gr.Button("Stop", variant="stop", scale=1, min_width=80)
                        v_log = gr.Textbox(label="Log", lines=11, max_lines=11, interactive=False, autoscroll=True, elem_classes="log")
                        gr.Markdown("### Results of this dataset", elem_classes="prose")
                        v_line = gr.Markdown(elem_classes="hint")
                        v_table = gr.Dataframe(headers=SUMMARY_COLS, label="Summary   counts are images, other values are means", wrap=True, interactive=False)
                        v_fig = gr.Image(show_label=False, height=380)
                        gr.Markdown("### Save this state as a run\n<span class='hint'>A run is a frozen copy of the Mooneys, their values and the "
                                    "results under a name, in the runs folder. Saved runs are listed in **Runs → History** and compared in "
                                    "**Analysis → Compare runs**.</span>", elem_classes="prose")
                        with gr.Row():
                            v_run_name = gr.Textbox("", label="Save run as…", placeholder="for example: final run", scale=2, max_lines=1)
                            v_run_note = gr.Textbox("", label="Note (optional, stored with the run)", scale=3, max_lines=1)
                            v_run_save = gr.Button("Save run", scale=1)
                        v_keep = gr.CheckboxGroup([v[0] for v in an.RUN_GROUPS.values()], value=[an.RUN_GROUPS[k][0] for k in an.RUN_DEFAULT],
                                                  label="The saved run includes, besides the Mooneys and their values")
                        v_run_msg = gr.Markdown(elem_classes="hint")
                        with gr.Accordion("Settings of the systems", open=False):
                            v_check = gr.Button("Check which systems are ready", size="sm", scale=0)
                            with gr.Row():
                                v_s1 = gr.Textbox(_vs["face"], label="1 Face model", interactive=False)
                                v_s2 = gr.Textbox(_vs["vlm"], label="2 Vision-language model", interactive=False)
                                v_s3 = gr.Textbox(_vs["depth"], label="3 Learned depth", interactive=False)
                                v_s4 = gr.Textbox(_vs["shading"], label="4 Depth from shading", interactive=False)
                                v_s5 = gr.Textbox(_vs["mesh"], label="5 3D face mesh", interactive=False)
                            gr.Markdown("**System 2 needs Ollama**, a separate program that serves the vision-language model on this "
                                        "computer.", elem_classes="hint")
                            o_line = gr.Markdown(ollama_status(), elem_classes="hint")
                            with gr.Row():
                                o_check = gr.Button("Check", size="sm", scale=0, min_width=90)
                                o_start = gr.Button("Start Ollama", size="sm", scale=0, min_width=130)
                                o_name = gr.Textbox(cfg.OLLAMA_MODEL, label="Model to download", max_lines=1, scale=2)
                                o_pull = gr.Button("Download this model", size="sm", scale=0, min_width=170)
                            gr.Markdown("*Download* shows its progress in the log above and can be stopped with the Stop button. "
                                        "A stopped download continues where it was when you start it again.", elem_classes="hint")
                            with gr.Row():
                                v_model = gr.Dropdown(choices=_vs["vlm_models"], value=pick_model(_vs["vlm_models"]),
                                                      label="System 2: Ollama model", allow_custom_value=True, scale=1)
                                v_prompt = gr.Textbox(vs.VLM_PROMPT, label="System 2: prompt (saved with the results)", lines=3, scale=3)
                            with gr.Row():
                                v_scope = gr.Dropdown(photo_set_choices(), value=vlm_scope_label(), label="System 2 runs on", scale=1)
                                gr.Markdown("System 2 is heavy: about 5 GB of memory and several seconds per image, so all photos keep the laptop busy "
                                            "for a long time. Choose a subset or your favourites. Systems 1, 3 and 4 are light and always run on every "
                                            "photo, so their tables stay complete.", elem_classes="hint", scale=3)
                            v_identify = gr.Checkbox(False, label="Identity matching (a later step): also test whether systems 1 and 2 recognise the "
                                                                  "person (system 1 ranks against the other photos; system 2 is asked for the name)")
                            v_src = gr.Radio(["Original", "Raw"], value="Original", label="Source folder of the photos")

                    # -------------------------------------------------------------- Runs
                    with gr.Tab("Runs") as tab_runs:
                        gr.Markdown("Build a run from stages, put runs into groups, and let a group work through its runs on its own. "
                                    "A run takes one source (the photos or a 3D dataset) through the stages you switch on; every stage "
                                    "does exactly what its own page does. " + QUEUE_NOTE, elem_classes="lead")

                        gr.HTML("<h2 class='zone'><span>1</span> New run</h2>")
                        with gr.Row():
                            r_name = gr.Textbox("", label="Run name", placeholder="for example: photos, final values", scale=3, max_lines=1)
                            r_group = gr.Dropdown(choices=r_groups(), value=r_groups()[0], label="Group (pick one or type a new name)",
                                                  allow_custom_value=True, scale=2)
                            r_note = gr.Textbox("", label="Note (optional, stored with the run)", scale=4, max_lines=1,
                                                placeholder="for example: blur 6.5, 20 % white, 71 adjusted by hand")
                        r_on = gr.State(dict(R_ON))
                        r_sel = gr.State("source")
                        r_edit = gr.State("")
                        _r0 = (R_SOURCES[R_PHOTOS], R_NO_SET, vlm_scope_label(), _st["sigma"], _st["white"],
                               _st["bg_mode"] if _st["bg_mode"] in ("black", "white") else "black", _st["bg_level"],
                               [R_SYS[n] for n in (1, 3, 4)], [an.RUN_GROUPS[k][0] for k in an.RUN_DEFAULT], dict(R_ON))
                        r_cards_html = gr.HTML(r_cards("source", *_r0), js_on_load=rb.JS, elem_id="r-cards")
                        with gr.Group(visible=True, elem_classes="stage-panel") as r_p_source:
                            with gr.Row():
                                r_source = gr.Radio(r_sources(), value=R_SOURCES[R_PHOTOS], label="Source of this run", scale=3,
                                                    info="A 3D source is offered once it has renders (Dataset → 3D datasets).")
                                r_saved = gr.Dropdown([R_NO_SET] + [f for _l, f in d3_set_choices()], value=R_NO_SET, scale=2,
                                                      label="Or a saved 3D dataset (loaded as the live one first)")
                                r_vlm_set = gr.Dropdown(photo_set_choices(), value=vlm_scope_label(), label="System 2 runs on", scale=2,
                                                        info="For a 3D source: the faces made from these photos. Invented Basel faces: the first 10.")
                        with gr.Group(visible=False, elem_classes="stage-panel") as r_p_mooneys:
                            with gr.Row():
                                r_sigma = gr.Slider(0, 20, value=_st["sigma"], step=0.1, label="Blur σ (px)")
                                r_white = gr.Slider(5, 60, value=_st["white"], step=0.5, label="White share (%)")
                                r_bg = gr.Radio(["black", "white"], value=_r0[5], label="Background of the Mooney")
                                r_bgl = gr.Slider(0, 255, value=_st["bg_level"], step=1, label="Phase-1 gray")
                            gr.Markdown("Switched off, the run uses the Mooneys as they are. Switched on, it makes them again with these values "
                                        "and saves the values as the project's defaults; for photos, Mooneys adjusted by hand are kept. "
                                        "Other options (white rule, preprocessing) come from **Mooney → Batch**.", elem_classes="hint")
                        with gr.Group(visible=False, elem_classes="stage-panel") as r_p_systems:
                            r_systems = gr.CheckboxGroup([R_SYS[n] for n in (1, 2, 3, 4)], value=_r0[7],
                                                         label="Vision systems of this run, on every original (or render) and its Mooney")
                            gr.Markdown("For a 3D source the run then scores systems 3 and 4 against the true depth. System 2 runs only on "
                                        "photos, on the set chosen in the Source stage, and is skipped if Ollama is not running.",
                                        elem_classes="hint")
                        with gr.Group(visible=False, elem_classes="stage-panel") as r_p_export:
                            gr.Markdown("Writes the summary table and figure to `figures/` (for photos also one panel per favourite; for a 3D "
                                        "source the table by light direction), and the comparison of the stimulus sets. Nothing to set.",
                                        elem_classes="hint")
                        with gr.Group(visible=False, elem_classes="stage-panel") as r_p_save:
                            r_keep = gr.CheckboxGroup([v[0] for v in an.RUN_GROUPS.values()], value=_r0[8],
                                                      label="The saved copy includes, besides the Mooneys and their values")
                        with gr.Row():
                            r_add_btn = gr.Button("Add to group", variant="primary", scale=2)
                            r_now_btn = gr.Button("Run now", scale=1)
                            r_dup_btn = gr.Button("Duplicate last run", scale=1)
                            r_cancel_btn = gr.Button("Cancel edit", scale=1)
                        r_edit_md = gr.Markdown(elem_classes="hint")
                        r_msg = gr.Markdown(elem_classes="hint")

                        gr.HTML("<h2 class='zone'><span>2</span> Groups and queue</h2>")
                        with gr.Row():
                            r_line = gr.Markdown(elem_classes="prose", elem_id="r-line")
                        with gr.Row():
                            r_all = gr.Button("Run all groups", variant="primary", scale=2)
                            r_stop_after_btn = gr.Button("Stop after the current run", scale=2)
                            r_stop_now_btn = gr.Button("Stop now", variant="stop", scale=1)
                            r_clear_btn = gr.Button("Clear finished", scale=1)
                        r_board_html = gr.HTML(r_board(), js_on_load=rb.JS, elem_id="r-board")
                        r_open = gr.State([])
                        r_sig = gr.State(0)
                        r_want = gr.State("")
                        gr.Markdown("The board is kept in the project (`queue.json`), so it is still there after a restart; a run that was "
                                    "working when the app was closed is marked *stopped*. The app must stay open and the computer awake "
                                    "while runs work. This page follows them by itself.", elem_classes="hint")

                        gr.HTML("<h2 class='zone'><span>3</span> History</h2>")
                        h_html = gr.HTML(r_history(), js_on_load=rb.JS, elem_id="r-history")
                        h_picked = gr.State([])
                        h_armed = gr.State(None)
                        h_opened = gr.State(None)
                        h_sig = gr.State(0)
                        h_msg = gr.Markdown(elem_classes="hint")
                        with gr.Accordion("Publish results as tables (for the repository's data/ folder)", open=False):
                            gr.Markdown("*publish* on a saved run, or the button here for the project as it is now, writes a copy of the "
                                        "numbers to `data/<name>/` next to the code: scores, settings and the per-photo tables of the "
                                        "systems. **No photo, Mooney, depth map, render or panel is copied, and nothing of the Basel Face "
                                        "Model.** The app does not commit or push; `data/README.md` lists what is there.",
                                        elem_classes="hint")
                            with gr.Row():
                                h_pub_name = gr.Textbox("", label="Name for the published project state", placeholder="for example: final state",
                                                        max_lines=1, scale=3)
                                h_pub_btn = gr.Button("Publish the current project state", scale=2)
                        sr_facts = gr.Markdown(elem_classes="prose")
                        sr_table = gr.Dataframe(headers=SUMMARY_COLS, label="Summary of the opened run", wrap=True, interactive=False, visible=False)
                        sr_fig = gr.Image(show_label=False, height=380, visible=False)
                        h_cmp_note = gr.Markdown(elem_classes="hint")
                        h_cmp = gr.Dataframe(interactive=False, wrap=True, label="Chosen runs side by side", visible=False)
                        h_cmp_settings = gr.Dataframe(interactive=False, wrap=True, label="What each run was made with", visible=False)
                        r_timer = gr.Timer(3, active=False)

            # ================================================================== Analysis
            with gr.Column(visible=False) as sec_analysis:
                pg_analysis = gr.State("Results")
                with gr.Tabs():
                    # -------------------------------------------------------------- Results
                    with gr.Tab("Results") as tab_results:
                        gr.Markdown("What the vision systems returned, for the originals and for the Mooneys. Nothing is computed here; "
                                    "press **Refresh** after running a system.", elem_classes="lead")
                        with gr.Row():
                            r_refresh = gr.Button("Refresh", variant="primary", size="sm", scale=0)
                            r_export = gr.Button("Export figures to figures/", size="sm", scale=0)
                            r_msg = gr.Markdown(elem_classes="hint")
                        r_table = gr.Dataframe(headers=SUMMARY_COLS, label="Summary   counts are images, other values are means", wrap=True, interactive=False)
                        r_fig = gr.Image(show_label=False, height=380)
                        with gr.Row():
                            r_stem = gr.Dropdown(choices=[], label="Photo", scale=3)
                            r_prev = gr.Button("← Previous", size="sm", scale=1)
                            r_next = gr.Button("Next →", size="sm", scale=1)
                        r_panel = gr.Image(label="Original and Mooney with each system's output", height=520)
                        r_detail = gr.Markdown(elem_classes="prose")
                        gr.Markdown("<span class='hint'>To keep this state under a name, save it as a run in **Experiments**.</span>",
                                    elem_classes="prose")

                    # -------------------------------------------------------------- By photo properties
                    with gr.Tab("By photo properties") as tab_analysis:
                        gr.Markdown("How the results relate to the photos. Each photo is one data point: its brightness, head turn and Mooney settings on one "
                                    "side, what each vision system returned on the other. Nothing is computed by a model here; the page only reads the "
                                    "result tables. Press **Refresh** after running a system.", elem_classes="lead")
                        with gr.Row():
                            a_refresh = gr.Button("Refresh", variant="primary", size="sm", scale=0)
                            a_export = gr.Button("Export analysis to figures/", size="sm", scale=0)
                            a_msg = gr.Markdown(elem_classes="hint")
                        gr.Markdown("### The dataset", elem_classes="prose")
                        with gr.Row():
                            a_ds_table = gr.Dataframe(headers=["", "count"], interactive=False, scale=1, label="What the dataset holds")
                            a_ds_fig = gr.Image(show_label=False, scale=3)
                        gr.Markdown("### Each system: original versus Mooney\n<span class='hint'>One bar chart or histogram per value. A histogram shows how many "
                                    "photos fall into each range; the dashed line is the mean.</span>", elem_classes="prose")
                        a_sys_fig = gr.Image(show_label=False)
                        gr.Markdown("### By group", elem_classes="prose")
                        a_by = gr.Radio(list(an.GROUPINGS), value="Face brightness", label="Group the photos by")
                        gr.Markdown("Face brightness: three equally large groups by the mean gray value of the face. Head turn: frontal, slightly turned, strongly "
                                    "turned, from the position of the nose between the eyes. Groups with fewer than 10 photos are marked, because their "
                                    "percentages are unreliable.", elem_classes="hint")
                        a_grp_table = gr.Dataframe(interactive=False, wrap=True, label="One row per group; every value with the number of photos behind it")
                        a_grp_fig = gr.Image(show_label=False)
                        gr.Markdown("### Do the systems agree?\n<span class='hint'>Correlation across photos between two results: +1 means the same photos score "
                                    "high on both, 0 means no relation, -1 means the opposite. Yes/no results count as 1 and 0.</span>", elem_classes="prose")
                        with gr.Row():
                            a_agree_table = gr.Dataframe(headers=["result A", "result B", "correlation", "photos"], interactive=False, scale=3,
                                                         label="Strongest relation first")
                            a_agree_fig = gr.Image(show_label=False, scale=2)
                        gr.Markdown("### Hardest and easiest photos\n<span class='hint'>Ranked by the average of the photo's rank on every available result "
                                    "(similarity, similarity after re-blurring, both depth matches, face found after re-blurring, language model says face). "
                                    "Hard and easy for the models, not for people.</span>", elem_classes="prose")
                        a_hard = gr.Gallery(label="Hardest for the systems", columns=8, height=190, object_fit="contain")
                        a_easy = gr.Gallery(label="Easiest for the systems", columns=8, height=190, object_fit="contain")
                        a_rank = gr.Dataframe(headers=an.RANK_COLS, interactive=False, label="The values behind the two rows of pictures")
                        gr.Markdown("### In words", elem_classes="prose")
                        a_words = gr.Markdown(elem_classes="prose")

                    # -------------------------------------------------------------- Compare runs
                    with gr.Tab("Compare runs") as tab_compare:
                        gr.Markdown("Put saved runs side by side. *current* is the project as it is now. The first table is the results summary "
                                    "with one column per run; the second shows what each run's Mooneys were made with. Runs are saved in "
                                    "**Experiments**.", elem_classes="lead")
                        c_runs = gr.Dropdown(choices=run_choices(), value=[], multiselect=True, label="Runs to put side by side")
                        c_msg = gr.Markdown(elem_classes="hint")
                        c_table = gr.Dataframe(interactive=False, wrap=True, label="Results, one column per run")
                        c_settings = gr.Dataframe(interactive=False, wrap=True, label="What each run's Mooneys were made with (from its params.csv)")

                    # -------------------------------------------------------------- 3D comparison
                    with gr.Tab("3D comparison") as tab_truth:
                        gr.Markdown("The systems against the truth. For a 3D dataset the depth of every pixel is known, so a system's depth map "
                                    "can be compared with the depth the picture was drawn from, for the render and for its Mooney. Make a 3D "
                                    "dataset in **Dataset → 3D datasets** and run the systems on it in **Experiments → Vision systems**.",
                                    elem_classes="lead")
                        t_source = gr.Radio(list(D3_SOURCES), value=list(D3_SOURCES)[0], label="3D dataset of the source")
                        d_truth_line = gr.Markdown(elem_classes="hint")
                        gr.Markdown("### By light direction\n<span class='hint'>n = renders in the row. *Face found* counts images. Similarity is the "
                                    "face model's (−1 to 1). *Depth vs truth* is the correlation, inside the face, between a system's depth map and "
                                    "the depth the render was drawn from: 1 = the same relief, 0 = unrelated.</span>", elem_classes="prose")
                        d_table = gr.Dataframe(headers=a3.HEADERS, interactive=False, wrap=True, label="One row per light direction")
                        d_fig = gr.Image(show_label=False)
                        gr.Markdown("### One face under every light", elem_classes="prose")
                        d_person = gr.Dropdown(choices=[], label="Face", scale=0)
                        d_sheet = gr.Image(show_label=False)
                        with gr.Row():
                            d_refresh = gr.Button("Refresh", size="sm", scale=0)
                            d_export = gr.Button("Export to 3D/figures/", size="sm", scale=0)
                            d_msg = gr.Markdown(elem_classes="hint")
                        gr.Markdown("### Compare stimulus sets\n<span class='hint'>The photos of the main project, the faces reconstructed from "
                                    "photos, the Basel model fitted to photos and the invented Basel faces, each with what the same systems returned. "
                                    + a3.COMPARE_NOTE + "</span>", elem_classes="prose")
                        x_table = gr.Dataframe(headers=["Measure"] + [a3.SET_NAMES[k] for k in a3.SETS], interactive=False, wrap=True, max_height=760,
                                               column_widths=["40%"] + ["15%"] * len(a3.SETS), label="One row per measure, one column per set")
                        x_fig = gr.Image(show_label=False)
                        x_lights = gr.Dataframe(headers=a3.LIGHT_HEADERS, interactive=False, wrap=True, max_height=760,
                                                label="The 3D sets by light direction")
                        with gr.Row():
                            x_refresh = gr.Button("Refresh comparison", size="sm", scale=0)
                            x_export = gr.Button("Export comparison to figures/", size="sm", scale=0)
                            x_msg = gr.Markdown(elem_classes="hint")

            # ================================================================== Docs
            with gr.Column(visible=False) as sec_docs:
                with gr.Tabs():
                    with gr.Tab("Manual"):
                        gr.Markdown(read_doc("MANUAL.md", "_docs/MANUAL.md is missing._"), elem_classes="prose")
                    with gr.Tab("Vision systems"):
                        gr.Markdown(read_doc("VISION_SYSTEMS.md", "_docs/VISION_SYSTEMS.md is missing._"), elem_classes="prose")
                    with gr.Tab("References") as tab_refs:
                        refs_md = gr.Markdown(references_md(), elem_classes=["prose", "refs"])
                    # Notes: only if the working folder has any
                    _notes = note_files()
                    if _notes:
                        with gr.Tab("Notes"):
                            gr.Markdown(f"Your own notes: the Markdown files in <span class='mono'>{NOTES_ROOT}</span>, with their pictures. "
                                        "They are read when the app starts and are not part of the app.", elem_classes="lead")
                            note_pick = gr.Radio([(note_title(f), f) for f in _notes], value=_notes[0], label="Note")
                            note_boxes = []
                            for _i, _f in enumerate(_notes):
                                with gr.Column(visible=_i == 0) as _box:
                                    for kind, body in doc_segments(os.path.join(NOTES_ROOT, _f)):
                                        if kind == "text":
                                            gr.Markdown(body, elem_classes="prose")
                                        elif os.path.exists(body[1]):
                                            gr.Image(body[1], label=body[0], interactive=False)
                                note_boxes.append(_box)
                            note_pick.change(lambda f: [gr.update(visible=f == g) for g in _notes], note_pick, note_boxes)

    # ====================================================================== wiring
    sections = [sec_home, sec_dataset, sec_mooney, sec_exp, sec_analysis, sec_docs]
    refresh_overview = lambda: (overview_md(), status_html())

    def go(k):
        """Show section k and mark its button."""
        return ([gr.update(visible=i == k) for i in range(len(NAV))]
                + [gr.update(variant="primary" if i == k else "secondary") for i in range(len(NAV))])

    # one chain of events per menu button: show the section, then refresh the page that is open in it
    nav_ev = [b.click(lambda k=k: go(k), None, sections + nav_btns, show_progress="hidden") for k, b in enumerate(nav_btns)]

    def pages(k, state, items):
        """Wire the pages of section k. items: (tab, name, refresher or None, inputs, outputs).
        A page is refreshed when its tab is chosen, and when the menu opens the section while that page is the open one."""
        for tab, name, fn, inputs, outputs in items:
            tab.select(lambda name=name: name, None, state, show_progress="hidden")
            if fn is None:
                continue
            tab.select(fn, inputs, outputs)

            def guarded(page, *args, fn=fn, name=name, n=len(outputs)):
                if page != name:
                    return gr.update() if n == 1 else tuple(gr.update() for _ in range(n))
                return fn(*args)
            nav_ev[k] = nav_ev[k].then(guarded, [state] + list(inputs or []), outputs, show_progress="hidden")

    def refresh_page():
        """Runs on every page load, so a reload always shows what is on disk right now."""
        return (status_html(), overview_md(), gallery_of("Original"), dataset_table(), gallery_of("Mooney"), params_table(),
                fav_text(), fav_gallery(), d3_suggest(), gr.update(choices=run_choices()))

    demo.load(refresh_page, None, [status, overview, ds_gallery, ds_table, b_gal, b_tab, b_favs, fav_gal, d_files, c_runs])

    # Stop: the job that is working ends after the image or step it is on (see run_logged)
    for _b in (ds_stop, d_stop, b_stop, v_stop):
        _b.click(stop_job, None, None, show_progress="hidden")

    # ---------------------------------------------------------------------- Home
    nav_ev[0] = nav_ev[0].then(refresh_overview, None, [overview, status], show_progress="hidden")
    ov_refresh.click(refresh_overview, None, [overview, status])
    ov_work_btn.click(change_work_dir, ov_work, ov_work_msg)
    ov_work.submit(change_work_dir, ov_work, ov_work_msg)

    # ---------------------------------------------------------------------- Dataset → Photos
    collect_btn.click(collect, [names_box, per_id, n_cand, min_face], [ds_log, ds_gallery, ds_table, status])
    ds_gallery.select(pick_photo, None, [ds_sel, ds_msg])
    ds_del.click(remove_photo, [ds_sel, lab_src], [ds_sel, ds_msg, ds_gallery, ds_table, status, b_gal, b_tab, fav_gal, lab_file])

    # ---------------------------------------------------------------------- Dataset → Subsets
    s_shown = [s_line, s_cand, s_in, s_orig, s_msg, s_chosen]
    s_args = [s_name, s_size, s_show, s_chosen]
    subset_page = lambda n, z, sh, ch: subset_view(n, z, sh, ch) if ch else subset_open(n, sh)[1:]
    s_pick.input(lambda n, sh: (n,) + subset_open(n, sh), [s_pick, s_show], [s_name, s_size] + s_shown)
    s_name.submit(subset_view, s_args, s_shown)
    s_name.blur(subset_view, s_args, s_shown)
    s_show.change(subset_view, s_args, s_shown)
    s_size.change(subset_view, s_args, s_shown)
    s_cand.select(subset_toggle, s_args, s_shown)
    s_in.select(subset_remove, s_args, s_shown)
    s_fill.click(subset_fill, s_args, s_shown)
    s_save.click(subset_save, s_args + [s_active], s_shown + [s_pick, status])
    s_del.click(subset_delete, [s_name, s_show], [s_name, s_size] + s_shown + [s_pick, status])
    sx_go.click(subset_random, [sx_pool, sx_size, sx_count, sx_stem, sx_seed, sx_person, sx_disjoint, s_show],
                [s_name, s_size] + s_shown + [s_pick, status])
    s_show.change(lambda v: v, s_show, sx_pool, show_progress="hidden")
    s_size.change(lambda v: v, s_size, sx_size, show_progress="hidden")

    # ---------------------------------------------------------------------- Dataset → 3D datasets
    d_live = [d_state, d_gal_o, d_gal_m]
    d_saved = [d_set_table, d_set_pick, d_set_msg]

    def d3_page(source, model):
        return ((gr.update(choices=photo_set_choices(suggest=True)), basel_note(model)) + d3_data_view(source) + d3_sets_view())

    for button, steps in ((d_b1, ["render"]), (d_b2, ["Mooneys"]), (d_all, ["render", "Mooneys"])):
        button.click(d3_run(steps), [d_source, d_files, d_lights, d_n, d_model, d_poses, d_shadows, d_fit_pose], [d_log] + d_live)
    d_source.change(d3_switch, [d_source, d_model], [d_box_photos, d_box_basel, d_caveat, d_model_note, d_n, d_poses, d_fit_pose,
                                                     d_set, d_files] + d_live)
    d_model.change(basel_note, d_model, d_model_note)
    d_suggest.click(d3_suggest, d_set, d_files)
    d_set.change(d3_suggest, d_set, d_files)
    d_set_save.click(d3_set_save, [d_source, d_set_name], d_saved)
    d_set_load.click(d3_set_load, d_set_pick, [d_source] + d_live + d_saved)
    d_set_del.click(d3_set_delete, d_set_pick, d_saved)

    pages(1, pg_dataset, [
        (tab_photos, "Photos", lambda: (gallery_of("Original"), dataset_table(), status_html()), None, [ds_gallery, ds_table, status]),
        (tab_subset, "Subsets", subset_page, s_args, s_shown),
        (tab_3d, "3D datasets", d3_page, [d_source, d_model], [d_set, d_model_note] + d_live + d_saved[:2]),
    ])

    # ---------------------------------------------------------------------- Mooney → Batch
    b_values = [b_sigma, b_white, b_bgm, b_bgl, b_sep, b_rule, b_bw, b_bb, b_pre]
    b_run.click(run_batch, [b_src] + b_values + [b_measure, b_keep, b_sweep, b_sig, b_wht], [b_log, b_gal, b_tab, status])
    b_save_def.click(save_defaults, b_values, b_log)
    b_gal.select(star_in_gallery, None, [b_favs, fav_gal])

    # ---------------------------------------------------------------------- Mooney → By hand
    manual.change(lambda m: gr.update(visible=m), manual, thr)
    lab_controls = [inp, separate, bg_level, bg_mode, sigma, white, preprocess, manual, thr]
    lab_inputs = lab_controls + [lab_src, lab_file]
    for c in lab_controls:                               # one at a time and in order, so the picture and the saved state match
        c.change(run_lab, lab_inputs, [out_sep, out_gray, out_blur, out_moon, info], show_progress="minimal",
                 concurrency_limit=1, concurrency_id="lab")
    lab_src.change(lab_files, lab_src, lab_file)
    # another photo: first its own saved values into the controls, then the photo itself
    lab_file.change(lab_values, [lab_src, lab_file], [separate, bg_level, bg_mode, sigma, white, preprocess, manual, thr]
                    ).then(lab_load, [lab_src, lab_file], inp)
    lab_file.change(lab_position, [lab_src, lab_file], lab_pos)
    lab_file.change(lab_star_label, [lab_src, lab_file], lab_fav)
    lab_prev.click(lambda s, f: lab_step(s, f, -1), [lab_src, lab_file], lab_file)
    lab_next.click(lambda s, f: lab_step(s, f, +1), [lab_src, lab_file], lab_file)

    def lab_save_now():
        msg = save_current()
        note = lab_saved_note(_last["stem"], _last["params"]) if _last["params"] else ""
        return msg, _last["head"] + "  \n" + note, status_html(), gallery_of("Mooney"), params_table()
    lab_save.click(lab_save_now, None, [lab_msg, info, status, b_gal, b_tab], concurrency_limit=1, concurrency_id="lab")
    lab_file.change(lambda: "", None, lab_msg)
    lab_fav.click(star_current, None, [lab_msg, fav_gal, lab_fav, b_favs])
    lab_default.click(lab_reset, None, [separate, bg_level, bg_mode, sigma, white, preprocess, manual])
    demo.load(lab_files, lab_src, lab_file)

    pages(2, pg_mooney, [
        (tab_batch, "Batch", lambda: (gallery_of("Mooney"), params_table(), fav_text(), status_html()), None, [b_gal, b_tab, b_favs, status]),
        (tab_lab, "By hand", fav_gallery, None, [fav_gal]),
    ])

    # ---------------------------------------------------------------------- Experiments → Vision systems
    v_shown = [v_line, v_table, v_fig]
    v_check.click(vision_status, None, [v_s1, v_s2, v_s3, v_s4, v_s5, v_model])
    v_identify.change(lambda on: vs.VLM_PROMPT_IDENTITY if on else vs.VLM_PROMPT, v_identify, v_prompt)
    v_data.change(v_switch, [v_data, v_ticks], [v_ticks] + v_shown)
    v_go.click(vision_run, [v_data, v_ticks, v_src, v_identify, v_model, v_prompt, v_scope], [v_log, status] + v_shown)
    v_run_save.click(v_save_run, [v_run_name, v_keep, v_run_note], [v_run_msg, c_runs])
    o_check.click(ollama_refresh, None, [o_line, v_model, v_s2])
    o_start.click(ollama_start, None, [o_line, v_model, v_s2])
    o_pull.click(ollama_pull, o_name, [v_log, o_line, v_model, v_s2])

    # ---------------------------------------------------------------------- Experiments → Runs
    r_form = [r_name, r_group, r_note, r_source, r_saved, r_vlm_set, r_sigma, r_white, r_bg, r_bgl, r_systems, r_keep, r_on]
    r_card_in = [r_source, r_saved, r_vlm_set, r_sigma, r_white, r_bg, r_bgl, r_systems, r_keep]
    r_panel_boxes = [r_p_source, r_p_mooneys, r_p_systems, r_p_export, r_p_save]
    r_fill_out = [r_name, r_group, r_note, r_source, r_saved, r_vlm_set, r_sigma, r_white, r_bg, r_bgl, r_systems, r_keep, r_on, r_sel,
                  r_cards_html] + r_panel_boxes
    assert len(r_fill_out) == R_FORM_N
    r_shown = [r_board_html, r_line, r_sig]

    # the stage cards: a click selects a stage or flips its switch; any change of a field redraws the summaries
    r_cards_html.click(r_card_click, [r_sel, r_on] + r_card_in, [r_sel, r_on, r_cards_html] + r_panel_boxes, show_progress="hidden")
    for _c in r_card_in:
        _c.change(r_cards, [r_sel] + r_card_in + [r_on], r_cards_html, show_progress="hidden")
    r_source.change(r_source_changed, [r_source, r_saved, r_systems], [r_systems, r_vlm_set], show_progress="hidden")
    r_saved.change(r_source_changed, [r_source, r_saved, r_systems], [r_systems, r_vlm_set], show_progress="hidden")

    r_add_btn.click(r_add, [r_open, r_edit] + r_form, r_shown + [r_msg, r_edit, r_edit_md]).then(
        lambda: gr.update(choices=r_groups()), None, r_group, show_progress="hidden")
    r_now_btn.click(r_run_now, [r_open] + r_form, r_shown + [r_msg])
    r_dup_btn.click(r_duplicate, None, r_fill_out + [r_msg])
    r_cancel_btn.click(lambda: ("", ""), None, [r_edit, r_edit_md], show_progress="hidden")

    r_all.click(r_start, r_open, r_shown + [r_msg])
    r_stop_after_btn.click(r_stop_after, r_open, r_shown + [r_msg])
    r_stop_now_btn.click(r_stop_now, r_open, r_shown + [r_msg])
    r_clear_btn.click(r_clear, r_open, r_shown + [r_msg])
    h_show = [sr_facts, sr_table, sr_fig]
    r_board_html.click(r_board_click, [r_open, r_edit], r_shown + [r_msg, r_open, r_edit, r_edit_md, r_want] + r_fill_out,
                       show_progress="hidden").then(r_open_result, [r_want, h_picked, h_armed, h_opened],
                                                    [h_html, h_sig, h_opened] + h_show, show_progress="hidden")
    h_html.click(r_history_click, [h_picked, h_armed, h_opened],
                 [h_html, h_picked, h_armed, h_opened] + h_show + [h_cmp, h_cmp_settings, h_cmp_note, c_runs, h_msg, h_sig],
                 show_progress="hidden")
    h_pub_btn.click(r_publish_state, h_pub_name, h_msg)
    r_timer.tick(r_tick, [r_open, r_sig, h_sig, h_picked, h_armed, h_opened], [r_board_html, r_line, r_sig, h_html, h_sig, status],
                 show_progress="hidden")

    pages(3, pg_exp, [
        (tab_vision, "Vision systems", v_page, [v_data, v_ticks], [v_data, v_scope, v_ticks] + v_shown),
        (tab_runs, "Runs", r_page, [r_open, h_picked, h_armed, h_opened, r_edit],
         [r_board_html, r_line, r_sig, h_html, h_sig, r_source, r_saved, r_vlm_set, r_group, r_edit_md]),
    ])
    # the timer of the Runs page ticks only while that page is the one shown
    tab_runs.select(lambda: gr.Timer(active=True), None, r_timer, show_progress="hidden")
    tab_vision.select(lambda: gr.Timer(active=False), None, r_timer, show_progress="hidden")
    for _k in range(len(NAV)):
        if _k == 3:
            nav_ev[_k] = nav_ev[_k].then(lambda page: gr.Timer(active=page == "Runs"), pg_exp, r_timer, show_progress="hidden")
        else:
            nav_ev[_k] = nav_ev[_k].then(lambda: gr.Timer(active=False), None, r_timer, show_progress="hidden")

    # ---------------------------------------------------------------------- Analysis → Results
    r_refresh.click(results_load, None, [r_table, r_fig, r_stem, r_msg])
    r_stem.change(results_show, r_stem, [r_panel, r_detail])
    r_prev.click(lambda s: results_step(s, -1), r_stem, r_stem)
    r_next.click(lambda s: results_step(s, +1), r_stem, r_stem)
    r_export.click(results_export, None, r_msg)
    demo.load(results_load, None, [r_table, r_fig, r_stem, r_msg])

    # ---------------------------------------------------------------------- Analysis → By photo properties, Compare runs
    def analysis_groups(by):
        head, rows = an.group_table(PROJECT_ROOT, by)
        return gr.update(value=rows, headers=head), an.fig_groups(PROJECT_ROOT, by), an.statements(PROJECT_ROOT, by)

    def analysis_load(by):
        hard, easy, rank_rows = an.ranking(PROJECT_ROOT)
        has_results = an.fig_systems(PROJECT_ROOT) is not None
        msg = "" if has_results else "No results to analyse yet. Run the vision systems in Experiments → Vision systems, then press Refresh."
        return (an.dataset_table(PROJECT_ROOT), an.fig_dataset(PROJECT_ROOT), an.fig_systems(PROJECT_ROOT), *analysis_groups(by),
                an.agreement_table(PROJECT_ROOT), an.fig_agreement(PROJECT_ROOT), hard, easy, rank_rows, msg)

    def analysis_export(by):
        made = an.export(PROJECT_ROOT, by)
        return f"Wrote {len(made)} files to figures/: " + ", ".join(made) if made else "Nothing to export yet."

    a_outputs = [a_ds_table, a_ds_fig, a_sys_fig, a_grp_table, a_grp_fig, a_words, a_agree_table, a_agree_fig, a_hard, a_easy, a_rank, a_msg]
    a_refresh.click(analysis_load, a_by, a_outputs)
    a_by.change(analysis_groups, a_by, [a_grp_table, a_grp_fig, a_words])
    a_export.click(analysis_export, a_by, a_msg)
    c_runs.change(runs_compare, c_runs, [c_table, c_settings, c_msg])

    # ---------------------------------------------------------------------- Analysis → 3D comparison
    t_shown = [d_truth_line, d_table, d_fig, d_person, d_sheet, d_export]
    x_shown = [x_table, x_lights, x_fig, x_msg]
    truth_page = lambda source, person: d3_truth_view(source, person) + d3_compare()
    t_source.change(d3_truth_view, [t_source, d_person], t_shown)
    d_person.change(d3_sheet, [t_source, d_person], d_sheet)
    d_refresh.click(d3_truth_view, [t_source, d_person], t_shown)
    d_export.click(d3_export, t_source, d_msg)
    x_refresh.click(d3_compare, None, x_shown)
    x_export.click(d3_compare_export, None, x_msg)

    pages(4, pg_analysis, [
        (tab_results, "Results", results_load, None, [r_table, r_fig, r_stem, r_msg]),
        (tab_analysis, "By photo properties", analysis_load, [a_by], a_outputs),
        (tab_compare, "Compare runs", lambda: gr.update(choices=run_choices()), None, [c_runs]),
        (tab_truth, "3D comparison", truth_page, [t_source, d_person], t_shown + x_shown),
    ])

    # ---------------------------------------------------------------------- Docs
    tab_refs.select(references_md, None, refs_md)


def launch_options():
    """Address, port, public link and login, from config.py, config.json, the environment or the command line."""
    auth = (cfg.AUTH_USER, cfg.AUTH_PASSWORD) if cfg.AUTH_USER and cfg.AUTH_PASSWORD else None
    if cfg.HOST not in ("127.0.0.1", "localhost") and auth is None:
        print("Note: the app is reachable from other machines without a login. Set MOONEY_USER and MOONEY_PASSWORD to add one.")
    return {"server_name": cfg.HOST, "server_port": cfg.PORT or None, "share": bool(cfg.SHARE), "auth": auth,
            "allowed_paths": [PROJECT_ROOT, RUNS_ROOT, NOTES_ROOT, HERE]}


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD, **launch_options())
