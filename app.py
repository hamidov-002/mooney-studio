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
"""
import argparse
import csv
import json
import os
import sys
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
from PIL import Image

import analysis as an
import analysis3d as a3
import batch as bt
import bfm
import collect_faces as cf
import config as cfg
import face3d as f3
import results as rs
import vision as vs
from mooney import make_mooney, subject_mask, resize_max, face_brightness

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_ROOT = cfg.WORK_DIR                           # the working folder: config.py, config.json, MOONEY_WORK or --work
PROJECT_ROOT = cfg.PROJECT_DIR                     # <working folder>/project unless set otherwise
RUNS_ROOT = cfg.RUNS_DIR                           # <working folder>/runs unless set otherwise
NOTES_ROOT = cfg.NOTES_DIR                         # <working folder>/notes: the user's own notes, never part of the code
DOCS_ROOT = os.path.join(HERE, "docs")
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
        "vlm_target": len(subset_stems() & orig) if load_settings()["vlm_scope"] == "subset" else len(orig),
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
    chips.append(chip("Subset", f"{s['subset']} / {TARGET_SUBSET}", (s["subset"] >= TARGET_SUBSET) if s["subset"] else None))
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
               "1. Open **Dataset**, enter a few names and press *Collect*. "
               "Or copy your own square face photos into the `Original/` folder shown below and reload the page.\n"
               "2. Make the Mooneys in **Batch**.\n"
               "3. Run the vision systems in **Vision systems** and read the outcome in **Results**.")
    elif s["mooneys"] < n:
        nxt = f"Make the Mooneys in **Batch** ({n - s['mooneys']} of {n} photos have none yet)."
    elif any(r[k] < tgt[k] for k in tgt):
        k = next(k for k in (1, 2, 3, 4) if r[k] < tgt[k])
        nxt = f"Run system {k} in **Vision systems** ({r[k]} of {tgt[k]} photos done)."
    elif s["subset"] < TARGET_SUBSET:
        nxt = f"Mark your best {TARGET_SUBSET} Mooneys with the star in **Mooney**, then press *Save favourites to Subset/* in **Batch**."
    elif not s["figures"]:
        nxt = "Press *Export figures* in **Results**."
    else:
        nxt = "Every step is done. Save the run in **Results** to keep it, or go on with **3D faces**."
    if 0 < n < TARGET_PHOTOS:
        nxt += f"  \nThe dataset has {n} of the {TARGET_PHOTOS} photos set as target; collect more in **Dataset** if you want the full set."

    rule = ("the same for every photo" if st["white_rule"] == "fixed"
            else f"by face brightness ({st['bright_white']:g} % for faces brighter than {st['bright_border']:g})")
    lines = [
        "### Next step", nxt, "",
        "### Progress",
        "| Step | Status | Detail |", "|---|---|---|",
        f"| Dataset | {word(n >= TARGET_PHOTOS, n > 0)} | {n} photos of {s['people']} people; {s['pairs']} people have two or more photos |",
        f"| Batch | {word(n > 0 and s['mooneys'] == n, s['mooneys'] > 0)} | {s['mooneys']} Mooneys for {n} photos |",
        f"| Mooney (adjust by hand) | optional | {s['manual']} adjusted by hand, {s['favs']} favourites |",
        f"| Vision systems | {word(n > 0 and all(r[k] >= tgt[k] for k in tgt), max(r.values()) > 0)} | "
        f"face model {r[1]} / {n} · language model {r[2]} / {tgt[2]}{' (subset only)' if st['vlm_scope'] == 'subset' else ''} · "
        f"learned depth {r[3]} / {n} · depth from shading {r[4]} / {n} |",
        f"| Results | {word(s['figures'] > 0)} | {s['figures']} per-photo panels exported to figures/ |",
        f"| Subset of {TARGET_SUBSET} | {word(s['subset'] >= TARGET_SUBSET, s['subset'] > 0)} | {s['subset']} pairs in Subset/ |",
        f"| 3D faces from photos | {d3_word()} | {d3_line()} |",
        f"| 3D faces from the Basel Face Model | {d3_word(bfm.SUB)} | {d3_line(bfm.SUB)} |",
        "",
        "### Current generation settings",
        f"Background **{st['bg_mode']}** · phase-1 gray **{st['bg_level']}** · blur σ **{st['sigma']:g} px** · "
        f"white **{st['white']:g} %** of the face · white share **{rule}** · preprocessing **{st['preprocess']}**.  ",
        "<span class='hint'>Change them in Batch; they are stored in the project's settings.json.</span>",
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
            ("Saved runs", RUNS_ROOT, "frozen copies made in Results"),
            ("Notes", NOTES_ROOT, f"{notes} Markdown file{'s' if notes != 1 else ''}, shown in the Notes tab" if notes
             else "optional: Markdown files put here appear in a Notes tab after a restart"),
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
    return f"{len(favs)} of {TARGET_SUBSET}: " + ", ".join(favs) if favs else "none yet"


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
def run_logged(job, n_wait, final):
    """
    Run job(log) in a thread and stream its log to a textbox.
    Yields (log_text, n_wait x no-change) while it runs, then (log_text, *final()) once.
    Every log line gets the clock time in front; errors are shown in the log, not swallowed.
    """
    lines, err = [], {}

    def log(msg=""):
        msg = str(msg)
        lines.append(f"{time.strftime('%H:%M:%S')}  {msg}" if msg.strip() else "")

    def work():
        try:
            job(log)
        except Exception:
            err["tb"] = traceback.format_exc()

    t = threading.Thread(target=work, daemon=True); t.start()
    while t.is_alive():
        time.sleep(0.5)
        yield ("\n".join(lines[-400:]),) + (gr.update(),) * n_wait
    if err:
        log(""); log("STOPPED WITH AN ERROR"); lines.extend(err["tb"].strip().splitlines()[-12:])
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
                                 min_face_frac=float(min_face), log=log, progress=lambda p: progress(p))
    yield from run_logged(job, 3, lambda: (gallery_of("Original"), dataset_table(), status_html()))


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
        keep_manual=bool(keep_manual), log=log, progress=lambda p: progress(p))
    yield from run_logged(job, 3, lambda: (gallery_of("Mooney"), params_table(), status_html()))


def save_defaults(sigma, white, bg_mode, bg_level, separate, rule, bright_white, bright_border, preprocess):
    save_settings(sigma=float(sigma), white=float(white), bg_mode=bg_mode, bg_level=int(bg_level), separate=bool(separate),
                  white_rule=RULES[rule], bright_white=float(bright_white), bright_border=float(bright_border), preprocess=preprocess)
    return "Saved as defaults (settings.json). The Mooney tab starts from these values after a reload."


def star_in_gallery(evt: gr.SelectData):
    items = images_in("Mooney")
    if evt.index is not None and evt.index < len(items):
        toggle_favourite(os.path.splitext(items[evt.index])[0])
    return fav_text(), fav_gallery()


def save_subset(source):
    favs = load_favs()
    if not favs:
        return "No favourites yet: click Mooneys in the gallery, or use the star in Mooney.", status_html()
    n = bt.save_subset(PROJECT_ROOT, [f + ".png" for f in favs], source=source)
    note = "" if n == TARGET_SUBSET else f" The target is {TARGET_SUBSET}."
    return f"{n} favourite pairs copied to Subset/Original and Subset/Mooney.{note}", status_html()


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
    return (f"{'Added to' if on else 'Removed from'} favourites: {_last['stem']} ({len(favs)} of {TARGET_SUBSET}).",
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
def vision_job(fn, **kw):
    def handler(*args, progress=gr.Progress()):
        init_project()
        extra = kw["args"](*args) if "args" in kw else {}
        if extra is None:
            yield kw["refuse"], gr.update(); return
        job = lambda log: fn(project=PROJECT_ROOT, log=log, progress=lambda p: progress(p), **extra)
        yield from run_logged(job, 1, lambda: (status_html(),))
    return handler


VLM_SCOPES = {"The subset only": "subset", "All photos": "all"}
VLM_SCOPE_LABEL = {v: k for k, v in VLM_SCOPES.items()}


def vlm_args(source, identify, model, prompt, scope):
    """Arguments for system 2, or None (= refuse to start) when no model is chosen or the subset is empty."""
    if not model:
        return None
    save_settings(vlm_scope=VLM_SCOPES[scope])
    only = subset_stems() if VLM_SCOPES[scope] == "subset" else None
    if only is not None and not only:
        return None
    return {"source": source, "identify": bool(identify), "model": model, "prompt": prompt, "only": only}


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
    msg = "" if rows else "No results yet. Run a system in Vision systems, then press Refresh."
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
        return gr.update(value=None), gr.update(value=None), "Choose two or more runs. Save a run in Results first."
    note = "" if len(names or []) > 1 else "Choose at least one more run to compare."
    return gr.update(value=rows or None, headers=head), gr.update(value=srows, headers=shead), note


# ============================================================================ 3D faces
D3_SOURCES = {"Faces reconstructed from your photos": f3.SUB, "Invented faces from the Basel Face Model": bfm.SUB}
D3_LEAD = {
    f3.SUB: "For a few photos a 3D face mesh is estimated and drawn again under light directions we choose.",
    bfm.SUB: "The faces are drawn in the head poses and under the light directions chosen below.",
}
BASEL_MODELS = {"Whole head": cfg.BASEL_MODEL, "Face only": cfg.BASEL_MODEL_FACE}


def d3_sub(source):
    """Folder of the chosen source of 3D faces (3D or 3D_basel)."""
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


def d3_results(source, person=None):
    """Everything the 3D tab shows below the log: state line, table by light, chart, face chooser and sheet, galleries."""
    sub = d3_sub(source)
    head, rows = a3.light_table(PROJECT_ROOT, sub)
    people = a3.sources(PROJECT_ROOT, sub)
    person = person if person in people else (people[0] if people else None)
    return (d3_line(sub), gr.update(value=rows or None, headers=head), a3.fig_truth(PROJECT_ROOT, sub) if rows else None,
            gr.update(choices=people, value=person), a3.person_sheet(PROJECT_ROOT, person, sub=sub) if person else None,
            gr.update(value=d3_gallery("Original", sub), label=f"{sub}/Original/"),
            gr.update(value=d3_gallery("Mooney", sub), label=f"{sub}/Mooney/"))


def d3_switch(source, which):
    """The source of 3D faces was changed: show its controls, its explanation and its results."""
    sub = d3_sub(source)
    basel = sub == bfm.SUB
    return (gr.update(visible=not basel), gr.update(visible=basel), D3_LEAD[sub] + " " + a3.caveat(sub)
            + ("" if basel else " Strongly turned heads are skipped, because this simple mesh breaks for them."),
            basel_note(which), gr.update(value=f"Export to {sub}/figures/")) + d3_results(source)


def d3_suggest():
    return gr.update(choices=images_in("Original"), value=a3.default_photos(PROJECT_ROOT))


def d3_job(steps, sub, files, lights, basel, progress):
    """The chosen steps of the 3D chain as one job for run_logged. Each step checks that the one before it was done."""
    def job(log):
        root = P(sub)
        for k, step in enumerate(steps):
            if len(steps) > 1:
                if k:
                    log("")
                log(f"Step {k + 1} of {len(steps)}: {step}")
            prog = lambda p, k=k, step=step: progress((k + p) / len(steps), desc=step)
            if step == "render" and sub == bfm.SUB:
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
                    log("No renders yet - press 1 Render first."); return
                st = load_settings()
                a3.clear_results(PROJECT_ROOT, sub)                 # results of earlier Mooneys would no longer match
                log(f"Settings as saved in Batch: blur {st['sigma']:g} · white {st['white']:g} % · background {st['bg_mode']}.")
                bt.run_batch(root, source="Original", sigmas=(float(st["sigma"]),), whites=(float(st["white"]) / 100,),
                             bg_level=int(st["bg_level"]), separate=bool(st["separate"]), bg_mooney=st["bg_mode"],
                             white_rule=st["white_rule"], bright_white=float(st["bright_white"]) / 100,
                             bright_border=float(st["bright_border"]), preprocess=st["preprocess"], keep_manual=False,
                             log=log, progress=prog)
            else:
                if not d3_gallery("Mooney", sub):
                    log("No Mooneys of the renders yet - press 2 Make Mooneys first."); return
                vs.run_face(root, "Original", log=log, progress=lambda p: prog(0.2 * p))
                log(""); vs.run_depth(root, "Original", log=log, progress=lambda p: prog(0.2 + 0.4 * p))
                log(""); vs.run_shading(root, "Original", log=log, progress=lambda p: prog(0.6 + 0.3 * p))
                log(""); a3.score_truth(PROJECT_ROOT, log=log, progress=lambda p: prog(0.9 + 0.1 * p), sub=sub)
    return job


def d3_run(steps):
    def handler(source, files, lights, person, n_faces, model, poses, shadows, progress=gr.Progress()):
        init_project()
        basel = {"faces": n_faces or cfg.BASEL_FACES, "model": model, "poses": poses or [], "shadows": shadows}
        yield from run_logged(d3_job(steps, d3_sub(source), files, lights, basel, progress), 7, lambda: d3_results(source, person))
    return handler


def d3_export(source):
    sub = d3_sub(source)
    made = a3.export(PROJECT_ROOT, sub)
    return f"Wrote {len(made)} files to {sub}/figures/: " + ", ".join(made) if made else "Nothing to export yet."


def d3_compare():
    """The comparison of the stimulus sets: table by measure, table by light, chart, and a line if there is nothing yet."""
    head, rows = a3.compare_sets(PROJECT_ROOT)
    lhead, lrows = a3.compare_lights(PROJECT_ROOT)
    msg = "" if rows else "Nothing to compare yet. Run the vision systems on the photos and the three steps above for a 3D source."
    return (gr.update(value=rows or None, headers=head), gr.update(value=lrows or None, headers=lhead),
            a3.fig_compare(PROJECT_ROOT) if rows else None, msg)


def d3_compare_export():
    made = a3.export_compare(PROJECT_ROOT)
    return f"Wrote {len(made)} files to figures/: " + ", ".join(made) if made else "Nothing to export yet."


# ============================================================================ Run (the whole pipeline with one button)
PIPE_SYSTEMS = ["1 Face model", "2 Vision-language model (subset only, needs Ollama)", "3 Learned depth", "4 Depth from shading"]
RUNS_COLS = ["Name", "Saved", "Mooneys", "By hand", "Blur", "White", "Background", "Systems", "Also holds", "Note"]
RUNS_WIDTHS = ["14%", "12%", "9%", "7%", "6%", "7%", "10%", "8%", "14%", "13%"]
RUNS_SHORT = {"figures": "figures", "Subset": "subset", "3D": "3D from photos", "3D_basel": "3D Basel", "Original": "photos"}


def runs_table():
    """One row per saved run, newest first, read from each run's run.json."""
    rows = []
    for d in reversed(an.list_runs(RUNS_ROOT)):
        try:
            j = json.load(open(os.path.join(RUNS_ROOT, d, "run.json"), encoding="utf-8"))
        except Exception:                                   # a run saved by hand, or a damaged file
            j = {}
        u, c = j.get("used", {}), j.get("counts", {})
        systems = ", ".join(p[-1] for p in ("Result1", "Result2", "Result3", "Result4") if c.get(p)) or "–"
        extra = ", ".join(short for part, short in RUNS_SHORT.items() if os.path.isdir(os.path.join(RUNS_ROOT, d, part))
                          and any(not f.startswith(".") for _r, _d, fs in os.walk(os.path.join(RUNS_ROOT, d, part)) for f in fs))
        rows.append([j.get("name", d), j.get("saved", ""), u.get("Mooneys", c.get("Mooney", "")), u.get("Adjusted by hand", ""),
                     u.get("Blur σ (px)", ""), u.get("White target", ""), u.get("Background", ""), systems, extra or "–",
                     j.get("note", "")])
    return rows or None


def pipeline_view(msg=""):
    """What the Run tab shows below the log: message, summary table, figure, comparison of the stimulus sets, saved runs."""
    rows = rs.summary(PROJECT_ROOT)
    table = [[r.get(c, "") for c in SUMMARY_COLS] for r in rows] or None
    head, crows = a3.compare_sets(PROJECT_ROOT)
    return (msg, table, rs.figure(PROJECT_ROOT) if rows else None, gr.update(value=crows or None, headers=head),
            gr.update(value=runs_table(), headers=RUNS_COLS))


def pipeline(name, note, keep, do_batch, sigma, white, bg_mode, bg_level, systems, d3, n_faces, do_export, do_save,
             progress=gr.Progress()):
    """
    One full run: Mooneys, vision systems, 3D faces, figures, saved copy - each part only if it is ticked.
    Nothing here computes anything itself; every step calls the same function as its own tab.
    """
    init_project()
    systems, d3 = list(systems or []), list(d3 or [])
    refuse = None
    if do_save and not an.run_slug(name):
        refuse = "Not started: give the run a name first, or untick *Save the run*."
    elif not images_in("Original"):
        refuse = "Not started: there are no photos yet. Collect some in **Dataset** first."
    elif not do_batch and not images_in("Mooney"):
        refuse = "Not started: there are no Mooneys yet. Tick *Make the Mooneys*."
    elif not (do_batch or systems or d3 or do_export or do_save):
        refuse = "Not started: nothing is ticked."
    if refuse:
        yield (gr.update(), refuse) + (gr.update(),) * 6
        return

    steps, saved = [], {}
    if do_batch:
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
            model, only = pick_model(vs.status()["vlm_models"]), subset_stems()
            if not model:
                log("Skipped: Ollama is not running or has no vision model (start it with `ollama serve`)."); return
            if not only:
                log("Skipped: the subset is empty and no favourites are marked."); return
            vs.run_vlm(PROJECT_ROOT, model, source="Original", log=log, progress=prog, only=only)
        steps.append(("System 2 Vision-language model, on the subset", 3, step_vlm))
    for label in d3:
        def step_3d(log, prog, label=label):
            basel = {"faces": int(n_faces or cfg.BASEL_FACES), "model": "Whole head", "poses": ["frontal"], "shadows": cfg.BASEL_SHADOWS}
            d3_job(["render", "Mooneys", "systems"], D3_SOURCES[label], a3.default_photos(PROJECT_ROOT),
                   [l for l in cfg.LIGHTS_3D if l in f3.LIGHTS], basel, lambda v, desc=None: prog(v))(log)
        steps.append((f"3D faces: {label[0].lower() + label[1:]}", 4, step_3d))
    if do_export:
        def step_export(log, prog):
            favs = load_favs()
            made = rs.export(PROJECT_ROOT, favs or rs.stems(PROJECT_ROOT)[:10])
            log(f"figures/: {len(made)} files (summary table, figure, one panel each for "
                + ("your favourites" if favs else "the first 10 photos") + ").")
            for s in (f3.SUB, bfm.SUB):
                if d3_gallery("Mooney", s):
                    log(f"{s}/figures/: {len(a3.export(PROJECT_ROOT, s))} files.")
            made = a3.export_compare(PROJECT_ROOT)
            if made:
                log(f"figures/: comparison of the stimulus sets, {len(made)} files.")
        steps.append(("Export figures", 1, step_export))
    if do_save:
        def step_save(log, prog):
            saved["folder"], saved["msg"] = an.save_run(PROJECT_ROOT, RUNS_ROOT, name, include=[RUN_LABELS[x] for x in (keep or [])], note=note)
            log(saved["msg"])
        steps.append(("Save the run", 1, step_save))

    total = sum(w for _, w, _ in steps)

    def job(log):
        t0, done = time.time(), 0
        log(f"Run{' “' + str(name).strip() + '”' if str(name).strip() else ''}: {len(steps)} steps.")
        for k, (title, w, fn) in enumerate(steps):
            log(""); log(f"── Step {k + 1} of {len(steps)}: {title}")
            fn(log, lambda p, done=done, w=w, title=title: progress((done + w * min(max(float(p), 0), 1)) / total, desc=title))
            done += w
        log(""); log(f"Finished in {(time.time() - t0) / 60:.1f} min.")

    def final():
        msg = saved.get("msg") or ("Finished. Not saved as a run (the box was not ticked)." if not do_save else "Finished, but the run was not saved; see the log.")
        return pipeline_view(msg) + (status_html(), gr.update(choices=run_choices()))
    yield from run_logged(job, 7, final)


def pipeline_save_only(name, keep, note):
    folder, msg = an.save_run(PROJECT_ROOT, RUNS_ROOT, name, include=[RUN_LABELS[x] for x in (keep or [])], note=note)
    return msg, gr.update(value=runs_table(), headers=RUNS_COLS), gr.update(choices=run_choices())


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


# ============================================================================ look
ACCENT, ACCENT_DARK = "#1f4f82", "#7fb2e5"
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm, spacing_size=gr.themes.sizes.spacing_sm, text_size=gr.themes.sizes.text_sm,
).set(
    body_background_fill="#f4f6f8", body_background_fill_dark="#12151a",
    body_text_color="#1b1f24", body_text_color_dark="#e4e8ee",
    block_background_fill="#ffffff", block_background_fill_dark="#1a1e24",
    block_border_width="1px", block_border_color="#d5dbe3", block_border_color_dark="#2b323b",
    block_label_background_fill="#ffffff", block_label_background_fill_dark="#1a1e24",
    block_label_text_color="#55606d", block_label_text_color_dark="#9ba6b4", block_label_text_size="11px",
    block_title_text_color="#55606d", block_title_text_color_dark="#9ba6b4", block_title_text_size="11px",
    input_background_fill="#ffffff", input_background_fill_dark="#12151a",
    input_border_color="#c5ccd6", input_border_color_dark="#343c47",
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover="#2a639e", button_primary_text_color="#ffffff",
    button_primary_background_fill_dark="#2f6aa8", button_primary_background_fill_hover_dark="#3a7ac0", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#1a1e24",
    button_secondary_border_color="#aeb7c3", button_secondary_border_color_dark="#3a434f",
    button_secondary_text_color="#1b1f24", button_secondary_text_color_dark="#e4e8ee",
    slider_color=ACCENT, slider_color_dark=ACCENT_DARK,
    checkbox_background_color_selected=ACCENT, checkbox_border_color_selected=ACCENT,
    checkbox_background_color_selected_dark="#2f6aa8", checkbox_border_color_selected_dark="#2f6aa8",
)
CSS = """
:root, .gradio-container { --ink:#1b1f24; --muted:#5b6672; --line:#d5dbe3; --card:#ffffff; --accent:#1f4f82;
  --ok:#17764f; --ok-bg:#e7f4ee; --part:#8a5a00; --part-bg:#fbf2dd; }
.dark, .dark .gradio-container { --ink:#e4e8ee; --muted:#9ba6b4; --line:#2b323b; --card:#1a1e24; --accent:#7fb2e5;
  --ok:#62cfa2; --ok-bg:#15302a; --part:#e3b964; --part-bg:#33290f; }
.gradio-container { max-width: 1480px !important; }
footer { display: none !important; }
#head { display: flex; align-items: center; gap: 12px; padding: 4px 0 2px; }
#head .mark { width: 30px; height: 30px; border-radius: 50%; flex: none; border: 1.5px solid var(--ink);
  background: linear-gradient(115deg, var(--ink) 0 52%, var(--card) 52% 100%); }
#head h1 { font-size: 21px; font-weight: 600; letter-spacing: -0.01em; margin: 0; color: var(--ink); line-height: 1.15; }
#head .sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.chips { margin-top: 2px; }
.chip { display: inline-block; border: 1px solid var(--line); background: var(--card); color: var(--muted);
  padding: 2px 9px; margin: 0 6px 6px 0; border-radius: 3px; font-size: 12px; white-space: nowrap; }
.chip b { color: var(--ink); font-weight: 600; margin-left: 3px; font-variant-numeric: tabular-nums; }
.chip.ok { border-color: var(--ok); background: var(--ok-bg); } .chip.ok b { color: var(--ok); }
.chip.part { border-color: var(--part); background: var(--part-bg); } .chip.part b { color: var(--part); }
.tabs > .tab-nav button, .tab-wrapper button { font-size: 12.5px; letter-spacing: 0.02em; }
.tabs > .tab-nav button.selected, .tab-wrapper button.selected { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; font-weight: 600; }
.hint, .hint p { font-size: 12px; color: var(--muted); }
.mono, .mono * { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.lead p { font-size: 13px; color: var(--ink); max-width: 110ch; }
.prose { max-width: 105ch; } .prose h1 { font-size: 20px; } .prose h2 { font-size: 16px; margin-top: 22px; border-bottom: 1px solid var(--line); padding-bottom: 3px; }
.prose h3 { font-size: 13.5px; margin-top: 16px; } .prose p, .prose li { font-size: 13px; line-height: 1.55; }
table { font-size: 12.5px; } table td, table th { padding: 3px 9px !important; font-variant-numeric: tabular-nums; }
.prose table { border-collapse: collapse; border: 1px solid var(--line); }
.prose table td, .prose table th { border: 1px solid var(--line) !important; padding: 5px 11px !important; vertical-align: top; }
.prose table th { background: color-mix(in srgb, var(--line) 35%, transparent); color: var(--ink); font-weight: 600; text-align: left; white-space: nowrap; }
.ov table td:first-child, .ov table td:nth-child(2), .folders table td:first-child { white-space: nowrap; }
.folders table td .mono { word-break: break-all; font-size: 12px; }
.prose code { font-size: 12px; }
.block label > span, .block .label-wrap span { font-size: 11.5px; color: var(--muted); }
button.secondary { border: 1px solid var(--line) !important; }
button.stop { border: 1px solid #b3261e !important; }
.log textarea { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; font-size: 12px !important; line-height: 1.5 !important; white-space: pre !important; overflow-x: auto !important; }
.system { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 2px; }
.system h4, .system strong { color: var(--ink); }
.pos { text-align: center; font-size: 12px; color: var(--muted); padding-top: 6px; font-variant-numeric: tabular-nums; }
.gallery-item, .thumbnail-item { border-radius: 2px !important; }
"""


# ============================================================================ interface
init_project()
try:
    cf.reconcile(PROJECT_ROOT, NAMES_FILE)                      # repair dataset.csv after an interrupted collection
except Exception as _e:                                         # noqa
    print("reconcile:", _e)
_st, _vs = load_settings(), vs.status()

with gr.Blocks(title="Mooney Studio", analytics_enabled=False) as demo:
    gr.HTML("<div id='head'><div class='mark'></div><div><h1>Mooney Studio</h1><div class='sub'>Two-tone face images, "
            "and what vision systems make of them</div></div></div>")
    status = gr.HTML(status_html())

    with gr.Tabs():
        # ------------------------------------------------------------------ Overview
        with gr.Tab("Overview") as tab_overview:
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

        # ------------------------------------------------------------------ Run
        with gr.Tab("Run") as tab_run:
            gr.Markdown("One full run with one button. Give it a name, tick the steps and set their values, then press **Run**. "
                        "Every step does exactly what its own tab does, in this order; the log shows where it is. "
                        "At the end the run is saved under its name and date and the results appear below.", elem_classes="lead")
            with gr.Row():
                p_name = gr.Textbox("", label="Name of the run", placeholder="for example: final run", scale=2, max_lines=1)
                p_note = gr.Textbox("", label="Note (optional, stored with the run)", scale=3, max_lines=1,
                                    placeholder="for example: blur 6, 23 % white, 22 Mooneys adjusted by hand")
            with gr.Group():
                p_batch = gr.Checkbox(True, label="Step 1 · Make the Mooneys (same as Batch; Mooneys adjusted by hand are kept)")
                with gr.Row():
                    p_sigma = gr.Slider(0, 20, value=_st["sigma"], step=0.1, label="Blur σ (px)")
                    p_white = gr.Slider(5, 60, value=_st["white"], step=0.5, label="White share of the face (%)")
                    p_bg = gr.Radio(["black", "white"], value=_st["bg_mode"] if _st["bg_mode"] in ("black", "white") else "black",
                                    label="Background of the Mooney")
                    p_bgl = gr.Slider(0, 255, value=_st["bg_level"], step=1, label="Phase-1 gray")
            with gr.Row():
                p_systems = gr.CheckboxGroup(PIPE_SYSTEMS, value=[PIPE_SYSTEMS[0], PIPE_SYSTEMS[2], PIPE_SYSTEMS[3]],
                                             label="Step 2 · Vision systems, on every original and every Mooney", scale=3)
            with gr.Row():
                p_3d = gr.CheckboxGroup(list(D3_SOURCES), value=[], scale=3,
                                        label="Step 3 · 3D faces: render, make Mooneys, run systems 1, 3 and 4 against the true depth")
                p_faces = gr.Slider(2, 30, value=cfg.BASEL_FACES, step=1, label="Basel faces", scale=1)
            with gr.Row():
                p_export = gr.Checkbox(True, label="Step 4 · Export figures", scale=1)
                p_save = gr.Checkbox(True, label="Step 5 · Save the run under its name", scale=1)
                p_keep = gr.CheckboxGroup([v[0] for v in an.RUN_GROUPS.values()], value=[an.RUN_GROUPS[k][0] for k in an.RUN_DEFAULT],
                                          label="The saved run includes, besides the Mooneys and their values", scale=3)
            gr.Markdown("3D from photos uses the suggested photos and the five default lights; Basel faces are whole heads, frontal, "
                        "with cast shadows. For other choices use the **3D faces** tab. Other Batch options (white rule, "
                        "preprocessing) are taken from the saved defaults.", elem_classes="hint")
            with gr.Row():
                p_run = gr.Button("Run", variant="primary", scale=2)
                p_only = gr.Button("Only save the current state as a run", scale=1)
            p_log = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False, autoscroll=True, elem_classes="log")
            p_msg = gr.Markdown(elem_classes="hint")
            gr.Markdown("### Results of the current state", elem_classes="prose")
            p_table = gr.Dataframe(headers=SUMMARY_COLS, label="Summary   counts are images, other values are means", wrap=True, interactive=False)
            p_fig = gr.Image(show_label=False, height=380)
            p_cmp = gr.Dataframe(interactive=False, wrap=True, max_height=520, label="Photos and 3D sets side by side (empty until a 3D step has run)")
            gr.Markdown("### Saved runs\n<span class='hint'>Newest first. Compare them in **Analysis**; each is a folder in the "
                        "runs folder shown on Overview.</span>", elem_classes="prose")
            p_runs = gr.Dataframe(headers=RUNS_COLS, interactive=False, wrap=True, column_widths=RUNS_WIDTHS)

        # ------------------------------------------------------------------ Dataset
        with gr.Tab("Dataset"):
            gr.Markdown("Collect openly licensed portraits from Wikimedia Commons, look through them, and remove the ones you do not want. "
                        "Each photo is cropped from hair to collar and saved at 512 × 512 px in `Original/`; the uncropped photo goes to `Raw/` "
                        "and its licence, author and source page to `dataset.csv`.", elem_classes="lead")
            with gr.Row():
                names_box = gr.Textbox(names_default(), label="People, one per line (as on Wikimedia Commons)", lines=9, scale=2)
                with gr.Column(scale=3):
                    collect_btn = gr.Button("Collect", variant="primary", size="md")
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

        # ------------------------------------------------------------------ Batch
        with gr.Tab("Batch"):
            gr.Markdown("Make a Mooney for every photo with one set of values. The values below are the project defaults; "
                        "afterwards you can adjust single photos by hand in **Mooney**, and Batch will leave those alone.", elem_classes="lead")
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
                b_save_def = gr.Button("Save these values as defaults", size="md", scale=1)
                b_subset = gr.Button("Save favourites to Subset/", size="md", scale=1)
            b_log = gr.Textbox(label="Log", lines=9, max_lines=9, interactive=False, autoscroll=True, elem_classes="log")
            b_favs = gr.Textbox(fav_text(), label="Favourites for the subset (click a Mooney below to add or remove it)", interactive=False)
            b_gal = gr.Gallery(gallery_of("Mooney"), label="Mooney/", columns=10, height=420, object_fit="contain", allow_preview=False)
            with gr.Accordion("params.csv (every value used, per photo)", open=False):
                b_tab = gr.Dataframe(params_table(), headers=PARAM_COLS, interactive=False, wrap=True)

        # ------------------------------------------------------------------ Mooney
        with gr.Tab("Mooney"):
            gr.Markdown("Look at one photo at a time and see the four phases. Move the sliders to adjust it by hand, then **Save this Mooney**. "
                        "A change affects this photo only, and only when you save it; another photo always opens with its own saved "
                        "values. Mark the best ones with the star; they become the subset of ten.", elem_classes="lead")
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

        # ------------------------------------------------------------------ Vision systems
        with gr.Tab("Vision systems"):
            gr.Markdown("Each system takes every original and its Mooney and returns values for both. One result file per image goes to "
                        "`ResultN/Original` and `ResultN/Mooney`, and a table per system to `analysis/`.", elem_classes="lead")
            v_check = gr.Button("Check which systems are ready", size="sm", scale=0)
            with gr.Row(elem_classes="system"):
                gr.Markdown("**1 · Face model**   YuNet detector and SFace embedding (OpenCV)  \n"
                            "<span class='hint'>Is a face detected? How similar is the Mooney's embedding to its original's? "
                            "The same again for the Mooney blurred once more.</span>", scale=3)
                v_s1 = gr.Textbox(_vs["face"], label="Status", interactive=False, scale=2)
                v_run1 = gr.Button("Run", variant="primary", size="sm", scale=1)
            with gr.Row(elem_classes="system"):
                gr.Markdown("**2 · Vision-language model**   a local model through Ollama, same prompt for every image, temperature 0  \n"
                            "<span class='hint'>Does it say a face is present? Head pose, light direction and a one-sentence description.</span>", scale=3)
                v_s2 = gr.Textbox(_vs["vlm"], label="Status", interactive=False, scale=2)
                v_run2 = gr.Button("Run", variant="primary", size="sm", scale=1)
            with gr.Row():
                v_model = gr.Dropdown(choices=_vs["vlm_models"], value=pick_model(_vs["vlm_models"]),
                                      label="Ollama model", allow_custom_value=True, scale=1)
                v_prompt = gr.Textbox(vs.VLM_PROMPT, label="Prompt (saved with the results)", lines=3, scale=3)
            with gr.Row():
                v_scope = gr.Radio(list(VLM_SCOPES), value=VLM_SCOPE_LABEL[_st["vlm_scope"]], label="Run system 2 on", scale=1)
                gr.Markdown("This model is heavy: about 5 GB of memory and several seconds per image, so all photos keep the laptop busy "
                            "for a long time. *The subset only* runs it on the pairs in `Subset/` (or on your favourites while that folder "
                            "is empty). Systems 1, 3 and 4 are light and always run on everything.", elem_classes="hint", scale=3)
            with gr.Row(elem_classes="system"):
                gr.Markdown("**3 · Learned depth**   Depth Anything V2, ViT-S: a neural network trained on millions of images  \n"
                            "<span class='hint'>A depth map for each image, and how well the Mooney's depth map matches its original's.</span>", scale=3)
                v_s3 = gr.Textbox(_vs["depth"], label="Status", interactive=False, scale=2)
                v_run3 = gr.Button("Run", variant="primary", size="sm", scale=1)
            with gr.Row(elem_classes="system"):
                gr.Markdown("**4 · Depth from shading**   shape from shading after Tsai & Shah (1994): a formula, nothing learned  \n"
                            "<span class='hint'>The same two values as system 3, from a model that assumes a matte surface and light from above "
                            "and knows nothing about faces.</span>", scale=3)
                v_s4 = gr.Textbox(_vs["shading"], label="Status", interactive=False, scale=2)
                v_run4 = gr.Button("Run", variant="primary", size="sm", scale=1)
            v_log = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False, autoscroll=True, elem_classes="log")
            with gr.Accordion("Where every value comes from: model output, our settings, what we calculate", open=False):
                gr.Markdown(read_doc("VISION_SYSTEMS.md", "_docs/VISION_SYSTEMS.md is missing._"), elem_classes="prose")
            with gr.Accordion("Later steps (off by default)", open=False):
                v_identify = gr.Checkbox(False, label="Identity matching: also test whether systems 1 and 2 recognise the person "
                                                      "(system 1 ranks against the other photos; system 2 is asked for the name)")
                v_src = gr.Radio(["Original", "Raw"], value="Original", label="Source folder")
                with gr.Row(elem_classes="system"):
                    gr.Markdown("**5 · 3D face mesh**   MediaPipe Face Landmarker, 478 3D landmarks  \n"
                                "<span class='hint'>Is a mesh obtained, and how close is it to the original's mesh? Writes to Result5/. "
                                "The controlled 3D experiment is in tab 6.</span>", scale=3)
                    v_s5 = gr.Textbox(_vs["mesh"], label="Status", interactive=False, scale=2)
                    v_run5 = gr.Button("Run", size="sm", scale=1)

        # ------------------------------------------------------------------ Results
        with gr.Tab("Results"):
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
            gr.Markdown("<span class='hint'>To keep this state under a name, use the **Run** tab. Saved runs can be compared "
                        "in **Analysis**.</span>", elem_classes="prose")

        # ------------------------------------------------------------------ Analysis
        with gr.Tab("Analysis") as tab_analysis:
            gr.Markdown("How the results relate to the photos. Each photo is one data point: its brightness, head turn and Mooney settings on one "
                        "side, what each vision system returned on the other. Nothing is computed by a model here; the tab only reads the "
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
            gr.Markdown("### Compare runs\n<span class='hint'>Runs are saved in Results. *current* is the project as it is now. "
                        "The first table is the Results summary with one column per run; the second shows what each run was made with.</span>",
                        elem_classes="prose")
            c_runs = gr.Dropdown(choices=run_choices(), value=[], multiselect=True, label="Runs to put side by side")
            c_msg = gr.Markdown(elem_classes="hint")
            c_table = gr.Dataframe(interactive=False, wrap=True, label="Results, one column per run")
            c_settings = gr.Dataframe(interactive=False, wrap=True, label="What each run's Mooneys were made with (from its params.csv)")

        # ------------------------------------------------------------------ 3D faces
        with gr.Tab("3D faces") as tab_3d:
            gr.Markdown("A controlled version of the experiment: pictures of 3D faces drawn under light directions we choose. For these "
                        "pictures the light and the depth of every pixel are known, so the depth systems can be checked against the "
                        "truth, for the render and for its Mooney. The renders go through the same Mooney generator and the same "
                        "systems as the photos. Each source of 3D faces has its own folder with the same layout as the project.",
                        elem_classes="lead")
            d_source = gr.Radio(list(D3_SOURCES), value=list(D3_SOURCES)[0], label="Source of the 3D faces",
                                info="Reconstructed faces are written to 3D/, invented faces to 3D_basel/. The two sets are kept side by side.")
            d_caveat = gr.Markdown(D3_LEAD[f3.SUB] + " " + a3.caveat(f3.SUB) + " Strongly turned heads are skipped, because this simple "
                                   "mesh breaks for them.", elem_classes="hint")
            with gr.Column(visible=True) as d_box_photos:
                with gr.Row():
                    d_files = gr.Dropdown(choices=images_in("Original"), value=a3.default_photos(PROJECT_ROOT), multiselect=True, scale=4,
                                          label="Photos to turn into 3D faces   suggested: favourites first, one photo per person")
                    d_suggest = gr.Button("Suggest again", size="sm", scale=0, min_width=120)
            with gr.Column(visible=False) as d_box_basel:
                d_model_note = gr.Markdown(basel_note(), elem_classes="hint")
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
                d_b1 = gr.Button("1 Render", size="md")
                d_b2 = gr.Button("2 Make Mooneys", size="md")
                d_b3 = gr.Button("3 Run systems 1, 3 and 4", size="md")
                d_all = gr.Button("Run all three steps", variant="primary", size="md")
            gr.Markdown("The buttons work on the source chosen above. *1 Render* replaces the earlier renders of that source together "
                        "with their Mooneys and results. *2 Make Mooneys* uses the values saved as defaults in Batch and removes "
                        "earlier results of step 3. *3* runs the face model and both depth systems and compares them with the truth.",
                        elem_classes="hint")
            d_state = gr.Markdown(elem_classes="hint")
            d_log = gr.Textbox(label="Log", lines=10, max_lines=10, interactive=False, autoscroll=True, elem_classes="log")
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
            with gr.Accordion("All renders and their Mooneys", open=False):
                d_gal_o = gr.Gallery(label="3D/Original/", columns=10, height=260, object_fit="contain", allow_preview=False)
                d_gal_m = gr.Gallery(label="3D/Mooney/", columns=10, height=260, object_fit="contain", allow_preview=False)
            gr.Markdown("### Compare stimulus sets\n<span class='hint'>The photos of the main project, the faces reconstructed from "
                        "photos and the invented Basel faces, each with what the same systems returned. " + a3.COMPARE_NOTE + "</span>",
                        elem_classes="prose")
            x_table = gr.Dataframe(headers=["Measure"] + [a3.SET_NAMES[k] for k in a3.SETS], interactive=False, wrap=True, max_height=760,
                                   label="One row per measure, one column per set")
            x_fig = gr.Image(show_label=False)
            x_lights = gr.Dataframe(headers=a3.LIGHT_HEADERS, interactive=False, wrap=True, max_height=760,
                                    label="The two 3D sets by light direction")
            with gr.Row():
                x_refresh = gr.Button("Refresh comparison", size="sm", scale=0)
                x_export = gr.Button("Export comparison to figures/", size="sm", scale=0)
                x_msg = gr.Markdown(elem_classes="hint")

        # ------------------------------------------------------------------ Notes (only if the working folder has any)
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

        # ------------------------------------------------------------------ Docs
        with gr.Tab("Docs"):
            gr.Markdown(read_doc("MANUAL.md", "_docs/MANUAL.md is missing._"), elem_classes="prose")

    # ====================================================================== wiring
    refresh_overview = lambda: (overview_md(), status_html())

    def refresh_page():
        """Runs on every page load, so a reload always shows what is on disk right now."""
        return (status_html(), overview_md(), gallery_of("Original"), dataset_table(), gallery_of("Mooney"), params_table(),
                fav_text(), fav_gallery(), d3_suggest(), gr.update(choices=run_choices()))

    demo.load(refresh_page, None, [status, overview, ds_gallery, ds_table, b_gal, b_tab, b_favs, fav_gal, d_files, c_runs])
    tab_overview.select(refresh_overview, None, [overview, status])
    ov_refresh.click(refresh_overview, None, [overview, status])
    ov_work_btn.click(change_work_dir, ov_work, ov_work_msg)
    ov_work.submit(change_work_dir, ov_work, ov_work_msg)

    # Dataset
    collect_btn.click(collect, [names_box, per_id, n_cand, min_face], [ds_log, ds_gallery, ds_table, status])
    ds_gallery.select(pick_photo, None, [ds_sel, ds_msg])
    ds_del.click(remove_photo, [ds_sel, lab_src], [ds_sel, ds_msg, ds_gallery, ds_table, status, b_gal, b_tab, fav_gal, lab_file])

    # Batch
    b_values = [b_sigma, b_white, b_bgm, b_bgl, b_sep, b_rule, b_bw, b_bb, b_pre]
    b_run.click(run_batch, [b_src] + b_values + [b_measure, b_keep, b_sweep, b_sig, b_wht], [b_log, b_gal, b_tab, status])
    b_save_def.click(save_defaults, b_values, b_log)
    b_gal.select(star_in_gallery, None, [b_favs, fav_gal])
    b_subset.click(save_subset, b_src, [b_log, status])

    # Mooney
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

    # Vision systems
    v_check.click(vision_status, None, [v_s1, v_s2, v_s3, v_s4, v_s5, v_model])
    v_identify.change(lambda on: vs.VLM_PROMPT_IDENTITY if on else vs.VLM_PROMPT, v_identify, v_prompt)
    v_run1.click(vision_job(vs.run_face, args=lambda src, ident: {"source": src, "identify": bool(ident)}),
                 [v_src, v_identify], [v_log, status])
    v_run2.click(vision_job(vs.run_vlm, refuse="Not started. Either no Ollama model is selected (start Ollama with `ollama serve`, pull a vision model, press the check button), "
                                                    "or the subset is empty (mark favourites with the star first, or choose All photos).",
                            args=vlm_args),
                 [v_src, v_identify, v_model, v_prompt, v_scope], [v_log, status])
    v_run3.click(vision_job(vs.run_depth, args=lambda src: {"source": src}), [v_src], [v_log, status])
    v_run4.click(vision_job(vs.run_shading, args=lambda src: {"source": src}), [v_src], [v_log, status])
    v_run5.click(vision_job(vs.run_mesh, args=lambda src: {"source": src}), [v_src], [v_log, status])

    # Results
    r_refresh.click(results_load, None, [r_table, r_fig, r_stem, r_msg])
    r_stem.change(results_show, r_stem, [r_panel, r_detail])
    r_prev.click(lambda s: results_step(s, -1), r_stem, r_stem)
    r_next.click(lambda s: results_step(s, +1), r_stem, r_stem)
    r_export.click(results_export, None, r_msg)
    demo.load(results_load, None, [r_table, r_fig, r_stem, r_msg])
    # Run
    p_shown = [p_msg, p_table, p_fig, p_cmp, p_runs]
    p_run.click(pipeline, [p_name, p_note, p_keep, p_batch, p_sigma, p_white, p_bg, p_bgl, p_systems, p_3d, p_faces, p_export, p_save],
                [p_log] + p_shown + [status, c_runs])
    p_only.click(pipeline_save_only, [p_name, p_keep, p_note], [p_msg, p_runs, c_runs])
    tab_run.select(pipeline_view, None, p_shown)

    # Analysis
    def analysis_groups(by):
        head, rows = an.group_table(PROJECT_ROOT, by)
        return gr.update(value=rows, headers=head), an.fig_groups(PROJECT_ROOT, by), an.statements(PROJECT_ROOT, by)

    def analysis_load(by):
        hard, easy, rank_rows = an.ranking(PROJECT_ROOT)
        has_results = an.fig_systems(PROJECT_ROOT) is not None
        msg = "" if has_results else "No results to analyse yet. Run the vision systems in tab 4, then press Refresh."
        return (an.dataset_table(PROJECT_ROOT), an.fig_dataset(PROJECT_ROOT), an.fig_systems(PROJECT_ROOT), *analysis_groups(by),
                an.agreement_table(PROJECT_ROOT), an.fig_agreement(PROJECT_ROOT), hard, easy, rank_rows, msg)

    def analysis_export(by):
        made = an.export(PROJECT_ROOT, by)
        return f"Wrote {len(made)} files to figures/: " + ", ".join(made) if made else "Nothing to export yet."

    a_outputs = [a_ds_table, a_ds_fig, a_sys_fig, a_grp_table, a_grp_fig, a_words, a_agree_table, a_agree_fig, a_hard, a_easy, a_rank, a_msg]
    a_refresh.click(analysis_load, a_by, a_outputs)
    tab_analysis.select(analysis_load, a_by, a_outputs)
    a_by.change(analysis_groups, a_by, [a_grp_table, a_grp_fig, a_words])
    a_export.click(analysis_export, a_by, a_msg)
    c_runs.change(runs_compare, c_runs, [c_table, c_settings, c_msg])

    # 3D faces
    d_shown = [d_state, d_table, d_fig, d_person, d_sheet, d_gal_o, d_gal_m]
    x_shown = [x_table, x_lights, x_fig, x_msg]
    for button, steps in ((d_b1, ["render"]), (d_b2, ["Mooneys"]), (d_b3, ["systems"]), (d_all, ["render", "Mooneys", "systems"])):
        button.click(d3_run(steps), [d_source, d_files, d_lights, d_person, d_n, d_model, d_poses, d_shadows],
                     [d_log] + d_shown).then(d3_compare, None, x_shown)
    d_source.change(d3_switch, [d_source, d_model], [d_box_photos, d_box_basel, d_caveat, d_model_note, d_export] + d_shown)
    d_model.change(basel_note, d_model, d_model_note)
    d_suggest.click(d3_suggest, None, d_files)
    d_person.change(d3_sheet, [d_source, d_person], d_sheet)
    d_refresh.click(d3_results, [d_source, d_person], d_shown)
    tab_3d.select(d3_results, [d_source, d_person], d_shown).then(d3_compare, None, x_shown)
    tab_3d.select(basel_note, d_model, d_model_note)
    d_export.click(d3_export, d_source, d_msg)
    x_refresh.click(d3_compare, None, x_shown)
    x_export.click(d3_compare_export, None, x_msg)


def launch_options():
    """Address, port, public link and login, from config.py, config.json, the environment or the command line."""
    auth = (cfg.AUTH_USER, cfg.AUTH_PASSWORD) if cfg.AUTH_USER and cfg.AUTH_PASSWORD else None
    if cfg.HOST not in ("127.0.0.1", "localhost") and auth is None:
        print("Note: the app is reachable from other machines without a login. Set MOONEY_USER and MOONEY_PASSWORD to add one.")
    return {"server_name": cfg.HOST, "server_port": cfg.PORT or None, "share": bool(cfg.SHARE), "auth": auth,
            "allowed_paths": [PROJECT_ROOT, RUNS_ROOT, NOTES_ROOT, HERE]}


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, **launch_options())
