"""
Analysis: relate what the vision systems returned to properties of the photos and of the Mooneys.

Everything here reads files that other steps wrote (dataset.csv, params.csv, analysis/result*.csv) and never
runs a model. One record per photo is built by `photos()`; all tables, figures and sentences derive from it.
Any input may be missing or partial: functions then return None, an empty list, or a short message.

Measures per photo (all about the Mooney unless stated):
    det_conf      detector confidence on the original                       value 0..1
    sim           similarity of the Mooney's embedding to its original's    value -1..1
    sim_reblur    the same after the Mooney was blurred again               value -1..1
    depth         learned depth: correlation with the original's map        value -1..1
    shading       depth from shading: correlation with the original's map   value -1..1
    det_mooney    face detected in the Mooney                               yes / no
    det_reblur    face detected in the re-blurred Mooney                    yes / no
    vlm_mooney    language model says a face is present in the Mooney       yes / no

Saved runs (save_run, list_runs, compare_runs): a run is a frozen copy of the Mooneys, their parameters and all
result tables in runs/<date>_<name>/, so that different generation settings can be compared later.
"""
import csv
import io
import json
import os
import re
import shutil
import time

import numpy as np
from PIL import Image

INK, MUTED, GRID = "#15181d", "#5c6672", "#e6e9ed"
BLUE, ORANGE, GREEN = "#2b5a9e", "#d9622b", "#1f9a6e"           # Original, Mooney, Mooney re-blurred (fixed everywhere)
FEW = 10                                                        # groups with fewer photos are flagged

VALUES = {"det_conf": "Detector confidence (original)", "sim": "Similarity to original (Mooney)",
          "sim_reblur": "Similarity to original (re-blurred)", "depth": "Learned depth: match with original",
          "shading": "Depth from shading: match with original"}
SHARES = {"det_mooney": "Face detected in the Mooney", "det_reblur": "Face detected after re-blurring",
          "vlm_mooney": "Language model says face (Mooney)"}
SHARE_COLOR = {"det_mooney": ORANGE, "det_reblur": GREEN, "vlm_mooney": ORANGE}
SHORT = {"det_conf": "detector confidence", "sim": "similarity", "sim_reblur": "similarity after re-blur", "depth": "learned depth",
         "shading": "depth from shading", "det_mooney": "face detected", "det_reblur": "face detected after re-blur",
         "vlm_mooney": "language model says face"}
GROUPINGS = {"Face brightness": "brightness", "Head turn": "turn", "How the Mooney was made": "mode", "White share reached": "white"}
PROPERTIES = {"face_brightness": "face brightness", "turn": "head turn", "white": "white share"}


# ============================================================================ reading
def _csv(project, *parts):
    p = os.path.join(project, *parts)
    if not os.path.exists(p):
        return []
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _yes(v):
    if v in (None, ""):
        return None
    return str(v).strip().lower() in ("true", "1", "yes")


def photos(project):
    """One dict per photo, keyed by what the other functions need. Missing values are None."""
    recs = {}
    for r in _csv(project, "dataset.csv"):
        stem = os.path.splitext(r.get("file", ""))[0]
        if not stem or not os.path.exists(os.path.join(project, "Original", r["file"])):
            continue
        yaw = _num(r.get("yaw"))
        recs[stem] = {"stem": stem, "person": stem.rsplit("_", 1)[0], "identity": r.get("identity", ""), "license": r.get("license", ""),
                      "turn": None if yaw is None else abs(yaw)}
    blank = lambda stem: recs.setdefault(stem, {"stem": stem, "person": stem.rsplit("_", 1)[0], "identity": "", "license": "", "turn": None})
    for r in _csv(project, "params.csv"):
        stem = os.path.splitext(r.get("file", ""))[0]
        if not stem:
            continue
        rec = blank(stem); wf = _num(r.get("white_fraction"))
        rec.update(mode=r.get("mode") or "batch", white=None if wf is None else wf * 100, face_brightness=_num(r.get("face_brightness")),
                   sigma=_num(r.get("sigma")))
    for r in _csv(project, "analysis", "result1_face.csv"):
        rec = blank(r.get("file", ""))
        if r.get("condition") == "Original":
            rec.update(det_orig=_yes(r.get("face_detected")), det_conf=_num(r.get("detector_score")))
        elif r.get("condition") == "Mooney":
            rec.update(det_mooney=_yes(r.get("face_detected")), det_reblur=_yes(r.get("reblur_face_detected")),
                       sim=_num(r.get("similarity_to_own_original")), sim_reblur=_num(r.get("reblur_similarity_to_own_original")))
    for r in _csv(project, "analysis", "result2_vlm.csv"):
        ans = str(r.get("face_present", "")).strip().lower()
        if ans in ("yes", "no"):
            blank(r.get("file", ""))["vlm_orig" if r.get("condition") == "Original" else "vlm_mooney"] = ans == "yes"
    for name, key in (("result3_depth.csv", "depth"), ("result4_shading.csv", "shading")):
        for r in _csv(project, "analysis", name):
            v = _num(r.get("depth_correlation_original_vs_mooney"))
            if v is not None:
                blank(r.get("file", ""))[key] = v
    recs.pop("", None)
    return [recs[k] for k in sorted(recs)]


def _vals(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)


def _share(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return sum(v), len(v)


# ============================================================================ grouping
def groups(rows, by):
    """Ordered list of (label, rows). Photos without the property are left out; the count is returned as well."""
    key = GROUPINGS.get(by, by)
    if key == "brightness":
        have = sorted((r for r in rows if r.get("face_brightness") is not None), key=lambda r: r["face_brightness"])
        if len(have) < 3:
            return [], len(rows) - len(have)
        k = len(have) // 3; parts = [have[:k], have[k:len(have) - k], have[len(have) - k:]]
        lab = lambda name, p: f"{name} ({p[0]['face_brightness']:.0f} to {p[-1]['face_brightness']:.0f})"
        return [(lab(n, p), p) for n, p in zip(("darker", "medium", "brighter"), parts) if p], len(rows) - len(have)
    if key == "turn":
        have = [r for r in rows if r.get("turn") is not None]
        cut = [("frontal", lambda t: t < 0.15), ("slightly turned", lambda t: 0.15 <= t < 0.35), ("strongly turned", lambda t: t >= 0.35)]
        return [(n, [r for r in have if f(r["turn"])]) for n, f in cut if any(f(r["turn"]) for r in have)], len(rows) - len(have)
    if key == "mode":
        have = [r for r in rows if r.get("mode")]
        names = (("made by Batch", "batch"), ("adjusted by hand", "manual"))
        return [(n, [r for r in have if r["mode"] == m]) for n, m in names if any(r["mode"] == m for r in have)], len(rows) - len(have)
    if key == "white":
        have = [r for r in rows if r.get("white") is not None]
        cut = [("under 22 % white", lambda w: w < 22), ("22 to 27 % white", lambda w: 22 <= w < 27),
               ("27 to 32 % white", lambda w: 27 <= w < 32), ("32 % white or more", lambda w: w >= 32)]
        return [(n, [r for r in have if f(r["white"])]) for n, f in cut if any(f(r["white"]) for r in have)], len(rows) - len(have)
    return [], len(rows)


# ============================================================================ tables
def dataset_table(project):
    rows = photos(project)
    if not rows:
        return None
    people = {}
    for r in rows:
        people[r["person"]] = people.get(r["person"], 0) + 1
    out = [["Photos", len(rows)], ["People", len(people)], ["People with two or more photos", sum(1 for v in people.values() if v >= 2)],
           ["Photos with a Mooney", sum(1 for r in rows if r.get("white") is not None)],
           ["Mooneys adjusted by hand", sum(1 for r in rows if r.get("mode") == "manual")]]
    lic = {}
    for r in rows:
        name = r.get("license") or "not recorded"
        short = ("Public domain" if "public" in name.lower() or name.upper().startswith("PD") else
                 "CC0" if "cc0" in name.lower() else "CC BY-SA" if "sa" in name.lower() else "CC BY" if "by" in name.lower() else name)
        lic[short] = lic.get(short, 0) + 1
    out += [[f"Licence: {k}", v] for k, v in sorted(lic.items(), key=lambda kv: -kv[1])]
    return out


def group_table(project, by):
    """(headers, rows): one row per group, one column per measure, always with the number of photos."""
    rows = photos(project)
    gs, missing = groups(rows, by)
    if not gs:
        return ["group"], None
    head = ["group", "n"] + [SHORT[k] for k in SHARES] + ["mean " + SHORT[k] for k in VALUES if k != "det_conf"]
    out = []
    for label, part in gs + [("all photos", [r for _, p in gs for r in p])]:
        line = [label + ("  (few photos)" if len(part) < FEW else ""), len(part)]
        for k in SHARES:
            a, n = _share(part, k)
            line.append(f"{100 * a / n:.0f} %  ({a} of {n})" if n else "no data")
        for k in VALUES:
            if k == "det_conf":
                continue
            v = _vals(part, k)
            line.append(f"{v.mean():.2f}  (n {len(v)})" if len(v) else "no data")
        out.append(line)
    return head, out


def agreement_table(project):
    """Pairwise Pearson correlations between the per-photo measures, with the number of photos behind each."""
    rows = photos(project)
    keys = [k for k in ("sim", "sim_reblur", "depth", "shading", "det_reblur", "vlm_mooney") if sum(r.get(k) is not None for r in rows) >= 5]
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pair = [(float(r[a]), float(r[b])) for r in rows if r.get(a) is not None and r.get(b) is not None]
            if len(pair) < 5:
                continue
            x, y = np.array(pair).T
            if x.std() == 0 or y.std() == 0:
                continue
            out.append([SHORT[a], SHORT[b], round(float(np.corrcoef(x, y)[0, 1]), 2), len(pair)])
    return sorted(out, key=lambda r: -abs(r[2])) or None


def ranking(project, n=8):
    """
    (hardest, easiest, table). The first two are lists of (Mooney path, caption) for galleries; the table has the
    values behind them (columns RANK_COLS). Score = mean of the photo's ranks over the measures available.
    """
    rows = photos(project)
    keys = [k for k in ("sim", "sim_reblur", "depth", "shading", "det_reblur", "vlm_mooney") if sum(r.get(k) is not None for r in rows) >= 5]
    if not keys:
        return [], [], None
    score = {r["stem"]: [] for r in rows}
    for k in keys:
        have = [r for r in rows if r.get(k) is not None]
        v = np.array([float(r[k]) for r in have])
        order = v.argsort().argsort().astype(float)
        for val in np.unique(v):                                   # ties share their mean rank
            order[v == val] = order[v == val].mean()
        for r, o in zip(have, order / max(len(have) - 1, 1)):
            score[r["stem"]].append(o)
    scored = sorted(((float(np.mean(s)), stem) for stem, s in score.items() if s), key=lambda t: t[0])
    by_stem = {r["stem"]: r for r in rows}

    f2 = lambda v: "" if v is None else f"{v:.2f}"
    yn = lambda v: "" if v is None else ("yes" if v else "no")

    def items(order, name):
        gallery, table = [], []
        for pos, (sc, stem) in enumerate(order[:n], 1):
            r = by_stem[stem]; path = os.path.join(project, "Mooney", stem + ".png")
            if os.path.exists(path):
                gallery.append((path, f"{pos} · {stem}"))
            table.append([name, pos, stem, f2(r.get("sim")), f2(r.get("sim_reblur")), f2(r.get("depth")), f2(r.get("shading")),
                          yn(r.get("det_reblur")), yn(r.get("vlm_mooney"))])
        return gallery, table

    hard, hard_rows = items(scored, "hardest")
    easy, easy_rows = items(scored[::-1], "easiest")
    return hard, easy, hard_rows + easy_rows


RANK_COLS = ["list", "place", "photo", "similarity", "after re-blur", "learned depth", "depth from shading", "face after re-blur", "language model says face"]


# ============================================================================ figures
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b9c1cb", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "axes.linewidth": 0.8})
    return plt


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, pad=8); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


def _empty(ax, title, text="no data yet"):
    ax.set_title(title, pad=8); ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _image(fig, plt):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _hist(ax, series, title, xlabel, lo, hi, bins=12):
    """series: list of (label, values, colour). Bars of several series stand side by side, never on top of each other."""
    series = [(l, v, c) for l, v, c in series if len(v)]
    if not series:
        return _empty(ax, title)
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist([np.clip(v, lo, hi) for _, v, _ in series], bins=edges, color=[c for _, _, c in series],
            label=[f"{l} (n {len(v)})" for l, v, _ in series], rwidth=0.86, linewidth=0)
    for (l, v, c) in series:
        ax.axvline(float(np.mean(v)), color=c, linewidth=1.2, linestyle=(0, (4, 3)))
    if len(series) > 1:
        ax.legend(fontsize=8, loc="upper right", handlelength=1.2)
        note = "dashed lines: means " + " and ".join(f"{np.mean(v):.2f}" for _, v, _ in series)
    else:
        m = float(np.mean(series[0][1]))
        note = f"n {len(series[0][1])} · dashed line: mean {m:.0f}" if abs(m) >= 10 else f"n {len(series[0][1])} · dashed line: mean {m:.2f}"
    _style(ax, title, f"{xlabel}\n{note}", "photos")
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))


def fig_dataset(project):
    """Distributions of the two photo properties used for grouping."""
    plt = _plt(); rows = photos(project)
    if plt is None or not rows:
        return None
    fb, turn = _vals(rows, "face_brightness"), _vals(rows, "turn")
    if not len(fb) and not len(turn):
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 2.9))
    if len(fb):
        _hist(ax[0], [("photos", fb, BLUE)], "Face brightness of the photos", "mean gray value of the face (0 = black, 255 = white)", 40, 200, 16)
    else:
        _empty(ax[0], "Face brightness of the photos", "appears after Batch has run")
    if len(turn):
        _hist(ax[1], [("photos", turn, BLUE)], "Head turn of the photos", "0 = facing the camera, larger = turned further", 0, 1, 16)
    else:
        _empty(ax[1], "Head turn of the photos")
    fig.tight_layout(w_pad=3)
    return _image(fig, plt)


def _bars(ax, title, items):
    """items: list of (label, yes, n, colour). Share of photos as bars with the counts written on them."""
    items = [it for it in items if it[2]]
    if not items:
        return _empty(ax, title)
    x = np.arange(len(items)); h = [100 * a / n for _, a, n, _ in items]
    ax.bar(x, h, width=0.56, color=[c for *_, c in items], linewidth=0)
    for xi, hi, (_, a, n, _) in zip(x, h, items):
        ax.text(xi, hi + 2.5, f"{hi:.0f} %\n{a} of {n}", ha="center", va="bottom", fontsize=8, color=INK, linespacing=1.15)
    ax.set_xticks(x, [l for l, *_ in items], fontsize=8.5, color=INK); ax.set_ylim(0, 128); ax.set_yticks([0, 25, 50, 75, 100])
    _style(ax, title, "", "share of photos (%)")


def fig_systems(project):
    """Small multiples: what each system returned, per photo, for the original and for the Mooney."""
    plt = _plt(); rows = photos(project)
    if plt is None or not rows or not any(r.get(k) is not None for r in rows for k in list(VALUES) + list(SHARES)):
        return None
    fig, ax = plt.subplots(2, 3, figsize=(12.4, 6.2)); ax = ax.ravel()
    _hist(ax[0], [("original", _vals(rows, "det_conf"), BLUE)], "1 Face model: detector confidence on the originals", "confidence (0 to 1)", 0, 1)
    _hist(ax[1], [("Mooney", _vals(rows, "sim"), ORANGE), ("Mooney, re-blurred", _vals(rows, "sim_reblur"), GREEN)],
          "1 Face model: similarity to the original", "cosine similarity (0 = nothing in common)", -0.2, 0.8)
    a = [_share(rows, k) for k in ("det_orig", "det_mooney", "det_reblur")]
    _bars(ax[2], "1 Face model: a face is detected", [("original", *a[0], BLUE), ("Mooney", *a[1], ORANGE), ("Mooney,\nre-blurred", *a[2], GREEN)])
    v = [_share(rows, k) for k in ("vlm_orig", "vlm_mooney")]
    _bars(ax[3], "2 Language model: says a face is present", [("original", *v[0], BLUE), ("Mooney", *v[1], ORANGE)])
    _hist(ax[4], [("Mooney", _vals(rows, "depth"), ORANGE)], "3 Learned depth: Mooney's map vs original's map", "correlation (1 = same shape)", -0.4, 1)
    _hist(ax[5], [("Mooney", _vals(rows, "shading"), ORANGE)], "4 Depth from shading: Mooney's map vs original's map", "correlation (1 = same shape)", -0.4, 1)
    fig.tight_layout(w_pad=3, h_pad=2.6)
    return _image(fig, plt)


def fig_groups(project, by):
    """The three yes/no measures per group, one panel per measure."""
    plt = _plt(); rows = photos(project)
    gs, _ = groups(rows, by)
    if plt is None or not gs or not any(_share(p, k)[1] for _, p in gs for k in SHARES):
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4), sharey=True)
    for a, (k, title) in zip(ax, SHARES.items()):
        items = []
        for label, part in gs:
            yes, n = _share(part, k)
            name = label.split(" (")[0].replace(" white", "").replace(" to ", "–") + ("\n(few photos)" if 0 < n < FEW else "")
            items.append((name, yes, n, SHARE_COLOR[k]))
        _bars(a, title, items)
        if a is not ax[0]:
            a.set_ylabel("")
    fig.suptitle(f"Photos grouped by {by[0].lower() + by[1:]}", x=0.012, y=1.02, ha="left", fontsize=10.5, color=MUTED)
    fig.tight_layout(w_pad=2.4)
    return _image(fig, plt)


def fig_agreement(project):
    """Scatter: does a photo whose Mooney keeps its depth for the learned model also keep it for the formula?"""
    plt = _plt(); rows = [r for r in photos(project) if r.get("depth") is not None and r.get("shading") is not None]
    if plt is None or len(rows) < 3:
        return None
    x, y = np.array([r["depth"] for r in rows]), np.array([r["shading"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    lo, hi = min(x.min(), y.min(), -0.1) - 0.05, 1.02
    ax.plot([lo, hi], [lo, hi], color="#c3cad3", linewidth=0.9, zorder=1)
    ax.scatter(x, y, s=26, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
    _style(ax, "Each point is one photo", "3 Learned depth: Mooney vs original", "4 Depth from shading: Mooney vs original")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.text(0.03, 0.97, f"n {len(x)}, correlation {r:.2f}\nline: both systems equal", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    return _image(fig, plt)


# ============================================================================ sentences
CAVEATS = """
**Read these with care**

- Every value comes from a model, not from people. Whether a person recognises a Mooney has not been tested.
- A group with fewer than 10 photos can swing by tens of points from one photo to the next.
- Face brightness partly reflects skin tone, so a difference between brightness groups is also a difference between people.
- The face detector and embedding are also vision system 1, so "easy" and "hard" here mean easy and hard for these models.
- A correlation across photos says the two values move together in this dataset, not that one causes the other.
"""


def statements(project, by):
    """Plain sentences built only from the numbers, each with the number of photos behind it."""
    rows = photos(project)
    if not rows:
        return "No photos yet. Collect the dataset first."
    if not any(r.get(k) is not None for r in rows for k in list(VALUES) + list(SHARES)):
        return "No results yet. Run the vision systems in tab 4, then press Refresh." + "\n" + CAVEATS
    s = []
    o, m, b = (_share(rows, k) for k in ("det_orig", "det_mooney", "det_reblur"))
    if o[1] or m[1]:
        s.append(f"The face detector finds a face in {o[0]} of {o[1]} originals, {m[0]} of {m[1]} Mooneys and {b[0]} of {b[1]} Mooneys that were blurred again.")
    sim, simb = _vals(rows, "sim"), _vals(rows, "sim_reblur")
    if len(sim):
        s.append(f"The face model's similarity between a Mooney and its original averages {sim.mean():.2f} (n {len(sim)}); after re-blurring it is "
                 f"{simb.mean():.2f} (n {len(simb)}). Two photos of one person usually score above about 0.36." if len(simb) else
                 f"The face model's similarity between a Mooney and its original averages {sim.mean():.2f} (n {len(sim)}).")
    vo, vm = _share(rows, "vlm_orig"), _share(rows, "vlm_mooney")
    if vo[1] or vm[1]:
        s.append(f"The language model says a face is present for {vo[0]} of {vo[1]} originals and {vm[0]} of {vm[1]} Mooneys.")
    for k, name in (("depth", "learned depth model"), ("shading", "shading formula")):
        v = _vals(rows, k)
        if len(v):
            s.append(f"For the {name}, the Mooney's depth map correlates {v.mean():.2f} on average with its original's "
                     f"(n {len(v)}, from {v.min():.2f} to {v.max():.2f}).")
    # the chosen grouping, on the measure with the largest spread between adequately sized groups
    gs, missing = groups(rows, by)
    big = [(l, p) for l, p in gs if len(p) >= FEW]
    best = None
    for k in SHARES:
        sh = [(l, *_share(p, k)) for l, p in big]
        sh = [(l, a, n) for l, a, n in sh if n >= FEW]
        if len(sh) >= 2:
            hi, lo = max(sh, key=lambda t: t[1] / t[2]), min(sh, key=lambda t: t[1] / t[2])
            gap = hi[1] / hi[2] - lo[1] / lo[2]
            if best is None or gap > best[0]:
                best = (gap, k, hi, lo)
    if best and best[0] > 0:
        _, k, hi, lo = best
        s.append(f'Grouped by {by[0].lower() + by[1:]}, the largest difference is in "{SHORT[k]}": {100 * hi[1] / hi[2]:.0f} % of the '
                 f"{hi[0].split(' (')[0]} photos ({hi[1]} of {hi[2]}) against {100 * lo[1] / lo[2]:.0f} % of the "
                 f"{lo[0].split(' (')[0]} photos ({lo[1]} of {lo[2]}).")
    elif gs:
        s.append(f"Grouped by {by[0].lower() + by[1:]}, no two groups have at least {FEW} photos with results, so no comparison is stated.")
    if missing and gs:
        s.append(f"{missing} photos could not be placed in a group for {by[0].lower() + by[1:]} because the value is missing.")
    # strongest link between a photo property and a measure
    link = None
    for pk, pname in PROPERTIES.items():
        for mk in ("sim", "sim_reblur", "depth", "shading", "det_reblur", "vlm_mooney"):
            pair = [(float(r[pk]), float(r[mk])) for r in rows if r.get(pk) is not None and r.get(mk) is not None]
            if len(pair) < FEW:
                continue
            x, y = np.array(pair).T
            if x.std() == 0 or y.std() == 0:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if link is None or abs(c) > abs(link[0]):
                link = (c, pname, mk, len(pair))
    if link:
        often = {"det_reblur": "a face is found after re-blurring", "vlm_mooney": "the language model says a face is present",
                 "det_mooney": "a face is found in the Mooney"}
        effect = (f"{often[link[2]]} {'more' if link[0] > 0 else 'less'} often" if link[2] in often
                  else f"{SHORT[link[2]]} is {'higher' if link[0] > 0 else 'lower'}")
        s.append(f"Of the photo properties, {link[1]} is tied most closely to a result: in photos with higher {link[1]}, "
                 f"{effect} (correlation {link[0]:+.2f}, n {link[3]}).")
    agree = agreement_table(project)
    if agree:
        a = agree[0]
        s.append(f"The two results that move together most across photos are {a[0]} and {a[1]} (correlation {a[2]:+.2f}, n {a[3]}).")
    return "\n".join(f"- {t}" for t in s) + "\n" + CAVEATS


# ============================================================================ export
def export(project, by):
    """Write the figures as figures/analysis_*.png and the tables as figures/analysis_*.csv. Returns the file names."""
    out = os.path.join(project, "figures"); os.makedirs(out, exist_ok=True); made = []
    slug = GROUPINGS.get(by, "group")
    for name, img in (("analysis_dataset.png", fig_dataset(project)), ("analysis_systems.png", fig_systems(project)),
                      (f"analysis_groups_{slug}.png", fig_groups(project, by)), ("analysis_agreement.png", fig_agreement(project))):
        if img is not None:
            img.save(os.path.join(out, name)); made.append(name)

    def write(name, head, rows):
        if rows:
            with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh); w.writerow(head); w.writerows(rows)
            made.append(name)

    write("analysis_dataset.csv", ["item", "value"], dataset_table(project))
    head, rows = group_table(project, by); write(f"analysis_groups_{slug}.csv", head, rows)
    write("analysis_agreement.csv", ["measure A", "measure B", "correlation", "photos"], agreement_table(project))
    cols = ["stem", "identity", "license", "turn", "mode", "white", "face_brightness"] + list(VALUES) + ["det_orig"] + list(SHARES) + ["vlm_orig"]
    write("analysis_hardest_easiest.csv", RANK_COLS, ranking(project)[2])
    recs = photos(project)
    write("analysis_per_photo.csv", cols, [[("" if r.get(c) is None else (round(r[c], 4) if isinstance(r[c], float) else r[c])) for c in cols] for r in recs])
    return made


# ============================================================================ saved runs
CURRENT = "current"
RUN_PARTS = ["Mooney", "Result1", "Result2", "Result3", "Result4", "Result5", "Result6", "analysis", "figures", "Subset", "Subsets",
             "3D", "3D_basel_fit", "3D_basel"]                                                             # folders copied into a run
RUN_FILES = ["params.csv", "settings.json", "favorites.json", "subsets.json", "dataset.csv"]              # files copied into a run
# What a run can hold, in the order shown in the app: key -> (label, folders). The Mooneys and RUN_FILES are always kept.
RUN_GROUPS = {
    "results": ("Vision system results and tables", ["Result1", "Result2", "Result3", "Result4", "Result5", "Result6", "analysis"]),
    "figures": ("Exported figures", ["figures"]),
    "subset": ("Subsets", ["Subset", "Subsets"]),
    "3d": ("3D faces from photos", ["3D"]),
    "fit": ("3D faces fitted with the Basel Face Model", ["3D_basel_fit"]),
    "basel": ("3D faces from the Basel Face Model", ["3D_basel"]),
    "photos": ("Original photos", ["Original"]),
}
RUN_DEFAULT = ["results", "figures", "subset", "3d", "fit", "basel"]                              # everything except the photos


def _count(folder):
    return len([f for f in os.listdir(folder) if not f.startswith(".")]) if os.path.isdir(folder) else 0


def used_settings(project):
    """What the Mooneys in a project were actually made with, read from params.csv (not the saved defaults)."""
    rows = _csv(project, "params.csv")

    def distinct(key, fmt=str):
        vals = sorted({r.get(key, "") for r in rows if r.get(key, "") != ""}, key=lambda v: (_num(v) is None, _num(v) or 0, v))
        return ", ".join(fmt(v) for v in vals) if vals else "–"
    pct = lambda v: f"{100 * float(v):g} %" if _num(v) is not None else v
    reached = [100 * float(r["white_fraction"]) for r in rows if _num(r.get("white_fraction")) is not None]
    return {
        "Photos": _count(os.path.join(project, "Original")) or len(rows),
        "Mooneys": _count(os.path.join(project, "Mooney")),
        "Adjusted by hand": sum(1 for r in rows if r.get("mode") == "manual"),
        "Blur σ (px)": distinct("sigma", lambda v: f"{float(v):g}" if _num(v) is not None else v),
        "White target": distinct("white_target", pct),
        "White reached (mean)": f"{np.mean(reached):.1f} %" if reached else "–",
        "Background": distinct("bg_mooney"),
        "Phase-1 gray": distinct("bg_level"),
        "White rule": distinct("white_rule"),
        "Preprocessing": distinct("preprocess"),
        "White share counted on": distinct("stats_on"),
    }


def run_slug(name):
    """'Blur 4, 30 % white' -> 'blur-4-30-white' (safe as a folder name)."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower())).strip("-")[:60]


def _size_mb(folder):
    total = 0
    for root, _dirs, files in os.walk(folder):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e6


def save_run(project, runs_dir, name, include=None, note="", group="", run=None):
    """
    Freeze the current state as runs/<date>_<name>/. Always kept: Mooney/, params.csv, settings.json,
    favorites.json, dataset.csv and run.json (settings, counts, date, what was included, the note).
    include: keys of RUN_GROUPS to copy as well (None = RUN_DEFAULT: everything except the photos).
    note: free text stored in run.json. The photos are listed in run.json whether they are copied or not.
    group: the group of runs this one belongs to; run: a plain description of what the run did (source, stages).
    Both are only written to run.json.
    Returns (folder name, message). An existing run of the same name is kept: the new one gets a number at the end.
    """
    slug = run_slug(name)
    if not slug:
        return None, "Give the run a name first (letters or digits)."
    if not _count(os.path.join(project, "Mooney")):
        return None, "Nothing to save yet: there are no Mooneys."
    include = [k for k in (RUN_DEFAULT if include is None else include) if k in RUN_GROUPS]
    parts = ["Mooney"] + [p for k in RUN_GROUPS if k in include for p in RUN_GROUPS[k][1]]
    base = f"{time.strftime('%Y-%m-%d')}_{slug}"
    folder, k = base, 2
    while os.path.exists(os.path.join(runs_dir, folder)):
        folder = f"{base}-{k}"; k += 1
    dst = os.path.join(runs_dir, folder)
    os.makedirs(dst)
    for part in parts:
        if os.path.isdir(os.path.join(project, part)):
            shutil.copytree(os.path.join(project, part), os.path.join(dst, part))
    for f in RUN_FILES:
        if os.path.exists(os.path.join(project, f)):
            shutil.copy2(os.path.join(project, f), os.path.join(dst, f))
    settings = {}
    try:
        settings = json.load(open(os.path.join(project, "settings.json"), encoding="utf-8"))
    except Exception:                                   # no defaults were saved in this project
        pass
    src = os.path.join(project, "Original")
    info = {"name": str(name).strip(), "folder": folder, "saved": time.strftime("%Y-%m-%d %H:%M"),
            "note": str(note or "").strip(), "group": str(group or "").strip(), "run": run or {}, "included": include,
            "saved_defaults": settings, "used": used_settings(project),
            "counts": {p: _count(os.path.join(dst, p, "Mooney")) if p.startswith("Result") else _count(os.path.join(dst, p))
                       for p in RUN_PARTS},
            "photos": sorted(os.path.splitext(f)[0] for f in os.listdir(src) if not f.startswith(".")) if os.path.isdir(src) else []}
    json.dump(info, open(os.path.join(dst, "run.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    try:                                                    # the scores in a fixed format, and the index of all runs
        write_scores(dst); write_scores_index(runs_dir)
    except Exception:
        pass
    done = [f"system {p[-1]}" for p in RUN_PARTS if p.startswith("Result") and info["counts"][p]]
    has = lambda k: any(os.path.isdir(os.path.join(dst, p)) for p in RUN_GROUPS[k][1])
    kept = [RUN_GROUPS[k][0] for k in include if k != "results" and has(k)]
    empty = [RUN_GROUPS[k][0] for k in include if not has(k)]
    mb = _size_mb(dst)
    msg = (f"Saved as runs/{folder}/ ({mb:.1f} MB): {info['counts']['Mooney']} Mooneys" if mb < 10 else
           f"Saved as runs/{folder}/ ({mb:.0f} MB): {info['counts']['Mooney']} Mooneys")
    msg += (f"; results of {', '.join(done)}" if done else "") + (f"; also: {', '.join(kept)}" if kept else "") + "."
    if empty:
        msg += f" Nothing to copy yet for: {', '.join(empty)}."
    if "results" not in include:
        msg += " Without the result tables this run cannot be used in Compare runs."
    return folder, msg


# ---------------------------------------------------------------------------- scores of a run in a fixed format
SCORES_SCHEMA = "mooney-studio.scores/1"
SCORES_FILE, INDEX_JSON, INDEX_CSV = "scores.json", "scores_index.json", "scores_index.csv"
_SETS_3D = ["3D", "3D_basel_fit", "3D_basel"]


def _r4(x):
    return None if x is None else round(float(x), 4)


def _stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": _r4(np.mean(v)), "median": _r4(np.median(v)), "min": _r4(min(v)), "max": _r4(max(v)), "n": len(v)}


def _depth_block(rows):
    pair = [r for r in rows if r.get("depth_correlation_original_vs_mooney") not in (None, "")]
    if not pair:
        return None
    return {"n": len(pair), "mooney_vs_original": _stats([_num(r.get("depth_correlation_original_vs_mooney")) for r in pair]),
            "mooney_vs_original_centre": _stats([_num(r.get("depth_correlation_centre_crop")) for r in pair])}


def run_scores(path):
    """
    The scores of one run (a saved run folder, or a project) as one dict in a fixed shape, read from the result
    tables the run already holds. Nothing is computed again and nothing is written. A block whose table is absent
    is left out; numbers are numbers, rounded to 4 decimals; null where a value does not exist.
    """
    try:
        j = json.load(open(os.path.join(path, "run.json"), encoding="utf-8"))
    except Exception:
        j = {}
    used = j.get("used") or (used_settings(path) if os.path.exists(os.path.join(path, "params.csv")) else {})
    params = _csv(path, "params.csv")
    out = {"schema": SCORES_SCHEMA,
           "run": {"name": j.get("name", os.path.basename(path.rstrip(os.sep))), "group": j.get("group", ""),
                   "folder": j.get("folder", os.path.basename(path.rstrip(os.sep))), "saved": j.get("saved", ""), "note": j.get("note", ""),
                   "source": (j.get("run") or {}).get("source", "")},
           "settings": used,
           "photos": {"n": len(j["photos"]) if j.get("photos") else (_count(os.path.join(path, "Original")) or None),
                      "mooneys": _count(os.path.join(path, "Mooney")), "adjusted_by_hand": sum(1 for r in params if r.get("mode") == "manual")},
           "systems": {}, "sets_3d": {}}
    f = _csv(path, "analysis", "result1_face.csv")
    if f:
        o = [r for r in f if r.get("condition") == "Original"]; m = [r for r in f if r.get("condition") == "Mooney"]
        out["systems"]["1_face_model"] = {
            "n": len(m) or len(o),
            "face_detected": {"original": sum(1 for r in o if _yes(r.get("face_detected"))), "original_n": len(o),
                              "mooney": sum(1 for r in m if _yes(r.get("face_detected"))), "mooney_n": len(m),
                              "mooney_reblurred": sum(1 for r in m if _yes(r.get("reblur_face_detected")))},
            "similarity_to_own_original": {"mooney": _stats([_num(r.get("similarity_to_own_original")) for r in m]),
                                           "mooney_reblurred": _stats([_num(r.get("reblur_similarity_to_own_original")) for r in m])}}
    v = _csv(path, "analysis", "result2_vlm.csv")
    if v:
        o = [r for r in v if r.get("condition") == "Original"]; m = [r for r in v if r.get("condition") == "Mooney"]
        out["systems"]["2_vision_language_model"] = {
            "n_original": len(o), "n_mooney": len(m), "model": v[0].get("model", ""),
            "face_present_yes": {"original": sum(1 for r in o if r.get("face_present") == "yes"),
                                 "mooney": sum(1 for r in m if r.get("face_present") == "yes")}}
    for key, name in (("3_learned_depth", "result3_depth.csv"), ("4_depth_from_shading", "result4_shading.csv")):
        block = _depth_block(_csv(path, "analysis", name))
        if block:
            out["systems"][key] = block
    for sub in _SETS_3D:
        if os.path.isdir(os.path.join(path, sub, "Original")):
            block = _scores_3d(path, sub)
            if block:
                out["sets_3d"][sub] = block
    return out


def _scores_3d(path, sub):
    """The block of one 3D set: counts, faces found, depth against the truth, and the same by light direction."""
    import analysis3d as a3                                 # here, because analysis3d imports this module
    try:
        recs = a3.per_render(path, sub)
    except Exception:
        return None
    if not recs:
        return None

    def count(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return {"yes": int(sum(v)), "n": len(v)} if v else None

    def mean(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return _r4(np.mean(v)) if v else None

    def block(rows):
        return {"renders": len(rows),
                "face_detected": {"render": count(rows, "det_render"), "mooney": count(rows, "det_mooney"),
                                  "mooney_reblurred": count(rows, "det_reblur")},
                "similarity_mooney_to_render": mean(rows, "sim_mooney_render"),
                "depth_vs_truth": {"3_learned_depth": {"render": mean(rows, "d3_render"), "mooney": mean(rows, "d3_mooney")},
                                   "4_depth_from_shading": {"render": mean(rows, "d4_render"), "mooney": mean(rows, "d4_mooney")}}}
    lights = []
    for r in recs:
        if r["light"] not in lights:
            lights.append(r["light"])
    out = block(recs)
    out["mooneys"] = _count(os.path.join(path, sub, "Mooney"))
    out["by_light"] = [{"light": l, **block([r for r in recs if r["light"] == l])} for l in lights]
    return out


def has_scores_source(path):
    """Whether a run folder holds any result table to read scores from."""
    return (os.path.isdir(os.path.join(path, "analysis"))
            or any(os.path.isdir(os.path.join(path, sub, "analysis")) for sub in _SETS_3D))


def write_scores(path):
    """Write <run>/scores.json from the run's own tables. Returns the dict. Nothing else in the run is touched."""
    d = run_scores(path)
    tmp = os.path.join(path, SCORES_FILE + ".tmp")
    json.dump(d, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(path, SCORES_FILE))
    return d


def read_scores(path):
    try:
        return json.load(open(os.path.join(path, SCORES_FILE), encoding="utf-8"))
    except Exception:
        return None


def score_row(d):
    """The key numbers of one run as a flat dict: one row of scores_index.csv."""
    s, t = d.get("systems", {}), d.get("sets_3d", {})
    f, v = s.get("1_face_model") or {}, s.get("2_vision_language_model") or {}
    fd, sim = f.get("face_detected") or {}, f.get("similarity_to_own_original") or {}
    row = {"folder": d["run"].get("folder", ""), "name": d["run"].get("name", ""), "group": d["run"].get("group", ""),
           "saved": d["run"].get("saved", ""), "source": d["run"].get("source", ""), "note": d["run"].get("note", ""),
           "blur": (d.get("settings") or {}).get("Blur σ (px)", ""), "white_target": (d.get("settings") or {}).get("White target", ""),
           "background": (d.get("settings") or {}).get("Background", ""),
           "photos": d["photos"].get("n"), "mooneys": d["photos"].get("mooneys"), "adjusted_by_hand": d["photos"].get("adjusted_by_hand"),
           "face_original": fd.get("original"), "face_mooney": fd.get("mooney"), "face_mooney_reblurred": fd.get("mooney_reblurred"),
           "face_n": fd.get("mooney_n"),
           "similarity_mooney": (sim.get("mooney") or {}).get("mean"), "similarity_mooney_reblurred": (sim.get("mooney_reblurred") or {}).get("mean"),
           "vlm_face_original": (v.get("face_present_yes") or {}).get("original"), "vlm_face_mooney": (v.get("face_present_yes") or {}).get("mooney"),
           "vlm_n_mooney": v.get("n_mooney"),
           "depth3_mooney_vs_original": ((s.get("3_learned_depth") or {}).get("mooney_vs_original") or {}).get("mean"),
           "depth4_mooney_vs_original": ((s.get("4_depth_from_shading") or {}).get("mooney_vs_original") or {}).get("mean")}
    for sub in _SETS_3D:
        b = t.get(sub) or {}
        dv = b.get("depth_vs_truth") or {}
        row.update({f"{sub}_renders": b.get("renders"),
                    f"{sub}_depth3_truth_render": (dv.get("3_learned_depth") or {}).get("render"),
                    f"{sub}_depth3_truth_mooney": (dv.get("3_learned_depth") or {}).get("mooney"),
                    f"{sub}_depth4_truth_render": (dv.get("4_depth_from_shading") or {}).get("render"),
                    f"{sub}_depth4_truth_mooney": (dv.get("4_depth_from_shading") or {}).get("mooney"),
                    f"{sub}_face_reblurred": ((b.get("face_detected") or {}).get("mooney_reblurred") or {}).get("yes")})
    return row


def write_scores_index(runs_dir):
    """Rewrite runs/scores_index.json and runs/scores_index.csv from the scores.json of every saved run."""
    rows = []
    for d in list_runs(runs_dir):
        sc = read_scores(os.path.join(runs_dir, d))
        if sc:
            rows.append(score_row(sc))
    if not os.path.isdir(runs_dir):
        return rows
    json.dump({"schema": SCORES_SCHEMA + "/index", "written": time.strftime("%Y-%m-%d %H:%M"), "runs": rows},
              open(os.path.join(runs_dir, INDEX_JSON), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(os.path.join(runs_dir, INDEX_CSV), "w", newline="", encoding="utf-8") as fh:
        fields = list(dict.fromkeys(k for r in rows for k in r)) or ["folder"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def backfill_scores(runs_dir):
    """Write scores.json for every saved run that has result tables but no scores.json yet, then the index.
    Returns the folders that were filled. No other file of a run is changed."""
    done = []
    for d in list_runs(runs_dir):
        path = os.path.join(runs_dir, d)
        if not os.path.exists(os.path.join(path, SCORES_FILE)) and has_scores_source(path):
            try:
                write_scores(path); done.append(d)
            except Exception:                               # a run with unusual tables: leave it without scores
                pass
    if done or not os.path.exists(os.path.join(runs_dir, INDEX_JSON)):
        try:
            write_scores_index(runs_dir)
        except Exception:                                   # the index is a convenience; a page must not fail over it
            pass
    return done


def list_runs(runs_dir):
    """Names of the saved runs (folders in runs/ that hold result tables), oldest first."""
    if not os.path.isdir(runs_dir):
        return []
    return sorted(d for d in os.listdir(runs_dir)
                  if os.path.isdir(os.path.join(runs_dir, d, "analysis")) or os.path.exists(os.path.join(runs_dir, d, "run.json")))


def _run_path(project, runs_dir, name):
    return project if name == CURRENT else os.path.join(runs_dir, name)


def compare_runs(project, runs_dir, names):
    """
    Put runs side by side. `names` are folder names from list_runs() and/or "current".
    Returns (headers, rows, settings_headers, settings_rows):
      rows           the Results summary with one column per run: for every measure a row for the original, the
                     Mooney and the re-blurred Mooney (rows that are empty in every run are left out);
      settings_rows  what each run's Mooneys were made with (from its params.csv) and when it was saved.
    """
    import results as rs
    names = [n for n in names or [] if n == CURRENT or os.path.isdir(os.path.join(runs_dir, n))]
    if not names:
        return ["system", "measure", "image"], [], ["setting"], []
    tables = {n: {(r["system"], r["measure"]): r for r in rs.summary(_run_path(project, runs_dir, n))} for n in names}
    keys = list(dict.fromkeys(k for n in names for k in tables[n]))
    headers = ["system", "measure", "image"] + names
    rows = []
    for k in keys:
        for cond, label in (("Original", "Original"), ("Mooney", "Mooney"), ("Mooney, re-blurred", "Mooney, re-blurred")):
            cells = [tables[n][k][cond] if k in tables[n] else "—" for n in names]
            if any(c != "—" for c in cells):
                rows.append([k[0], k[1], label] + cells)
    used, saved = {}, {}
    for n in names:
        path = _run_path(project, runs_dir, n)
        used[n] = used_settings(path)
        saved[n] = "not saved (live project)" if n == CURRENT else "–"
        if n != CURRENT and os.path.exists(os.path.join(path, "run.json")):
            try:
                info = json.load(open(os.path.join(path, "run.json"), encoding="utf-8"))
                saved[n] = info.get("saved", "–")
                used[n]["Photos"] = len(info.get("photos") or []) or used[n]["Photos"]
            except Exception:
                pass
    srows = [["Saved"] + [saved[n] for n in names]]
    for key in list(used[names[0]]):
        srows.append([key] + [str(used[n].get(key, "–")) for n in names])
    return headers, rows, ["setting"] + names, srows


if __name__ == "__main__":                                  # python analysis.py --scores <runs folder>
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--scores":
        filled = backfill_scores(sys.argv[2])
        rows = write_scores_index(sys.argv[2])
        print(f"scores.json written for {len(filled)} run(s): {', '.join(filled) or 'none were missing'}")
        print(f"{INDEX_JSON} and {INDEX_CSV} list {len(rows)} run(s) in {sys.argv[2]}")
    else:
        print("usage: python analysis.py --scores <runs folder>")
