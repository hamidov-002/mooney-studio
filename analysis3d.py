"""
Analysis of the 3D step: renders of 3D faces under known lights, their Mooneys, and what the vision systems
return for both, compared with what is known exactly for every render.

There are three sources of 3D faces, each in its own folder inside the project:

    3D/             faces reconstructed from your photos (face3d.run_render_set)
    3D_basel_fit/   the Basel Face Model fitted to your photos' landmarks (bfm_fit.run_fit_set)
    3D_basel/       invented faces from the Basel Face Model (bfm.run_render_set)

Both folders have the same layout as the project (Original/, Mooney/, Result1..4/, analysis/, figures/) plus
truth/. The render step fills Original/ and truth/, Batch fills Mooney/, and systems 1, 3 and 4 run on the
folder unchanged. Every function below takes `sub`, the folder name; the default is 3D/.

    score_truth(project)      per render: depth of systems 3 and 4 against the true depth inside the face,
                              for the render and for its Mooney, and (3D/ only) the similarity of the render
                              to the photo   ->  <sub>/analysis/truth_scores.csv
    light_table(project)      one row per light direction, always with n
    fig_truth(project)        depth against truth, by light
    person_sheet(project, s)  render and Mooney of one face under every light
    export(project)           all of the above to <sub>/figures/

    compare_sets(project)     the stimulus sets side by side: photos, reconstructed 3D, Basel fitted 3D, Basel 3D
    compare_lights(project)   the 3D sets by light direction
    fig_compare(project)      one chart of the comparison
    export_compare(project)   the comparison to <project>/figures/

What "truth" means: the depth and the light are exact for the RENDER, because the picture is drawn from that
mesh under that light. In 3D/ the mesh is an estimate from one photo, so nothing there is the true shape of the
real person; the same holds in 3D_basel_fit/, where the Basel model is fitted to the photo's landmarks. In
3D_basel/ the face is invented, so shape, light and pose are all set by us and exact.

No interface code in this file.
"""
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import analysis as an
import bfm_fit
import bfm
import config as cfg
import face3d as f3

SUB = f3.SUB                 # faces reconstructed from photos
BASEL = bfm.SUB              # invented faces from the Basel Face Model
FIT = bfm_fit.SUB            # the Basel Face Model fitted to photos
TURNED = 0.6                 # |head turn| in dataset.csv above which a photo is not suggested (mesh tends to break)
SCORE_FIELDS = ["file", "source", "identity", "light", "mask_pixels", "similarity_render_to_photo",
                "depth3_vs_truth_render", "depth3_vs_truth_mooney", "depth4_vs_truth_render", "depth4_vs_truth_mooney"]
HEADERS = ["Light", "n", "Face found: render", "Face found: Mooney", "Face found: re-blurred",
           "Similarity Mooney to render", "Similarity render to photo",
           "Learned depth vs truth: render", "Learned depth vs truth: Mooney",
           "Shading depth vs truth: render", "Shading depth vs truth: Mooney"]


def root(project, sub=SUB):
    return os.path.join(project, sub)


def has_photos(sub):
    """The reconstructed and the fitted faces come from photos; the invented Basel faces do not."""
    return sub in (SUB, FIT)


def headers(sub=SUB):
    """Column names of the table by light. Sets without source photos have no 'render to photo' column."""
    return HEADERS if has_photos(sub) else [h for h in HEADERS if h != "Similarity render to photo"]


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def renders(project, sub=SUB):
    """Rows of <sub>/truth/renders.csv whose render file exists."""
    ori = os.path.join(root(project, sub), "Original")
    return [r for r in _rows(os.path.join(root(project, sub), "truth", "renders.csv")) if os.path.exists(os.path.join(ori, r["file"]))]


def sources(project, sub=SUB):
    """The rendered faces, in order: stems of the photos (3D/) or names like basel-007 (3D_basel/)."""
    return list(dict.fromkeys(os.path.splitext(r["source"])[0] for r in renders(project, sub)))


def lights_of(project, sub=SUB):
    present = {r["light"] for r in renders(project, sub)}
    return [l for l in f3.LIGHTS if l in present]


def clear_results(project, sub=SUB):
    """Remove the system outputs of a 3D set (Result*/ and analysis/). Called when its Mooneys are made again."""
    import shutil
    r = root(project, sub)
    gone = [d for d in (os.listdir(r) if os.path.isdir(r) else []) if d.startswith("Result") or d == "analysis"]
    for d in gone:
        shutil.rmtree(os.path.join(r, d), ignore_errors=True)
    os.makedirs(os.path.join(r, "analysis"), exist_ok=True)
    return bool(gone)


# ---------------------------------------------------------------------------- which photos
def default_photos(project, n=None, source="Original"):
    """
    Suggested photos for the 3D step: the favourites in their order, at most one photo per person, then filled
    up from the dataset, most frontal first. Strongly turned heads are left out (their mesh has holes).
    Returns file names in <source>/.
    """
    n = n or cfg.PHOTOS_3D
    folder = os.path.join(project, source)
    files = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))} if os.path.isdir(folder) else {}
    meta = {os.path.splitext(r.get("file", ""))[0]: r for r in _rows(os.path.join(project, "dataset.csv"))}

    def person(stem):
        return (meta.get(stem, {}).get("identity") or stem.rsplit("_", 1)[0]).strip().lower()

    def turn(stem):
        v = an._num(meta.get(stem, {}).get("yaw"))
        return abs(v) if v is not None else 0.0

    favs = []
    path = os.path.join(project, "favorites.json")
    if os.path.exists(path):
        try:
            favs = [os.path.splitext(s)[0] for s in json.load(open(path, encoding="utf-8"))]
        except Exception:
            favs = []
    chosen, people = [], set()
    for stem in [s for s in favs if s in files] + sorted((s for s in files if s not in favs), key=turn):
        if len(chosen) >= n:
            break
        if person(stem) in people or turn(stem) > TURNED:
            continue
        chosen.append(files[stem]); people.add(person(stem))
    return chosen


# ---------------------------------------------------------------------------- scoring against the truth
def _corr(a, b, mask):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if a.size < 50 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _depth_png(project, n, cond, stem, shape, sub=SUB):
    """A system's depth picture as float array at the size of the truth, or None."""
    path = os.path.join(root(project, sub), f"Result{n}", cond, stem + ".png")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L")
    if img.size != (shape[1], shape[0]):
        img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def score_truth(project, source="Original", log=print, progress=None, sub=SUB):
    """
    Compare the systems' outputs with what is known for every render, and write <sub>/analysis/truth_scores.csv.

    depthN_vs_truth_*   Pearson correlation, inside the face mask, between the system's depth picture
                        (Result3 / Result4, the saved PNG) and the true depth of the render. 1 = the same relief.
    similarity_render_to_photo   SFace similarity between the render and the photo it was made from. Both are
                        aligned with the landmarks found in the PHOTO (the render has the same geometry);
                        for the fitted set, the render is aligned with its own landmarks.
                        Left empty for the Basel set: its faces are invented, there is no photo.
    """
    rs = renders(project, sub)
    if not rs:
        log("No renders yet - press 1 Render first."); return []
    tru = os.path.join(root(project, sub), "truth")
    have = {n: os.path.isdir(os.path.join(root(project, sub), f"Result{n}", "Original")) for n in (3, 4)}
    fr = None
    if has_photos(sub):
        try:
            import vision as vs
            fr = vs.FaceRec(log=lambda *_: None)
        except Exception as e:                                       # the depth comparison does not need the face model
            log(f"Face model not available ({e}); similarity to the photo is left empty.")
    photo_emb, truth, out = {}, {}, []
    for i, r in enumerate(rs):
        stem = os.path.splitext(r["file"])[0]; src = r["source"]
        if src not in truth:
            depth = np.load(os.path.join(tru, r["depth_file"]))
            mask = np.asarray(Image.open(os.path.join(tru, r["mask_file"])).convert("L")) > 127
            truth[src] = (depth, mask)
            if fr is not None:
                photo = Image.open(os.path.join(project, source, src)).convert("RGB")
                row = fr.detect(photo)
                photo_emb[src] = (row, fr.embed(photo, row) if row is not None else None)
        depth, mask = truth[src]
        rec = {"file": r["file"], "source": src, "identity": r["identity"], "light": r["light"], "mask_pixels": int(mask.sum())}
        if fr is not None and photo_emb[src][1] is not None:
            row, emb = photo_emb[src]
            render = Image.open(os.path.join(root(project, sub), "Original", r["file"])).convert("RGB")
            if sub == FIT:                       # the fitted head is drawn in its own frame, so it is aligned with its own landmarks
                row = fr.detect(render)
            if row is not None:
                rec["similarity_render_to_photo"] = round(float(np.dot(emb, fr.embed(render, row))), 3)
        for n in (3, 4):
            for cond, key in (("Original", "render"), ("Mooney", "mooney")):
                z = _depth_png(project, n, cond, stem, depth.shape, sub) if have[n] else None
                c = None if z is None else _corr(z, depth, mask)
                rec[f"depth{n}_vs_truth_{key}"] = "" if c is None else round(c, 3)
        out.append(rec)
        if progress: progress((i + 1) / len(rs))
    os.makedirs(os.path.join(root(project, sub), "analysis"), exist_ok=True)
    with open(os.path.join(root(project, sub), "analysis", "truth_scores.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS); w.writeheader(); w.writerows(out)

    def mean(key):
        v = [x[key] for x in out if x.get(key) not in ("", None)]
        return f"{np.mean(v):.2f} (n {len(v)})" if v else "–"
    log(f"Compared with the truth · {len(out)} renders")
    log(f"  learned depth vs true depth        render {mean('depth3_vs_truth_render')}   Mooney {mean('depth3_vs_truth_mooney')}")
    log(f"  depth from shading vs true depth   render {mean('depth4_vs_truth_render')}   Mooney {mean('depth4_vs_truth_mooney')}")
    if has_photos(sub):
        log(f"  similarity of the render to the photo   {mean('similarity_render_to_photo')}")
    log(f"Written to {sub}/analysis/truth_scores.csv.")
    return out


# ---------------------------------------------------------------------------- per render, per light
def _system_rows(folder):
    """What systems 1, 3 and 4 wrote for the images of one folder (a project or a 3D set), keyed by image stem."""
    face = {}
    for r in _rows(os.path.join(folder, "analysis", "result1_face.csv")):
        face[(r.get("file"), r.get("condition"))] = r
    pair = {}
    for n, name in ((3, "result3_depth.csv"), (4, "result4_shading.csv")):
        for r in _rows(os.path.join(folder, "analysis", name)):
            if r.get("condition") == "Mooney":
                pair[(r.get("file"), n)] = (an._num(r.get("depth_correlation_original_vs_mooney")), an._num(r.get("depth_correlation_centre_crop")))
    return face, pair


def _system_values(face, pair, stem):
    """The values of one image pair that exist for photos and renders alike."""
    o, m = face.get((stem, "Original")), face.get((stem, "Mooney"))
    return {"det_render": None if o is None else an._yes(o.get("face_detected")),
            "det_mooney": None if m is None else an._yes(m.get("face_detected")),
            "det_reblur": None if m is None else an._yes(m.get("reblur_face_detected")),
            "sim_mooney_render": None if m is None else an._num(m.get("similarity_to_own_original")),
            "pair3": pair.get((stem, 3), (None, None))[0], "pair3_centre": pair.get((stem, 3), (None, None))[1],
            "pair4": pair.get((stem, 4), (None, None))[0], "pair4_centre": pair.get((stem, 4), (None, None))[1]}


def per_render(project, sub=SUB):
    """One record per render with everything known about it (truth scores and the systems), numbers as float or None."""
    face, pair = _system_rows(root(project, sub))
    scores = {os.path.splitext(r["file"])[0]: r for r in _rows(os.path.join(root(project, sub), "analysis", "truth_scores.csv"))}
    out = []
    for r in renders(project, sub):
        stem = os.path.splitext(r["file"])[0]
        o, m, s = face.get((stem, "Original")), face.get((stem, "Mooney")), scores.get(stem, {})
        out.append({
            "file": r["file"], "stem": stem, "source": os.path.splitext(r["source"])[0], "identity": r["identity"], "light": r["light"],
            "head_pose": r.get("head_pose", ""),
            "det_render": None if o is None else an._yes(o.get("face_detected")),
            "det_mooney": None if m is None else an._yes(m.get("face_detected")),
            "det_reblur": None if m is None else an._yes(m.get("reblur_face_detected")),
            "sim_mooney_render": None if m is None else an._num(m.get("similarity_to_own_original")),
            "sim_render_photo": an._num(s.get("similarity_render_to_photo")),
            "d3_render": an._num(s.get("depth3_vs_truth_render")), "d3_mooney": an._num(s.get("depth3_vs_truth_mooney")),
            "d4_render": an._num(s.get("depth4_vs_truth_render")), "d4_mooney": an._num(s.get("depth4_vs_truth_mooney")),
            **{k: v for k, v in _system_values(face, pair, stem).items() if k.startswith("pair")},
        })
    return out


def _count(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return f"{sum(v)} of {len(v)}" if v else "–"


def _mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    if not v:
        return "–"
    return f"{np.mean(v):.2f}" if len(v) == len(rows) else f"{np.mean(v):.2f} (n {len(v)})"


def _light_row(sel, sub):
    """The cells of one table row after the name and n, for the renders in `sel`."""
    return ([_count(sel, "det_render"), _count(sel, "det_mooney"), _count(sel, "det_reblur"), _mean(sel, "sim_mooney_render")]
            + ([_mean(sel, "sim_render_photo")] if has_photos(sub) else [])
            + [_mean(sel, "d3_render"), _mean(sel, "d3_mooney"), _mean(sel, "d4_render"), _mean(sel, "d4_mooney")])


def light_table(project, sub=SUB):
    """(headers, rows): one row per light direction and one for all lights. n = renders behind the row."""
    rows = per_render(project, sub)
    if not rows:
        return headers(sub), []
    table = []
    for name, sel in [(l, [r for r in rows if r["light"] == l]) for l in lights_of(project, sub)] + [("all lights", rows)]:
        table.append([name, len(sel)] + _light_row(sel, sub))
    return headers(sub), table


def status(project, sub=SUB):
    """Counts for the Overview page and the state line of the 3D tab."""
    r = root(project, sub)

    def count(*parts):
        d = os.path.join(r, *parts)
        return len([f for f in os.listdir(d) if not f.startswith(".")]) if os.path.isdir(d) else 0
    rs = renders(project, sub)
    first = rs[0] if rs else {}
    return {"faces": len({x["identity"] for x in rs}) if not has_photos(sub) else len(sources(project, sub)),
            "lights": len(lights_of(project, sub)), "renders": len(rs),
            "poses": list(dict.fromkeys(x.get("head_pose", "") for x in rs)),
            "model": first.get("model_file", ""), "shadows": first.get("cast_shadows", ""),
            "mooneys": count("Mooney"), "systems": [n for n in (1, 3, 4) if count(f"Result{n}", "Mooney")],
            "scored": os.path.exists(os.path.join(r, "analysis", "truth_scores.csv"))}


# ---------------------------------------------------------------------------- figures
def fig_truth(project, sub=SUB):
    """Depth against the true depth by light direction: one panel per depth system, render next to Mooney."""
    plt = an._plt(); rows = per_render(project, sub)
    if plt is None:
        return None
    lights = lights_of(project, sub)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    panels = [("Learned depth (system 3) against the true depth", "d3"), ("Depth from shading (system 4) against the true depth", "d4")]
    every = [r[f"{key}_{cond}"] for r in rows for _, key in panels for cond in ("render", "mooney") if r[f"{key}_{cond}"] is not None]
    bottom = min(-0.05, min(every, default=0.0) - 0.08)                  # the panels share the y axis, so one range for both
    for ax, (title, key) in zip(axes, panels):
        data = [(l, [r[f"{key}_render"] for r in rows if r["light"] == l and r[f"{key}_render"] is not None],
                 [r[f"{key}_mooney"] for r in rows if r["light"] == l and r[f"{key}_mooney"] is not None]) for l in lights]
        if not any(a or b for _, a, b in data):
            an._empty(ax, title, "run systems 3 and 4 on the renders first"); continue
        x = np.arange(len(data)); w = 0.34
        for off, idx, colour, label in ((-w / 2 - 0.02, 1, an.BLUE, "Render"), (w / 2 + 0.02, 2, an.ORANGE, "Mooney of the render")):
            means = [float(np.mean(d[idx])) if d[idx] else 0.0 for d in data]
            ax.bar(x + off, means, width=w, color=colour, linewidth=0, label=label)
            for xi, m, d in zip(x + off, means, data):
                if d[idx]:
                    ax.text(xi, 1.07, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=an.INK)   # one row, clear of the dots
                    ax.scatter(np.full(len(d[idx]), xi), d[idx], s=7, color="white", edgecolor=an.INK, linewidth=0.5, zorder=3)
        ax.axhline(0, color="#b9c1cb", linewidth=0.8)
        ax.set_xticks(x, [f"{l}\nn {max(len(a), len(b))}" for l, a, b in data], fontsize=8.5, color=an.INK)
        ax.set_ylim(bottom, 1.18)
        ax.set_yticks([t for t in (-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0) if t >= bottom])
        an._style(ax, title, "", "correlation inside the face" if ax is axes[0] else "")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", ncol=2, fontsize=8.5, bbox_to_anchor=(0.995, -0.065), handlelength=1.2)
    fig.text(0.008, -0.105, f"{SET_NAMES.get(sub, sub)}. Bars and numbers: mean over the rendered faces. Dots: single renders. "
             "1 = same relief as the mesh the picture was drawn from; 0 = unrelated.", fontsize=8, color=an.MUTED)
    fig.tight_layout(w_pad=2.5)
    return an._image(fig, plt)


def _font(size):
    try:
        from results import _font as f
        return f(size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def person_sheet(project, stem, cell=190, source="Original", sub=SUB):
    """
    One picture for one rendered face: on the left the photo (3D/) or the face without shading (3D_basel/) and
    the true depth, then one column per light with the render above and its Mooney below. PIL image or None.
    """
    rs = [r for r in renders(project, sub) if os.path.splitext(r["source"])[0] == stem]
    if not rs:
        return None
    r3, pad, head, left = root(project, sub), 8, 22, 64
    cols = 1 + len(rs)
    sheet = Image.new("RGB", (left + cols * (cell + pad) + pad, head + 2 * (cell + pad) + pad), "white")
    d = ImageDraw.Draw(sheet); font, small = _font(13), _font(12)

    def put(path, col, row):
        if os.path.exists(path):
            sheet.paste(Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS),
                        (left + pad + col * (cell + pad), head + pad + row * (cell + pad)))

    if has_photos(sub):
        put(os.path.join(project, source, rs[0]["source"]), 0, 0)
    else:
        put(os.path.join(r3, "truth", f"{stem}__texture.png"), 0, 0)
    put(os.path.join(r3, "truth", f"{stem}__depth.png"), 0, 1)
    d.text((left + pad, 4), "photo / true depth" if has_photos(sub) else "no shading / true depth", fill="#1b1f24", font=font)
    for j, r in enumerate(rs, start=1):
        put(os.path.join(r3, "Original", r["file"]), j, 0)
        put(os.path.join(r3, "Mooney", r["file"]), j, 1)
        d.text((left + pad + j * (cell + pad), 4), f"light: {r['light']}", fill="#1b1f24", font=font)
    d.text((6, head + pad + cell // 2 - 8), "render", fill="#5b6672", font=small)
    d.text((6, head + pad + cell + pad + cell // 2 - 8), "Mooney", fill="#5b6672", font=small)
    return sheet


CAVEAT = ("The true depth and the light direction are exact for the render, because the picture is drawn from that mesh "
          "under that light. They are not the true shape of the real person: the mesh is estimated from one photo. "
          "The renders show the face region only, without hair, in one skin colour.")
CAVEAT_BASEL = ("These faces are invented: each is a random sample from the Basel Face Model 2019, a statistical model of face "
                "shape and skin colour, and shows no real person. Shape, light and head pose are set by us, so the true depth "
                "is exact, and cast shadows (the nose shading the cheek) are drawn. The heads have no hair. The model file is "
                "not part of Mooney Studio.")
CAVEAT_FIT = ("The Basel Face Model is fitted to the landmarks of each photo (eyes, nose, mouth, chin), so the head has that "
              "person's overall proportions and head pose, but it stays close to the model's average face: landmarks do not fix "
              "fine detail, and the skin has one colour matched to the photo. The true depth and the light are exact for the "
              "render, an estimate for the person. Cast shadows are drawn; the heads have no hair. Check each fit in "
              "`truth/<photo>__fit.png`. The model file is not part of Mooney Studio.")
SET_NAMES = {None: "Photos", SUB: "Reconstructed 3D", FIT: "Basel fitted 3D", BASEL: "Basel 3D"}


def caveat(sub=SUB):
    return CAVEAT_FIT if sub == FIT else CAVEAT if has_photos(sub) else CAVEAT_BASEL


def export(project, sub=SUB):
    """Write the table, the chart and one sheet per face to <sub>/figures/. Returns the list of files written."""
    fig_dir = os.path.join(root(project, sub), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    headers, rows = light_table(project, sub)
    if rows:
        with open(os.path.join(fig_dir, "3d_by_light.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(headers); w.writerows(rows)
        written.append("3d_by_light.csv")
        recs = per_render(project, sub)
        with open(os.path.join(fig_dir, "3d_per_render.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
        written.append("3d_per_render.csv")
    img = fig_truth(project, sub) if rows else None
    if img is not None:
        img.save(os.path.join(fig_dir, "3d_depth_vs_truth.png")); written.append("3d_depth_vs_truth.png")
    for stem in sources(project, sub):
        sheet = person_sheet(project, stem, sub=sub)
        if sheet is not None:
            sheet.save(os.path.join(fig_dir, f"3d_{stem}.png")); written.append(f"3d_{stem}.png")
    return written


# ---------------------------------------------------------------------------- compare the stimulus sets
SETS = [None, SUB, FIT, BASEL]           # None = the photos of the main project
SETS_3D = [SUB, FIT, BASEL]
NO_TRUTH = "no truth"
# (label, key, kind); kind: n, count, mean, truth (a mean that exists only where the truth is known)
MEASURES = [
    ("Image pairs (original or render, and its Mooney)", None, "n"),
    ("Face found in the original / render", "det_render", "count"),
    ("Face found in the Mooney", "det_mooney", "count"),
    ("Face found in the re-blurred Mooney", "det_reblur", "count"),
    ("Similarity of the Mooney to its original / render (−1 to 1)", "sim_mooney_render", "mean"),
    ("Learned depth: Mooney against original / render, whole picture", "pair3", "mean"),
    ("Learned depth: Mooney against original / render, central half", "pair3_centre", "mean"),
    ("Depth from shading: Mooney against original / render, whole picture", "pair4", "mean"),
    ("Depth from shading: Mooney against original / render, central half", "pair4_centre", "mean"),
    ("Learned depth against the true depth: render", "d3_render", "truth"),
    ("Learned depth against the true depth: Mooney", "d3_mooney", "truth"),
    ("Depth from shading against the true depth: render", "d4_render", "truth"),
    ("Depth from shading against the true depth: Mooney", "d4_mooney", "truth"),
]
COMPARE_NOTE = ("Each cell carries its own n. *Face found* counts images. Similarity is the face model's. The depth rows are "
                "correlations (1 = the same relief, 0 = unrelated). *Mooney against original / render* compares a system's two "
                "depth maps with each other; in the 3D sets both pictures have the same black background, so the whole-picture "
                "value mostly reflects the outline, and the central half is the fairer comparison with the photos. *Against the "
                "true depth* exists only for the 3D sets and is measured inside the face.")


def set_records(project, sub=None):
    """
    One record per image pair of a stimulus set, with the same keys for all three sets.
    sub=None: the photos of the project (every image in Original/ that has a Mooney). Otherwise the renders of
    that 3D set that have a Mooney, with the truth scores in addition.
    """
    folder = project if sub is None else root(project, sub)
    moon = os.path.join(folder, "Mooney")
    has_mooney = lambda stem: os.path.exists(os.path.join(moon, stem + ".png"))
    if sub is not None:
        return [r for r in per_render(project, sub) if has_mooney(r["stem"])]
    ori = os.path.join(folder, "Original")
    stems = sorted(os.path.splitext(f)[0] for f in (os.listdir(ori) if os.path.isdir(ori) else [])
                   if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".") and "__mesh" not in f)
    face, pair = _system_rows(folder)
    blank = {"d3_render": None, "d3_mooney": None, "d4_render": None, "d4_mooney": None, "light": "", "head_pose": ""}
    return [{"stem": s, **_system_values(face, pair, s), **blank} for s in stems if has_mooney(s)]


def _cell(rows, key, kind, sub):
    if kind == "n":
        return str(len(rows)) if rows else "–"
    if kind == "truth" and sub is None:
        return NO_TRUTH
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return "–"
    if kind == "count":
        return f"{sum(v)} of {len(v)} ({100 * sum(v) / len(v):.0f} %)"
    return f"{np.mean(v):.2f} (n {len(v)})"


def compare_sets(project):
    """(headers, rows): one row per measure, one column per stimulus set. Empty rows list if no set has anything."""
    recs = {sub: set_records(project, sub) for sub in SETS}
    head = ["Measure"] + [SET_NAMES[sub] for sub in SETS]
    if not any(recs.values()):
        return head, []
    return head, [[label] + [_cell(recs[sub], key, kind, sub) for sub in SETS] for label, key, kind in MEASURES]


LIGHT_HEADERS = ["Light", "Set", "n"] + [h for h in HEADERS[2:] if h != "Similarity render to photo"]


def compare_lights(project):
    """(headers, rows): for every light direction one row per 3D set, so the sets stand directly under each other."""
    recs = {sub: per_render(project, sub) for sub in SETS_3D}
    present = {r["light"] for rows in recs.values() for r in rows}
    table = []
    for light in [l for l in f3.LIGHTS if l in present] + (["all lights"] if present else []):
        for sub in SETS_3D:
            sel = [r for r in recs[sub] if light == "all lights" or r["light"] == light]
            if sel:
                table.append([light, SET_NAMES[sub], len(sel), _count(sel, "det_render"), _count(sel, "det_mooney"), _count(sel, "det_reblur"),
                              _mean(sel, "sim_mooney_render"), _mean(sel, "d3_render"), _mean(sel, "d3_mooney"),
                              _mean(sel, "d4_render"), _mean(sel, "d4_mooney")])
    return LIGHT_HEADERS, table


def fig_compare(project):
    """
    One chart for the comparison, the sets side by side on the x axis. Colours mean the same as everywhere:
    blue = original or render, orange = Mooney, green = re-blurred Mooney.
    Left: how often the detector finds a face. Middle and right: the two depth systems against the true depth
    (bars, 3D sets only) and the Mooney's depth map against that of its own original or render (diamonds).
    """
    plt = an._plt()
    recs = {sub: set_records(project, sub) for sub in SETS}
    if plt is None or not any(recs.values()):
        return None
    sets = [sub for sub in SETS if recs[sub]]
    x = np.arange(len(sets))
    ticks = [f"{SET_NAMES[sub]}\nn {len(recs[sub])}" for sub in sets]
    mean = lambda sub, key: (lambda v: float(np.mean(v)) if v else None)([r[key] for r in recs[sub] if r.get(key) is not None])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    def bars(ax, series, factor, fmt, slots=None):
        w = 0.8 / (slots or len(series))
        for j, (label, key, colour) in enumerate(series):
            pts = [(xi + (j - ((slots or len(series)) - 1) / 2) * w, mean(sub, key)) for xi, sub in zip(x, sets)]
            pts = [(xi, factor * m) for xi, m in pts if m is not None]
            if pts:
                ax.bar([p[0] for p in pts], [p[1] for p in pts], width=w * 0.88, color=colour, linewidth=0, label=label)
                for xi, h in pts:
                    ax.text(xi, max(h, 0) + 0.02 * factor, fmt.format(h), ha="center", va="bottom", fontsize=8, color=an.INK)

    bars(axes[0], [("original or render", "det_render", an.BLUE), ("Mooney", "det_mooney", an.ORANGE),
                   ("Mooney, re-blurred", "det_reblur", an.GREEN)], 100.0, "{:.0f}")
    axes[0].set_ylim(0, 114); axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_xticks(x, ticks, fontsize=8.5, color=an.INK)
    an._style(axes[0], "Face found by the detector (system 1)", "", "share of images (%)")
    low = 0.0
    for ax, n, title in ((axes[1], 3, "Learned depth (system 3)"), (axes[2], 4, "Depth from shading (system 4)")):
        bars(ax, [("original or render", f"d{n}_render", an.BLUE), ("Mooney", f"d{n}_mooney", an.ORANGE)], 1.0, "{:.2f}", slots=3)
        pair = [(xi, mean(sub, f"pair{n}_centre")) for xi, sub in zip(x, sets)]
        pair = [(xi, m) for xi, m in pair if m is not None]
        if pair:
            ax.scatter([p[0] + 0.8 / 3 for p in pair], [p[1] for p in pair], marker="D", s=26, color="white", edgecolor=an.INK, linewidth=1.1,
                       zorder=3, label="Mooney against its own original or render (central half)")
            for xi, m in pair:
                ax.text(xi + 0.8 / 3, m + 0.045, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=an.INK)
        for xi, sub in zip(x, sets):
            if sub is None:
                ax.text(xi - 0.4 / 3, 0.03, "no true\ndepth", ha="center", va="bottom", fontsize=8, color=an.MUTED)
        vals = [m for sub in sets for m in (mean(sub, f"d{n}_render"), mean(sub, f"d{n}_mooney"), mean(sub, f"pair{n}_centre")) if m is not None]
        low = min([low] + vals)
        ax.axhline(0, color="#b9c1cb", linewidth=0.8)
        ax.set_xticks(x, ticks, fontsize=8.5, color=an.INK); ax.set_xlim(-0.55, len(sets) - 0.45)
        an._style(ax, title, "", "correlation" if n == 3 else "")
    for ax in axes[1:]:
        ax.set_ylim(min(0.0, low - 0.1), 1.14); ax.set_yticks([t for t in (-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0) if t >= min(0.0, low - 0.1)])
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower left", ncol=len(handles), fontsize=8.5, bbox_to_anchor=(0.005, -0.075), handlelength=1.2)
    fig.text(0.008, -0.115, "Bars in the two depth panels: correlation with the true depth inside the face (3D sets only). "
             "All values are means over the image pairs of a set.", fontsize=8, color=an.MUTED)
    fig.tight_layout(w_pad=2.5)
    return an._image(fig, plt)


def export_compare(project):
    """Write the two comparison tables and the chart to <project>/figures/. Returns the list of files written."""
    fig_dir = os.path.join(project, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    for name, (head, rows) in (("compare_sets.csv", compare_sets(project)), ("compare_sets_by_light.csv", compare_lights(project))):
        if rows:
            with open(os.path.join(fig_dir, name), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh); w.writerow(head); w.writerows(rows)
            written.append(name)
    img = fig_compare(project)
    if img is not None:
        img.save(os.path.join(fig_dir, "compare_sets.png")); written.append("compare_sets.png")
    return written
