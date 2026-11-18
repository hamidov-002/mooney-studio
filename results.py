"""
Results: summary table, summary figure and per-image panels, built from the files the vision
systems wrote (ResultN/ and analysis/*.csv). Nothing here runs a model.

  summary(project)        -> rows for the table (one per measure)
  figure(project, path)   -> bar chart "share of images on which each system succeeds" (PNG)
  panel(project, stem)    -> one image: Original and Mooney with every system's output next to them
  detail(project, stem)   -> the text results for one image (identity ranking, VLM answer, ...)
  export(project)         -> figures/summary.png, figures/summary.csv, figures/panel_<stem>.png
"""
import csv, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXT = (".jpg", ".jpeg", ".png")
CONDS = ("Original", "Mooney", "Mooney, re-blurred")
# categorical slots 1-3 of the validated palette (blue, orange, aqua); aqua is below 3:1 on the
# light surface, so every bar carries a visible value label and the table shows the same numbers
COLORS = {"Original": "#2a78d6", "Mooney": "#eb6834", "Mooney, re-blurred": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------- reading
def _csv(project, name):
    p = os.path.join(project, "analysis", name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _true(v):
    return str(v) == "True"


def _share(rows, key):
    """(k, n) = how many rows have key == True, out of how many."""
    return sum(_true(r.get(key)) for r in rows), len(rows)


def _find(folder, stem):
    if os.path.isdir(folder):
        for e in IMG_EXT:
            p = os.path.join(folder, stem + e)
            if os.path.exists(p):
                return p
    return None


def stems(project, source="Original"):
    d = os.path.join(project, source)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and "__mesh" not in f)


# ----------------------------------------------------------------- summary
def measures(project):
    """
    List of measures. Each: {system, measure, kind ('share' | 'value'), values: {condition: (k, n) or float}, note}
    'share' measures are counts of images and go into the figure; 'value' measures are means.
    """
    out = []
    f = _csv(project, "result1_face.csv")
    if f:
        o = [r for r in f if r["condition"] == "Original"]; m = [r for r in f if r["condition"] == "Mooney"]
        out.append({"system": "1 Face model", "measure": "Face detector finds a face", "kind": "share",
                    "values": {"Original": _share(o, "face_detected"), "Mooney": _share(m, "face_detected"),
                               "Mooney, re-blurred": _share(m, "reblur_face_detected")}})
        if any(r.get("has_gallery_mate") not in (None, "") for r in f):      # only when identity matching was run
            om = [r for r in o if _true(r["has_gallery_mate"])]; mm = [r for r in m if _true(r["has_gallery_mate"])]
            k = max([int(r["n_identities"]) for r in f if r.get("n_identities")] or [0])
            out.append({"system": "1 Face model", "measure": f"Identity correct, 1 of {k}", "kind": "share",
                        "chance": 1 / k if k else None,
                        "values": {"Original": _share(om, "top1_correct"), "Mooney": _share(mm, "top1_correct"),
                                   "Mooney, re-blurred": _share(mm, "reblur_top1_correct")},
                        "note": f"chance {100/k:.1f} %; only images whose identity has a second photo" if k else ""})
        sims = [float(r["similarity_to_own_original"]) for r in m if r.get("similarity_to_own_original") not in (None, "")]
        simb = [float(r["reblur_similarity_to_own_original"]) for r in m if r.get("reblur_similarity_to_own_original") not in (None, "")]
        if sims:
            out.append({"system": "1 Face model", "measure": "Embedding similarity to own original (cosine)", "kind": "value",
                        "values": {"Original": 1.0, "Mooney": float(np.mean(sims)), "Mooney, re-blurred": float(np.mean(simb)) if simb else None},
                        "n": len(sims)})
    v = _csv(project, "result2_vlm.csv")
    if v:
        o = [r for r in v if r["condition"] == "Original"]; m = [r for r in v if r["condition"] == "Mooney"]
        model = v[0].get("model", "")
        yes = lambda rows: (sum(r.get("face_present") == "yes" for r in rows), len(rows))
        out.append({"system": f"2 VLM ({model})", "measure": "Model says a face is present", "kind": "share",
                    "values": {"Original": yes(o), "Mooney": yes(m)}})
        if any(r.get("named_correctly") not in (None, "") for r in v):       # only when identity matching was run
            out.append({"system": f"2 VLM ({model})", "measure": "Model names the person correctly", "kind": "share",
                        "values": {"Original": _share(o, "named_correctly"), "Mooney": _share(m, "named_correctly")}})
    d = _csv(project, "result3_depth.csv")
    cs = [float(r["depth_correlation_original_vs_mooney"]) for r in d if r.get("depth_correlation_original_vs_mooney") not in (None, "")]
    if cs:
        out.append({"system": "3 Learned depth", "measure": "Depth map correlation with the original's depth map", "kind": "value",
                    "values": {"Original": 1.0, "Mooney": float(np.mean(cs))}, "n": len(cs),
                    "note": f"range {min(cs):.2f} to {max(cs):.2f}"})
    sh = _csv(project, "result4_shading.csv")
    cs = [float(r["depth_correlation_original_vs_mooney"]) for r in sh if r.get("depth_correlation_original_vs_mooney") not in (None, "")]
    if cs:
        out.append({"system": "4 Depth from shading", "measure": "Depth map correlation with the original's depth map", "kind": "value",
                    "values": {"Original": 1.0, "Mooney": float(np.mean(cs))}, "n": len(cs),
                    "note": f"range {min(cs):.2f} to {max(cs):.2f}"})
    g = _csv(project, "result5_mesh.csv")
    if g:
        o = [r for r in g if r["condition"] == "Original"]; m = [r for r in g if r["condition"] == "Mooney"]
        out.append({"system": "5 3D face mesh", "measure": "A 3D face mesh is obtained", "kind": "share",
                    "values": {"Original": _share(o, "face_found"), "Mooney": _share(m, "face_found"),
                               "Mooney, re-blurred": _share(m, "reblur_face_found")}})
        le = [float(r["landmark_error_interocular"]) for r in m if r.get("landmark_error_interocular") not in (None, "")]
        dc = [float(r["depth_profile_correlation"]) for r in m if r.get("depth_profile_correlation") not in (None, "")]
        which = "Mooney" if any(_true(r["face_found"]) for r in m) else "Mooney, re-blurred"
        if le:
            ref = ""
            p = os.path.join(project, "analysis", "result5_mesh_reference.json")
            if os.path.exists(p):
                ref = f"; between different people's originals it is {json.load(open(p))['depth_profile_correlation_between_different_identities']:.2f}"
            out.append({"system": "5 3D face mesh", "measure": "Landmark error vs the original's mesh (inter-ocular distances)",
                        "kind": "value", "values": {"Original": 0.0, which: float(np.mean(le))}, "n": len(le)})
            out.append({"system": "5 3D face mesh", "measure": "Depth-profile correlation with the original's mesh",
                        "kind": "value", "values": {"Original": 1.0, which: float(np.mean(dc))}, "n": len(dc), "note": "reference" + ref if ref else ""})
    return out


def _fmt(m, cond):
    v = m["values"].get(cond)
    if v is None:
        return "—"
    if m["kind"] == "share":
        k, n = v
        return f"{k} / {n}  ({100*k/n:.0f} %)" if n else "—"
    return f"{v:.2f}"


def summary(project):
    """Rows for the UI table / summary.csv."""
    rows = []
    for m in measures(project):
        n = m.get("n", "")
        rows.append({"system": m["system"], "measure": m["measure"], **{c: _fmt(m, c) for c in CONDS},
                     "n": n, "note": m.get("note", "")})
    return rows


# ----------------------------------------------------------------- figure
def figure(project, path=None):
    """Horizontal grouped bars: share of images (%) per measure and condition. Returns the PNG path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    ms = [m for m in measures(project) if m["kind"] == "share"]
    if not ms:
        return None
    path = path or os.path.join(project, "figures", "summary.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_img = max(n for m in ms for (_, n) in m["values"].values())
    bar_h, gap = 0.2, 0.04
    fig, ax = plt.subplots(figsize=(9.5, 1.3 + 0.95 * len(ms)), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for i, m in enumerate(ms):
        conds = [c for c in CONDS if c in m["values"]]
        for j, c in enumerate(conds):
            k, n = m["values"][c]
            pct = 100 * k / n if n else 0
            y = i + (j - (len(conds) - 1) / 2) * (bar_h + gap)
            ax.barh(y, pct, height=bar_h, color=COLORS[c], edgecolor=SURFACE, linewidth=0, zorder=3,
                    label=c if i == 0 or c not in ax.get_legend_handles_labels()[1] else None)
            x_lab = pct + 1.2
            ch = 100 * m["chance"] if m.get("chance") else None
            if ch is not None and pct - 1 < ch < pct + 16:        # keep the label clear of the chance line
                x_lab = ch + 1.5
            ax.text(x_lab, y, f"{pct:.0f} %  ({k}/{n})", va="center", ha="left", fontsize=8.5, color=INK, zorder=6)
        if m.get("chance"):
            x = 100 * m["chance"]
            ax.plot([x, x], [i - 0.46, i + 0.46], color=INK2, linewidth=1, linestyle=(0, (2, 2)), zorder=5)
            ax.text(x, i - 0.5, "chance", va="bottom", ha="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{m['measure']}\n{m['system']}" for m in ms], fontsize=9, color=INK)
    for t in ax.get_yticklabels():
        t.set_linespacing(1.35)
    ax.invert_yaxis()
    ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of images (%)", fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5, length=0); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(c) for c in CONDS if c in l]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK, handlelength=1.2, columnspacing=1.6, borderaxespad=0.3)
    ax.set_title(f"Where each vision system still works: originals vs Mooney images (up to {n_img} images)",
                 loc="left", pad=30, fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- per-image panel and text
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def panel(project, stem, source="Original", cell=300):
    """Two rows (Original, Mooney) x columns (image, learned depth, depth from shading, 3D mesh). Missing outputs are left out."""
    cols = [("image", {"Original": _find(os.path.join(project, source), stem), "Mooney": _find(os.path.join(project, "Mooney"), stem)}),
            ("3 · learned depth", {c: _find(os.path.join(project, "Result3", c), stem) for c in ("Original", "Mooney")}),
            ("4 · depth from shading", {c: _find(os.path.join(project, "Result4", c), stem) for c in ("Original", "Mooney")}),
            ("5 · 3D face mesh", {c: _find(os.path.join(project, "Result5", c), stem) for c in ("Original", "Mooney")})]
    cols = [(t, d) for t, d in cols if any(d.values())]
    if not cols or not cols[0][1]["Original"]:
        return None
    pad, head, side = 8, 26, 78
    W, H = side + len(cols) * (cell + pad) + pad, head + 2 * (cell + pad) + pad
    img = Image.new("RGB", (W, H), "#ffffff"); dr = ImageDraw.Draw(img); f = _font(14); fs = _font(12)
    for ci, (title, d) in enumerate(cols):
        x = side + pad + ci * (cell + pad)
        dr.text((x, 6), title, fill="#333333", font=f)
        for ri, c in enumerate(("Original", "Mooney")):
            y = head + pad + ri * (cell + pad)
            if d.get(c):
                im = Image.open(d[c]).convert("RGB"); im.thumbnail((cell, cell))
                img.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
                if title.startswith("5") and c == "Mooney":           # say what the mesh render is based on
                    jp = os.path.join(project, "Result5", "Mooney", stem + ".json")
                    shows = json.load(open(jp)).get("render_shows", "") if os.path.exists(jp) else ""
                    note = {"none": "no face found", "Mooney, re-blurred": "no face found in the Mooney; this mesh is from the re-blurred Mooney"}.get(shows, "")
                    if note:
                        words, line, yy = note.split(), "", y + 6
                        for w_ in words + [""]:
                            if not w_ or dr.textlength(line + " " + w_, font=fs) > cell - 12:
                                dr.text((x + 6, yy), line.strip(), fill="#ffffff", font=fs); yy += 15; line = w_
                            else:
                                line += " " + w_
            else:
                dr.rectangle([x, y, x + cell, y + cell], outline="#cccccc"); dr.text((x + 10, y + 10), "not run", fill="#999999", font=f)
    for ri, c in enumerate(("Original", "Mooney")):
        dr.text((8, head + pad + ri * (cell + pad) + cell // 2 - 8), c, fill="#333333", font=f)
    return img


def detail(project, stem):
    """Markdown with the text results for one image."""
    lines = [f"**{stem}**"]
    for c in ("Original", "Mooney"):
        p = os.path.join(project, "Result1", c, stem + ".json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            s = (f"- **1 · face model, {c}:** face detected {'yes' if r['face_detected'] else 'no'}"
                 + (f" (score {r['detector_score']})" if r.get("detector_score") is not None else ""))
            if c == "Mooney":
                s += (f"; similarity to own original {r.get('similarity_to_own_original')}"
                      f"; after re-blur: detected {'yes' if r.get('reblur_face_detected') else 'no'}, "
                      f"similarity {r.get('reblur_similarity_to_own_original')}")
            if "rank_of_true_identity" in r:                     # identity matching was run
                top = ", ".join(f"{i} {v:.2f}" for i, v in r.get("top5", [])[:3])
                s += f"; true identity ranked {r['rank_of_true_identity'] or '— (no second photo)'} of {r['n_identities']} (top: {top})"
            lines.append(s)
    for c in ("Original", "Mooney"):
        p = os.path.join(project, "Result2", c, stem + ".txt")
        if os.path.exists(p):
            t = " ".join(open(p, encoding="utf-8").read().split())
            lines.append(f"- **2 · VLM, {c}:** `{t[:400]}`")
    for r in _csv(project, "result3_depth.csv"):
        if r["file"] == stem and r.get("depth_correlation_original_vs_mooney"):
            lines.append(f"- **3 · learned depth:** correlation between the Mooney's and the original's depth map {r['depth_correlation_original_vs_mooney']}")
    for r in _csv(project, "result4_shading.csv"):
        if r["file"] == stem and r.get("depth_correlation_original_vs_mooney"):
            lines.append(f"- **4 · depth from shading:** correlation between the Mooney's and the original's depth map {r['depth_correlation_original_vs_mooney']}")
    p = os.path.join(project, "Result5", "Mooney", stem + ".json")
    if os.path.exists(p):
        r = json.load(open(p))
        s = f"- **5 · 3D mesh, Mooney:** face found {'yes' if r['face_found'] else 'no'}; after re-blur {'yes' if r['reblur_face_found'] else 'no'}; the render shows: {r['render_shows']}"
        if "landmark_error_interocular" in r:
            s += f"; landmark error {r['landmark_error_interocular']} inter-ocular distances; depth-profile correlation {r['depth_profile_correlation']}"
        lines.append(s)
    return "\n".join(lines)


# ----------------------------------------------------------------- export
def export(project, stems_to_export=None, source="Original"):
    out = os.path.join(project, "figures"); os.makedirs(out, exist_ok=True)
    rows = summary(project)
    made = []
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        made.append("summary.csv")
    if figure(project, os.path.join(out, "summary.png")):
        made.append("summary.png")
    for s in (stems_to_export or []):
        im = panel(project, s, source)
        if im:
            im.save(os.path.join(out, f"panel_{s}.png")); made.append(f"panel_{s}.png")
    return made
