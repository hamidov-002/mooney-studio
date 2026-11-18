"""
Publishable results: a copy of a project's or a saved run's numbers without any picture of a person.

    python publish.py <project or run folder> <data folder> "<name>"

What is copied (tables and settings only):
    scores.json, run.json, params.csv, settings.json, subsets.json, favorites.json, dataset.csv
    analysis/*.csv and analysis/*.json                 the per-photo tables of the systems and their protocols
    3D*/analysis/*.csv, 3D*/analysis/*.json, 3D*/truth/renders.csv
    figures/summary.png, figures/compare_sets.png      charts without faces
What is never copied:
    photos, Mooneys, depth maps, renders, panels (pictures of people), the face embeddings in Result1/,
    anything of the Basel Face Model. dataset.csv lists author, licence and source page of every photo, so the
    photo set can be collected again by anyone.
"""
import csv
import json
import os
import re
import shutil
import sys
import time

TOP_FILES = ["scores.json", "run.json", "params.csv", "settings.json", "subsets.json", "favorites.json", "dataset.csv"]
TABLE_EXT = (".csv", ".json")
CHARTS = ["summary.png", "compare_sets.png"]                 # charts known to hold no face
SETS_3D = ["3D", "3D_basel_fit", "3D_basel"]
NEVER = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".npy", ".npz", ".h5", ".txt", ".onnx", ".task")


def slug(name):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower())).strip("-")[:60]


def _copy_tables(src, dst):
    """Copy the .csv and .json files of one folder (not its sub-folders). Returns how many."""
    n = 0
    if os.path.isdir(src):
        for f in sorted(os.listdir(src)):
            if f.lower().endswith(TABLE_EXT) and os.path.isfile(os.path.join(src, f)):
                os.makedirs(dst, exist_ok=True)
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f)); n += 1
    return n


def publish(src, data_dir, name, note=""):
    """
    Write <data_dir>/<slug of name>/ from the project or run folder `src`. An existing folder of that name is
    replaced. Returns (folder, number of files, message).
    """
    if not os.path.isdir(os.path.join(src, "analysis")):
        return None, 0, "Nothing to publish: this folder holds no result tables (analysis/)."
    s = slug(name)
    if not s:
        return None, 0, "Give the published results a name first."
    out = os.path.join(data_dir, s)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    n = 0
    for f in TOP_FILES:
        if os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(out, f)); n += 1
    n += _copy_tables(os.path.join(src, "analysis"), os.path.join(out, "analysis"))
    for sub in SETS_3D:
        n += _copy_tables(os.path.join(src, sub, "analysis"), os.path.join(out, sub, "analysis"))
        r = os.path.join(src, sub, "truth", "renders.csv")
        if os.path.isfile(r):
            os.makedirs(os.path.join(out, sub, "truth"), exist_ok=True)
            shutil.copy2(r, os.path.join(out, sub, "truth", "renders.csv")); n += 1
    for f in CHARTS:
        if os.path.isfile(os.path.join(src, "figures", f)):
            os.makedirs(os.path.join(out, "figures"), exist_ok=True)
            shutil.copy2(os.path.join(src, "figures", f), os.path.join(out, "figures", f)); n += 1
    if not os.path.isfile(os.path.join(out, "scores.json")):            # a project state or an older run: compute them
        try:
            import analysis as an
            json.dump(an.run_scores(src), open(os.path.join(out, "scores.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            n += 1
        except Exception:                                               # no scores function, or tables in an older shape
            pass
    json.dump({"name": str(name).strip(), "published": time.strftime("%Y-%m-%d %H:%M"), "note": str(note or "").strip(),
               "holds": "tables and settings only; no picture of a person, no face embeddings, nothing of the Basel Face Model"},
              open(os.path.join(out, "published.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n += 1
    leaked = check(out)
    if leaked:                                                          # cannot happen with the lists above; guard anyway
        shutil.rmtree(out)
        return None, 0, "Stopped: files that must not be published were found: " + ", ".join(leaked[:5])
    write_index(data_dir)
    return out, n, f"Published {n} files to {os.path.join(os.path.basename(data_dir), s)}/ (tables and settings only, no pictures of people)."


def check(folder):
    """Files in a published folder that must not be there (anything but tables, the two charts and the readme)."""
    bad = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), folder)
            if f.lower().endswith(NEVER) and not (os.path.basename(root) == "figures" and f in CHARTS):
                bad.append(rel)
    return bad


def _key_numbers(folder):
    try:
        s = json.load(open(os.path.join(folder, "scores.json"), encoding="utf-8"))
    except Exception:
        return {}
    sy = s.get("systems", {})
    f1 = sy.get("1_face_model", {})
    det = f1.get("face_detected", {})
    n = f1.get("n_mooney") or f1.get("n") or ""
    mean = lambda d: (d or {}).get("mean", "") if isinstance(d, dict) else ""
    return {"photos": n, "face found in Mooney": det.get("mooney", ""), "after re-blur": det.get("mooney_reblurred", ""),
            "learned depth, Mooney vs original": mean(sy.get("3_learned_depth", {}).get("mooney_vs_original")),
            "depth from shading, Mooney vs original": mean(sy.get("4_depth_from_shading", {}).get("mooney_vs_original")),
            "3D sets": ", ".join(sorted(s.get("sets_3d", {}))) or "–"}


def write_index(data_dir):
    """data/README.md and data/index.csv: one row per published result."""
    rows = []
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d, "published.json")
        if os.path.isfile(p):
            meta = json.load(open(p, encoding="utf-8"))
            rows.append({"folder": d, "name": meta.get("name", d), "published": meta.get("published", ""), "note": meta.get("note", ""),
                         **_key_numbers(os.path.join(data_dir, d))})
    cols = ["folder", "name", "published", "photos", "face found in Mooney", "after re-blur", "learned depth, Mooney vs original",
            "depth from shading, Mooney vs original", "3D sets", "note"]
    with open(os.path.join(data_dir, "index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows([{c: r.get(c, "") for c in cols} for r in rows])
    lines = ["# Results", "",
             "Results of experiments made with Mooney Studio: tables and settings, one folder per run.", "",
             "| Run | Photos | Face found in Mooney | After re-blur | Learned depth | Depth from shading | 3D sets |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| [{r['name']}]({r['folder']}/) | {r.get('photos', '')} | {r.get('face found in Mooney', '')} | {r.get('after re-blur', '')} | "
                     f"{r.get('learned depth, Mooney vs original', '')} | {r.get('depth from shading, Mooney vs original', '')} | {r.get('3D sets', '')} |")
    lines += ["",
              "The two depth columns are the mean correlation between the depth map of a Mooney and that of its original.", "",
              "## What a folder holds", "",
              "- `scores.json`: every score of the run in one file (format in `docs/MANUAL.md`).",
              "- `analysis/`: the per-photo tables of each vision system and the settings each system ran with.",
              "- `params.csv`: the values every Mooney was made with. `settings.json`, `subsets.json`: defaults and subsets.",
              "- `dataset.csv`: person, author, licence and source page of every photo.",
              "- `3D*/`: for the 3D datasets, the scores against the true depth and the list of renders.", "",
              "## What is not here, and why", "",
              "- **No photos, Mooneys, depth maps or renders.** They show real people. Each photo keeps its own licence; `dataset.csv` "
              "links to every source page, so the same photo set can be collected again with the tool.",
              "- **No face embeddings.** The face model's embeddings are biometric data of the people shown.",
              "- **Nothing of the Basel Face Model.** It is licensed to registered users only.", ""]
    open(os.path.join(data_dir, "README.md"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    folder, count, msg = publish(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    print(msg)
