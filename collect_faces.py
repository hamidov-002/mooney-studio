#!/usr/bin/env python3
"""
Collect openly licensed portraits of public figures from Wikimedia Commons.

For each name:
  * ask the Commons API for files in Category:<Name>, plus a search for "<Name> portrait"
  * keep only CC0 / CC BY / CC BY-SA / public-domain files whose title names the person
  * download candidates and detect the face (YuNet through OpenCV; Haar cascade as fallback)
  * keep photos that pass the quality rules listed in collect()
  * choose photos from different occasions with the most different head poses

Writes
  Raw/<id>_<k>.jpg        the photo as downloaded (longer side at most 1024 px)
  Original/<id>_<k>.jpg   512 x 512 crop from hair to collar (what the systems see)
  dataset.csv             identity, file, licence, author, source page, head turn, face share, sharpness, date
  rejected.csv            photos removed by hand; they are never collected again

Usage:  python collect_faces.py --out ./project --names identities.txt --per-identity 2
"""
import argparse, csv, io, os, re, time
import numpy as np
import requests
from PIL import Image, ImageOps

import config as cfg

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks tools to say who they are; put a project page or an e-mail address in COLLECT_CONTACT (config.json)
UA = {"User-Agent": f"MooneyStudio/1.0 ({cfg.COLLECT_CONTACT or 'research tool for two-tone face images'})"}
OK_LICENSE = re.compile(r"(cc0|cc[- ]by(?!-nc)(-sa)?[- ]?\d|public domain|pd-)", re.I)
BAD_LICENSE = re.compile(r"(nc|nd|all rights|copyright)", re.I)


# ----------------------------------------------------------------- Commons
def api(params):
    """Commons API call. On 'too many requests' (429) or a server error it waits and tries again (up to ~2 minutes)."""
    params = {**params, "format": "json", "formatversion": 2}
    for attempt, pause in enumerate((5, 15, 30, 60, 0)):
        r = requests.get(API, params=params, headers=UA, timeout=30)
        if r.ok:
            return r.json()
        if r.status_code not in (429, 500, 502, 503, 504) or not pause:
            break
        try:
            pause = max(pause, min(int(r.headers.get("Retry-After", 0)), 120))
        except ValueError:
            pass
        time.sleep(pause)
    r.raise_for_status()


def category_files(name, limit=60):
    """Files directly in Category:<Name> plus files found by search, deduplicated."""
    titles = []
    cat = "Category:" + name.replace(" ", "_")
    j = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file", "cmlimit": limit})
    titles += [m["title"] for m in j.get("query", {}).get("categorymembers", [])]
    if len(titles) < limit:
        j = api({"action": "query", "list": "search", "srsearch": f'"{name}" portrait', "srnamespace": 6, "srlimit": limit})
        titles += [m["title"] for m in j.get("query", {}).get("search", [])]
    seen, out = set(), []
    for t in titles:
        if t not in seen and t.lower().endswith((".jpg", ".jpeg", ".png")):
            seen.add(t); out.append(t)
    return out[:limit]


def file_info(titles):
    """imageinfo with extmetadata (licence, author, description) and a 1600px-wide download url."""
    out = []
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        j = api({"action": "query", "prop": "imageinfo", "titles": "|".join(chunk),
                 "iiprop": "url|size|extmetadata", "iiurlwidth": 1600})
        for page in j.get("query", {}).get("pages", []):
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                continue
            md = ii.get("extmetadata", {})
            lic = md.get("LicenseShortName", {}).get("value", "") or md.get("License", {}).get("value", "")
            out.append({
                "title": page["title"], "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl"),
                "description": re.sub("<[^>]+>", " ", md.get("ImageDescription", {}).get("value", ""))[:600],
                "date": _date_of(md.get("DateTimeOriginal", {}).get("value", "")),
                "width": ii.get("width", 0), "height": ii.get("height", 0), "license": lic,
                "author": " ".join(re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "")).split())[:120],
            })
    return out


def licence_ok(lic):
    return bool(OK_LICENSE.search(lic)) and not BAD_LICENSE.search(lic)


# ----------------------------------------------------------------- faces
YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class FaceFinder:
    """YuNet face detector (OpenCV >= 4.5.4). Returns box + 5 landmarks; yaw from nose offset."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        if not os.path.exists(YUNET) or os.path.getsize(YUNET) < 100_000:
            os.makedirs(os.path.dirname(YUNET), exist_ok=True)
            r = requests.get(YUNET_URL, headers=UA, timeout=60); r.raise_for_status()
            open(YUNET, "wb").write(r.content)
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score_threshold=0.7)
        self.kind = "yunet"

    def find(self, img: Image.Image):
        """List of dicts: box (x,y,w,h) in pixels, yaw in [-1,1] (0 = frontal, sign = direction)."""
        bgr = self.cv2.cvtColor(np.asarray(img.convert("RGB")), self.cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        scale = 640 / max(H, W)                          # detector input size; keep it bounded
        small = self.cv2.resize(bgr, (max(1, int(W * scale)), max(1, int(H * scale))))
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.det.detect(small)
        faces = []
        for d in (dets if dets is not None else []):
            x, y, w, h = d[:4] / scale
            re_, le_, nose = d[4:6] / scale, d[6:8] / scale, d[8:10] / scale
            eye_mid = (re_[0] + le_[0]) / 2
            yaw = (nose[0] - eye_mid) / max(w, 1e-6) * 4   # nose offset from eye midline, normalised by face width
            faces.append({"box": (float(x), float(y), float(w), float(h)), "yaw": float(np.clip(yaw, -1, 1)),
                          "score": float(d[14])})
        return faces


def sharpness(img: Image.Image, box):
    x, y, w, h = [int(v) for v in box]
    g = np.asarray(ImageOps.grayscale(img).crop((x, y, x + w, y + h)), dtype=np.float32)
    if g.size < 100:
        return 0.0
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


CROP_MARGIN, CROP_UP = 0.75, 0.08     # crop side = 1.75 x face box; centre moved up by 8 % of the box so hair fits


def crop_square(box, margin=CROP_MARGIN, up=CROP_UP):
    """(left, top, side) of the square crop around a face box: wide enough for hair, chin and neck."""
    x, y, w, h = box
    m = max(w, h); s = m * (1 + margin)
    cx, cy = x + w / 2, y + h / 2 - up * m
    return cx - s / 2, cy - s / 2, s


def fit_square(img_size, box, max_shift=0.12):
    """
    The crop square, slid sideways/vertically by at most max_shift of its side so that it lies inside the photo.
    Returns (left, top, side) or None when it cannot fit - such photos are rejected, so Original/ has no padding.
    """
    left, top, s = crop_square(box); W, H = img_size
    if s > min(W, H):
        return None
    nl, nt = min(max(left, 0), W - s), min(max(top, 0), H - s)
    if abs(nl - left) > max_shift * s or abs(nt - top) > max_shift * s:
        return None
    return nl, nt, s


def crop_fits(img_size, box):
    return fit_square(img_size, box) is not None


def exposure(img: Image.Image, box):
    """Mean and standard deviation of the gray levels inside the face box (0..255)."""
    x, y, w, h = [int(v) for v in box]
    g = np.asarray(ImageOps.grayscale(img).crop((x, y, x + w, y + h)), dtype=np.float32)
    return (float(g.mean()), float(g.std())) if g.size else (0.0, 0.0)


def _plain(t):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", t or "") if not unicodedata.combining(c)).lower()


def _date_of(text):
    """First YYYY-MM-DD in a Commons date field ('' if there is no full date)."""
    m = re.search(r"(\d{4})[-:/.](\d{2})[-:/.](\d{2})", re.sub("<[^>]+>", " ", text or ""))
    return "-".join(m.groups()) if m else ""


def occasion(rec):
    """What identifies the occasion a photo was taken at: its date, its author, and its title without numbers."""
    from urllib.parse import unquote
    title = rec.get("title") or unquote((rec.get("commons_url") or "").split("File:")[-1])
    stem = re.sub(r"\([^)]*\)", "", _plain(os.path.splitext(title.replace("File:", ""))[0]))
    return {"date": rec.get("date", "") or "", "author": _plain(rec.get("author", ""))[:60],
            "stem": re.sub(r"[^a-z]+", "", stem), "thumb": rec.get("_thumb")}


def _thumb(crop: Image.Image) -> np.ndarray:
    """64 x 64 gray version of a face crop, used to recognise the same photograph uploaded twice."""
    return np.asarray(crop.convert("L").resize((64, 64)), dtype=np.float32).ravel()


def same_occasion(a, b):
    """Same day, same photographer with the same title apart from a running number, or visibly the same photograph."""
    if a["date"] and a["date"] == b["date"]:
        return True
    ta, tb = a.get("thumb"), b.get("thumb")
    if ta is not None and tb is not None and np.corrcoef(ta, tb)[0, 1] > 0.85:
        return True                                     # the same photograph, uploaded as two files or two crops
    return bool(a["stem"]) and a["stem"] == b["stem"] and bool(a["author"]) and a["author"] == b["author"]


def pick_poses(good, kept, need):
    """
    Choose `need` photos from `good` so that all photos of the person (chosen + kept) come from different
    occasions and their head poses (yaw) differ as much as possible. Returns fewer than `need` if impossible.
    """
    occ = lambda g: occasion(g[1]); yaw = lambda g: g[2]["yaw"] or 0.0
    taken_occ = [occasion(r) for r in kept]
    taken_yaw = [float(r["yaw"]) for r in kept if r.get("yaw") not in (None, "")]
    chosen = []
    if not kept and need >= 2:                          # start with the most different pair of poses
        pairs = [(abs(yaw(a) - yaw(b)), i, j) for i, a in enumerate(good) for j, b in enumerate(good)
                 if i < j and not same_occasion(occ(a), occ(b))]
        if not pairs:
            return []
        _, i, j = max(pairs); chosen = [good[i], good[j]]
    while len(chosen) < need:
        occs = taken_occ + [occ(c) for c in chosen]; yaws = taken_yaw + [yaw(c) for c in chosen]
        pool = [g for g in good if all(g is not c for c in chosen) and not any(same_occasion(occ(g), o) for o in occs)]
        if not pool:
            break
        chosen.append(max(pool, key=(lambda g: min(abs(yaw(g) - y) for y in yaws)) if yaws else (lambda g: g[4])))
    return chosen[:need]


ARTWORK = re.compile(r"\b(oil|painting|painted|drawing|sketch|caricature|cartoon|illustration|engraving|lithograph|"
                     r"statue|sculpture|waxwork|wax figure|mural|graffiti|stamp|poster|artwork|fan ?art)\b")


def name_match(name, info):
    """
    2 = the file title contains every word of the name (first name and surname), so it is this person
    1 = the title has only the surname, or only the description has the full name (used when 2 is not enough)
    0 = neither, or the file is a painting / drawing / statue rather than a photograph
    """
    words = [w for w in re.split(r"[^a-z0-9]+", _plain(name)) if len(w) > 2 and w not in ("the", "of", "busts", "bust")]
    if not words:
        return 2
    title, desc = _plain(info.get("title", "")), _plain(info.get("description", ""))
    if "bust" not in _plain(name) and ARTWORK.search(title + " " + desc):
        return 0
    if all(w in title for w in words):
        return 2
    if words[-1] in title or all(w in desc for w in words):
        return 1
    return 0


def align_crop(img: Image.Image, box, size=512):
    """Square crop around the face (hair to collar), slid to stay inside the photo if possible, resized to size."""
    left, top, s = fit_square(img.size, box) or crop_square(box)       # padding only for photos that cannot fit
    left, top = int(round(left)), int(round(top))
    crop = Image.new("RGB", (int(s), int(s)), (128, 128, 128))
    crop.paste(img.convert("RGB"), (-left, -top))
    return crop.resize((size, size), Image.LANCZOS)


# ----------------------------------------------------------------- main
def collect(out, names, per_identity=cfg.COLLECT_PER_PERSON, candidates=cfg.COLLECT_CANDIDATES,
            min_face_frac=cfg.COLLECT_MIN_FACE, min_sharp=cfg.COLLECT_MIN_SHARPNESS, log=print, progress=None,
            min_crop_px=cfg.COLLECT_MIN_CROP_PX, bright=tuple(cfg.COLLECT_BRIGHTNESS), min_contrast=cfg.COLLECT_MIN_CONTRAST):
    """
    Quality rules (an image is kept only if all hold):
      licence is open - the file title (or, failing that, its description) names the person - exactly one face -
      the wide crop (hair to neck) lies fully inside the photo - that crop is at least min_crop_px wide in the
      download (no upscaling) - the face is sharp, neither too dark nor too bright, and has contrast -
      the photos of one person come from different occasions (different day, or different photographer/series) -
      no photo is used for two people.
    """
    raw_dir, ori_dir = os.path.join(out, "Raw"), os.path.join(out, "Original")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(ori_dir, exist_ok=True)
    ff = FaceFinder()
    rows, rejected = [], read_rejected(out)
    width = max((len(n) for n in names), default=10)
    tally = {"added": 0, "complete": 0, "skipped": 0, "error": 0}

    def one(name):
        """Collect the missing photos of one person. Returns (kind, text) for the log; kind is a key of `tally`."""
        slug = slug_of(name)
        # photos of this person that are already in the dataset with a full record are kept as they are
        kept = [dict(r, identity=name) for r in _read_dataset(out)
                if os.path.splitext(r.get("file", ""))[0].rsplit("_", 1)[0] == slug and r.get("commons_url")
                and os.path.exists(os.path.join(ori_dir, r["file"]))]
        need = per_identity - len(kept)
        if need <= 0:
            return "complete", f"already has {len(kept)} photos"
        for r in kept:
            r["_thumb"] = _thumb(Image.open(os.path.join(ori_dir, r["file"])))
        skip_pages = rejected | {r["commons_url"] for r in _read_dataset(out) if r.get("commons_url")}   # no photo twice
        kept_occ = [occasion(r) for r in kept]
        try:
            infos = [i for i in file_info(category_files(name, candidates))
                     if licence_ok(i["license"]) and i["page"] not in skip_pages]
        except Exception as e:
            return "error", f"Commons did not answer ({str(e)[:60]})"
        for i in infos:
            i["named"] = name_match(name, i)
        n_open = len(infos)
        infos = [i for i in infos if i["named"] and min(i["width"], i["height"]) >= 700]
        if any(i["named"] == 2 for i in infos):            # titles with the full name exist: use only those,
            infos = [i for i in infos if i["named"] == 2]  # a surname alone could be a relative (wrong person)
        good, why = [], {"no single face": 0, "face too small": 0, "crop leaves photo": 0, "low resolution": 0,
                         "blurred": 0, "too dark/bright/flat": 0, "same occasion": 0, "download failed": 0}
        for info in infos:
            if len([g for g in good if g[1]["named"] == 2]) >= 8 or (info["named"] == 1 and len(good) >= need + 2):
                break
            o = occasion(info)
            if any(same_occasion(o, k) for k in kept_occ) or sum(same_occasion(o, occasion(g[1])) for g in good) >= 2:
                why["same occasion"] += 1; continue      # same event as a photo we have / already 2 candidates from it
            img = None
            for attempt in range(3):
                try:
                    r = requests.get(info["url"], headers=UA, timeout=30)
                    if r.status_code == 429:
                        time.sleep(2 * (attempt + 1)); continue
                    r.raise_for_status()
                    img = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
                    break
                except Exception:
                    pass
            if img is None:
                why["download failed"] += 1; continue
            time.sleep(0.3)   # be polite to Commons
            faces = ff.find(img)
            if len(faces) != 1:
                why["no single face"] += 1; continue
            f = faces[0]; frac = (f["box"][2] * f["box"][3]) / (img.width * img.height)
            if frac < min_face_frac:
                why["face too small"] += 1; continue
            if not crop_fits(img.size, f["box"]):
                why["crop leaves photo"] += 1; continue
            if crop_square(f["box"])[2] < min_crop_px:
                why["low resolution"] += 1; continue
            sh = sharpness(img, f["box"])
            if sh < min_sharp:
                why["blurred"] += 1; continue
            mean, std = exposure(img, f["box"])
            if not (bright[0] <= mean <= bright[1]) or std < min_contrast:
                why["too dark/bright/flat"] += 1; continue
            info["_thumb"] = _thumb(align_crop(img, f["box"]))
            if any(same_occasion(occasion(info), k) for k in kept_occ):
                why["same occasion"] += 1; continue      # the same photograph as one we already have
            good.append((img, info, f, frac, sh))
        detail = (f"{n_open} open-licence files, {len(infos)} name the person, {len(good)} usable"
                  + ("; rejected: " + ", ".join(f"{k} {v}" for k, v in why.items() if v) if any(why.values()) else ""))
        if len([g for g in good if g[1]["named"] == 2]) >= need:
            good = [g for g in good if g[1]["named"] == 2]
        if len(good) < need:
            return "skipped", f"needs {need}, found {len(good)} usable  ({detail})"
        # all photos of a person must come from different occasions; among those, the most different head poses
        chosen = pick_poses(good, kept, need)
        if len(chosen) < need:
            return "skipped", f"usable photos are all from one occasion  ({detail})"
        used = {os.path.splitext(r["file"])[0].rsplit("_", 1)[-1] for r in kept}
        slots = [k for k in range(1, per_identity + len(used) + 1) if str(k) not in used][:need]
        rows.extend(kept)
        for k, (img, info, f, frac, sh) in zip(slots, chosen):
            fname = f"{slug}_{k}.jpg"
            big = img.copy(); big.thumbnail((1024, 1024), Image.LANCZOS); big.save(os.path.join(raw_dir, fname), quality=92)
            align_crop(img, f["box"]).save(os.path.join(ori_dir, fname), quality=92)
            rows.append({"_new": 1, "identity": name, "file": fname, "license": info["license"], "author": info["author"],
                         "commons_url": info["page"], "yaw": "" if f["yaw"] is None else f"{f['yaw']:.2f}",
                         "face_frac": f"{frac:.3f}", "sharpness": f"{sh:.0f}", "date": info.get("date", "")})
        _write_dataset(out, rows)                      # save after every person, so a stopped run loses nothing
        return "added", f"{len(chosen)} photo(s) added" + (f", {len(kept)} kept" if kept else "") + f"  ({detail})"

    marks = {"added": "+", "complete": "=", "skipped": "-", "error": "!"}
    for ni, name in enumerate(names):
        kind, text = one(name)
        tally[kind] += 1
        log(f"{marks[kind]} {name:<{width}}  {text}")
        if progress: progress((ni + 1) / len(names))
        if kind != "complete":
            time.sleep(1.0)                            # short pause between people keeps Commons from rate-limiting
    all_rows = _write_dataset(out, rows)
    new_n = len({r["file"] for r in rows if r.get("_new")})
    log("")
    log(f"Finished: {new_n} new photos for {tally['added']} {'person' if tally['added'] == 1 else 'people'}; {tally['skipped']} skipped, "
        f"{tally['error']} not reached, {tally['complete']} already complete.")
    log(f"Dataset now holds {len(all_rows)} photos.  (+ added, = already complete, - skipped, ! Commons error)")
    return rows


FIELDS = ["identity", "file", "license", "author", "commons_url", "yaw", "face_frac", "sharpness", "date"]


def slug_of(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _read_dataset(out):
    p = os.path.join(out, "dataset.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_dataset(out, rows):
    """Merge `rows` into dataset.csv: existing rows stay if their file still exists and is not rewritten by `rows`."""
    csv_path, ori_dir = os.path.join(out, "dataset.csv"), os.path.join(out, "Original")
    new = {}
    for r in rows:
        new[r["file"]] = r
    old = [r for r in _read_dataset(out) if r.get("file") not in new
           and os.path.exists(os.path.join(ori_dir, r.get("file", "")))]
    allr = old + list(new.values())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(allr)
    return allr


# ----------------------------------------------------------------- manual review: remove an image
REJECT_FIELDS = ["identity", "file", "commons_url", "removed_at"]


def read_rejected(out):
    """Commons pages the user removed by hand; the collector never picks them again."""
    p = os.path.join(out, "rejected.csv")
    if not os.path.exists(p):
        return set()
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["commons_url"] for r in csv.DictReader(fh) if r.get("commons_url")}


def remove_image(out, fname):
    """
    Take one image out of the dataset.
      * Original/<file> and Raw/<file> are moved to _removed/ (recoverable, not deleted)
      * everything derived from it is deleted: Mooney, Subset, Result*/, Original3D renders, params.csv row, favourite
      * its row leaves dataset.csv and its Commons page goes to rejected.csv so it is not collected again
    """
    import json, shutil
    stem = os.path.splitext(fname)[0]
    same = lambda f: os.path.splitext(f)[0] == stem
    row = next((r for r in _read_dataset(out) if r.get("file") == fname), {})
    moved, deleted = 0, 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for rel in ("Original", "Raw"):
        src = os.path.join(out, rel, fname)
        if os.path.exists(src):
            dst_dir = os.path.join(out, "_removed", rel); os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(dst):
                dst = os.path.join(dst_dir, f"{stem}__{stamp}{os.path.splitext(fname)[1]}")
            shutil.move(src, dst); moved += 1
    derived = ["Mooney", os.path.join("Subset", "Original"), os.path.join("Subset", "Mooney")]
    for d in sorted(os.listdir(out)):
        if re.fullmatch(r"Result\d+", d):
            derived += [os.path.join(d, "Original"), os.path.join(d, "Mooney")]
    for rel in derived:
        p = os.path.join(out, rel)
        if os.path.isdir(p):
            for f in os.listdir(p):
                if same(f):
                    os.remove(os.path.join(p, f)); deleted += 1
    pf = os.path.join(out, "figures")
    if os.path.isdir(pf):
        for f in os.listdir(pf):
            if os.path.splitext(f)[0] == "panel_" + stem:
                os.remove(os.path.join(pf, f)); deleted += 1
    p3 = os.path.join(out, "Original3D")
    if os.path.isdir(p3):
        for f in os.listdir(p3):
            if f.startswith(stem + "__"):
                os.remove(os.path.join(p3, f)); deleted += 1
    pp = os.path.join(out, "params.csv")
    if os.path.exists(pp):
        with open(pp, newline="") as fh:
            rd = csv.DictReader(fh); fields = rd.fieldnames or []; prow = list(rd)
        keep = [r for r in prow if not same(r.get("file", ""))]
        if len(keep) != len(prow):
            with open(pp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(keep)
    fp = os.path.join(out, "favorites.json")
    if os.path.exists(fp):
        try:
            favs = list(json.load(open(fp)))
            if stem in favs:
                json.dump(sorted(set(favs) - {stem}), open(fp, "w"), indent=2)
        except Exception:
            pass
    rest = [r for r in _read_dataset(out) if r.get("file") != fname]
    with open(os.path.join(out, "dataset.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rest)
    if row.get("commons_url"):
        rp = os.path.join(out, "rejected.csv"); first = not os.path.exists(rp)
        with open(rp, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=REJECT_FIELDS)
            if first: w.writeheader()
            w.writerow({"identity": row.get("identity", ""), "file": fname, "commons_url": row["commons_url"],
                        "removed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {"file": fname, "identity": row.get("identity", ""), "moved": moved, "deleted": deleted,
            "blocked": bool(row.get("commons_url"))}


def reconcile(out, names_file=None):
    """
    Make dataset.csv agree with Original/: drop rows whose file is gone and add a minimal row for every
    image that has none (e.g. a collection run that was stopped before it wrote the CSV).
    Returns the number of rows added.
    """
    ori_dir, csv_path = os.path.join(out, "Original"), os.path.join(out, "dataset.csv")
    if not os.path.isdir(ori_dir):
        return 0
    files = sorted(f for f in os.listdir(ori_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("file") in files]
    have = {r["file"] for r in rows}
    names = {}
    if names_file and os.path.exists(names_file):
        names = {slug_of(n): n for n in read_names(names_file)}
    added = 0
    for f in files:
        if f in have:
            continue
        slug = os.path.splitext(f)[0].rsplit("_", 1)[0]
        rows.append({"identity": names.get(slug, slug.replace("-", " ").title()), "file": f,
                     "license": "not recorded (collection interrupted) - re-collect this identity", "author": "",
                     "commons_url": "", "yaw": "", "face_frac": "", "sharpness": ""})
        added += 1
    if added or len(rows) != len(have):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    return added


def read_names(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="project"); ap.add_argument("--names", default="identities.txt")
    ap.add_argument("--per-identity", type=int, default=cfg.COLLECT_PER_PERSON)
    ap.add_argument("--candidates", type=int, default=cfg.COLLECT_CANDIDATES)
    a = ap.parse_args()
    collect(a.out, read_names(a.names), a.per_identity, a.candidates)
