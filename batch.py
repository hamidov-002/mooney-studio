"""
Batch Mooney generation for the whole dataset, with an optional σ-sweep that picks
the σ maximising a quality score. Writes Mooney/<same filename>.png and params.csv.

Quality score for a candidate Mooney (higher = better):

    quality = fidelity × simplicity

    fidelity    F1 between the Mooney's black/white boundaries and the strong edges
                (top 15 %) of a σ=3-blurred reference of the gray image — does the
                two-tone image follow the real structure, without speckle?
    simplicity  exp(−L / 12), L = boundary length / sqrt(subject area) — are the
                regions few and large, as in a Mooney, rather than a half-tone?

    Also logged, not used for selection:
    integrity   share of white area in the 3 largest components
    face_conf   YuNet face-detector confidence on the Mooney (detectors fail on
                Mooneys — a first result in itself)
"""
import csv, os, time
import numpy as np
from PIL import Image, ImageFilter
from mooney import make_mooney, subject_mask, resize_max

import config as cfg

IMG_EXT = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------- quality
def _dilate(b: np.ndarray, k=5) -> np.ndarray:
    return np.asarray(Image.fromarray((b * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k))) > 0


def _grad_mag(gray_u8: np.ndarray) -> np.ndarray:
    g = gray_u8.astype(np.float32)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def edge_similarity(gray: np.ndarray, mooney: np.ndarray, region=None, ref_sigma=3.0, pct=85) -> float:
    """
    F1 between the Mooney's black/white boundaries and the strong edges of a moderately blurred
    reference image (gradient magnitude above the `pct` percentile inside the region).
    Punishes speckle (boundaries where the reference has no edge) and over-blurring (reference
    edges with no boundary nearby).
    """
    from mooney import gaussian_blur
    mag = _grad_mag(gaussian_blur(gray, ref_sigma))
    vals = mag[region] if region is not None else mag.ravel()
    thr = np.percentile(vals, pct) if vals.size else 0
    ref = mag > thr
    b = mooney > 0
    boundary = b ^ np.asarray(Image.fromarray(b.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))).astype(bool)
    if region is not None:
        ref &= region; boundary &= region
    ref_d, bnd_d = _dilate(ref), _dilate(boundary)
    if boundary.sum() == 0 or ref.sum() == 0:
        return 0.0
    precision = (boundary & ref_d).sum() / boundary.sum()
    recall = (ref & bnd_d).sum() / ref.sum()
    return float(2 * precision * recall / (precision + recall + 1e-9))


def integrity(mooney: np.ndarray, region=None, k=3) -> float:
    """Share of white pixels that belong to the k largest white components (1 = compact, 0 = speckle)."""
    try:
        from scipy import ndimage
    except ImportError:
        return 1.0
    white = mooney > 0
    if region is not None:
        white &= region
    lab, n = ndimage.label(white)
    if n == 0:
        return 0.0
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
    return float(sizes[:k].sum() / sizes.sum())


class FaceConf:
    def __init__(self):
        try:
            from collect_faces import FaceFinder
            self.ff = FaceFinder()
        except Exception:
            self.ff = None

    def __call__(self, mooney: np.ndarray) -> float | None:
        if self.ff is None:
            return None
        faces = self.ff.find(Image.fromarray(mooney).convert("RGB"))
        return max((f["score"] for f in faces), default=0.0)


def simplicity(mooney: np.ndarray, region=None, scale=12.0) -> float:
    b = mooney > 0
    boundary = b ^ np.asarray(Image.fromarray(b.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))).astype(bool)
    area = region.sum() if region is not None else b.size
    if region is not None:
        boundary &= region
    L = boundary.sum() / max(np.sqrt(area), 1.0)
    return float(np.exp(-L / scale))


def quality(gray, mooney, region, face_conf: FaceConf | None):
    e = edge_similarity(gray, mooney, region)
    sp = simplicity(mooney, region)
    i = integrity(mooney, region)
    f = face_conf(mooney) if face_conf else None
    return e * sp, e, sp, i, f


# ----------------------------------------------------------------- batch
def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT) and "__mesh" not in f)


def face_region(img: Image.Image, ff, grow=0.15):
    """Boolean mask of the (slightly enlarged) largest detected face box, or None."""
    faces = ff.find(img) if ff else []
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f["box"][2] * f["box"][3])["box"]
    x0, y0 = max(0, int(x - grow * w)), max(0, int(y - grow * h))
    x1, y1 = min(img.width, int(x + w * (1 + grow))), min(img.height, int(y + h * (1 + grow)))
    reg = np.zeros((img.height, img.width), dtype=bool); reg[y0:y1, x0:x1] = True
    return reg


def run_batch(project, source="Original", sigmas=(cfg.MOONEY_SIGMA,), whites=(cfg.MOONEY_WHITE / 100,),
              bg_level=cfg.MOONEY_BG_LEVEL, separate=cfg.MOONEY_SEPARATE, bg_mooney=cfg.MOONEY_BACKGROUND, use_face_conf=True,
              stats_on="face", white_rule=cfg.MOONEY_WHITE_RULE, bright_white=cfg.MOONEY_BRIGHT_WHITE / 100,
              bright_border=cfg.MOONEY_BRIGHT_BORDER, preprocess=cfg.MOONEY_PREPROCESS, keep_manual=True, log=print, progress=None):
    """
    Make Mooney/<stem>.png for every image in <source>/ and write all parameters to params.csv.

    sigmas / whites   one value each -> the same setting for every photo; several -> every combination is tried
                      per photo and the one with the highest quality score is kept (sweep).
    stats_on          "face": the white share is counted on the person's pixels inside the face box (default);
                      "subject": on the whole person.
    white_rule        "fixed" (default): `whites` for every photo.
                      "brightness": photos whose face brightness is at least `bright_border` get `bright_white`
                      instead (bright, evenly lit faces tend to need more white).
    preprocess        "none" (default) or "clahe".
    keep_manual       photos whose Mooney was saved by hand in the Mooney tab (mode = manual) are left untouched.
    """
    from mooney import face_brightness
    src = os.path.join(project, source)
    dst = os.path.join(project, "Mooney")
    os.makedirs(dst, exist_ok=True)
    files = list_images(src)
    if not files:
        log(f"No images in {source}/ - collect the dataset first."); return []
    old = {}
    pcsv = os.path.join(project, "params.csv")
    if os.path.exists(pcsv):
        with open(pcsv, newline="") as fh:
            old = {r.get("file"): r for r in csv.DictReader(fh)}
    fc = FaceConf() if (use_face_conf or stats_on == "face") else None
    rows, t0, kept, bright_n = [], time.time(), 0, 0
    width = max(len(os.path.splitext(f)[0]) for f in files)
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]; out = stem + ".png"
        tag = f"[{i + 1:>{len(str(len(files)))}}/{len(files)}] {stem:<{width}}"
        if keep_manual and old.get(out, {}).get("mode") == "manual" and os.path.exists(os.path.join(dst, out)):
            rows.append(old[out]); kept += 1
            log(f"{tag}  kept as adjusted by hand")
            if progress: progress((i + 1) / len(files))
            continue
        img = Image.open(os.path.join(src, f)).convert("RGB")
        img = resize_max(img, 1024)
        mask = subject_mask(img) if separate else None
        stat = face_region(img, fc.ff if fc else None) if stats_on == "face" else None
        fb = face_brightness(img, mask, stat)
        is_bright = white_rule == "brightness" and fb is not None and fb >= bright_border
        bright_n += is_bright
        best = None
        for s in sigmas:
            for w in ((bright_white,) if is_bright else whites):
                r = make_mooney(img, sigma=s, white_fraction=w, separate=separate and mask is not None, bg_level=bg_level,
                                mask=mask, bg_mooney=bg_mooney, stat_region=stat, preprocess=preprocess)
                region = (r.mask > 0.5) if r.mask is not None else None
                q, e, sp, it, fconf = quality(r.gray, r.mooney, stat if stat is not None else region, fc if use_face_conf else None)
                if best is None or q > best[0]:
                    best = (q, e, sp, it, fconf, s, w, r)
        q, e, sp, it, fconf, s, w, r = best
        Image.fromarray(r.mooney).save(os.path.join(dst, out))
        rows.append({"file": out, "source": f, "sigma": s, "white_target": w, "white_fraction": f"{r.white_fraction:.4f}",
                     "threshold": f"{r.threshold:.2f}", "bg_level": bg_level, "bg_mooney": bg_mooney, "separated": int(r.mask is not None),
                     "stats_on": "face" if stat is not None else "subject",
                     "face_brightness": "" if fb is None else f"{fb:.0f}", "white_rule": white_rule, "preprocess": preprocess,
                     "quality": f"{q:.3f}", "fidelity": f"{e:.3f}", "simplicity": f"{sp:.3f}", "integrity": f"{it:.3f}",
                     "face_conf": "" if fconf is None else f"{fconf:.3f}", "mode": "batch",
                     "sweep": "sigma=" + ";".join(str(x) for x in sigmas) + " white=" + ";".join(str(x) for x in whites)})
        log(f"{tag}  blur {s:g} · white {r.white_fraction * 100:4.1f} % (target {w * 100:g}) · threshold {r.threshold:3.0f}"
            + ("" if fb is None else f" · face brightness {fb:3.0f}") + ("  [bright face]" if is_bright else "")
            + ("" if stat is not None or stats_on != "face" else "  [no face found: measured on the person]"))
        if progress: progress((i + 1) / len(files))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(pcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    log("")
    log(f"Finished: {len(rows) - kept} Mooneys made in {time.time() - t0:.0f} s"
        + (f", {kept} hand-adjusted ones kept" if kept else "")
        + (f", {bright_n} treated as bright faces ({bright_white * 100:g} % white)" if white_rule == "brightness" else "") + ".")
    log("Written to Mooney/ and params.csv.")
    return rows


def save_subset(project, files, source="Original"):
    """Copy chosen pairs into Subset/Original and Subset/Mooney."""
    import shutil
    so, sm = os.path.join(project, "Subset", "Original"), os.path.join(project, "Subset", "Mooney")
    for d in (so, sm):
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try: os.remove(os.path.join(d, old))
            except OSError: pass
    n = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        src_o = next((os.path.join(project, source, stem + e) for e in IMG_EXT if os.path.exists(os.path.join(project, source, stem + e))), None)
        src_m = os.path.join(project, "Mooney", stem + ".png")
        if src_o and os.path.exists(src_m):
            shutil.copy(src_o, so); shutil.copy(src_m, sm); n += 1
    return n
