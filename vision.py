"""
Vision systems evaluated on originals and Mooneys.

Every system runs on <project>/<source>/* and on <project>/Mooney/* and writes one result file per input
image into ResultN/Original and ResultN/Mooney, a table in <project>/analysis/resultN_*.csv, and its
settings in <project>/analysis/resultN_*_protocol.json.

  Result1  face model          YuNet detector + SFace embedding (OpenCV)
           -> <stem>.json : what the models returned (box, landmarks, confidence, embedding), our settings,
                            and what we calculated from them (detected yes/no, similarity to the original)
  Result2  vision-language     local model through Ollama, fixed prompt, temperature 0
           -> <stem>.txt  : the model's answer exactly as it came back
  Result3  learned depth       Depth Anything V2 (ViT-S, ONNX)
           -> <stem>.png  : depth map, near = bright
  Result4  depth from shading  Tsai & Shah (1994) shape from shading: a formula, nothing learned
           -> <stem>.png  : depth map, near = bright
  Result5  3D face mesh        MediaPipe Face Landmarker (478 3D landmarks)   [later step]
           -> <stem>.png, <stem>.json

VISION_SYSTEMS.md explains every value: which come straight from a model, which settings are ours, and
what we calculate.

Extra condition "Mooney, re-blurred": the Mooney low-pass filtered again (Gaussian, sigma = 8 px at 512 px).
It contains no information that is not in the Mooney, but removes the hard edges.
"""
import base64, csv, io, json, os, random, re, time
import numpy as np
import requests
from PIL import Image, ImageFilter

import config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
IMG_EXT = (".jpg", ".jpeg", ".png")
UA = {"User-Agent": "MooneyStudio/1.0 (student project)"}

YUNET = ("face_detection_yunet_2023mar.onnx",
         "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = ("face_recognition_sface_2021dec.onnx",
         "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
DEPTH = ("depth_anything_v2_vits.onnx",
         "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx")

VLM_PROMPT = (
    "You are shown one image. Answer with a single JSON object and nothing else, using exactly these keys: "
    '"face_present" ("yes" or "no"), '
    '"head_pose" (one of "frontal", "turned left", "turned right"), '
    '"light_direction" (where the light comes from, one of "left", "right", "top", "front", "below", "unknown"), '
    '"description" (one sentence describing what the image shows).'
)
# used only when identity matching is switched on (a later step of the project)
VLM_PROMPT_IDENTITY = VLM_PROMPT.replace(
    '"head_pose"', '"person" (the name of the person if you recognise them, otherwise "unknown"), "head_pose"')
OLLAMA = cfg.OLLAMA_HOST                # config.py, or the OLLAMA_HOST environment variable
REBLUR_SIGMA = cfg.REBLUR_SIGMA         # px, for a 512-px-wide image


DETECT_MIN_CONFIDENCE = cfg.DETECT_MIN_CONFIDENCE   # a detection counts as a face from this confidence (OpenCV's default is 0.9)
DETECT_LONG_SIDE = cfg.DETECT_LONG_SIDE             # px: the image is resized to this before detection
SFS = {"slant_deg": cfg.SFS_SLANT_DEG, "tilt_deg": cfg.SFS_TILT_DEG, "iterations": cfg.SFS_ITERATIONS,
       "smooth_sigma_px": cfg.SFS_SMOOTH_SIGMA_PX, "working_size_px": cfg.SFS_WORKING_SIZE_PX}


def protocol(project, name, settings):
    """Write a system's settings next to its results, so every number can be traced to how it was made."""
    os.makedirs(os.path.join(project, "analysis"), exist_ok=True)
    json.dump(settings, open(os.path.join(project, "analysis", name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def _model_out(row, vec):
    """The raw outputs of the two face models for one image, in a form that can be stored as JSON."""
    det = None
    if row is not None:
        r = [float(v) for v in row]
        det = {"box_x_y_w_h": [round(v, 1) for v in r[0:4]],
               "landmarks_x_y": {"right_eye": [round(r[4], 1), round(r[5], 1)], "left_eye": [round(r[6], 1), round(r[7], 1)],
                                 "nose_tip": [round(r[8], 1), round(r[9], 1)], "mouth_right": [round(r[10], 1), round(r[11], 1)],
                                 "mouth_left": [round(r[12], 1), round(r[13], 1)]},
               "confidence": round(r[14], 4)}
    return {"detector": det, "embedding_128": [round(float(v), 4) for v in vec]}


def reblur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(REBLUR_SIGMA * img.width / 512))


# ----------------------------------------------------------------- common
def model_path(spec, log=print):
    name, url = spec
    p = os.path.join(MODELS, name)
    if not os.path.exists(p) or os.path.getsize(p) < 100_000:
        os.makedirs(MODELS, exist_ok=True)
        log(f"downloading {name} ...")
        r = requests.get(url, headers=UA, timeout=600); r.raise_for_status()
        open(p, "wb").write(r.content)
    return p


def stems(folder):
    if not os.path.isdir(folder):
        return {}
    return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXT) and "__mesh" not in f}


def identities(project):
    """stem -> identity, from dataset.csv (falls back to the file-name prefix)."""
    out = {}
    p = os.path.join(project, "dataset.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("file"):
                    out[os.path.splitext(r["file"])[0]] = r.get("identity") or ""
    return out


def ident_of(stem, table):
    return table.get(stem) or stem.split("__")[0].rsplit("_", 1)[0]


def pairs(project, source):
    """[(stem, original_path, mooney_path or None)]"""
    o, m = stems(os.path.join(project, source)), stems(os.path.join(project, "Mooney"))
    return [(s, p, m.get(s)) for s, p in o.items()]


def out_dirs(project, n):
    d = {c: os.path.join(project, f"Result{n}", c) for c in ("Original", "Mooney")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(project, "analysis"), exist_ok=True)
    return d


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------- 1. face recognition
class FaceRec:
    def __init__(self, log=print):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(model_path(YUNET, log), "", (320, 320), score_threshold=DETECT_MIN_CONFIDENCE)
        self.rec = cv2.FaceRecognizerSF.create(model_path(SFACE, log), "")

    def detect(self, img: Image.Image):
        """Largest face as a raw YuNet row in the image's own pixel coordinates, or None."""
        bgr = self.cv2.cvtColor(np.asarray(img.convert("RGB")), self.cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        sc = DETECT_LONG_SIDE / max(H, W)
        small = self.cv2.resize(bgr, (max(1, int(W * sc)), max(1, int(H * sc))))
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.det.detect(small)
        if dets is None or len(dets) == 0:
            return None
        d = max(dets, key=lambda r: r[2] * r[3]).copy()
        d[:14] /= sc
        return d

    def embed(self, img: Image.Image, row):
        """L2-normalised SFace embedding; aligned with `row` if given, else a plain centre resize."""
        bgr = self.cv2.cvtColor(np.asarray(img.convert("RGB")), self.cv2.COLOR_RGB2BGR)
        if row is not None:
            face = self.rec.alignCrop(bgr, row.reshape(1, -1).astype(np.float32))
        else:
            face = self.cv2.resize(bgr, (112, 112))
        f = self.rec.feature(face).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)


def _rank(probe_vec, probe_stem, true_id, gallery, afc_draws=200, seed=0):
    """gallery: list of (stem, identity, vec). Leave the probe's own source image out."""
    best = {}
    for s, i, v in gallery:
        if s == probe_stem:
            continue
        sim = float(probe_vec @ v)
        if sim > best.get(i, -2):
            best[i] = sim
    order = sorted(best.items(), key=lambda kv: -kv[1])
    has_mate = true_id in best
    rank = 1 + [i for i, _ in order].index(true_id) if has_mate else None
    afc = None
    if has_mate and len(best) >= 8:
        rng = random.Random(f"{seed}-{probe_stem}")
        others = [i for i in best if i != true_id]
        wins = sum(best[true_id] > max(best[i] for i in rng.sample(others, 7)) for _ in range(afc_draws))
        afc = wins / afc_draws
    return order, rank, afc, has_mate


def run_face(project, source="Original", log=print, progress=None, identify=False):
    """
    System 1 - face model (YuNet detector + SFace embedding). For every original and its Mooney:
      face_detected / detector_score      does the detector find a face?
      similarity_to_own_original          cosine similarity of the Mooney's embedding to its original's embedding
      reblur_*                            the same for the Mooney blurred again (sigma 8 px at 512 px)
    identify=True additionally ranks each image against all originals of the other photos (identity matching);
    that is a later step of the project and off by default.
    Writes Result1/{Original,Mooney}/<stem>.json and analysis/result1_face.csv.
    """
    d = out_dirs(project, 1)
    fr = FaceRec(log)
    table = identities(project)
    items = pairs(project, source)
    if not items:
        log(f"No images in {source}/."); return []
    no_mooney = sum(1 for _, _, pm in items if not pm)
    emb, rows_det, t0 = {}, {}, time.time()
    for k, (stem, po, pm) in enumerate(items):
        io_ = Image.open(po).convert("RGB")
        row = fr.detect(io_)
        emb[("Original", stem)] = fr.embed(io_, row)
        rows_det[("Original", stem)] = row
        if pm:
            im = Image.open(pm).convert("RGB")
            row_m = fr.detect(im)                              # can the detector find a face in the Mooney?
            rows_det[("Mooney", stem)] = row_m
            r = row
            if r is not None and im.size != io_.size:          # reuse the original's landmarks, rescaled
                r = r.copy(); r[0:14:2] *= im.width / io_.width; r[1:14:2] *= im.height / io_.height
            emb[("Mooney", stem)] = fr.embed(im, r)
            imb = reblur(im)
            rows_det[("MooneyBlur", stem)] = fr.detect(imb)
            emb[("MooneyBlur", stem)] = fr.embed(imb, r)
        if progress: progress(0.8 * (k + 1) / len(items))
    gallery = [(s, ident_of(s, table), emb[("Original", s)]) for s, _, _ in items] if identify else None
    rows = []
    for k, (stem, po, pm) in enumerate(items):
        true_id = ident_of(stem, table)
        for cond in ("Original", "Mooney"):
            if (cond, stem) not in emb:
                continue
            det = rows_det.get((cond, stem))
            rec = {"file": stem, "condition": cond, "identity": true_id, "system": "YuNet detector + SFace embedding (OpenCV)",
                   "alignment": "own landmarks" if cond == "Original" else "landmarks of the original image",
                   "face_detected": det is not None, "detector_score": None if det is None else round(float(det[14]), 3)}
            if identify:
                order, rank, afc, mate = _rank(emb[(cond, stem)], stem, true_id, gallery)
                rec.update({"has_gallery_mate": mate, "rank_of_true_identity": rank, "n_identities": len(order),
                            "top1_correct": (rank == 1) if mate else None, "afc8_accuracy": afc,
                            "top5": [[i, round(v, 3)] for i, v in order[:5]]})
            rec["model_output"] = _model_out(det, emb[(cond, stem)])
            if cond == "Mooney":
                rec["model_output_reblurred"] = _model_out(rows_det.get(("MooneyBlur", stem)), emb[("MooneyBlur", stem)])
                rec["similarity_to_own_original"] = round(float(emb[("Mooney", stem)] @ emb[("Original", stem)]), 3)
                rec["reblur_face_detected"] = rows_det.get(("MooneyBlur", stem)) is not None
                rec["reblur_similarity_to_own_original"] = round(float(emb[("MooneyBlur", stem)] @ emb[("Original", stem)]), 3)
                if identify:
                    _, rank_b, _, mate = _rank(emb[("MooneyBlur", stem)], stem, true_id, gallery)
                    rec["reblur_rank_of_true_identity"] = rank_b
                    rec["reblur_top1_correct"] = (rank_b == 1) if mate else None
            json.dump(rec, open(os.path.join(d[cond], stem + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            rows.append({k2: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k2, v in rec.items()
                         if not k2.startswith("model_output")})          # raw outputs stay in the JSON files
        if progress: progress(0.8 + 0.2 * (k + 1) / len(items))
    write_csv(os.path.join(project, "analysis", "result1_face.csv"), rows)
    protocol(project, "result1_face_protocol.json", {
        "detector": "YuNet, face_detection_yunet_2023mar.onnx (OpenCV FaceDetectorYN)",
        "embedding": "SFace, face_recognition_sface_2021dec.onnx (OpenCV FaceRecognizerSF), 128 numbers",
        "detector_input": f"image resized so that its longer side is {DETECT_LONG_SIDE} px",
        "face_counts_as_detected_from_confidence": DETECT_MIN_CONFIDENCE,
        "if_several_faces": "the largest box is used",
        "alignment_original": "SFace alignCrop with the image's own five landmarks, to 112 x 112 px",
        "alignment_mooney": "the landmarks found in the ORIGINAL are used, because the detector finds none in a Mooney",
        "alignment_if_original_has_no_face": "plain resize of the whole image to 112 x 112 px",
        "embedding_normalisation": "divided by its length (L2)",
        "similarity": "dot product of the two normalised embeddings = cosine similarity, range -1 to 1",
        "reblurred_condition": f"Gaussian blur of the Mooney, sigma {REBLUR_SIGMA:g} px at 512 px width, then the same steps",
        "identity_matching": bool(identify)})
    pct = lambda xs: f"{np.mean(xs) * 100:3.0f} %" if len(xs) else "  –"
    o = [r for r in rows if r["condition"] == "Original"]; m = [r for r in rows if r["condition"] == "Mooney"]
    log(f"System 1 · face model · {len(o)} originals, {len(m)} Mooneys" + (f" ({no_mooney} originals have no Mooney yet)" if no_mooney else ""))
    log(f"  face detected          original {pct([r['face_detected'] for r in o])}   Mooney {pct([r['face_detected'] for r in m])}"
        f"   Mooney re-blurred {pct([r['reblur_face_detected'] for r in m])}")
    if m:
        log(f"  similarity to original                   Mooney {np.mean([r['similarity_to_own_original'] for r in m]):5.2f}"
            f"   Mooney re-blurred {np.mean([r['reblur_similarity_to_own_original'] for r in m]):5.2f}")
    if identify:
        mo = [r for r in o if r["has_gallery_mate"]]; mm = [r for r in m if r["has_gallery_mate"]]
        log(f"  identity correct       original {pct([bool(r['top1_correct']) for r in mo])}   Mooney {pct([bool(r['top1_correct']) for r in mm])}"
            f"   Mooney re-blurred {pct([bool(r['reblur_top1_correct']) for r in mm])}   ({len(mo)} photos with a second photo of the person)")
    log(f"Finished in {time.time() - t0:.0f} s. Written to Result1/ and analysis/result1_face.csv.")
    return rows


# ----------------------------------------------------------------- 2. VLM (Ollama)
def ollama_models():
    try:
        r = requests.get(OLLAMA + "/api/tags", timeout=3); r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None                                             # server not reachable


def _b64(path, max_side=768):
    img = Image.open(path).convert("RGB"); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower().replace("-", " ")).strip()


def run_vlm(project, model, prompt=VLM_PROMPT, source="Original", log=print, progress=None, resume=True, identify=False,
            only=None):
    """
    System 2 - vision-language model through Ollama. The same prompt for every original and every Mooney,
    temperature 0, seed 0. The raw answer goes to Result2/{Original,Mooney}/<stem>.txt, the parsed fields to
    analysis/result2_vlm.csv, model + prompt to analysis/result2_vlm_protocol.json.
    Answers from an earlier run are reused only if model and prompt are unchanged.
    identify=True also checks whether the model names the person (needs a prompt with a "person" key).
    only: a set of file stems; if given, the model runs on those photos only (it is heavy, so the subset may be enough).
    """
    d = out_dirs(project, 2)
    table = identities(project)
    items = [it for it in pairs(project, source) if only is None or it[0] in only]
    todo = [(s, c, p) for s, po, pm in items for c, p in (("Original", po), ("Mooney", pm)) if p]
    if not todo:
        log(f"No images in {source}/."); return []
    protocol = {"model": model, "prompt": prompt, "options": {"temperature": 0, "seed": 0}}
    ppath = os.path.join(project, "analysis", "result2_vlm_protocol.json")
    try:
        same = json.load(open(ppath)) == protocol
    except Exception:
        same = False
    json.dump(protocol, open(ppath, "w"), indent=1)
    json.dump({"image_sent_as": "PNG, longer side at most 768 px", "answer_format": "JSON, requested through Ollama's format option",
               "read_as": "the four fields are copied from the model's answer as they are; an answer that is not valid JSON counts as no answer",
               "calculated": "share of images whose face_present is 'yes'"},
              open(os.path.join(project, "analysis", "result2_vlm_how_read.json"), "w"), indent=1)
    rows, t0, reused = [], time.time(), 0
    log(f"System 2 · {model} · {len(todo)} images" + (" (subset only)" if only is not None else "")
        + ("" if same or not resume else " · model or prompt changed, earlier answers are not reused"))
    for k, (stem, cond, path) in enumerate(todo):
        out = os.path.join(d[cond], stem + ".txt")
        if resume and same and os.path.exists(out) and os.path.getsize(out) > 0:
            text = open(out, encoding="utf-8").read(); reused += 1
        else:
            r = requests.post(OLLAMA + "/api/generate", timeout=600, json={
                "model": model, "prompt": prompt, "images": [_b64(path)], "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": 0}})
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            open(out, "w", encoding="utf-8").write(text)
        try:
            ans = json.loads(text)
            if not isinstance(ans, dict): ans = {}
        except Exception:
            ans = {}
        true_id = ident_of(stem, table)
        row = {"file": stem, "condition": cond, "identity": true_id, "model": model,
               "face_present": str(ans.get("face_present", "")).lower(),
               "head_pose": ans.get("head_pose", ""), "light_direction": ans.get("light_direction", ""),
               "description": ans.get("description", ""), "parsed": bool(ans)}
        if identify:
            said = str(ans.get("person", ""))
            toks = [t for t in _norm(true_id).split() if len(t) > 2]
            row["person"] = said
            row["named_correctly"] = bool(toks) and (_norm(true_id) in _norm(said) or (toks[-1] in _norm(said).split()))
        rows.append(row)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            el = time.time() - t0; left = el / (k + 1) * (len(todo) - k - 1)
            log(f"  {k + 1:>{len(str(len(todo)))}}/{len(todo)}  {stem} ({cond}): face {row['face_present'] or '?'} - "
                f"{str(row['description'])[:60]}   [about {left / 60:.0f} min left]")
        if progress: progress((k + 1) / len(todo))
    write_csv(os.path.join(project, "analysis", "result2_vlm.csv"), rows)
    pct = lambda xs: f"{np.mean(xs) * 100:3.0f} %" if len(xs) else "  –"
    o = [r for r in rows if r["condition"] == "Original"]; m = [r for r in rows if r["condition"] == "Mooney"]
    log(f"  says a face is present   original {pct([r['face_present'] == 'yes' for r in o])}   Mooney {pct([r['face_present'] == 'yes' for r in m])}")
    if identify:
        log(f"  names the person         original {pct([r['named_correctly'] for r in o])}   Mooney {pct([r['named_correctly'] for r in m])}")
    bad = sum(1 for r in rows if not r["parsed"])
    log(f"Finished in {time.time() - t0:.0f} s" + (f", {reused} answers reused" if reused else "") + (f", {bad} answers could not be read as JSON" if bad else "")
        + ". Written to Result2/ and analysis/result2_vlm.csv.")
    return rows


# ----------------------------------------------------------------- 3. monocular depth
class Depth:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, log=print):
        import onnxruntime as ort
        self.s = ort.InferenceSession(model_path(DEPTH, log), providers=["CPUExecutionProvider"])
        i, o = self.s.get_inputs()[0], self.s.get_outputs()[0]
        self.name, self.size = i.name, int(i.shape[2]) if isinstance(i.shape[2], int) else 518
        # the file must be what the settings below describe: one RGB image in, one depth map out
        if len(i.shape) != 4 or i.shape[1] != 3 or len(o.shape) != 3:
            raise RuntimeError(f"{DEPTH[0]} is not the expected Depth Anything V2 export (input {i.shape}, output {o.shape}). "
                               "Delete the file in models/ and run the system again to download it afresh.")

    def __call__(self, img: Image.Image) -> np.ndarray:
        """
        Relative inverse depth (large = near), float32, at the image's own resolution.
        The steps are those of the authors' own inference code (DepthAnythingV2.infer_image / image2tensor):
        RGB scaled to 0-1, bicubic resize as floating point (so overshoot at hard edges is kept, not clipped),
        ImageNet mean and standard deviation, and the output brought back with bilinear interpolation, corners aligned.
        Our images are square, so the fixed square input keeps the aspect ratio, as the authors' resize does.
        """
        import cv2
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        y = self.s.run(None, {self.name: x})[0].squeeze().astype(np.float32)
        w, h = img.size
        if (h, w) == y.shape:
            return y
        gx = np.linspace(0, y.shape[1] - 1, w, dtype=np.float32)          # corners aligned, as in the authors' code
        gy = np.linspace(0, y.shape[0] - 1, h, dtype=np.float32)
        return cv2.remap(y, np.tile(gx, (h, 1)), np.tile(gy[:, None], (1, w)), interpolation=cv2.INTER_LINEAR)


def _to_png(depth):
    lo, hi = np.percentile(depth, 1), np.percentile(depth, 99)
    return Image.fromarray((np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8))


def _depth_row(stem, cond, system, z):
    """Table row for one depth map: the range of the model's raw values and how the PNG maps onto them."""
    return {"file": stem, "condition": cond, "system": system, "raw_min": round(float(z.min()), 4), "raw_max": round(float(z.max()), 4),
            "png_0_is_raw": round(float(np.percentile(z, 1)), 4), "png_255_is_raw": round(float(np.percentile(z, 99)), 4)}


def _run_depth_like(project, n, system, estimate, csv_name, source, log, progress, label):
    """Shared loop of the two depth systems: a depth map per image, and Mooney-vs-original correlation."""
    d = out_dirs(project, n)
    items = pairs(project, source)
    if not items:
        log(f"No images in {source}/."); return []
    rows, t0 = [], time.time()
    log(f"System {n} · {label} · {len(items)} originals, {sum(1 for _, _, pm in items if pm)} Mooneys")
    for k, (stem, po, pm) in enumerate(items):
        io_ = Image.open(po).convert("RGB")
        do = estimate(io_); _to_png(do).save(os.path.join(d["Original"], stem + ".png"))
        rows.append(_depth_row(stem, "Original", system, do))
        if pm:
            im = Image.open(pm).convert("RGB").resize(io_.size)
            dmo = estimate(im); _to_png(dmo).save(os.path.join(d["Mooney"], stem + ".png"))
            row = _depth_row(stem, "Mooney", system, dmo)
            row["depth_correlation_original_vs_mooney"] = round(float(np.corrcoef(do.ravel(), dmo.ravel())[0, 1]), 3)
            c = slice(io_.height // 4, 3 * io_.height // 4), slice(io_.width // 4, 3 * io_.width // 4)
            row["depth_correlation_centre_crop"] = round(float(np.corrcoef(do[c].ravel(), dmo[c].ravel())[0, 1]), 3)
            rows.append(row)
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            log(f"  {k + 1:>{len(str(len(items)))}}/{len(items)}  {stem}")
        if progress: progress((k + 1) / len(items))
    write_csv(os.path.join(project, "analysis", csv_name), rows)
    cs = [r["depth_correlation_original_vs_mooney"] for r in rows if "depth_correlation_original_vs_mooney" in r]
    if cs:
        log(f"  depth map of the Mooney vs depth map of its original: correlation mean {np.mean(cs):.2f}, "
            f"median {np.median(cs):.2f}, range {min(cs):.2f} to {max(cs):.2f}  ({len(cs)} pairs)")
    log(f"Finished in {time.time() - t0:.0f} s. Written to Result{n}/ and analysis/{csv_name}.")
    return rows


DEPTH_READING = {
    "png": "the raw values clipped to their 1st and 99th percentile and scaled to 0-255; bright = near. "
           "The two percentiles are stored per image in the table (png_0_is_raw, png_255_is_raw).",
    "depth_correlation_original_vs_mooney": "Pearson correlation between all raw values of the Mooney's map and of its original's map",
    "depth_correlation_centre_crop": "the same, on the central half of the image only"}


def run_depth(project, source="Original", log=print, progress=None):
    """System 3 - learned monocular depth (Depth Anything V2, ViT-S)."""
    dm = Depth(log)
    protocol(project, "result3_depth_protocol.json", {
        "model": "Depth Anything V2, ViT-S (the authors' weights, ONNX export by fabio-sim, release v2.0.0), run on the CPU",
        "model_file": DEPTH[1],
        "model_input": f"RGB scaled to 0-1, resized to {dm.size} x {dm.size} px (bicubic, floating point), ImageNet mean and standard "
                       "deviation removed: the steps of the authors' inference code",
        "model_output": "one number per pixel: relative inverse depth, larger = nearer, no physical unit; brought back to the image "
                        "size with bilinear interpolation (corners aligned), as in the authors' code",
        **DEPTH_READING})
    return _run_depth_like(project, 3, "Depth Anything V2 ViT-S (ONNX)", dm, "result3_depth.csv", source, log, progress, "learned depth")


# ----------------------------------------------------------------- 4. depth from shading (a formula, nothing learned)
def shape_from_shading(img: Image.Image, slant_deg=SFS["slant_deg"], tilt_deg=SFS["tilt_deg"], iterations=SFS["iterations"],
                       smooth=SFS["smooth_sigma_px"], size=SFS["working_size_px"]) -> np.ndarray:
    """
    Shape from shading after Tsai & Shah (1994, "Shape from shading using linear approximation").

    Assumptions: the surface is matte with one albedo (Lambertian), lit by one distant light whose direction
    is given by slant (angle from the viewing direction) and tilt (angle in the image plane; 90 deg = from
    above). The brightness of a pixel then depends only on the surface slope there. The slope is written as
    differences of the depth Z, the brightness equation is linearised in Z, and Z is improved by Newton steps.
    Returns relative depth (large = near) at the image's resolution. Nothing here is learned from data.
    """
    from scipy import ndimage
    gray = np.asarray(img.convert("L").resize((size, size), Image.BICUBIC), dtype=np.float64)
    E = ndimage.gaussian_filter(gray, smooth)
    E = E / max(float(E.max()), 1e-6)                                   # brightness 0..1 (albedo assumed constant)
    s, t = np.deg2rad(slant_deg), np.deg2rad(tilt_deg)
    Ps, Qs = np.cos(t) * np.tan(s), np.sin(t) * np.tan(s)               # the light direction as a surface slope
    Z = np.zeros_like(E); Sk = np.full_like(E, 0.01); Wn = 1e-8
    for _ in range(int(iterations)):
        p = Z - np.roll(Z, 1, axis=1); q = Z - np.roll(Z, 1, axis=0)   # slopes from neighbouring depths
        pq = 1 + p * p + q * q; PQs = 1 + Ps * Ps + Qs * Qs
        R = np.maximum(0, (1 + p * Ps + q * Qs) / (np.sqrt(pq) * np.sqrt(PQs)))        # brightness the current shape would have
        f = -(E - R)
        df = -((Ps + Qs) / (np.sqrt(pq) * np.sqrt(PQs)) - (p + q) * (1 + p * Ps + q * Qs) / (np.sqrt(pq ** 3) * np.sqrt(PQs)))
        K = Sk * df / (Wn + df * Sk * df)
        Sk = (1 - K * df) * Sk
        Z = Z + K * f
    return np.asarray(Image.fromarray(Z.astype(np.float32)).resize(img.size, Image.BICUBIC))


def run_shading(project, source="Original", log=print, progress=None):
    """System 4 - depth from shading with a classical formula (no training data, no knowledge of faces)."""
    protocol(project, "result4_shading_protocol.json", {
        "method": "Shape from shading, Tsai & Shah 1994 (linear approximation, Newton iterations)",
        "assumptions": "matte surface with constant albedo; one distant light; no cast shadows",
        "light_direction": f"slant {SFS['slant_deg']:g} deg from the viewing direction, tilt {SFS['tilt_deg']:g} deg (from above)",
        "input": f"grayscale, resized to {SFS['working_size_px']} px, Gaussian smoothing sigma {SFS['smooth_sigma_px']:g} px, divided by its maximum",
        "iterations": SFS["iterations"],
        "output": "one number per pixel: relative depth, larger = nearer, no physical unit; resized back to the image size",
        **DEPTH_READING})
    return _run_depth_like(project, 4, "Shape from shading (Tsai & Shah 1994)", shape_from_shading, "result4_shading.csv",
                           source, log, progress, "depth from shading")


# ----------------------------------------------------------------- 5. 3D face mesh (later step)
MESH_LIGHT = (-0.45, -0.30, 0.84)      # fixed light for all renders: upper left, towards the viewer
EYE_R, EYE_L = 33, 263                 # outer eye corners in the MediaPipe topology


def _mesh_render(verts, size):
    import face3d
    tris = face3d.triangulate(verts)
    v = face3d.smooth_z(verts, tris)
    v, t = face3d.subdivide(v, tris); v, t = face3d.subdivide(v, t)
    return face3d.render(v, t, MESH_LIGHT, size)


def run_mesh(project, source="Original", log=print, progress=None):
    import face3d
    d = out_dirs(project, 5)
    items = pairs(project, source)
    if not items:
        log(f"No images in {source}/."); return []
    rows, t0, zprof = [], time.time(), {}
    table = identities(project)
    for k, (stem, po, pm) in enumerate(items):
        io_ = Image.open(po).convert("RGB")
        vo, _ = face3d.fit(io_)
        if vo is not None:
            zprof[stem] = vo[:, 2].copy()
        (_mesh_render(vo, io_.size) if vo is not None else Image.new("L", io_.size, 0)).save(os.path.join(d["Original"], stem + ".png"))
        rec_o = {"file": stem, "condition": "Original", "system": "MediaPipe Face Landmarker (478 3D landmarks)",
                 "face_found": vo is not None}
        json.dump(rec_o, open(os.path.join(d["Original"], stem + ".json"), "w"), indent=1)
        rows.append(rec_o)
        if pm:
            im = Image.open(pm).convert("RGB").resize(io_.size)
            vm, _ = face3d.fit(im)
            vb = None
            if vm is None:
                vb, _ = face3d.fit(reblur(im))
            used, v = ("Mooney as is", vm) if vm is not None else (("Mooney, re-blurred", vb) if vb is not None else ("none", None))
            (_mesh_render(v, io_.size) if v is not None else Image.new("L", io_.size, 0)).save(os.path.join(d["Mooney"], stem + ".png"))
            rec = {"file": stem, "condition": "Mooney", "system": rec_o["system"],
                   "face_found": vm is not None, "reblur_face_found": (vm is not None) or (vb is not None),
                   "render_shows": used, "reblur_sigma_px": REBLUR_SIGMA}
            if v is not None and vo is not None:
                iod = float(np.linalg.norm(vo[EYE_R, :2] - vo[EYE_L, :2])) + 1e-9
                rec["landmark_error_interocular"] = round(float(np.linalg.norm(v[:, :2] - vo[:, :2], axis=1).mean() / iod), 3)
                rec["depth_profile_correlation"] = round(float(np.corrcoef(v[:, 2], vo[:, 2])[0, 1]), 3)
            json.dump(rec, open(os.path.join(d["Mooney"], stem + ".json"), "w"), indent=1)
            rows.append(rec)
        if (k + 1) % 5 == 0 or k + 1 == len(items):
            log(f"  {k + 1:>{len(str(len(items)))}}/{len(items)}  {stem}")
        if progress: progress((k + 1) / len(items))
    write_csv(os.path.join(project, "analysis", "result5_mesh.csv"), rows)
    ro = [r for r in rows if r["condition"] == "Original"]; rm = [r for r in rows if r["condition"] == "Mooney"]
    log(f"Original n={len(ro)}  face found {np.mean([r['face_found'] for r in ro])*100:.0f}%")
    if rm:
        le = [r["landmark_error_interocular"] for r in rm if "landmark_error_interocular" in r]
        dc = [r["depth_profile_correlation"] for r in rm if "depth_profile_correlation" in r]
        log(f"Mooney   n={len(rm)}  face found {np.mean([r['face_found'] for r in rm])*100:.0f}%  "
            f"after re-blur {np.mean([r['reblur_face_found'] for r in rm])*100:.0f}%")
        if le:
            log(f"where a mesh was obtained (n={len(le)}): landmark error {np.mean(le):.2f} inter-ocular distances, "
                f"depth-profile correlation {np.mean(dc):.2f}")
        # reference level: how similar are the depth profiles of two DIFFERENT people's originals?
        ks = list(zprof)
        base = [float(np.corrcoef(zprof[a], zprof[b])[0, 1]) for i, a in enumerate(ks) for b in ks[i + 1:]
                if ident_of(a, table) != ident_of(b, table)]
        if base:
            log(f"reference: depth-profile correlation between originals of different people {np.mean(base):.2f} "
                f"(the model has a strong face-shape prior, so only the excess over this is informative)")
            json.dump({"depth_profile_correlation_between_different_identities": round(float(np.mean(base)), 3), "n_pairs": len(base)},
                      open(os.path.join(project, "analysis", "result5_mesh_reference.json"), "w"), indent=1)
    log(f"Finished in {time.time() - t0:.0f} s. Written to Result5/ and analysis/result5_mesh.csv.")
    return rows


# ----------------------------------------------------------------- status for the UI
def status():
    have = lambda spec: os.path.exists(os.path.join(MODELS, spec[0]))
    om = ollama_models()
    return {
        "face": "ready" if have(SFACE) and have(YUNET) else "model downloads on first run (39 MB)",
        "vlm": "Ollama not running" if om is None else (f"{len(om)} model(s) installed" if om else "Ollama running, no model pulled"),
        "vlm_models": om or [],
        "depth": "ready" if have(DEPTH) else "model downloads on first run (99 MB)",
        "shading": "ready (a formula, no model file)",
        "mesh": "ready" if os.path.exists(os.path.join(MODELS, "face_landmarker.task")) else "model downloads on first run (4 MB)",
    }
