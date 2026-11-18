"""
The Basel Face Model fitted to your photos: a third source of 3D faces, drawn like the invented Basel faces.

For each photo the shape coefficients and the head pose of the Basel Face Model 2019 are fitted to the face's
2D landmarks (found by MediaPipe), so the result is a Basel head shaped like the person in the photo. That head
is then drawn exactly like the invented faces in bfm.py (512 px, z-buffer, ambient light, cast shadows, true depth).

The fit (landmarks only, no pixels):
    camera   scaled orthographic: image point = s * R[:2] @ X + t
    shape    X = mean + basis @ (sd * a), only the first BASEL_FIT_COMPONENTS components
    loop     a few times: (1) s, R, t from the current 3D points by least squares, made a true rotation;
             (2) a by regularised linear least squares on the 2D residuals, with the model's own prior
             a ~ N(0, 1) and an expected landmark error of BASEL_FIT_SIGMA_MM (Tikhonov)
    colour   the model's mean skin colour; optionally one RGB gain so the overall tone matches the photo
    check    mean landmark error in pixels, also as a share of the eye distance; above BASEL_FIT_MAX_ERROR the
             photo is skipped

What is known exactly: the shape of the fitted head (so the depth of every pixel of the render), the light and
the pose. It is NOT the true shape of the person: landmarks fix the overall proportions (face width, jaw, nose
length, eye and mouth position) but not fine detail, and the heads have no hair.

    landmarks_from_photo(img)           MediaPipe landmarks in pixels (needs mediapipe, like face3d.py)
    correspondences(model_path)         which Basel vertex belongs to which MediaPipe landmark
    fit(model, points, ...)             coefficients, pose, fitted vertices and the landmark error
    run_fit_set(project, files, ...)    the set for the app -> <project>/3D_basel_fit/ (same layout as 3D/)

The fitting and drawing need only numpy, h5py and Pillow; MediaPipe is needed only to find the landmarks.
The model file is NOT part of Mooney Studio and may not be redistributed (see bfm.py).
"""
import csv
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import bfm

try:
    import config as cfg
    COMPONENTS, SIGMA_MM = int(cfg.BASEL_FIT_COMPONENTS), float(cfg.BASEL_FIT_SIGMA_MM)
    MAX_ERROR, TONE = float(cfg.BASEL_FIT_MAX_ERROR), bool(cfg.BASEL_FIT_TONE)
except Exception:                                    # run on its own, without the app's settings
    COMPONENTS, SIGMA_MM, MAX_ERROR, TONE = 80, 1.5, 0.15, True

SUB = "3D_basel_fit"                                 # sub-project folder inside the project
ITERATIONS = 8
POSES = {"frontal": "frontal", "as in the photo": "photo"}

# Basel landmark (from the model file's metadata) -> MediaPipe Face Landmarker index, weight.
# Found by aligning MediaPipe's 3D landmarks of 15 frontal photos to the Basel landmarks and taking, for each
# Basel landmark, the nearest MediaPipe point (the eye and mouth corners use MediaPipe's own corner points).
# Left out: ears (MediaPipe has none), hairline and nasolabial folds (not located reliably), the eyebrows
# (they lie on skin, move with expression, and MediaPipe and Basel place them differently: 10-16 px apart in a
# first test) and the inner lip points (they move when the mouth opens).
CORRESPONDENCES = [
    ("right.eye.corner_outer", 33, 1.0), ("right.eye.corner_inner", 133, 1.0),
    ("left.eye.corner_outer", 263, 1.0), ("left.eye.corner_inner", 362, 1.0),
    ("right.eye.bottom", 145, 0.5), ("left.eye.bottom", 374, 0.5),
    ("center.nose.tip", 4, 1.0), ("center.nose.attachement_to_philtrum", 2, 1.0),
    ("right.nose.wing.outer", 129, 1.0), ("left.nose.wing.outer", 358, 1.0),
    ("right.nose.wing.tip", 98, 1.0), ("left.nose.wing.tip", 327, 1.0),
    ("right.lips.philtrum_ridge", 37, 0.5), ("left.lips.philtrum_ridge", 267, 0.5),
    ("center.lips.upper.outer", 0, 1.0), ("center.lips.lower.outer", 18, 1.0),
    ("right.lips.corner", 61, 1.0), ("left.lips.corner", 291, 1.0),
    ("center.chin.tip", 152, 0.7),
]
EYES = (33, 263)                                     # outer eye corners: their distance is the scale of the error


# ----------------------------------------------------------------- landmarks
def landmarks_from_photo(img: Image.Image):
    """MediaPipe landmarks of the face as an (N, 2) array in pixels (x right, y down), or None if no face is found."""
    import face3d as f3                              # needs mediapipe; only the landmarks come from here
    verts, _ = f3.fit(img)
    return None if verts is None else np.asarray(verts[:, :2], dtype=np.float64)


_corr = {}


def correspondences(model):
    """
    (names, vertex indices, MediaPipe indices, weights) for the landmarks this model file has. The vertex is the
    mean-shape vertex nearest to the landmark position given in the file's metadata.
    """
    if model.path in _corr:
        return _corr[model.path]
    with bfm.h5py.File(model.path, "r") as f:
        raw = f["metadata/landmarks/json"][()]
    if isinstance(raw, np.ndarray):                      # stored as an array holding one string
        raw = raw.ravel()[0]
    raw = raw.decode() if isinstance(raw, bytes) else str(raw)
    pos = {d["id"]: np.asarray(d["coordinates"], dtype=np.float64) for d in json.loads(raw)}
    mean = model.shape_mean.reshape(-1, 3).astype(np.float64)
    names, verts, mp, w = [], [], [], []
    for name, idx, weight in CORRESPONDENCES:
        if name in pos:
            names.append(name); verts.append(int(np.argmin(((mean - pos[name]) ** 2).sum(1)))); mp.append(idx); w.append(weight)
    _corr[model.path] = (names, np.array(verts), np.array(mp), np.array(w, dtype=np.float64))
    return _corr[model.path]


# ----------------------------------------------------------------- the fit
def _camera(q, X, w):
    """Scaled orthographic camera that maps X (K, 3) to q (K, 2): s, R (3 x 3 rotation), t (2,)."""
    sw = np.sqrt(w)[:, None]
    A = np.c_[X, np.ones(len(X))] * sw
    M = np.linalg.lstsq(A, q * sw, rcond=None)[0].T                 # 2 x 4 affine camera
    U, S, Vt = np.linalg.svd(M[:, :3], full_matrices=False)
    R2 = U @ Vt                                                      # nearest 2 x 3 with orthonormal rows
    s = float(S.mean())
    R = np.vstack([R2, np.cross(R2[0], R2[1])])
    t = ((q - s * X @ R2.T) * w[:, None]).sum(0) / w.sum()
    return s, R, t


def fit(model, points, components=None, sigma_mm=None, iterations=ITERATIONS):
    """
    Fit shape and pose to MediaPipe landmarks `points` ((N, 2) pixels, x right, y down).
    Returns a dict: coefficients (components,), verts (V, 3) in mm, rotation (3 x 3), scale (px per mm),
    shift (2,), error_px (mean distance between the landmarks and the fitted ones), error_eyes (the same as a
    share of the distance between the outer eye corners), predicted (K, 2) fitted landmark positions in pixels,
    observed (K, 2), names. None if the points do not cover the needed landmarks.
    """
    names, vidx, mpidx, w = correspondences(model)
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= int(mpidx.max()):
        return None
    n = min(int(components or COMPONENTS), model.shape_sd.size)
    sigma = float(sigma_mm or SIGMA_MM)
    q = points[mpidx] * np.array([1.0, -1.0])                        # y up, as in the model
    rows = (3 * vidx[:, None] + np.arange(3)).ravel()
    mean_L = model.shape_mean[rows].astype(np.float64).reshape(-1, 3)
    basis_L = (np.asarray(model.shape_basis[rows, :n], dtype=np.float64) * model.shape_sd[:n]).reshape(len(vidx), 3, n)
    a = np.zeros(n)
    X = mean_L.copy()
    for _ in range(iterations):
        s, R, t = _camera(q, X, w)
        R2 = R[:2]
        r = ((q - t) / s - mean_L @ R2.T).ravel()                    # residual in mm, (2K,)
        J = np.einsum("ij,kjn->kin", R2, basis_L).reshape(-1, n)     # (2K, n)
        W = np.repeat(w, 2)
        a = np.linalg.solve(J.T @ (J * W[:, None]) + sigma ** 2 * np.eye(n), J.T @ (W * r))
        X = mean_L + basis_L @ a
    s, R, t = _camera(q, X, w)
    pred = s * X @ R[:2].T + t
    err = np.linalg.norm(pred - q, axis=1)
    eyes = float(np.linalg.norm(points[EYES[0]] - points[EYES[1]]))
    coeff = np.zeros(model.shape_sd.size, dtype=np.float32); coeff[:n] = a
    verts = (model.shape_mean + model.shape_basis[:, :n] @ (model.shape_sd[:n] * a.astype(np.float32))).reshape(-1, 3)
    return {"coefficients": coeff, "verts": verts.astype(np.float32), "rotation": R, "scale": s, "shift": t,
            "error_px": float(err.mean()), "error_eyes": float(err.mean() / max(eyes, 1e-6)),
            "predicted": pred * np.array([1.0, -1.0]), "observed": points[mpidx], "names": names, "components": n}


def project(model, verts, res):
    """Vertices (V, 3) in mm to photo pixels (V, 2) with the fitted camera."""
    p = res["scale"] * np.asarray(verts, dtype=np.float64) @ res["rotation"][:2].T + res["shift"]
    return p * np.array([1.0, -1.0])


def rotate(model, verts, R):
    """The fitted head turned as in the photo (about the middle of the mean head), for 'as in the photo'."""
    c = model.centre.astype(np.float64)
    return ((np.asarray(verts, dtype=np.float64) - c) @ R.T + c).astype(np.float32)


def skin_colours(model, img, res, verts):
    """
    The model's mean skin colour, scaled by one RGB gain so that the overall tone matches the photo inside the
    fitted face (cheeks, forehead and nose, as seen from the camera). Gain limited to 0.4 - 1.6.
    """
    colours = np.clip(model.colour_mean.reshape(-1, 3), 0, 1).astype(np.float32)
    if not TONE:
        return colours, np.ones(3)
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    H, W = arr.shape[:2]
    p = project(model, verts, res)
    normals = bfm.vertex_normals(np.asarray(verts, dtype=np.float32) @ res["rotation"].T.astype(np.float32), model.tris, model.outward)
    facing = normals[:, 2] > 0.6                                     # surface turned towards the camera
    inside = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    obs = res["observed"]
    lo, hi = obs.min(0), obs.max(0)                                  # the box of the landmarks: the face, without hair and neck
    box = (p[:, 0] > lo[0]) & (p[:, 0] < hi[0]) & (p[:, 1] > lo[1]) & (p[:, 1] < hi[1])
    use = facing & inside & box
    if use.sum() < 50:
        return colours, np.ones(3)
    px = arr[p[use, 1].astype(int), p[use, 0].astype(int)]
    gain = np.clip(np.median(px, 0) / np.maximum(np.median(colours[use], 0), 1e-3), 0.4, 1.6)
    return np.clip(colours * gain, 0, 1).astype(np.float32), gain


def _hull(p):
    """Convex hull of 2D points (Andrew's monotone chain), as a list of points in order."""
    p = sorted(map(tuple, np.round(p, 1)))
    if len(p) < 3:
        return p
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def overlay(img, model, res, verts):
    """
    The photo with the fitted head drawn over it: its outline (white), the landmarks MediaPipe found (green), the
    fitted model's landmarks (red), joined by a yellow line where they differ.
    """
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    p = project(model, verts, res)
    normals = bfm.vertex_normals(np.asarray(verts, dtype=np.float32) @ res["rotation"].T.astype(np.float32), model.tris, model.outward)
    front = p[normals[:, 2] > 0.2]
    if len(front) > 2:
        hull = _hull(front[:: max(1, len(front) // 4000)])
        d.line(hull + hull[:1], fill=(245, 245, 245), width=2)
    for (ox, oy), (fx, fy) in zip(res["observed"], res["predicted"]):
        d.line([(ox, oy), (fx, fy)], fill=(255, 210, 0), width=1)
        d.ellipse([ox - 3, oy - 3, ox + 3, oy + 3], fill=(0, 220, 120))
        d.ellipse([fx - 4, fy - 4, fx + 4, fy + 4], outline=(255, 50, 50), width=2)
    return out


# ----------------------------------------------------------------- the set for the app
TRUTH_FIELDS = bfm.TRUTH_FIELDS + ["fit_error_px", "fit_error_eyes", "fit_components", "coefficients_file", "tone_gain"]


def run_fit_set(project, files, lights, pose="frontal", model_path=None, shadows=True, source="Original", identities=None,
                landmarks=None, log=print, progress=None):
    """
    The fitted set: for every photo in `files` (names in <project>/<source>/) fit the Basel model to the
    face's landmarks and draw the fitted head under every light in `lights` (keys of bfm.LIGHTS).
    pose: "frontal" (default, comparable with the invented faces) or "as in the photo".
    landmarks: optional {photo stem: (N, 2) pixels}; photos without an entry get theirs from MediaPipe.

    Writes into <project>/3D_basel_fit/, the same layout as <project>/3D/:
        Original/<stem>__<light>.png     the render
        truth/<stem>__depth.npy|png      true depth of the fitted head (larger = nearer, mm, 0 outside)
        truth/<stem>__mask.png           where the head is
        truth/<stem>__texture.png        the head without shading
        truth/<stem>__fit.png            the photo with the fitted landmarks: check the fit here
        truth/<stem>__coefficients.npy   the fitted shape coefficients
        truth/renders.csv, dataset.csv   as for the other sets, plus the landmark error of every fit
    Earlier content of 3D_basel_fit/ is removed, so the folder holds one consistent set.
    """
    import shutil
    problem = bfm.check(model_path)
    if problem:
        log(problem); return []
    try:
        model = bfm.load(model_path)
    except bfm.ModelProblem as e:
        log(str(e)); return []
    names, _, _, _ = correspondences(model)
    if len(names) < 10:
        log(f"{model.name} has too few landmarks in its metadata for the fit ({len(names)})."); return []
    root = os.path.join(project, SUB)
    ori, tru = os.path.join(root, "Original"), os.path.join(root, "truth")
    for sub in ("Original", "Mooney", "truth", "analysis", "figures"):
        if sub != "figures" and os.path.isdir(os.path.join(root, sub)):
            shutil.rmtree(os.path.join(root, sub))
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for d in os.listdir(root):
        if d.startswith("Result") and os.path.isdir(os.path.join(root, d)):
            shutil.rmtree(os.path.join(root, d))
    if os.path.exists(os.path.join(root, "params.csv")):
        os.remove(os.path.join(root, "params.csv"))
    lights = [l for l in bfm.LIGHTS if l in lights]
    identities, landmarks = identities or {}, landmarks or {}
    log(f"Model {model.name} · {len(names)} landmarks · {COMPONENTS} shape components · expected landmark error {SIGMA_MM:g} mm · "
        f"pose {pose} · cast shadows {'on' if shadows else 'off'}")
    rows, data, t0, missed, errors = [], [], time.time(), [], []
    width = max((len(os.path.splitext(f)[0]) for f in files), default=10)
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        tag = f"[{i + 1:>{len(str(len(files)))}}/{len(files)}] {stem:<{width}}"
        img = Image.open(os.path.join(project, source, f)).convert("RGB")
        pts = landmarks.get(stem)
        pts = landmarks_from_photo(img) if pts is None else np.asarray(pts, dtype=np.float64)
        res = None if pts is None else fit(model, pts)
        if res is None:
            missed.append(stem); log(f"{tag}  no face found, skipped")
            if progress: progress((i + 1) / len(files))
            continue
        if res["error_eyes"] > MAX_ERROR:
            missed.append(stem); log(f"{tag}  fit too poor (landmark error {res['error_px']:.1f} px = "
                                     f"{res['error_eyes']:.0%} of the eye distance), skipped")
            if progress: progress((i + 1) / len(files))
            continue
        errors.append(res["error_px"])
        verts = res["verts"] if POSES.get(pose, pose) == "frontal" else rotate(model, res["verts"], res["rotation"])
        colours, gain = skin_colours(model, img, res, res["verts"])
        overlay(img, model, res, res["verts"]).save(os.path.join(tru, f"{stem}__fit.png"))
        np.save(os.path.join(tru, f"{stem}__coefficients.npy"), res["coefficients"])
        vw = bfm.view(model, verts, colours, 0.0)
        mask, depth = bfm.mask_and_depth(vw)
        np.save(os.path.join(tru, f"{stem}__depth.npy"), depth)
        bfm.depth_picture(depth, mask).save(os.path.join(tru, f"{stem}__depth.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(tru, f"{stem}__mask.png"))
        bfm.shade(vw, bfm.LIGHTS["front"], shadows=False, ambient=1.0).save(os.path.join(tru, f"{stem}__texture.png"))
        who = identities.get(stem) or stem.rsplit("_", 1)[0].replace("-", " ")
        yaw = float(np.degrees(np.arctan2(res["rotation"][0, 2], res["rotation"][2, 2])))
        for light in lights:
            name = f"{stem}__{bfm.light_slug(light)}.png"
            bfm.shade(vw, bfm.LIGHTS[light], shadows=shadows).save(os.path.join(ori, name))
            x, y, z = bfm.LIGHTS[light]
            rows.append({"file": name, "source": f, "identity": who, "light": light, "light_x": x, "light_y": y, "light_z": z,
                         "head_pose": "frontal" if POSES.get(pose, pose) == "frontal" else "as in the photo",
                         "photo_yaw_deg": f"{yaw:.1f}", "depth_file": f"{stem}__depth.npy", "mask_file": f"{stem}__mask.png",
                         "method": "Basel Face Model 2019 fitted to MediaPipe landmarks, matte shading", "yaw_deg": 0.0, "seed": "",
                         "model_file": model.name, "coefficient_scale": "", "cast_shadows": "yes" if shadows else "no",
                         "expression": "neutral", "ambient": bfm.AMBIENT, "fit_error_px": round(res["error_px"], 2),
                         "fit_error_eyes": round(res["error_eyes"], 4), "fit_components": res["components"],
                         "coefficients_file": f"{stem}__coefficients.npy", "tone_gain": " ".join(f"{g:.2f}" for g in gain)})
            data.append({"identity": who, "file": name, "license": f"render of a Basel Face Model fit to {f}", "author": "",
                         "commons_url": "", "yaw": "", "face_frac": "", "sharpness": "", "date": ""})
        log(f"{tag}  landmark error {res['error_px']:4.1f} px ({res['error_eyes']:.0%} of the eye distance) · {len(lights)} renders")
        if progress: progress((i + 1) / len(files))
    with open(os.path.join(tru, "renders.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRUTH_FIELDS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(root, "dataset.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["identity", "file", "license", "author", "commons_url", "yaw", "face_frac", "sharpness", "date"])
        w.writeheader(); w.writerows(data)
    log("")
    log(f"Finished: {len(rows)} renders of {len(files) - len(missed)} fitted faces under {len(lights)} lights in {time.time() - t0:.0f} s"
        + (f"; mean landmark error {np.mean(errors):.1f} px" if errors else "") + (f"; skipped: {', '.join(missed)}" if missed else "") + ".")
    log(f"Written to {SUB}/Original/ and {SUB}/truth/. Check the fits in {SUB}/truth/<photo>__fit.png.")
    return rows
