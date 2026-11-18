"""
Invented 3D faces from the Basel Face Model 2019, drawn under chosen lights, with exact truth.

The Basel Face Model (BFM) is a statistical model of 3D face shape and skin colour, built from 3D scans.
A face is the model's mean plus a weighted sum of its principal components:

    face = mean + pcaBasis @ (sqrt(pcaVariance) * a)          a ~ N(0, 1), one number per component

for shape (positions of all vertices, in mm) and, separately, for colour (RGB per vertex, 0 to 1).
A seed fixes the numbers `a`, so `basel-007` is the same invented face every time. Nobody looks like this:
the faces are samples from the model, not people.

What is known exactly for every picture made here: the shape (so the depth of every pixel), the light
direction, the head pose, and whether cast shadows were drawn. The heads have no hair.

The model file is NOT part of Mooney Studio and may not be redistributed. It is free for non-commercial
research after registration at https://faces.dmi.unibas.ch/bfm/bfm2019.html ; put the .h5 file in models/.

    check(path)                          None if the model can be used, otherwise a plain sentence saying what is missing
    load(path)                           the model (kept in memory until another file is loaded)
    sample(model, seed, scale)           vertices (N, 3) in mm and colours (N, 3)
    view(model, verts, colours, yaw)     the head as seen by the camera: which surface point is in which pixel
    shade(view, light, shadows)          the picture for one light
    render(...)                          view + shade in one call: picture, face mask, true depth
    run_render_set(project, seeds, ...)  the set for the app -> <project>/3D_basel/ (same layout as 3D/)

Coordinates of the model: x to the viewer's right, y up, z towards the viewer, in mm.
Light directions are given as in face3d.py: x right, y DOWN, z towards the viewer (image convention).

This file needs only numpy, h5py and Pillow.
"""
import csv
import os
import time

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:                                  # the rest of the app works without it
    h5py = None

SUB = "3D_basel"                                     # sub-project folder inside the project
URL = "https://faces.dmi.unibas.ch/bfm/bfm2019.html"
FILES = {"Whole head": "model2019_fullHead.h5", "Face only": "model2019_face12.h5"}

# the same names and vectors as face3d.LIGHTS: x right, y down, z towards the viewer
LIGHTS = {
    "front":      (0.0, 0.0, 1.0),
    "left 45°":   (-0.71, 0.0, 0.71),
    "right 45°":  (0.71, 0.0, 0.71),
    "top":        (0.0, -0.71, 0.71),
    "below":      (0.0, 0.71, 0.71),
    "back-light": (0.0, -0.3, -0.95),
}
# head turn about the vertical axis, in degrees; negative = the face points to the viewer's left
POSES = {"frontal": 0.0, "turned 30° left": -30.0, "turned 30° right": 30.0}

SIZE = 512                                           # picture size in pixels
FRAME_MM = 300.0                                     # the picture shows a square of this many mm (same scale for every face)
AMBIENT = 0.1                                        # light that reaches every point, also in shadow (0 to 1)
SCALE = 0.9                                          # coefficient scale: 0 = the mean face, 1 = the model's full variation
SUPERSAMPLE = 2                                      # the picture is drawn at 2 x the size and averaged down (smooth outline)
SHADOW_BIAS_MM = 1.0                                 # a point counts as shadowed only if the blocker is at least this much nearer the light
_BUCKETS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192])


def light_slug(name):
    """'left 45°' -> 'left45'; used in file names."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def pose_slug(name):
    """'turned 30° left' -> 'left30'; '' for the frontal view."""
    deg = POSES[name]
    return "" if abs(deg) < 1e-6 else f"{'left' if deg < 0 else 'right'}{abs(deg):.0f}"


def face_name(seed, pose="frontal"):
    """'basel-007' for the frontal view, 'basel-007-left30' for a turned one."""
    return f"basel-{int(seed):03d}" + (f"-{pose_slug(pose)}" if pose_slug(pose) else "")


# ----------------------------------------------------------------- the model file
def how_to_get(path):
    return (f"The Basel Face Model file was not found ({path}). It is not part of Mooney Studio and may not be passed on. "
            f"It is free for non-commercial research: register at {URL}, download {FILES['Whole head']} (whole head) or "
            f"{FILES['Face only']} (face only) and put the file in the models/ folder.")


def check(path):
    """None if faces can be drawn from this model file, otherwise one plain sentence saying what is missing."""
    if h5py is None:
        return "The Basel source needs the Python package h5py. Install it with:  pip install h5py"
    if not path or not os.path.exists(path):
        return how_to_get(path)
    return None


class ModelProblem(Exception):
    """The model cannot be used; the message is written for the user."""


class Model:
    """Mean, principal components and triangles of one model file."""
    def __init__(self, path):
        problem = check(path)
        if problem:
            raise ModelProblem(problem)
        self.path, self.name = path, os.path.basename(path)
        try:
            with h5py.File(path, "r") as f:
                get = lambda key: np.asarray(f[key])
                self.shape_mean = get("shape/model/mean").astype(np.float32)
                self.shape_basis = get("shape/model/pcaBasis")
                self.shape_sd = np.sqrt(get("shape/model/pcaVariance")).astype(np.float32)
                self.colour_mean = get("color/model/mean").astype(np.float32)
                self.colour_basis = get("color/model/pcaBasis")
                self.colour_sd = np.sqrt(get("color/model/pcaVariance")).astype(np.float32)
                self.tris = get("shape/representer/cells").T.astype(np.int64)
                self.has_expression = "expression" in f and "model" in f["expression"]
        except (KeyError, OSError) as e:
            raise ModelProblem(f"{self.name} could not be read as a Basel Face Model 2019 file ({e}). "
                               f"Download it again from {URL}.")
        self._expression = None
        mean = self.shape_mean.reshape(-1, 3)
        self.n_vertices, self.n_triangles = len(mean), len(self.tris)
        self.centre = ((mean.min(0) + mean.max(0)) / 2).astype(np.float32)          # middle of the mean head: picture centre and turning axis
        # which way the triangles are wound: +1 if their normals point away from the middle of the head
        fn = np.cross(mean[self.tris[:, 1]] - mean[self.tris[:, 0]], mean[self.tris[:, 2]] - mean[self.tris[:, 0]])
        self.outward = 1.0 if float((fn * (mean[self.tris].mean(1) - mean.mean(0))).sum()) >= 0 else -1.0

    def expression(self):
        """Mean, basis and standard deviations of the expression part, read on first use."""
        if self._expression is None:
            if not self.has_expression:
                raise ModelProblem(f"{self.name} has no expression part.")
            with h5py.File(self.path, "r") as f:
                self._expression = (np.asarray(f["expression/model/mean"]).astype(np.float32), np.asarray(f["expression/model/pcaBasis"]),
                                    np.sqrt(np.asarray(f["expression/model/pcaVariance"])).astype(np.float32))
        return self._expression


_loaded = {}


def load(path):
    """The model for this file. One model is kept in memory at a time (the whole head needs about 300 MB)."""
    path = os.path.abspath(path)
    if path not in _loaded:
        _loaded.clear()
        _loaded[path] = Model(path)
    return _loaded[path]


def sample(model, seed, scale=SCALE, expression=False):
    """
    One invented face. Returns (verts, colours): (N, 3) positions in mm and (N, 3) RGB values between 0 and 1.
    The seed fixes the face. `scale` multiplies the random coefficients: 0 gives the mean face, 1 the model's
    full variation; 0.9 keeps the faces a little nearer to the mean. With expression=True a random facial
    expression is added to the neutral shape.
    """
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(model.shape_sd.size).astype(np.float32) * scale
    b = rng.standard_normal(model.colour_sd.size).astype(np.float32) * scale
    verts = model.shape_mean + model.shape_basis @ (model.shape_sd * a)
    colours = model.colour_mean + model.colour_basis @ (model.colour_sd * b)
    if expression:
        mean, basis, sd = model.expression()
        e = np.random.default_rng([int(seed), 1]).standard_normal(sd.size).astype(np.float32) * scale
        verts = verts + mean + basis @ (sd * e)
    return verts.reshape(-1, 3).astype(np.float32), np.clip(colours.reshape(-1, 3), 0, 1).astype(np.float32)


# ----------------------------------------------------------------- geometry
def rotate_yaw(verts, deg, centre):
    """Turn the head about the vertical axis through `centre`. Negative = the face points to the viewer's left."""
    if abs(deg) < 1e-6:
        return verts
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]], dtype=np.float32)
    return ((verts - centre) @ R.T + centre).astype(np.float32)


def vertex_normals(verts, tris, outward=1.0):
    """Unit normal per vertex: the area-weighted mean of the normals of its triangles, pointing out of the head."""
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]) * outward
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def _raster(px, py, z, tris, width, height, want_bary=True):
    """
    Z-buffer rasteriser. px, py: vertex positions in pixels, pixel centres at whole numbers; z: larger = nearer.
    A pixel belongs to a triangle if its centre lies inside it (edges included, so neighbours leave no gap);
    where several triangles cover a pixel the nearest one wins.
    Returns (tri, bary, zbuf): triangle index per pixel (-1 = empty), the three barycentric weights, the depth.
    Triangles are handled in groups of similar size, each group in one numpy operation.
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ax, ay, bx, by, cx, cy = px[a], py[a], px[b], py[b], px[c], py[c]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    eps = 1e-6
    lox = np.maximum(np.ceil(np.minimum(np.minimum(ax, bx), cx) - eps), 0).astype(np.int64)
    hix = np.minimum(np.floor(np.maximum(np.maximum(ax, bx), cx) + eps), width - 1).astype(np.int64)
    loy = np.maximum(np.ceil(np.minimum(np.minimum(ay, by), cy) - eps), 0).astype(np.int64)
    hiy = np.minimum(np.floor(np.maximum(np.maximum(ay, by), cy) + eps), height - 1).astype(np.int64)
    ok = (np.abs(area) > 1e-12) & (hix >= lox) & (hiy >= loy)
    zbuf = np.full(width * height, -np.inf)
    tri = np.full(width * height, -1, dtype=np.int64)
    bary = np.zeros((width * height, 3), dtype=np.float32) if want_bary else None
    wi = np.searchsorted(_BUCKETS, hix - lox + 1); hi = np.searchsorted(_BUCKETS, hiy - loy + 1)
    key = np.where(ok, wi * 100 + hi, -1)
    for k in np.unique(key[ok]):
        ids = np.nonzero(key == k)[0]
        bw, bh = int(_BUCKETS[k // 100]), int(_BUCKETS[k % 100])
        step = max(1, int(1_500_000 // (bw * bh)))
        ox = np.arange(bw)[None, None, :]; oy = np.arange(bh)[None, :, None]
        for s in range(0, len(ids), step):
            t = ids[s:s + step]
            col = lambda v: v[t][:, None, None]
            X = col(lox) + ox; Y = col(loy) + oy                                    # (n, bh, bw) by broadcasting
            w0 = ((col(bx) - X) * (col(cy) - Y) - (col(by) - Y) * (col(cx) - X)) / col(area)
            w1 = ((col(cx) - X) * (col(ay) - Y) - (col(cy) - Y) * (col(ax) - X)) / col(area)
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7) & (X <= col(hix)) & (Y <= col(hiy))
            n, yy, xx = np.nonzero(inside)
            if not len(n):
                continue
            f0, f1, f2 = w0[n, yy, xx], w1[n, yy, xx], w2[n, yy, xx]
            tt = t[n]
            zz = f0 * z[a[tt]] + f1 * z[b[tt]] + f2 * z[c[tt]]
            pix = (loy[tt] + yy) * width + (lox[tt] + xx)
            np.maximum.at(zbuf, pix, zz)
            win = zz >= zbuf[pix]
            tri[pix[win]] = tt[win]
            if want_bary:
                bary[pix[win]] = np.stack([f0[win], f1[win], f2[win]], 1)
    return tri, bary, zbuf


def view(model, verts, colours, yaw=0.0, size=SIZE, supersample=SUPERSAMPLE):
    """
    The head as the camera sees it (parallel projection, camera on the z axis): for every pixel the surface
    point, its normal and its colour. Independent of the light, so one view serves all lights of a face.
    """
    v = rotate_yaw(np.asarray(verts, dtype=np.float32), yaw, model.centre)
    n = size * supersample
    k = n / FRAME_MM                                                              # pixels per mm
    px = (v[:, 0] - model.centre[0]) * k + n / 2 - 0.5
    py = n / 2 - 0.5 - (v[:, 1] - model.centre[1]) * k
    tri, bary, zbuf = _raster(px.astype(np.float64), py.astype(np.float64), v[:, 2].astype(np.float64), model.tris, n, n)
    seen = tri >= 0
    t = model.tris[tri[seen]]; w = bary[seen][:, :, None]
    mix = lambda attr: (attr[t] * w).sum(1)
    normal = mix(vertex_normals(v, model.tris, model.outward))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
    return {"verts": v, "tris": model.tris, "size": size, "ss": supersample, "seen": seen, "points": mix(v), "normals": normal,
            "colours": mix(np.asarray(colours, dtype=np.float32)), "z": zbuf[seen].astype(np.float32)}


def _down(arr, ss):
    """Average ss x ss blocks."""
    h, w = arr.shape[0] // ss, arr.shape[1] // ss
    return arr.reshape(h, ss, w, ss, *arr.shape[2:]).mean((1, 3))


def mask_and_depth(vw):
    """
    (mask, depth) of a view at picture size. mask: True where the head covers the pixel. depth: float32,
    larger = nearer to the camera, in mm relative to the mean over the head, 0 outside the head.
    """
    n = vw["size"] * vw["ss"]
    cover = np.zeros(n * n, dtype=np.float32); cover[vw["seen"]] = 1.0
    zfull = np.zeros(n * n, dtype=np.float32); zfull[vw["seen"]] = vw["z"]
    cover, zsum = _down(cover.reshape(n, n), vw["ss"]), _down(zfull.reshape(n, n), vw["ss"])
    mask = cover >= 0.5
    depth = np.where(mask, zsum / np.maximum(cover, 1e-6), 0.0).astype(np.float32)
    if mask.any():
        depth[mask] -= float(depth[mask].mean())
    return mask, depth


def _to_model(light):
    """A light direction from the image convention (y down) to the model's (y up), as a unit vector."""
    l = np.array([light[0], -light[1], light[2]], dtype=np.float64)
    return l / np.linalg.norm(l)


def lit_share(vw, light, n_dot_l=None):
    """
    Cast shadows: for every visible surface point, how much of the light reaches it (1 = all, 0 = something lies
    between the point and the light). The head is drawn once as seen from the light (a shadow map); a point is
    lit if nothing in that picture is nearer to the light than the point itself.
    """
    l = _to_model(light)
    up = np.array([0.0, 1.0, 0.0]) if abs(l[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(up, l); u /= np.linalg.norm(u)
    w = np.cross(l, u)
    v = vw["verts"].astype(np.float64)
    su, sv, sd = v @ u, v @ w, v @ l
    res = 2 * vw["size"]
    k = (res - 3) / max(float(np.ptp(su)), float(np.ptp(sv)), 1e-6)               # shadow-map pixels per mm
    _, _, zs = _raster((su - su.min()) * k + 1, (sv - sv.min()) * k + 1, sd, vw["tris"], res, res, want_bary=False)
    zs = zs.reshape(res, res)
    p = vw["points"].astype(np.float64)
    qx, qy, qd = (p @ u - su.min()) * k + 1, (p @ w - sv.min()) * k + 1, p @ l
    if n_dot_l is None:
        n_dot_l = vw["normals"].astype(np.float64) @ l
    c = np.clip(n_dot_l, 1e-3, 1.0)
    bias = SHADOW_BIAS_MM + np.clip(np.sqrt(1 - c * c) / c, 0, 8) / k             # more tolerance where the light grazes the surface
    x0, y0 = np.floor(qx).astype(np.int64), np.floor(qy).astype(np.int64)
    fx, fy = qx - x0, qy - y0
    lit = np.zeros(len(p))
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        blocker = zs[np.clip(y0 + dy, 0, res - 1), np.clip(x0 + dx, 0, res - 1)]
        lit += wt * (qd + bias >= blocker)
    return lit


def shade(vw, light, shadows=True, ambient=AMBIENT):
    """
    The picture of a view under one distant light (matte surface): colour x (ambient + (1 - ambient) x cos of the
    angle between surface normal and light), and in a cast shadow only the ambient part. Black background.
    Returns a PIL RGB image of vw['size'] pixels.
    """
    cos = np.clip(vw["normals"].astype(np.float64) @ _to_model(light), 0, 1)
    if shadows and cos.any():
        cos = cos * lit_share(vw, light, cos)
    rgb = vw["colours"] * (ambient + (1 - ambient) * cos)[:, None]
    n = vw["size"] * vw["ss"]
    img = np.zeros((n * n, 3), dtype=np.float32); img[vw["seen"]] = rgb
    img = _down(img.reshape(n, n, 3), vw["ss"])
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def render(model, verts, colours, light, yaw=0.0, shadows=True, ambient=AMBIENT, size=SIZE, supersample=SUPERSAMPLE):
    """One picture. Returns (image, mask, depth): PIL RGB, bool H x W, float32 H x W (larger = nearer, mm, 0 outside)."""
    vw = view(model, verts, colours, yaw, size, supersample)
    mask, depth = mask_and_depth(vw)
    return shade(vw, light, shadows, ambient), mask, depth


def depth_picture(depth, mask):
    """The depth as a gray picture for looking at: bright = near, black outside the head."""
    if not mask.any():
        return Image.fromarray(np.zeros(depth.shape, dtype=np.uint8))
    lo, hi = np.percentile(depth[mask], 1), np.percentile(depth[mask], 99)
    g = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1) * 235 + 20
    return Image.fromarray(np.where(mask, g, 0).astype(np.uint8))


# ----------------------------------------------------------------- the set for the app
TRUTH_FIELDS = ["file", "source", "identity", "light", "light_x", "light_y", "light_z", "head_pose", "photo_yaw_deg",
                "depth_file", "mask_file", "method", "yaw_deg", "seed", "model_file", "coefficient_scale", "cast_shadows",
                "expression", "ambient"]


def run_render_set(project, seeds, lights, poses=("frontal",), model_path=None, scale=SCALE, shadows=True, expression=False,
                   log=print, progress=None):
    """
    The Basel set: for every seed one invented face, drawn in every pose in `poses` (keys of POSES) under every
    light in `lights` (keys of LIGHTS).

    Writes into <project>/3D_basel/, which has the same layout as <project>/3D/ and as a project, so that Batch
    and the vision systems run on it unchanged:
        Original/<face>__<light>.png      the render; <face> is basel-<seed>, with -left30 / -right30 for a turned head
        truth/<face>__depth.npy           true depth, float32, larger = nearer, mm, 0 outside the head
        truth/<face>__depth.png           the same as a picture
        truth/<face>__mask.png            where the head is
        truth/<face>__texture.png         the face without shading (skin colour only)
        truth/renders.csv                 one row per render: seed, model file, coefficient scale, light, pose, shadows
        dataset.csv                       render -> identity (basel-<seed>), so the systems know who is who
    Earlier content of 3D_basel/ (renders, Mooneys, results) is removed, so the folder holds one consistent set.
    """
    import shutil
    problem = check(model_path)
    if problem:
        log(problem); return []
    try:
        model = load(model_path)
    except ModelProblem as e:
        log(str(e)); return []
    if expression and not model.has_expression:
        log(f"{model.name} has no expression part; the faces are drawn with a neutral expression."); expression = False
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
    seeds, lights, poses = [int(s) for s in seeds], [l for l in LIGHTS if l in lights], [p for p in POSES if p in poses]
    log(f"Model {model.name}: {model.n_vertices} points, {model.n_triangles} triangles · coefficient scale {scale:g} · "
        f"cast shadows {'on' if shadows else 'off'}")
    rows, data, t0, faces = [], [], time.time(), len(seeds) * len(poses)
    k = 0
    for seed in seeds:
        verts, colours = sample(model, seed, scale, expression)
        who = face_name(seed)
        for pose in poses:
            stem = face_name(seed, pose)
            vw = view(model, verts, colours, POSES[pose])
            mask, depth = mask_and_depth(vw)
            np.save(os.path.join(tru, f"{stem}__depth.npy"), depth)
            depth_picture(depth, mask).save(os.path.join(tru, f"{stem}__depth.png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(tru, f"{stem}__mask.png"))
            shade(vw, LIGHTS["front"], shadows=False, ambient=1.0).save(os.path.join(tru, f"{stem}__texture.png"))
            for light in lights:
                name = f"{stem}__{light_slug(light)}.png"
                shade(vw, LIGHTS[light], shadows=shadows).save(os.path.join(ori, name))
                x, y, z = LIGHTS[light]
                rows.append({"file": name, "source": stem, "identity": who, "light": light, "light_x": x, "light_y": y, "light_z": z,
                             "head_pose": pose, "photo_yaw_deg": "", "depth_file": f"{stem}__depth.npy", "mask_file": f"{stem}__mask.png",
                             "method": "Basel Face Model 2019, random sample, matte shading", "yaw_deg": POSES[pose], "seed": seed,
                             "model_file": model.name, "coefficient_scale": scale, "cast_shadows": "yes" if shadows else "no",
                             "expression": "random" if expression else "neutral", "ambient": AMBIENT})
                data.append({"identity": who, "file": name, "license": "render of an invented face (Basel Face Model 2019)", "author": "",
                             "commons_url": "", "yaw": "", "face_frac": "", "sharpness": "", "date": ""})
            k += 1
            log(f"[{k:>{len(str(faces))}}/{faces}] {stem:<16}  {len(lights)} renders")
            if progress: progress(k / faces)
    with open(os.path.join(tru, "renders.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRUTH_FIELDS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(root, "dataset.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["identity", "file", "license", "author", "commons_url", "yaw", "face_frac", "sharpness", "date"])
        w.writeheader(); w.writerows(data)
    log("")
    log(f"Finished: {len(rows)} renders of {len(seeds)} invented faces"
        + (f" in {len(poses)} poses" if len(poses) > 1 else "") + f" under {len(lights)} lights in {time.time() - t0:.0f} s.")
    log(f"Written to {SUB}/Original/ and {SUB}/truth/.")
    return rows


if __name__ == "__main__":                           # python bfm.py [model.h5] [out.png]: one test picture
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "models", FILES["Whole head"])
    problem = check(path)
    if problem:
        print(problem); sys.exit(1)
    t = time.time(); m = load(path); print(f"{m.name}: {m.n_vertices} points, {m.n_triangles} triangles, loaded in {time.time() - t:.1f} s")
    t = time.time(); picture, _, _ = render(m, *sample(m, 1), LIGHTS["left 45°"]); print(f"one render: {time.time() - t:.1f} s")
    out = sys.argv[2] if len(sys.argv) > 2 else "bfm_test.png"
    picture.save(out); print("written to", out)
