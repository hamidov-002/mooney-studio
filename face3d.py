"""
3D faces from single photos, re-lit under chosen light directions, with known depth.

Geometry: MediaPipe Face Landmarker returns 478 3D landmarks (x, y in pixels, z = relative depth in the same
scale). They are triangulated in the image plane (Delaunay), the depth is smoothed, the mesh is subdivided
twice and drawn with a small software rasteriser (matte surface, one skin colour, painter's algorithm).

Because we make the renders ourselves, two things are known exactly for each of them:
  * the light direction, and
  * the depth of every pixel (true_depth), since the image is drawn from that very mesh.
The truth is exact for the RENDER. It is not the true shape of the real person: the mesh is an estimate.
The renders show the face region only, without hair, in one skin colour.

This is not a statistical 3D morphable model. `fit()` is the only part that would change if one is used later.

run_render_set(project, files, lights)  ->  <project>/3D/Original/<stem>__<light>.png, 3D/truth/*, 3D/dataset.csv
run_3d(...)                             ->  the older free-form renders in Original3D/ (kept for the pose option)
"""
import csv, os, time
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
             "float16/latest/face_landmarker.task")

# unit light directions in camera space: x right, y down, z towards the viewer
LIGHTS = {
    "front":      (0.0, 0.0, 1.0),
    "left 45°":   (-0.71, 0.0, 0.71),
    "right 45°":  (0.71, 0.0, 0.71),
    "top":        (0.0, -0.71, 0.71),
    "below":      (0.0, 0.71, 0.71),
    "back-light": (0.0, -0.3, -0.95),
}
POSES = {"as in photo": 0.0, "yaw −30°": -30.0, "yaw +30°": 30.0}
SUB = "3D"                                  # sub-project folder inside the project, same layout as the project itself


def light_slug(name):
    """'left 45°' -> 'left45'; used in file names."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


# ----------------------------------------------------------------- fitting
_landmarker = None


def load_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
    if not os.path.exists(MODEL) or os.path.getsize(MODEL) < 1_000_000:
        import requests
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        r = requests.get(MODEL_URL, timeout=120); r.raise_for_status()
        open(MODEL, "wb").write(r.content)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                        num_faces=1, output_facial_transformation_matrixes=True)
    _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def fit(img: Image.Image):
    """Return verts (468,3) in pixel units (z relative, same scale as x) and the 4x4 head pose matrix."""
    import mediapipe as mp
    lm = load_landmarker()
    arr = np.asarray(img.convert("RGB"))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None, None
    H, W = arr.shape[:2]
    pts = res.face_landmarks[0]
    verts = np.array([[p.x * W, p.y * H, p.z * W] for p in pts], dtype=np.float32)
    mat = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
    return verts, mat


# ----------------------------------------------------------------- mesh
def triangulate(verts):
    from scipy.spatial import Delaunay
    tri = Delaunay(verts[:, :2]).simplices
    # drop very elongated hull triangles (they bridge concavities at the face outline)
    a, b, c = verts[tri[:, 0], :2], verts[tri[:, 1], :2], verts[tri[:, 2], :2]
    edges = np.stack([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)], 1)
    med = np.median(edges)
    return tri[edges.max(1) < 4.5 * med]


def subdivide(verts, tris):
    """One midpoint subdivision step (4 triangles per triangle) for smoother shading."""
    mid = {}
    new_v = [verts]
    nv = len(verts)

    def m(i, j):
        nonlocal nv
        key = (min(i, j), max(i, j))
        if key not in mid:
            mid[key] = nv; nv += 1
            new_v.append(((verts[i] + verts[j]) / 2)[None])
        return mid[key]

    out = []
    for i, j, k in tris:
        a, b, c = m(i, j), m(j, k), m(k, i)
        out += [[i, a, c], [a, j, b], [c, b, k], [a, b, c]]
    return np.concatenate(new_v, 0), np.array(out)


def smooth_z(verts, tris, iters=2):
    """Laplacian smoothing of depth only (landmark z is noisy)."""
    v = verts.copy()
    nbr = [[] for _ in range(len(v))]
    for i, j, k in tris:
        nbr[i] += [j, k]; nbr[j] += [i, k]; nbr[k] += [i, j]
    for _ in range(iters):
        z = v[:, 2].copy()
        for i, n in enumerate(nbr):
            if n:
                z[i] = 0.5 * v[i, 2] + 0.5 * v[n, 2].mean()
        v[:, 2] = z
    return v


def rotate_yaw(verts, deg):
    if abs(deg) < 1e-6:
        return verts
    c = verts.mean(0)
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]], dtype=np.float32)
    return (verts - c) @ R.T + c


# ----------------------------------------------------------------- rendering
def vertex_normals(verts, tris):
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)                       # area-weighted face normals
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
    return vn


def render(verts, tris, light, size, ambient=0.08, bg=0, smooth=True):
    """
    Lambertian render, painter's algorithm. verts in pixel coords (z: smaller = closer to camera).
    smooth=True shades each triangle with the mean of its vertex normals (Gouraud-like), which
    removes the faceted look; smooth=False uses flat face normals.
    """
    W, H = size
    L = np.asarray(light, dtype=np.float32); L /= np.linalg.norm(L)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if smooth:
        vn = vertex_normals(verts, tris)
        n = (vn[tris[:, 0]] + vn[tris[:, 1]] + vn[tris[:, 2]]) / 3
    else:
        n = np.cross(p1 - p0, p2 - p0)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    n[:, 2] *= -1                       # MediaPipe z grows away from camera -> flip so +z faces viewer
    n[n[:, 2] < 0] *= -1
    shade = np.clip(n @ L, 0, 1) * (1 - ambient) + ambient
    depth = (p0[:, 2] + p1[:, 2] + p2[:, 2]) / 3
    order = np.argsort(-depth)
    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img)
    for t in order:
        g = int(round(255 * shade[t]))
        d.polygon([tuple(p0[t, :2]), tuple(p1[t, :2]), tuple(p2[t, :2])], fill=g, outline=g)
    return img


def true_depth(verts, tris, size):
    """
    Depth of the mesh for the view it is rendered in.
    Returns (depth, mask): depth is float32 H x W, larger = nearer to the camera, in pixel units relative to
    the mean of the face (0 outside the face); mask is True where the mesh covers the pixel.
    The mesh was triangulated in the image plane, so for the unrotated head every pixel sees exactly one
    surface point and the depth can be interpolated linearly inside each triangle.
    """
    from scipy.interpolate import LinearNDInterpolator
    W, H = size
    cover = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(cover)
    for i, j, k in tris:
        d.polygon([tuple(verts[i, :2]), tuple(verts[j, :2]), tuple(verts[k, :2])], fill=255, outline=255)
    mask = np.asarray(cover) > 0
    near = LinearNDInterpolator(verts[:, :2].astype(np.float64), -verts[:, 2].astype(np.float64))   # MediaPipe z grows away from the camera
    ys, xs = np.nonzero(mask)
    vals = near(np.column_stack([xs, ys]))
    ok = np.isfinite(vals)
    depth = np.zeros((H, W), dtype=np.float32)
    depth[ys[ok], xs[ok]] = vals[ok]
    mask = np.zeros((H, W), dtype=bool); mask[ys[ok], xs[ok]] = True
    if mask.any():
        depth[mask] -= float(depth[mask].mean())
    return depth, mask


def depth_picture(depth, mask):
    """The depth as a gray picture for looking at: bright = near, black outside the face."""
    if not mask.any():
        return Image.fromarray(np.zeros(depth.shape, dtype=np.uint8))
    lo, hi = np.percentile(depth[mask], 1), np.percentile(depth[mask], 99)
    g = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1) * 235 + 20
    return Image.fromarray(np.where(mask, g, 0).astype(np.uint8))


def overlay(img: Image.Image, verts, tris, color=(0, 255, 160)):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for i, j, k in tris:
        d.line([tuple(verts[i, :2]), tuple(verts[j, :2])], fill=color, width=1)
        d.line([tuple(verts[j, :2]), tuple(verts[k, :2])], fill=color, width=1)
        d.line([tuple(verts[k, :2]), tuple(verts[i, :2])], fill=color, width=1)
    return out


# ----------------------------------------------------------------- pipeline
def reconstruct(img: Image.Image):
    """fit + mesh. Returns dict or None."""
    verts, mat = fit(img)
    if verts is None:
        return None
    tris = triangulate(verts)
    verts = smooth_z(verts, tris)
    v2, t2 = subdivide(verts, tris)
    v3, t3 = subdivide(v2, t2)
    return {"verts": verts, "tris": tris, "verts_fine": v3, "tris_fine": t3, "pose": mat}


MIN_COVER = 0.95                            # below this the mesh has holes (head turned too far for this simple method)


def mesh_cover(verts, tris):
    """Share of the landmark outline (convex hull) that the kept triangles cover; 1.0 = a closed mesh."""
    from scipy.spatial import ConvexHull
    p, q, r = verts[tris[:, 0], :2], verts[tris[:, 1], :2], verts[tris[:, 2], :2]
    area = 0.5 * np.abs((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])).sum()
    return float(area / max(ConvexHull(verts[:, :2]).volume, 1e-9))


def photo_yaw(pose):
    """Head turn in degrees from MediaPipe's pose matrix (0 = frontal), or None."""
    if pose is None:
        return None
    R = np.asarray(pose)[:3, :3]
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))


TRUTH_FIELDS = ["file", "source", "identity", "light", "light_x", "light_y", "light_z", "head_pose", "photo_yaw_deg",
                "depth_file", "mask_file", "method"]


def run_render_set(project, files, lights, source="Original", identities=None, log=print, progress=None):
    """
    The 3D step: for every photo in `files` (names in <project>/<source>/) reconstruct the mesh and draw it under
    every light in `lights` (keys of LIGHTS), head pose as in the photo.

    Writes into <project>/3D/, which has the same layout as a project so that Batch and the vision systems
    can run on it unchanged:
        Original/<stem>__<light>.png      the render (plays the role of the original)
        truth/<stem>__depth.npy           true depth, float32, larger = nearer, 0 outside the face
        truth/<stem>__depth.png           the same as a picture
        truth/<stem>__mask.png            where the face is
        truth/<stem>__mesh.png            the mesh drawn over the photo
        truth/renders.csv                 one row per render: source photo, person, light name and vector, pose
        dataset.csv                       render -> person, so the systems know who is who
    Earlier renders in 3D/Original that are not part of this call are removed, together with their Mooneys and
    results, so the folder always holds exactly one consistent set.
    """
    import shutil
    root = os.path.join(project, SUB)
    ori, tru = os.path.join(root, "Original"), os.path.join(root, "truth")
    for sub in ("Original", "Mooney", "truth", "analysis", "figures"):
        if sub != "figures" and os.path.isdir(os.path.join(root, sub)):
            shutil.rmtree(os.path.join(root, sub))
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for d in os.listdir(root):
        if d.startswith("Result") and os.path.isdir(os.path.join(root, d)):
            shutil.rmtree(os.path.join(root, d))
    identities = identities or {}
    rows, data, t0, missed = [], [], time.time(), []
    width = max((len(os.path.splitext(f)[0]) for f in files), default=10)
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        tag = f"[{i + 1:>{len(str(len(files)))}}/{len(files)}] {stem:<{width}}"
        img = Image.open(os.path.join(project, source, f)).convert("RGB")
        rec = reconstruct(img)
        if rec is None:
            missed.append(stem); log(f"{tag}  no face found, skipped")
            if progress: progress((i + 1) / len(files))
            continue
        cover = mesh_cover(rec["verts"], rec["tris"])
        if cover < MIN_COVER:
            missed.append(stem); log(f"{tag}  head turned too far for a closed mesh (covers {cover:.0%}), skipped")
            if progress: progress((i + 1) / len(files))
            continue
        depth, mask = true_depth(rec["verts_fine"], rec["tris_fine"], img.size)
        np.save(os.path.join(tru, f"{stem}__depth.npy"), depth)
        depth_picture(depth, mask).save(os.path.join(tru, f"{stem}__depth.png"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(tru, f"{stem}__mask.png"))
        overlay(img, rec["verts"], rec["tris"]).save(os.path.join(tru, f"{stem}__mesh.png"))
        yaw = photo_yaw(rec["pose"])
        yaw = "" if yaw is None else f"{yaw:.1f}"
        who = identities.get(stem) or stem.rsplit("_", 1)[0].replace("-", " ")
        for light in lights:
            name = f"{stem}__{light_slug(light)}.png"
            render(rec["verts_fine"], rec["tris_fine"], LIGHTS[light], img.size).convert("RGB").save(os.path.join(ori, name))
            x, y, z = LIGHTS[light]
            rows.append({"file": name, "source": f, "identity": who, "light": light, "light_x": x, "light_y": y, "light_z": z,
                         "head_pose": "as in photo", "photo_yaw_deg": yaw, "depth_file": f"{stem}__depth.npy",
                         "mask_file": f"{stem}__mask.png", "method": "MediaPipe landmarks, Delaunay mesh, matte shading"})
            data.append({"identity": who, "file": name, "license": f"render of {f}", "author": "", "commons_url": "",
                         "yaw": "", "face_frac": "", "sharpness": "", "date": ""})
        log(f"{tag}  {len(lights)} renders")
        if progress: progress((i + 1) / len(files))
    with open(os.path.join(tru, "renders.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRUTH_FIELDS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(root, "dataset.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["identity", "file", "license", "author", "commons_url", "yaw", "face_frac", "sharpness", "date"])
        w.writeheader(); w.writerows(data)
    log("")
    log(f"Finished: {len(rows)} renders of {len(files) - len(missed)} faces under {len(lights)} lights in {time.time() - t0:.0f} s"
        + (f"; skipped: {', '.join(missed)}" if missed else "") + ".")
    log("Written to 3D/Original/ and 3D/truth/.")
    return rows


def run_3d(project, files, lights, poses, source="Original", log=print, progress=None):
    """Older free-form runner (any light, turned heads) writing to Original3D/. The app uses run_render_set."""
    src, dst = os.path.join(project, source), os.path.join(project, "Original3D")
    os.makedirs(dst, exist_ok=True)
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert("RGB")
        rec = reconstruct(img)
        stem = os.path.splitext(f)[0]
        if rec is None:
            log(f"[{f}] no face found"); continue
        overlay(img, rec["verts"], rec["tris"]).save(os.path.join(dst, f"{stem}__mesh.png"))
        yaw = ""
        if rec["pose"] is not None:
            R = rec["pose"][:3, :3]; yaw = f"{np.degrees(np.arctan2(R[0, 2], R[2, 2])):.1f}"
        for pose in poses:
            v = rotate_yaw(rec["verts_fine"], POSES[pose])
            for light in lights:
                name = f"{stem}__{light.replace(' ', '_').replace('°','deg')}__{pose.replace(' ', '_').replace('°','deg').replace('−','-')}.png"
                render(v, rec["tris_fine"], LIGHTS[light], img.size).save(os.path.join(dst, name))
                rows.append({"file": name, "source": f, "light": light, "light_xyz": LIGHTS[light],
                             "pose": pose, "yaw_delta": POSES[pose], "photo_yaw_est": yaw,
                             "method": "mediapipe-facemesh-468+delaunay+lambert"})
        log(f"[{i+1}/{len(files)}] {f}: {len(lights) * len(poses)} renders")
        if progress: progress((i + 1) / len(files))
    with open(os.path.join(project, "renders.csv"), "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    log(f"Done: {len(rows)} renders in {time.time() - t0:.0f}s -> {dst}")
    return rows
