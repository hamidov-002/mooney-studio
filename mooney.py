"""
Mooney image generation — core algorithm (no UI).

Pipeline (4 phases):
    1. background separation: segment the subject, paint the background one
       uniform gray level (the subject keeps its own shading)
    2. grayscale
    3. Gaussian blur with standard deviation `sigma` (pixels)
    4. global threshold chosen so that `white_fraction` of the measured region
       becomes white. The region is the person's pixels inside the face box when
       a face box is given, otherwise the whole person, otherwise the whole image.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- phase 1
_session = None


def subject_mask(img: Image.Image) -> np.ndarray | None:
    """
    Soft foreground mask in [0,1] (HxW float32) from a U²-Net segmentation
    model (package `rembg`). Returns None if rembg is not installed.
    """
    global _session
    try:
        from rembg import remove, new_session
    except ImportError:
        return None
    if _session is None:
        _session = new_session("u2net")           # ~170 MB, downloaded once
    rgba = remove(img.convert("RGB"), session=_session, only_mask=True)
    return np.asarray(rgba, dtype=np.float32) / 255.0


def separate_background(img: Image.Image, mask: np.ndarray | None,
                        bg_level: int = 128) -> Image.Image:
    """Composite the subject over a uniform gray background of value bg_level."""
    if mask is None:
        return img.convert("RGB")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    m = mask[..., None]
    out = rgb * m + bg_level * (1.0 - m)
    return Image.fromarray(out.round().astype(np.uint8))


# ---------------------------------------------------------------- phases 2-4
def resize_max(img: Image.Image, max_side: int | None) -> Image.Image:
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


PREPROCESS = ("none", "clahe")


def preprocess_gray(gray: np.ndarray, method: str = "none") -> np.ndarray:
    """
    Optional step between grayscale and blur. "none" is the default and the recommended setting.
    "clahe" = local contrast equalisation (OpenCV, clip 2.0, 8x8 tiles). In tests it made the Mooneys
    more fragmented and did not help bright faces; it is here to be testable.
    """
    if method == "clahe":
        try:
            import cv2
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        except ImportError:
            return gray
    return gray


def face_brightness(img: Image.Image, mask: np.ndarray | None, stat_region: np.ndarray | None) -> float | None:
    """Mean gray value (0-255) of the person's pixels inside the face box - the 'face brightness' of the notes."""
    if stat_region is None:
        return None
    g = to_gray(img)
    if stat_region.shape != g.shape:
        return None
    reg = stat_region & (mask > 0.5) if mask is not None and (stat_region & (mask > 0.5)).any() else stat_region
    return float(g[reg].mean()) if reg.any() else None


def gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return gray.copy()
    return np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.uint8)


def threshold_for_white_fraction(blurred: np.ndarray, white_fraction: float,
                                 region: np.ndarray | None = None) -> float:
    """
    The threshold t for which mean(blurred[region] > t) is as close as possible to white_fraction.
    `region` is a boolean mask (e.g. the face); None = whole image.

    All 256 gray levels are tried and the closest one wins. (An earlier version used a binary search
    that always stopped on the "not more than the target" side, so a large area of equal gray values
    could leave far less white than asked for - in the extreme, none.)
    """
    vals = blurred if region is None else blurred[region]
    if vals.size == 0:
        return 127.5
    levels = np.clip(np.rint(vals.ravel()), 0, 255).astype(np.int64)
    hist = np.bincount(levels, minlength=256)
    above = (hist[::-1].cumsum()[::-1] - hist) / levels.size       # share of pixels strictly brighter than level k
    return float(np.argmin(np.abs(above - white_fraction)))


def binarize(blurred: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(blurred > threshold, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------- driver
@dataclass
class MooneyResult:
    separated: np.ndarray     # RGB uint8, phase 1 output
    gray: np.ndarray          # uint8 HxW
    blurred: np.ndarray       # uint8 HxW
    mooney: np.ndarray        # uint8 HxW, values {0, 255}
    mask: np.ndarray | None   # float32 HxW or None
    threshold: float
    white_fraction: float     # achieved, measured on the subject (or whole image)
    sigma: float
    bg_level: int


def make_mooney(img: Image.Image, sigma: float = 4.0, white_fraction: float = 0.16,
                threshold: float | None = None, separate: bool = True,
                bg_level: int = 128, max_side: int | None = 1024,
                mask: np.ndarray | None = None, bg_mooney: str = "black",
                stat_region: np.ndarray | None = None, preprocess: str = "none") -> MooneyResult:
    """
    bg_mooney:   what the background becomes in the final Mooney when a mask exists:
                 "black", "white" or "auto" (whatever the threshold makes of bg_level).
    preprocess:  "none" (default) or "clahe"; see preprocess_gray.
    stat_region: boolean HxW mask on which the white fraction is measured (e.g. the face box);
                 defaults to the subject mask, or the whole image without a mask. When a subject
                 mask exists, only the part of stat_region that lies on the subject is counted.
    """
    img = resize_max(img, max_side)
    if separate and mask is None:
        mask = subject_mask(img)
    if not separate:
        mask = None
    separated = separate_background(img, mask, bg_level)        # phase 1
    gray = to_gray(separated)                                    # phase 2
    if preprocess != "none":                                     # optional: equalise the photo, then re-apply the flat background
        pre = preprocess_gray(to_gray(img), preprocess).astype(np.float32)
        gray = pre.astype(np.uint8) if mask is None else (pre * mask + bg_level * (1.0 - mask)).round().astype(np.uint8)
    blurred = gaussian_blur(gray, sigma)                         # phase 3
    region = (mask > 0.5) if mask is not None else None
    stat = stat_region if stat_region is not None else region
    if stat is not None and stat.shape != blurred.shape:
        stat = np.asarray(Image.fromarray(stat.astype(np.uint8) * 255).resize((blurred.shape[1], blurred.shape[0]))) > 127
    if stat_region is not None and region is not None and (stat & region).any():
        stat = stat & region        # count only the person inside the face box: background pixels there are
                                    # painted over afterwards and must not use up the white share
    if threshold is None:                                        # phase 4
        threshold = threshold_for_white_fraction(blurred, white_fraction, stat)
    mooney = binarize(blurred, threshold)
    if region is not None and bg_mooney in ("black", "white"):
        mooney[~region] = 0 if bg_mooney == "black" else 255
    achieved = float((mooney[stat] > 0).mean()) if stat is not None else float((mooney > 0).mean())
    return MooneyResult(np.asarray(separated), gray, blurred, mooney, mask,
                        float(threshold), achieved, float(sigma), int(bg_level))
