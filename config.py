"""
Settings in one place.

Every value below is a default. To change one without editing this file, create `config.json` next to it
with only the keys you want to change (see config.example.json), for example:

    {"TARGET_PHOTOS": 60, "OLLAMA_MODEL": "moondream"}

Where your data goes: WORK_DIR is the working folder. The project (photos, Mooneys, results) is kept in
<WORK_DIR>/project and saved runs in <WORK_DIR>/runs, unless PROJECT_DIR or RUNS_DIR name another place.
The working folder can also be changed on the Overview page of the app, or with  ./run.sh --work PATH.

Environment variables win over both: MOONEY_WORK (the working folder), MOONEY_PROJECT (the project folder),
MOONEY_RUNS (the folder for saved runs) and OLLAMA_HOST; for running on another machine MOONEY_HOST,
MOONEY_PORT, MOONEY_SHARE, MOONEY_USER and MOONEY_PASSWORD.

Other modules read the values as attributes:   import config as cfg;  cfg.TARGET_PHOTOS
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (default, what it is)
SETTINGS = {
    # ---- where your data is kept (never inside the code folder by default)
    "WORK_DIR": ("~/MooneyStudio", "working folder: holds project/, runs/ and, if you keep any, notes/"),
    "PROJECT_DIR": ("", "folder with the photos, Mooneys and results; empty = <WORK_DIR>/project"),
    "RUNS_DIR": ("", "folder for saved runs (frozen copies of Mooneys and results); empty = <WORK_DIR>/runs"),
    "TARGET_PHOTOS": (100, "how many photos the dataset should have"),
    "TARGET_SUBSET": (10, "how many pairs go into Subset/"),

    # ---- Mooney generation (first-start defaults; the app saves later changes in project/settings.json)
    "MOONEY_SIGMA": (6.0, "blur, standard deviation in pixels at 512 px"),
    "MOONEY_WHITE": (23.0, "white share, per cent of the face"),
    "MOONEY_BACKGROUND": ("black", "background of the Mooney: black, white or auto"),
    "MOONEY_BG_LEVEL": (146, "gray value given to the background before blurring; it only changes the outline of the person"),
    "MOONEY_SEPARATE": (True, "separate the person from the background first"),
    "MOONEY_WHITE_RULE": ("fixed", "fixed = same white share for all; brightness = more white for bright faces"),
    "MOONEY_BRIGHT_WHITE": (35.0, "white share for bright faces, per cent (brightness rule only)"),
    "MOONEY_BRIGHT_BORDER": (124.0, "mean gray value of the face from which a face counts as bright"),
    "MOONEY_PREPROCESS": ("none", "step before the blur: none or clahe"),

    # ---- dataset collection (Wikimedia Commons)
    "COLLECT_PER_PERSON": (2, "photos per person"),
    "COLLECT_CANDIDATES": (80, "files looked at per person"),
    "COLLECT_MIN_FACE": (0.04, "smallest face, as a share of the photo's area"),
    "COLLECT_MIN_CROP_PX": (480, "the crop must be at least this wide in the download (no upscaling)"),
    "COLLECT_MIN_SHARPNESS": (40.0, "lowest accepted sharpness of the face"),
    "COLLECT_BRIGHTNESS": ([60, 200], "accepted range of the face's mean gray value"),
    "COLLECT_MIN_CONTRAST": (28.0, "lowest accepted standard deviation of the face's gray values"),
    "COLLECT_CONTACT": ("", "how Wikimedia can reach you (a web page or an e-mail address); sent with every request to Commons"),

    # ---- vision systems
    "DETECT_MIN_CONFIDENCE": (0.6, "a detection counts as a face from this confidence"),
    "DETECT_LONG_SIDE": (640, "images are resized to this longer side before face detection"),
    "REBLUR_SIGMA": (8.0, "blur of the re-blurred Mooney, pixels at 512 px width"),
    "SFS_SLANT_DEG": (45.0, "depth from shading: light angle away from the viewing direction"),
    "SFS_TILT_DEG": (90.0, "depth from shading: light angle in the image plane (90 = from above)"),
    "SFS_ITERATIONS": (120, "depth from shading: number of iterations"),
    "SFS_SMOOTH_SIGMA_PX": (2.0, "depth from shading: smoothing before the calculation"),
    "SFS_WORKING_SIZE_PX": (256, "depth from shading: image size used for the calculation"),
    "OLLAMA_HOST": ("http://127.0.0.1:11434", "where the Ollama server listens"),
    "OLLAMA_MODEL": ("llava:7b", "vision-language model suggested when several are installed"),
    "VLM_SCOPE": ("subset", "language model runs on: subset (the ten pairs or the favourites) or all"),

    # ---- 3D faces
    "LIGHTS_3D": (["front", "left 45°", "right 45°", "top", "below"], "light directions rendered for every 3D face"),
    "PHOTOS_3D": (10, "how many people are suggested for the 3D step"),

    # ---- 3D faces from the Basel Face Model 2019 (the model file is not included; see README)
    "BASEL_MODEL": ("models/model2019_fullHead.h5", "Basel Face Model file, whole head"),
    "BASEL_MODEL_FACE": ("models/model2019_face12.h5", "Basel Face Model file, face only"),
    "BASEL_FACES": (10, "how many invented faces are drawn (seeds 1, 2, 3, ...)"),
    "BASEL_SCALE": (0.9, "coefficient scale: 0 = the mean face, 1 = the model's full variation"),
    "BASEL_SHADOWS": (True, "draw cast shadows (the nose shading the cheek)"),
    "BASEL_FIT_COMPONENTS": (80, "fitting the Basel model to a photo: how many shape components are fitted"),
    "BASEL_FIT_SIGMA_MM": (1.5, "fitting: expected landmark error in mm; larger = the face stays nearer the mean"),
    "BASEL_FIT_MAX_ERROR": (0.15, "fitting: a photo is skipped if the landmark error exceeds this share of the eye distance"),
    "BASEL_FIT_TONE": (True, "fitting: match the overall skin tone of the photo (one colour gain, no texture)"),

    # ---- serving
    "HOST": ("127.0.0.1", "address the app listens on; 0.0.0.0 makes it reachable from other machines"),
    "PORT": (0, "port of the app; 0 = the first free port from 7860"),
    "SHARE": (False, "create a temporary public gradio.live link"),
    "AUTH_USER": ("", "user name for a simple login (empty = no login)"),
    "AUTH_PASSWORD": ("", "password for that login"),
}

_ENV = {"MOONEY_WORK": "WORK_DIR", "MOONEY_PROJECT": "PROJECT_DIR", "MOONEY_RUNS": "RUNS_DIR", "OLLAMA_HOST": "OLLAMA_HOST", "MOONEY_HOST": "HOST",
        "MOONEY_PORT": "PORT", "MOONEY_SHARE": "SHARE", "MOONEY_USER": "AUTH_USER", "MOONEY_PASSWORD": "AUTH_PASSWORD"}


def _cast(value, default):
    """Bring a value from config.json or the environment to the type of the default."""
    try:
        if isinstance(default, bool):
            return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, list):
            return list(value)
        return str(value)
    except (TypeError, ValueError):
        return default


def load():
    """Defaults, then config.json, then environment variables. Unknown keys in config.json are ignored."""
    values = {k: v for k, (v, _) in SETTINGS.items()}
    problems = []
    path = os.path.join(HERE, "config.json")
    if os.path.exists(path):
        try:
            for k, v in json.load(open(path, encoding="utf-8")).items():
                if k in values:
                    values[k] = _cast(v, SETTINGS[k][0])
                elif not k.startswith("_"):
                    problems.append(f"unknown setting {k!r}")
        except Exception as e:                                   # a broken file must not stop the app
            problems.append(f"config.json could not be read ({e})")
    for env, key in _ENV.items():
        if os.environ.get(env):
            values[key] = _cast(os.environ[env], SETTINGS[key][0])
    return values, problems


def _path(value, from_env):
    """~ is the home folder. Relative paths in config.json are relative to this folder; from the environment, to where you are."""
    value = os.path.expanduser(value)
    return os.path.abspath(value) if from_env or os.path.isabs(value) else os.path.abspath(os.path.join(HERE, value))


_values, PROBLEMS = load()
globals().update(_values)
WORK_DIR = _path(_values["WORK_DIR"], bool(os.environ.get("MOONEY_WORK")))
PROJECT_DIR = (_path(_values["PROJECT_DIR"], bool(os.environ.get("MOONEY_PROJECT"))) if _values["PROJECT_DIR"]
               else os.path.join(WORK_DIR, "project"))
RUNS_DIR = (_path(_values["RUNS_DIR"], bool(os.environ.get("MOONEY_RUNS"))) if _values["RUNS_DIR"]
            else os.path.join(WORK_DIR, "runs"))
NOTES_DIR = os.path.join(WORK_DIR, "notes")          # your own Markdown notes; shown in the app if there are any
CONFIG_FILE = os.path.join(HERE, "config.json")
BASEL_MODEL = _path(_values["BASEL_MODEL"], False)
BASEL_MODEL_FACE = _path(_values["BASEL_MODEL_FACE"], False)
for _p in PROBLEMS:
    print("config:", _p)


def set_work_dir(folder):
    """
    Make `folder` the working folder from the next start on: create it and write WORK_DIR to config.json.
    PROJECT_DIR and RUNS_DIR are taken out of config.json so that they follow the working folder.
    Returns the absolute path. Raises ValueError with a plain message if the folder cannot be used.
    """
    folder = (folder or "").strip().strip('"').strip("'")
    if not folder:
        raise ValueError("Enter a folder.")
    path = os.path.abspath(os.path.expanduser(folder))
    if os.path.isfile(path):
        raise ValueError(f"{path} is a file, not a folder.")
    if path == HERE or path.startswith(HERE + os.sep):
        raise ValueError("Choose a folder outside the code folder, so that your data stays separate from the app.")
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        open(probe, "w").close()
        os.remove(probe)
    except OSError as e:
        raise ValueError(f"{path} cannot be created or written to ({e.strerror or e}).")
    data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            data = json.load(open(CONFIG_FILE, encoding="utf-8"))
        except Exception:
            raise ValueError("config.json exists but cannot be read; fix or delete it first.")
    data.pop("PROJECT_DIR", None)
    data.pop("RUNS_DIR", None)
    data["WORK_DIR"] = path
    with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return path


def describe():
    """All settings with their current value and meaning, for the README and for debugging:  python config.py"""
    return [(k, globals()[k], SETTINGS[k][1]) for k in SETTINGS]


if __name__ == "__main__":
    for name, value, meaning in describe():
        print(f"{name:24s} {value!r:38}  {meaning}")
