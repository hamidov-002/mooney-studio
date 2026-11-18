#!/usr/bin/env bash
# One-command setup for macOS and Linux:   ./setup.sh
#
# Creates the Python environment and installs everything in requirements.txt.
#   - with conda installed: a conda environment called "mooney" (Python 3.11)
#   - otherwise:            a virtual environment in .venv/   (force this with  ./setup.sh --venv)
# It is safe to run again: an existing environment is reused and only missing packages are installed.
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME=mooney
USE_VENV=0
[ "${1:-}" = "--venv" ] && USE_VENV=1

if [ "$USE_VENV" = 0 ] && command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Using the existing conda environment '$ENV_NAME'."
    else
        echo "Creating the conda environment '$ENV_NAME' (Python 3.11)..."
        conda create -y -n "$ENV_NAME" python=3.11
    fi
    PY="conda run --no-capture-output -n $ENV_NAME python"
    ACTIVATE="conda activate $ENV_NAME"
else
    BASE=""
    for c in python3.11 python3.12 python3.13 python3; do
        if command -v "$c" >/dev/null 2>&1; then BASE="$c"; break; fi
    done
    if [ -z "$BASE" ]; then
        echo "No Python found. Install Python 3.11 (https://www.python.org/downloads/) or Miniconda, then run ./setup.sh again."
        exit 1
    fi
    if ! "$BASE" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
        echo "Note: $("$BASE" --version) found. The tool is tested with Python 3.11 to 3.13; some packages may not install on other versions."
    fi
    if [ -x .venv/bin/python ]; then
        echo "Using the existing environment in .venv/."
    else
        echo "Creating a virtual environment in .venv/ with $("$BASE" --version)..."
        "$BASE" -m venv .venv
    fi
    PY=".venv/bin/python"
    ACTIVATE="source .venv/bin/activate"
fi

echo "Installing the packages from requirements.txt (the first time this takes a few minutes)..."
$PY -m pip install --quiet --upgrade pip
$PY -m pip install -r requirements.txt

echo
$PY -c '
import importlib
missing = []
for name in ("gradio", "numpy", "PIL", "scipy", "cv2", "onnxruntime", "rembg", "mediapipe", "matplotlib", "requests", "h5py"):
    try:
        importlib.import_module(name)
    except Exception as e:
        missing.append(name + " (" + e.__class__.__name__ + ")")
print("All packages import correctly." if not missing else "These packages did not import: " + ", ".join(missing))
'

cat <<MSG

Setup finished. To start:

    ./run.sh                    opens at http://127.0.0.1:7860

Or by hand:   $ACTIVATE   and then   python app.py

Your photos and results go to the working folder ~/MooneyStudio, not into this folder.
Choose another one on the Overview page of the app, or with  ./run.sh --work FOLDER
Model files (about 300 MB in total) are downloaded the first time each step is used.
More settings: copy config.example.json to config.json and change what you need.
MSG
