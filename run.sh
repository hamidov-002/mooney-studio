#!/usr/bin/env bash
# Start Mooney Studio with the environment that setup.sh made.
#   ./run.sh                       opens http://127.0.0.1:7860
#   ./run.sh --work ~/my-study     use another working folder for this start
#   ./run.sh --port 7870 --share   other options of app.py; ./run.sh --help lists them
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx mooney; then
    eval "$(conda shell.bash hook)"
    conda activate mooney
    PY="python"
else
    echo "No environment found. Run ./setup.sh first."
    exit 1
fi

exec $PY app.py "$@"
