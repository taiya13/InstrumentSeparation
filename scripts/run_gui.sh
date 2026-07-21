#!/usr/bin/env bash
# Launch the research GUI on Linux/macOS (dev). Requires a display + python tkinter.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate || true
python src/app/gui.py "$@"
