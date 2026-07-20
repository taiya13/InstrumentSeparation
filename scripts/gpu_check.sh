#!/usr/bin/env bash
# Report GPU / VRAM / CUDA / driver and the auto-selected production config.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
python src/mss/gpu.py
