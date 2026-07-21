#!/usr/bin/env bash
# One-command production run: autoselect config from GPU/VRAM, then
# data check -> train -> validation -> checkpoint/resume -> infer -> evaluate -> report.
#
# Usage:
#   bash scripts/train_gpu.sh                              # autoselect tier config
#   bash scripts/train_gpu.sh configs/prod/roformer_24gb.yaml myrun
#   bash scripts/train_gpu.sh "" myrun --resume checkpoints/prod_24gb/final.pt
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate

CONFIG="${1:-}"
NAME="${2:-}"
shift || true; shift || true

ARGS=()
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")
[ -n "$NAME" ]   && ARGS+=(--name "$NAME")
python src/run_pipeline.py "${ARGS[@]}" "$@"
