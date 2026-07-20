#!/usr/bin/env bash
# Multi-GPU launch via torchrun (Distributed Training — DDP). The trainer is DDP-aware:
# rank-gated logging/checkpoint, checkpoints saved with the DDP wrapper unwrapped.
#
# Usage:
#   bash scripts/launch_distributed.sh 4 configs/prod/roformer_48gb.yaml myrun
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate

NGPU="${1:-2}"
CONFIG="${2:-configs/prod/roformer_48gb.yaml}"
NAME="${3:-ddp_run}"

torchrun --standalone --nproc_per_node="$NGPU" \
  src/run_pipeline.py --config "$CONFIG" --name "$NAME"
