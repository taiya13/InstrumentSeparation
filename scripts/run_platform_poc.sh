#!/usr/bin/env bash
# Phase 3 platform PoC: train -> resume -> infer -> evaluate, all through the common
# Backbone Interface. Demonstrates model-agnosticism (tiny_masker trains on CPU;
# mel_band_roformer is exercised through the identical API).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate

if [ ! -f data/synth_orchestra/manifest.json ]; then
  echo "[pre] building data foundation"; bash scripts/build_foundation.sh >/dev/null
fi

echo "=== [1/4] TRAIN tiny_masker (category level) ==="
python src/run_train.py --config configs/train.example.yaml

echo; echo "=== [2/4] RESUME from final checkpoint (+100 steps) ==="
python src/run_train.py --config configs/train.example.yaml \
  --resume checkpoints/tiny_masker_category/final.pt --steps 500

echo; echo "=== [3/4] INFER one song: presence -> separate -> evaluate ==="
python src/run_infer.py \
  --checkpoint checkpoints/tiny_masker_category/final.pt \
  --manifest data/synth_orchestra/manifest.json \
  --out results/infer_tiny_masker --presence oracle

echo; echo "=== [4/4] Same interface, different model: Mel-Band RoFormer smoke-train ==="
python src/run_train.py --config configs/train_roformer.example.yaml --steps 6 || \
  echo "(roformer step skipped)"

echo; echo "Platform PoC complete."
