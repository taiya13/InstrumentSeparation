#!/usr/bin/env bash
# Phase 2: validate the taxonomy + build a manifest + run presence + evaluation.
# Depends on the Phase 1 synthetic bench (run scripts/run_poc.sh first if missing).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q pyyaml jsonschema >/dev/null 2>&1 || true

if [ ! -f data/synth_orchestra/mixture.wav ]; then
  echo "[pre] generating Phase 1 bench (missing)"; python src/synth_orchestra.py
fi

echo "=== [1/4] validate taxonomy ==="
python src/taxonomy.py
echo; echo "=== [2/4] build + schema-validate manifest ==="
python src/make_manifest.py
echo; echo "=== [3/4] presence detection demo ==="
python src/presence.py 2>/dev/null
echo; echo "=== [4/4] instrument-aware evaluation (oracle demo) ==="
python src/evaluation.py --manifest data/synth_orchestra/manifest.json --demo-oracle 2>/dev/null

echo; echo "Done. Foundation artifacts: configs/*.yaml, schemas/*.json, results/eval_*.{json,md}"
