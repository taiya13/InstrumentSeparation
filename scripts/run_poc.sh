#!/usr/bin/env bash
# Reproduce the entire Phase 1 PoC that runs fully offline (CPU, no model downloads).
# Usage:  bash scripts/run_poc.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r environment/requirements-poc.txt

echo "==================================================================="
echo "[1/4] Generate controlled synthetic orchestra (ground-truth stems)"
echo "==================================================================="
python src/synth_orchestra.py

echo
echo "==================================================================="
echo "[2/4] Oracle separation ceilings + HPSS blind baseline"
echo "==================================================================="
python src/oracle_and_baselines.py

echo
echo "==================================================================="
echo "[3/4] 2x2 same-instrument diagnostic (the model-independent wall)"
echo "==================================================================="
python src/diagnostic_pairs.py

echo
echo "==================================================================="
echo "[4/4] HDemucs (torchaudio) CPU architecture smoke-test"
echo "==================================================================="
python src/hdemucs_smoketest.py 2>/dev/null || \
  echo "(skipped: torch/torchaudio not installed)"

echo
echo "Done. Metrics -> results/metrics.json, results/diagnostic_pairs.json"
echo "Audio    -> data/synth_orchestra/, results/estimates*/"
