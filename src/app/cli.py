"""
cli.py — Headless runner for the separation engine (same engine the GUI uses).

Useful for: testing without a display, batch research over many files, and CI.

    python src/app/cli.py --input song.flac --output out --model roformer_baseline
    python src/app/cli.py --input mix.wav --output out --model tiny_masker \
        --reference data/synth_orchestra/manifest.json     # also evaluate
    python src/app/cli.py --list-models
"""
import os
import sys
import argparse
import json

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from app.engine import SeparationEngine   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_SRC, "..", "configs", "app_default.yaml"))
    ap.add_argument("--input")
    ap.add_argument("--output", default="separated")
    ap.add_argument("--model", default=None)
    ap.add_argument("--presence", default="all", choices=["all", "heuristic"])
    ap.add_argument("--reference", default=None, help="manifest.json for evaluation")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    eng = SeparationEngine(a.config)
    if a.list_models or not a.input:
        print("device:", eng.device, "| GPU:", eng.gpu_info.get("gpu_name"), eng.gpu_info.get("vram_gb"))
        print("available models:")
        for m in eng.list_models():
            print(f"  {m['key']:20s} trained={m['trained']} available={m['available']}  {m['display']}")
        if not a.input:
            return

    model = a.model or eng.list_models()[0]["key"]
    rec = eng.separate_file(a.input, a.output, model, presence_mode=a.presence,
                            reference_manifest=a.reference,
                            progress=lambda f, m: print(f"  [{int(f*100):3d}%] {m}"))
    print("\n=== RESULT ===")
    if rec.get("error"):
        print("ERROR:", rec["error"]["type"], rec["error"]["message"])
        return
    mt = rec["metrics"]
    print(f"model         : {rec['model']['display']} ({rec['model']['backbone']}, "
          f"trained={rec['model']['trained']}, {rec['model']['params_m']}M)")
    print(f"device        : {rec['device']}")
    print(f"inference time: {mt['inference_time_sec']} s  (RTF {mt['realtime_factor']}, "
          f"audio {mt['audio_duration_sec']}s)")
    g = mt["gpu"]
    print(f"GPU           : util avg/peak = {g['gpu_util_avg_pct']}/{g['gpu_util_peak_pct']} % | "
          f"VRAM peak = {g['vram_used_peak_mb']} / {g['vram_total_mb']} MB  (src={g['source']})")
    print(f"instruments   : {[i['id']+':'+i['name_ja'] for i in rec['instruments']]}")
    print(f"outputs       : {rec['output_subdir']}/")
    if rec.get("evaluation"):
        print(f"evaluation    : {json.dumps(rec['evaluation']['per_category'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
