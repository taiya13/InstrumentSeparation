"""
pipeline.py — Single-song inference pipeline + automatic evaluation hook.

run_song():  1 mixture  ->  Presence estimate  ->  model separation (active set only)
             ->  per-instrument stems on disk  ->  Phase 2 evaluation (SI-SDR/SDR/SIR/SAR).

This is the "put a song in, get evaluated per-instrument stems out" glue that ties the
whole platform (Taxonomy + Presence + Backbone + Evaluation) together.
"""
import os
import json
import numpy as np
import librosa
import soundfile as sf

from taxonomy import Taxonomy
from presence import OraclePresence, EnergyHeuristicPresence, select_active_set
import evaluation


def estimate_presence(manifest, target_level, tx, mode="oracle", threshold=0.15):
    """Return the active output-id set at target_level."""
    if mode == "oracle":
        present_ids = list(OraclePresence().predict(manifest).keys())
    else:
        y, _ = librosa.load(manifest["mixture"], sr=manifest["sample_rate"], mono=True)
        scores = EnergyHeuristicPresence(tx, sr=manifest["sample_rate"]).predict(y)
        present_ids = select_active_set(scores, tx, threshold=threshold)
    active = set()
    for pid in present_ids:
        lvl = tx.nodes[pid].level
        roll = pid if lvl == target_level else tx.rollup(pid, target_level)
        if roll:
            active.add(roll)
    return sorted(active)


def run_song(backbone, manifest_path, out_dir, target_level, presence_mode="oracle",
             evaluate=True):
    tx = Taxonomy()
    manifest = json.load(open(manifest_path))
    sr = manifest["sample_rate"]
    mix, _ = librosa.load(manifest["mixture"], sr=sr, mono=True)

    # 1) presence -> active output set
    active = estimate_presence(manifest, target_level, tx, mode=presence_mode)
    active = [a for a in active if a in backbone.output_targets]
    print(f"[pipeline] presence({presence_mode}) active@{target_level}: {active}")

    # 2) model separation (only active targets)
    stems = backbone.separate(mix, sr, targets=active)

    # 3) write per-instrument stems named by taxonomy id
    os.makedirs(out_dir, exist_ok=True)
    for tid, wav in stems.items():
        sf.write(os.path.join(out_dir, f"{tid}.wav"),
                 np.asarray(wav, dtype=np.float32), sr)
    print(f"[pipeline] wrote {len(stems)} stems -> {out_dir}/")

    # 4) automatic evaluation (SI-SDR/SDR/SIR/SAR) at target_level
    report = None
    if evaluate:
        report = evaluation.evaluate(manifest_path, estimates_dir=out_dir,
                                     demo_oracle=False,
                                     instrument_level=target_level,
                                     bss_level=target_level, out_dir=out_dir)
    return dict(active=active, stems=list(stems.keys()), report=report)
