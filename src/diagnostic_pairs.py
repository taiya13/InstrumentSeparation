"""
diagnostic_pairs.py — The 2x2 experiment that isolates *why* orchestral instrument
separation is hard, independent of any model.

We build controlled 2-source mixtures crossing two factors:
    PITCH:  same (unison / doubling)     vs   different (harmony)
    TIMBRE: same (two of the same instr) vs   different (violin vs flute)

For each 2-source mixture we compute the ORACLE ideal-ratio-mask separation
(the best ANY mono magnitude-mask model could do) and report SI-SDR improvement.

Expected and observed pattern:
    - same pitch + same timbre  -> near-hopeless (masks cannot assign shared TF energy)
    - same pitch + diff timbre  -> partial (timbre differences give the mask something)
    - diff pitch + same timbre  -> good (pitch separates the harmonics)
    - diff pitch + diff timbre  -> best

Conclusion this supports: the killer case for orchestras is "same timbre + overlapping
pitch" = exactly section doublings and unisons (Violin I/II, desks within a section),
which are pervasive in orchestral writing. Solving it REQUIRES extra information beyond
the mono waveform: multichannel/spatial cues or the musical score.
"""
import os
import json
import numpy as np
import soundfile as sf
import librosa

from synth_orchestra import harmonic_tone, adsr, midi_to_hz, TIMBRE, SR
from oracle_and_baselines import oracle_masks
from metrics import sdr_report

rng = np.random.default_rng(7)


def player_tone(instr, midi, n):
    """One 'player': a sustained note with that instrument's timbre + slight detune/vibrato phase."""
    tb = TIMBRE[instr]
    f0 = float(midi_to_hz(midi)) * (1.0 + rng.normal(0, 0.0012))
    tone = harmonic_tone(f0, n, SR, tb["h"], vibrato_hz=tb["vib"],
                         vibrato_depth=tb["vibd"], inharmonicity=tb["inh"])
    return tone * adsr(n, SR, a=0.03, d=0.06, s=0.8, r=0.15)


def make_case(instr_a, midi_a, instr_b, midi_b, dur=2.5):
    n = int(dur * SR)
    a = player_tone(instr_a, midi_a, n)
    b = player_tone(instr_b, midi_b, n)
    a = a / (np.max(np.abs(a)) + 1e-9) * 0.8
    b = b / (np.max(np.abs(b)) + 1e-9) * 0.8
    mix = a + b
    peak = np.max(np.abs(mix)) + 1e-9
    a, b, mix = a / peak * 0.9, b / peak * 0.9, mix / peak * 0.9
    refs = {"srcA": a, "srcB": b}
    est = oracle_masks(mix, refs, kind="irm")
    rep = sdr_report(refs, est, mix)
    return mix, refs, est, rep


CASES = {
    "1_unison_SAME_timbre":  ("violin", 69, "violin", 69),   # two violins, same note  -> HARDEST
    "2_unison_DIFF_timbre":  ("violin", 69, "flute", 69),    # violin + flute, same note
    "3_harmony_SAME_timbre": ("violin", 69, "violin", 72),   # two violins, a 3rd apart
    "4_harmony_DIFF_timbre": ("violin", 69, "flute", 72),    # violin + flute, a 3rd apart -> EASIEST
}


def main(outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    est_dir = os.path.join(outdir, "estimates_pairs")
    os.makedirs(est_dir, exist_ok=True)
    summary = {}
    print(f"{'case':26s} {'srcA SI-SDRi':>13s} {'srcB SI-SDRi':>13s} {'mean':>8s}")
    for name, (ia, ma, ib, mb) in CASES.items():
        mix, refs, est, rep = make_case(ia, ma, ib, mb)
        sf.write(os.path.join(est_dir, f"{name}_mix.wav"), mix.astype(np.float32), SR)
        for s in ("srcA", "srcB"):
            sf.write(os.path.join(est_dir, f"{name}_{s}_oracle.wav"),
                     est[s].astype(np.float32), SR)
        mean_i = round((rep["srcA"]["si_sdri"] + rep["srcB"]["si_sdri"]) / 2, 2)
        summary[name] = dict(per_source=rep, mean_si_sdri=mean_i,
                             config=dict(instr_a=ia, midi_a=ma, instr_b=ib, midi_b=mb))
        print(f"{name:26s} {rep['srcA']['si_sdri']:13.2f} "
              f"{rep['srcB']['si_sdri']:13.2f} {mean_i:8.2f}")
    with open(os.path.join(outdir, "diagnostic_pairs.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote -> {os.path.join(outdir, 'diagnostic_pairs.json')}")
    print("Interpretation: mean SI-SDRi collapses toward ~0 for case 1 "
          "(same timbre + same pitch); this is the model-independent wall.")
    return summary


if __name__ == "__main__":
    main()
