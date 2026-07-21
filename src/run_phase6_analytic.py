"""
run_phase6_analytic.py — Torch-free, training-free verification of the Phase 6 hypothesis.

Hypothesis (H1): SCORE information enables same-instrument (Violin I/II) separation from a
MONO mixture, which audio alone cannot achieve.

This variant isolates the *information* question directly (no neural net, no training
budget to confound the result): build a soft mask purely from each target's SCORE
(harmonic-comb map from its notes) and measure how well it separates the two identical-
timbre violins — versus the audio-only floor (do nothing) and the oracle ceiling (mask
built from the true stems, an unfair upper bound that peeks at the answer).

    audio-only floor  : est = mixture                (no way to assign shared energy)
    SCORE-informed    : mask_i = score_i / sum_j score_j   (uses score, NOT the answer)
    oracle IRM        : mask_i = |S_i| / sum_j |S_j|       (uses true stems; upper bound)

Runs in seconds on CPU. Writes results/phase6_results.json.
"""
import os
import json
import numpy as np
import librosa

from synth_orchestra import harmonic_tone, adsr, midi_to_hz, TIMBRE
from metrics import si_sdr

try:
    import mir_eval
    HAVE_MIR = True
except Exception:
    HAVE_MIR = False

SR = 22050
NFFT = 1024
HOP = 256
NOTE_DUR = 0.375
N_NOTES = 8
HARMONICS = 8
SCALE = [69, 71, 72, 74, 76, 77, 79, 81]     # narrow band -> frequent pitch overlap/unison
NOTE_N = int(NOTE_DUR * SR)
CLIP_N = NOTE_N * N_NOTES


def render_line(midis):
    tb = TIMBRE["violin"]
    segs = []
    for m in midis:
        f0 = float(midi_to_hz(m)) * (1 + np.random.normal(0, 0.001))
        tone = harmonic_tone(f0, NOTE_N, SR, tb["h"], vibrato_hz=tb["vib"],
                             vibrato_depth=tb["vibd"], inharmonicity=tb["inh"])
        segs.append(tone * adsr(NOTE_N, SR))
    return np.concatenate(segs)[:CLIP_N]


def stft(x):
    return librosa.stft(x, n_fft=NFFT, hop_length=HOP)


def istft(X, n):
    return librosa.istft(X, hop_length=HOP, length=n)


def score_map(midis, n_frames):
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    F = len(freqs)
    m = np.zeros((F, n_frames), dtype=np.float64)
    for t in range(n_frames):
        time = t * HOP / SR
        idx = min(int(time / NOTE_DUR), N_NOTES - 1)
        f0 = float(midi_to_hz(midis[idx]))
        for k in range(1, HARMONICS + 1):
            b = int(round(k * f0 * NFFT / SR))
            for db, ww in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                bb = b + db
                if 0 <= bb < F:
                    m[bb, t] += ww / k
    return m


def gen_pool(n, seed):
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n):
        def walk():
            idx = int(rng.integers(0, len(SCALE)))
            seq = []
            for _ in range(N_NOTES):
                idx = int(np.clip(idx + rng.integers(-2, 3), 0, len(SCALE) - 1))
                seq.append(SCALE[idx])
            return seq
        m1, m2 = walk(), walk()
        s1, s2 = render_line(m1), render_line(m2)
        s1 = s1 / (np.max(np.abs(s1)) + 1e-9) * 0.8
        s2 = s2 / (np.max(np.abs(s2)) + 1e-9) * 0.8
        mix = s1 + s2
        p = np.max(np.abs(mix)) + 1e-9
        pool.append(dict(stems=[s1 / p * 0.9, s2 / p * 0.9], mix=mix / p * 0.9,
                         notes=[m1, m2]))
    return pool


def separate(pool, method, floor=1e-3):
    """Return per-target estimates for each piece using the given method."""
    out = []
    for pc in pool:
        mix = pc["mix"]
        X = stft(mix)
        T = X.shape[1]
        if method == "mixture":
            out.append([mix, mix])
            continue
        if method == "score":
            maps = [score_map(pc["notes"][i], T) + floor for i in range(2)]
        else:  # oracle
            maps = [np.abs(stft(pc["stems"][i])) + 1e-9 for i in range(2)]
            maps = [m[:, :T] for m in maps]
        denom = maps[0] + maps[1]
        ests = [istft((maps[i] / denom) * X, len(mix)) for i in range(2)]
        out.append(ests)
    return out


def score_method(pool, ests):
    per = [[], []]
    base = [[], []]
    bss = []
    for pc, es in zip(pool, ests):
        for i in range(2):
            per[i].append(si_sdr(pc["stems"][i], es[i]))
            base[i].append(si_sdr(pc["stems"][i], pc["mix"]))
        if HAVE_MIR:
            n = min(len(pc["stems"][0]), len(es[0]))
            R = np.stack([pc["stems"][0][:n], pc["stems"][1][:n]])
            E = np.stack([es[0][:n], es[1][:n]])
            if min(np.sum(R[0] ** 2), np.sum(R[1] ** 2)) > 1e-8:
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                    R, E, compute_permutation=False)
                bss.append((sdr.mean(), sir.mean(), sar.mean()))
    allv = per[0] + per[1]
    allb = base[0] + base[1]
    r = dict(si_sdr=round(float(np.mean(allv)), 2),
             si_sdri=round(float(np.mean(allv) - np.mean(allb)), 2),
             si_sdr_vln1=round(float(np.mean(per[0])), 2),
             si_sdr_vln2=round(float(np.mean(per[1])), 2))
    if bss:
        b = np.mean(bss, axis=0)
        r.update(sdr=round(float(b[0]), 2), sir=round(float(b[1]), 2), sar=round(float(b[2]), 2))
    return r


def main(out="results/phase6_results.json", n_test=16):
    pool = gen_pool(n_test, seed=999)
    print(f"Held-out test: {n_test} pieces, {CLIP_N/SR:.1f}s, two identical-timbre violins\n")
    methods = {"audio_only_floor": "mixture", "score_informed": "score", "oracle_irm": "oracle"}
    results = {}
    for name, meth in methods.items():
        results[name] = score_method(pool, separate(pool, meth))
    imp = {k: round(results["score_informed"][k] - results["audio_only_floor"][k], 2)
           for k in ("si_sdr", "si_sdr_vln1", "si_sdr_vln2")}
    for k in ("sdr", "sir", "sar"):
        if k in results["score_informed"] and k in results["audio_only_floor"]:
            imp[k] = round(results["score_informed"][k] - results["audio_only_floor"][k], 2)
    payload = dict(
        hypothesis="score conditioning enables same-instrument (Vln I/II) separation from mono",
        method="analytic score-informed soft mask (torch-free, training-free)",
        n_test=n_test, results=results,
        improvement_score_vs_audio_only=imp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2, ensure_ascii=False)

    def row(name, r):
        s = (f"{name:18s} SI-SDR={r['si_sdr']:6.2f}  SI-SDRi={r['si_sdri']:6.2f}  "
             f"vln1={r['si_sdr_vln1']:6.2f} vln2={r['si_sdr_vln2']:6.2f}")
        if "sdr" in r:
            s += f"  SDR={r['sdr']:.2f} SIR={r['sir']:.2f} SAR={r['sar']:.2f}"
        return s
    print("===== Phase 6 verification (held-out test) =====")
    for name in methods:
        print(row(name, results[name]))
    print("\nScore-informed improvement over audio-only floor:", imp)
    print(f"\nWrote {out}")
    return payload


if __name__ == "__main__":
    main()
