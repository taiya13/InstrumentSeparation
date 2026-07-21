"""
run_exp002_position.py — Experiment 002: POSITION-only conditioning for same-instrument
(Violin I/II) separation. NO score is used anywhere.

Kept DIRECTLY comparable to Experiment 001 (src/run_phase6_analytic.py): identical violin
lines / test pieces (gen_pool seed 999), identical SR/NFFT/HOP, identical metrics, and the
identical mono "audio-only floor" baseline. The ONLY change is the conditioning signal:
Score (Exp 001) -> Position (Exp 002).

Single hypothesis (H1-002):
    Using ONLY Position — each violin placed at a different azimuth so the STEREO mixture
    carries interchannel level differences (ILD) — as the conditioning cue lets a separator
    split two identical-timbre violins significantly better than mono audio alone, WITHOUT
    any score information.

Definition of Position (see report §1):
    - Spatial info used : interchannel level difference (ILD) from constant-power panning
      (pan-pot stereo, as in most orchestral stereo mixes). Each target has a known pan.
    - Representation    : per-target expected interchannel log-level-ratio rho_i = log(gR/gL).
    - Why it helps      : at TF bins dominated by one source, the observed interchannel ratio
      reveals which position (source) that energy came from -> assign the bin accordingly.
      This works for identical timbres because it uses WHERE, not WHAT the sound is.

Method (analytic, torch-free, training-free — mirrors Exp 001's analytic mask):
    obs(f,t) = log(|R|/|L|);  w_i = exp(-(obs - rho_i)^2 / 2 sigma^2);  mask_i = w_i / sum_j w_j
    est_i = iSTFT(mask_i * MID),  MID = (L+R)/2   (SI-SDR is scale-invariant)

Baseline = audio-only floor = mono mixture (identical to Exp 001).
"""
import os
import json
import numpy as np
import librosa

from run_phase6_analytic import gen_pool, stft, istft, score_method, separate, SR

PRIMARY_SEP = 0.3          # primary pan separation (Vln I at -0.3, Vln II at +0.3)
SWEEP = [0.1, 0.2, 0.3, 0.5, 0.8]
SIGMA = 0.5               # width of the ILD soft-assignment (log-ratio units)


def pan_gains(p):
    a = (p + 1) * np.pi / 4          # constant-power pan, p in [-1, 1]
    return np.cos(a), np.sin(a)      # gL, gR


def make_stereo(s1, s2, pan1, pan2):
    gL1, gR1 = pan_gains(pan1)
    gL2, gR2 = pan_gains(pan2)
    L = gL1 * s1 + gL2 * s2
    R = gR1 * s1 + gR2 * s2
    rho = (np.log(gR1 / gL1), np.log(gR2 / gL2))   # expected interchannel log-ILD per target
    return L, R, rho


def position_separate(pool, sep, sigma=SIGMA):
    """POSITION-only soft mask from the stereo ILD cue. No score used."""
    pan1, pan2 = -sep, +sep
    out = []
    for pc in pool:
        s1, s2 = pc["stems"]
        L, R, rho = make_stereo(s1, s2, pan1, pan2)
        Lf, Rf = stft(L), stft(R)
        mid = stft((L + R) / 2.0)
        obs = np.log((np.abs(Rf) + 1e-9) / (np.abs(Lf) + 1e-9))     # observed log-ILD
        w = [np.exp(-((obs - r) ** 2) / (2 * sigma ** 2)) for r in rho]
        denom = w[0] + w[1] + 1e-9
        ests = [istft((w[i] / denom) * mid, len(s1)) for i in range(2)]
        out.append(ests)
    return out


def main(out="results/exp002_position_results.json", n_test=16):
    pool = gen_pool(n_test, seed=999)            # IDENTICAL pieces to Experiment 001
    print(f"Held-out test: {n_test} pieces (same as Exp 001), two identical-timbre violins\n")

    floor = score_method(pool, [[pc["mix"], pc["mix"]] for pc in pool])   # audio-only (mono)
    oracle = score_method(pool, separate(pool, "oracle"))                 # upper bound (mono IRM)
    primary = score_method(pool, position_separate(pool, PRIMARY_SEP))

    sweep = {}
    for sep in SWEEP:
        sweep[f"pan_pm_{sep}"] = score_method(pool, position_separate(pool, sep))

    imp = {k: round(primary[k] - floor[k], 2)
           for k in ("si_sdr", "si_sdr_vln1", "si_sdr_vln2")}
    for k in ("sdr", "sir", "sar"):
        if k in primary and k in floor:
            imp[k] = round(primary[k] - floor[k], 2)

    payload = dict(
        hypothesis="Position (ILD) only enables same-instrument (Vln I/II) separation from a "
                   "stereo mix, without any score",
        method="analytic ILD soft mask (torch-free, training-free); constant-power panning",
        n_test=n_test, primary_pan_separation=PRIMARY_SEP, sigma=SIGMA,
        results=dict(audio_only_floor=floor, position_informed=primary, oracle_irm=oracle),
        improvement_position_vs_audio_only=imp,
        pan_separation_sweep=sweep)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2, ensure_ascii=False)

    def row(name, r):
        s = (f"{name:20s} SI-SDR={r['si_sdr']:6.2f}  SI-SDRi={r['si_sdri']:6.2f}  "
             f"vln1={r['si_sdr_vln1']:6.2f} vln2={r['si_sdr_vln2']:6.2f}")
        if "sdr" in r:
            s += f"  SDR={r['sdr']:.2f} SIR={r['sir']:.2f} SAR={r['sar']:.2f}"
        return s
    print("===== Experiment 002: Position-only (held-out test) =====")
    print(row("audio_only_floor", floor))
    print(row(f"position (pan +-{PRIMARY_SEP})", primary))
    print(row("oracle_irm", oracle))
    print("\nPosition improvement over audio-only floor:", imp)
    print("\n--- pan-separation sweep (SI-SDR / SIR) ---")
    for k, r in sweep.items():
        print(f"  {k:12s} SI-SDR={r['si_sdr']:6.2f}  SIR={r.get('sir','-')}")
    print(f"\nWrote {out}")
    return payload


if __name__ == "__main__":
    main()
