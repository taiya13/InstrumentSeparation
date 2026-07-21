"""
run_exp003_combined.py — Experiment 003: do SCORE and POSITION complement each other?

Compares 4 conditions under identical settings (same violin synthesis, SR/STFT, metrics as
Exp 001/002): audio-only (baseline), Score-only, Position-only, Score+Position.

Single hypothesis (H1-003):
    Combining Score and Position covers each single cue's failure mode — Score fails at
    UNISON (same pitch), Position fails at CLOSE position — so Score+Position matches the
    better single cue in every case and stays robust across BOTH failure modes; only when
    BOTH cues are degenerate (unison AND co-located) does it fail (no information exists).

Combination = multiplicative fusion of the two soft-assignment weight maps:
    w_i = w_score_i * w_position_i ;  mask_i = w_i / sum_j w_j
A bin is claimed by target i only where BOTH pitch and position agree, so if one cue is
ambiguous the other dominates -> automatic complementarity.

The 2x2 stress test (hardest cases requested):
    A diff-pitch  + separated : both cues informative (easy)
    B UNISON      + separated : Score's weakness  -> Position should save it
    C diff-pitch  + CLOSE     : Position's weakness -> Score should save it
    D UNISON      + CLOSE     : both degenerate     -> unsolvable (honest limit)

Torch-free, training-free (analytic masks), runs in seconds on CPU.
"""
import os
import json
import numpy as np

from run_phase6_analytic import (render_line, stft, istft, score_map,
                                  SR, N_NOTES, NOTE_DUR, SCALE)
from run_exp002_position import pan_gains
from metrics import si_sdr

try:
    import mir_eval
    HAVE_MIR = True
except Exception:
    HAVE_MIR = False

SIGMA = 0.5          # position ILD soft-assignment width (same as Exp 002)
FLOOR = 1e-3         # score-map floor (same as Exp 001)
SEP_FAR = 0.3        # "separated" pan (as Exp 002 primary)
SEP_NEAR = 0.05      # "close" pan (near co-located)
N_TEST = 16

SCENARIOS = {
    "A_diff_separated":  dict(unison=False, sep=SEP_FAR),
    "B_unison_separated": dict(unison=True,  sep=SEP_FAR),   # Score weak
    "C_diff_close":      dict(unison=False, sep=SEP_NEAR),   # Position weak
    "D_unison_close":    dict(unison=True,  sep=SEP_NEAR),   # both weak (hardest)
}
METHODS = ["audio_only", "score", "position", "score_position"]


def gen_scenario(n, seed, unison):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)                       # make render detune reproducible too
    pool = []
    for _ in range(n):
        def walk():
            idx = int(rng.integers(0, len(SCALE)))
            seq = []
            for _ in range(N_NOTES):
                idx = int(np.clip(idx + rng.integers(-2, 3), 0, len(SCALE) - 1))
                seq.append(SCALE[idx])
            return seq
        m1 = walk()
        m2_raw = walk()                        # always drawn (keeps rng aligned across scenarios)
        m2 = list(m1) if unison else m2_raw
        s1, s2 = render_line(m1), render_line(m2)
        s1 = s1 / (np.max(np.abs(s1)) + 1e-9) * 0.8
        s2 = s2 / (np.max(np.abs(s2)) + 1e-9) * 0.8
        pool.append(dict(stems=[s1, s2], notes=[m1, m2]))
    return pool


def separate(pool, sep, method):
    """All methods mask the SAME signal MID=(L+R)/2 -> differences are purely the mask."""
    gL1, gR1 = pan_gains(-sep)
    gL2, gR2 = pan_gains(+sep)
    rho = (np.log(gR1 / gL1), np.log(gR2 / gL2))
    out = []
    for pc in pool:
        s1, s2 = pc["stems"]
        L = gL1 * s1 + gL2 * s2
        R = gR1 * s1 + gR2 * s2
        mid = (L + R) / 2.0
        if method == "audio_only":
            out.append([mid, mid])
            continue
        Mf = stft(mid)
        T = Mf.shape[1]
        ws = wp = None
        if method in ("score", "score_position"):
            ws = [score_map(pc["notes"][i], T) + FLOOR for i in range(2)]
        if method in ("position", "score_position"):
            Lf, Rf = stft(L), stft(R)
            obs = np.log((np.abs(Rf) + 1e-9) / (np.abs(Lf) + 1e-9))
            obs = obs[:, :T]
            wp = [np.exp(-((obs - rho[i]) ** 2) / (2 * SIGMA ** 2)) for i in range(2)]
        if method == "score":
            w = ws
        elif method == "position":
            w = wp
        else:
            w = [ws[i] * wp[i] for i in range(2)]
        denom = w[0] + w[1] + 1e-9
        out.append([istft((w[i] / denom) * Mf, len(s1)) for i in range(2)])
    return out


def metrics(pool, ests):
    per = [[], []]
    bss = []
    for pc, es in zip(pool, ests):
        for i in range(2):
            per[i].append(si_sdr(pc["stems"][i], es[i]))
        if HAVE_MIR:
            n = min(len(pc["stems"][0]), len(es[0]))
            R = np.stack([pc["stems"][0][:n], pc["stems"][1][:n]])
            E = np.stack([es[0][:n], es[1][:n]])
            if min(np.sum(R[0] ** 2), np.sum(R[1] ** 2)) > 1e-8:
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                    R, E, compute_permutation=False)
                bss.append((sdr.mean(), sir.mean(), sar.mean()))
    r = dict(si_sdr=round(float(np.mean(per[0] + per[1])), 2),
             vln1=round(float(np.mean(per[0])), 2),
             vln2=round(float(np.mean(per[1])), 2))
    if bss:
        b = np.mean(bss, axis=0)
        r.update(sdr=round(float(b[0]), 2), sir=round(float(b[1]), 2), sar=round(float(b[2]), 2))
    return r


def main(out="results/exp003_combined_results.json"):
    all_res = {}
    for sc, cfg in SCENARIOS.items():
        pool = gen_scenario(N_TEST, seed=999, unison=cfg["unison"])
        res = {}
        for m in METHODS:
            res[m] = metrics(pool, separate(pool, cfg["sep"], m))
        # improvements over audio-only floor, and complementarity check
        floor = res["audio_only"]["si_sdr"]
        res["_si_sdri"] = {m: round(res[m]["si_sdr"] - floor, 2) for m in METHODS}
        all_res[sc] = res

    payload = dict(
        hypothesis="Score and Position complement each other's failure modes; combined is "
                   "robust across both (fails only when both cues are degenerate)",
        config=dict(sep_far=SEP_FAR, sep_near=SEP_NEAR, sigma=SIGMA, n_test=N_TEST),
        scenarios=all_res)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2, ensure_ascii=False)

    print("SI-SDR (dB) by scenario x method  [vln I/II, held-out 16 pieces]")
    print(f"{'scenario':20s} {'audio':>7s} {'score':>7s} {'position':>9s} {'score+pos':>10s}")
    for sc in SCENARIOS:
        r = all_res[sc]
        print(f"{sc:20s} {r['audio_only']['si_sdr']:7.2f} {r['score']['si_sdr']:7.2f} "
              f"{r['position']['si_sdr']:9.2f} {r['score_position']['si_sdr']:10.2f}")
    print("\nSIR (dB) by scenario x method")
    print(f"{'scenario':20s} {'audio':>7s} {'score':>7s} {'position':>9s} {'score+pos':>10s}")
    for sc in SCENARIOS:
        r = all_res[sc]
        g = lambda m: r[m].get('sir', float('nan'))
        print(f"{sc:20s} {g('audio_only'):7.2f} {g('score'):7.2f} "
              f"{g('position'):9.2f} {g('score_position'):10.2f}")
    print(f"\nWrote {out}")
    return payload


if __name__ == "__main__":
    main()
