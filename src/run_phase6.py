"""
run_phase6.py — Phase 6 original research: does SCORE conditioning enable same-instrument
(Violin I vs Violin II) separation that is impossible from mono audio alone?

Single hypothesis (H1):
    Two violins share an IDENTICAL timbre. From a MONO mix, an unconditioned extractor
    cannot tell which is which (Phase 1: oracle SI-SDRi ~+0.26 dB on unison+same timbre).
    Conditioning the extractor on each TARGET's score (a harmonic-comb pitch map derived
    from the notes it plays) should let it route the right partials -> separable.

Design (clean ablation): ONE model class (`ConditionedMasker`, 2 input channels =
[mixture log-mag, score map]). The conditioning is an ADDED INPUT MODULE; the conv
backbone is otherwise unchanged.
    - BASELINE   : trained/eval with the score channel ZEROED (audio only).
    - CONDITIONED: trained/eval with the real per-target score map.
Identical architecture, params, data, and training budget -> the ONLY variable is score.

Runs on CPU (no GPU, no downloads). Outputs results/phase6_results.json.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

from synth_orchestra import harmonic_tone, adsr, midi_to_hz, TIMBRE
from metrics import si_sdr

try:
    import mir_eval
    HAVE_MIR = True
except Exception:
    HAVE_MIR = False

SR = 16000
NFFT = 512
HOP = 256
FBINS = NFFT // 2 + 1
NOTE_DUR = 0.3
N_NOTES = 6
SCALE = [69, 71, 72, 74, 76, 77, 79, 81]   # A4..A5 (C major) — narrow band => frequent overlap
NOTE_N = int(NOTE_DUR * SR)
CLIP_N = NOTE_N * N_NOTES
HARMONICS = 6
DEVICE = "cpu"

# number of STFT frames for a CLIP_N-length signal (center=True)
_WIN = torch.hann_window(NFFT)
_T = torch.stft(torch.zeros(CLIP_N), NFFT, HOP, window=_WIN, return_complex=True).shape[-1]


def render_line(midis):
    tb = TIMBRE["violin"]
    segs = []
    for m in midis:
        f0 = float(midi_to_hz(m)) * (1 + np.random.normal(0, 0.001))
        tone = harmonic_tone(f0, NOTE_N, SR, tb["h"], vibrato_hz=tb["vib"],
                             vibrato_depth=tb["vibd"], inharmonicity=tb["inh"])
        segs.append(tone * adsr(NOTE_N, SR))
    return np.concatenate(segs)[:CLIP_N]


def build_score_map(midis):
    """Harmonic-comb time-frequency salience for a line of notes -> [F, T]."""
    m = np.zeros((FBINS, _T), dtype=np.float32)
    for t in range(_T):
        time = t * HOP / SR
        idx = min(int(time / NOTE_DUR), N_NOTES - 1)
        f0 = float(midi_to_hz(midis[idx]))
        for k in range(1, HARMONICS + 1):
            b = int(round(k * f0 * NFFT / SR))
            w = 1.0 / k
            for db, ww in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                bb = b + db
                if 0 <= bb < FBINS:
                    m[bb, t] += w * ww
    mx = m.max() + 1e-9
    return m / mx


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
        s1, s2, mix = s1 / p * 0.9, s2 / p * 0.9, mix / p * 0.9
        pool.append(dict(mix=mix.astype(np.float32),
                         stems=[s1.astype(np.float32), s2.astype(np.float32)],
                         scores=[build_score_map(m1), build_score_map(m2)],
                         notes=[m1, m2]))
    return pool


class ConditionedMasker(nn.Module):
    """2-ch input [log-mag mixture, score map] -> 1 sigmoid mask -> extracted target.
    The score channel is the ADDED conditioning module; the conv stack is unchanged."""
    def __init__(self, hidden=24):
        super().__init__()
        self.register_buffer("window", torch.hann_window(NFFT))
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, mix, score):                 # mix [B,T], score [B,F,Tf]
        spec = torch.stft(mix, NFFT, HOP, window=self.window, return_complex=True)  # [B,F,Tf]
        mag = torch.log1p(spec.abs()).unsqueeze(1)                                   # [B,1,F,Tf]
        inp = torch.cat([mag, score.unsqueeze(1)], dim=1)                            # [B,2,F,Tf]
        mask = torch.sigmoid(self.net(inp))[:, 0]                                    # [B,F,Tf]
        est = torch.istft(mask * spec, NFFT, HOP, window=self.window, length=mix.shape[-1])
        return est


def batch(pool, rng, bs, use_score):
    idx = rng.integers(0, len(pool), size=bs)
    tgt = rng.integers(0, 2, size=bs)
    mix = np.stack([pool[i]["mix"] for i in idx])
    ref = np.stack([pool[i]["stems"][t] for i, t in zip(idx, tgt)])
    if use_score:
        sc = np.stack([pool[i]["scores"][t] for i, t in zip(idx, tgt)])
    else:
        sc = np.zeros((bs, FBINS, _T), dtype=np.float32)
    return (torch.tensor(mix), torch.tensor(sc), torch.tensor(ref))


def train(pool, steps, use_score, lr=2e-3, bs=8, seed=0, tag=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ConditionedMasker().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(1, steps + 1):
        mix, sc, ref = batch(pool, rng, bs, use_score)
        est = model(mix, sc)
        n = min(est.shape[-1], ref.shape[-1])
        loss = torch.mean(torch.abs(est[..., :n] - ref[..., :n]))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(steps // 5, 1) == 0:
            print(f"  [{tag}] step {s}/{steps} loss={loss.item():.4f}")
    return model


def evaluate(model, pool, use_score):
    model.eval()
    per_tgt = {0: [], 1: []}
    mix_base = {0: [], 1: []}
    conf = 0
    bss = []
    with torch.no_grad():
        for pc in pool:
            mix = torch.tensor(pc["mix"])[None]
            ests = []
            for t in (0, 1):
                sc = torch.tensor(pc["scores"][t])[None] if use_score \
                    else torch.zeros(1, FBINS, _T)
                est = model(mix, sc)[0].numpy()
                ests.append(est)
                per_tgt[t].append(si_sdr(pc["stems"][t], est))
                mix_base[t].append(si_sdr(pc["stems"][t], pc["mix"]))
            # identity confusion: does swapping estimates score better?
            idn = si_sdr(pc["stems"][0], ests[0]) + si_sdr(pc["stems"][1], ests[1])
            swp = si_sdr(pc["stems"][0], ests[1]) + si_sdr(pc["stems"][1], ests[0])
            conf += int(swp > idn)
            if HAVE_MIR:
                n = min(len(pc["stems"][0]), len(ests[0]))
                R = np.stack([pc["stems"][0][:n], pc["stems"][1][:n]])
                E = np.stack([ests[0][:n], ests[1][:n]])
                if min(np.sum(R[0] ** 2), np.sum(R[1] ** 2)) > 1e-8:
                    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                        R, E, compute_permutation=False)
                    bss.append((sdr.mean(), sir.mean(), sar.mean()))
    all_sisdr = per_tgt[0] + per_tgt[1]
    all_base = mix_base[0] + mix_base[1]
    out = dict(
        si_sdr=round(float(np.mean(all_sisdr)), 2),
        si_sdri=round(float(np.mean(all_sisdr) - np.mean(all_base)), 2),
        si_sdr_vln1=round(float(np.mean(per_tgt[0])), 2),
        si_sdr_vln2=round(float(np.mean(per_tgt[1])), 2),
        identity_confusion_rate=round(conf / len(pool), 3),
        n_test=len(pool))
    if bss:
        b = np.mean(bss, axis=0)
        out.update(sdr=round(float(b[0]), 2), sir=round(float(b[1]), 2),
                   sar=round(float(b[2]), 2))
    return out


def main(steps=250, out="results/phase6_results.json"):
    print(f"Generating data pools ({_T} STFT frames/clip, {CLIP_N/SR:.1f}s clips)...")
    train_pool = gen_pool(24, seed=1)
    test_pool = gen_pool(8, seed=999)           # held-out
    print(f"train={len(train_pool)} test={len(test_pool)} pieces\n")

    print("== Training BASELINE (audio only, score channel zeroed) ==")
    base = train(train_pool, steps, use_score=False, seed=0, tag="base")
    print("== Training CONDITIONED (score map) ==")
    cond = train(train_pool, steps, use_score=True, seed=0, tag="cond")

    r_base = evaluate(base, test_pool, use_score=False)
    r_cond = evaluate(cond, test_pool, use_score=True)

    delta = {k: round(r_cond[k] - r_base[k], 2)
             for k in ("si_sdr", "si_sdri", "si_sdr_vln1", "si_sdr_vln2")
             if k in r_base and k in r_cond}
    for k in ("sdr", "sir", "sar"):
        if k in r_base and k in r_cond:
            delta[k] = round(r_cond[k] - r_base[k], 2)
    result = dict(hypothesis="score conditioning enables same-instrument (Vln I/II) "
                             "separation from mono", steps=steps,
                  baseline=r_base, conditioned=r_cond, improvement=delta)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)

    def row(name, r):
        return (f"{name:12s} SI-SDR={r['si_sdr']:6.2f} SI-SDRi={r['si_sdri']:6.2f} "
                f"vln1={r['si_sdr_vln1']:6.2f} vln2={r['si_sdr_vln2']:6.2f} "
                f"conf={r['identity_confusion_rate']:.2f}"
                + (f" SDR={r.get('sdr')} SIR={r.get('sir')} SAR={r.get('sar')}"
                   if 'sdr' in r else ""))
    print("\n===== RESULTS (held-out test) =====")
    print(row("BASELINE", r_base))
    print(row("CONDITIONED", r_cond))
    print("\nImprovement (conditioned - baseline):", delta)
    print(f"\nWrote {out}")
    return result


if __name__ == "__main__":
    main()
