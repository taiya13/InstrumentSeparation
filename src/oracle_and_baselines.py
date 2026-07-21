"""
oracle_and_baselines.py — Evaluate separation ceilings and weight-free baselines.

Why "oracle" masks?
  An ideal ratio/binary mask is computed FROM the ground truth. It is the best any
  magnitude-mask-based separator (Spleeter, Open-Unmix, MDX, Demucs' spectral branch,
  RoFormer's masking, ...) could possibly achieve on a *mono* mixture. It is
  therefore a MODEL-INDEPENDENT UPPER BOUND. If the oracle can't separate two
  sources, no amount of model/architecture work will — you need extra information
  (multichannel/spatial cues, or the score). This directly quantifies Phase 0's
  central claim about same-instrument sections.

Baselines that need NO downloaded weights (so they run in this sandbox):
  - Mixture-as-estimate (do nothing) -> the floor.
  - Oracle IRM / IBM -> the ceiling.
  - HPSS (librosa) -> a real, classical blind method (harmonic vs percussive).

Real neural SOTA (HT-Demucs / RoFormer) is provided as a reproducible recipe in
src/run_demucs_reference.py and the Phase 1 report; its trained weights are blocked
by this sandbox's egress policy, so it is meant to be run on the team's GPU box.
"""
import os
import json
import glob
import numpy as np
import librosa
import soundfile as sf

from metrics import si_sdr, sdr_report

NFFT = 2048
HOP = 512


def load_stems(datadir):
    mix, sr = librosa.load(os.path.join(datadir, "mixture.wav"), sr=None, mono=True)
    stems = {}
    for p in sorted(glob.glob(os.path.join(datadir, "stem_*.wav"))):
        name = os.path.basename(p)[len("stem_"):-len(".wav")]
        s, _ = librosa.load(p, sr=None, mono=True)
        stems[name] = s
    families = {}
    for p in sorted(glob.glob(os.path.join(datadir, "family_*.wav"))):
        name = os.path.basename(p)[len("family_"):-len(".wav")]
        s, _ = librosa.load(p, sr=None, mono=True)
        families[name] = s
    return mix, stems, families, sr


def stft(x):
    return librosa.stft(x, n_fft=NFFT, hop_length=HOP)


def istft(X, length):
    return librosa.istft(X, hop_length=HOP, length=length)


def oracle_masks(mix, refs, kind="irm"):
    """Return dict name->estimate using ideal ratio (irm) or binary (ibm) masks."""
    Xmix = stft(mix)
    mags = {k: np.abs(stft(v)) for k, v in refs.items()}
    # align time frames
    T = min(Xmix.shape[1], min(m.shape[1] for m in mags.values()))
    Xmix = Xmix[:, :T]
    mags = {k: m[:, :T] for k, m in mags.items()}
    denom = np.sum([m for m in mags.values()], axis=0) + 1e-9
    ests = {}
    if kind == "ibm":
        stacknames = list(mags.keys())
        stack = np.stack([mags[k] for k in stacknames], axis=0)
        argmax = np.argmax(stack, axis=0)
        for i, k in enumerate(stacknames):
            mask = (argmax == i).astype(float)
            ests[k] = istft(mask * Xmix, length=len(mix))
    else:  # irm (soft)
        for k, m in mags.items():
            mask = m / denom
            ests[k] = istft(mask * Xmix, length=len(mix))
    return ests


def hpss_baseline(mix, refs):
    """librosa HPSS: percussive -> timpani; harmonic -> everything else (grouped)."""
    h, p = librosa.effects.hpss(mix)
    ests = {}
    if "timpani" in refs:
        ests["timpani"] = p
    # harmonic component approximates the pitched ensemble as a whole
    return ests, h, p


def spectrogram_similarity(refs):
    """Pairwise cosine similarity of magnitude spectrograms (TF overlap diagnostic)."""
    mags = {}
    for k, v in refs.items():
        m = np.abs(stft(v))
        mags[k] = (m / (np.linalg.norm(m) + 1e-9)).ravel()
    names = list(mags.keys())
    sim = {}
    for a in names:
        for b in names:
            if a < b:
                sim[f"{a}|{b}"] = round(float(np.dot(mags[a], mags[b])), 3)
    return sim


def main(datadir="data/synth_orchestra", outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    est_dir = os.path.join(outdir, "estimates")
    os.makedirs(est_dir, exist_ok=True)
    mix, stems, families, sr = load_stems(datadir)

    report = {"sr": sr, "n_instruments": len(stems)}

    # ---- Oracle IRM (per instrument) ----
    irm = oracle_masks(mix, stems, kind="irm")
    report["oracle_irm_per_instrument"] = sdr_report(stems, irm, mix)
    # save a few representative estimates for listening
    for k in ["violin1", "violin2", "flute", "oboe", "timpani", "trumpet"]:
        if k in irm:
            sf.write(os.path.join(est_dir, f"oracle_irm_{k}.wav"),
                     irm[k].astype(np.float32), sr)

    # ---- Oracle IBM (per instrument) ----
    ibm = oracle_masks(mix, stems, kind="ibm")
    report["oracle_ibm_per_instrument"] = sdr_report(stems, ibm, mix)

    # ---- Oracle IRM at FAMILY level ----
    irm_fam = oracle_masks(mix, families, kind="irm")
    report["oracle_irm_per_family"] = sdr_report(families, irm_fam, mix)
    for k, v in irm_fam.items():
        sf.write(os.path.join(est_dir, f"oracle_irm_family_{k}.wav"),
                 v.astype(np.float32), sr)

    # ---- HPSS blind baseline (real, weight-free) ----
    hpss_est, hcomp, pcomp = hpss_baseline(mix, stems)
    report["hpss_baseline"] = sdr_report({k: stems[k] for k in hpss_est}, hpss_est, mix)
    sf.write(os.path.join(est_dir, "hpss_percussive.wav"), pcomp.astype(np.float32), sr)
    sf.write(os.path.join(est_dir, "hpss_harmonic.wav"), hcomp.astype(np.float32), sr)

    # ---- TF-overlap diagnostic (why same-instrument is hard) ----
    report["spectrogram_cosine_similarity"] = spectrogram_similarity(stems)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- pretty print ----
    def tbl(title, d):
        print(f"\n=== {title} ===")
        print(f"{'source':12s} {'SI-SDR':>8s} {'mix-base':>9s} {'SI-SDRi':>8s}")
        for k, v in sorted(d.items(), key=lambda kv: -kv[1]['si_sdr']):
            print(f"{k:12s} {v['si_sdr']:8.2f} {v['mixture_baseline']:9.2f} {v['si_sdri']:8.2f}")
    tbl("Oracle IRM  (per instrument)  [ceiling for mono mask separation]",
        report["oracle_irm_per_instrument"])
    tbl("Oracle IRM  (per family)", report["oracle_irm_per_family"])
    tbl("HPSS blind baseline", report["hpss_baseline"])

    print("\n=== Same-instrument vs same-pitch diagnostic (cosine sim of spectrograms) ===")
    s = report["spectrogram_cosine_similarity"]
    for key in ["violin1|violin2", "flute|violin1", "oboe|violin1", "timpani|violin1"]:
        # keys are sorted alphabetically; look up robustly
        val = s.get(key) or s.get("|".join(sorted(key.split("|"))))
        print(f"  {key:22s} -> {val}")
    print(f"\nWrote metrics -> {os.path.join(outdir, 'metrics.json')}")
    print(f"Wrote example estimates -> {est_dir}/")
    return report


if __name__ == "__main__":
    main()
