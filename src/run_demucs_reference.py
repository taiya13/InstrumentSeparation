"""
run_demucs_reference.py — Reproducible REAL HT-Demucs inference + objective evaluation.

Run this on a machine with unrestricted network (weights auto-download from
dl.fbaipublicfiles.com on first use). It:
  1) runs HT-Demucs (Demucs v4) on an orchestral mixture,
  2) if ground-truth stems are provided, maps Demucs' pop stems
     (drums/bass/other/vocals[/guitar/piano]) onto our instrument families and
     reports SI-SDR — quantifying how a pop-trained SOTA model behaves on orchestra.

This is intentionally NOT run inside the project sandbox (weights are blocked there);
it is the recipe the team executes to reproduce the world-class baseline. See
environment/README.md.

Usage:
  pip install -r environment/requirements-sota.txt
  python src/run_demucs_reference.py --input data/synth_orchestra/mixture_stereo.wav \
         --refs data/synth_orchestra --model htdemucs_6s --out results/demucs
"""
import argparse
import os
import sys
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to mixture wav (stereo recommended)")
    ap.add_argument("--refs", default=None, help="dir with ground-truth stem_*.wav / family_*.wav")
    ap.add_argument("--model", default="htdemucs_6s",
                    choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"])
    ap.add_argument("--out", default="results/demucs")
    args = ap.parse_args()

    try:
        import torch
        import numpy as np
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, convert_audio
    except Exception as e:
        print("ERROR importing demucs stack. Install: pip install -r environment/requirements-sota.txt")
        print(repr(e)); sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{args.model}' on {device} (weights auto-download on first run)...")
    model = get_model(args.model)          # downloads from dl.fbaipublicfiles.com
    model.to(device).eval()
    sr = model.samplerate
    print(f"sources = {model.sources}  samplerate = {sr}")

    wav = AudioFile(args.input).read(streams=0, samplerate=sr, channels=model.audio_channels)
    ref_mix = wav.mean(0)
    wav = (wav - wav.mean()) / (wav.std() + 1e-8)
    with torch.no_grad():
        est = apply_model(model, wav[None].to(device), split=True, overlap=0.25)[0]
    est = est.cpu().numpy()               # [n_sources, channels, time]

    stems = {}
    for i, name in enumerate(model.sources):
        mono = est[i].mean(0)
        stems[name] = mono
        sf.write(os.path.join(args.out, f"{name}.wav"), est[i].T.astype(np.float32), sr)
    print(f"Wrote {len(stems)} separated stems -> {args.out}/")

    # ---- optional objective evaluation vs ground truth ----
    if args.refs:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from metrics import si_sdr
        import librosa

        def load(p):
            y, _ = librosa.load(p, sr=sr, mono=True)
            return y

        # Map Demucs pop-stems -> our families (best-effort, documents the mismatch):
        #   'other'  ~ pitched ensemble (strings+winds+brass)
        #   'drums'  ~ percussion (timpani)
        #   'bass'   ~ low strings/brass
        refs = {}
        for p in glob.glob(os.path.join(args.refs, "family_*.wav")):
            fam = os.path.basename(p)[len("family_"):-len(".wav")]
            refs[fam] = load(p)
        mapping = {"other": "strings", "drums": "percussion", "bass": "strings"}
        rows = {}
        for src, fam in mapping.items():
            if src in stems and fam in refs:
                n = min(len(stems[src]), len(refs[fam]))
                rows[f"{src}->{fam}"] = round(si_sdr(refs[fam][:n], stems[src][:n]), 2)
        print("\nSI-SDR of Demucs pop-stems vs orchestral families (mismatch expected):")
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        with open(os.path.join(args.out, "eval_vs_families.json"), "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print("\nInterpretation: a pop-trained SOTA model has NO orchestral-instrument "
              "outputs; the whole orchestra collapses into 'other'. This is the domain "
              "gap that Phase 2 must close by training on orchestral stems.")


if __name__ == "__main__":
    main()
