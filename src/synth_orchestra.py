"""
synth_orchestra.py — Controlled synthetic orchestra test-bench with ground-truth stems.

Purpose (Phase 1 PoC):
  Generate a short multi-instrument "orchestral-like" piece where we know the
  exact per-instrument audio (ground truth), so that any separation method can be
  evaluated objectively (SI-SDR etc.).

Design choices that make this a *diagnostic* bench, not just a demo:
  - Violin I and Violin II share an IDENTICAL timbre model but play DIFFERENT lines
    -> probes the "same-instrument section" problem (Phase 0's hardest case).
  - Flute plays in UNISON with Violin I (same pitch) but with a DIFFERENT timbre
    -> probes timbre-based separation when pitch overlaps (the "easy-ish" case).
  - Instruments play a shared chord progression -> realistic harmonic overlap and
    inter-source correlation, like a real orchestral texture.

This is synthetic additive synthesis (not a real orchestra). It is fully
reproducible with no downloads, and its purpose is to expose *fundamental limits*
of separation, which are model-independent. Real-recording evaluation (URMP /
Aalto / Cadenza) is described in the Phase 1 report and the reproducible recipe.
"""
import os
import json
import numpy as np
import soundfile as sf

SR = 22050
BPM = 100
BEATS_PER_CHORD = 2          # each chord lasts 2 beats
N_CHORDS = 8
SEED = 20260720

rng = np.random.default_rng(SEED)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def adsr(n, sr, a=0.02, d=0.08, s=0.7, r=0.15):
    """Simple ADSR amplitude envelope of length n samples."""
    env = np.zeros(n)
    A = int(a * sr); D = int(d * sr); R = int(r * sr)
    A = max(A, 1); D = max(D, 1); R = max(R, 1)
    sus = max(n - A - D - R, 0)
    idx = 0
    env[idx:idx + A] = np.linspace(0, 1, A, endpoint=False); idx += A
    if idx < n:
        dd = min(D, n - idx)
        env[idx:idx + dd] = np.linspace(1, s, dd, endpoint=False); idx += dd
    if idx < n and sus > 0:
        ss = min(sus, n - idx)
        env[idx:idx + ss] = s; idx += ss
    if idx < n:
        rr = n - idx
        env[idx:idx + rr] = np.linspace(env[idx - 1] if idx > 0 else s, 0, rr)
    return env


def harmonic_tone(f0, n, sr, harmonic_weights, vibrato_hz=0.0, vibrato_depth=0.0,
                  inharmonicity=0.0):
    """Additive-synthesis tone. harmonic_weights: list of amplitudes for k=1..K."""
    t = np.arange(n) / sr
    # vibrato -> instantaneous frequency
    if vibrato_hz > 0 and vibrato_depth > 0:
        vib = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    else:
        vib = 1.0
    sig = np.zeros(n)
    for k, w in enumerate(harmonic_weights, start=1):
        if w == 0:
            continue
        # slight inharmonic stretch (brass/strings) and per-partial phase
        fk = f0 * k * (1.0 + inharmonicity * (k - 1))
        phase = rng.uniform(0, 2 * np.pi)
        sig += w * np.sin(2 * np.pi * fk * (t * vib) + phase)
    m = np.max(np.abs(sig)) + 1e-9
    return sig / m


# Timbre models: harmonic amplitude profiles per instrument family.
# (These are stylised, not measured; the point is *distinct* vs *identical* timbres.)
TIMBRE = {
    "violin":   dict(h=[1.0, .6, .5, .35, .3, .22, .18, .12, .09, .06], vib=5.5, vibd=0.006, inh=0.0004),
    "viola":    dict(h=[1.0, .65, .45, .3, .25, .18, .12, .08], vib=5.0, vibd=0.006, inh=0.0004),
    "cello":    dict(h=[1.0, .7, .5, .35, .25, .18, .12, .08, .05], vib=4.5, vibd=0.005, inh=0.0005),
    "bass":     dict(h=[1.0, .8, .55, .35, .22, .14, .08], vib=3.5, vibd=0.004, inh=0.0006),
    "flute":    dict(h=[1.0, .28, .12, .06, .03], vib=5.0, vibd=0.010, inh=0.0),      # nearly pure -> very different from violin
    "oboe":     dict(h=[.5, 1.0, .8, .55, .4, .3, .2, .12], vib=5.0, vibd=0.008, inh=0.0),  # strong 2nd/3rd -> reedy
    "clarinet": dict(h=[1.0, .05, .7, .05, .45, .04, .25, .03, .12], vib=4.0, vibd=0.004, inh=0.0),  # odd harmonics
    "bassoon":  dict(h=[.4, 1.0, .7, .5, .35, .22, .14], vib=4.0, vibd=0.005, inh=0.0),
    "horn":     dict(h=[1.0, .7, .55, .4, .3, .22, .15, .1], vib=0.0, vibd=0.0, inh=0.0003),
    "trumpet":  dict(h=[.7, 1.0, .9, .7, .55, .42, .3, .2, .12], vib=0.0, vibd=0.0, inh=0.0004),  # bright
    "trombone": dict(h=[1.0, .8, .65, .5, .38, .28, .18, .1], vib=0.0, vibd=0.0, inh=0.0004),
}

# Instrument score: list of MIDI pitches per chord (0/None = rest). 8 chords.
# Chord progression C - G - Am - F - C - G - F - C.
SCORE = {
    # STRINGS
    "violin1": dict(timbre="violin", section=3, pan=-0.4,
                    notes=[76, 74, 72, 77, 79, 71, 69, 67]),         # melody (high)
    "violin2": dict(timbre="violin", section=3, pan=-0.2,             # SAME timbre as violin1, DIFFERENT line
                    notes=[72, 71, 69, 72, 76, 67, 65, 64]),
    "viola":   dict(timbre="viola", section=2, pan=0.0,
                    notes=[64, 62, 60, 65, 64, 59, 57, 60]),
    "cello":   dict(timbre="cello", section=2, pan=0.2,
                    notes=[48, 43, 45, 41, 48, 43, 41, 48]),
    "bass":    dict(timbre="bass", section=1, pan=0.35,
                    notes=[36, 31, 33, 29, 36, 31, 29, 36]),
    # WOODWINDS
    "flute":   dict(timbre="flute", section=1, pan=-0.15,            # UNISON with violin1 (same pitch, diff timbre)
                    notes=[76, 74, 72, 77, 79, 71, 69, 67]),
    "oboe":    dict(timbre="oboe", section=1, pan=-0.05,
                    notes=[72, 74, 76, 72, 71, 74, 72, 71]),
    "clarinet":dict(timbre="clarinet", section=1, pan=0.05,
                    notes=[60, 62, 64, 60, 59, 62, 60, 59]),
    "bassoon": dict(timbre="bassoon", section=1, pan=0.15,
                    notes=[48, 50, 52, 47, 48, 43, 41, 48]),
    # BRASS
    "horn":    dict(timbre="horn", section=2, pan=-0.25,
                    notes=[55, 55, 57, 53, 55, 55, 53, 55]),
    "trumpet": dict(timbre="trumpet", section=1, pan=0.1,            # sparse (fanfare)
                    notes=[67, 0, 0, 0, 79, 0, 0, 72]),
    "trombone":dict(timbre="trombone", section=1, pan=0.25,
                    notes=[43, 43, 45, 41, 43, 43, 41, 43]),
}

PERC = {
    "timpani": dict(pan=0.0, hits=[36, 31, 0, 0, 36, 31, 0, 36]),   # percussive root hits
}

FAMILIES = {
    "strings":   ["violin1", "violin2", "viola", "cello", "bass"],
    "woodwinds": ["flute", "oboe", "clarinet", "bassoon"],
    "brass":     ["horn", "trumpet", "trombone"],
    "percussion":["timpani"],
}


def render_instrument(spec, total_n, chord_n):
    """Render a sustained-tone instrument (string/wind/brass) to a mono signal."""
    tb = TIMBRE[spec["timbre"]]
    out = np.zeros(total_n)
    for ci, midi in enumerate(spec["notes"]):
        if not midi:
            continue
        start = ci * chord_n
        f0 = float(midi_to_hz(midi))
        # section = number of players -> detune/chorus for realism (esp. strings)
        players = spec.get("section", 1)
        tone = np.zeros(chord_n)
        for _ in range(players):
            detune = 1.0 + rng.normal(0, 0.0015)          # cents-level detune per player
            delay = int(abs(rng.normal(0, 0.004)) * SR)   # small timing spread
            seg = harmonic_tone(f0 * detune, chord_n, SR, tb["h"],
                                vibrato_hz=tb["vib"], vibrato_depth=tb["vibd"],
                                inharmonicity=tb["inh"])
            if delay > 0:
                seg = np.concatenate([np.zeros(delay), seg])[:chord_n]
            tone += seg
        tone /= players
        env = adsr(chord_n, SR, a=0.03, d=0.06, s=0.75, r=0.12)
        out[start:start + chord_n] += tone * env
    return out


def render_timpani(spec, total_n, chord_n):
    out = np.zeros(total_n)
    for ci, midi in enumerate(spec["hits"]):
        if not midi:
            continue
        start = ci * chord_n
        f0 = float(midi_to_hz(midi))
        n = chord_n
        t = np.arange(n) / SR
        # inharmonic membrane partials + noise transient, fast decay
        sig = (np.sin(2 * np.pi * f0 * t) +
               0.6 * np.sin(2 * np.pi * f0 * 1.59 * t) +
               0.4 * np.sin(2 * np.pi * f0 * 2.14 * t))
        noise = rng.normal(0, 1, n) * np.exp(-t * 40)
        decay = np.exp(-t * 6.0)
        seg = (sig * decay + 0.5 * noise)
        seg /= (np.max(np.abs(seg)) + 1e-9)
        out[start:start + n] += seg
    return out


def _transpose_spec(spec, semis):
    s = {k: (v[:] if isinstance(v, list) else v) for k, v in spec.items()}
    if "notes" in s:
        s["notes"] = [(n + semis if n else 0) for n in s["notes"]]
    if "hits" in s:
        s["hits"] = [(n + semis if n else 0) for n in s["hits"]]
    return s


def generate_piece(outdir="data/synth_orchestra", transpose=0, bpm=BPM, seed=SEED,
                   drop=()):
    """Generate one piece. `transpose` (semitones), `bpm`, `seed`, and `drop` (set of
    instrument keys to omit -> variable instrumentation) create a distinct piece so a
    corpus of non-overlapping train/valid/test pieces can be built."""
    global rng
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    beat_sec = 60.0 / bpm
    chord_sec = beat_sec * BEATS_PER_CHORD
    chord_n = int(chord_sec * SR)
    total_n = chord_n * N_CHORDS

    score = {k: _transpose_spec(v, transpose) for k, v in SCORE.items() if k not in drop}
    perc = {k: _transpose_spec(v, transpose) for k, v in PERC.items() if k not in drop}

    stems = {}
    for name, spec in score.items():
        stems[name] = render_instrument(spec, total_n, chord_n)
    for name, spec in perc.items():
        stems[name] = render_timpani(spec, total_n, chord_n)

    # per-instrument loudness balance (roughly orchestral)
    gains = {
        "violin1": 1.0, "violin2": 0.9, "viola": 0.8, "cello": 0.85, "bass": 0.7,
        "flute": 0.75, "oboe": 0.7, "clarinet": 0.65, "bassoon": 0.6,
        "horn": 0.8, "trumpet": 0.9, "trombone": 0.75, "timpani": 0.9,
    }
    for k in stems:
        s = stems[k]
        s = s / (np.max(np.abs(s)) + 1e-9) * gains[k]
        stems[k] = s

    # mono mixture
    mix = np.sum([stems[k] for k in stems], axis=0)
    peak = np.max(np.abs(mix)) + 1e-9
    norm = 0.9 / peak
    for k in stems:
        stems[k] = stems[k] * norm
        sf.write(os.path.join(outdir, f"stem_{k}.wav"), stems[k].astype(np.float32), SR)
    mix = mix * norm
    sf.write(os.path.join(outdir, "mixture.wav"), mix.astype(np.float32), SR)

    # stereo mixture (with panning) -> for future neural models & the spatial-cue point
    all_specs = {**score, **perc}
    L = np.zeros(total_n); R = np.zeros(total_n)
    for k, s in stems.items():
        pan = all_specs[k].get("pan", 0.0)          # -1 left .. +1 right
        gl = np.cos((pan + 1) * np.pi / 4); gr = np.sin((pan + 1) * np.pi / 4)
        L += s * gl; R += s * gr
    st = np.stack([L, R], axis=1)
    st = st / (np.max(np.abs(st)) + 1e-9) * 0.9
    sf.write(os.path.join(outdir, "mixture_stereo.wav"), st.astype(np.float32), SR)

    # family stems (sum of present members)
    present_families = {}
    for fam, members in FAMILIES.items():
        present = [m for m in members if m in stems]
        if not present:
            continue
        present_families[fam] = present
        fs = np.sum([stems[m] for m in present], axis=0)
        sf.write(os.path.join(outdir, f"family_{fam}.wav"), fs.astype(np.float32), SR)

    meta = dict(sr=SR, bpm=bpm, n_chords=N_CHORDS, duration_sec=total_n / SR,
                instruments=list(stems.keys()), families=present_families,
                transpose=transpose, drop=list(drop), seed=seed)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(stems)} stems + mixture(s) to {outdir} "
          f"({total_n/SR:.1f}s @ {SR} Hz, transpose={transpose}, bpm={bpm}, drop={list(drop)})")
    return outdir


def main(outdir="data/synth_orchestra"):
    return generate_piece(outdir)


if __name__ == "__main__":
    main()
