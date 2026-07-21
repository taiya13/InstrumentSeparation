"""
make_corpus.py — Generate a multi-piece synthetic corpus with CLEAN train/valid/test
separation (distinct pieces), and emit a schema-valid manifest per piece + a splits file.

Each piece varies by transposition / tempo / seed / instrumentation (drop), so train,
valid and test contain NON-OVERLAPPING pieces (song-level split, no leakage) — this is
the Phase 2 data policy applied for the Phase 4 baseline.

NOTE: This is a small SYNTHETIC corpus so a real baseline can be trained+evaluated in an
environment without GPUs or access to the (blocked) real datasets. The same manifests +
splits format is what the real-dataset adapters (SynthSOD/URMP) emit on the GPU box.

    python src/make_corpus.py
"""
import os
import json
import glob
import jsonschema
import soundfile as sf

from taxonomy import Taxonomy
from synth_orchestra import generate_piece

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "..", "schemas", "manifest.schema.json")))

# (transpose_semitones, bpm, seed, drop, split) — distinct pieces per split
PIECES = [
    # ---- train (8 distinct pieces) ----
    (0,  100, 101, (),            "train"),
    (2,   96, 102, ("trumpet",),  "train"),
    (-2, 104, 103, (),            "train"),
    (3,   92, 104, ("bassoon",),  "train"),
    (-3, 108, 105, (),            "train"),
    (5,  100, 106, ("clarinet",), "train"),
    (-5,  88, 107, (),            "train"),
    (7,  112, 108, ("trombone",), "train"),
    # ---- valid (2) ----
    (1,  102, 201, (),            "valid"),
    (-1,  98, 202, ("oboe",),     "valid"),
    # ---- test (2) ----
    (4,   95, 301, (),            "test"),
    (-4, 105, 302, (),            "test"),
]


def build_manifest(datadir, song_id, split, tx):
    meta = json.load(open(os.path.join(datadir, "meta.json")))
    sr = meta["sr"]
    sources, present = [], []
    for p in sorted(glob.glob(os.path.join(datadir, "stem_*.wav"))):
        key = os.path.basename(p)[len("stem_"):-len(".wav")]
        tid = tx.map_label("synth_bench", key)
        if tid is None:
            continue
        sources.append(dict(taxonomy_id=tid, dataset_label=key, stem=p,
                            level=tx.nodes[tid].level))
        present.append(tid)
    manifest = dict(
        song_id=song_id, dataset="synth_corpus", split=split, source_type="synthetic",
        sample_rate=sr, duration_sec=round(meta["duration_sec"], 2),
        mixture=os.path.join(datadir, "mixture.wav"),
        mixture_stereo=os.path.join(datadir, "mixture_stereo.wav"),
        present_ids=present, sources=sources,
        provenance=dict(renderer="numpy_additive_synth", midi_source="progression",
                        augmentations=[f"transpose={meta['transpose']}",
                                       f"bpm={meta['bpm']}"]))
    jsonschema.validate(manifest, SCHEMA)
    out = os.path.join(datadir, "manifest.json")
    json.dump(manifest, open(out, "w"), indent=2, ensure_ascii=False)
    return out


def main(root="data/synth_corpus"):
    tx = Taxonomy()
    os.makedirs(root, exist_ok=True)
    splits = {"train": [], "valid": [], "test": []}
    for i, (tr, bpm, seed, drop, split) in enumerate(PIECES):
        d = os.path.join(root, f"piece_{i:02d}")
        generate_piece(d, transpose=tr, bpm=bpm, seed=seed, drop=drop)
        mp = build_manifest(d, f"corpus_{i:02d}", split, tx)
        splits[split].append(mp)
    json.dump(splits, open(os.path.join(root, "splits.json"), "w"), indent=2)
    print(f"\nCorpus: train={len(splits['train'])} valid={len(splits['valid'])} "
          f"test={len(splits['test'])} pieces -> {root}/splits.json")
    return splits


if __name__ == "__main__":
    main()
