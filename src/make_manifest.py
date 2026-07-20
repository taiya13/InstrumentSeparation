"""
make_manifest.py — Build a schema-valid manifest from a rendered piece.

Demo: turn the Phase 1 synthetic bench (data/synth_orchestra) into a manifest that
the presence + evaluation harnesses consume. In Phase 3 the same schema is emitted by
the rendering pipeline for every synthesized/real piece.

    python src/make_manifest.py
"""
import os
import json
import glob
import jsonschema
import soundfile as sf

from taxonomy import Taxonomy

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "..", "schemas", "manifest.schema.json")


def build_synth_bench_manifest(datadir="data/synth_orchestra", split="test",
                               out="data/synth_orchestra/manifest.json"):
    tx = Taxonomy()
    meta_path = os.path.join(datadir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    sr = meta.get("sr", 22050)

    sources = []
    present = []
    for p in sorted(glob.glob(os.path.join(datadir, "stem_*.wav"))):
        stem_key = os.path.basename(p)[len("stem_"):-len(".wav")]
        tid = tx.map_label("synth_bench", stem_key)
        if tid is None:
            print(f"  WARN: no taxonomy mapping for '{stem_key}' (skipped)")
            continue
        level = tx.nodes[tid].level
        info = sf.info(p)
        sources.append(dict(taxonomy_id=tid, dataset_label=stem_key, stem=p, level=level))
        present.append(tid)

    manifest = dict(
        song_id="synth_bench_0001",
        dataset="synth_bench",
        split=split,
        source_type="synthetic",
        sample_rate=sr,
        duration_sec=round(meta.get("duration_sec", info.frames / sr), 2),
        mixture=os.path.join(datadir, "mixture.wav"),
        mixture_stereo=os.path.join(datadir, "mixture_stereo.wav"),
        present_ids=present,
        sources=sources,
        provenance=dict(renderer="numpy_additive_synth",
                        midi_source="hardcoded_progression",
                        augmentations=[]),
    )
    schema = json.load(open(SCHEMA))
    jsonschema.validate(manifest, schema)     # raises if invalid
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(manifest, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote schema-valid manifest -> {out}")
    print(f"  {len(sources)} sources, present_ids: {present}")
    return out


if __name__ == "__main__":
    build_synth_bench_manifest()
