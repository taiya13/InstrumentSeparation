"""
prepare_datasets.py — Turn a downloaded real dataset into our manifest/splits format.

Runs on a machine with the dataset already on disk (the sandbox blocks the CDNs, so the
DOWNLOAD happens elsewhere — see environment/README.md). Given a local dataset root, it
emits one schema-valid manifest per piece + a song-level splits.json, ready for the
platform's `--config ... data.splits: ...`.

Supported layouts:
  musdb18hq: <root>/{train,test}/<song>/{mixture,vocals,drums,bass,other}.wav
  synthsod:  <root>/<piece>/{mixture.wav, <instrument>.wav ...} + a label map

    python src/prepare_datasets.py --dataset musdb18hq --root /data/musdb18hq --out data/musdb18hq
"""
import os
import json
import glob
import argparse
import jsonschema
import soundfile as sf

from taxonomy import Taxonomy

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = json.load(open(os.path.join(HERE, "..", "schemas", "manifest.schema.json")))


def _manifest(song_id, dataset, split, mixture, sources, sr, dur):
    m = dict(song_id=song_id, dataset=dataset, split=split, source_type="real",
             sample_rate=sr, duration_sec=round(dur, 2), mixture=mixture,
             present_ids=[s["taxonomy_id"] for s in sources], sources=sources,
             provenance=dict(renderer="real_recording"))
    jsonschema.validate(m, SCHEMA)
    return m


def prepare_musdb(root, out, valid_frac=0.15):
    """MUSDB18-HQ -> manifests. vocals/drums/bass mapped; 'other' skipped (mixed)."""
    tx = Taxonomy()
    os.makedirs(out, exist_ok=True)
    splits = {"train": [], "valid": [], "test": []}
    train_songs = sorted(glob.glob(os.path.join(root, "train", "*")))
    n_valid = int(len(train_songs) * valid_frac)
    for i, song_dir in enumerate(sorted(glob.glob(os.path.join(root, "train", "*"))) +
                                 sorted(glob.glob(os.path.join(root, "test", "*")))):
        if not os.path.isdir(song_dir):
            continue
        is_test = os.path.basename(os.path.dirname(song_dir)) == "test"
        split = "test" if is_test else ("valid" if i < n_valid else "train")
        name = os.path.basename(song_dir)
        mixture = os.path.join(song_dir, "mixture.wav")
        if not os.path.exists(mixture):
            continue
        info = sf.info(mixture)
        sources = []
        for stem in ["vocals", "drums", "bass"]:      # 'other' is __mixed__ -> skip
            p = os.path.join(song_dir, f"{stem}.wav")
            tid = tx.map_label("musdb", stem)
            if os.path.exists(p) and tid and tid != "__mixed__":
                sources.append(dict(taxonomy_id=tid, dataset_label=stem, stem=p,
                                    level=tx.nodes[tid].level))
        if not sources:
            continue
        m = _manifest(name, "musdb18hq", split, mixture, sources,
                      info.samplerate, info.frames / info.samplerate)
        mp = os.path.join(out, f"{name}.json")
        json.dump(m, open(mp, "w"), ensure_ascii=False)
        splits[split].append(mp)
    json.dump(splits, open(os.path.join(out, "splits.json"), "w"), indent=2)
    print(f"MUSDB18-HQ: train={len(splits['train'])} valid={len(splits['valid'])} "
          f"test={len(splits['test'])} -> {out}/splits.json")
    return splits


def prepare_synthsod(root, out, label_map=None, valid_frac=0.1, test_frac=0.1):
    """SynthSOD-like: <root>/<piece>/{mixture.wav, <label>.wav}. label_map: label->id."""
    tx = Taxonomy()
    os.makedirs(out, exist_ok=True)
    pieces = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    n = len(pieces); n_test = int(n * test_frac); n_valid = int(n * valid_frac)
    splits = {"train": [], "valid": [], "test": []}
    for i, pdir in enumerate(pieces):
        split = "test" if i < n_test else ("valid" if i < n_test + n_valid else "train")
        mixture = os.path.join(pdir, "mixture.wav")
        if not os.path.exists(mixture):
            continue
        info = sf.info(mixture)
        sources = []
        for p in sorted(glob.glob(os.path.join(pdir, "*.wav"))):
            label = os.path.basename(p)[:-4]
            if label == "mixture":
                continue
            tid = (label_map or {}).get(label) or tx.map_label("urmp", label) \
                or tx.map_label("synthsod_family", label)
            if tid and tid != "__mixed__":
                sources.append(dict(taxonomy_id=tid, dataset_label=label, stem=p,
                                    level=tx.nodes[tid].level))
        if not sources:
            continue
        m = _manifest(os.path.basename(pdir), "synthsod", split, mixture, sources,
                      info.samplerate, info.frames / info.samplerate)
        mp = os.path.join(out, f"{os.path.basename(pdir)}.json")
        json.dump(m, open(mp, "w"), ensure_ascii=False)
        splits[split].append(mp)
    json.dump(splits, open(os.path.join(out, "splits.json"), "w"), indent=2)
    print(f"SynthSOD: train={len(splits['train'])} valid={len(splits['valid'])} "
          f"test={len(splits['test'])} -> {out}/splits.json")
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musdb18hq", "synthsod"])
    ap.add_argument("--root", required=True, help="local path to the downloaded dataset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dataset == "musdb18hq":
        prepare_musdb(a.root, a.out)
    else:
        prepare_synthsod(a.root, a.out)
