"""
run_train.py — Train ANY registered backbone on manifest data (config-driven).

    python src/run_train.py --config configs/train.example.yaml
    python src/run_train.py --config configs/train.example.yaml --resume checkpoints/.../final.pt

Swapping models is a one-line change: `model.name: mel_band_roformer` (or htdemucs, ...).
"""
import argparse
import yaml

import json
import mss.backbones  # noqa: F401  (populates the registry)
from mss.registry import get_backbone, available
from mss.dataset import build_loaders, build_split_loaders, resolve_output_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    mono = cfg.get("mono", True)
    target_level = d["target_level"]

    # Corpus mode (distinct pieces) if a splits file / explicit lists are given;
    # otherwise single-corpus time-region split.
    if "splits" in d:
        sp = json.load(open(d["splits"]))
        train_manifests, valid_manifests = sp["train"], sp["valid"]
    elif "train_manifests" in d:
        train_manifests, valid_manifests = d["train_manifests"], d["valid_manifests"]
    else:
        train_manifests = valid_manifests = None

    all_manifests = (train_manifests + valid_manifests) if train_manifests \
        else d["manifests"]
    output_targets = resolve_output_targets(all_manifests, target_level)
    print(f"backbones available: {available()}")
    print(f"target_level={target_level}  output_targets={output_targets}")

    model_cfg = {**m, "output_targets": output_targets, "sr": d["sr"],
                 "mono": mono, "device": cfg.get("device", "cpu")}
    backbone = get_backbone(m["name"]).build(model_cfg)
    print(f"model '{m['name']}'  params={backbone.num_params()/1e6:.3f}M")

    kw = dict(sr=d["sr"], crop_seconds=d.get("crop_seconds", 1.5),
              batch_size=d.get("batch_size", 4),
              samples_per_epoch=d.get("samples_per_epoch", 256))
    if train_manifests:
        train_loader, valid_loader = build_split_loaders(
            train_manifests, valid_manifests, output_targets, target_level, **kw)
    else:
        train_loader, valid_loader = build_loaders(
            d["manifests"], output_targets, target_level, **kw)

    steps = a.steps if a.steps is not None else t["steps"]
    backbone.train(train_loader, valid_loader, steps=steps, lr=t.get("lr", 1e-3),
                   ckpt_dir=t["ckpt_dir"], ckpt_every=t.get("ckpt_every", 100),
                   valid_every=t.get("valid_every", 100),
                   log_every=t.get("log_every", 20), resume=a.resume)


if __name__ == "__main__":
    main()
