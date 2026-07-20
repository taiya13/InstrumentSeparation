"""
run_infer.py — Full inference pipeline for one song: presence -> separation -> eval.

    python src/run_infer.py --checkpoint checkpoints/.../final.pt \
        --manifest data/synth_orchestra/manifest.json --out results/infer_tiny --presence oracle

Rebuilds the correct backbone from the checkpoint metadata, so no separate config is
needed to run inference on a trained model.
"""
import argparse
import torch

import mss.backbones  # noqa: F401
from mss.registry import get_backbone
from mss.pipeline import run_song


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results/infer")
    ap.add_argument("--presence", default="oracle", choices=["oracle", "heuristic"])
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    name = ckpt["name"]
    model_cfg = {**ckpt.get("cfg", {}), "output_targets": ckpt["output_targets"],
                 "sr": ckpt["sr"], "mono": ckpt["mono"], "device": "cpu"}
    backbone = get_backbone(name).build(model_cfg)
    backbone.load_checkpoint(a.checkpoint)
    target_level = model_cfg["cfg"]["data"]["target_level"] if "data" in model_cfg.get("cfg", {}) \
        else _infer_level(ckpt["output_targets"])
    print(f"loaded '{name}' | output_targets={backbone.output_targets} | level={target_level}")

    run_song(backbone, a.manifest, a.out, target_level=target_level,
             presence_mode=a.presence, evaluate=True)


def _infer_level(output_targets):
    from taxonomy import Taxonomy
    tx = Taxonomy()
    levels = {tx.nodes[t].level for t in output_targets if t in tx.nodes}
    # pick the coarsest present level
    for lv in ["category", "family", "instrument", "part"]:
        if lv in levels:
            return lv
    return "category"


if __name__ == "__main__":
    main()
