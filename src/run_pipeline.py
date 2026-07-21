"""
run_pipeline.py — One-command automated training pipeline.

    data check -> train (validation + checkpoint + resume) -> inference -> evaluation
    -> report, with per-run experiment tracking and automatic GPU config selection.

Examples:
    python src/run_pipeline.py                       # autoselect config from GPU/VRAM
    python src/run_pipeline.py --config configs/prod/roformer_24gb.yaml --name musdb24
    python src/run_pipeline.py --resume checkpoints/prod_small/final.pt
    torchrun --nproc_per_node=4 src/run_pipeline.py --config configs/prod/roformer_48gb.yaml

The moment a GPU is available, `python src/run_pipeline.py` picks the right tier config
and runs the whole thing end-to-end. No GPU -> it selects the small/CPU config.
"""
import os
import json
import argparse
import glob
import yaml

import mss.backbones  # noqa: F401
from mss.registry import get_backbone
from mss.dataset import build_split_loaders, build_loaders, resolve_output_targets
from mss.gpu import detect, autoselect_config, print_report
from mss.experiment import Experiment
from mss.pipeline import run_song


# ---------------------------------------------------------------- data check
def check_data(cfg):
    d = cfg["data"]
    if "splits" in d:
        if not os.path.exists(d["splits"]):
            raise FileNotFoundError(f"splits file not found: {d['splits']} "
                                    "(run src/make_corpus.py or src/prepare_datasets.py)")
        sp = json.load(open(d["splits"]))
        train_m, valid_m, test_m = sp.get("train", []), sp.get("valid", []), sp.get("test", [])
    else:
        train_m = valid_m = d.get("manifests", [])
        test_m = d.get("manifests", [])
    missing = [m for m in (train_m + valid_m + test_m) if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest(s) missing, e.g. {missing[:2]}")
    n_src = 0
    for m in (train_m[:1] or valid_m[:1]):
        man = json.load(open(m))
        n_src = len(man.get("sources", []))
        if not os.path.exists(man["mixture"]):   # audio referenced by the manifest
            raise FileNotFoundError(
                f"manifest audio missing: {man['mixture']}\n"
                "  -> regenerate the synthetic corpus:  python src/make_corpus.py\n"
                "  -> or prepare a real dataset:         python src/prepare_datasets.py ...")
    print(f"[data] train={len(train_m)} valid={len(valid_m)} test={len(test_m)} "
          f"pieces | ~{n_src} sources/piece | splits={d.get('splits','(manifests)')}")
    return train_m, valid_m, test_m


# ---------------------------------------------------------------- DDP setup
def maybe_init_ddp():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 0, False
    import torch.distributed as dist
    dist.init_process_group(backend="nccl" if os.environ.get("LOCAL_RANK") else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, True


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="config path; autoselected if omitted")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args()

    info = print_report()
    config_path = a.config or autoselect_config(info)
    print(f"[config] using {config_path}")
    cfg = yaml.safe_load(open(config_path))

    # device reconciliation (fall back to CPU if CUDA requested but absent)
    want = cfg.get("device", "cpu")
    device = "cuda" if (want == "cuda" and info["cuda"]) else "cpu"
    if want == "cuda" and not info["cuda"]:
        print("[warn] config requests CUDA but none available -> falling back to CPU")
    cfg["device"] = device

    rank, local_rank, is_ddp = maybe_init_ddp()
    name = a.name or os.path.splitext(os.path.basename(config_path))[0]

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    target_level = d["target_level"]
    train_m, valid_m, test_m = check_data(cfg)
    output_targets = resolve_output_targets(train_m + valid_m + test_m, target_level)
    print(f"[targets] level={target_level} -> {output_targets}")

    exp = Experiment(name, cfg, info) if rank == 0 else None

    model_cfg = {**m, "output_targets": output_targets, "sr": d["sr"], "mono": cfg.get("mono", True),
                 "device": device, "grad_checkpointing": t.get("grad_checkpointing", False)}
    backbone = get_backbone(m["name"]).build(model_cfg)
    if rank == 0:
        print(f"[model] {m['name']} params={backbone.num_params()/1e6:.3f}M device={device}")
    if is_ddp:
        backbone.wrap_ddp(local_rank)

    # -------- train --------
    if not (a.skip_train or a.test_only):
        kw = dict(sr=d["sr"], crop_seconds=d.get("crop_seconds", 2.0),
                  batch_size=d.get("batch_size", 2),
                  samples_per_epoch=d.get("samples_per_epoch", 200))
        if "splits" in d or train_m is not valid_m:
            tl, vl = build_split_loaders(train_m, valid_m, output_targets, target_level, **kw)
        else:
            tl, vl = build_loaders(train_m, output_targets, target_level, **kw)
        steps = a.steps if a.steps is not None else t["steps"]
        hist = backbone.train(
            tl, vl, steps=steps, lr=t.get("lr", 3e-4), ckpt_dir=t["ckpt_dir"],
            ckpt_every=t.get("ckpt_every", 100), valid_every=t.get("valid_every", 100),
            log_every=t.get("log_every", 20), resume=a.resume,
            amp=t.get("amp", False), amp_dtype=t.get("amp_dtype", "auto"),
            grad_accum=t.get("grad_accum", 1), grad_clip=t.get("grad_clip", 0.0),
            optimizer=t.get("optimizer", "adam"), weight_decay=t.get("weight_decay", 0.0),
            scheduler=t.get("scheduler"), warmup=t.get("warmup", 0), rank=rank,
            on_valid=(lambda s, v: exp.update(last_valid={"step": s, "si_sdr": v})) if exp else None,
            on_checkpoint=(lambda p, s: exp.add_checkpoint(p)) if exp else None)
        if exp:
            exp.update(train_history=hist["history"], final_checkpoint=hist["final_checkpoint"])
        ckpt = hist["final_checkpoint"]
    else:
        ckpt = a.resume or os.path.join(t["ckpt_dir"], "final.pt")
        if os.path.exists(ckpt):
            backbone.load_checkpoint(ckpt)

    # -------- inference + evaluation on the test split (rank 0) --------
    if rank == 0 and test_m:
        infer_dir = os.path.join(exp.dir if exp else "results", "infer")
        per_song, cat_acc = {}, {}
        for mf in test_m:
            song = json.load(open(mf))["song_id"]
            res = run_song(backbone, mf, os.path.join(infer_dir, song),
                           target_level=target_level, presence_mode="oracle", evaluate=True)
            rep = res["report"]
            per_song[song] = rep["per_category"]
            for c, v in rep["per_category"].items():
                cat_acc.setdefault(c, []).append(v["mean_si_sdr"])
        agg = {c: round(sum(vs) / len(vs), 2) for c, vs in cat_acc.items()}
        metrics = dict(test_songs=list(per_song), per_category_mean_si_sdr=agg,
                       per_song=per_song)
        print(f"[eval] test per-category mean SI-SDR: {agg}")
        if exp:
            exp.set_metrics(metrics)

    if exp:
        exp.finish("completed")
        print(f"[experiment] recorded -> {exp.dir}/run.json + report.md")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
