"""
gpu.py — GPU/CUDA environment detection and automatic config selection.

At startup we inspect GPU / VRAM / CUDA / driver and pick the matching production
config so the user does not have to. Works with no GPU (selects the small/CPU config).
"""
import os
import subprocess
import torch


def detect():
    cuda = torch.cuda.is_available()
    info = dict(cuda=cuda, device="cuda" if cuda else "cpu",
                torch=torch.__version__, cuda_version=torch.version.cuda,
                n_gpus=0, gpu_name=None, vram_gb=None, driver=None, bf16=False)
    if cuda:
        info["n_gpus"] = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["vram_gb"] = round(p.total_memory / 1024 ** 3, 1)
        try:
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            info["bf16"] = False
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL).decode().strip()
            info["driver"] = out.splitlines()[0].strip()
        except Exception:
            pass
    return info


def autoselect_config(info=None, base="configs/prod"):
    """Map detected VRAM to the right production config file."""
    info = info or detect()
    if not info["cuda"]:
        return os.path.join(base, "roformer_small.yaml")
    v = info["vram_gb"] or 0
    if v < 16:
        return os.path.join(base, "roformer_15gb.yaml")
    if v < 32:
        return os.path.join(base, "roformer_24gb.yaml")
    return os.path.join(base, "roformer_48gb.yaml")


def print_report(info=None):
    info = info or detect()
    print("=== GPU / environment ===")
    print(f"  CUDA available : {info['cuda']}")
    if info["cuda"]:
        print(f"  GPU            : {info['gpu_name']}  x{info['n_gpus']}")
        print(f"  VRAM           : {info['vram_gb']} GB")
        print(f"  Driver         : {info['driver']}")
        print(f"  bf16 supported : {info['bf16']}")
    print(f"  torch / cuda   : {info['torch']} / {info['cuda_version']}")
    print(f"  -> auto config : {autoselect_config(info)}")
    return info


if __name__ == "__main__":
    print_report()
