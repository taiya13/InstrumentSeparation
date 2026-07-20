"""
hdemucs_smoketest.py — Verify the genuine SOTA architecture (torchaudio HDemucs,
the same Hybrid-Demucs family as Demucs v4) builds and runs on this CPU, and measure
inference cost.

IMPORTANT: The TRAINED weights (MUSDB) download from download.pytorch.org /
dl.fbaipublicfiles.com, which this sandbox's egress policy blocks. So here we run the
architecture with RANDOM weights: the OUTPUT IS NOT A MEANINGFUL SEPARATION. The
purpose is only to (1) confirm the model constructs and the forward pass runs on CPU,
(2) report parameter count, and (3) measure CPU latency & memory -> to size the GPU
requirement for real reproduction (done on the team's GPU box; see run_demucs_reference.py).
"""
import time
import resource
import numpy as np
import torch
import torchaudio
from torchaudio.models import HDemucs

SR = 44100


def main():
    torch.manual_seed(0)
    sources = ["drums", "bass", "other", "vocals"]
    model = HDemucs(sources=sources, audio_channels=2)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: torchaudio HDemucs  sources={sources}")
    print(f"Parameters: {n_params/1e6:.1f} M")
    print(f"Torch: {torch.__version__} | threads={torch.get_num_threads()} | CUDA={torch.cuda.is_available()}")

    for dur in (3.0, 6.0):
        n = int(dur * SR)
        x = torch.randn(1, 2, n) * 0.1        # stereo
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        dt = time.time() - t0
        rtf = dt / dur
        print(f"  forward {dur:>4.1f}s stereo @ {SR}Hz -> out {tuple(y.shape)} "
              f"| {dt:6.2f}s wall | RTF={rtf:5.2f}x")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak RSS: {peak_mb:.0f} MB")
    print("\nNOTE: random weights -> output is NOT a real separation. "
          "This only proves the inference pipeline runs on CPU and sizes the cost.")


if __name__ == "__main__":
    main()
