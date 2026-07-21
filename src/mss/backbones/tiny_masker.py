"""
tiny_masker.py — A small STFT-masking model registered on the platform.

Its ONLY purpose is to be a lightweight, CPU-trainable backbone so the whole
train->infer->evaluate loop can be demonstrated end-to-end without a GPU. It is a
plain reference masker (log-mag -> conv stack -> sigmoid masks -> mixture-phase iSTFT),
NOT a novel model. The SOTA backbones (RoFormer/HT-Demucs) plug into the exact same
interface for the GPU box.
"""
import torch
import torch.nn as nn

from mss.interface import SeparationBackbone
from mss.registry import register


class TinyMaskerNet(nn.Module):
    def __init__(self, n_stems, n_fft=1024, hop=256, hidden=32):
        super().__init__()
        self.n_fft, self.hop, self.n_stems = n_fft, hop, n_stems
        self.use_checkpoint = False
        self.register_buffer("window", torch.hann_window(n_fft))
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, n_stems, 3, padding=1),
        )

    def forward(self, mix):                     # [B,1,T] -> [B,S,1,T]
        B, C, T = mix.shape
        x = mix[:, 0, :]
        spec = torch.stft(x, self.n_fft, self.hop, window=self.window,
                          return_complex=True)               # [B,F,Fr]
        mag = spec.abs().unsqueeze(1)                          # [B,1,F,Fr]
        inp = torch.log1p(mag)
        if self.use_checkpoint and inp.requires_grad:
            from torch.utils.checkpoint import checkpoint
            logits = checkpoint(self.net, inp, use_reentrant=False)
        else:
            logits = self.net(inp)
        # softmax over stems -> masks partition the mixture (sum to 1), a ratio-mask
        # inductive bias that guarantees the outputs re-sum to the input.
        masks = torch.softmax(logits, dim=1)   # [B,S,F,Fr]
        ests = []
        for s in range(self.n_stems):
            est_spec = masks[:, s] * spec                     # [B,F,Fr] (real*complex)
            wav = torch.istft(est_spec, self.n_fft, self.hop,
                              window=self.window, length=T)   # [B,T]
            ests.append(wav)
        return torch.stack(ests, dim=1).unsqueeze(2)          # [B,S,1,T]


@register("tiny_masker")
class TinyMasker(SeparationBackbone):
    def _build(self, cfg):
        return TinyMaskerNet(n_stems=len(cfg["output_targets"]),
                             n_fft=cfg.get("n_fft", 1024),
                             hop=cfg.get("hop", 256),
                             hidden=cfg.get("hidden", 32))

    def _forward(self, mix):
        return self.net(mix)

    def enable_grad_checkpointing(self):
        self.net.use_checkpoint = True
        return True
    # default _loss (L1 on waveform) is appropriate for a mixture-phase masker
