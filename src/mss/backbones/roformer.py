"""
roformer.py — Mel-Band RoFormer integration (Phase 0/1 chosen backbone).

Wraps lucidrains' `bs_roformer.MelBandRoformer` (MIT) WITHOUT modifying it. We use the
model's own forward (infer) and its native training loss (passing `target=`), so this
is a faithful integration, not a re-implementation. Custom modifications are deferred
to a later phase per the project plan.

Notes:
  - The real SOTA config is large and GPU-oriented. Hyperparameters are read from cfg so
    a tiny CPU config can be used for platform smoke-tests, and the full config on a GPU.
  - Input: [B,C,T]. Mono uses (B,T); stereo uses (B,2,T). Output: [B,S,C,T].
"""
import torch
from bs_roformer import MelBandRoformer

from mss.interface import SeparationBackbone
from mss.registry import register


@register("mel_band_roformer")
class MelBandRoformerBackbone(SeparationBackbone):
    def _build(self, cfg):
        n = len(cfg["output_targets"])
        return MelBandRoformer(
            dim=cfg.get("dim", 32),
            depth=cfg.get("depth", 1),
            num_stems=n,
            stereo=not cfg.get("mono", True),
            dim_head=cfg.get("dim_head", 16),
            heads=cfg.get("heads", 2),
            num_bands=cfg.get("num_bands", 16),
            stft_n_fft=cfg.get("n_fft", 512),
            stft_win_length=cfg.get("win_length", 512),
            stft_hop_length=cfg.get("hop", 128),
        )

    def _in(self, mix):
        return mix[:, 0, :] if self.mono else mix          # (B,T) or (B,2,T)

    def _forward(self, mix):                                # -> [B,S,C,T]
        return self.net(self._in(mix))

    def _loss(self, mix, targets):
        # RoFormer computes its native (L1 + multi-res STFT) loss when given target.
        # target must be [B, num_stems, channels, T].
        return self.net(self._in(mix), target=targets)
