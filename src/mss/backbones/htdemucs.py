"""
htdemucs.py — HT-Demucs integration via torchaudio's HDemucs (Phase 1 base model).

Wraps torchaudio.models.HDemucs (the Hybrid-Demucs architecture) unmodified. HDemucs'
output stems are fixed at construction (== output_targets order). Best used stereo on a
GPU with proper chunking; here it demonstrates that a second, architecturally very
different model plugs into the identical interface.
"""
import torch
from torchaudio.models import HDemucs

from mss.interface import SeparationBackbone
from mss.registry import register


@register("htdemucs")
class HTDemucsBackbone(SeparationBackbone):
    def _build(self, cfg):
        ch = 1 if cfg.get("mono", True) else 2
        return HDemucs(sources=cfg["output_targets"], audio_channels=ch)

    def _forward(self, mix):                # [B,C,T] -> [B,S,C,T]
        return self.net(mix)
