"""
dataset.py — Manifest-driven Dataset/DataLoader connecting Taxonomy + data to the model.

ManifestDataset reads Phase 2 manifests, rolls each ground-truth stem up to the chosen
`target_level` (category/instrument/part) via the taxonomy, and yields fixed-length
random crops as (mixture, targets, presence). Targets are ordered to match the model's
`output_targets`. `region` restricts crops to a time span so train/valid can be split
within the same pieces without leakage.
"""
import json
import random
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from taxonomy import Taxonomy


def resolve_output_targets(manifests, target_level, taxonomy=None):
    """Ordered list of ids present across manifests, rolled up to target_level."""
    tx = taxonomy or Taxonomy()
    present = set()
    for m in manifests:
        man = json.load(open(m))
        for s in man["sources"]:
            tid = s["taxonomy_id"]
            lvl = tx.nodes[tid].level
            roll = tid if lvl == target_level else tx.rollup(tid, target_level)
            if roll:
                present.add(roll)
    # order by taxonomy declaration
    order = [n.id for n in tx.nodes.values()]
    return [i for i in order if i in present]


class ManifestDataset(Dataset):
    def __init__(self, manifests, output_targets, target_level, sr=22050,
                 crop_seconds=1.5, samples_per_epoch=256, region=(0.0, 1.0),
                 taxonomy=None):
        self.tx = taxonomy or Taxonomy()
        self.output_targets = list(output_targets)
        self.level = target_level
        self.sr = sr
        self.crop = int(crop_seconds * sr)
        self.N = samples_per_epoch
        self.region = region
        self.items = [self._load(m) for m in manifests]

    def _load(self, mpath):
        man = json.load(open(mpath))
        mix, _ = librosa.load(man["mixture"], sr=self.sr, mono=True)
        tgt = {t: np.zeros_like(mix) for t in self.output_targets}
        for s in man["sources"]:
            tid = s["taxonomy_id"]
            lvl = self.tx.nodes[tid].level
            roll = tid if lvl == self.level else self.tx.rollup(tid, self.level)
            if roll in tgt:
                y, _ = librosa.load(s["stem"], sr=self.sr, mono=True)
                n = min(len(y), len(tgt[roll]))
                tgt[roll][:n] += y[:n]
        present = [t for t in self.output_targets if np.sum(tgt[t] ** 2) > 1e-8]
        return dict(mix=mix, tgt=tgt, present=present)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        item = random.choice(self.items)
        T = len(item["mix"])
        c = self.crop
        lo = int(self.region[0] * T)
        hi = max(int(self.region[1] * T) - c, lo + 1)
        st = random.randint(lo, max(lo, hi - 1))
        def crop(a):
            seg = a[st:st + c]
            return np.pad(seg, (0, c - len(seg))) if len(seg) < c else seg
        mix = crop(item["mix"])
        tgts = np.stack([crop(item["tgt"][t]) for t in self.output_targets])   # [S,T]
        return dict(
            mixture=torch.tensor(mix, dtype=torch.float32)[None],              # [1,T]
            targets=torch.tensor(tgts, dtype=torch.float32)[:, None, :],       # [S,1,T]
            presence=torch.tensor([1.0 if t in item["present"] else 0.0
                                   for t in self.output_targets]),             # [S]
        )


def build_loaders(manifests, output_targets, target_level, sr=22050,
                  crop_seconds=1.5, batch_size=4, samples_per_epoch=256):
    """Single-corpus mode: split crops by TIME region within the same pieces."""
    common = dict(output_targets=output_targets, target_level=target_level, sr=sr,
                  crop_seconds=crop_seconds)
    train = ManifestDataset(manifests, samples_per_epoch=samples_per_epoch,
                            region=(0.0, 0.7), **common)
    valid = ManifestDataset(manifests, samples_per_epoch=max(samples_per_epoch // 4, 8),
                            region=(0.7, 1.0), **common)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(valid, batch_size=batch_size, shuffle=False))


def build_split_loaders(train_manifests, valid_manifests, output_targets, target_level,
                        sr=22050, crop_seconds=1.5, batch_size=4, samples_per_epoch=256):
    """Corpus mode: DISTINCT pieces for train vs valid (song-level split, full region)."""
    common = dict(output_targets=output_targets, target_level=target_level, sr=sr,
                  crop_seconds=crop_seconds, region=(0.0, 1.0))
    train = ManifestDataset(train_manifests, samples_per_epoch=samples_per_epoch, **common)
    valid = ManifestDataset(valid_manifests,
                            samples_per_epoch=max(samples_per_epoch // 4, 8), **common)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(valid, batch_size=batch_size, shuffle=False))
