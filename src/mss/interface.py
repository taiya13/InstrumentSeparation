"""
interface.py — The common Backbone Interface for the research platform.

SeparationBackbone is a WRAPPER (not an nn.Module) around any underlying model, so
that different existing implementations (Mel-Band RoFormer, HT-Demucs, ...) are all
driven through one API without being modified:

    build()            construct from a config dict
    train()            generic training loop (checkpoint + resume)   [platform code]
    infer()            model-native forward: [B,C,T] -> [B,S,C,T]     [subclass]
    separate()         high-level single-file separation -> {id: wav} [platform code]
    export()           TorchScript/ONNX export
    save_checkpoint()  persist net(+optim+step)
    load_checkpoint()  restore net(+optim+step) -> enables resume

Subclasses implement only the model-specific parts:
    _build(cfg)   -> underlying nn.Module
    _forward(mix) -> [B,S,C,T]     (shape adaptation to the wrapped model)
    _loss(mix, targets) -> scalar  (default = L1 on waveform; override for native losses)
    configure_optimizer(lr)        (optional override)

`output_targets` is the ordered list of taxonomy ids the model emits (its output
vocabulary), so the platform can map stem index -> instrument/category id and connect
to Presence + the Phase 2 evaluation harness.
"""
import os
import time
import json
import numpy as np
import torch
import librosa

from metrics import si_sdr


def _align(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


class SeparationBackbone:
    name = "base"

    def __init__(self, output_targets, sr=44100, mono=True, cfg=None, device="cpu"):
        self.output_targets = list(output_targets)
        self.sr = sr
        self.mono = mono
        self.cfg = cfg or {}
        self.device = device
        self.net = self._build(self.cfg)
        self.net.to(device)
        self._step = 0

    # ---- factory --------------------------------------------------------
    @classmethod
    def build(cls, cfg):
        return cls(output_targets=cfg["output_targets"],
                   sr=cfg.get("sr", 44100), mono=cfg.get("mono", True),
                   cfg=cfg, device=cfg.get("device", "cpu"))

    # ---- subclass hooks -------------------------------------------------
    def _build(self, cfg):
        raise NotImplementedError

    def _forward(self, mix):            # [B,C,T] -> [B,S,C,T]
        raise NotImplementedError

    def _loss(self, mix, targets):      # default: L1 on waveform (mixture-phase models)
        est = self._forward(mix)
        est, targets = _align(est, targets)
        return torch.mean(torch.abs(est - targets))

    def configure_optimizer(self, lr=1e-3):
        return torch.optim.Adam(self.net.parameters(), lr=lr)

    # ---- API: infer -----------------------------------------------------
    def infer(self, mix):               # [B,C,T] -> [B,S,C,T]
        self.net.eval()
        with torch.no_grad():
            return self._forward(mix.to(self.device))

    # ---- API: train (generic loop; checkpoint + resume) -----------------
    def train(self, train_loader, valid_loader=None, *, steps=200, lr=1e-3,
              ckpt_dir="checkpoints", ckpt_every=100, valid_every=100,
              log_every=20, resume=None):
        os.makedirs(ckpt_dir, exist_ok=True)
        opt = self.configure_optimizer(lr)
        start = 0
        if resume and os.path.exists(resume):
            meta = self.load_checkpoint(resume, optimizer=opt)
            start = meta.get("step", 0)
            print(f"[resume] from {resume} at step {start}")
        history = []
        self.net.train()
        it = iter(train_loader)
        step = start
        t0 = time.time()
        while step < steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            mix = batch["mixture"].to(self.device)
            tgt = batch["targets"].to(self.device)
            opt.zero_grad()
            loss = self._loss(mix, tgt)
            loss.backward()
            opt.step()
            step += 1
            self._step = step
            if step % log_every == 0:
                history.append({"step": step, "loss": round(loss.detach().item(), 5)})
                print(f"  step {step:4d}/{steps}  loss={loss.detach().item():.5f}  "
                      f"({(time.time()-t0)/max(step-start,1):.2f}s/step)")
            if valid_loader is not None and step % valid_every == 0:
                v = self.validate(valid_loader)
                print(f"  [valid] step {step}: mean SI-SDR = {v:.2f} dB")
            if step % ckpt_every == 0:
                self.save_checkpoint(os.path.join(ckpt_dir, f"step{step}.pt"), opt, step)
        final = os.path.join(ckpt_dir, "final.pt")
        self.save_checkpoint(final, opt, step)
        print(f"[train] done. final checkpoint -> {final}")
        return {"history": history, "final_checkpoint": final, "steps": step}

    def validate(self, valid_loader, max_batches=8):
        self.net.eval()
        vals = []
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                if i >= max_batches:
                    break
                est = self._forward(batch["mixture"].to(self.device))
                tgt = batch["targets"].to(self.device)
                est, tgt = _align(est, tgt)
                e = est.cpu().numpy(); t = tgt.cpu().numpy()
                for b in range(e.shape[0]):
                    for s in range(e.shape[1]):
                        r = t[b, s].reshape(-1)
                        if np.sum(r ** 2) < 1e-8:
                            continue
                        vals.append(si_sdr(r, e[b, s].reshape(-1)))
        self.net.train()
        return float(np.mean(vals)) if vals else float("nan")

    # ---- API: separate (single file -> {id: wav}) -----------------------
    def separate(self, wav, sr, targets=None, presence=None):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]                       # [C,T]
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        if self.mono and wav.shape[0] > 1:
            wav = wav.mean(0, keepdims=True)
        x = torch.tensor(wav, dtype=torch.float32)[None]      # [1,C,T]
        est = self.infer(x)[0].cpu().numpy()                  # [S,C,T]
        active = set(targets) if targets else set(self.output_targets)
        if presence is not None:
            active &= set(presence)
        out = {}
        for i, tid in enumerate(self.output_targets):
            if tid in active:
                s = est[i]
                out[tid] = s.mean(0) if s.ndim > 1 else s     # mono output
        return out

    # ---- API: checkpointing --------------------------------------------
    def save_checkpoint(self, path, optimizer=None, step=None, extra=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"name": self.name, "output_targets": self.output_targets,
                    "sr": self.sr, "mono": self.mono, "cfg": self.cfg,
                    "step": step if step is not None else self._step,
                    "net": self.net.state_dict(),
                    "opt": optimizer.state_dict() if optimizer is not None else None,
                    "extra": extra}, path)
        return path

    def load_checkpoint(self, path, optimizer=None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        if optimizer is not None and ckpt.get("opt") is not None:
            optimizer.load_state_dict(ckpt["opt"])
        self._step = ckpt.get("step", 0)
        return ckpt

    # ---- API: export ----------------------------------------------------
    def export(self, path, fmt="torchscript", example_seconds=1.0):
        self.net.eval()
        c = 1 if self.mono else 2
        ex = torch.randn(1, c, int(example_seconds * self.sr), device=self.device)
        if fmt == "torchscript":
            try:
                ts = torch.jit.trace(lambda m: self._forward(m), ex, strict=False)
            except Exception:
                # fall back to saving the underlying module + metadata
                torch.save({"net": self.net.state_dict(), "meta": self.cfg}, path)
                return path
            ts.save(path)
        elif fmt == "onnx":
            torch.onnx.export(self.net, ex, path, opset_version=17)
        else:
            raise ValueError(f"unknown export fmt {fmt}")
        return path

    def to(self, device):
        self.device = device
        self.net.to(device)
        return self

    def num_params(self):
        return sum(p.numel() for p in self.net.parameters())
