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
        if self.cfg.get("grad_checkpointing"):
            ok = self.enable_grad_checkpointing()
            print(f"[grad_checkpointing] {'enabled' if ok else 'not supported for this model'}")

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

    def configure_optimizer(self, lr=1e-3, optimizer="adam", weight_decay=0.0,
                            betas=(0.9, 0.999)):
        params = self.net.parameters()
        if optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        return torch.optim.Adam(params, lr=lr, betas=betas)

    def _make_scheduler(self, opt, name, steps, warmup):
        if not name or name == "none":
            return None
        if name == "cosine":
            import math
            from torch.optim.lr_scheduler import LambdaLR
            def fn(s):
                if warmup and s < warmup:
                    return (s + 1) / max(1, warmup)
                p = (s - warmup) / max(1, steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * min(max(p, 0.0), 1.0)))
            return LambdaLR(opt, fn)
        return None

    def enable_grad_checkpointing(self):
        """Hook: subclasses enable gradient checkpointing to trade compute for VRAM."""
        return False

    def wrap_ddp(self, local_rank):
        """Wrap the underlying net in DistributedDataParallel (call after dist.init)."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        dev = [local_rank] if self.device.startswith("cuda") else None
        self.net = DDP(self.net, device_ids=dev)
        return self

    def _core_net(self):
        net = self.net
        return net.module if hasattr(net, "module") else net

    # ---- API: infer -----------------------------------------------------
    def infer(self, mix):               # [B,C,T] -> [B,S,C,T]
        self.net.eval()
        with torch.no_grad():
            return self._forward(mix.to(self.device))

    # ---- API: train (generic loop; AMP + grad-accum + clip + sched + DDP) ----
    def train(self, train_loader, valid_loader=None, *, steps=200, lr=1e-3,
              ckpt_dir="checkpoints", ckpt_every=100, valid_every=100,
              log_every=20, resume=None, amp=False, amp_dtype="auto",
              grad_accum=1, grad_clip=0.0, optimizer="adam", weight_decay=0.0,
              scheduler=None, warmup=0, rank=0, on_valid=None, on_checkpoint=None):
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        opt = self.configure_optimizer(lr, optimizer=optimizer, weight_decay=weight_decay)
        sched = self._make_scheduler(opt, scheduler, steps, warmup)
        start = 0
        if resume and os.path.exists(resume):
            meta = self.load_checkpoint(resume, optimizer=opt)
            start = meta.get("step", 0)
            print(f"[resume] from {resume} at step {start}")

        # --- device-aware mixed precision ---
        is_cuda = self.device.startswith("cuda")
        amp_device = "cuda" if is_cuda else "cpu"
        if amp_dtype == "auto":
            dtype = torch.bfloat16 if (not is_cuda or torch.cuda.is_bf16_supported()) \
                else torch.float16
        else:
            dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        use_scaler = amp and is_cuda and dtype == torch.float16
        scaler = torch.amp.GradScaler(amp_device, enabled=use_scaler)

        history = []
        self.net.train()
        it = iter(train_loader)

        def next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(train_loader)
                return next(it)

        step = start
        t0 = time.time()
        while step < steps:
            opt.zero_grad(set_to_none=True)
            last = 0.0
            for _ in range(grad_accum):
                batch = next_batch()
                mix = batch["mixture"].to(self.device)
                tgt = batch["targets"].to(self.device)
                with torch.amp.autocast(amp_device, dtype=dtype, enabled=amp):
                    loss = self._loss(mix, tgt) / grad_accum
                scaler.scale(loss).backward()
                last += loss.detach().item()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            step += 1
            self._step = step
            if rank == 0 and step % log_every == 0:
                history.append({"step": step, "loss": round(last, 5),
                                "lr": opt.param_groups[0]["lr"]})
                print(f"  step {step:4d}/{steps}  loss={last:.5f}  "
                      f"lr={opt.param_groups[0]['lr']:.2e}  "
                      f"({(time.time()-t0)/max(step-start,1):.2f}s/step)")
            if valid_loader is not None and rank == 0 and step % valid_every == 0:
                v = self.validate(valid_loader)
                print(f"  [valid] step {step}: mean SI-SDR = {v:.2f} dB")
                if on_valid:
                    on_valid(step, v)
            if rank == 0 and step % ckpt_every == 0:
                p = self.save_checkpoint(os.path.join(ckpt_dir, f"step{step}.pt"), opt, step)
                if on_checkpoint:
                    on_checkpoint(p, step)
        final = os.path.join(ckpt_dir, "final.pt")
        if rank == 0:
            self.save_checkpoint(final, opt, step)
            if on_checkpoint:
                on_checkpoint(final, step)
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
                    "net": self._core_net().state_dict(),   # unwrap DDP if present
                    "opt": optimizer.state_dict() if optimizer is not None else None,
                    "extra": extra}, path)
        return path

    def load_checkpoint(self, path, optimizer=None):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._core_net().load_state_dict(ckpt["net"])
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
