"""
engine.py — Headless separation engine for the research tool.

Model-agnostic by design: a model is just a checkpoint path. The backbone TYPE,
architecture, output targets, sr and channel layout are read FROM the checkpoint
metadata (as run_infer.py does), so adding/swapping a model never requires touching the
engine or the GUI — you only edit configs/app_default.yaml (or drop a new checkpoint).

Reuses the existing platform as-is:
  - Backbone Interface (mss.registry / SeparationBackbone.separate / load_checkpoint)
  - Taxonomy (display names, target-level rollup)
  - Presence (EnergyHeuristicPresence + select_active_set)
  - Evaluation harness (evaluation.evaluate) when a reference manifest is provided
  - GPU detection (mss.gpu) + GPUMonitor + ResearchLogger
"""
import os
import sys
import time
import traceback

# --- make the repo's src/ importable regardless of entry point ---
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import soundfile as sf
import yaml

import mss.backbones  # noqa: F401  (populate registry)
from mss.registry import get_backbone
from mss import gpu as gpu_mod
from taxonomy import Taxonomy
from presence import EnergyHeuristicPresence, select_active_set
import evaluation

from app.gpu_monitor import GPUMonitor
from app.research_log import ResearchLogger

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_audio(path):
    """Return (wav[C,T] float32, sr). soundfile handles WAV/FLAC/MP3; librosa fallback."""
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")   # [T, C]
        return data.T, sr
    except Exception:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = y[None, :]
        return y.astype(np.float32), sr


class SeparationEngine:
    def __init__(self, config):
        self.cfg = load_config(config) if isinstance(config, str) else dict(config)
        self.device = self._resolve_device(self.cfg.get("device", "auto"))
        self.gpu_info = gpu_mod.detect()
        self.tx = Taxonomy()
        self.logger = ResearchLogger(self.cfg.get("logging", {}).get("dir", "logs"))
        self._loaded = {}         # key -> backbone
        self._models = {m["key"]: m for m in self.cfg.get("models", [])}

    # ---- device ----
    @staticmethod
    def _resolve_device(want):
        try:
            import torch
            has = torch.cuda.is_available()
        except Exception:
            has = False
        if want == "auto":
            return "cuda" if has else "cpu"
        if want == "cuda" and not has:
            return "cpu"
        return want

    # ---- model discovery ----
    def list_models(self):
        out = []
        for key, m in self._models.items():
            ckpt = m.get("checkpoint")
            available = (ckpt and os.path.exists(ckpt)) or ("backbone" in m and "output_targets" in m)
            out.append(dict(key=key, display=m.get("display", key),
                            checkpoint=ckpt, available=bool(available),
                            trained=bool(ckpt and os.path.exists(ckpt))))
        return out

    def _infer_level(self, output_targets):
        levels = {self.tx.nodes[t].level for t in output_targets if t in self.tx.nodes}
        for lv in ["category", "family", "instrument", "part"]:
            if lv in levels:
                return lv
        return "category"

    def load_model(self, key):
        if key in self._loaded:
            return self._loaded[key]
        if key not in self._models:
            raise KeyError(f"unknown model '{key}'. available: {list(self._models)}")
        m = self._models[key]
        ckpt = m.get("checkpoint")
        import torch
        if ckpt and os.path.exists(ckpt):
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            model_cfg = {**meta.get("cfg", {}), "output_targets": meta["output_targets"],
                         "sr": meta["sr"], "mono": meta["mono"], "device": self.device}
            backbone = get_backbone(meta["name"]).build(model_cfg)
            backbone.load_checkpoint(ckpt)
            trained = True
        else:
            # untrained build (pipeline validation) — needs backbone + model + output_targets
            model_cfg = {**m.get("model", {}), "output_targets": m["output_targets"],
                         "sr": m.get("sr", 44100), "mono": m.get("mono", True),
                         "device": self.device}
            backbone = get_backbone(m["backbone"]).build(model_cfg)
            trained = False
        info = dict(key=key, display=m.get("display", key), backbone=backbone.name,
                    checkpoint=ckpt, trained=trained, sr=backbone.sr, mono=backbone.mono,
                    output_targets=backbone.output_targets,
                    target_level=self._infer_level(backbone.output_targets),
                    params_m=round(backbone.num_params() / 1e6, 3))
        self._loaded[key] = (backbone, info)
        return self._loaded[key]

    # ---- active target selection (presence) ----
    def _active_targets(self, backbone, info, wav_ct, sr, mode, threshold):
        if mode != "heuristic":
            return list(backbone.output_targets), "all"
        import librosa
        mono = wav_ct.mean(0)
        if sr != backbone.sr:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=backbone.sr)
        scores = EnergyHeuristicPresence(self.tx, sr=backbone.sr).predict(
            mono, candidates=self.tx.leaves("instrument"))
        picked = select_active_set(scores, self.tx, threshold=threshold)
        level = info["target_level"]
        active = []
        for pid in picked:
            lv = self.tx.nodes[pid].level
            roll = pid if lv == level else self.tx.rollup(pid, level)
            if roll in backbone.output_targets and roll not in active:
                active.append(roll)
        if not active:
            return list(backbone.output_targets), "heuristic(fallback=all)"
        return active, "heuristic"

    # ---- main entry ----
    def separate_file(self, input_path, output_dir, model_key, presence_mode="all",
                      threshold=0.15, reference_manifest=None, progress=None):
        def _p(frac, msg):
            if progress:
                progress(frac, msg)

        record = dict(input=input_path, output_dir=output_dir, model_key=model_key,
                      device=self.device, presence_mode=presence_mode,
                      gpu_env=self.gpu_info, config=self.cfg.get("_name", "app_default"))
        try:
            _p(0.02, f"モデル読み込み: {model_key}")
            backbone, info = self.load_model(model_key)
            record["model"] = info
            if not info["trained"]:
                _p(0.03, "警告: 未学習モデル（パイプライン検証用・分離品質は無意味）")

            _p(0.10, f"音源読み込み: {os.path.basename(input_path)}")
            wav_ct, sr = _read_audio(input_path)
            dur = wav_ct.shape[1] / sr
            record["input_info"] = dict(sr=sr, channels=wav_ct.shape[0],
                                        duration_sec=round(dur, 2))

            active, presence_used = self._active_targets(
                backbone, info, wav_ct, sr, presence_mode, threshold)
            record["active_targets"] = active
            record["presence_used"] = presence_used

            base = os.path.splitext(os.path.basename(input_path))[0]
            out_sub = os.path.join(output_dir, base) if self.cfg.get(
                "output", {}).get("subfolder_per_input", True) else output_dir
            os.makedirs(out_sub, exist_ok=True)

            _p(0.30, f"分離実行中 ({len(active)} 楽器, device={self.device}) ...")
            mon = GPUMonitor().start()
            t0 = time.time()
            stems = backbone.separate(wav_ct, sr, targets=active)
            infer_time = time.time() - t0
            gpu_stats = mon.stop()

            _p(0.75, "楽器別に保存中 ...")
            instruments = []
            index = {}
            for i, tid in enumerate([t for t in backbone.output_targets if t in stems]):
                fn = f"{tid}.wav"
                path = os.path.join(out_sub, fn)
                sf.write(path, np.asarray(stems[tid], dtype=np.float32), backbone.sr)
                name_ja = self.tx.display(tid, "ja")
                name_en = self.tx.display(tid, "en")
                instruments.append(dict(id=tid, name_ja=name_ja, name_en=name_en, file=fn))
                index[tid] = dict(name_ja=name_ja, name_en=name_en, file=fn)
            with open(os.path.join(out_sub, "instruments.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(index, f, ensure_ascii=False, indent=2)

            rtf = infer_time / dur if dur > 0 else None
            record.update(
                output_subdir=out_sub, instruments=instruments,
                num_instruments=len(instruments),
                metrics=dict(inference_time_sec=round(infer_time, 3),
                             realtime_factor=round(rtf, 3) if rtf else None,
                             audio_duration_sec=round(dur, 2), gpu=gpu_stats))

            # optional evaluation against a reference manifest
            if reference_manifest and os.path.exists(reference_manifest):
                _p(0.88, "評価中 (SI-SDR/SDR/SIR/SAR) ...")
                try:
                    rep = evaluation.evaluate(
                        reference_manifest, estimates_dir=out_sub, demo_oracle=False,
                        instrument_level=info["target_level"],
                        bss_level=info["target_level"], out_dir=out_sub)
                    record["evaluation"] = dict(
                        overall=rep.get("overall"), per_category=rep.get("per_category"),
                        sdr_sir_sar=rep.get("sdr_sir_sar_by_level"))
                except Exception as e:
                    record["evaluation_error"] = str(e)
            else:
                record["evaluation"] = None

            record["error"] = None
            _p(0.97, "ログ保存中 ...")
            self.logger.log(record, per_run_dir=out_sub)
            _p(1.0, "完了")
            return record

        except Exception as e:
            record["error"] = dict(type=type(e).__name__, message=str(e),
                                   traceback=traceback.format_exc())
            self.logger.log(record, per_run_dir=output_dir)
            _p(1.0, f"エラー: {e}")
            return record
