"""
gpu_monitor.py — Sample GPU utilization and VRAM during inference.

Robust fallbacks so it works on the RTX 5060 Ti (pynvml), any NVIDIA box (nvidia-smi),
or CPU-only dev machines (torch / none). Runs a background sampler thread that you
start() before inference and stop() after; stop() returns a summary.
"""
import threading
import time
import subprocess


def _try_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


class GPUMonitor:
    def __init__(self, device_index=0, interval=0.1):
        self.device_index = device_index
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._util = []
        self._mem = []          # MB used
        self.source = "none"
        self.vram_total_mb = None
        self._nvml = None
        self._handle = None
        self._init_backend()

    def _init_backend(self):
        self._nvml = _try_pynvml()
        if self._nvml is not None:
            try:
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self.vram_total_mb = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).total / 1024 ** 2
                self.source = "pynvml"
                return
            except Exception:
                self._nvml = None
        # nvidia-smi fallback: probe once
        if self._nvidia_smi_sample() is not None:
            self.source = "nvidia-smi"
            return
        # torch fallback
        try:
            import torch
            if torch.cuda.is_available():
                self.vram_total_mb = torch.cuda.get_device_properties(
                    self.device_index).total_memory / 1024 ** 2
                self.source = "torch"
        except Exception:
            pass

    def _nvidia_smi_sample(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(self.device_index)],
                stderr=subprocess.DEVNULL, timeout=2).decode().strip().splitlines()[0]
            util, used, total = [float(x) for x in out.split(",")]
            self.vram_total_mb = total
            return util, used
        except Exception:
            return None

    def _sample(self):
        if self.source == "pynvml":
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1024 ** 2
                return u, m
            except Exception:
                return None
        if self.source == "nvidia-smi":
            return self._nvidia_smi_sample()
        if self.source == "torch":
            try:
                import torch
                m = torch.cuda.memory_reserved(self.device_index) / 1024 ** 2
                return None, m
            except Exception:
                return None
        return None

    def _loop(self):
        while not self._stop.is_set():
            s = self._sample()
            if s is not None:
                u, m = s
                if u is not None:
                    self._util.append(u)
                if m is not None:
                    self._mem.append(m)
            self._stop.wait(self.interval)

    def start(self):
        if self.source == "none":
            return self
        try:
            import torch
            if self.source == "torch":
                torch.cuda.reset_peak_memory_stats(self.device_index)
        except Exception:
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        util_avg = round(sum(self._util) / len(self._util), 1) if self._util else None
        util_peak = round(max(self._util), 1) if self._util else None
        vram_peak = round(max(self._mem), 1) if self._mem else None
        # torch peak (more accurate for allocated)
        if self.source == "torch":
            try:
                import torch
                vram_peak = round(torch.cuda.max_memory_reserved(self.device_index) / 1024 ** 2, 1)
            except Exception:
                pass
        return dict(source=self.source,
                    gpu_util_avg_pct=util_avg, gpu_util_peak_pct=util_peak,
                    vram_used_peak_mb=vram_peak, vram_total_mb=round(self.vram_total_mb, 1)
                    if self.vram_total_mb else None)
