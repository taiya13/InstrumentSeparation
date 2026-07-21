"""
research_log.py — Persistent research log for every separation run.

Writes one line per run to logs/research_log.jsonl (for easy grep/analysis across runs)
AND a per-run JSON next to the outputs. Records: timestamp, config, model, checkpoint,
device, GPU info, inputs/outputs, instruments, metrics, evaluation results, and errors.
"""
import os
import json
import time


class ResearchLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl = os.path.join(log_dir, "research_log.jsonl")

    def log(self, record: dict, per_run_dir=None):
        record = dict(record)
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if per_run_dir:
            os.makedirs(per_run_dir, exist_ok=True)
            ts = record["timestamp"].replace(":", "").replace("-", "")
            p = os.path.join(per_run_dir, f"run_{ts}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, default=str)
            return p
        return self.jsonl
