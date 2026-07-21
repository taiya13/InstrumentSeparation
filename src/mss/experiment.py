"""
experiment.py — Per-run experiment tracking.

Every training/pipeline run gets a directory under `runs/<timestamp>_<name>/` that records
model, dataset, training conditions, git commit, environment, and evaluation results.
This makes every result reproducible and comparable without an external tracker (W&B can
be added later, but this works offline).
"""
import os
import json
import time
import subprocess


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_commit():
    return _git("rev-parse", "HEAD")


def git_dirty():
    s = _git("status", "--porcelain")
    return None if s is None else bool(s)


class Experiment:
    def __init__(self, name, config, environment, root="runs"):
        self.id = time.strftime("%Y%m%d_%H%M%S") + "_" + name
        self.dir = os.path.join(root, self.id)
        os.makedirs(self.dir, exist_ok=True)
        d = config.get("data", {})
        self.record = dict(
            id=self.id, name=name,
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            git_commit=git_commit(), git_dirty=git_dirty(),
            git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
            model=config.get("model", {}),
            dataset=dict(splits=d.get("splits"), manifests=d.get("manifests"),
                         target_level=d.get("target_level"), sr=d.get("sr")),
            train_conditions=config.get("train", {}),
            environment=environment,
            status="running", metrics=None, checkpoints=[], notes=[])
        self.save()

    @property
    def path(self):
        return self.dir

    def update(self, **kw):
        self.record.update(kw)
        self.save()

    def add_checkpoint(self, path):
        self.record["checkpoints"].append(path)
        self.save()

    def set_metrics(self, metrics):
        self.record["metrics"] = metrics
        self.save()

    def finish(self, status="completed"):
        self.record["status"] = status
        self.record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()
        self.write_report()

    def save(self):
        json.dump(self.record, open(os.path.join(self.dir, "run.json"), "w"),
                  indent=2, ensure_ascii=False, default=str)

    def write_report(self):
        r = self.record
        L = [f"# Experiment: {r['id']}", "",
             f"- status: **{r['status']}**",
             f"- model: `{r['model'].get('name')}`  ({r['model']})",
             f"- dataset: {r['dataset']}",
             f"- git: `{r['git_commit']}` (branch {r['git_branch']}, dirty={r['git_dirty']})",
             f"- environment: {r['environment']}",
             f"- train conditions: {r['train_conditions']}",
             f"- checkpoints: {len(r['checkpoints'])}", ""]
        if r.get("metrics"):
            L += ["## Metrics", "```json", json.dumps(r["metrics"], ensure_ascii=False, indent=2), "```"]
        open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
