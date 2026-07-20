"""
presence.py — Per-piece instrument PRESENCE detection spec + reference implementations.

Rationale (Phase 2 requirement #2): pieces use different instruments, so the system
must output ONLY the instruments estimated to be present, not the full training
vocabulary every time. This module defines the contract between a presence stage and
the separator, plus the active-set selection logic (with hierarchy fallback).

Contract:
    PresenceDetector.predict(...) -> {taxonomy_id: probability in [0,1]}
    select_active_set(scores, tx, threshold, ...) -> ordered list of ids to output

Implementations here:
    - OraclePresence:  ground-truth presence from a manifest. Used to (a) supervise the
      presence head during training on synthetic data, (b) provide an upper bound at eval.
    - EnergyHeuristicPresence: a PLACEHOLDER audio-based detector (per-instrument band
      energy vs the taxonomy midi_range). It only demonstrates the interface. Phase 3
      replaces it with a real audio tagger (PANNs / CLAP / a trained instrument-recognition
      head), which is the correct tool for real recordings.
"""
import numpy as np
import librosa

from taxonomy import Taxonomy


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


class PresenceDetector:
    def predict(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class OraclePresence(PresenceDetector):
    """Presence straight from a manifest's ground truth (prob 1.0 for present)."""
    def predict(self, manifest: dict) -> dict:
        return {s["taxonomy_id"]: 1.0 for s in manifest["sources"]}


class EnergyHeuristicPresence(PresenceDetector):
    """PLACEHOLDER: crude band-energy heuristic. NOT for production; interface demo only."""
    def __init__(self, tx: Taxonomy, sr=22050):
        self.tx = tx
        self.sr = sr

    def predict(self, audio: np.ndarray, candidates=None) -> dict:
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        total = S.sum() + 1e-9
        cand = candidates or self.tx.leaves("instrument")
        scores = {}
        for tid in cand:
            node = self.tx.nodes[tid]
            rng = node.attrs.get("midi_range")
            if not rng:
                scores[tid] = 0.0
                continue
            lo, hi = midi_to_hz(rng[0]), midi_to_hz(rng[1]) * 4  # +harmonics headroom
            band = (freqs >= lo) & (freqs <= hi)
            scores[tid] = float(S[band].sum() / total)
        # normalise to [0,1] for a comparable "score" (heuristic only)
        mx = max(scores.values()) + 1e-9
        return {k: v / mx for k, v in scores.items()}


def select_active_set(scores: dict, tx: Taxonomy, threshold=0.5,
                      min_conf_for_instrument=0.5, fallback_level="category"):
    """
    Turn presence scores into the set of ids the separator should output.

    - ids scoring >= threshold are active at their own level.
    - Hierarchy fallback: if an instrument's members belong to a same-timbre group and
      confidence is low/ambiguous, we can back off to the parent (handled by caller/policy).
      Here we implement the simple, robust rule: output instrument-level ids over
      threshold; if NONE clear the bar in a category that clearly has energy, fall back
      to emitting that category id.
    """
    active = [tid for tid, sc in scores.items() if sc >= threshold]
    # category-level fallback for categories with energy but no confident instrument
    cat_energy = {}
    for tid, sc in scores.items():
        cat = tx.rollup(tid, "category")
        cat_energy[cat] = max(cat_energy.get(cat, 0.0), sc)
    active_cats = {tx.rollup(t, "category") for t in active}
    for cat, e in cat_energy.items():
        if e >= threshold and cat not in active_cats and fallback_level == "category":
            active.append(cat)
    # stable order: by score desc
    active = sorted(set(active), key=lambda t: -scores.get(t, 1.0))
    return active


def presence_metrics(pred_ids, true_ids):
    pred, true = set(pred_ids), set(true_ids)
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                tp=tp, fp=fp, fn=fn,
                false_positives=sorted(pred - true), false_negatives=sorted(true - pred))


def _demo():
    import json, os
    tx = Taxonomy()
    mpath = "data/synth_orchestra/manifest.json"
    if not os.path.exists(mpath):
        print("run src/make_manifest.py first"); return
    manifest = json.load(open(mpath))
    true_ids = manifest["present_ids"]

    # Oracle presence -> perfect
    oracle = OraclePresence().predict(manifest)
    print("Oracle presence set size:", len(oracle))

    # Heuristic presence on the mixture (placeholder)
    y, _ = librosa.load(manifest["mixture"], sr=manifest["sample_rate"], mono=True)
    heur = EnergyHeuristicPresence(tx, sr=manifest["sample_rate"]).predict(y)
    active = select_active_set(heur, tx, threshold=0.15)
    # evaluate heuristic vs truth (roll truth parts up to instrument for a fair compare)
    true_instr = sorted({tx.rollup(t, "instrument") or t for t in true_ids})
    active_instr = sorted({tx.rollup(t, "instrument") or t for t in active
                           if tx.nodes[t].level in ("instrument", "part")})
    print("\nHeuristic active (instrument-level):", active_instr)
    print("True (instrument-level):            ", true_instr)
    print("Presence metrics (heuristic, PLACEHOLDER):",
          json.dumps(presence_metrics(active_instr, true_instr), ensure_ascii=False))
    print("\nNOTE: the heuristic is a stub to demonstrate the interface. Phase 3 uses a "
          "trained audio tagger (PANNs/CLAP) for real presence detection.")


if __name__ == "__main__":
    _demo()
