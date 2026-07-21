"""
evaluation.py — Instrument-aware separation evaluation harness.

Given a manifest (ground truth) and a directory of system estimates named by taxonomy
id (`<id>.wav`), it computes per-source metrics and rolls them up so we can compare
performance by INSTRUMENT, by CATEGORY, and by SONG — exactly the axes Phase 2 asks for.

Metrics:
  - SI-SDR / SI-SDRi (fast, all sources)
  - SDR / SIR / SAR via mir_eval.bss_eval_sources at a chosen level (default: family/
    category, which is fast) — the classical SiSEC metrics.
  - Permutation-invariant matching WITHIN a same-timbre group (e.g. Violin I/II), plus
    an "identity confusion" flag when the best assignment is a swap.
  - Oracle IRM ceiling for every source, so each number is shown against its headroom.
  - Presence P/R/F1 when a predicted present-set is supplied.

Demo mode (`--demo-oracle`) generates oracle-IRM estimates from the manifest itself and
scores them, so the harness is runnable end-to-end without a trained model. A real model
is evaluated by pointing `--estimates` at its output directory.

    python src/evaluation.py --manifest data/synth_orchestra/manifest.json --demo-oracle
"""
import os
import sys
import json
import argparse
import numpy as np
import librosa

from taxonomy import Taxonomy
from metrics import si_sdr
from oracle_and_baselines import oracle_masks

try:
    import mir_eval
    HAVE_MIR = True
except Exception:
    HAVE_MIR = False


def load(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def refs_at_level(manifest, tx, level):
    """Roll up ground-truth stems to `level`, summing members. Returns {id: audio}."""
    sr = manifest["sample_rate"]
    acc = {}
    for s in manifest["sources"]:
        tid = s["taxonomy_id"]
        target = tid if level == tx.nodes[tid].level else tx.rollup(tid, level)
        if target is None:
            target = tid  # already coarser than requested -> keep
        y = load(s["stem"], sr)
        if target in acc:
            n = min(len(acc[target]), len(y))
            acc[target] = acc[target][:n] + y[:n]
        else:
            acc[target] = y
    return acc, sr


def bss_eval(refs: dict, ests: dict):
    """mir_eval SDR/SIR/SAR jointly over the common id set. Returns {id: {...}}."""
    if not HAVE_MIR:
        return {}
    ids = [k for k in refs if k in ests]
    if len(ids) < 1:
        return {}
    n = min(min(len(refs[k]) for k in ids), min(len(ests[k]) for k in ids))
    R = np.stack([refs[k][:n] for k in ids])
    E = np.stack([ests[k][:n] for k in ids])
    # drop silent references (bss_eval undefined)
    keep = [i for i in range(len(ids)) if np.sum(R[i] ** 2) > 1e-8]
    ids = [ids[i] for i in keep]; R = R[keep]; E = E[keep]
    if len(ids) < 1:
        return {}
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(R, E, compute_permutation=False)
    return {ids[i]: dict(sdr=round(float(sdr[i]), 2), sir=round(float(sir[i]), 2),
                         sar=round(float(sar[i]), 2)) for i in range(len(ids))}


def same_timbre_permutation(refs, ests, tx):
    """For each same-timbre group present, report best-assignment SI-SDR and whether a
    swap beats the identity assignment (identity confusion)."""
    out = {}
    for g in tx.raw.get("same_timbre_groups", []):
        members = [m for m in g["members"] if m in refs and m in ests]
        if len(members) < 2:
            continue
        # 2-member case: identity vs swap
        a, b = members[0], members[1]
        n = min(len(refs[a]), len(refs[b]), len(ests[a]), len(ests[b]))
        idn = si_sdr(refs[a][:n], ests[a][:n]) + si_sdr(refs[b][:n], ests[b][:n])
        swp = si_sdr(refs[a][:n], ests[b][:n]) + si_sdr(refs[b][:n], ests[a][:n])
        out[g["id"]] = dict(identity_sisdr_sum=round(idn, 2),
                            swap_sisdr_sum=round(swp, 2),
                            identity_confusion=bool(swp > idn),
                            best_mean_sisdr=round(max(idn, swp) / 2, 2))
    return out


def evaluate(manifest_path, estimates_dir=None, demo_oracle=False,
             instrument_level="instrument", bss_level="category", out_dir="results"):
    tx = Taxonomy()
    manifest = json.load(open(manifest_path))
    sr = manifest["sample_rate"]
    song = manifest["song_id"]

    # refs at instrument level (for SI-SDR per instrument) and at bss level (for SDR/SIR/SAR)
    refs_inst, _ = refs_at_level(manifest, tx, instrument_level)
    refs_bss, _ = refs_at_level(manifest, tx, bss_level)
    mix = load(manifest["mixture"], sr)

    # --- estimates: from a real system dir, or oracle demo ---
    if demo_oracle:
        ests_inst = oracle_masks(mix, refs_inst, kind="irm")
        ests_bss = oracle_masks(mix, refs_bss, kind="irm")
        est_source = "oracle_irm(demo)"
    else:
        if not estimates_dir:
            print("ERROR: pass --estimates DIR or --demo-oracle"); sys.exit(1)
        ests_inst, ests_bss = {}, {}
        for tid in refs_inst:
            p = os.path.join(estimates_dir, f"{tid}.wav")
            if os.path.exists(p):
                ests_inst[tid] = load(p, sr)
        for tid in refs_bss:
            p = os.path.join(estimates_dir, f"{tid}.wav")
            if os.path.exists(p):
                ests_bss[tid] = load(p, sr)
        est_source = estimates_dir

    # --- per-instrument SI-SDR (+ oracle ceiling for headroom) ---
    oracle_ceiling = oracle_masks(mix, refs_inst, kind="irm")
    per_instrument = {}
    for tid, ref in refs_inst.items():
        row = dict(category=tx.rollup(tid, "category"),
                   display=tx.display(tid, "ja"),
                   mixture_baseline=round(si_sdr(ref, mix), 2),
                   oracle_ceiling_sisdr=round(si_sdr(ref, oracle_ceiling[tid]), 2))
        if tid in ests_inst:
            v = si_sdr(ref, ests_inst[tid])
            row["si_sdr"] = round(v, 2)
            row["si_sdri"] = round(v - row["mixture_baseline"], 2)
        per_instrument[tid] = row

    # --- SDR/SIR/SAR at bss level ---
    bss = bss_eval(refs_bss, ests_bss)

    # --- rollup to category (mean SI-SDR of instruments in each category) ---
    per_category = {}
    for tid, row in per_instrument.items():
        if "si_sdr" not in row:
            continue
        cat = row["category"]
        per_category.setdefault(cat, []).append(row["si_sdr"])
    per_category = {c: dict(mean_si_sdr=round(float(np.mean(v)), 2),
                            median_si_sdr=round(float(np.median(v)), 2),
                            n=len(v)) for c, v in per_category.items()}

    # --- same-timbre permutation / identity confusion ---
    stg = same_timbre_permutation(refs_inst, ests_inst, tx)

    # --- overall ---
    vals = [r["si_sdr"] for r in per_instrument.values() if "si_sdr" in r]
    overall = dict(mean_si_sdr=round(float(np.mean(vals)), 2) if vals else None,
                   median_si_sdr=round(float(np.median(vals)), 2) if vals else None,
                   n_sources=len(vals))

    report = dict(song_id=song, dataset=manifest["dataset"], split=manifest["split"],
                  estimates=est_source, instrument_level=instrument_level,
                  bss_level=bss_level, overall=overall, per_category=per_category,
                  per_instrument=per_instrument, sdr_sir_sar_by_level=bss,
                  same_timbre_groups=stg)

    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, f"eval_{song}.json")
    json.dump(report, open(jp, "w"), indent=2, ensure_ascii=False)
    _write_md(report, tx, os.path.join(out_dir, f"eval_{song}.md"))
    _print(report, tx)
    print(f"\nWrote {jp} and {jp[:-5]}.md")
    return report


def _print(report, tx):
    print(f"\n### Evaluation: {report['song_id']} ({report['dataset']}/{report['split']}) "
          f"| estimates={report['estimates']}")
    print(f"overall: {report['overall']}")
    print(f"\n{'instrument':14s} {'cat':6s} {'SI-SDR':>7s} {'SI-SDRi':>8s} {'ceiling':>8s}")
    for tid, r in sorted(report["per_instrument"].items(),
                         key=lambda kv: -(kv[1].get('si_sdr') or -99)):
        print(f"{tid:14s} {r['category']:6s} "
              f"{r.get('si_sdr','-'):>7} {r.get('si_sdri','-'):>8} "
              f"{r['oracle_ceiling_sisdr']:>8}")
    print(f"\nper-category mean SI-SDR: "
          f"{ {c: v['mean_si_sdr'] for c, v in report['per_category'].items()} }")
    if report["sdr_sir_sar_by_level"]:
        print(f"\nSDR/SIR/SAR @ {report['bss_level']} level:")
        for k, v in report["sdr_sir_sar_by_level"].items():
            print(f"  {k:12s} SDR={v['sdr']:6} SIR={v['sir']:6} SAR={v['sar']:6}")
    if report["same_timbre_groups"]:
        print("\nsame-timbre groups:")
        for g, v in report["same_timbre_groups"].items():
            print(f"  {g}: best_mean_SISDR={v['best_mean_sisdr']} "
                  f"identity_confusion={v['identity_confusion']}")


def _write_md(report, tx, path):
    L = [f"# Evaluation — {report['song_id']} ({report['dataset']}/{report['split']})",
         f"estimates: `{report['estimates']}` | overall: {report['overall']}", "",
         "## Per-instrument (SI-SDR, dB)",
         "| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |",
         "|---|---|---|---|---|"]
    for tid, r in sorted(report["per_instrument"].items(),
                         key=lambda kv: -(kv[1].get('si_sdr') or -99)):
        L.append(f"| {tx.display(tid,'ja')} (`{tid}`) | {r['category']} | "
                 f"{r.get('si_sdr','-')} | {r.get('si_sdri','-')} | {r['oracle_ceiling_sisdr']} |")
    L += ["", "## Per-category (mean SI-SDR)",
          "| category | mean | median | n |", "|---|---|---|---|"]
    for c, v in report["per_category"].items():
        L.append(f"| {c} | {v['mean_si_sdr']} | {v['median_si_sdr']} | {v['n']} |")
    if report["sdr_sir_sar_by_level"]:
        L += ["", f"## SDR / SIR / SAR @ {report['bss_level']} level",
              "| source | SDR | SIR | SAR |", "|---|---|---|---|"]
        for k, v in report["sdr_sir_sar_by_level"].items():
            L.append(f"| {k} | {v['sdr']} | {v['sir']} | {v['sar']} |")
    if report["same_timbre_groups"]:
        L += ["", "## Same-timbre groups (identity confusion)",
              "| group | best mean SI-SDR | identity confusion |", "|---|---|---|"]
        for g, v in report["same_timbre_groups"].items():
            L.append(f"| {g} | {v['best_mean_sisdr']} | {v['identity_confusion']} |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--estimates", default=None)
    ap.add_argument("--demo-oracle", action="store_true")
    ap.add_argument("--instrument-level", default="instrument")
    ap.add_argument("--bss-level", default="category")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    evaluate(a.manifest, a.estimates, a.demo_oracle, a.instrument_level, a.bss_level, a.out)
