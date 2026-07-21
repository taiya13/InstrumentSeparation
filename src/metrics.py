"""
metrics.py — Separation quality metrics.

SI-SDR (scale-invariant SDR) is the modern standard and is O(n) (no expensive
bss_eval projections), so it scales to many sources. We also expose SI-SDR
*improvement* over the trivial "mixture-as-estimate" baseline, which is what
actually matters (how much better than doing nothing).
"""
import numpy as np


def si_sdr(reference, estimate, eps=1e-9):
    """Scale-invariant SDR in dB. reference/estimate: 1-D arrays (mono)."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    if np.sum(reference ** 2) < eps:
        return float("nan")
    # zero-mean
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def sdr_report(references: dict, estimates: dict, mixture):
    """Return per-source SI-SDR, mixture-baseline SI-SDR, and improvement (SI-SDRi)."""
    rows = {}
    for name, ref in references.items():
        est = estimates.get(name)
        if est is None:
            continue
        base = si_sdr(ref, mixture)          # feeding the mix as the estimate
        val = si_sdr(ref, est)
        rows[name] = dict(si_sdr=round(val, 2),
                          mixture_baseline=round(base, 2),
                          si_sdri=round(val - base, 2))
    return rows
