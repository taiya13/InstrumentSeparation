# Evaluation — synth_bench_0001 (synth_bench/test)
estimates: `results/infer_tiny_masker` | overall: {'mean_si_sdr': -6.99, 'median_si_sdr': 0.15, 'n_sources': 4}

## Per-instrument (SI-SDR, dB)
| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |
|---|---|---|---|---|
| 木管楽器 (`ww`) | ww | 0.5 | 2.34 | 5.38 |
| 金管楽器 (`br`) | br | 0.26 | 6.08 | 4.75 |
| 弦楽器 (`str`) | str | 0.05 | 2.06 | 4.35 |
| 打楽器 (`perc`) | perc | -28.78 | -8.21 | 0.75 |

## Per-category (mean SI-SDR)
| category | mean | median | n |
|---|---|---|---|
| str | 0.05 | 0.05 | 1 |
| ww | 0.5 | 0.5 | 1 |
| br | 0.26 | 0.26 | 1 |
| perc | -28.78 | -28.78 | 1 |

## SDR / SIR / SAR @ category level
| source | SDR | SIR | SAR |
|---|---|---|---|
| str | 0.52 | 1.66 | 9.14 |
| ww | 0.84 | 2.46 | 7.85 |
| br | 1.39 | 5.06 | 5.0 |
| perc | -22.05 | -15.51 | -5.33 |
