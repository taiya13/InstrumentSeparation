# Evaluation — corpus_11 (synth_corpus/test)
estimates: `results/infer_roformer_baseline` | overall: {'mean_si_sdr': -10.81, 'median_si_sdr': -5.02, 'n_sources': 4}

## Per-instrument (SI-SDR, dB)
| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |
|---|---|---|---|---|
| 木管楽器 (`ww`) | ww | -0.85 | 0.25 | 6.24 |
| 弦楽器 (`str`) | str | -1.46 | 0.85 | 4.04 |
| 金管楽器 (`br`) | br | -8.57 | -2.55 | 3.93 |
| 打楽器 (`perc`) | perc | -32.34 | -9.87 | -1.69 |

## Per-category (mean SI-SDR)
| category | mean | median | n |
|---|---|---|---|
| str | -1.46 | -1.46 | 1 |
| ww | -0.85 | -0.85 | 1 |
| br | -8.57 | -8.57 | 1 |
| perc | -32.34 | -32.34 | 1 |

## SDR / SIR / SAR @ category level
| source | SDR | SIR | SAR |
|---|---|---|---|
| str | -0.39 | -0.06 | 14.07 |
| ww | 0.62 | 1.53 | 10.16 |
| br | -5.03 | -3.68 | 5.92 |
| perc | -0.99 | 4.57 | 1.72 |
