# Evaluation — corpus_10 (synth_corpus/test)
estimates: `results/infer_roformer_baseline` | overall: {'mean_si_sdr': -9.39, 'median_si_sdr': -4.88, 'n_sources': 4}

## Per-instrument (SI-SDR, dB)
| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |
|---|---|---|---|---|
| 木管楽器 (`ww`) | ww | -0.48 | 1.37 | 5.86 |
| 弦楽器 (`str`) | str | -0.85 | 0.75 | 4.88 |
| 金管楽器 (`br`) | br | -8.9 | -2.79 | 4.47 |
| 打楽器 (`perc`) | perc | -27.34 | -5.45 | -0.25 |

## Per-category (mean SI-SDR)
| category | mean | median | n |
|---|---|---|---|
| str | -0.85 | -0.85 | 1 |
| ww | -0.48 | -0.48 | 1 |
| br | -8.9 | -8.9 | 1 |
| perc | -27.34 | -27.34 | 1 |

## SDR / SIR / SAR @ category level
| source | SDR | SIR | SAR |
|---|---|---|---|
| str | 0.23 | 0.48 | 15.46 |
| ww | 0.05 | 0.51 | 12.79 |
| br | -3.05 | -1.61 | 6.36 |
| perc | -1.33 | 4.54 | 1.28 |
