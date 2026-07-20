# Evaluation — synth_bench_0001 (synth_bench/test)
estimates: `results/infer_roformer` | overall: {'mean_si_sdr': -12.81, 'median_si_sdr': -2.44, 'n_sources': 4}

## Per-instrument (SI-SDR, dB)
| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |
|---|---|---|---|---|
| 木管楽器 (`ww`) | ww | -0.8 | 1.04 | 5.38 |
| 弦楽器 (`str`) | str | -2.14 | -0.13 | 4.35 |
| 金管楽器 (`br`) | br | -2.74 | 3.08 | 4.75 |
| 打楽器 (`perc`) | perc | -45.56 | -24.99 | 0.75 |

## Per-category (mean SI-SDR)
| category | mean | median | n |
|---|---|---|---|
| str | -2.14 | -2.14 | 1 |
| ww | -0.8 | -0.8 | 1 |
| br | -2.74 | -2.74 | 1 |
| perc | -45.56 | -45.56 | 1 |

## SDR / SIR / SAR @ category level
| source | SDR | SIR | SAR |
|---|---|---|---|
| str | -1.23 | -0.68 | 11.36 |
| ww | 0.1 | 1.05 | 9.69 |
| br | 0.88 | 2.4 | 8.16 |
| perc | -21.45 | -17.87 | -1.0 |
