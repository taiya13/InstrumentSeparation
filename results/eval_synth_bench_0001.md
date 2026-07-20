# Evaluation — synth_bench_0001 (synth_bench/test)
estimates: `oracle_irm(demo)` | overall: {'mean_si_sdr': 1.92, 'median_si_sdr': 1.92, 'n_sources': 12}

## Per-instrument (SI-SDR, dB)
| instrument | category | SI-SDR | SI-SDRi | oracle ceiling |
|---|---|---|---|---|
| トランペット (`br.tpt`) | br | 5.51 | 20.01 | 5.51 |
| フルート (`ww.fl`) | ww | 3.61 | 11.53 | 3.61 |
| オーボエ (`ww.ob`) | ww | 3.57 | 13.82 | 3.57 |
| ヴァイオリン (`str.vln`) | str | 3.47 | 9.79 | 3.47 |
| クラリネット (`ww.cl`) | ww | 3.41 | 12.74 | 3.41 |
| コントラバス (`str.cb`) | str | 2.55 | 13.64 | 2.55 |
| ホルン (`br.hn`) | br | 1.29 | 11.61 | 1.29 |
| ヴィオラ (`str.vla`) | str | 1.17 | 12.77 | 1.17 |
| ティンパニ (`perc.timp`) | perc | 0.62 | 21.19 | 0.62 |
| トロンボーン (`br.tbn`) | br | -0.1 | 9.59 | -0.1 |
| チェロ (`str.vlc`) | str | -0.46 | 11.38 | -0.46 |
| ファゴット (`ww.bn`) | ww | -1.59 | 9.59 | -1.59 |

## Per-category (mean SI-SDR)
| category | mean | median | n |
|---|---|---|---|
| str | 1.68 | 1.86 | 4 |
| ww | 2.25 | 3.49 | 4 |
| br | 2.23 | 1.29 | 3 |
| perc | 0.62 | 0.62 | 1 |

## SDR / SIR / SAR @ category level
| source | SDR | SIR | SAR |
|---|---|---|---|
| str | 4.89 | 6.94 | 9.92 |
| ww | 5.66 | 8.3 | 9.67 |
| br | 5.58 | 9.43 | 8.36 |
| perc | 2.27 | 9.38 | 3.69 |
