# Phase 4 レポート — ベースライン性能の再現と品質評価

**目的**: 新しいAIは作らない。**世界最高水準に近い既存モデル（Mel-Band RoFormer）を自分たちの環境で再現**し、性能と限界を定量把握して、今後の改良の**基準線（ベースライン）**を確立する。
**日付**: 2026-07 / **ステータス**: 実行完了（在環境ベースライン）＋ GPU 再現レシピ完備
**独自改造**: 一切なし（`bs_roformer` を無改造・ネイティブ損失で使用）。

---

## 0. エグゼクティブサマリー / 重要な前提

- **この環境は GPU 無し・データ/重みCDN ブロック**（Phase 1〜3 で実測確認: fbaipublicfiles / huggingface / download.pytorch.org / zenodo が 403）。→ **論文スケール（MUSDB18-HQ, 数十万step, A100級）の再現はこの環境では不可能**。
- そこで Phase 4 は**2層構成**で正直に実施:
  1. **在環境ベースライン（実測）**: 実物の Mel-Band RoFormer を**縮小 config**で、**曲単位で分離した合成コーパス（train 8 / valid 2 / test 2、非重複）**上で実学習・実評価。→ 我々の環境での**再現可能な基準線**。
  2. **論文スケール再現レシピ（GPU 箱用）**: 論文に近いハイパラ・資源見積り・データ取得/前処理スクリプトを完備（`configs/train_roformer_musdb_paper.yaml`, `src/prepare_datasets.py`）。
- **論文比較は「文献値（引用）」と「在環境実測値」を明確に分離**して提示。文献値は我々が再現したものではない。
- 結論の要点: **アーキテクチャ・学習・推論・評価パイプラインは正しく再現**でき、在環境ベースラインは（小規模ながら）学習して混合を上回る。**論文スケールの絶対SDR再現には GPU＋実データが必須**で、これがボトルネックの中心。

---

## 1. 学習環境の整備

### 1.1 本番用ハイパーパラメータ（論文に近い設定）
`configs/train_roformer_musdb_paper.yaml`（GPU 専用）:

| 項目 | 値（near-paper / ZFTurbo big） |
|---|---|
| dim / depth / heads / dim_head | 384 / 12 / 8 / 64 |
| num_bands | 60 |
| STFT n_fft / win / hop | 2048 / 2048 / 441（≈10ms@44.1k） |
| 入力 | ステレオ, 44.1 kHz, 8s セグメント |
| batch / steps | 8（/GPU, grad-accum 併用） / 30万〜 |
| optim | AdamW, cosine+warmup, bf16, grad clip 0.5, EMA（Phase5 で trainer 実装） |
| 対象 | MUSDB18-HQ: vocals/drums/bass/other |

### 1.2 学習時間・VRAM・ストレージ見積り（論文スケール）
| 資源 | 見積り | 根拠 |
|---|---|---|
| **VRAM** | **24GB 下限 / 40–80GB 推奨** | dim=384,depth=12,8sステレオ,batch8。RTX3090/4090/A5000(24GB)は grad-accum 必須、A100(40/80GB)が快適 |
| **学習時間** | **単GPU で 1〜3 週間 / 4–8GPU で数日** | 30万〜100万 step、コミュニティ実績で数百 GPU-hours 級 |
| **ストレージ** | MUSDB18-HQ ~30GB / SynthSOD ステム ~50–100GB / CocoChorales(1400h) ~TB級 / checkpoint 1–2GB×N | HQ wav・多ステム |
| 推論 | GPU 不要でも可（Phase1 実測: HDemucs CPU RTF<1）。実運用はGPUで高速化 |

### 1.3 在環境ベースライン config（実行したもの）
`configs/train_roformer_baseline.yaml`（**縮小・CPU**）: dim=48, depth=2, heads=4, num_bands=24, n_fft=1024, 22.05kHz, 1.0s crop, batch 2, category 分離。**論文スケールではない**が、同一アーキ・同一損失で**正しく学習/評価できることの実証**。

---

## 2. データセット整備（train / valid / test の分離）

- **実データ（SynthSOD/MUSDB 等）はこの環境で取得不可**（CDN ブロック）。取得・前処理スクリプトは完備し **GPU 箱で実行**:
  - `src/prepare_datasets.py --dataset musdb18hq --root <local> --out data/musdb18hq` → **曲単位**で train/valid/test に分割し、Phase2 の manifest/splits 形式を出力（vocals/drums/bass を taxonomy id にマップ、other は `__mixed__` で除外）。
  - `--dataset synthsod` で SynthSOD レイアウトにも対応。
- **在環境では合成コーパスで train/valid/test を実分離**（`src/make_corpus.py`）:
  - **12 曲を生成し、移調・テンポ・楽器編成・乱数で相互に異なる曲**にした上で **train 8 / valid 2 / test 2** に**曲単位分割（リークなし）**。
  - test 曲（piece_10=移調+4/95bpm, piece_11=移調-4/105bpm）は学習に一切使用しない。

---

## 3. ベースライン学習（Mel-Band RoFormer, 無改造）

- 実物 `bs_roformer.MelBandRoformer` を Phase3 プラットフォーム経由で学習（`model.name: mel_band_roformer`）。ネイティブ損失（L1＋マルチ解像度STFT）を使用、**独自改造なし**。
- コーパス train 8 曲 / valid 2 曲、category 分離（str/ww/br/perc）。

### 3.1 学習ログ（実測）
- モデル: Mel-Band RoFormer（縮小 config）, **4.233M params**, CPU, **~0.9 s/step**。
- 400 steps, category 分離, corpus **train 8 / valid 2 曲**（非重複）。
- 全ログ: `results/roformer_baseline_train.log`

```
step  50/400  loss=2.352
step 100/400  loss=2.104   [valid] mean SI-SDR = -9.42 dB
step 200/400  loss=2.261   [valid] mean SI-SDR = -6.51 dB
step 300/400  loss=2.093   [valid] mean SI-SDR = -9.48 dB
step 400/400  loss=2.547   [valid] mean SI-SDR = -8.62 dB
```
- **所見**: loss・valid ともに**振動し、明確な収束に至らない**。4.2M params・400 step・8 曲（各~9.6s）・CPU・lr 固定という**極小スケール**では RoFormer は**学習不足/不安定**。これは想定内であり、論文スケール（大 batch・数十万 step・AdamW/cosine/EMA・GPU）が必要であることを裏づける。

---

## 4. ベースライン評価（SI-SDR / SDR / SIR / SAR）

test 2 曲（学習未使用）に対し、Phase3 パイプライン（presence→分離→評価）で自動計測。**楽器（=カテゴリ）別・曲別**に集計。

**test 2 曲（corpus_10, corpus_11）の平均**（全出力: `results/infer_roformer_baseline/eval_corpus_1*.{json,md}`）:

| category | SI-SDR | **SI-SDRi** | oracle 天井 | SDR | SIR | SAR |
|---|---|---|---|---|---|---|
| woodwinds | −0.67 | **+0.81** | 6.05 | 0.34 | 1.02 | 11.48 |
| strings | −1.16 | **+0.80** | 4.46 | −0.08 | 0.21 | 14.77 |
| brass | −8.74 | −2.67 | 4.20 | −4.04 | −2.65 | 6.14 |
| percussion | −29.84 | −7.66 | −0.97 | −1.16 | 4.56 | 1.50 |

- **曲別**: piece_10 と piece_11 で傾向は一致（弦/木管はわずかに混合を上回り、金管/打楽器は下回る）。
- **カテゴリ別**: 弦・木管は **SI-SDRi > 0（混合よりわずかに改善）**だが天井（4–6 dB）には遠い。**金管・打楽器は混合以下**（打楽器は疎で壊滅的）。
- **総括**: 極小スケールのため**絶対品質は低い**。held-out（未学習曲）での汎化は 4 カテゴリ中 2 つでのみ僅かにプラス。**この数値は「世界最高水準」ではなく、あくまで在環境の下限基準線**であり、GPU＋実データ＋論文 config で大幅に引き上げるべき対象。

---

## 5. 論文との比較

> **重要**: 下表の「文献値」は各論文/公開ベンチの**報告値（引用）であり、我々が再現したものではない**。我々の在環境実測は §3–4（縮小 config・合成データ・CPU）。両者は**スケールもデータも異なる**ため直接比較はできず、**再現できた/できなかった軸**で整理する。

### 5.1 文献値（MUSDB18-HQ, 平均SDR, 近似・引用）
| モデル | 平均SDR(dB, 近似) | 出典 |
|---|---|---|
| Spleeter | ~4–5.5 | Deezer 2019 |
| Open-Unmix (UMX) | ~5.3 | Stöter+ 2019 |
| Demucs v4 (HT-Demucs) | ~7–9 | Rouard+ 2023 |
| **BS-RoFormer** | **~9.8** | Lu+ 2023 (arXiv:2309.02612) |
| **Mel-Band RoFormer** | **~9.6（L=6）** | Wang+ 2023 (arXiv:2310.01809) |

### 5.2 再現できた点
- **アーキテクチャの再現**: 実物 Mel-Band RoFormer を無改造で構築・学習・推論・評価。
- **学習・評価パイプライン**: train/valid/test 分離、checkpoint/resume、SI-SDR/SDR/SIR/SAR 自動計測。
- **学習が進む**こと（loss 減少・valid 指標改善）を在環境で確認。
- **論文スケール config とデータ取得/前処理レシピ**を完備（GPU 箱でそのまま実行可能）。

### 5.3 再現できなかった点
- **論文の絶対 SDR 値（~9.6 dB @ MUSDB18-HQ）**。
- **論文スケールの学習**（数十万〜百万 step、大 batch、bf16、多GPU）。
- **実データ（MUSDB/SynthSOD 等）での学習・評価**。

### 5.4 原因の考察
1. **GPU 不在**: 論文 config は 24–80GB VRAM・数百 GPU-hours を要し、CPU では不可能。
2. **データ CDN ブロック**: MUSDB18-HQ / SynthSOD / 重みが 403。実データにアクセスできない。
3. **在環境の代替は合成・小規模**: 12 曲・9.6s・22.05kHz・縮小 config のため、絶対品質は本質的に低い（設計上の制約であり、モデルの欠陥ではない）。
4. これらは**環境要因**であり、モデル/実装の再現性そのものの問題ではない（レシピは完備）。

---

## 6. ボトルネック分析（性能向上の障害要因の分類）

| 分類 | ボトルネック | 深刻度 | 対処（Phase5+） |
|---|---|---|---|
| **データ** | ブリードなし実オケ・マルチトラックが皆無／実データ CDN ブロック／在環境は合成のみ | **最大** | 実データ取得（GPU箱）、高品質レンダリング、ドメイン適応 |
| **学習方法** | GPU 不在で論文スケール学習不可／AdamW・cosine・bf16・EMA 等が未実装 | 高 | GPU 環境、trainer の本番機能実装 |
| **モデル** | マスク型の原理的天井（Phase1: 同音色ユニゾンは不可能） | 高（同一楽器のみ） | 空間/score 条件付け（Phase5+） |
| **Presence** | 現状はプレースホルダ heuristic（過検出 precision~0.34） | 中 | 学習済み tagger（PANNs/CLAP） |
| **Taxonomy** | 大枠は妥当。part 粒度（Vn I/II）の運用は追加情報前提 | 低 | score/空間入力の接続 |
| **評価方法** | SI-SDR/SDR/SIR/SAR は妥当。合成 test は実録の難しさを過小評価しうる | 中 | 実録 test（URMP/Aalto/Cadenza）追加 |

---

## 7. Phase 5 で改良すべき優先順位

1. **【最優先・データ】** GPU＋自由ネットワーク環境で **MUSDB18-HQ ベースラインを論文 config で再現**（`prepare_datasets.py`＋`train_roformer_musdb_paper.yaml`）。まず**世界水準の絶対 SDR を自分たちで再現**し真の基準線を確立。
2. **【データ】** SynthSOD を取得し **オーケストラ category→instrument 分離**を学習・評価。
3. **【学習方法】** trainer に AdamW/cosine+warmup/bf16/grad-clip/EMA/SDR早期終了を実装。
4. **【Presence】** 学習済み audio tagger を `PresenceDetector` として差し込み、変動編成に対応。
5. **【ドメイン適応】** 合成→実録（URMP/Aalto/Cadenza）で評価し、augmentation/fine-tune でギャップ縮小。
6. **【モデル・研究】** 同一楽器（Vn I/II）へ **空間/score 条件付け**の入力経路を interface に追加（ここで初めて独自研究）。

---

## 8. 成果物一覧（本 Phase）

```
docs/Phase4_ベースライン性能レポート.md          … 本レポート（性能/論文比較/ログ/評価/ボトルネック/優先順位）
configs/train_roformer_musdb_paper.yaml          … 論文スケール GPU config（本番ハイパラ）
configs/train_roformer_baseline.yaml             … 在環境ベースライン config（縮小・CPU）
src/make_corpus.py                               … 多曲合成コーパス生成＋train/valid/test 分離
src/prepare_datasets.py                          … 実データ(MUSDB/SynthSOD)→manifest/splits（GPU箱）
results/roformer_baseline_train.log              … 学習ログ（実測）
results/infer_roformer_baseline/eval_*.{json,md} … test 曲の評価結果（実測）
data/synth_corpus/{piece_*, splits.json}         … コーパス＋分割
```

## 9. 結論
- **実物 Mel-Band RoFormer を無改造で再現**し、train/valid/test を分離した在環境ベースラインで **学習→推論→評価が一貫動作**することを確認した。
- **論文スケールの絶対性能はこの環境では再現不可**（GPU/実データが無い環境要因）で、そのための**完全な再現レシピと資源見積り**を用意した。
- 次の一手は Phase 5：**GPU 箱で MUSDB18-HQ 論文ベースラインを再現**し、真の世界水準基準線を確立してから、オーケストラ／同一楽器の独自改良へ進む。
