# Phase 3 設計・実装レポート — AIバックボーン統合／学習・推論・評価プラットフォーム

**目的**: 特定の1モデルを作ることではなく、**「どの音源分離モデルでも同じ基盤上で学習・推論・評価できる研究プラットフォーム」**を完成させる。
**日付**: 2026-07 / **ステータス**: 実装完了・エンドツーエンド動作確認済み（CPU）
**方針**: 独自AIは作らない。既存実装（Mel-Band RoFormer / HT-Demucs）を**改造せず**共通インターフェースへ統合し、`学習 → 推論 → 評価`が一貫して動く状態を作る。**新モデルを数時間で差し替えられる構造**を最優先。

---

## 0. エグゼクティブサマリー

- **共通 Backbone Interface** を実装（`src/mss/interface.py`）。`build / train / infer / separate / export / save_checkpoint / load_checkpoint` の統一 API。**backbone はラッパ（nn.Module を継承しない）**設計にし、`nn.Module.train()` と衝突させず、既存モデルを**無改造で**包む。
- **レジストリで差し替え1行**（`src/mss/registry.py`）。`model.name: tiny_masker → mel_band_roformer → htdemucs` を YAML で切り替えるだけ。登録済み backbone: **`tiny_masker` / `mel_band_roformer` / `htdemucs`**。
- **Mel-Band RoFormer を最初の統合対象**として組込み（`bs_roformer` v1.2.4, MIT を無改造利用）。モデル自身の forward と**ネイティブ損失**（`target=` 渡し）をそのまま使用。
- **Taxonomy / Presence と接続**。Dataset が manifest の正解ステムを Taxonomy で任意粒度（category/instrument/part）へ集約して教師化。推論時は Presence が「存在する楽器のみ」を active set として出力。
- **学習ループ完備**: Dataset / DataLoader / Train / Validation / Checkpoint / **Resume**。GPU 無しでも回るよう、軽量な `tiny_masker` を CPU で実学習してループ全体を実証。
- **推論パイプライン**: 1曲入力 → **Presence 推定 → モデル推論 → 楽器別出力** を自動実行（`src/mss/pipeline.py`）。
- **評価基盤と自動接続**: 推論後に Phase 2 の評価へ自動連結し **SI-SDR / SDR / SIR / SAR** を自動計算。
- **結果**: `train → resume → infer → evaluate` を**エンドツーエンドで実行成功**（本書 §7 に実測値）。

---

## 1. Backbone Interface（共通インターフェース）

**ファイル**: `src/mss/interface.py`, `src/mss/registry.py`

### 1.1 API
| メソッド | 役割 | 実装場所 |
|---|---|---|
| `build(cfg)` | config からモデル構築 | 基底（＋ subclass `_build`） |
| `train(loaders, ...)` | 汎用学習ループ（checkpoint / resume 込み） | **基底（共通）** |
| `infer(mix)` | モデル素の順伝播 `[B,C,T]→[B,S,C,T]` | subclass `_forward` |
| `separate(wav, sr, targets, presence)` | 1ファイル分離 → `{taxonomy_id: wav}` | **基底（共通）** |
| `export(path, fmt)` | TorchScript / ONNX 書き出し | 基底 |
| `save_checkpoint / load_checkpoint` | 重み・optim・step の保存/復元（resume の要） | 基底 |

### 1.2 設計の要点
- **backbone は「ラッパ」**: 実モデルを `self.net` に保持。`nn.Module` を継承しないので、要望どおり `train()` を素直に実装できる（`nn.Module.train()` と非衝突）。既存実装を**そのまま包む**思想と一致。
- **subclass が書くのはモデル固有部分のみ**: `_build(cfg)`（実モデル生成）、`_forward(mix)`（入出力形状の整合）、必要なら `_loss(mix, targets)`（既定は波形 L1、ネイティブ損失があれば上書き）。
- **`output_targets`**（出力楽器 id の順序付きリスト）を backbone が保持 → stem index ↔ taxonomy id を対応づけ、Presence・評価と接続。
- **レジストリ**: `@register("name")` で登録、`get_backbone(name).build(cfg)` で取得。**新モデル追加＝ラッパ1ファイル**。

### 1.3 登録済みバックボーン（3種・アーキ差異が大きい＝汎用性の証明）
| name | 実体 | ライセンス | 役割 |
|---|---|---|---|
| `mel_band_roformer` | lucidrains `bs_roformer.MelBandRoformer` | MIT | **本命 SOTA バックボーン**（無改造） |
| `htdemucs` | `torchaudio.models.HDemucs` | BSD/MIT系 | 別系統アーキの統合実証（波形＋STFTハイブリッド） |
| `tiny_masker` | STFT マスキング小型ネット | 本repo | **CPU 学習デモ用**（ループ全体を GPU 無しで実証） |

> `tiny_masker` は「独自AI」ではなく、GPU 無し環境でも `train→infer→evaluate` を通すためのリファレンス小型器。SOTA は上2つ。

---

## 2. Mel-Band RoFormer 統合

**ファイル**: `src/mss/backbones/roformer.py`
- `bs_roformer`（PyPI, MIT）の `MelBandRoformer` を**そのまま**利用。**独自改造なし**（プロジェクト計画どおり後回し）。
- 推論: `self.net(raw_audio)`（mono は `(B,T)`, stereo は `(B,2,T)`）→ `(B, num_stems, C, T)`。
- 学習: `self.net(raw_audio, target=targets)` で**モデル自身の損失**（L1＋マルチ解像度 STFT）を使用。
- ハイパラは cfg 化：CPU スモークは小さく（dim=32, depth=1…）、**GPU では本来の SOTA config へスケール**。
- 動作確認: tiny config（約1.3M params）で CPU 順伝播・ネイティブ損失計算・数ステップ学習まで確認（§7）。

---

## 3. Presence と Taxonomy の接続

**ファイル**: `src/mss/dataset.py`, `src/mss/pipeline.py`（＋Phase 2 の `taxonomy.py` / `presence.py`）

- **学習教師の生成**: `ManifestDataset` が manifest の各ステムを **Taxonomy の `rollup` で `target_level`（category/instrument/part）へ集約**し、`output_targets` 順にスタック。→ 同じデータで category 分離 → instrument 分離 → part 分離へ段階拡張可能。
- **presence ベクトル**も各サンプルに付与（存在楽器＝1）。
- **推論時**: `pipeline.estimate_presence` が Presence（Oracle または heuristic）で**存在楽器を推定 → target_level へ集約 → active set** を決め、**その楽器のみ** `separate()` が出力。曲ごとに登場楽器が違っても「存在するものだけ」出す Phase 2 仕様を実現。

---

## 4. 学習ループ（Dataset / DataLoader / Train / Validation / Checkpoint / Resume）

**ファイル**: `src/mss/dataset.py`, `src/mss/interface.py`（`train`）, `src/run_train.py`

- **Dataset**: manifest 群からランダム固定長クロップ（`region` で train=前70% / valid=後30% に時間分割しリーク回避）。返り値 `mixture[1,T]`, `targets[S,1,T]`, `presence[S]`。
- **DataLoader**: `build_loaders` が train/valid を生成（バッチ化・シャッフル）。
- **Train**: 汎用ループ（`SeparationBackbone.train`）。optimizer 構築 → step ループ → 損失 backward → ログ。
- **Validation**: 一定 step ごとに valid の平均 SI-SDR を計算。
- **Checkpoint**: `ckpt_every` ごと＋終了時に `net / optim / step / output_targets / cfg` を保存。
- **Resume**: `--resume <ckpt>` で `net`＋`optim`＋`step` を復元して継続（§7 で実証）。

設定は `configs/train.example.yaml`（tiny_masker）/ `configs/train_roformer.example.yaml`（RoFormer）。

---

## 5. 推論パイプライン（1曲 → 楽器別出力）

**ファイル**: `src/mss/pipeline.py`, `src/run_infer.py`
```
run_song(backbone, manifest, out_dir, target_level, presence_mode):
  1) Presence 推定  -> active set（存在楽器のみ）
  2) backbone.separate(mixture, targets=active)  -> 楽器別 stem
  3) out_dir/<taxonomy_id>.wav に書き出し
  4) 評価基盤へ自動連結（§6）
```
- checkpoint から backbone を**メタ情報だけで再構築**（別 config 不要）→ `load_checkpoint` → 実行。

---

## 6. 評価基盤との統合（自動）

**ファイル**: `src/mss/pipeline.py` → `src/evaluation.py`（Phase 2）
- 推論直後に `evaluation.evaluate(manifest, estimates_dir, level)` を自動呼び出し。
- 出力: **SI-SDR / SI-SDRi / SDR / SIR / SAR** を**楽器別・カテゴリ別・曲別**に集計、**オラクル天井**併記、同一楽器の identity confusion も。
- 成果物: `results/infer_.../eval_<song>.{json,md}`。

---

## 7. 動作確認（実測・エンドツーエンド）

> 実行: `bash scripts/run_platform_poc.sh`（train → resume → infer → evaluate → RoFormer smoke）

### 7.1 学習ループ（`tiny_masker`, CPU, category 分離）
- 0.023M params / 400 steps / ~1.2s/step。
- **loss 単調減少** 0.049 → 0.043。
- **validation SI-SDR が単調改善**（step 100/200/300/400）: **−10.25 → −7.71 → −6.64 → −6.07 dB** ＝確かに学習している。
- checkpoint（step100〜400＋final）保存。

### 7.2 Resume
- `final.pt`（step 400）から復元 → step 400 起点で継続 → **OK**（`[resume] from ... at step 400`）。

### 7.3 推論 ＋ 自動評価（`tiny_masker`, 全曲）
`presence(oracle) → active=[br,perc,str,ww] → 4 stem 出力 → 自動評価`:

| category | SI-SDR | **SI-SDRi** | oracle 天井 | SDR | SIR | SAR |
|---|---|---|---|---|---|---|
| brass | 0.26 | **+6.08** | 4.75 | 1.39 | 5.06 | 5.00 |
| woodwinds | 0.50 | **+2.34** | 5.38 | 0.84 | 2.46 | 7.85 |
| strings | 0.05 | **+2.06** | 4.35 | 0.52 | 1.66 | 9.14 |
| percussion | −28.78 | −8.21 | 0.75 | −22.05 | −15.51 | −5.33 |

- **音程系3カテゴリ（弦/木管/金管）で混合より改善（SI-SDRi > 0）**し、オラクル天井の一部に到達。
- 打楽器（時間的に疎なティンパニ）は失敗＝softmax マスカが疎音源を取りこぼす**既知の難所**を可視化。
- SI-SDR / SDR / SIR / SAR が**カテゴリ別に自動計算**され `results/infer_tiny_masker/eval_*.{json,md}` に出力。

### 7.4 モデル差し替え（同一 API で Mel-Band RoFormer）
- 変更は `model.name: mel_band_roformer` の**1行のみ**（`configs/train_roformer.example.yaml`）。1.305M params。
- **ネイティブ損失で学習**（6 step で 3.36 → 2.70）、checkpoint 保存、`run_infer` で推論＋評価も同一パイプラインで完走。
- **Dataset / 学習ループ / 推論 / 評価は一切変更せず再利用** ＝「新モデルを数時間で差し替える」構造を実証。

> 注: `tiny_masker`/RoFormer とも**小型 config・合成1曲・CPU・少 step** のため絶対品質は低い。本 Phase の目的は**プラットフォームが一貫動作すること**であり、品質は Phase 4 で実バックボーン＋実データ＋GPU により追求する。

---

## 8. 「新モデルを数時間で差し替える」手順

1. `src/mss/backbones/<newmodel>.py` に 15〜30 行のラッパを書く:
   ```python
   @register("newmodel")
   class NewModel(SeparationBackbone):
       def _build(self, cfg): return SomeExistingNet(...)
       def _forward(self, mix): return self.net(...)   # -> [B,S,C,T]
       # 必要なら _loss を上書き
   ```
2. `configs/train.example.yaml` の `model.name: newmodel` に変更。
3. `python src/run_train.py --config ...` → `python src/run_infer.py ...`。
   Dataset・学習ループ・推論・評価は**一切変更不要**で再利用される。

---

## 9. 成果物一覧（本 Phase）

```
src/mss/interface.py            … Backbone Interface（train/infer/separate/export/checkpoint）
src/mss/registry.py             … モデル・レジストリ（差し替え1行）
src/mss/backbones/roformer.py   … Mel-Band RoFormer 統合（無改造）
src/mss/backbones/htdemucs.py   … HT-Demucs 統合（torchaudio）
src/mss/backbones/tiny_masker.py… CPU 学習デモ用リファレンス器
src/mss/dataset.py              … Manifest 駆動 Dataset/DataLoader（Taxonomy 集約）
src/mss/pipeline.py             … 1曲: presence→分離→楽器別出力→評価
src/run_train.py / run_infer.py … 学習 / 推論 CLI（config 駆動）
configs/train.example.yaml      … 学習設定（tiny_masker）
configs/train_roformer.example.yaml … 学習設定（RoFormer）
scripts/run_platform_poc.sh     … train→resume→infer→evaluate の一括再現
```

## 10. 次フェーズ（Phase 4）への接続
本プラットフォーム上で、**独自改良**を安全に始められる状態になった。想定する最初の改良:
- Mel-Band RoFormer を GPU で本 config に拡張し、SynthSOD で **category → instrument 分離**を学習。
- Presence ヘッド（PANNs/CLAP）を `PresenceDetector` 実装として差し込み、推論を自動化。
- **同一楽器（Vn I/II）**: 空間/score 条件の入力経路を interface に追加（`_forward` の入力を拡張）。
- 評価のバッチ/リーダーボード化（データセット別・モデル別）。
