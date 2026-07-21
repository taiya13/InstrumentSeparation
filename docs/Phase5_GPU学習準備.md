# Phase 5 — GPU 学習準備の完成

**目的**: AI の改良はしない。**GPU を用意した瞬間に「本番学習をすぐ開始できる」状態**を、GPU 無しの現環境で完成させる。
**日付**: 2026-07 / **ステータス**: 完了・全機能を CPU 上でエンドツーエンド動作確認済み。

---

## 0. エグゼクティブサマリー

- **GPU 移行機能を実装**: CUDA 自動判定、**Mixed Precision（AMP, bf16/fp16 自動）**、**Gradient Accumulation**、**Gradient Checkpointing**、**AdamW + cosine+warmup + grad-clip**、**Distributed（DDP, torchrun）対応**（将来用）。すべて**device 非依存**で、GPU 無しでも動作（CPU にフォールバック）。
- **本番設定を 4 段階**用意: `configs/prod/roformer_{small,15gb,24gb,48gb}.yaml`。VRAM に応じて arch/segment/batch/grad-accum を調整。
- **ワンコマンド自動化**（`src/run_pipeline.py`）: **データ確認 → 学習 → Validation → Checkpoint → Resume → 推論 → 評価 → レポート出力**を一括実行。
- **実験管理**（`src/mss/experiment.py`）: 実験ごとに **モデル / データセット / 学習条件 / Git commit / 環境 / 評価結果**を `runs/<id>/run.json`＋`report.md` に自動保存。
- **GPU 自動判定**（`src/mss/gpu.py`）: 起動時に **GPU / VRAM / CUDA / Driver** を検出し、**適切な config を自動選択**。
- **完全な README / 手順書**を整備。→ **`python src/run_pipeline.py` を叩くだけ**で、GPU があれば本番学習が始まる。

---

## 1. GPU 移行準備（実装済み・CPU 検証済み）

`src/mss/interface.py` の学習ループに以下を実装（**モデル本体は無改造**、学習インフラのみ）:

| 機能 | 実装 | 備考 |
|---|---|---|
| **CUDA 対応** | `gpu.detect()` で device 自動判定、`.to(device)` | CUDA 無しは CPU |
| **Mixed Precision** | `torch.amp.autocast`＋`GradScaler`（device 別） | bf16 優先、非対応時 fp16＋scaler。`amp: true` |
| **Gradient Accumulation** | micro-batch を `grad_accum` 回蓄積して 1 step | 実効 batch = batch×grad_accum |
| **Gradient Checkpointing** | backbone の `enable_grad_checkpointing()` フック | tiny_masker は対応。**bs_roformer 1.2.4 は非公開**→segment/band で VRAM 調整 |
| **AdamW / cosine / warmup / grad-clip** | `optimizer/scheduler/warmup/grad_clip` 設定 | 学習方法の本番機能 |
| **Distributed（DDP）** | `wrap_ddp()`＋rank ゲート＋DDP 解除 checkpoint | `torchrun` で起動（`scripts/launch_distributed.sh`） |

> 検証: CPU 上で `AMP(bf16)+grad_accum+grad_ckpt+cosine` 学習、Resume、DDP 単プロセス経路が動作。GPU では同コードがそのまま高速化・省メモリ動作する。

---

## 2. 本番学習設定（4 段階）

`configs/prod/`（Mel-Band RoFormer, 論文アーキ準拠。データは既定 SynthSOD＝オーケストラ、コメントで MUSDB へ切替可）:

| tier | 対象 | dim/depth/bands | seg | batch×accum | AMP | 用途 |
|---|---|---|---|---|---|---|
| **small** | CPU / GPU 無し | 48 / 2 / 24 | 1.0s | 2×2 | off | パイプライン検証（合成コーパス） |
| **15gb** | RTX 16GB級 | 256 / 8 / 48 | 4.0s | 1×8 | bf16 | 縮小・実データ |
| **24gb** | 3090/4090 | 384 / 10 / 60 | 6.0s | 2×4 | bf16 | 準・論文 |
| **48gb** | A6000/A100 | 384 / 12 / 60 | 8.0s | 4×2 | bf16 | 論文スケール |

- **論文どおりの再現**は 48gb（＋必要なら MUSDB splits）。GPU 自動判定が VRAM から自動選択。
- 参考: 別途 `configs/train_roformer_musdb_paper.yaml`（Phase 4）が MUSDB の純・論文設定。

---

## 3. 学習自動化（`src/run_pipeline.py`）

**一括実行**: `データ確認 → 学習 → Validation → Checkpoint → Resume → 推論 → 評価 → レポート`

```bash
python src/run_pipeline.py                          # GPU/VRAM から config 自動選択
python src/run_pipeline.py --config configs/prod/roformer_24gb.yaml --name musdb24
python src/run_pipeline.py --resume checkpoints/prod_24gb/final.pt   # 再開
python src/run_pipeline.py --skip-train --resume <ckpt>             # 評価のみ
```
- **データ確認**: splits/manifest の存在・件数・音源数を検査（無ければ取得コマンドを案内）。
- **学習**: 上記 GPU 機能をすべて適用、valid/checkpoint 込み。
- **推論＋評価**: test split の各曲で `presence→分離→SI-SDR/SDR/SIR/SAR`（楽器/カテゴリ/曲別）。
- **レポート**: 実験ディレクトリに集約。

---

## 4. 実験管理（`src/mss/experiment.py`）

各 run を `runs/<timestamp>_<name>/` に自動記録:
- **モデル**（name＋ハイパラ）、**データセット**（splits/target_level/sr）、**学習条件**（optimizer/lr/amp/accum…）
- **Git commit / branch / dirty**、**環境**（GPU/VRAM/CUDA/Driver）
- **評価結果**（per-category / per-song SI-SDR ほか）
- 出力: `run.json`（機械可読）＋ `report.md`（人間可読）

> 検証済み: run.json に model/dataset/git_commit/environment/metrics が正しく保存されることを確認。

---

## 5. GPU 環境チェック（`src/mss/gpu.py`）

起動時に **GPU / VRAM / CUDA / Driver** を検出し、**config を自動選択**:
```bash
bash scripts/gpu_check.sh
# CUDA available / GPU名 / VRAM / Driver / bf16 / -> auto config
```
選択規則: GPU 無し→small、<16GB→15gb、<32GB→24gb、それ以上→48gb。

---

## 6. ドキュメント / 学習手順

新環境での手順（GPU 箱）:
```bash
# 1) 環境
python -m venv .venv && . .venv/bin/activate
# GPU の CUDA ビルドを pytorch.org から入れてから:
pip install -r environment/requirements-platform.txt

# 2) データ用意（自由ネットワーク環境）
python src/prepare_datasets.py --dataset synthsod --root /data/synthsod --out data/synthsod
#   （または MUSDB: --dataset musdb18hq --root /data/musdb18hq --out data/musdb18hq）

# 3) 環境チェック（自動 config 選択の確認）
bash scripts/gpu_check.sh

# 4) 本番学習〜評価〜レポートを一括
bash scripts/train_gpu.sh                 # 自動選択
#   もしくは tier 指定: bash scripts/train_gpu.sh configs/prod/roformer_24gb.yaml run24
#   マルチGPU:          bash scripts/launch_distributed.sh 4 configs/prod/roformer_48gb.yaml run48

# 5) 結果は runs/<id>/ に run.json + report.md + infer/eval_*
```

---

## 7. 動作確認（CPU で実施済み）

| 項目 | 結果 |
|---|---|
| GPU 自動判定 | CUDA False → `roformer_small.yaml` 自動選択 ✓ |
| フルパイプライン（small/CPU） | データ確認→学習→valid→ckpt→推論→評価→実験記録まで完走 ✓ |
| AMP(bf16)+grad_accum+grad_clip+AdamW+cosine | 動作（lr が cosine で減衰）✓ |
| Gradient Checkpointing（tiny_masker） | `enabled` で学習成功 ✓ |
| Resume / 評価のみ（--skip-train） | チェックポイント復元→評価まで動作 ✓ |
| 実験記録 | run.json に model/dataset/git/env/metrics 保存 ✓ |
| DDP 経路 | 単プロセスで rank ゲート・DDP 解除 checkpoint を確認（多GPU は GPU 箱で）|

> 数値自体は極小 config・CPU・少 step のため低品質（品質追求は本番学習の役割）。本 Phase の目的は**準備の完成**であり、それは達成済み。

---

## 8. 成果物一覧

```
src/mss/gpu.py                      … GPU/VRAM/CUDA/Driver 判定＋config 自動選択
src/mss/experiment.py               … 実験管理（run.json + report.md）
src/mss/interface.py（拡張）        … AMP/grad-accum/grad-ckpt/AdamW/cosine/DDP 対応の学習ループ
src/run_pipeline.py                 … 一括自動化（データ確認→学習→推論→評価→レポート）
configs/prod/roformer_{small,15gb,24gb,48gb}.yaml … VRAM 段階別 本番設定
scripts/gpu_check.sh                … GPU 環境チェック
scripts/train_gpu.sh                … ワンコマンド本番実行（自動選択）
scripts/launch_distributed.sh       … マルチGPU（torchrun/DDP）
environment/requirements-platform.txt … プラットフォーム依存一括
docs/Phase5_GPU学習準備.md          … 本書
```

## 9. 結論
- **GPU を挿した瞬間に `python src/run_pipeline.py` だけで本番学習〜評価〜レポートが走る**状態を完成させた。
- CUDA/AMP/accum/checkpoint/DDP・VRAM 別設定・自動化・実験管理・GPU 自動判定・ドキュメントを、**GPU 無し環境で実装・検証**済み。
- 次は実際に GPU 環境で本番学習（Phase 6 以降）。ここで初めて、Phase 0–5 の土台の上に**独自改良**（同一楽器の空間/score 条件付け 等）を載せていく。
