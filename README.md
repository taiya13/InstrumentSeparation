# InstrumentSeparation

オーケストラ音源を 15〜20 種類程度の楽器単位へ分離することを目標とする研究プロジェクト。

## 現在の進捗

- **Phase 0（調査）** … `docs/Phase0_技術調査レポート.md`
  既存の音源分離AI・オーケストラ特化研究・データセットの徹底調査、比較表、推奨ベースモデル、ロードマップ、実現可能性評価。
- **Phase 1（再現・評価 PoC）** … `docs/Phase1_再現評価レポート.md`
  ベースモデル（HT-Demucs）選定、再現可能な環境構築、CPU 動作確認、**オラクル上限による到達可能品質の実測**、**同一楽器分離が原理的に不可能であることの定量実証**、ボトルネック分析。
- **Phase 2（データ基盤・学習仕様設計）** … `docs/Phase2_データ基盤_学習仕様設計.md`
  拡張可能な **Instrument Taxonomy**（`configs/taxonomy.yaml`）、曲別 **presence 検出**仕様、**マニフェスト駆動データ基盤**（`schemas/manifest.schema.json`）、**楽器別/カテゴリ別/曲別の評価基盤**（`src/evaluation.py`）、データセット整理表、Phase 3 実装一覧。すべて動作確認済み。
- **Phase 3（AIバックボーン統合・学習基盤）** … `docs/Phase3_AIバックボーン統合_学習基盤.md`
  **共通 Backbone Interface**（`src/mss/`）で任意モデルを `train/infer/separate/export/checkpoint` の統一APIに統合。**Mel-Band RoFormer / HT-Demucs / tiny_masker** を1行で差し替え可能。`学習→推論→評価`が一貫動作するプラットフォーム。
- **Phase 4（ベースライン再現・品質評価）** … `docs/Phase4_ベースライン性能レポート.md`
  実物 Mel-Band RoFormer を**無改造**で、train/valid/test を分離した合成コーパスで学習・評価（在環境ベースライン）。**論文スケール GPU config・資源見積り・データ取得/前処理**を完備。論文比較・ボトルネック分析・Phase5 優先順位。GPU/実データ CDN はサンドボックスでブロックのため論文スケール再現は GPU 箱で実施。
- **Phase 5（GPU学習準備の完成）** … `docs/Phase5_GPU学習準備.md`
  **GPU を挿した瞬間に本番学習が始まる**状態を GPU 無しで完成。CUDA自動判定・**AMP/勾配蓄積/勾配チェックポイント/AdamW+cosine/DDP**・VRAM別本番設定4種・**ワンコマンド自動化**（データ確認→学習→推論→評価→レポート）・**実験管理**（`runs/<id>/run.json`）。全機能を CPU で動作確認済み。
- **Phase 6 / Experiment 001（独自研究①：Score条件）** … `docs/Phase6_条件付き分離_研究レポート.md`
  単一仮説「**楽譜（score）条件で同一楽器（第1/第2ヴァイオリン）をモノ分離できるか**」を検証。**音声のみ 0 dB（不可能）→ 楽譜条件 +6.99 dB SI-SDR / +10.5 dB SIR**（オラクル上限の約6割）で**仮説を支持**。`python src/run_phase6_analytic.py` で再現。
- **Experiment 002（独自研究②：Position条件）** … `docs/Experiment002_Position条件_研究レポート.md`
  Score を一切使わず「**空間情報（ステレオILD）のみ**で同一楽器を分離できるか」を Exp 001 と同一条件で検証。**+10.25 dB SI-SDR（pan±0.3）で支持**、ただし**空間分離量に強く依存**（±0.1で+1.5dB）。`python src/run_exp002_position.py` で再現。
- **Experiment 003（独自研究③：Score＋Position 併用）** … `docs/Experiment003_Score_Position併用_研究レポート.md`
  4条件（音声のみ/Score/Position/併用）を最難条件（ユニゾン・近接・両方）で比較。**相補性は非対称・部分的**：Position の近接失敗は Score が補うが、**ユニゾンは Score・Position（マスキング）共通の壁**で併用でも解けない（B: 併用0.64＝両方失敗）。次段（多ch空間フィルタ/生成的手法）への指針を提示。`python src/run_exp003_combined.py` で再現。
- **研究戦略（2ch生成的分離への転換）** … `docs/研究戦略_2ch生成的分離への転換.md`
  最終目標を「**2chステレオの市販/YouTubeクラシック音源**からの楽器分離」に確定（多ch/マイクアレイは対象外）。Exp001-003 で TFマスキングの壁（ユニゾン=同一TF占有）を確認、*Separate and Diffuse*（生成は決定論の理論上限を超える）を根拠に、**「決定論フロントエンド(RoFormer)＋Score条件付き生成的分離(拡散/Flow)」**へ転換。最新手法調査・限界克服分析・優先テーマ・3〜5年ロードマップを提示。
- **研究ツール（GUI/CLI, RTX 5060 Ti 向け）** … `docs/研究ツール_GUI_使い方.md`
  「音源読込→分離→評価」を反復する研究用ツール。**GUI(View)と推論エンジン(Core)を分離**し、**モデルは checkpoint パスのみ＝差し替えても GUI 不変**。既存の Backbone Interface/Taxonomy/Presence/評価基盤をそのまま利用。推論時間・GPU使用率・VRAM・使用モデル・楽器一覧を表示し、研究ログ（`logs/research_log.jsonl`）を保存。GUI: `scripts/run_gui.bat`、ヘッドレス/バッチ: `python src/app/cli.py`。
- **Architecture Design 001（Score-informed 生成分離）** … `docs/ArchitectureDesign001_ScoreInformed生成分離.md`
  システム全体アーキテクチャを設計・決定。採用: **潜在 Rectified Flow-Matching DiT（Score条件・混合アンカー・per-target＋同一セクション同時生成）＋ RoFormer フロントエンド ＋ consistency 蒸留**。入力/エンコーダ/潜在/生成モデル/条件付け/デコーダ/出力/損失/学習/推論を設計、6候補（Diffusion/Flow/Consistency/AR/Bridge等）を比較し採用理由を詳述。既存 Taxonomy/Presence/Manifest/評価/GPU 基盤に直結。

## PoC の再現（CPU のみ・完全オフライン）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-poc.txt
bash scripts/run_poc.sh
```

生成物: `data/synth_orchestra/`（正解付き合成オケ）、`results/metrics.json`・`results/diagnostic_pairs.json`（実測メトリクス）、`results/estimates*/`（分離例の音声）。

Phase 2 のデータ基盤・評価基盤の検証:

```bash
pip install pyyaml jsonschema
bash scripts/build_foundation.sh   # taxonomy検証 → manifest → presence → evaluation
```

Phase 3 の研究プラットフォーム（学習→推論→評価が一貫動作）:

```bash
pip install bs-roformer            # 実 Mel-Band RoFormer (MIT)
bash scripts/run_platform_poc.sh   # train → resume → infer → evaluate → RoFormer smoke
# モデル差し替えは configs/train.example.yaml の model.name を変更するだけ
```

## GPU 本番学習クイックスタート（Phase 5）

**GPU を用意したら、これだけで本番学習〜評価〜レポートが走ります。**

```bash
# 1) 環境（GPU箱）
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-platform.txt     # GPUは先に CUDA版 torch を導入

# 2) データ用意（自由ネットワーク環境で取得済みのものを変換）
python src/prepare_datasets.py --dataset synthsod --root /data/synthsod --out data/synthsod

# 3) GPU/VRAM/CUDA/Driver を確認し、適切な config を自動選択
bash scripts/gpu_check.sh

# 4) データ確認→学習→Validation→Checkpoint→Resume→推論→評価→レポート を一括
bash scripts/train_gpu.sh                                  # 自動選択（VRAMで tier 決定）
#   tier 指定 : bash scripts/train_gpu.sh configs/prod/roformer_24gb.yaml run24
#   マルチGPU : bash scripts/launch_distributed.sh 4 configs/prod/roformer_48gb.yaml run48

# 結果: runs/<id>/run.json + report.md（実験管理）, runs/<id>/infer/eval_*（評価）
```

GPU が無い環境でも同じコマンドで小規模 config が自動選択され、パイプライン全体を検証できます:
```bash
python src/run_pipeline.py --config configs/prod/roformer_small.yaml --name smoke
```

## 主要な発見（Phase 1）

- ファミリー単位（弦/木管/金管/打）の分離はオラクル上限で **+6〜+21 dB SI-SDRi** ＝原理的に可能。
- **同一音色＋同一音高（第1/第2ヴァイオリンのユニゾン重複）は理想オラクルでも +0.26 dB ＝分離不可能**。→ モノ音源＋マスク型の原理的限界であり、空間情報 or 楽譜情報が必須。
- 推論は GPU 不要（HT-Demucs 83.6M を CPU で RTF<1 実測）。GPU が要るのは学習。

詳細・環境・学習/推論手順・データ入手は `environment/README.md` を参照。

## ライセンス / 依存

採用候補モデル（Demucs, Mel-Band RoFormer 実装）はいずれも MIT。詳細は各レポート参照。
