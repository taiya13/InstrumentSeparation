# InstrumentSeparation

オーケストラ音源を 15〜20 種類程度の楽器単位へ分離することを目標とする研究プロジェクト。

## 現在の進捗

- **Phase 0（調査）** … `docs/Phase0_技術調査レポート.md`
  既存の音源分離AI・オーケストラ特化研究・データセットの徹底調査、比較表、推奨ベースモデル、ロードマップ、実現可能性評価。
- **Phase 1（再現・評価 PoC）** … `docs/Phase1_再現評価レポート.md`
  ベースモデル（HT-Demucs）選定、再現可能な環境構築、CPU 動作確認、**オラクル上限による到達可能品質の実測**、**同一楽器分離が原理的に不可能であることの定量実証**、ボトルネック分析。
- **Phase 2（データ基盤・学習仕様設計）** … `docs/Phase2_データ基盤_学習仕様設計.md`
  拡張可能な **Instrument Taxonomy**（`configs/taxonomy.yaml`）、曲別 **presence 検出**仕様、**マニフェスト駆動データ基盤**（`schemas/manifest.schema.json`）、**楽器別/カテゴリ別/曲別の評価基盤**（`src/evaluation.py`）、データセット整理表、Phase 3 実装一覧。すべて動作確認済み。

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

## 主要な発見（Phase 1）

- ファミリー単位（弦/木管/金管/打）の分離はオラクル上限で **+6〜+21 dB SI-SDRi** ＝原理的に可能。
- **同一音色＋同一音高（第1/第2ヴァイオリンのユニゾン重複）は理想オラクルでも +0.26 dB ＝分離不可能**。→ モノ音源＋マスク型の原理的限界であり、空間情報 or 楽譜情報が必須。
- 推論は GPU 不要（HT-Demucs 83.6M を CPU で RTF<1 実測）。GPU が要るのは学習。

詳細・環境・学習/推論手順・データ入手は `environment/README.md` を参照。

## ライセンス / 依存

採用候補モデル（Demucs, Mel-Band RoFormer 実装）はいずれも MIT。詳細は各レポート参照。
