# Phase 1 環境構築ガイド（再現可能な学習・推論環境）

本ドキュメントは Phase 1「既存ベースモデルの再現・評価」の環境を、**この開発サンドボックスで実際に動く部分**と、**GPU＋自由なネットワークが必要な部分**に分けて記載する。

---

## 0. この環境の重要な制約（先に把握すること）

本プロジェクトのサンドボックスの egress ポリシーは、**学習済みモデルの配布 CDN を軒並みブロック**している。実測で確認済み:

| ホスト | 用途 | このサンドボックス |
|---|---|---|
| pypi.org / files.pythonhosted.org | pip パッケージ | ✅ 到達可 |
| github.com | コード | ✅ 到達可 |
| dl.fbaipublicfiles.com | **Demucs 重み** | ❌ 403 ブロック |
| download.pytorch.org | **torchaudio / torch 重み** | ❌ 403 ブロック |
| huggingface.co | **RoFormer 等の重み** | ❌ 403 ブロック |
| zenodo.org | **SynthSOD 等データセット** | ❌ 403 ブロック |

**帰結**: このサンドボックスでは学習済みの Demucs/RoFormer を**ダウンロードできない**（＝そのまま実分離を回せない）。
本 Phase の「実際に動かして評価する」部分は、**重み不要の手法（オラクル・マスク上限、HPSS、合成テストベンチ）**で実行した。
**本物の SOTA（HT-Demucs / RoFormer）の再現は、GPU と自由なネットワークを持つチームのマシンで**、本ディレクトリのレシピに従って行う。

> なお **推論そのものは CPU で十分動く**ことを実測で確認（HDemucs 83.6M params, RTF≈0.16x, RAM 1.5GB）。
> GPU が本質的に必要なのは **学習**と、実運用の**バッチ高速化**。

---

## 1. Python 環境

- Python 3.11（サンドボックス実測: 3.11.15）
- 仮想環境を推奨:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
```

## 2. PoC 用ライブラリ（このサンドボックスで実行済み・完全オフライン動作）

```bash
pip install -r environment/requirements-poc.txt
```

これで以下が動く（`scripts/run_poc.sh`）:
- 合成オーケストラ・テストベンチ生成（`src/synth_orchestra.py`）
- オラクル分離上限＋HPSS ベースライン評価（`src/oracle_and_baselines.py`）
- 同一楽器 2×2 診断実験（`src/diagnostic_pairs.py`）
- HDemucs アーキテクチャの CPU 動作確認（`src/hdemucs_smoketest.py`）

## 3. GPU 環境（本物の SOTA を再現する場合）

- 推奨: NVIDIA GPU (VRAM 12GB 以上、学習なら 24GB 以上が快適)。CUDA 12.x。
- CPU-only の PoC には GPU 不要（本リポジトリの実行結果は全て CPU）。
- CPU-only の torch を入れる場合（重みは別途）:
  ```bash
  pip install torch torchaudio            # pypi 既定（CUDA ビルド。CPU でも動く）
  # もしくは公式 CPU ホイール（要 download.pytorch.org 到達）:
  # pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

## 4. ベースモデルの推論方法（HT-Demucs, Phase 1 採用モデル）

自由ネットワークのマシンで:

```bash
pip install -r environment/requirements-sota.txt

# 4/6 ステム分離（重みは初回自動 DL: dl.fbaipublicfiles.com）
demucs -n htdemucs        path/to/orchestra.wav       # drums/bass/other/vocals
demucs -n htdemucs_6s     path/to/orchestra.wav       # +guitar +piano (6 stems)

# 出力: separated/<model>/<track>/{vocals,drums,bass,other[,guitar,piano]}.wav
```

オーケストラ音源に対する客観評価（正解ステムがある合成/URMP 等）:

```bash
python src/run_demucs_reference.py --input path/to/mixture.wav \
       --refs path/to/refs_dir --model htdemucs_6s
```

## 5. 学習方法

### 5-A. HT-Demucs の学習/微調整（Demucs 公式）
```bash
git clone https://github.com/facebookresearch/demucs
cd demucs && pip install -e .
# データセットを Demucs の形式（train/valid, 各トラックにステム）で用意し dora で起動
dora run -d          # 設定は conf/ 参照。GPU 必須。
```

### 5-B. Mel-Band RoFormer の学習（Phase 2+ の本命バックボーン）
```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training
cd Music-Source-Separation-Training && pip install -r requirements.txt

# 例: mel_band_roformer を独自ステム定義で学習
python train.py \
  --model_type mel_band_roformer \
  --config_path configs/config_mel_band_roformer_<your>.yaml \
  --data_path /data/train --valid_path /data/valid \
  --results_path ./results --device_ids 0
```
- config の `training.instruments` に**オーケストラ用ステム定義**（弦/木管/… または楽器名）を書く。
- 学習データは**ブリードのないマルチトラック**が必要 → Phase 1 の合成ベンチ、SynthSOD、EnsembleSet、CocoChorales を利用。

## 6. データセットの入手（ネットワーク制限のない環境で）

| データ | 入手先 | 用途 |
|---|---|---|
| 合成オケ（本リポジトリ） | `python src/synth_orchestra.py` | 制御された正解付きベンチ |
| SynthSOD (47h) | Zenodo 13759492 | オケ学習 |
| EnsembleSet | ISMIR2022 配布 | 室内アンサンブル学習 |
| CocoChorales (1400h) | Magenta | 4声・13楽器 |
| URMP | 公式サイト | 実録評価 |
| Aalto anechoic orchestral | 公式 | 実録評価 |
| MUSDB18-HQ | Zenodo | ベンチ整合性確認 |

> これらのホスト（Zenodo 等）は本サンドボックスではブロックされている。チームのマシンで取得すること。
