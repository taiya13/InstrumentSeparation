# 研究用分離ツール（GUI / CLI）— 使い方・テスト手順

RTX 5060 Ti（15GB）上で「**自分で音源を読み込み → 実際に分離 → 結果を何度も評価**」する研究ループを回すためのツール。**見た目より研究の反復速度を優先**した設計。

---

## 0. 設計方針（重要）

- **GUI（View）と 推論エンジン（Core）を分離**：`src/app/gui.py`（Tkinter）は薄く、実処理はすべて `src/app/engine.py`。
- **モデル非依存**：モデルは「**チェックポイントのパス**」でしかない。backbone 種別・アーキ・出力楽器・sr はチェックポイントのメタ情報から復元。→ **モデルを差し替えても GUI は一切変更不要**（`configs/app_default.yaml` を編集するだけ）。
- **既存基盤をそのまま利用**：Backbone Interface（`separate`/`load_checkpoint`）、Taxonomy（楽器名・粒度）、Presence、評価基盤（SI-SDR/SDR/SIR/SAR）、GPU 判定（`mss.gpu`）。
- **ヘッドレスでも動く**：GUI と同じエンジンを CLI（`src/app/cli.py`）から実行可能（テスト・バッチ研究・GUI 無し環境用）。

構成:
```
src/app/engine.py       … 推論エンジン（音源読込→分離→保存→計測→評価→ログ）
src/app/gui.py          … Tkinter GUI（Windows対応・薄い）
src/app/cli.py          … ヘッドレス実行（同一エンジン）
src/app/gpu_monitor.py  … GPU使用率/VRAM 計測（pynvml→nvidia-smi→torch フォールバック）
src/app/research_log.py … 研究ログ（logs/research_log.jsonl + 実行毎 json）
configs/app_default.yaml… モデル一覧・出力・presence・ログ設定
scripts/run_gui.bat     … Windows 起動
```

---

## 1. GUI の機能（要件対応）

| 要件 | 対応 |
|---|---|
| Windows 対応 | Tkinter（Python 標準・python.org 版に同梱） |
| 音楽ファイル選択（WAV/FLAC/MP3） | ファイル選択ダイアログ（soundfile が 3 形式対応） |
| 「分離開始」ボタン | あり（別スレッド実行で UI 応答維持） |
| 進捗表示 | プログレスバー＋ステータス（モデル読込→読込→分離→保存→評価→ログ） |
| 出力フォルダ選択 | あり |
| 楽器ごとに自動保存 | `<出力>/<入力名>/<taxonomy_id>.wav` ＋ `instruments.json`（日本語名対応） |
| 推論時間 | 表示（RTF も） |
| GPU 使用率 | 表示（分離中サンプリングの avg/peak） |
| VRAM 使用量 | 表示（peak / total MB） |
| 使用モデル | 表示（display 名・backbone・trained・params） |
| 分離楽器一覧 | 表（id / 日本語名 / ファイル / SI-SDR） |
| 研究ログ | `logs/research_log.jsonl`＋実行毎 json（設定・モデル・評価・エラー・日時） |

---

## 2. Windows（RTX 5060 Ti）セットアップ

> ⚠️ **RTX 5060 Ti は Blackwell 世代（sm_120）**。**CUDA 12.8 対応の PyTorch（cu128）以降**が必要です（古い torch は起動しません）。

```bat
:: 1) Python 3.11 (python.org 版。tkinter 同梱にチェック)
python -m venv .venv
.venv\Scripts\activate

:: 2) PyTorch（Blackwell 対応 = cu128 以降）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

:: 3) プロジェクト依存＋ツール依存
pip install -r environment\requirements-app.txt

:: 4) GPU が見えるか確認
python src\app\cli.py --list-models
::   -> device: cuda | GPU: NVIDIA GeForce RTX 5060 Ti 15.0 と出れば OK
```

---

## 3. モデルの用意（分離するための重み）

このツールは**チェックポイントを読み込んで分離**します。手順:

1. **学習**（Phase 5 パイプライン）: 例）SynthSOD を用意し
   ```bat
   python src\run_pipeline.py --config configs\prod\roformer_15gb.yaml --name orch15
   ```
   → `checkpoints\prod_15gb\final.pt` が生成。
2. **ツールに登録**: `configs\app_default.yaml` の該当エントリの `checkpoint` を上記パスに（既定で `checkpoints/prod_15gb/final.pt` を指しています）。
3. 学習前の**パイプライン検証**だけなら、`tiny_masker` / `roformer_untrained`（チェックポイント不要）で GUI 動作を先に確認できます（**分離品質は無意味**、動作確認用）。

> **モデル追加/差し替え** = `configs\app_default.yaml` に 1 エントリ足す（または checkpoint を置く）だけ。**GUI/エンジンのコード変更は不要**。

---

## 4. 使い方

### GUI
```bat
scripts\run_gui.bat
```
1. 「音源ファイル」を選択（WAV/FLAC/MP3）
2. 「出力フォルダ」を選択
3. 「モデル」を選択（一覧は config から自動生成）
4. （任意）「参照 manifest.json」を指定すると分離後に **SI-SDR/SDR/SIR/SAR** を自動評価
5. **「分離開始」** → 進捗表示 → 完了後に推論時間・GPU使用率・VRAM・使用モデル・楽器一覧を表示

### CLI（同一エンジン・バッチ/自動化向き）
```bat
python src\app\cli.py --input song.flac --output out --model roformer_15gb
python src\app\cli.py --input mix.wav --output out --model tiny_masker ^
    --reference data\synth_orchestra\manifest.json    :: 評価も実行
python src\app\cli.py --list-models
```

### 出力物
```
out\<入力名>\
   str.wav, ww.wav, ...          … 楽器（taxonomy id）別
   instruments.json              … id → 日本語/英語名・ファイル
   run_<日時>.json               … この実行の全記録
   eval_<song>.json/.md          … 評価（--reference 指定時）
logs\research_log.jsonl          … 全実行の追記ログ（研究用）
```

---

## 5. テスト手順（動作確認のやり方）

### (A) まず CPU / ヘッドレスで配管を検証（GPU 不要・数秒）
```bash
python src/synth_orchestra.py            # 正解付きテスト音源を生成
python src/make_manifest.py              # 評価用 manifest
python src/app/cli.py --input data/synth_orchestra/mixture.wav --output out \
    --model tiny_masker --reference data/synth_orchestra/manifest.json
```
確認項目:
- 進捗が 0→100% で流れる
- `out/mixture/` に `str.wav/ww.wav/br.wav/perc.wav` と `instruments.json`
- 推論時間・(CPUなので GPU は src=none)・楽器一覧（弦/木管/金管/打）が表示
- `--reference` 指定で **SI-SDR/SDR/SIR/SAR** が表示・保存
- `logs/research_log.jsonl` に 1 行追記（設定/モデル/評価/エラー/日時）

> ※ `tiny_masker` は未学習なので SI-SDR ≈ 0（**配管の検証が目的**。品質は学習済みモデルで評価）。

### (B) RTX 5060 Ti で実機検証
1. `python src\app\cli.py --list-models` → `device: cuda` と GPU 名・VRAM を確認。
2. 学習済み checkpoint を用意（§3）。
3. `scripts\run_gui.bat` で音源を選び「分離開始」。
4. **別ターミナルで `nvidia-smi -l 1`** を回し、分離中に本プロセスが GPU を使うことを確認。ツール表示の GPU使用率/VRAM と概ね一致するはず。
5. 出力 stem を試聴し、`--reference`（正解ありの合成/URMP 等）で SI-SDR を確認。
6. **研究ループ**: モデル/設定を変えて再実行 → `logs/research_log.jsonl` を横断比較（`grep`/`jq`/pandas）。

### (C) GPU 計測が出ない場合
- `pip install nvidia-ml-py`（pynvml）で使用率が取得可能に。未導入でも `nvidia-smi` → `torch` にフォールバック（VRAM のみ等）。

---

## 6. よくある落とし穴
- **torch が起動しない/CUDA error**: Blackwell は **cu128 以降**が必須（§2）。
- **MP3 が読めない**: soundfile（libsndfile 1.1+）同梱で通常読めます。古い環境なら WAV/FLAC を使用、または `pip install -U soundfile`。
- **モデルが `[未整備]`**: checkpoint 未生成。§3 で学習するか、`tiny_masker` で配管確認。
- **GUI が出ない（ヘッドレス）**: サーバ等ディスプレイ無し環境では CLI を使用。

---

## 7. 将来のモデル差し替え（生成モデル等）
Architecture Design 001 の生成分離モデルも、**Backbone Interface に載せて checkpoint を出力**すれば、`configs/app_default.yaml` に 1 行追加するだけで本ツールから利用可能（GUI 変更不要）。
