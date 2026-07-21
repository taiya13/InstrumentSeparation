# Phase 2 設計書 — オーケストラ・データ基盤／学習仕様

**目的**: AI の改良はまだ行わない。**今後の学習・評価の土台（データ基盤・楽器辞書・評価基盤）を設計**し、Phase 3 で「モデルを改良できる状態」を作る。
**日付**: 2026-07 / **ステータス**: 設計完了・基盤コード動作確認済み
**方針**: 設計を「読む文書」だけでなく **機械可読な設定＋動くコード**として提出し、Phase 3 が直接消費できる形にした。すべて Phase 1 の合成ベンチ上で**実際に動作確認済み**。

---

## 0. エグゼクティブサマリー

- **拡張可能な楽器辞書（Instrument Taxonomy）**を設計・実装。7カテゴリ / 12ファミリー / 41楽器 / パート（第1・第2Vn）を定義し、**category ▷ family ▷ instrument ▷ part の4階層**で任意の粒度に集約可能。**同一音色グループ（g.violins）を明示フラグ化**（Phase 1 で実証した「モノでは分離不可能」を辞書レベルで表現）。→ `configs/taxonomy.yaml`（バリデーション OK）。
- **曲ごとの楽器検出（presence）仕様**を設計。「毎回全楽器を出す」のではなく、**曲に存在する楽器のみ出力**する。presence 段（audio tagger）→ active set 選択（階層フォールバック付き）→ separator を条件付け、というインターフェースを定義し参照実装を用意。
- **データ基盤**をマニフェスト駆動で設計（`schemas/manifest.schema.json`）。曲＝1マニフェスト。**合成＝学習の主軸**（ブリードなし・presence 既知）、**実録＝検証/テスト**（ドメイン差の計測）と役割分担。
- **評価基盤**を実装。**SI-SDR / SDR / SIR / SAR** を、**楽器別・カテゴリ別・曲別**に集計し、**オラクル天井**と**同一楽器の取り違え（identity confusion）**も同時に出す。→ `src/evaluation.py`（Phase 1 データで実行確認済み）。
- **成果物はすべて動く**: `make_manifest → presence → evaluation` をエンドツーエンドで実行し、実数値を出力（本書 §7）。

---

## 1. Instrument Taxonomy 仕様書

**ファイル**: `configs/taxonomy.yaml`（正）、`src/taxonomy.py`（ローダ/検証/クエリ API）
**規模**: category 7 / family 12 / instrument 41 / part 2（合計 62 ノード）。検証: **VALIDATION OK**。

### 1.1 構造（4階層）

```
category (弦/木管/金管/打/鍵盤/声楽/その他)
  └ family (bowed, plucked, flutes, double_reeds, brass, pitched, ...)
       └ instrument (violin, oboe, horn, timpani, ...)   ← 既定の出力粒度
            └ part (Violin I / Violin II, ...)             ← 追加情報がある時のみ分離
```

- **ID 規約**: ドット区切りの**安定 ID**（例 `str.vln`=ヴァイオリン、`str.vln.1`=第1ヴァイオリン）。**既存 ID は再利用・付番変更しない**＝辞書を拡張しても後方互換。
- **カテゴリ**: Strings / Woodwinds / Brass / Percussion / Keyboard / Voices / **Other（拡張スロット）**。要望どおり、将来の楽器追加は該当ファミリーに1行追記するだけ。
- **メタ属性**: `midi_range`（音域）、`gm_programs`（GM 音色番号）、`typical_pan`（配置）、`same_timbre_group`。

### 1.2 多粒度学習・評価のサポート
`rollup(id, level)` で任意ノードを上位へ集約。例: `str.vln.1 → (instrument) str.vln → (category) str`。
→ **同じデータで「4カテゴリ分離」「41楽器分離」「パート分離」を切替可能**。Phase 3 はまず category、次に instrument、最後に part、と段階的に難度を上げられる。

### 1.3 同一楽器問題を辞書で明示
```yaml
same_timbre_groups:
  - id: g.violins
    members: [str.vln.1, str.vln.2]
    requires: [multichannel_or_spatial, score_informed]
```
Phase 1 で「同音色ユニゾンはオラクルでも +0.26 dB＝分離不能」を実証済み。**辞書がこれを機械可読に持つ**ことで、学習・評価コードが「このグループは追加情報（空間/楽譜）が必須」と判断できる。

### 1.4 異種データセットの統一
`mappings` で GM program / SynthSOD 4系統 / URMP コード / MUSDB / 本合成ベンチのラベルを**共通 ID に正規化**。→ 複数コーパスを1つの学習集合にプールできる。
（検証済み例: `urmp:vn → str.vln`、`gm_program:40 → str.vln`、`synthsod_family:brass → br`。）

---

## 2. 曲ごとの楽器検出（presence）仕様

**ファイル**: `src/presence.py`
**課題**: 曲ごとに登場楽器が違う。全楽器を常時出力すると、存在しない楽器に無音/雑音が出て評価も破綻する。

### 2.1 インターフェース（契約）
```
PresenceDetector.predict(audio|manifest) -> {taxonomy_id: prob∈[0,1]}
select_active_set(scores, taxonomy, threshold, fallback_level) -> [出力する id ...]
separator は active set のみを対象に分離（query/条件付けと接続）
```

### 2.2 実装
- **OraclePresence**: マニフェストの正解から presence を返す。**合成データ学習時の presence ヘッド教師**、および評価時の上限として使用。
- **EnergyHeuristicPresence（プレースホルダ）**: 音域バンドのエネルギー比で presence を近似。**インターフェース実証用**であり、Phase 3 では **学習済み audio tagger（PANNs / CLAP / 楽器認識ヘッド）**に置換する。
- **階層フォールバック**: instrument 単位で自信が無いが category 単位でエネルギーがある場合、**category ノードを出力**（`policy.fallback_level`）。同一音色グループは part を出さず instrument に丸める、という運用も辞書フラグで判断可能。

### 2.3 出力契約と評価
- モデルは **active set の楽器のみ stem を出力**（非存在は出さない）。
- 評価は **マニフェストの present_ids のみ**を対象にし、**取りこぼし（false negative）**と**幻の出力（false positive）**を presence の P/R/F1 で測る（§5）。

---

## 3. データ基盤設計書

**ファイル**: `schemas/manifest.schema.json`（正）、`src/make_manifest.py`（生成）

### 3.1 マニフェスト（曲＝1エントリ）
学習・presence・評価が共有する**単一の真実**。主要フィールド:
| フィールド | 意味 |
|---|---|
| `song_id / dataset / split` | 一意 ID・コーパス・train/valid/test |
| `source_type` | synthetic / real |
| `mixture` / `mixture_stereo` | モノ／ステレオ混合 |
| `score` | 整合済み MIDI/MusicXML（あれば score-informed 分離を有効化） |
| `present_ids` | この曲に存在する楽器 id（＝presence 正解） |
| `sources[]` | 各音源: `taxonomy_id, stem(ブリードなし), level, gain, pan, n_players` |
| `provenance` | renderer / midi_source / reverb_ir / augmentations（再現性・ドメイン追跡） |

- **スキーマ検証必須**（`jsonschema`）。Phase 1 ベンチから生成したマニフェストは**検証 OK**（13音源）。
- Phase 3 のレンダリング・パイプラインは、合成でも実録でも**この同一スキーマ**を出力する。

### 3.2 分割ポリシー
- **曲単位分割**（同一曲・同一演奏を複数 split に跨がせない＝リーク防止）。
- **楽器層化**（各 split の楽器カバレッジを保つ）。
- **クロスドメイン・テスト**（合成で学習→実録でテスト）で**ドメイン差を明示計測**。

---

## 4. 学習データ生成計画

**ゴール**: ブリードなしの多様なオーケストラ・マルチトラックを大量に得る。実録が希少なので**合成を主軸**に、**実録へ適応**する。

### 4.1 生成パイプライン
```
(1) 記号ソース   : MIDI / MusicXML  (SOD, Lakh, Bach chorales, 自作スコア)
(2) レンダリング : 高品質サンプル音源で楽器を個別レンダ
                   - 商用: Spitfire BBC SO (SynthSOD 準拠)
                   - オープン: sfizz + GM/SGM/VS soundfont, MIDI-DDSP(CocoChorales流)
(3) 個別ステム   : 楽器ごとに分離レンダ → 正解ステム＋presence を自動生成
(4) リアリズム拡張:
      - 畳み込みリバーブ(実ホール IR)
      - セクション配置/パンニング  ← 同一楽器分離のための空間手がかりを付与
      - 強弱/テンポ/表情のゆらぎ、奏者数(divisi)チューニング
      - マイク・ブリード模擬 / コーデック・ノイズ拡張
(5) ミキシング   : 現実的バランスで合算 → モノ＋ステレオ(＋任意で多ch)
(6) ドメイン適応 : (a)実録の音響に合わせた拡張 (b)実録少量でfine-tune
                   (c)無ラベル実録での自己教師/mixture-invariant training
                   (d)音源ライブラリのランダム化(1音源への過学習回避)
(7) マニフェスト : 各曲を schemas/manifest.schema.json で出力
```

### 4.2 データ拡張（学習時オンザフライ）
- **ランダム・リミックス**（曲を跨いでステムを再合成）で組合せ爆発を活用（MSS の定石）。
- ゲイン/ピッチ/タイム・ストレッチ、残響・イコライズ、チャンネル入替。
- **空間拡張**（同一楽器を別パンで配置）→ Phase 4 の空間手がかり学習の土台。

### 4.3 ドメイン適応（合成→実録：実用化の生命線）
1. 音響マッチング拡張（実ホール IR・実マイク特性）。
2. 実録少量（URMP 等）での fine-tune。
3. 無ラベル実録での自己教師（mixture-invariant / teacher-student）。
4. 音源ライブラリの多様化で timbre の過学習を防ぐ。

---

## 5. 評価基盤設計書

**ファイル**: `src/evaluation.py`（実装・実行確認済み）、`configs/eval_config.example.yaml`

### 5.1 指標
| 指標 | 用途 |
|---|---|
| **SI-SDR / SI-SDRi** | 主指標（スケール不変・O(n)・全楽器で高速） |
| **SDR / SIR / SAR** | 古典 SiSEC 指標（`mir_eval.bss_eval_sources`）。干渉(SIR)・雑音(SAR)を分離評価 |
| **オラクル天井(IRM)** | 各音源の**到達可能上限**を併記＝ヘッドルームが見える（Phase 1 の思想を標準化） |
| **presence P/R/F1** | 楽器検出の正しさ（false positive/negative） |
| **identity confusion** | 同一楽器グループで**取り違え**（swap が identity を上回る）を検出 |

### 5.2 比較軸（要望どおり）
- **楽器別**: 各 `taxonomy_id` の SI-SDR / SI-SDRi / 天井。
- **カテゴリ別**: rollup で弦/木管/金管/打に集約、mean・median。
- **曲別**: `eval_<song_id>.json/.md` を曲ごとに出力。
- **データセット別・モデル別**: `eval_config.yaml` のバッチ集計（Phase 3）でリーダーボード化。
- **統計**: mean＋**median**（MUSDB/SiSEC 慣行）。

### 5.3 同一楽器の扱い
same-timbre グループでは**置換不変（PIT）**で最良割当を採り、**swap が勝てば identity_confusion=true**として別途報告。→「分離できたが第1/第2を取り違えた」を可視化。

### 5.4 変動する出力集合の扱い
評価はマニフェストの present_ids のみを対象。存在しない楽器への出力は presence 側の false positive として計上（分離指標とは分離して評価）。

---

## 6. データセット整理表

**ファイル**: `configs/datasets.yaml`（機械可読・正）

| データ | 種別 | 役割 | 規模 | 辞書粒度 | このサンドボックス | 備考 |
|---|---|---|---|---|---|---|
| synth_bench(本repo) | 合成 | 診断/単体テスト | ~0.003h | instrument/part | ✅ local | パイプライン検証用 |
| **SynthSOD** | 合成 | **train/valid** | 47h | category/instrument | ❌ Zenodo 403 | 現状最良のオケ学習源 |
| EnsembleSet | 合成 | train/valid | ~80h | instrument | ❌ | 室内アンサンブル |
| **CocoChorales** | 合成 | **train** | 1400h | instrument | ❌ | 大規模＋整合MIDI(score) |
| Slakh2100 | 合成 | train(事前) | 145h | category/instrument | ❌ | 広い楽器カバレッジ |
| **URMP** | 実録 | **valid/test/DA** | 1.3h | instrument | ？ | 実録の定番ベンチ |
| Aalto anechoic | 実録 | test | 0.3h | category | ？ | 無響・family 評価 |
| **Cadenza CAD2** | 実録 | valid/test | ~1h | instrument/part | ？ | **目標に最も近い**（同一楽器含む） |
| Spheres | 実録 | test | - | instrument | ？ | 新規オケ multitrack |
| Bach10 | 実録 | test | 0.1h | instrument | ？ | 4声実録 |
| MUSDB18-HQ | 実録 | sanity | 10h | category | ❌ Zenodo 403 | 既知SDR再現で整合性確認 |

**役割整理**:
- **合成 = 学習の主軸**（ブリードなし・presence 既知・大量）。
- **実録 = 検証/テスト**（ドメイン現実性、希少なので温存。少量のみ DA fine-tune へ）。
- **MUSDB = 健全性チェック**（非オケだが、既知の公開 SDR を再現できるかでパイプラインの正しさを担保）。

> ⚠️ このサンドボックスでは Zenodo 等がブロックのため、実データ取得は**GPU＋自由ネットワークの環境**で行う（`environment/README.md`）。

---

## 7. 基盤の動作確認（実行済み・実数値）

`make_manifest → presence → evaluation` をエンドツーエンドで実行した実結果:

- **マニフェスト生成**: 13音源、スキーマ検証 **OK**。
- **presence（プレースホルダ heuristic）**: recall 1.0 / precision 0.34（過検出）→ **インターフェースと指標が機能**（真の精度は Phase 3 の学習済み tagger で担保）。
- **評価（demo-oracle）**: 楽器別 SI-SDR、カテゴリ別集計、`mir_eval` の SDR/SIR/SAR を出力。例（カテゴリ別 SDR/SIR/SAR @ オラクル）:

| category | SDR | SIR | SAR |
|---|---|---|---|
| woodwinds | 5.66 | 8.30 | 9.67 |
| brass | 5.58 | 9.43 | 8.36 |
| strings | 4.89 | 6.94 | 9.92 |
| percussion | 2.27 | 9.38 | 3.69 |

→ **評価基盤が SI-SDR/SDR/SIR/SAR を楽器別・カテゴリ別・曲別に出す**ことを実証。出力: `results/eval_synth_bench_0001.{json,md}`。

---

## 8. 次フェーズ（Phase 3）で実装する内容一覧

「不足しているデータ・実装・環境」を洗い出した。

### 8.1 データ（取得・生成）
- [ ] SynthSOD / EnsembleSet / CocoChorales / Slakh の取得（自由ネットワーク環境）。
- [ ] URMP / Aalto / Cadenza / Spheres / Bach10 の取得（実録評価用）。
- [ ] **独自レンダリング・パイプライン**（§4）: MIDI→個別ステム→拡張→ミックス→マニフェスト。
- [ ] 実ホール IR コレクション、実マイク特性プロファイル。
- [ ] 各コーパス→マニフェスト変換アダプタ（`mappings` を使用）。

### 8.2 実装（コード）
- [ ] **バックボーン統合**: Mel-Band RoFormer（ZFTurbo 枠組）を本辞書の出力粒度に接続。
- [ ] **presence ヘッド / audio tagger**（PANNs/CLAP）を `PresenceDetector` 実装として差し込み。
- [ ] **条件付け機構**: query（Banquet 型）／ score-informed（整合 MIDI）で active set を指定。
- [ ] **同一楽器分離**（Phase 4）: マルチチャンネル/空間 or 楽譜条件の入力経路。
- [ ] 学習ループ（オンザフライ・リミックス拡張、ドメイン適応）。
- [ ] `evaluation.py` の**バッチ/リーダーボード**モード（`eval_config.yaml` 駆動、データセット別・モデル別集計）。
- [ ] マニフェスト・バリデータの CI 化。

### 8.3 環境
- [ ] **GPU**（学習。VRAM 24GB 目安）。推論は CPU 可（Phase 1 実測）。
- [ ] 自由ネットワーク（重み＋データ CDN 到達）。
- [ ] 大容量ストレージ（CocoChorales 1400h 等）。
- [ ] 実験管理（W&B 等）とデータ・バージョニング。

---

## 9. 成果物一覧（本 Phase）

```
docs/Phase2_データ基盤_学習仕様設計.md   … 本設計書（6成果物を統合）
configs/taxonomy.yaml                     … Instrument Taxonomy 仕様（機械可読・検証OK）
configs/datasets.yaml                     … データセット整理表（機械可読）
configs/eval_config.example.yaml          … 評価バッチ設定例
schemas/manifest.schema.json              … データ基盤マニフェスト・スキーマ
src/taxonomy.py                           … 辞書ローダ/検証/rollup/mapping API
src/presence.py                           … 曲別楽器検出 仕様＋参照実装
src/make_manifest.py                      … 曲→マニフェスト生成（スキーマ検証）
src/evaluation.py                         … 楽器別/カテゴリ別/曲別 SI-SDR/SDR/SIR/SAR 評価
results/eval_synth_bench_0001.{json,md}   … 評価基盤の実行結果
```

## 10. 結論

- Phase 2 で **「拡張可能な楽器辞書」「曲別 presence 仕様」「マニフェスト駆動データ基盤」「楽器別/カテゴリ別/曲別の評価基盤」**を設計し、**すべて動くコード＋機械可読設定**として提出、Phase 1 データ上でエンドツーエンド実行を確認した。
- これにより **Phase 3 は「モデルを差し込めば学習・評価できる」状態**になった。まずは **category 分離**から始め、辞書の rollup で instrument → part へ段階的に難度を上げる設計。
- 次は **Phase 3: モデル改良（Mel-Band RoFormer ＋ presence 条件付け）**へ。最初の一歩は、自由ネットワーク環境で SynthSOD を取得し、本基盤に流し込んで**category 分離のベースライン学習**を回すこと。
