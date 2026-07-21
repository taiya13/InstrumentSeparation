# Architecture Design 001 — Score-informed Generative Source Separation

**位置づけ**: 実装は行わない。研究方針「**2ch ステレオ・生成的オーケストラ楽器分離**」を実現するための**システム全体アーキテクチャを設計・決定**する。
**前提**: Exp 001-003（マスキングの壁＝ユニゾン/同一 TF 占有）、研究戦略（P1 Score-informed 生成 ＋ P2 生成的リファインメント ＋ P3 決定論フロントエンド）、既存基盤（Taxonomy / Presence / Manifest / 評価 = Phase 2-5）。
**日付**: 2026-07 / **ステータス**: 設計完了（採用アーキ決定）

---

## 0. エグゼクティブサマリー（採用結論）

**採用アーキテクチャ**: **潜在空間 Rectified Flow-Matching DiT による Score 条件付き・混合アンカー型（bridge）ターゲット抽出 ＋ RoFormer 決定論フロントエンド ＋ 推論時 consistency 蒸留**。

- **潜在空間**: 事前学習ステレオ音声 VAE（DAC/Stable-Audio 系）の連続潜在で生成 → 高忠実・低計算（波形/スペクトル直接生成より効率的）。
- **生成核**: **Rectified Flow Matching**（拡散より高速・安定・SOTA。FlowSep/SAM Audio の先行）。**混合潜在から源潜在へ橋渡し（mixture-anchored）**し、**ノイズからの自由生成を避けて忠実性を確保**。
- **条件付け**: ①**混合潜在**（強アンカー、concat＋cross-attn）②**Score**（音符列 → score encoder → cross-attention ＋ 倍音サリエンス map concat）③**target-id 埋め込み**（Taxonomy の楽器/パート id）。
- **難所対応**: **Taxonomy の `same_timbre_group`（Vn I/II）は「グループ同時生成＋sum=混合 拘束」モード**で分解 → Exp 001-003 で 0 dB だったユニゾン/同一セクションを、**Score 拘束下の生成**で越える（Separate-and-Diffuse の理論＝生成は決定論上限を超える）。
- **忠実性制御**: mixture-consistency 損失（Σ源=混合）＋ Score-fidelity（音高一致）損失で**音符の捏造を抑制**。
- **既存基盤に統合**: Backbone Interface の新 backbone として実装可能。Presence が active set、Manifest が (mix, stems, score) を供給、評価は Phase 2 ハーネス（SI-SDR/SDR/SIR/SAR）。

---

## 1. システム全体構成（データフロー）

```
                        ┌──────────────────────── 条件 (Conditioning) ─────────────────────────┐
                        │                                                                      │
 [Stereo Mix 2ch] ─► Audio VAE Encoder ─► z_mix (latent) ─┐                                    │
                        │                                  │                                   │
 [Score/MIDI] ─► Align (audio-to-score) ─► per-target      │      ┌─ Score Encoder ─┐          │
                pianoroll + note events ───────────────────┼─────►│ (small Transformer)│─► c_score (cross-attn)
                        │                                  │      └────────────────────┘        │
 [Target id] (Taxonomy) ─────────────────────────────────┼─────► id-embedding ──────────────► FiLM/token
                        │                                  │                                   │
 (任意) RoFormer 決定論分離 ─► 初期推定 stems ─► VAE Enc ─► z_hat_target (warm-start / 追加条件) │
                        ▼                                  ▼                                    │
             ┌─────────────────────────────────────────────────────────────────────────────────┐
             │  Rectified Flow-Matching DiT  (latent transformer over time-frames)               │
             │   base: z_mix(+z_hat)  ──(ODE, few steps / 1-2 with consistency)──►  z_target      │
             │   条件: z_mix(concat) , c_score(cross-attn) , target-id , t(timestep)              │
             └─────────────────────────────────────────────────────────────────────────────────┘
                        ▼
             z_target ─► mixture-consistency 投影 (Σz_i ≈ z_mix) ─► Audio VAE Decoder ─► [Stereo Stem 2ch]
                        │
        （same_timbre_group は member を同時生成し Σ=混合 拘束で分解）
                        ▼
             Phase 2 評価ハーネス: SI-SDR / SDR / SIR / SAR（合成）+ 無参照指標（実録）
```

---

## 2. 各コンポーネント設計

### 2.1 入力
- **Stereo Audio (2ch)**: 44.1kHz。市販/YouTube 音源をそのまま。mid/side も派生可（補助）。
- **Score**:
  - 現行: **整合済み MIDI**（audio-to-score alignment 済み）。表現＝各ターゲットの**ピアノロール**＋**音符イベント列**(pitch, onset, offset, velocity, instrument)。
  - 将来拡張: **自動採譜**（AMT）→ alignment → 同表現。Score 無い実録に対応。
  - **倍音サリエンス map**（Exp 001 で有効性実証）: 音符→基音+倍音を潜在時間格子に射影した [F/latent, T] 特徴。安価で強い prior。

### 2.2 エンコーダ / 潜在表現
- **Audio VAE（連続潜在）**: ステレオ対応の事前学習音声オートエンコーダ（DAC-VAE / Stable-Audio VAE / Music2Latent 系）を**オーケストラ音源で微調整**。
  - 例: 44.1kHz → 潜在フレーム ~21.5ms、次元 ~64-128、連続。
  - 生成は**潜在で実施**（波形/複素スペクトル直接より 1-2 桁効率的、かつ知覚品質を AE が担保）。
  - Flow 学習中は原則 **凍結**（latent diffusion の定石）。
- **Score Encoder**: 音符イベント列を小型 Transformer で埋め込み（pitch/timing/velocity/instrument を token 化）。cross-attention の key/value に。

### 2.3 生成モデル（核）
- **Rectified Flow Matching**（連続時間 ODE）を潜在で。
  - **混合アンカー（bridge）**: base 分布を**混合潜在 `z_mix`（＋任意で RoFormer 初期推定 `z_hat`）**に取り、`z_mix → z_target` の直線輸送を回帰学習。**ノイズからの自由生成でなく混合からの写像**とすることで**忠実性を確保・ハルシネーション抑制**。
  - バックボーン: **Diffusion/Flow Transformer (DiT)**（時間フレーム列に self-attn、条件に cross-attn、timestep 埋め込み）。SAM Audio（DiT+FM+DAC 潜在）が先行例。
- **推論高速化**: 学習後に **consistency 蒸留**で 1-2 step 化（製品/YouTube 規模の実用速度）。

### 2.4 条件入力の方法
| 条件 | 表現 | 注入方法 |
|---|---|---|
| 混合 | `z_mix`（潜在） | **入力に concat（強アンカー）** ＋ cross-attn |
| Score | 音符列 → score encoder ／ 倍音サリエンス map | **cross-attention**（イベント列）＋ **map を concat** |
| ターゲット指定 | Taxonomy の instrument/part id | **学習埋め込み**（FiLM もしくは条件トークン） |
| （任意）初期推定 | RoFormer 出力の潜在 `z_hat` | concat（warm-start） |
| guidance | 条件ドロップ（CFG） | classifier-free guidance で分離強度を調整 |

### 2.5 デコーダ / 出力形式
- **Audio VAE Decoder**: `z_target → ステレオ stem (2ch)`。
- **出力**: 楽器/パートごとの**ステレオ stem 群**（Presence の active set のみ、Taxonomy id 命名）。同一セクションはグループ同時生成で member 分解。
- **mixture-consistency 投影**: 生成後、Σstem ≈ 混合 となるよう残差再配分（Wiener 風／潜在射影）で整合。

### 2.6 損失関数
- **Flow-matching 損失**（主）: 速度場（straight-path のベクトル）回帰。
- **mixture-consistency 損失**: Σ_i z_i ≈ z_mix（潜在）／Σ stem ≈ 混合（波形）。**劣決定問題の拘束＋忠実性**。
- **Score-fidelity 損失**: 生成 stem の音高内容が条件 Score に一致（微分可能 f0/chroma もしくは事前学習採譜器）。**音符捏造を防ぐ**。
- **知覚再構成損失**（1-step 推定 or 一部 step で復号し）: multi-res STFT ＋ SI-SDR（音声域）。
- **VAE 損失**（AE 微調整段のみ）: 再構成 ＋ KL ＋ 逆説（adversarial）。

### 2.7 学習方法（段階）
1. **AE 微調整**: オーケストラ stem/混合でステレオ VAE を微調整。
2. **Flow 学習**: 合成データ（SynthSOD/EnsembleSet/CocoChorales、**Score は合成で既知**）で score encoder＋DiT を学習。Phase 2/4 の augmentation（残響・強弱・配置・リミックス）併用。CFG のため条件ドロップ。
3. **ドメイン適応**: 実録（URMP/Aalto/Cadenza）＋**無ラベル YouTube** で自己教師（mixture-consistency）／**score-only 条件は synthetic→real 汎化に有利**（2503.07352）。
4. **consistency 蒸留**: 推論 1-2 step 化。

### 2.8 推論パイプライン
```
1. 入力: stereo mix + score(整合MIDI; 将来AMT)
2. Presence(Phase2) → active target 集合（+ same_timbre_group ルーティング）
3. audio-to-score alignment → 各ターゲットの整合ピアノロール/イベント
4. (推奨) RoFormer 決定論分離 → 初期 stems（warm-start / 追加条件）
5. VAE Encode: mix(→z_mix), 初期 stems(→z_hat)
6. 各 active target（or 同一セクションは同時）で Flow ODE を数 step（蒸留時1-2）
   条件: z_mix, score, target-id, (z_hat)  → z_target
7. mixture-consistency 投影で整合
8. VAE Decode → stereo stem 群
9. Phase 2 評価（合成: SI-SDR/SDR/SIR/SAR、実録: 無参照/採譜整合/参照録音）
```

---

## 3. アーキテクチャ候補の比較

| 候補 | 概要 | 品質(特にユニゾン) | 2ch忠実性 | 計算量(学習/推論) | 実装難易度 | 成熟度 | 将来性 | ハルシネーション |
|---|---|---|---|---|---|---|---|---|
| **A. 潜在 Flow-Matching DiT（混合アンカー・per-target＋同一セクション同時, 採用）** | 潜在で FM、混合橋渡し、Score条件 | **高** | 高 | 中/**低(蒸留)** | 中〜高 | 中(FlowSep/SAM先行) | **高** | 低(アンカー+拘束) |
| B. 潜在 Diffusion 同時多源（MSDM/MSG-LD 型） | 全 stem を同時生成、Σ=混合 | 高 | 高 | 高/高 | 高 | 中 | 高 | 中 |
| C. 生成的リファインメントのみ（Sep-and-Diffuse/consistency） | RoFormer 出力を後段生成で精緻化 | 中(残差補正止まり) | 高 | 低/中 | **低** | 高 | 中 | 低 |
| D. 波形/スペクトル直接 Diffusion（AEなし） | 生波形/複素スペクトルで拡散 | 中〜高 | 中 | **高**/高 | 中 | 中 | 中 | 中 |
| E. 離散トークン AR（codec+LM, MusicGen 型） | codec token を Score 条件で自己回帰生成 | 中 | 中 | 高/**高** | 高 | 中 | 中 | **高**(離散再構成誤差) |
| F. Schrödinger Bridge / 確率補間（混合→源直写像） | 混合→源の確率的橋 | 高 | 高 | 中/中 | **高** | 低(新) | 高 | 低 |

**評価軸の要点**:
- **速度**: FM(A) ＞ Diffusion(B,D)。蒸留で A は推論 1-2 step。
- **忠実性/低ハルシネーション**: 混合アンカー(A) と Bridge(F) が有利。自由生成の AR(E) は危険。
- **実装容易性**: C が最易（既存 RoFormer に載る）。A は中〜高、F は最難（新規理論）。
- **ユニゾン克服**: 生成の力を使う A/B/F が本命。C は残差補正に留まり原理的壁は越えにくい。E は離散誤差で不利。

---

## 4. 採用アーキテクチャと理由

### 採用: **候補 A ＝ 潜在 Rectified Flow-Matching DiT（Score条件・混合アンカー・per-target＋同一セクション同時生成）＋ RoFormer 決定論フロントエンド ＋ consistency 蒸留**

### 採用理由（詳細）
1. **ユニゾン/同一音色の壁を越える唯一筋の通った方向**: Exp 001-003 でマスキング（Score/Position 問わず）は同一 TF 占有を分離不能と実証。**生成は決定論上限を超えられる（Separate-and-Diffuse, ICLR'24）**。A は生成核を持ち、**Score が生成を正しい音符列へ拘束**するため、同一セクションを「もっともらしい各声部」に分解できる。
2. **忠実性（捏造の回避）を設計で担保**: base を**混合潜在にアンカー（bridge）**し、**mixture-consistency ＋ Score-fidelity 損失**で拘束。自由生成でないため、分離器に必要な「混合に忠実」性を保つ。
3. **2ch 制約と完全整合**: 多ch/マイクアレイ不要。ステレオ VAE 潜在で 2ch のまま動作。Position 依存を排し、方針（多ch 対象外）に適合。
4. **効率と実用性**: 潜在空間＋Flow Matching で学習安定・高速、**consistency 蒸留で YouTube 規模の実用推論**。波形/スペクトル直接や AR より軽い。
5. **二段構えでリスク分散（戦略 P2+P3+P1 を包含）**: 大半の楽器は**RoFormer（決定論・実績）**で解き、生成核は**難所（同一セクション/残差）に集中**。C（リファインメント）の利点も front-end 結合として吸収。
6. **既存基盤に直結**: Backbone Interface の新 backbone として実装可。**Taxonomy が target-id と same_timbre_group ルーティング**、**Presence が active set**、**Manifest が (mix, stems, score)**、**評価ハーネス**がそのまま使える。→ Phase 2-5 の投資を回収。
7. **将来拡張性**: Score→自動採譜（AMT）差し替え、per-target→同時多源（B）への発展、Bridge（F）への一般化、text/audio クエリ（GuideSep 型）併用まで拡張余地。

### 不採用理由（要点）
- **B（同時多源拡散）**: 音楽的整合は魅力だが**固定 N・高計算・実装難**。A の「同一セクションのみ同時生成」で利点を部分取り込みしつつ軽量化。将来 A→B 発展。
- **C（リファインメントのみ）**: 実装最易だが**残差補正止まりで原理的壁を越えにくい**。A の front-end 結合として活用（単独では非採用）。
- **D（AEなし直接拡散）**: 高計算・忠実性中程度。潜在採用で回避。
- **E（離散 AR）**: 忠実性/速度で不利。
- **F（Schrödinger Bridge）**: 理論的に魅力（A の混合アンカーは思想的に近い）だが**未成熟・実装最難**。**A を Bridge 方向へ発展させる**のが現実的。

---

## 5. 計算量・規模の見積り（設計目安）
- **AE**: 事前学習流用＋微調整。VRAM 中。
- **Flow DiT**: 潜在フレーム列（例 数百フレーム/10s）に Transformer。学習は 24-80GB GPU、数十万 step（Phase 5 の tier 設定を流用）。
- **推論**: 非蒸留で数〜数十 ODE step、**蒸留後 1-2 step**。per-target なので active 楽器数×step。
- Phase 5 の GPU 準備（AMP/accum/DDP/自動化）をそのまま適用可能。

## 6. リスクと対策
| リスク | 対策 |
|---|---|
| 音符/音色の捏造 | 混合アンカー＋mixture-consistency＋Score-fidelity 損失＋CFG 弱め |
| Score アライメント誤差 | score-only 条件（頑健）、アライメント誤差注入学習、AMT の信頼度重み |
| 実録の正解不在（2ch に GT なし） | 合成学習＋実録は採譜整合/参照録音/知覚評価で間接評価 |
| ブリードなし実オケデータ不足 | 高品質合成の規模化＋無ラベル実録の自己教師 |
| ユニゾンの一意解不在 | Score で一意性最大化、評価を「知覚的妥当性」へ再定義 |
| 計算コスト | 潜在＋FM＋consistency 蒸留 |

## 7. 実装フェーズ（将来）への接続
1. **P3**: RoFormer をオーケストラでスケール学習（front-end 確立）。
2. **AE**: ステレオ音声 VAE をオケ微調整。
3. **P1 core**: 潜在 Flow DiT を Score 条件で学習（合成）→ 同一セクション同時生成モード。
4. **P4**: ドメイン適応（実録＋自己教師）。
5. **蒸留 & 統合**: consistency 蒸留、Backbone Interface へ統合、評価ハーネス接続。

---

## 8. 結論
本プロジェクトは、**「潜在 Rectified Flow-Matching DiT による Score 条件付き・混合アンカー型ターゲット抽出（同一セクションは同時生成）＋ RoFormer 決定論フロントエンド ＋ consistency 蒸留」**を採用する。これは **Exp 001-003 が示したマスキングの壁（ユニゾン/同一音色）を、2ch 制約の枠内で生成的に越える**唯一筋の通った設計であり、**忠実性を設計で担保**し、**既存の Taxonomy/Presence/Manifest/評価/GPU 基盤（Phase 2-5）にそのまま載る**。実装は次フェーズ（Phase 8: P3→AE→P1 core の順）で開始する。
