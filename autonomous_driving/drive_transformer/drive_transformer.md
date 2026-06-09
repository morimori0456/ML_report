# DriveTransformer 完全解説 — Unified Transformer for Scalable End-to-End Autonomous Driving

> 論文: Xiaosong Jia, Junqi You, Zhiyuan Zhang, Junchi Yan, *DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving*, ICLR 2025
> arXiv: [2503.07656](https://arxiv.org/abs/2503.07656) / 公式実装: [Thinklab-SJTU/DriveTransformer](https://github.com/Thinklab-SJTU/DriveTransformer)

End-to-End 自動運転（E2E-AD）を **単一の Transformer** で統一する手法。本ドキュメントは「なぜこの設計なのか」を、従来手法の課題から逆算して理解することを目的とする。手を動かす最小実装は [drive_transformer_demo.ipynb](drive_transformer_demo.ipynb) を参照。

---

## 目次
1. [前提: E2E-AD の系譜と課題](#1-前提-e2e-ad-の系譜と課題)
2. [DriveTransformer の3本柱](#2-drivetransformer-の3本柱)
3. [全体アーキテクチャとデータフロー](#3-全体アーキテクチャとデータフロー)
4. [タスククエリ — Agent / Map / Ego](#4-タスククエリ--agent--map--ego)
5. [Sensor Cross-Attention（BEVを作らない理由）](#5-sensor-cross-attentionbevを作らない理由)
6. [Task Self-Attention（タスク並列）](#6-task-self-attentionタスク並列)
7. [Temporal Cross-Attention とストリーミングFIFO](#7-temporal-cross-attention-とストリーミングfifo)
8. [タスクヘッドと損失関数](#8-タスクヘッドと損失関数)
9. [ハイパーパラメータと計算量](#9-ハイパーパラメータと計算量)
10. [よくある誤解とつまずき所](#10-よくある誤解とつまずき所)

---

## 1. 前提: E2E-AD の系譜と課題

### 1.1 従来パラダイム = 逐次（sequential）
UniAD / VAD に代表される従来の E2E-AD は、人間の運転パイプラインを模して

```
知覚(Perception) → 予測(Prediction) → 計画(Planning)
```

を **直列** に並べる。各段の出力が次段の入力になる。この設計には2つの構造的弱点がある。

1. **累積誤差（cumulative error）**: 知覚の誤りが予測・計画に伝播し、後段で訂正できない。情報は前→後へ一方向にしか流れない。
2. **学習の不安定性**: 後段タスク（計画）の勾配が、前段（知覚）の重い backbone まで届きにくい。段ごとに役割が固定されるため、計画に有用な特徴を知覚段が学ぶ保証がない。

### 1.2 もう一つの軸 = BEV 表現
多くの手法は多視点画像を **BEV（Bird's-Eye-View）特徴**という密なグリッド（例: 200×200×C）へ変換してから処理する。BEV は直感的だが、

- 構築コストが高い（視点変換・voxel pooling）
- グリッド解像度に表現が縛られる
- 「知覚に最適な中間表現」であって「計画に最適」とは限らない

### 1.3 DriveTransformer の立場
> **逐次パイプラインも密なBEVも捨て、すべてのタスクを生のセンサー特徴の上で Transformer の attention だけで解く。**

タスク同士・タスクと生センサー・タスクと過去履歴の関係を、すべて attention に学習させる。これにより累積誤差を断ち、計画の勾配を直接 backbone へ流す。

---

## 2. DriveTransformer の3本柱

| 柱 | 何をするか | 解決する課題 |
|---|---|---|
| **Task Parallelism**（タスク並列） | 全タスククエリが各ブロックで相互に attention。明示的な階層を作らない | 累積誤差・勾配が後段で詰まる問題 |
| **Sparse Representation**（疎表現） | タスククエリが生センサー特徴に直接 cross-attention。BEVを作らない | BEV構築コスト・表現の硬直化 |
| **Streaming Processing**（ストリーミング） | 過去のタスククエリを FIFO キューに貯め、temporal cross-attention で時間融合 | 時系列の効率・特徴再利用 |

この3つが「1つの Transformer ブロック」の中に同居しているのが核心。ブロックは

```
Task Self-Attn → Sensor Cross-Attn → Temporal Cross-Attn → FFN
```

の4オペレーションからなり、これを **L=12 回**積む。各ブロック出力にタスクヘッドを付け、全ブロックで損失を取る（deep supervision）。

---

## 3. 全体アーキテクチャとデータフロー

```
                  ┌─────────────────── 6枚の多視点カメラ画像 ──────────────────┐
                  │  CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_L, BACK_R │
                  └───────────────────────────┬──────────────────────────────┘
                                              ▼
                        ResNet50 / EVA / VoVNet backbone
                                              ▼
                  画像特徴 [B, N_cam, H, W, D]  +  3D-ray 位置エンコーディング
                                              │  flatten
                                              ▼
                  img_feats [B, N_img_token, D]   img_pos_embed [B, N_img_token, D]
                                              │
   ┌──────────── タスククエリ（学習パラメータで初期化）─────────────┐  │
   │ Agent  [B, 900, D]   Map [B, 100, D]   Ego [B, 1, D]        │  │
   └──────────────────────────┬─────────────────────────────────┘  │
                              ▼                                      │
   ╔══════════════════ Decoder Block × 12 ════════════════════════╗ │
   ║  ① Task Self-Attn : [Agent;Map;Ego] が相互に attention       ║ │
   ║  ② Sensor Cross-Attn: タスククエリ → img_feats ◄─────────────╫─┘
   ║  ③ Temporal Cross-Attn: タスククエリ → FIFOキュー(過去L frame)║◄── history queue
   ║  ④ FFN : Agent/Map/Ego 別々の FFN                            ║
   ║  ─ 各ブロック出力にヘッド: det/motion/map/plan → 損失         ║
   ╚══════════════════════════════════════════════════════════════╝
                              ▼
   検出box / 動き予測 / オンライン地図 / 自車軌跡(6モード)
                              ▼
        Winner-Take-All で1モード選択 → 制御（CARLA closed-loop）
                              ▼
   上位 Top-K クエリを次フレームのため FIFO キューへ push
```

ポイント:
- **生センサー特徴を全ブロックが直接見続ける**（②）。一度BEVに潰してから捨てる、ではない。
- **時間方向は「特徴マップ」ではなく「クエリ」を貯める**（③）。疎なので軽い。
- ブロックを積むほど性能が上がる**スケーラビリティ**を持つ（タイトル "Scalable" の所以）。

---

## 4. タスククエリ — Agent / Map / Ego

DriveTransformer の状態はすべて「クエリ（query token）」で表現される。種類は3つ。

| クエリ | 個数(Large) | 表すもの | 担当タスク | 初期化 |
|---|---|---|---|---|
| **Agent query** | 900 | 動的物体（車・歩行者） | 3D検出 + 動き予測 | 学習パラメータ + 一様な位置エンコ |
| **Map query** | 100 | 静的要素（車線・標識） | オンラインマッピング | 学習パラメータ + 一様な位置エンコ |
| **Ego query** | 1 | 自車の取りうる挙動 | 計画（軌跡生成） | CAN bus を MLP に通して初期化、位置エンコ=0 |

3種を連結して `query = [Agent; Map; Ego]`（形状 `[B, 900+100+1, D]`）として1本の系列に扱う。**この連結がタスク並列の物理的な実体**である。Self-Attention をかければ、agent と ego が同じ attention 行列の中で相互作用する。

> 直感: 「あの車（agent）が右折しそう（motion）→ 自車（ego）は減速すべき」という推論を、別モジュール間のメッセージパッシングではなく、**1枚の attention 行列の非対角成分**として学習する。

---

## 5. Sensor Cross-Attention（BEVを作らない理由）

### 5.1 何をするか
タスククエリ（query）を Q、画像トークン `img_feats` を K=V として cross-attention する。

```
Q = LayerNorm(query + pos_embed) を Linear  (agent/map/ego で別々の重み cross_w_q[0..2])
K = V = img_feats  (+ key_pos = img_pos_embed)
out = identity + Attention(Q, K, V)        # identity は残差
```

公式実装では map query を `map_pts_per_vec`（折れ線の点数）に展開してから cross-attention し、より細かく画像を見る。

### 5.2 3D 位置エンコーディング（PETR 系）
BEV を作らない代わりに、**各画像パッチがどの3D方向を見ているか**を位置エンコで与える。

1. 各パッチ（u,v）から、カメラ内部・外部行列を使って3D空間へ **レイ（ray）** を飛ばす
2. レイ上を K 個の深度でサンプリングし、3D点列を作る
3. それらを連結して MLP に通し、`img_pos_embed` とする

これにより「画像の左上のこのパッチは、自車の右前方◯mあたりを向いている」という幾何が attention の key に埋め込まれる。クエリはこの位置情報を手がかりに、必要な画像領域へ注意を向ける。**カメラ外部キャリブレーション**（[../camera_calibration/extrinsic_calibration.md](../camera_calibration/extrinsic_calibration.md)）がここで効く。

### 5.3 なぜ疎（sparse）が良いか
- BEVグリッド全セルを計算せず、**900+100+1個のクエリが必要な画素だけ**に注意する → 計算が軽い
- 中間表現（BEV）を経由しないので、計画タスクの勾配が backbone まで一直線に届く（end-to-end 最適化と整合）

---

## 6. Task Self-Attention（タスク並列）

連結クエリ `[Agent; Map; Ego]` に通常の self-attention をかけるだけ。ただし意味が深い。

- agent ↔ agent: 物体間の相互作用（追い越し・追従）
- agent ↔ map: 「この車は車線に沿っているか」
- ego ↔ agent/map: 計画に必要な周辺文脈を ego query が直接吸い上げる

**階層がない**のが従来との違い。UniAD では「検出→追跡→予測→占有→計画」と固定順だが、ここでは全タスクが対等に、各ブロックで双方向に情報交換する。どの関係が重要かは attention が学習で決める。

---

## 7. Temporal Cross-Attention とストリーミングFIFO

単一フレームでは速度・意図が読めない。時間融合が要る。DriveTransformer は **密なBEV特徴を時系列スタックしない**。代わりに**疎なクエリを貯める**。

### 7.1 FIFO キュー
- タスク種別ごと（agent/map/ego）に別キュー
- 毎フレーム、**確信度上位 Top-K**（agent/map は各50個）のクエリだけを push（冗長を避ける）
- 過去 `memory_len_frame = 10` フレーム分を保持（nuScenes は 4）

### 7.2 時間融合の手順
現在クエリ Q が、キュー内の過去クエリ K=V に cross-attention する。ただし時系列で座標系が動くため補正が必要:

1. **自車座標変換**: 過去フレームの位置エンコを、変換行列 `T_{t→t0}` で現在の自車座標系へ移す
2. **動き補償**: agent は予測速度を使い、過去位置を現在時刻まで外挿
3. **相対時刻埋め込み**: 時間差 `(t − t0)` をエンコードして key に加算
4. 公式実装では各メモリ系列の先頭に **ゼロのレジスタトークン**を1つ付け、「過去に何も無い」場合の逃げ場を作る（attn_mask と併用）

> 効果: 「3フレーム前のこの agent query は、今のこの query と同一物体」という対応を attention が張れる。特徴（クエリ）をそのまま再利用するので **feature reuse** が効き、計算が軽い。

DiT 風の adaptive LayerNorm で時刻条件を注入する実装になっている（時刻に応じて正規化のスケール/シフトを変える）。

---

## 8. タスクヘッドと損失関数

各デコーダブロックの出力（agent/map/ego query）にヘッドを付け、**全ブロックで損失**を取る（deep supervision、推論時は最終ブロックのみ使用）。

| タスク | ヘッド出力 | 損失 |
|---|---|---|
| **検出 (detection)** | 3D box（中心・寸法・向き・クラス） | DETR 流 Hungarian マッチング損失 |
| **動き予測 (motion)** | 各 agent の将来軌跡（マルチモード） | Winner-Take-All（局所 agent 座標系） |
| **マッピング (mapping)** | 折れ線地図要素 | MapTR 流 Hungarian マッチング損失 |
| **計画 (planning)** | 自車軌跡 × **6モード** + 確信度 | Winner-Take-All（6モードの最良へ回帰）+ 分類 |

総損失は重み付き和で、各項のスケールが約1に揃うよう調整:

```
L = w_det·L_det + w_motion·L_motion + w_map·L_map + w_plan·L_plan
```

### 8.1 計画の6モード
ego query は `nn.Embedding(6, D)` の **モード埋め込み**を持つ。6モードは大まかに「直進 / 停止 / 左折(浅・深) / 右折(浅・深)」のような運転意図に対応。学習は WTA：6本の予測軌跡のうち GT に最も近い1本だけに回帰損失をかける（モード崩壊を防ぎ多峰性を保つ）。公式実装は等時間間隔(fix_time)と等距離間隔(fix_dist)の2系統で軌跡を出す。

---

## 9. ハイパーパラメータと計算量

公式 `drivetransformer_large.py` より:

| 項目 | 値 |
|---|---|
| Agent query 数 | 900（うち伝播 Top-50/frame/type） |
| Map query 数 | 100（うち伝播 Top-50/frame/type） |
| Ego query 数 | 1（モード埋め込み6） |
| デコーダ層数 L | 12 |
| 隠れ次元 D (Large) | 768 |
| メモリ長 | 10 frame（nuScenes 4） |
| パラメータ数 (Large) | 約 646M |
| backbone | ResNet50 / VoVNet / EVA02 |

スケール則: Tiny→Small→Base→Large と層・次元を増やすほど Bench2Drive の運転スコアが単調改善。BEVを持たない疎設計ゆえメモリ・FPS効率が良く、Large でも約211msのレイテンシで closed-loop SOTA。

### 評価
- **Bench2Drive（CARLA closed-loop）**: 運転スコア SOTA
- **nuScenes（open-loop）**: 高FPSで競争力ある成績

---

## 10. よくある誤解とつまずき所

1. **「BEVを一度作ってから捨てる」ではない** — 最初から作らない。タスククエリが毎ブロック生画像特徴を直接見る。
2. **「逐次パイプラインの並列化」ではない** — 段を並列に並べたのではなく、**段という概念自体を消し**、全タスククエリを1系列の self-attention に入れた。
3. **時間融合で貯めるのは特徴マップではなくクエリ** — 疎なので10フレーム分でも軽い。密BEVを10枚スタックする手法とコストが桁違い。
4. **temporal の座標変換を忘れると壊れる** — 過去クエリの位置を現在自車座標へ移し、agent は動き補償する。これを怠ると過去と現在の物体対応がズレる。
5. **deep supervision は必須級** — 全ブロックに損失を付けることで、深いブロックでも勾配が立ち、タスク並列が安定する。
6. **Ego query の位置エンコは0** — agent/map と違い自車は常に原点なので一様PE不要。CAN bus（速度・舵角）で初期化する点に注意。
7. **map query を点列に展開してから sensor cross-attn** — 折れ線地図を画素レベルで精緻に読むため。検出(agent)とは展開の仕方が異なる。

---

## 参考

- 論文: [arXiv:2503.07656](https://arxiv.org/abs/2503.07656) / [OpenReview](https://openreview.net/forum?id=M42KR4W9P5)
- 公式実装: [github.com/Thinklab-SJTU/DriveTransformer](https://github.com/Thinklab-SJTU/DriveTransformer)
- 関連: 位置エンコの原理は [PETR](https://arxiv.org/abs/2203.05625)、地図は [MapTR](https://arxiv.org/abs/2208.14437)、検出は [DETR](https://arxiv.org/abs/2005.12872)
- 本リポジトリ内の前提知識: [カメラ外部キャリブレーション](../camera_calibration/extrinsic_calibration.md)（3D位置エンコの幾何）、[VADデータローダー](../VAD/dataloader.md)（nuScenes/Bench2Drive入力）

> 実装で理解を固めるには [drive_transformer_demo.ipynb](drive_transformer_demo.ipynb) を実行。ResNetやデータセットを排した**最小の純PyTorch版 DriveTransformer**（3種のattention・FIFOキュー・6モード計画ヘッド）をCPUで動かし、各テンソル形状と勾配の流れを確認できる。
