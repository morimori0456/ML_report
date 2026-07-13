# ML Report — Roadmap & Gap Backlog

このレポート集の「まだ書けていないテーマ」を振り返り用にまとめたもの。優先度つきの backlog として、追加したら `- [ ]` を `- [x]` にしていく運用。

- 作成日: 2026-07-13
- 追加方法: `/add-report <topic>`(md → ipynb → 実行 → README更新 → push を自動化)
- 用語方針: 専門用語は英語のまま、初出に平易な説明を付ける

---

## 現状スナップショット

| カテゴリ | 本数 | 中身の傾向 |
|---|---|---|
| distillation | 7 | 明確な強み。KDは response/feature/relation/multi-teacher/self/FM をほぼ網羅 |
| autonomous_driving | 11 | **planning / E2E / 評価に偏り**。perception〜prediction が空白 |
| llm | 4 | inference(KV cache)+ fine-tuning 中心 |
| agentic_engineering | 2 | loop設計 |
| ema / experiment_tracking / infrastructure | 各1 | 単発 |

一言でいうと **「distillation に深く、AD は planning/評価に寄り、知覚(perception)・基盤(fundamentals)・deploy が空白」**。

---

## 次に足すなら(優先度つき Top 5)

プロフィール(AD AI・E2E・海外/起業志向・Jetson Thor 実機あり)に対する費用対効果順。

1. [ ] **BEV perception 入門(LSS → BEVFormer → BEVFusion)** — ADの最大の穴。既存 VAD/DriveTransformer と直結
2. [ ] **Edge deployment(TensorRT + INT8 量子化)hands-on** — Jetson実機で回せる、差別化最大
3. [ ] **Motion prediction(VectorNet / MTR / QCNet)** — planning評価(nuPlan/NAVSIM)群と対になり「予測→計画」が完結
4. [ ] **Occupancy networks(Occ3D / SurroundOcc / OccNet)** — 話題性が高く空白、面接ネタ
5. [ ] **VLM / VLA for driving(world model含む)** — LLM知見 × 運転の交差点、最も尖る

---

## 欠落テーマ(カテゴリ別 backlog)

### A. Autonomous Driving — 知覚〜予測スタックが丸ごと空白（最大の構造的穴）

現状は localization / calibration → いきなり planning・E2E・評価 に飛び、スタックの中間層が無い。

- [ ] **BEV perception**: LSS (Lift-Splat-Shoot), BEVFormer, BEVFusion — 現代ADの共通言語、VADの前提
- [ ] **3D object detection**: CenterPoint, PointPillars, TransFusion — 知覚の土台(camera/LiDAR/fusion)
- [ ] **Occupancy prediction**: Occ3D, SurroundOcc, OccNet — Tesla以降のホット領域
- [ ] **Motion prediction / forecasting**: VectorNet, MTR, QCNet — planningの入力(nuPlanは評価だけで手法が無い)
- [ ] **Online HD mapping**: MapTR, VectorMapNet — VADが「HD map生成」に触れるが専用解説なし
- [ ] **Multi-object tracking**: ByteTrack系 / 3D MOT — 知覚→予測を繋ぐ層
- [ ] **Sensor fusion 深掘り**: camera-LiDAR fusion(BEVFusion / TransFusion)を localization_tech から独立

### B. 基盤(Fundamentals）— 応用は深いのに土台の単発解説が無い

- [ ] **Transformer / attention 本体**: KV cache はあるのに attention・positional encoding の基礎が無い
- [ ] **Normalization**: BatchNorm / LayerNorm / GroupNorm(weight_ema が BN buffer に触れるだけ)
- [ ] **Optimizer & scheduler**: AdamW, warmup, cosine — 全レポートの前提だが未整理
- [ ] **Diffusion models 基礎**: 生成系ゼロ。planning でも diffusion policy が主流化
- [ ] **Self-supervised / contrastive**: SimCLR / MoCo / DINO(DINO は distillation で言及のみ)

### C. Model Compression — 三本柱のうち distillation だけ突出

- [ ] **Quantization**: PTQ / QAT, INT8, GPTQ / AWQ — distillation と対
- [ ] **Pruning**: structured / unstructured, movement pruning
- [ ] **NAS / efficient architectures**: MobileNet系, efficient attention(補助的)

### D. Deploy / Edge — 完全に空白（Jetson 実機があるのに未活用）

- [ ] **TensorRT / ONNX / INT8 量子化(PTQ・QAT)** — Jetson実機で回せる hands-on にできる(navsim_hands_on 型)
- [ ] **CUDA kernel / 推論プロファイリング** — Nsight, レイテンシ分解
- [ ] **モデルサービング**: Triton Inference Server, dynamic batching
- 補足: `infrastructure/` は学習側(Slurm/NCCL)だけで、推論・deploy 側が対で欠落

### E. Generative × Driving（最前線・最も尖る）

- [ ] **World models for driving**: GAIA-1, Vista, DriveDreamer
- [ ] **VLM / VLA for driving**: DriveGPT, LINGO系, vision-language-action(domain_finetune_driving はテキストLLM止まり)
- [ ] **Diffusion planners / policy**: 運転軌跡生成への diffusion 応用

### F. LLM — 言及止まりの主題を独立回に

- [ ] **RAG 専用回**: 各所で言及のみ、retrieval / chunking / reranking を主題化
- [ ] **Agent / tool-use**: agentic_engineering は loop 設計中心、LLM側の tool-calling 機構
- [ ] **LLM evaluation**: benchmark / LLM-as-judge / contamination
- [ ] **Inference serving 深掘り**: vLLM / SGLang(kv_cache から独立)

### G. その他

- [ ] **データ工学**: labeling / auto-labeling / dataset curation(nuPlanのauto-labelに触れるが主題化せず)
- [ ] **Classical control**: MPC / controller(PDMS回で LQR+bicycle を使ったが制御そのものは未解説)
- [ ] **Closed-loop simulation**: CARLA など(nuPlan CL は一部のみ)

---

## メモ

- distillation の深さは強みなので維持。次の重心は **AD perception** と **deploy/edge** に置くと、planning偏重を是正しつつ実機(Jetson Thor)資産を活かせる。
- 関連レポートは相互リンク(例: BEV perception ↔ VAD ↔ DriveTransformer、量子化 ↔ lora_qlora の NF4)。
