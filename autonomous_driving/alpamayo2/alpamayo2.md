# NVIDIA Alpamayo 2 — 推論する自動運転 VLA を実装レベルで理解する

> NVIDIA の open Vision-Language-Action (VLA) モデル `Alpamayo` ファミリー（1 Nano / 1.5 Nano / 2 Super）を、公式実装のソースコードと Alpamayo-R1 論文 (arXiv:2511.00088) に基づいて解剖する導入レポート。理論だけでなく **公式実装と同じ数式・同じハイパーパラメータで動く最小再実装** を companion notebook [`alpamayo2.ipynb`](alpamayo2.ipynb) に用意した（CPU のみで完走する）。さらに、実物の VLM バックボーンを実 GPU に載せて同じパイプラインを回した GPU leg を [`alpamayo2_gpu_smoke_kaggle.ipynb`](alpamayo2_gpu_smoke_kaggle.ipynb) に置いた（Kaggle Tesla T4 での実行結果を焼き込み済み）。

End-to-end (E2E) 自動運転モデルは、カメラ画像から直接 trajectory（自車が今後たどる経路）を出力することで、認識・予測・計画のモジュール間で情報が落ちる問題を解いた。しかし模倣学習ベースの E2E プランナは long-tail — つまり滅多に起きないが起きると危険なシナリオ — で脆く、「なぜその経路を選んだのか」を説明できない。Alpamayo の主張は単純である：**画像と trajectory の間に、明示的で構造化された「因果の言語」を一段挟む**。そうすることで (1) 安全性のクロスチェックが可能になり、(2) 意思決定が人間に読める形で説明され、(3) 強化学習で検証可能な報酬信号が得られる。本レポートはこの主張がコードのどこにどう実装されているかを、実際に動く最小実装まで落として確認する。

---

## Table of Contents
1. [Alpamayo ファミリーの全体像](#1-alpamayo-ファミリーの全体像)
2. [なぜ reasoning を挟むのか](#2-なぜ-reasoning-を挟むのか)
3. [Chain-of-Causation — 構造化された運転の因果記述](#3-chain-of-causation--構造化された運転の因果記述)
4. [アーキテクチャ全体とトークン設計](#4-アーキテクチャ全体とトークン設計)
5. [Action Space — unicycle 運動学で trajectory を制御量に変換する](#5-action-space--unicycle-運動学で-trajectory-を制御量に変換する)
6. [Action Expert と Flow Matching](#6-action-expert-と-flow-matching)
7. [学習レシピ — 3 段階と GRPO](#7-学習レシピ--3-段階と-grpo)
8. [評価結果と ablation が示すもの](#8-評価結果と-ablation-が示すもの)
9. [実際に動かす — 環境要件と現実的な選択肢](#9-実際に動かす--環境要件と現実的な選択肢)
10. [Common Pitfalls](#10-common-pitfalls)
11. [References](#11-references)

---

## 1. Alpamayo ファミリーの全体像

Alpamayo は NVIDIA が 2025 年末から 2026 年にかけて段階的に公開した、自動運転向けの open VLA モデル群である。VLA (Vision-Language-Action) とは、画像・言語・行動を単一のモデルで扱う枠組みで、ロボティクス由来の用語である。Alpamayo はこれを自動運転に持ち込み、**カメラ映像 → 因果推論テキスト → 走行 trajectory** を 1 パスで出力する。

| モデル | HF repo ID | パラメータ | 構成 | 公開時期 | 位置づけ |
|---|---|---|---|---|---|
| Alpamayo 1 Nano（旧 Alpamayo-R1） | `nvidia/Alpamayo-R1-10B` | 10.5B (VLM 8.2B + expert 2.3B) | Cosmos-Reason + diffusion decoder | 2025-12 | 論文 arXiv:2511.00088 の対象。リポジトリは deprecated |
| Alpamayo 1.5 Nano | `nvidia/Alpamayo-1.5-10B` | 10.5B (VLM 8.2B + expert 2.3B) | `nvidia/Cosmos-Reason2-8B` + RL post-train | 2026-03 | navigation guidance 入力と VQA を追加 |
| **Alpamayo 2 Super** | `nvidia/Alpamayo2-Super` | 34B (VLM 32B + expert 2.3B) | Cosmos 3 Super Reasoner + diffusion expert | 2026-08-04 | 商用利用可（OpenMDW-1.1）。360 度 6 カメラ |

「Nano」「Super」は NVIDIA のサイズ階層命名で、間の 7B/2B といった小型変種は **存在しない**。重みは OpenMDW-1.1、コードは Apache-2.0 という二重ライセンス構成になっている。

補足として、Alpamayo 2 Super の実チェックポイントの `config.json` は、ネストされた `vlm_config` の中に `model_type: "qwen3_vl"` / `vlm_class: "Qwen3VLForConditionalGeneration"` を持つ。NVIDIA は公式ブログで「Cosmos 3 Super Reasoner バックボーン」と表現するが、実体は **Qwen3-VL アーキテクチャを継続事前学習したもの** と読むのが正確である。この事実は後述する GPU 実験の設計に直結する（同じ `qwen3_vl` 系統の小型モデルなら、同じコードパスで動かせる）。

なお世代によってバックボーンの系統が違う点は注意を要する。Alpamayo 1.5 Nano の base model は HF 上では `nvidia/Cosmos-Reason2-8B` であり、Qwen3-VL 系統であることが config で確認できるのは Alpamayo 2 Super 側である。

ライセンスについても誤解しやすい点がある。3 つのモデルはいずれも HF 上で **gated ではなく、誰でも即ダウンロードできる**。当初 Alpamayo 1 / 1.5 は R&D 向けとして導入されたが、2026-08-04 の NVIDIA ブログは "The OpenMDW license is now being applied across the entire Alpamayo model family so developers can deploy any of the models commercially without requiring additional permissions" と述べており、**現在は 3 世代すべてが商用利用可**である。一方で **データセット `nvidia/PhysicalAI-Autonomous-Vehicles` は gated（自動承認）** で、ライセンス同意とアクセストークンが必要になる。「モデルは自由、データは要同意」と覚えるとよい。

エコシステムとしては、モデル本体のほかに以下が公開されている。

| リポジトリ | 役割 |
|---|---|
| [`NVlabs/alpamayo2`](https://github.com/NVlabs/alpamayo2) | Alpamayo 2 Super の推論コード |
| [`NVlabs/alpamayo-recipes`](https://github.com/NVlabs/alpamayo-recipes) | SFT / RL post-training / 量子化レシピのハブ |
| [`NVlabs/alpasim`](https://github.com/NVlabs/alpasim) | Gaussian Splatting ベースの closed-loop シミュレータ |
| [`NVlabs/alpamayo-coc-autolabeler`](https://github.com/NVlabs/alpamayo-coc-autolabeler) | Chain-of-Causation の自動ラベリングパイプライン |

### 1.1 世代間で何が変わったか

3 世代の差分を、モデルカード・各リポジトリの README・`config.json` の実値・実装コードから突き合わせた結果が以下である。

| 項目 | 1 Nano | 1.5 Nano | 2 Super |
|---|---|---|---|
| パラメータ | 10.5B (8.2B + 2.3B) | 10.5B (8.2B + 2.3B) | 34B (32B + 2.3B) |
| VLM バックボーン | Cosmos-Reason | `nvidia/Cosmos-Reason2-8B` | Cosmos 3 Super Reasoner（config 実体は `qwen3_vl`） |
| **RL post-training（重み）** | **なし**（論文には記載、リリースには未収録） | **あり** | あり（AlpaGym で closed-loop RL） |
| navigation conditioning | なし | **あり**（自然言語コマンド + nav-CFG） | あり（ただし 2 GPU 構成の別スクリプト） |
| VQA | なし | **あり** | あり + **2D grounding** |
| meta-action 出力 | なし | なし | **あり** |
| auto-labeling 出力 | なし | なし | **あり**（4 フィールド JSON） |
| カメラ | 4 台固定 | 4 台既定・**台数可変** | **7 台リング → タスク別に 6 台選択**（後方含む 360 度） |
| 学習データ（映像） | 80,000 時間 | 80,000 時間 | **115,000 時間** |
| CoC トレース | 700K | **3.0M** | **3.7M** |
| minADE$_6$@6.4s | 1.22 m | 0.916 m | 0.911 m |
| AlpaSim score | 0.73±0.01 | 1.37±0.10 | 1.50±0.13 |
| LingoQA (Lingo-Judge) | — (VQA 非対応) | 74.2 | 79.2 |
| リポジトリ | deprecated (2026-05) | active | active |

数字の読み方として重要なのは、**性能の伸びの大半が 1 → 1.5 で起きている**ことである。minADE は 1.22 → 0.916 m（25% 改善）に対し 1.5 → 2 Super は 0.916 → 0.911 m（0.5%）。AlpaSim も 0.73 → 1.37（88% 改善）に対し 1.37 → 1.50（9%）。パラメータを 3.4 倍にした 2 Super の寄与は、運転そのものよりタスクの幅（meta-action・grounding・auto-labeling）と言語理解に出ている。

#### 変わらなかったもの — action space

コードを世代間で読み比べて最も印象的だったのは、`action_space/unicycle_accel_curvature.py` が **3 世代で一字一句同一**だったことである。`n_waypoints=64`、`dt=0.1`、`accel_bounds=(-9.8, 9.8)`、`curvature_bounds=(-0.33, 0.33)`、Tikhonov 正則化の係数群、そして `_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.6` まで変わっていない。差分はロガーの実装と `register_buffer` の `persistent` 指定だけである。

さらに `config.json` の**正規化統計も 3 世代で完全に同一**である（`accel_std=0.6810426736454882` 等）。学習データが 80,000 → 115,000 時間に増えても再計算されていない。

つまり **車両の運動学モデルと action の表現は、最初から正しく設計されて以降触られていない**。世代交代で動いたのはバックボーンとデータとタスクの側であり、「trajectory をどう表現するか」という土台は固定されている。本レポートの companion notebook がこれらの定数を使っているのは、そういう意味で世代に依存しない。

#### 変わったもの — classifier-free guidance の一生

CFG 関連のコードは 3 世代で興味深い経路をたどる。

| 世代 | 状態 |
|---|---|
| 1 Nano | `FlowMatching.__init__` に `inference_guidance_weight` は**あるが、`sample()` からも `_euler()` からも一度も参照されない**（死んだパラメータ）。`unguided_step_fn` 引数も無い |
| 1.5 Nano | `BaseDiffusion` に `use_classifier_free_guidance` が追加され、`_guided_v()` が実装される。nav 区間を除いた unguided プロンプトを別途 prefill し、guided/unguided の KV キャッシュ両方をサンプラに渡す |
| 2 Super | `ExpertModel.__init__` が `if self.diffusion.use_classifier_free_guidance: raise ValueError("Classifier-free guidance is not supported by Alpamayo2Super.")` と**明示的に禁止**。nav-CFG は `examples/two_gpu_nav_cfg_demo.py` という 2 GPU 前提の別スクリプトへ追い出された |

1 Nano のコードにある `inference_guidance_weight` を見て「CFG が使える」と読むと間違う。実装は 1.5 で入り、2 Super では本体パスから外されている（VRAM の都合と読める。公式実測で 6 カメラ × 4 フレームが VLM 側 67 GiB / expert 側 71 GiB）。

#### その他の実装差分

| 項目 | 1 Nano | 1.5 Nano | 2 Super |
|---|---|---|---|
| expert `hidden_size` | 2048 | 2048 | **1536** |
| expert の実体 | 本体クラス内にインライン | 同左 | **独立した `PreTrainedModel`**（`models/expert.py`、`model_type="alpamayo2_super_expert"` として別登録） |
| 共有基底クラス | `ReasoningVLA` (`models/base_model.py`) | 同左 | **廃止**、`PreTrainedModel` 直継承 + `vlm_class` による動的ロード |
| `tokens_per_history_traj` | 48 | 48 | **45** |
| 軌道語彙 | `traj_vocab_size=4000` 単一空間 | 同左 | history 1000 / future 3000 に**分離** |
| サンプリング API | `sample_trajectories_from_data_with_vlm_rollout(...)`, `num_traj_samples=6` | 同左 + `..._cfg_nav(...)` | `sample_trajectories_from_data(...)`, `num_traj_samples=1`、`*args/**kwargs` を廃し型付き引数へ |
| カメラ定義 | コード内に無し（データセット側に委譲） | 同左 | `common/constants.py` + `input_profiles.py` で**明示的に体系化** |
| torch / transformers ピン | 2.8.0 / 4.57.1 | 同一 | 同一 |

`num_traj_samples` の既定が 6 → 1 に下がっているのは、34B ではメモリ的に複数サンプルを回せないためである（公式のサンプルスクリプトにも "set for GPU memory compatibility" と注記がある）。**minADE$_6$ という指標は 6 サンプル取ることが前提**なので、既定のまま動かすと論文の数字とは比較できない。

#### エコシステム側

- `NVlabs/alpamayo-recipes` に **Alpamayo 2 向けの SFT / RL レシピはまだ無い**（2026-08 時点）。あるのは `alpamayo1_sft/`、`alpamayo1_5_sft/`、`alpamayo1_x_rl/`（1 と 1.5 共用の GRPO）、`alpamayo1_5_quant/` である。2 Super を自社データで fine-tune したい場合、公式レシピが無い状態から始めることになる。
- `NVlabs/alpagym`（closed-loop RL フレームワーク、2026-06 公開）の README は "It currently supports the Alpamayo 1.5 model with 10b parameters" とあり、**2 Super 対応が明記されていない**。一方 NVIDIA のブログは AlpaGym を 2 Super の closed-loop RL 基盤として紹介しており、記述に食い違いがある。実際の対応状況は不明。
- `NVlabs/alpamayo`（1 Nano）は 2026-05 に fine-tuning スクリプトが recipes へ移管されたのを機に deprecated 化した。

> **バックボーンの系譜について**: Cosmos-Reason（1 Nano 用）のベースアーキテクチャには情報源間で食い違いがある。Cosmos-Reason1 論文 (arXiv:2503.15558) は InternViT + Mamba-MLP-Transformer ハイブリッドと記述する一方、Qwen2.5-VL ベースとする二次情報もある。また Cosmos 3 Super Reasoner が Qwen3-VL-32B ベースであることは `config.json` の実体から強く示唆されるが、NVIDIA の一次情報でそう明言している箇所は確認できなかった。ここは**断定できない**。

### Why this matters
Alpamayo は「重みを公開した大きなモデル」ではなく、**データ生成（autolabeler）・学習（recipes）・評価（alpasim）まで揃った開発スタック** として出されている。自社データで fine-tune して製品に載せることを前提とした構成であり、そこが単なる研究リリースとの違いである。ただし世代を追うと、**論文に書かれた機能とリリースされた重みの機能は一致しない**（1 Nano は RL 重みも navigation conditioning も VQA も入っていない）。触る前にリリースノートの対応表を確認する必要がある。

---

## 2. なぜ reasoning を挟むのか

Alpamayo-R1 論文は、自動運転 VLA が汎用 VLM から不足している能力を 4 つ挙げる（Sec.3 冒頭）。これがそのままアーキテクチャの設計要件になっている。

1. **マルチカメラ・マルチタイムステップの 360 度認識が必要だが、標準 VLM は各フレームを独立に処理するためトークン数が爆発する。** 7 カメラ × 4 フレームを素直に入れるとリアルタイム推論が不可能になる。
2. **意思決定は free-form な語りではなく、因果的に構造化された推論に基づくべきである。**
3. **waypoint をテキストトークンとして自己回帰デコードするのは非効率であり、かつ幾何学的・運動学的制約を持たない。**
4. **long-tail での安全性には、reasoning trace と実際の action が整合している必要がある。**

4 点目が最も本質的である。「歩行者がいるので減速します」と出力しながら実際には加速する trajectory を出すモデルは、説明可能どころか有害である。Alpamayo はこの reasoning-action consistency を **RL の報酬として明示的に最適化する**（第 7 節）。

従来手法との対比を整理する。

| アプローチ | 中間表現 | 説明可能性 | long-tail 耐性 | 代表例 |
|---|---|---|---|---|
| モジュラー型（認識→予測→計画） | 明示的な物体・車線 | 高い（ただし計画ロジックのみ） | ルール記述の網羅性に依存 | 従来の AV スタック |
| E2E 模倣学習 | なし（暗黙の特徴量） | ほぼ無い | 弱い（疎な教師信号） | UniAD, VAD |
| VLM + free-form CoT | 自由文の推論 | 見かけ上は高い | 推論が行動に反映されず改善しないことがある | 一般的な driving VLM |
| **Alpamayo (CoC + VLA)** | **閉集合の driving decision + 因果要因** | **高い（決定に紐づく）** | **RL で検証可能な報酬を設計できる** | Alpamayo 1 / 1.5 / 2 |

### Why this matters
「説明を出させる」ことと「説明に従って動かす」ことは別問題である。free-form な Chain-of-Thought は前者しか担保しない。Alpamayo の設計上の核心は、**推論を閉集合の決定に接地させ、行動との一致を機械的に検証できる形にした** 点にある。

---

## 3. Chain-of-Causation — 構造化された運転の因果記述

### 3.1 通常の CoT が失敗する 3 パターン

論文 Sec.4 / Fig.2 は、運転動画に free-form な Chain-of-Thought (CoT) を書かせたときの失敗を 3 つに分類している。

| 失敗パターン | 内容 |
|---|---|
| Vague behavior descriptions | 具体的な運転行動を特定できず、自車 trajectory と弱い相関しか持たない語を選ぶ |
| Superficial reasoning | 文脈的な観察や仮想的要因を並べるだけで、自車の行動への直接の因果リンクを欠く |
| Causal confusion | 推論トレースに**未来の時間窓で生じた要因**が混入する（ラベリング時に動画全体を見せ、過去と未来を区別しないことに起因） |

3 つ目は特に厄介である。「前方車が急停止したので減速した」というラベルが、実際には減速後に前方車が停止したケースから作られていると、モデルは未来を透視する前提で学習してしまう。

### 3.2 CoC の 3 コンポーネント

Chain-of-Causation (CoC) は、この失敗を構造で潰す。

**(1) Driving Decision（閉集合）** — 各クリップに縦方向 1 つ・横方向 1 つ（または None）を割り当てる。

| 方向 | 語彙 |
|---|---|
| 縦方向（7 種） | Set speed tracking / Lead obstacle following / Speed adaptation (road events) / Gap-searching (for LC/merge/zipper) / Acceleration for passing/overtaking / Yield (agent right-of-way) / Stop for static constraints |
| 横方向（8 種） | Lane keeping & centering / Merge・Split (facility change) / Out-of-lane nudge / In-lane nudge / Lane change (lateral push) / Pull-over・curb approach / Turn (intersection/roundabout/U-turn) / Lateral maneuver abort |

**(2) Critical Components（オープン集合、7 カテゴリ）** — Critical objects（種別・相対姿勢・動き、不確実性 Low/High のタグ付き）/ Traffic lights（R・Y・G、矢印、視認性）/ Yield・Stop control / Road events（曲率・段差など）/ Lane・lanelines / Routing intent / ODD constraints（天候・工事など）。

**(3) Composed CoC Trace** — 上記 2 つを自然言語で結合したもの。設計原則は 3 つ：

- **decision grounding**: 単一の明示的な決定に紐づくこと
- **causal locality**: 引用する証拠がすべて**観測済みの履歴窓内**にあること（causal confusion の直接的な対策）
- **annotation economy**: 決定に関係する要因のみを書くこと

### 3.3 ラベリングと品質

人手ラベリングは 2 段階に分かれる。Stage I (0–2 秒) で履歴窓内のみから critical components を特定し、Stage II (0–8 秒) で安全性フィルタをかけたうえで、**Stage I で特定した因果要因のみを参照して** CoC trace を書く。この「先に過去だけを見て証拠を確定させる」順序が causal locality を強制する仕組みである。

自動ラベリングでは、まずルールベース検出器で atomic meta action（Gentle/Strong accelerate, Gentle/Strong decelerate, Maintain speed, Stop, Reverse / Steer left・right, Sharp steer left・right, Reverse left・right, Go straight）の遷移点をキーフレームとして検出し、そこを起点に LLM で構造化 CoC を生成する。

品質評価は GPT-5 を評価者として、driving decision の一致・因果要因の有無・cause-effect の妥当性という 3 つの True/False サブタスクに分解して行われ、**人間評価との一致率 92%**、構造化 CoC は free-form reasoning に対し **causal relationship score を相対 132.8% 改善** したと報告されている。論文は「100% の一致は必要条件でも達成可能な目標でもない」と明記しており、因果推論の本質的曖昧さと GT 側のノイズを認めている点は誠実である。

### Why this matters
CoC は「良い CoT を書かせるプロンプト工夫」ではなく、**アノテーションのプロトコルとして因果の向きを担保するデータ設計** である。閉集合の decision を持つことで、後段の RL で「推論と行動が一致したか」をルールベースで判定できる（第 7 節）。

---

## 4. アーキテクチャ全体とトークン設計

### 4.1 系列の定式化

モデルが扱う系列は次のように構成される（論文 Eq.1）。

$$[\boldsymbol{o}_{\text{image}},\ \boldsymbol{o}_{\text{egomotion}},\ \textsc{Reason},\ \boldsymbol{\tau}]$$

各要素は先行するすべての要素に条件づけられる。予測する trajectory はデフォルトで 6.4 秒先まで、10 Hz の 64 点である（Eq.2）。

$$\boldsymbol{\tau}=\{(x^{i},y^{i},\theta_{\text{yaw}}^{i})\}_{i=1}^{64}$$

そして、これと等価な制御ベースの表現（unicycle dynamics、Eq.3）を持つ。

$$\boldsymbol{a}=\{(a^{i},\kappa^{i})\}_{i=1}^{64}$$

ここで $a^i$ は加速度、$\kappa^i$ は曲率である。**この二重表現が Alpamayo の設計の要** で、第 5・6 節で詳述する。

### 4.2 マルチカメラのトークン化

第 2 節で挙げた要件 1（トークン数の爆発）への対策として、論文は 3 方式を比較している。

| 方式 | 追加パラメータ | 画像あたりトークン数 | 相対 minADE$_6$ |
|---|---|---|---|
| Baseline (single-image) | 0 | 160 (1.0×) | 0% |
| Triplane | 6.3M | 104 (1.5×) / 45 (3.6×) | -3% / +4% |
| Flex | 61.6M | 50 (3.2×) / 32 (5.0×) / 16 (10×) / 8 (20×) | -3% / -3% / -2% / -2% |

Baseline は $W\times H$ の画像をパッチ特徴 $\mathbf{f}\in\mathbb{R}^{W/14\times H/14\times D}$ にエンコードし、2×2 バイリニアダウンサンプルして $\mathbf{f}'\in\mathbb{R}^{W/28\times H/28\times D}$ とする。$W{=}448, H{=}280$ で 160 トークン/画像になる。

Triplane 方式はカメラ画像を BEV 的な 3 平面のグリッドに投影してからパッチ化するため、トークン数が**カメラ台数にも解像度にも依存しない**。グリッドサイズ $S_x, S_y, S_z$、パッチ化パラメータ $p_x, p_y, p_z$ に対し（Eq.4）：

$$\left(\frac{S_x-p_x}{p_x}+1\right)\left(\frac{S_y-p_y}{p_y}+1\right)+\left(\frac{S_x-p_x}{p_x}+1\right)\left(\frac{S_z-p_z}{p_z}+1\right)+\left(\frac{S_y-p_y}{p_y}+1\right)\left(\frac{S_z-p_z}{p_z}+1\right)$$

$S_x{=}S_y{=}96,\ S_z{=}48,\ p_x{=}p_y{=}p_z{=}8$ で 288 トークン。7 カメラなら 1 画像あたり約 41.1 トークン相当で、single-image の 3.9 倍の圧縮になる。

注目すべきは Flex の結果で、**20 倍圧縮しても minADE の劣化が 2% に留まる**。運転の意思決定に必要な情報量は、画像の生のピクセル情報よりはるかに少ないことを示唆している。

### 4.3 入出力の実仕様

| 項目 | Alpamayo 1 / 1.5 (10B) | Alpamayo 2 Super (34B) |
|---|---|---|
| カメラ | 4 台（front-wide, front-tele, cross-left, cross-right） | 6 台（cross-left, front-wide, cross-right, rear-left, rear-right, front-tele）。定義上は 7 台環状 |
| フレーム | 各カメラ 4 フレーム @10Hz（0.4 秒の履歴） | 各カメラ 4 フレーム |
| 解像度 | 1080×1920 → 320×576 にダウンサンプル | processor が自動リサイズ（`min_pixels=163840`, `max_pixels=196608`） |
| ego 状態 | 3D 並進 (x,y,z) + 3×3 回転行列、履歴 16 waypoints @10Hz | 同左（複数タイムステップ） |
| 出力 trajectory | 64 waypoints、0.1 秒刻み 6.4 秒先まで、ego 座標系 | 同左 |
| 追加出力 | CoC 推論テキスト | CoC、VQA、meta-action、2D grounding 付き auto-label |

### 4.4 実装上の forward パス

`NVlabs/alpamayo2` のコードを追うと、推論は以下の流れになる（エントリは `Alpamayo2Super.sample_trajectories_from_data`）。

```
7 カメラ
  └─ input_profiles.select_task_input        # タスクごとに 6 台 × 4 フレームを選択
      └─ helper.prepare_model_inputs         # AutoProcessor で画像+テキストを同時トークン化
          └─ token_utils.fuse_traj_tokens    # ego 履歴を離散トークン化し masked_scatter で埋込
              └─ self.vlm.model(...)          # Qwen3-VL prefill → past_key_values を得る
                  └─ self.vlm.generate(...)   # CoC テキストを自己回帰生成（top_p=0.98, T=0.6）
                      └─ ExpertModel          # 同じ KV キャッシュを条件に flow matching を 10 step
                          └─ UnicycleAccelCurvatureActionSpace.action_to_traj
                              └─ (64, 3) xyz + (64, 3, 3) 回転行列
```

重要なのは、**VLM の prefill で作った `past_key_values` を action expert がそのまま再利用する** 点である。expert は画像やテキストを再エンコードしない。実装では `expert_non_causal_attention=True` により、expert 側のトークンは互いに非因果的（全 waypoint が相互に注目できる）に、しかし VLM のキャッシュに対しては cross attention のように振る舞う。

実チェックポイントの主要ハイパーパラメータは以下の通りである。

| 構成要素 | 値 |
|---|---|
| VLM text 側 | `hidden_size=5120`, `num_hidden_layers=64`, `num_attention_heads=64`, `num_key_value_heads=8`, `intermediate_size=25600`, `vocab_size=155776` |
| VLM vision 側 | `hidden_size=1152`, `depth=27`, `patch_size=16`, `spatial_merge_size=2`, `out_hidden_size=5120` |
| Expert | `hidden_size=1536`, `num_hidden_layers=64`, `num_attention_heads=16`, `num_key_value_heads=8`, `intermediate_size=6144` |
| Action space | `n_waypoints=64`, `dt=0.1`, `accel_bounds=[-9.8, 9.8]`, `curvature_bounds=[-0.33, 0.33]` |
| 正規化統計 | `accel_mean=0.02902694707164455`, `accel_std=0.6810426736454882`, `curvature_mean=0.0002692167976330542`, `curvature_std=0.026148280660833106` |
| Diffusion | `int_method="euler"`, `train_timestep_sampler="beta"`, `num_inference_steps=10` |
| 軌道トークナイザ | `history_vocab_size=1000`, `future_vocab_size=3000`, `tokens_per_history_traj=45`, `tokens_per_future_traj=128` |

Expert が VLM と同じ 64 層でありながら `hidden_size` が 1536（VLM は 5120）である点に注意したい。**層数を揃えているのは、各層の KV キャッシュに層ごとに attend するため** である。幅だけを削ってパラメータ数を 2.3B に抑えている。

### Why this matters
Alpamayo の「効率」は主に 2 箇所から来ている。マルチカメラトークン化による入力側の圧縮と、VLM の KV キャッシュを共有する細幅 expert による出力側の圧縮である。どちらも「VLM を賢く使う」のではなく「VLM の計算結果を使い回す」タイプの設計である。

---

## 5. Action Space — unicycle 運動学で trajectory を制御量に変換する

Alpamayo は waypoint 列を直接予測しない。**加速度と曲率の系列を予測し、それを運動学モデルで積分して trajectory を得る。** これが幾何学的・運動学的な妥当性を構造的に保証する。

### 5.1 順方向（action → trajectory）

Euler 離散化した unicycle model（$\Delta T = 0.1\ \text{s}$、論文 Eq.5）：

$$\mathbf{x}^{i+1}=\begin{pmatrix}x^{i+1}\\y^{i+1}\\\theta^{i+1}\\v^{i+1}\end{pmatrix}=\begin{pmatrix}x^{i}+\frac{\Delta T}{2}\left(v^{i}\cos\theta^{i}+v^{i+1}\cos\theta^{i+1}\right)\\[2pt] y^{i}+\frac{\Delta T}{2}\left(v^{i}\sin\theta^{i}+v^{i+1}\sin\theta^{i+1}\right)\\[2pt] \theta^{i}+\Delta T\,\kappa^{i}v^{i}+\frac{\Delta T^{2}}{2}\kappa^{i}a^{i}\\[2pt] v^{i}+\Delta T\,a^{i}\end{pmatrix}$$

位置の更新が台形則（$v^i$ と $v^{i+1}$ の平均）になっている点が実装の細部として重要である。公式実装 `UnicycleAccelCurvatureActionSpace.action_to_traj` はこれを cumsum で一括計算する。

```python
# NVlabs/alpamayo2 : src/alpamayo2_super/action_space/unicycle_accel_curvature.py
velocity = torch.cat([v0.unsqueeze(-1),
                      (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1))], dim=-1)
theta = torch.cat([initial_yaw.unsqueeze(-1),
                   (initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1))], dim=-1)
x = (initial_x.unsqueeze(-1)
     + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1)
     + torch.cumsum(velocity[..., 1:]  * torch.cos(theta[..., 1:])  * half_dt_term, dim=-1))
```

`half_dt_term = 0.5 * dt` の項が 2 つの cumsum に分かれているのが台形則の実装形である。

### 5.2 逆方向（trajectory → action）

学習時には GT の trajectory から $(a^i, \kappa^i)$ を復元する必要がある。これは素朴に差分を取るとノイズが増幅するため、**Tikhonov 正則化つき最小二乗**で解かれる（`solve_xs_eq_y`）。

加速度は $\Delta v_t = \Delta T \cdot a_t$ という関係を、jerk の平滑性（2 階微分）を正則化しながら解く。曲率は

$$s^i = \Delta T\, v^i + \frac{\Delta T^{2}}{2}a^i, \qquad \kappa^i = \frac{\Delta\theta^i}{s^i}$$

を同様に解く。実装には低速時の特別扱いがあり、これが実務的に効く：

```python
_LOW_SPEED_CURVATURE_THRESHOLD_MPS = 0.6
...
low_speed = (v[..., :-1].abs() < _LOW_SPEED_CURVATURE_THRESHOLD_MPS) | \
            (v[..., 1:].abs() < _LOW_SPEED_CURVATURE_THRESHOLD_MPS)
kappa = kappa.clamp(min=self.curvature_bounds[0], max=self.curvature_bounds[1])
return torch.where(low_speed, torch.zeros_like(kappa), kappa)
```

$\kappa = \Delta\theta / s$ は $v \to 0$ で発散する。速度 0.6 m/s 未満では曲率を 0 に潰すことでこれを回避している。停車中に微小なヨー揺れが巨大な曲率ラベルになる事故を防ぐ、地味だが必須の処理である。

### 5.3 二重表現（dual representation）

Alpamayo は同じ action を **離散トークン**と**連続ベクトル**の両方で扱う。

| 表現 | 用途 | 詳細 |
|---|---|---|
| 離散（128 トークン/軌道） | VLM 本体の学習と RL | 加速度・曲率を量子化して VLM の語彙に埋め込む。`future_vocab_size=3000` |
| 連続（64×2） | 推論時のデコード | flow matching で一括生成 |

論文 Sec.5.1 が挙げる採用理由は 4 つある。

1. reasoning と trajectory が共通トークン空間で結合され、next-token prediction で causal explanation と車両挙動を密結合できる
2. 離散表現が RL post-training（GRPO）での勾配伝播を可能にする
3. 車両ダイナミクスの強い教師信号になる
4. flow-matching デコードは 128 トークンの自己回帰サンプリングより高速で real-time 推論を可能にする

### Why this matters
「trajectory を直接回帰する」代わりに「制御量を予測して積分する」だけで、物理的にありえない経路（横滑り、瞬間的なヨー変化）が構造的に排除される。学習側が払うコストは、GT trajectory から制御量を安定に逆算する前処理であり、Alpamayo はそこに正則化つき最小二乗と低速クランプという実務的な手当てを入れている。

---

## 6. Action Expert と Flow Matching

### 6.1 Flow matching の定式化

Action expert は $\pi_{0.5}$-KI に倣った flow matching で学習される。損失は（論文 Eq.6）：

$$L_{\text{cfm}}(\Theta)=\mathbb{E}_{t\sim p_{\text{schedule}},\,(\boldsymbol{o},\textsc{Reason})\sim\mathcal{D}}\left\|\mathbf{v}_{\Theta}(\boldsymbol{a}_t,\boldsymbol{o},\textsc{Reason})-\mathbf{u}(\boldsymbol{a}_t\mid\boldsymbol{a})\right\|$$

Gaussian 条件付き最適輸送（OT）パスを使うので、ノイズ付き action は線形補間で作られる。

$$\boldsymbol{a}_t = t\,\boldsymbol{a} + (1-t)\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$

そしてターゲットのベクトル場は定数になる（Eq.7）。

$$\mathbf{u}(\boldsymbol{a}_t\mid\boldsymbol{a})=\boldsymbol{a}-\boldsymbol{\epsilon}$$

これが flow matching が diffusion より実装が単純な理由である。**ノイズスケジュールを設計する必要がなく、回帰ターゲットが $\boldsymbol{a}-\boldsymbol{\epsilon}$ という定数ベクトル**になる。公式実装もそのまま素直である。

```python
# NVlabs/alpamayo2 : src/alpamayo2_super/diffusion/flow_matching.py
def construct_training_data(self, x):
    ...
    noise = torch.randn_like(x)
    return {"x": x, "noisy_x": t * x + (1 - t) * noise,
            "timesteps": t, "noise": noise, "is_drop_guidance": None}

def compute_loss_from_pred(self, training_data, pred):
    x = training_data["x"]
    noise = training_data["noise"]
    target = (x - noise).to(dtype=pred.dtype)
    return torch.nn.functional.mse_loss(target, pred)
```

推論は $\boldsymbol{a}_0\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ から Euler 積分する（Eq.8）。

$$\boldsymbol{a}_{t+\delta_t}=\boldsymbol{a}_t+\delta_t\,\mathbf{v}_\Theta(\boldsymbol{a}_t,\boldsymbol{o},\textsc{Reason})$$

```python
x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
for i in range(inference_step):
    dt = time_steps[i + 1] - time_steps[i]
    v = step_fn(x=x, t=t_start)
    x = x + dt * v
```

デフォルトは `num_inference_steps=10`（$\delta_t = 0.1$）。論文 Table 12 では $\delta_t=0.2$（5 ステップ）でも性能劣化は無視できるとされている。

### 6.2 学習時の timestep サンプラ

`train_timestep_sampler="beta"` がデフォルトで、実装は $\text{Beta}(1.5, 1.0)$ からサンプルして反転する。

```python
self.beta_dist = torch.distributions.beta.Beta(torch.tensor(1.5), torch.tensor(1.0))
self.beta_scale_constant = 0.999
...
t = self.beta_dist.sample((batch_size,)).to(x.device)
t = self.beta_scale_constant - t * self.beta_scale_constant
```

$\text{Beta}(1.5, 1.0)$ は 1 に偏った分布なので、反転すると **$t$ が 0 側（＝ノイズが多い側）に偏る**。生成が難しいノイズの強い領域を重点的に学習させる意図である。uniform サンプリングとの違いは companion notebook で可視化した。

### 6.3 Action expert の入力射影

action（64×2 の実数）を expert のトークン埋め込みに変換するのが `PerWaypointActionInProjV2` である。素直な Linear ではなく **Fourier 特徴を経由する**。

```python
class FourierEncoderV2(nn.Module):
    def __init__(self, dim, max_freq=100.0):
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freqs", freqs[None, :])
    def forward(self, x):
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
```

加速度・曲率の各次元と diffusion の timestep をそれぞれ 20 次元の Fourier 特徴に展開し、連結して 4 層の MLP（RMSNorm + SiLU）で `hidden_size=1536` に射影する。低次元のスカラー入力を高周波成分まで表現できるようにする、NeRF 由来の定石である。

### 6.4 デコード方式の比較

| Strategy | minADE$_6$@6.4s ↓ | AlpaSim (at-fault) ↑ | Comfort (Accel) ↑ | 相対デコード速度 ↑ |
|---|---|---|---|---|
| Auto-Regressive (VQGAN 離散化) | 0.6811 | 0.59±0.17 | 44.05% | 1.00× |
| **Flow Matching** | **0.6440** | **1.27±0.34** | **97.38%** | **1.16×** |

Comfort（加速度が快適域に収まる割合）が 44% → 97% と劇的に改善している。自己回帰デコードは 1 トークンずつ独立にサンプリングするため、隣接 waypoint 間の滑らかさが保証されない。flow matching は 64 waypoint を**同時に**デノイズするので、系列全体の整合性が自然に保たれる。

レイテンシの内訳も示されている（RTX 6000 Pro Blackwell）。

| Config | Vision Enc | Prefilling | Reasoning Decoding | Trajectory Decoding | Total |
|---|---|---|---|---|---|
| Baseline (traj-only, flow matching) | 3.43 ms | 16.54 ms | – | 8.75 ms (5 steps) | **29 ms** |
| Alpamayo-R1 (flow matching) | 3.43 ms | 16.54 ms | 70 ms (40 tokens) | 8.75 ms (5 steps) | **99 ms** |
| Alpamayo-R1 (auto-regressive traj) | 3.43 ms | 16.54 ms | 70 ms (40 tokens) | 222 ms (127 tokens) | **312 ms** |

reasoning を挟むコストは 70 ms（40 トークンの自己回帰生成）で、これが全体の 7 割を占める。論文が future work に「reasoning on demand（安全上重要な場面でのみ推論を起動する）」を挙げているのはこの数字が理由である。

### Why this matters
flow matching の採用は「新しい生成手法を使った」以上の意味を持つ。自己回帰デコードが 222 ms かかり快適性も損なうのに対し、8.75 ms で滑らかな軌道が出る。**リアルタイム制約のある系では、生成手法の選択がそのまま製品要件の可否を決める。**

---

## 7. 学習レシピ — 3 段階と GRPO

### Stage 1: Action Modality Injection

離散トークンで action モダリティを注入する段階。80,000 時間（米国・EU 25 か国 2,500 都市以上）のデータで、Eq.1 の系列に対し cross-entropy 損失をかける。この段階では expert の勾配が VLM に流れないよう、**KV キャッシュに stop-gradient を適用する**。

### Stage 2: SFT による reasoning の誘発

700K の CoC 付き動画セグメント（うち約 10% が人手ラベル）で SFT する。

$$\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(\boldsymbol{o},\textsc{Reason},\boldsymbol{a})\sim\mathcal{D}_{\text{CoC}}}\left[\log\pi_{\theta}(\textsc{Reason},\boldsymbol{a}\mid\boldsymbol{o})\right]$$

reasoning トークンと離散 trajectory トークン（128 トークン/軌道）の両方に損失をかける。論文は SFT 単独の限界を 4 点挙げる：データバイアス・注釈ノイズ、汎化の限界（パターン記憶に留まる）、弱い視覚グラウンディング（hallucination）、reasoning-action 不整合。

### Stage 3: GRPO による RL post-training

GRPO（Group Relative Policy Optimization）を使う。ロールアウト群 $\{\tau_i\}_{i=1}^{K}$、報酬 $r_i$、相対アドバンテージ $A_i = r_i - \bar{r}$ に対し（Eq.10）：

$$\mathcal{L}_{\text{GRPO}}(\theta)=-\mathbb{E}_{\tau_i\sim\pi_\theta}\left[\frac{\exp(\beta A_i)}{\sum_j\exp(\beta A_j)}\Big(\log\pi_\theta(\tau_i)-\lambda_{\mathrm{KL}}\,\mathrm{KL}\big[\pi_\theta(\tau_i)\,\|\,\pi_{\text{ref}}(\tau_i)\big]\Big)\right]$$

報酬は 3 項の合成である。

| 報酬項 | 定義 | 役割 |
|---|---|---|
| $r_{\text{reason}}$ | LRM 批評家（DeepSeek-R1 等）が GT と予測の reasoning trace を behavior consistency / causal quality の 2 軸で 0–5 採点 | 推論の質 |
| $r_{\text{consistency}}$ | 予測 trajectory を meta-action に変換し、reasoning から読み取った意図とルールベースで照合。一致 1 / 不一致 0 の二値 | **推論と行動の一致** |
| $r_{\text{traj}}$ | 下式 | 軌道の低レベル品質 |

$$r_{\text{traj}}=\lambda_{\text{L2}}\left\|x_{\text{pred}}-x_{\text{expert}}\right\|_2^2+\lambda_{\text{coll}}\,\mathbb{I}\big[\text{collision}(x_{\text{pred}})\big]+\lambda_{\text{jerk}}\,J(x_{\text{pred}})$$

$r_{\text{consistency}}$ が第 3 節で「閉集合の decision」にこだわった理由である。**閉集合だからこそ、予測 trajectory から機械的に meta-action を復元して照合できる。**

データキュレーションも興味深い。RL をフルデータに拡大するのは計算上不可能なため、モデルの implicit reward（logits 由来の確率分布）と外部報酬モデルが誘導する Boltzmann 分布

$$p_{\text{reward}}(\tau_i)=\frac{\exp(\beta r_i)}{\sum_j\exp(\beta r_j)}$$

の乖離が大きいサンプルを優先する。ただし多様性維持のためランダムサンプルも同程度混ぜる。

### 7.1 $r_{\text{consistency}}$ が必要な理由を実測する

「reasoning と action がずれる」というのは抽象的な懸念に聞こえるが、実際に測れる現象である。
GPU notebook で `Qwen/Qwen3-VL-2B-Instruct`（Alpamayo 2 Super と同じ `qwen3_vl` 系統）に
CoC 形式の出力を求めたところ、次の結果になった。

| 測定項目 | 結果 |
|---|---|
| 信号の色（red/green）の正答率 | **0.958**（チャンス 0.5） |
| longitudinal decision の正答率 | 0.500（常に多数派を答えるだけで 0.500） |
| 出力された longitudinal decision の種類数 | **1 / 3** |
| `BECAUSE` 節が「止まるべき」と述べた件数 | 10 |
| うち `LONGITUDINAL` が `Stop` 以外だった割合 | **100%** |

実際の出力を 1 つ引く。

```
LIGHT: red
LONGITUDINAL: Set speed tracking
LATERAL: Lane keeping & centering
BECAUSE: The traffic light is red, so the vehicle must stop, and the forward camera
         shows a lead vehicle, which is not a significant obstacle for lane keeping.
```

**赤信号を正しく認識し、「だから停止しなければならない」と自分で書いておきながら、
選んだ行動は「設定速度の維持」である。** 問題は知覚ではない（信号は 95.8% 読めている）。
知覚と行動の接続が存在しないのである。

これが論文の言う "fluent but causally disconnected explanations that fail to translate into
coherent actions" の実物であり、$r_{\text{consistency}}$ という二値報酬がわざわざ設計されている理由である。
汎用 VLM にフォーマットを指示するだけで CoC が得られるなら、
700K セグメントの構造化アノテーションも SFT も GRPO も不要だったはずである。

### Why this matters
Stage 3 の報酬設計が Alpamayo の中核的な貢献である。特に $r_{\text{consistency}}$ は「説明可能な AI」を評価可能な最適化目標に落とし込んだ例であり、次節の ablation がその効果を定量的に示している。

---

## 8. 評価結果と ablation が示すもの

### 8.1 RL 報酬の ablation — 最も示唆的な結果

| Training strategy | ADE ↓ | Reasoning Grading ↑ | Reasoning-Action Consistency ↑ | Close Encounter Rate (%) ↓ |
|---|---|---|---|---|
| SFT | 2.12 m | 3.1 | 0.62 | 6.9 |
| SFT + RL($r_{\text{reason}}$) | **2.19 m** | 4.5 | **0.53** | 5.8 |
| SFT + RL($r_{\text{reason}}+r_{\text{consistency}}$) | 1.92 m | 4.5 | **0.85** | 6.2 |
| SFT + RL(全報酬 + $r_{\text{safety}}$) | 1.94 m | 4.4 | 0.83 | **3.7** |

2 行目に注目してほしい。**reasoning 品質だけを最適化すると、ADE も consistency も SFT より悪化する。** 論文はこれを "fluent but causally disconnected explanations that fail to translate into coherent actions"（流暢だが行動に反映されない推論）と表現している。consistency 報酬を足して初めて ADE が 9.4% 改善（2.12 → 1.92 m）、consistency が 37% 改善（0.62 → 0.85）する。

### 8.2 CoC の効果（open-loop, challenging dataset, 0.5B）

| Model | minADE$_6$@3s | minADE$_6$@6.4s |
|---|---|---|
| Ft. w/ Traj. のみ | 0.315 | 0.994 |
| Ft. w/ Meta-action & Traj. | 0.301 | 0.928 |
| Ft. w/ CoC & Traj. (AR1) | **0.290** | **0.868** |

trajectory のみに対して 12% の改善。meta-action だけでも効くが、CoC がさらに上回る。

### 8.3 closed-loop（AlpaSim, 75 シナリオ）

| Model | CE 率 all (%) | CE 率 at-fault (%) | Offroad 率 (%) | AlpaSim score all |
|---|---|---|---|---|
| Baseline (traj-only) | 17.0±3.0 | 6.0±1.0 | **3.0±2.0** | 0.38±0.04 |
| Alpamayo-R1 | **11.0±2.0** | 5.0±3.0 | 4.0±3.0 | **0.50±0.08** |

close encounter 率は 35% 改善する一方、**offroad 率はわずかに悪化している**（4% vs 3%）。論文は "comparable" とするのみで踏み込んだ考察はない。誠実に読むなら、reasoning の導入が全指標を一様に改善するわけではないという注意信号である。

### 8.4 スケールと Alpamayo 2 Super

PhysicalAI-AV 公開ベンチマークでのモデルサイズの効果：

| Model | minADE$_6$@6.4s | CE 率 (%) | Offroad 率 (%) | AlpaSim Score |
|---|---|---|---|---|
| Alpamayo-R1-0.5B | 0.913 | 9.0±1.0 | 19.0±0.0 | 0.35±0.01 |
| Alpamayo-R1-10B | **0.849** | **4.0±0.0** | **16.0±1.0** | **0.72±0.02** |

Alpamayo 2 Super（34B）の主要数値は以下（NVIDIA developer blog / モデルカードより。独立した arXiv 論文は存在しない）。

| Benchmark | Metric | Alpamayo 2 Super | 比較 |
|---|---|---|---|
| PhysicalAI-AV (1,434 例) | minADE$_6$@6.4s | 0.911 m | Alpamayo 1.5 Nano: 0.916 m |
| LingoQA | Lingo-Judge | **79.2**（約 40 モデル中 1 位） | Qwen3-VL 32B: 72.2 / Gemini 2.5 Pro: 64.1 / GPT-4o: 56.0 |
| VQA (8K QA) | Answer similarity | 0.652 | Qwen3-VL 32B: 0.450 |
| 2D Grounding | IoU | 0.71 | Qwen3-VL 32B: 0.17 |
| AlpaSim (913 scenes) | AlpaSim Score | 1.50±0.13 | Alpamayo 1.5 Nano: 1.37±0.10 |

**34B が 10B に対して minADE でほぼ横ばい（0.911 vs 0.916）である点は正直に読むべきである。** 大きく効いているのは言語側のタスク（LingoQA, VQA, grounding）で、trajectory の正確さそのものではない。パラメータを 3 倍にした恩恵は「説明と理解」に出ており、「運転そのもの」には open-loop 指標上ほとんど出ていない。closed-loop の AlpaSim score では 1.37 → 1.50 と改善しているので、シナリオ全体での挙動には効いていると読める。

### Why this matters
ablation の一貫したメッセージは「**推論の質を上げるだけでは運転は良くならない。推論と行動を結び付ける制約が要る**」である。そして 34B へのスケールは、その結び付きよりも言語理解側に効いた。auto-labeler としての用途が NVIDIA 側から強調されているのは、この数字と整合している。

---

## 9. 実際に動かす — 環境要件と現実的な選択肢

### 9.1 公式の要件

| モデル | 最低 VRAM | 実測ピーク |
|---|---|---|
| Alpamayo 1 / 1.5 (10B) | 24 GB（重み 22 GB） | 単一サンプル 約 24 GB / 16 サンプル 約 40 GB / CFG 付き 16 サンプル 約 60 GB |
| Alpamayo 2 Super (34B) | H100 80GB 前提 | 6 カメラ × 4 フレームで VLM GPU 約 67 GiB、expert GPU 約 71 GiB（2 GPU 構成） |

ソフトウェアは Python 3.12、`torch==2.8.0`、`transformers==4.57.1` のハードピン、CUDA Toolkit 12.x（`flash-attn` をソースビルドする）、Linux のみ。加えて Hugging Face 上でモデルと `nvidia/PhysicalAI-Autonomous-Vehicles` データセットの両方にアクセス承認が必要である。

```bash
# NVlabs/alpamayo2 : README より
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked --dev
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
hf auth login --token "$HF_TOKEN"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m alpamayo2_super.inference_smoke \
  --model-id nvidia/Alpamayo2-Super \
  --manifest examples/validation_samples.json \
  --sample-index 0 \
  --save-viz outputs/sample0.png \
  --save-json outputs/sample0.json
```

### 9.2 手元で理解するための 2 段構成

24 GB 以上の GPU がない環境で Alpamayo を理解するには、対象を分けるのが現実的である。本レポートは 2 つの notebook でそれを行う。

| Notebook | 何が「本物」か | 何が「代用」か |
|---|---|---|
| [`alpamayo2.ipynb`](alpamayo2.ipynb)（CPU） | unicycle action space の数式・低速クランプ・flow matching の損失と Euler サンプラ・beta timestep サンプラ・Fourier 射影・公式 config の実数値（`n_waypoints=64`, `dt=0.1`, `accel_std=0.681...` 等） | VLM バックボーンを小さな Transformer に置換。画像は合成シーン |
| [`alpamayo2_gpu_smoke_kaggle.ipynb`](alpamayo2_gpu_smoke_kaggle.ipynb)（Kaggle T4） | 実モデル `Qwen/Qwen3-VL-2B-Instruct` を実 GPU にロードして CoC 形式の推論テキストを生成。その hidden state を条件に action expert を実学習 | 34B ではなく T4 に載るサイズ。Alpamayo の重みそのものではない |

GPU leg のバックボーンに `Qwen/Qwen3-VL-2B-Instruct`（約 4.26 GB）を選んだのには理由がある。Alpamayo 2 Super の `vlm_config.model_type` は `qwen3_vl` であり、**同じアーキテクチャ系統の最小メンバー**だからである。`Qwen3-VL-8B-Instruct` は重みだけで約 17.5 GB あり T4 の 16 GB には収まらない。

CPU 側で「数式とアルゴリズム」を、GPU 側で「実 VLM の hidden state を条件にした expert が本当に学習するか」を確認する分担である。**Alpamayo の重みそのものは T4 には載らない**（10B で 22 GB）ため、そこは代用であることを明記している。

### Why this matters
公開されたモデルを「動かす」と「理解する」は別の作業である。34B の重みを H100 で回してもアーキテクチャの理解は進まない一方、`action_to_traj` の 10 行を自分で書き直すと、なぜ台形則なのか・なぜ低速クランプが要るのかが体感でわかる。本レポートは後者に重心を置いている。

---

## 10. Common Pitfalls

**曲率の逆算で $v \to 0$ の発散を踏む。** $\kappa = \Delta\theta / s,\ s = \Delta T v + \frac{\Delta T^2}{2}a$ は停車時に $s \to 0$ となり爆発する。公式実装は 0.6 m/s 未満で $\kappa=0$ に潰し、さらに `curvature_bounds=[-0.33, 0.33]` でクランプしている。自前で action space を書くとき最初に踏むバグである。

**bf16 対応の判定を `torch.cuda.is_bf16_supported()` で行う。** これは Ampere 以前の GPU でもソフトウェアエミュレーションのため `True` を返す。T4 (sm_75) で bf16 を選ぶと数値が壊れる。`torch.cuda.get_device_capability() >= (8, 0)` で判定すること。GPU notebook ではこれを assert で担保している。

**`flash-attn` が入らない環境で公式コードをそのまま動かそうとする。** `pyproject.toml` は `flash-attn>=2.8.3` を要求し、これは sm_80 以上（Ampere 以降）でしか動かない。T4 や古い GPU では `attn_implementation="sdpa"` への差し替えが必須である。

**`transformers` のバージョンずれ。** 公式は `transformers==4.57.1` にハードピンしているが、公開チェックポイントの `config.json` に記録された `transformers_version` は `4.57.6` である。Qwen3-VL のクラス定義は `transformers` 本体側にあるため、バージョンがずれると `vlm_class` の動的ロードで失敗しうる。

**reasoning 品質だけを最適化して満足する。** 第 8.1 節の ablation の通り、$r_{\text{reason}}$ 単独では ADE も reasoning-action consistency も SFT より悪化する。「もっともらしい説明を出すモデル」は「良い運転をするモデル」ではない。評価指標に必ず consistency を入れること。第 7.1 節の実測では、赤信号を正しく認識し「停止すべき」と書いたケースの 100% で、実際には別の行動を選んでいた。

**出力の「形式」を検査して検証したつもりになる。** GPU notebook の最初の版は「`LONGITUDINAL` と `LATERAL` を含むか」「閉集合の語彙が現れるか」を assert していたが、モデルがプロンプトのプレースホルダ記法 `<...>` をそのまま複写し、全サンプルで同一の答えを返していても緑になった。検証すべきは形式ではなく、**答えが入力によって変わるか**である。多数派クラスを常に答えるベースラインと比較すること。

**free-form CoT のアノテーションで未来を漏らす。** 動画全体を見せてラベリングすると causal confusion が入り、モデルは未来を透視する前提を学習する。CoC の 2 段階ラベリング（先に履歴窓だけで証拠を確定させる）はこの対策である。自前でデータを作るときも同じ順序を守る必要がある。

**モデルサイズを上げれば運転が良くなると期待する。** 34B の Alpamayo 2 Super は 10B の 1.5 Nano に対し minADE でほぼ横ばい（0.911 vs 0.916）である。スケールが効いたのは言語理解側であり、auto-labeler としての価値がそこにある。

**「gated」と「商用利用制限」を混同する。** Alpamayo のモデル 3 種は HF 上で gated ではなく無認証でダウンロードできる。逆に**データセット `nvidia/PhysicalAI-Autonomous-Vehicles` は gated（自動承認）** で、ライセンス同意とアクセストークンが必要。公式の `inference_smoke.py` はこのデータセットを引きに行くため、モデルだけ落として実行しようとすると認証で止まる。なおライセンス条件は途中で変わっている（第 1.1 節）ので、古い記事の「非商用限定」という記述を鵜呑みにしないこと。

**論文に書いてある機能が、公開された重みに入っていると思い込む。** Alpamayo 1 の README は paper と release の対応表を持っており、RL post-trained weights / route・navigation conditioning / meta-actions・VQA はいずれも `❌ Not in this release` と明記されている。本レポート第 7 節で解説した 3 段階レシピのうち **Stage 3 (GRPO) の成果物が実際に重みに入っているのは 1.5 以降**である。論文を読んで実装を触るときは、必ずリリースノート側の対応表を確認すること。

---

## 11. References

- Yan Wang et al., "Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail," arXiv:2511.00088 (2025). https://arxiv.org/abs/2511.00088
- NVIDIA, "Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning," arXiv:2503.15558 (2025). https://arxiv.org/abs/2503.15558
- Physical Intelligence et al., "$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization," arXiv:2504.16054 (2025). https://arxiv.org/abs/2504.16054
- Lipman et al., "Flow Matching for Generative Modeling," arXiv:2210.02747 (2022). https://arxiv.org/abs/2210.02747
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"（GRPO の原典）, arXiv:2402.03300 (2024). https://arxiv.org/abs/2402.03300
- NVlabs, Alpamayo 2 Super 公式実装. https://github.com/NVlabs/alpamayo2
- NVlabs, Alpamayo 1.5 Nano 公式実装（世代間の Key Features 比較表あり）. https://github.com/NVlabs/alpamayo1.5
- NVlabs, Alpamayo 1 Nano 公式実装（paper と release の対応表あり、deprecated）. https://github.com/NVlabs/alpamayo
- NVlabs, Alpamayo recipes（SFT / RL / 量子化）. https://github.com/NVlabs/alpamayo-recipes
- NVlabs, AlpaSim（closed-loop シミュレータ）. https://github.com/NVlabs/alpasim
- NVlabs, AlpaGym（closed-loop RL フレームワーク）. https://github.com/NVlabs/alpagym
- Hugging Face, `nvidia/Alpamayo-1.5-10B`. https://huggingface.co/nvidia/Alpamayo-1.5-10B
- Hugging Face, `nvidia/Alpamayo-R1-10B`. https://huggingface.co/nvidia/Alpamayo-R1-10B
- Hugging Face, `nvidia/Cosmos-Reason2-8B`（Alpamayo 1.5 のバックボーン）. https://huggingface.co/nvidia/Cosmos-Reason2-8B
- NVIDIA, "Alpamayo 2 Super, the Frontier Open Model for Robotaxis and Autonomous Vehicles, Now Available for Commercial Use," NVIDIA Blog (2026-08-04). https://blogs.nvidia.com/blog/alpamayo-2-super-open-model-now-available/
- NVIDIA, "Generate Trajectories, Reasoning Traces, and Auto-Labels with NVIDIA Alpamayo 2 Super," NVIDIA Developer Blog. https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/
- Hugging Face, `nvidia/Alpamayo2-Super`. https://huggingface.co/nvidia/Alpamayo2-Super
- Hugging Face, `nvidia/PhysicalAI-Autonomous-Vehicles`. https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
