# LoRA / QLoRA 完全解説

> コード参照:
> - `peft` (HuggingFace PEFT): `src/peft/tuners/lora/layer.py`, `src/peft/tuners/lora/model.py`, `src/peft/tuners/lora/config.py`
> - `bitsandbytes`: `bitsandbytes/nn/modules.py` (`Linear4bit`, `Params4bit`), `bitsandbytes/functional.py` (`quantize_4bit`, `dequantize_4bit`)
> - `transformers` (HuggingFace): `src/transformers/integrations/bitsandbytes.py`, `src/transformers/quantizers/quantizer_bnb_4bit.py`
> - 原論文: LoRA (Hu et al., 2021, arXiv:2106.09685) / QLoRA (Dettmers et al., 2023, arXiv:2305.14314)

---

## 0. この資料の読み方

LoRA と QLoRA は「巨大な事前学習済みモデルを、少ないメモリ・計算で微調整（fine-tuning）する」ための技術。
- **LoRA** = 重み更新を**低ランク行列**で近似する手法（メモリ削減の本体）
- **QLoRA** = ベースモデルを**4bit量子化**して固定し、その上に LoRA を載せる手法（さらにメモリ削減）

理解の順序は **Full FT のメモリ内訳 → LoRA → 量子化 → QLoRA**。
各章末に「手を動かす」ポイントを示すので、`lora_qlora_demo.ipynb`（numpyだけで今すぐ動く）と `lora_qlora_finetune.ipynb`（GPUで実学習）を併用すること。

---

## 1. なぜ PEFT が必要か — Full Fine-Tuning のメモリ問題

7B モデル（70億パラメータ）を **bfloat16 + Adam** でフル微調整する場合のGPUメモリ内訳：

| 項目 | 係数 | 7B での量 | 説明 |
|---|---|---|---|
| モデル重み | 2 bytes/param | 14 GB | bf16 で保持 |
| 勾配 (gradient) | 2 bytes/param | 14 GB | 各パラメータに1つ |
| Adam: 1次モーメント `m` | 4 bytes/param | 28 GB | fp32 で保持 |
| Adam: 2次モーメント `v` | 4 bytes/param | 28 GB | fp32 で保持 |
| (mixed precisionのfp32マスター重み) | 4 bytes/param | 28 GB | 実装による |
| **合計（活性化を除く）** | | **約 84〜112 GB** | |

ポイント：**重み本体は14GBなのに、optimizer state（Adamの m, v）が圧倒的に重い**。
A100 80GB 1枚でも 7B のフル微調整は厳しく、活性化メモリを足すと複数GPUが必要になる。

> **正しい努力の視点**：fine-tuning のボトルネックは「重みの大きさ」ではなく「学習可能パラメータに付随する optimizer state と gradient」。ここを削るのが PEFT の発想。

**PEFT (Parameter-Efficient Fine-Tuning)** = ベースの重みは凍結（freeze）し、ごく一部の追加パラメータだけを学習する。
学習対象が 0.1〜1% になれば、gradient と optimizer state もその分しか要らない。

→ 手を動かす: `demo` の「メモリ内訳の棒グラフ」セル

---

## 2. LoRA の原理 — 重み更新を低ランク分解する

### 2.1 中心アイデア

微調整とは、事前学習済み重み `W₀ ∈ ℝ^{d×k}` を `W₀ + ΔW` に更新すること。
LoRA の仮説（原論文）：**ΔW は「低い内在ランク（intrinsic rank）」を持つ** → 低ランク行列で十分近似できる。

```
ΔW = B · A

  B ∈ ℝ^{d×r}   （down→up の up 側）
  A ∈ ℝ^{r×k}   （down 側）
  r ≪ min(d, k)  （ランク。typ. r = 8, 16, 32, 64）
```

元の forward：
```
h = W₀ · x
```
LoRA 適用後の forward：
```
h = W₀ · x + (α/r) · B · (A · x)
        └ 凍結 ┘   └─ 学習対象（B, A のみ）─┘
```

- `W₀` は**凍結**（勾配を流さない）
- 学習するのは `A`, `B` だけ
- `α`（lora_alpha）は**スケーリング係数**。実効的な学習率を調整する

### 2.2 パラメータ数の比較

`d = k = 4096`（7Bの典型的な隠れ次元）の Linear 1層について：

| 方式 | パラメータ数 | 4096×4096 での値 |
|---|---|---|
| Full | d × k | 16,777,216 (約1678万) |
| LoRA (r=8) | r × (d + k) | 65,536 (約6.5万) |
| LoRA (r=16) | r × (d + k) | 131,072 (約13万) |

r=8 なら **0.39%** まで学習パラメータが減る。

### 2.3 初期化の妙：なぜ学習開始時に ΔW = 0 か

```
A ~ N(0, σ²)  （ガウス初期化、またはKaiming）
B = 0          （ゼロ初期化）
→ ΔW = B·A = 0  （学習開始時）
```

`B = 0` で始めるので、**学習開始時点では元モデルと完全に同じ出力**になる。
これにより「微調整の最初で性能が壊れる」ことを防ぎ、安定して `W₀` から出発できる。

> peft コード: `src/peft/tuners/lora/layer.py` の `reset_lora_parameters()`。
> `lora_A` は `kaiming_uniform_`、`lora_B` は `zeros_` で初期化される。

### 2.4 α / r スケーリングの意味

```
実効スケール = α / r
```
論文では「`α` を `r` と同じにして固定し、`r` を変えてもスケールを保つ」運用が多い（例: r=8, α=16 → スケール2）。
- α を上げる = ΔW の寄与を強める（学習率を上げるのに近い）
- **rsLoRA**（rank-stabilized LoRA）では `α/√r` を使い、高ランクでも安定させる

→ 手を動かす: `demo` の「低ランク近似で行列を復元」「パラメータ数カウント」セル

---

## 3. LoRA をどの層に適用するか

`target_modules` で指定する。Transformer の Attention/MLP の Linear 層が対象。

| 対象 | module名（LLaMA系） | 効果 |
|---|---|---|
| Query/Value 射影 | `q_proj`, `v_proj` | 原論文の最小構成。コスパ良 |
| Attention全体 | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 表現力UP |
| MLP も含む | + `gate_proj`, `up_proj`, `down_proj` | QLoRA論文推奨。全Linearが鉄板 |

QLoRA 論文の重要な発見：**「すべての Linear 層に LoRA を当てる」のが性能面で最も効く**（Attentionだけより良い）。
迷ったら全 Linear（`all-linear`）。peft では `target_modules="all-linear"` で自動指定可能。

```python
from peft import LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",   # 全Linear層
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

---

## 4. LoRA のメモリ削減はどこから来るか

学習可能パラメータが 0.4% になると、それに紐づくものが全部削れる：

| 項目 | Full FT (7B) | LoRA (7B, 約0.4%) |
|---|---|---|
| モデル重み（凍結） | 14 GB | 14 GB（変わらない） |
| 勾配 | 14 GB | **~0.06 GB** |
| Adam m, v | 56 GB | **~0.24 GB** |
| 合計（活性化除く） | 約84 GB | **約14.3 GB** |

**重みそのものは減らない**点に注意。LoRA が削るのは gradient と optimizer state。
→ だから「重み14GBをさらに削りたい」という次の動機が QLoRA（4bit量子化）につながる。

→ 手を動かす: `demo` の「Full vs LoRA vs QLoRA メモリ比較」セル

---

## 5. 量子化の基礎 — QLoRA の前提知識

QLoRA の "Q" は4bit量子化。まず一般的な量子化を押さえる。

### 5.1 量子化とは

fp16（2バイト=16bit）の重みを 4bit に圧縮する → メモリ 1/4。
基本は「実数の範囲を離散的な格子点に丸める」：

```
量子化:   q = round(x / scale)        scale = absmax / (2^(bits-1) - 1)
逆量子化: x̂ = q × scale
```

`scale`（量子化定数）をブロックごとに持つのが **block-wise quantization**（外れ値の影響を局所化）。

### 5.2 NF4 (NormalFloat4) — QLoRA の核心その1

通常の int4 は等間隔の格子。しかし**ニューラルネットの重みは平均0の正規分布に近い**。
→ 等間隔ではなく「**正規分布の分位点（quantile）に合わせた非等間隔の格子**」を使うと情報量的に最適。これが NF4。

```
NF4 の 16 個の格子点は、標準正規分布 N(0,1) の分位点から導出される（0を含む対称な16値）。
重みブロックを absmax で [-1, 1] に正規化 → 最も近い NF4 格子点の index（4bit）を保存。
```

- 「情報理論的に最適（information-theoretically optimal）」と論文が主張するのは、入力が正規分布のとき各ビンに均等にデータが入るから。
- int4 より同じ4bitで高精度。

> bitsandbytes コード: `bitsandbytes/functional.py` の `create_normal_map()` が NF4 の格子を生成。
> `quantize_4bit(x, quant_type="nf4")` / `dequantize_4bit(...)`。

### 5.3 Double Quantization (DQ) — 核心その2

block-wise だと、各ブロックに `scale`（fp32, 4バイト）が必要。ブロックサイズ64なら：
```
scale のオーバーヘッド = 32 bit / 64 params = 0.5 bit/param
```
これも無視できない。**DQ は「量子化定数そのものをさらに量子化」**して 8bit に圧縮：
```
0.5 bit/param → 約 0.127 bit/param（論文値）
→ パラメータあたり約 0.37 bit の節約
```

### 5.4 Paged Optimizers — 核心その3

学習中のメモリスパイク（長いシーケンス等）で OOM するのを防ぐため、
NVIDIA Unified Memory を使って optimizer state を必要に応じて CPU↔GPU で**ページング**（OS の仮想メモリのように退避）。

→ 手を動かす: `demo` の「NF4量子化シミュレーション」「int4 vs NF4 の誤差比較」セル

---

## 6. QLoRA — 全部を組み合わせる

### 6.1 構造

```
                  ┌─────────────── 4bit (NF4, 凍結) ───────────────┐
入力 x ──────────► W₀(4bit) を fp16/bf16 に逐次デクオンタイズして行列積
   │              └────────────────────────────────────────────────┘
   │                                                          │
   └──► LoRA: A(16bit) ─► B(16bit) ──(α/r)──────────────────► (+)──► h
              └─ 学習対象（fp16/bf16）─┘
```

ポイント：
1. **ベース重み W₀ は NF4 で保存（凍結）** → 重み本体が 14GB → 約 3.5GB に
2. forward/backward の計算時は、必要なブロックだけ **fp16 にデクオンタイズして** matmul（計算精度は16bit）
3. **勾配は LoRA の A, B にだけ流れる**（W₀ は凍結なので勾配不要）
4. LoRA パラメータは 16bit、optimizer は paged

### 6.2 メモリ比較（7B モデル）

| 方式 | 重み | 勾配 | Adam state | 合計(活性化除く) | 必要GPU目安 |
|---|---|---|---|---|---|
| Full FT (bf16) | 14 GB | 14 GB | 56 GB | **~84 GB** | A100 80GB×複数 |
| LoRA (bf16) | 14 GB | 0.06 GB | 0.24 GB | **~14.3 GB** | RTX 4090 24GB |
| **QLoRA (NF4)** | 3.5 GB | 0.06 GB | 0.24 GB | **~3.8 GB** + 活性化 | **RTX 3060 12GB / Colab T4** |

QLoRA 論文の代表的成果：**65B モデルを単一 48GB GPU で微調整**（Guanaco）。
これは「数百GB必要だったものが1枚で回る」という破壊的な改善。

### 6.3 精度は落ちないのか

QLoRA 論文の主張：**16bit フル微調整と同等の性能を 4bit ベース + LoRA で達成**。
理由：
- NF4 が重み分布に最適化されている
- 計算（matmul）自体は 16bit で行う（保存だけ4bit）
- LoRA が量子化誤差を学習で吸収する

---

## 7. PEFT / bitsandbytes コード解析

### 7.1 4bit ロード（transformers + bitsandbytes）

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4を使う（§5.2）
    bnb_4bit_use_double_quant=True,     # Double Quantization（§5.3）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 計算時のdtype（§6.1-2）
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", quantization_config=bnb_config, device_map="auto",
)
```

内部の流れ（`transformers/integrations/bitsandbytes.py`）：
- `nn.Linear` を `bnb.nn.Linear4bit` に置換
- 重みは `Params4bit` として NF4 量子化されGPUへ
- forward 時に `Linear4bit.forward()` が `dequantize_4bit` → matmul

### 7.2 LoRA を載せる（peft）

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)  # 量子化モデルの学習準備
#  └ layernormをfp32化、入力にrequires_grad、gradient checkpoint有効化など

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~10M || all params: ~1.2B || trainable%: ~0.8%
```

`peft/tuners/lora/layer.py` の `Linear.forward()` 抜粋イメージ：
```python
result = self.base_layer(x)                 # 凍結された(4bit)ベース
for active_adapter in self.active_adapters:
    lora_A = self.lora_A[active_adapter]
    lora_B = self.lora_B[active_adapter]
    scaling = self.scaling[active_adapter]  # = alpha / r
    x_ = dropout(x)
    result = result + lora_B(lora_A(x_)) * scaling
return result
```

---

## 8. ハイパーパラメータ実践ガイド

| パラメータ | 推奨初期値 | 効果と勘所 |
|---|---|---|
| `r`（rank） | 16（8〜64） | 大きいほど表現力↑だがメモリ・過学習リスク↑。まず16 |
| `lora_alpha` | r の2倍（=32） | スケール α/r。rと連動させる運用が定番 |
| `lora_dropout` | 0.05〜0.1 | 過学習防止。小データほど大きめ |
| `target_modules` | `all-linear` | QLoRA論文の推奨。全Linearが最も効く |
| `learning_rate` | 1e-4〜2e-4 | Full FTより**1桁高め**でOK（パラメータが少ないため） |
| `bias` | `"none"` | biasは通常学習しない |
| optimizer | `paged_adamw_8bit` | QLoRA定番。8bit Adam + ページング |

経験則：
- **r と alpha は連動**（alpha = 2r）させると lr の再調整が楽
- データが少ない（数百〜数千件）なら r を小さく、dropout を大きく
- うまくいかないときは「target_modules を広げる」が「r を上げる」より効くことが多い

---

## 9. 学習後の推論 — マージとアダプタ切り替え

### 9.1 merge（本番デプロイ用）

```python
merged = model.merge_and_unload()   # W₀ + (α/r)BA を計算して単一の重みに焼き込む
merged.save_pretrained("./merged-model")
```
- `merge_and_unload()` は `W = W₀ + scaling·B·A` を実体化 → LoRA分岐が消え、**推論時の追加レイテンシがゼロ**になる
- 注意：**4bit量子化済みベースにそのままマージはできない**（一度fp16で再ロードしてマージするのが安全）

### 9.2 アダプタの付け替え（マルチタスク）

```python
model.load_adapter("./adapter-task-A", adapter_name="A")
model.load_adapter("./adapter-task-B", adapter_name="B")
model.set_adapter("A")   # 実行時に切り替え
```
1つのベース（大きい）+ 複数の小さなアダプタ、で多タスクを安く運用できる。
これが LoRA の運用上の強力な利点（S-LoRA等のサービングはこれを大規模化したもの）。

---

## 10. LoRA の派生・発展（2024〜）

| 手法 | 一言 | いつ使う |
|---|---|---|
| **QLoRA** | 4bit NF4 + LoRA | メモリ最優先。本資料の主役 |
| **DoRA** | 重みを「大きさ」と「方向」に分解しLoRA適用 | LoRAより高品質。peftで`use_dora=True` |
| **rsLoRA** | スケールを α/√r に | 高ランク(r≥64)で安定 |
| **LoRA+** | A と B で学習率を変える（B側を高く） | 収束を速くしたい |
| **VeRA** | 共有ランダム行列＋小さなスケーリングベクトル | 究極の省パラメータ |
| **PiSSA** | 主特異成分でA,Bを初期化 | 収束・性能改善 |

まず QLoRA を完全に手の内に入れ、次に DoRA を試す、の順がおすすめ。

---

## 11. 落とし穴・実践tips

1. **`prepare_model_for_kbit_training` を忘れない** — 量子化モデルにLoRAを載せる前に必須。忘れると勾配が流れず学習が進まない。
2. **`compute_dtype` は bf16 推奨**（Ampere以降）。T4等の古いGPUは fp16。
3. **gradient checkpointing と併用**するとさらに活性化メモリ削減（速度とのトレードオフ）。
4. **学習率はFull FTより高め**（1e-4〜2e-4）。低すぎると全然動かない。
5. **`target_modules`の指定ミス**で「trainable% が極端に小さい/0」になりがち → `print_trainable_parameters()` で必ず確認。
6. **4bitベースに直接mergeしない** — マージはfp16ベースで。量子化したまま焼くと精度劣化。
7. **eval時はdropoutオフ**（`model.eval()`）を忘れずに。
8. **保存されるのはアダプタだけ**（数十MB）。ベースは別管理。再現には「ベースモデル名＋アダプタ」の両方が必要。

---

## 12. まとめ — 1枚で振り返る

```
Full FT:   重み14 + 勾配14 + Adam56 = 84GB   （optimizer stateが主犯）
   │
   ├─ LoRA:   学習を0.4%に → 勾配・Adamがほぼ消滅 → 14.3GB
   │            （でも重み14GBは残る）
   │
   └─ QLoRA:  重みをNF4 4bitに → 3.5GB + LoRA → 3.8GB
                ・NF4: 正規分布の分位点格子（情報理論的に最適）
                ・Double Quant: 量子化定数も量子化
                ・Paged Optimizer: OOM回避
                ・計算は16bit / 保存は4bit / 勾配はLoRAのみ
```

**学習ロードマップ**：
1. `lora_qlora_demo.ipynb` で「低ランク分解」「メモリ計算」「NF4量子化」を numpy で体感（この環境で今すぐ動く）
2. `lora_qlora_finetune.ipynb` で Colab/GPU 上で実際に小型LLMを QLoRA 微調整
3. r/alpha/target_modules を変えて挙動の差を観察
4. DoRA・rsLoRA に展開

---

## 参考文献

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314
- Dettmers et al., "8-bit Optimizers via Block-wise Quantization", arXiv:2110.02861
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", arXiv:2402.09353
- HuggingFace PEFT docs: https://huggingface.co/docs/peft
