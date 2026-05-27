# KV Cache 完全解説

> コード参照:
> - `transformers` (HuggingFace): `src/transformers/cache_utils.py`
> - `vllm` (vLLM project): `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`, `vllm/model_executor/layers/quantization/kv_cache.py`

---

## 1. そもそもなぜ KV Cache が必要か

Transformer の Self-Attention は次のように計算される：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q = X·Wq   shape: [seq_len, d_model]
K = X·Wk   shape: [seq_len, d_model]
V = X·Wv   shape: [seq_len, d_model]
```

LLM の **Autoregressive デコード** では、各ステップで新しいトークン 1 個を生成する。  
このとき、過去トークン全部に対して Q·Kᵀ を計算しなおすのは非常に非効率：

```
ステップ t で新トークン生成:
  新 Q(t) = x(t) · Wq          (新しいトークン 1 個分だけ)
  K(0..t) = [x(0)..x(t)] · Wk  (過去 t+1 個分全部 → 毎回再計算?)
  V(0..t) = [x(0)..x(t)] · Wv  (過去 t+1 個分全部 → 毎回再計算?)
```

**KV Cache** は K・V を一度計算したらメモリに保存しておき、再計算をスキップする仕組み。

```
ステップ t-1 の後: K(0..t-1), V(0..t-1) をキャッシュ済み
ステップ t では:
  K(t) = x(t) · Wk  ← 1 トークン分だけ新規計算
  K(0..t) = concat(K_cache, K(t))  ← キャッシュに append
```

### 計算量の比較

| 方式 | ステップあたり計算量 | N ステップの総計算量 |
|---|---|---|
| KV Cache なし | O(t²·d) | O(N³·d) |
| KV Cache あり | O(t·d) | O(N²·d) |

---

## 2. KV Cache のメモリサイズ

### 基本計算式

1 レイヤーあたりの KV cache サイズ：

```
bytes_per_layer = 2             (K と V)
               × seq_len        (トークン数)
               × num_kv_heads   (KV のヘッド数)
               × head_dim       (ヘッドの次元)
               × dtype_bytes    (float16=2, bfloat16=2, float32=4)
```

vLLM の `FullAttentionSpec.real_page_size_bytes` は以下で計算している
（`vllm/v1/kv_cache_interface.py` L170–175）：

```python
return (
    2              # K + V
    * block_size   # ブロック内トークン数
    * num_kv_heads
    * head_size
    * get_dtype_size(self.dtype)
)
```

### 具体例：LLaMA-3 70B

| パラメータ | 値 |
|---|---|
| レイヤー数 | 80 |
| KV ヘッド数（GQA） | 8 |
| ヘッド次元 | 128 |
| dtype | bfloat16（2 bytes） |

```
1 トークンあたりのキャッシュサイズ:
  2 × 8 × 128 × 2 × 80 = 327,680 bytes ≈ 320 KB

4096 トークンのシーケンス:
  320 KB × 4096 = 1.25 GB / リクエスト

バッチサイズ 8:
  1.25 GB × 8 = 10 GB
```

---

## 3. HuggingFace Transformers の実装

`transformers/cache_utils.py` では `Cache` → `CacheLayerMixin` の2層構造でキャッシュを管理する。

### 3-1. DynamicLayer（デフォルト）

デコード中に動的に grow するシンプルな実装（L109–188）：

```python
class DynamicLayer(CacheLayerMixin):
    """grows dynamically as more tokens are generated"""

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.keys  = torch.tensor([], dtype=...)
            self.values = torch.tensor([], dtype=...)
            self.is_initialized = True

        self.keys   = torch.cat([self.keys,   key_states],   dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
```

`torch.cat` を毎ステップ呼ぶため、シーケンスが長くなるほど **メモリ再確保コスト** が増える。

### 3-2. StaticLayer（推論最適化）

最大長を事前に確保し、スライスで更新する（L277–）：

```python
class StaticLayer(CacheLayerMixin):
    # 最初に [batch, heads, max_len, head_dim] を一括確保
    # update のたびに self.keys[:, :, cache_position] = key_states
```

`torch.cat` がなく、メモリ再確保不要。`torch.compile` との相性が良い。

### 3-3. DynamicSlidingWindowLayer

Mistral / Gemma 等のスライディングウィンドウ Attention で使用。  
直近 `sliding_window` トークン分のみを保持し、古い KV を自動的に破棄する（L190–275）。

### 3-4. QuantizedLayer（KIVI ペーパー準拠）

最新の `residual_length` トークン分だけ高精度で保持し、それ以降は量子化して圧縮する（L514–587）：

```python
class QuantizedLayer(DynamicLayer):
    """residual_length 分だけ高精度、それ以前は量子化"""
    nbits           = 4      # 量子化ビット数
    q_group_size    = 64     # グループ量子化のグループサイズ
    residual_length = 128    # 高精度保持トークン数

    def update(self, key_states, value_states, ...):
        # 高精度バッファが満杯になったら量子化キャッシュに移行
        if self.keys.shape[-2] + 1 >= self.residual_length:
            self._quantized_keys   = self._quantize(keys_to_return, axis=self.axis_key)
            self._quantized_values = self._quantize(values_to_return, axis=self.axis_value)
            self.keys   = torch.tensor([])  # 高精度バッファをリセット
            self.values = torch.tensor([])
```

### 3-5. Cache（コンテナ）

各レイヤーを束ねて CPU オフロードを制御するコンテナ（L890–998）：

```python
class Cache:
    layers: list[CacheLayerMixin]

    def update(self, key_states, value_states, layer_idx, ...):
        # CPU オフロード時は非デフォルトストリームで prefetch
        if self.offloading:
            torch.cuda.default_stream().wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1)

        keys, values = self.layers[layer_idx].update(key_states, value_states)

        if self.offloading:
            self.offload(layer_idx)  # GPU → CPU に追い出す
        return keys, values
```

---

## 4. vLLM PagedAttention

`transformers` の `DynamicLayer` がバッチごとに連続メモリを確保するのに対し、  
vLLM は **物理メモリをページ（ブロック）単位で管理** することで、メモリ断片化を解消する。

### 4-1. KVCacheBlock（vllm/v1/core/kv_cache_utils.py L116–163）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 0 〜 num_gpu_blocks-1
    ref_cnt:  int = 0      # 参照カウント（共有時に >1 になる）
    _block_hash: BlockHashWithGroupId | None = None  # プレフィックスキャッシュ用ハッシュ

    # 双方向リンクリストのポインタ（FreeKVCacheBlockQueue 専用）
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

### 4-2. FreeKVCacheBlockQueue（vllm/v1/core/kv_cache_utils.py L164–）

空きブロックを **O(1) で操作できる双方向リンクリスト** で管理する。  
Python の `deque` は C++ 実装だが、中間要素の削除が O(N)。  
この実装はブロック自体にポインタを持たせることで O(1) を達成している。

```
eviction 順序（LRU）:
  先頭 = 最長未使用ブロック（次に追い出される候補）
  末尾 = 最近使われたブロック

フリーリストに戻す際:
  リクエスト内の複数ブロックは逆順で末尾に追加
  → チェーンの末尾（ブロック列の先頭）が先に evict される
```

### 4-3. KV cache のブロックレイアウト（FlashAttention バックエンド）

`vllm/v1/attention/backends/flash_attn.py` L149：

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    # block_size は 16 の倍数必須
    return (num_blocks, 2, block_size, num_kv_heads, head_size)
    #        ^^^^^^^^^^^  ^  ^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^
    #        全ブロック数  K/V  トークン数   ヘッド数     ヘッド次元
```

物理的に **non-contiguous なブロック** を論理的に 1 つのシーケンスとして扱うため、  
`block_tables`（論理ブロック ID → 物理ブロック ID のマッピング）が別途管理される。

### 4-4. PagedAttention の利点

```
通常のキャッシュ（連続割り当て）:
  リクエスト A: [K0 K1 K2 K3 ... Kmax] 最大長分を事前確保
                → 短いリクエストは後半が無駄（内部フラグメンテーション）

PagedAttention（ブロック割り当て）:
  ブロックサイズ=16 として:
  リクエスト A (23 tokens): [block7][block12][block3(7tokens)]
  リクエスト B (8 tokens):  [block1]
  → 実際に使う分だけ確保。空きブロックは他リクエストが利用可能
```

| 指標 | 通常割り当て | PagedAttention |
|---|---|---|
| メモリ断片化 | 最大長分を事前確保 → 無駄が大きい | ブロック単位の割り当て → 無駄 <5% |
| 最大バッチサイズ | 小さい | 大幅向上（論文で最大 24倍のスループット） |
| 実装コスト | 低い | 高い（block_tables 管理が必要） |

---

## 5. プレフィックスキャッシング（Prefix Caching）

同じプレフィックス（システムプロンプト等）を持つ複数リクエストで KV を共有する技術。

### 5-1. ハッシュによる一致検出（vllm/v1/core/kv_cache_utils.py L541–570）

```python
def hash_block_tokens(
    hash_function,
    parent_block_hash: BlockHash | None,  # 直前ブロックのハッシュ（連鎖）
    curr_block_token_ids: Sequence[int],   # このブロックのトークン ID 列
    extra_keys: tuple | None = None,
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # 乱数シード（衝突回避）

    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

**ハッシュのチェーン構造：**

```
Block 0: hash(NONE_HASH, tokens[0:16])      = H0
Block 1: hash(H0,        tokens[16:32])     = H1
Block 2: hash(H1,        tokens[32:48])     = H2
...
```

各ブロックのハッシュは**過去全トークン列の情報を含む**ため、  
同じトークン列を持つブロックは必ず同じハッシュ値になる。  
（これにより `KVCacheBlock.ref_cnt > 1` で複数リクエストがブロックを共有可能）

### 5-2. LRU 退避

キャッシュが満杯になると `FreeKVCacheBlockQueue` の先頭（LRU）から退避。  
`ref_cnt > 0`（使用中）のブロックは退避されない。

---

## 6. アテンション機構とヘッド数最適化

### 6-1. MHA / MQA / GQA の比較

```
MHA（Multi-Head Attention）:  num_kv_heads = num_q_heads
  Q heads: [H0 H1 H2 H3]
  K heads: [H0 H1 H2 H3]  ← Q と同数、KV キャッシュが大きい
  V heads: [H0 H1 H2 H3]

MQA（Multi-Query Attention）: num_kv_heads = 1
  Q heads: [H0 H1 H2 H3]
  K heads: [H0]             ← 全 Q が単一の KV を共有
  V heads: [H0]             ← KV キャッシュが 1/num_heads

GQA（Grouped-Query Attention）: 1 < num_kv_heads < num_q_heads
  Q heads: [H0 H1 H2 H3]
  K heads: [G0     G1   ]   ← 2グループで共有（LLaMA-3 70B は Q=64, KV=8）
  V heads: [G0     G1   ]
```

| 手法 | KV サイズ | 精度 | 採用モデル |
|---|---|---|---|
| MHA | 基準(1x) | 高 | GPT-2、BERT、初期 LLaMA |
| MQA | 1/num_q_heads | やや低下 | PaLM、Falcon |
| GQA | 1/group_factor | MHA に近い | LLaMA-2/3、Gemma、Mistral |

### 6-2. MLA（Multi-head Latent Attention）- DeepSeek

vLLM の `MLAAttentionSpec`（`vllm/v1/kv_cache_interface.py` L337–397）が表しているのが、DeepSeek V2/V3 で採用された革新的なアーキテクチャ：

```
通常の KV cache:
  K: [seq_len, num_heads, head_dim]
  V: [seq_len, num_heads, head_dim]
  → KV = 2 × num_heads × head_dim per token

MLA の KV cache（latent vector に圧縮）:
  c_KV: [seq_len, kv_lora_rank]   ← KV を低ランクベクトルで表現
  k_R:  [seq_len, qk_rope_head_dim]  ← RoPE 用のロープ成分だけ別保存
  → KV ≈ kv_lora_rank + qk_rope_head_dim per token
```

DeepSeek V3 の例：

```python
# vllm/v1/kv_cache_interface.py の MLAAttentionSpec:
# fp8_ds_mla 形式では 1 トークンあたり 584 bytes (DeepSeek V4)
# 通常の GQA (head_size=128, num_kv_heads=8) なら:
#   2 × 8 × 128 × 2 bytes = 4096 bytes / token
# MLA では 584 bytes → 約 7 倍の圧縮
```

---

## 7. KV Cache の量子化

### 7-1. vLLM の量子化モード（vllm/v1/kv_cache_interface.py L32–80）

```python
class KVQuantMode(IntEnum):
    NONE              = 0   # FP16/BF16（デフォルト）
    FP8_PER_TENSOR    = 1   # テンソル全体を共通スケールで FP8 に量子化
    INT8_PER_TOKEN_HEAD = 2  # トークン×ヘッドごとに動的スケールで INT8
    FP8_PER_TOKEN_HEAD  = 3  # トークン×ヘッドごとに動的スケールで FP8
    NVFP4              = 4   # NVIDIA FP4 packed 形式 + FP8 ブロックスケール
```

スケールの管理（`vllm/model_executor/layers/quantization/kv_cache.py`）：

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0))  # チェックポイントから読み込み
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0))

    # per_token_head モードでは動的にカーネル内でスケールを計算
    # FP8_PER_TENSOR ではチェックポイントの k_scale/v_scale を使用
```

### 7-2. transformers の QuantizedLayer（KIVI）

```python
# 最新 residual_length=128 トークンは高精度（BF16）のまま
# それ以前は 4bit + グループ量子化でキャッシュ圧縮

# メモリ効果（理論値）:
# BF16 対比: 最初の 128 トークン = 1x, それ以降 ≈ 0.25x
# 長いシーケンスで大幅節約
```

### 7-3. 量子化による精度・メモリのトレードオフ

| 方式 | メモリ削減 | 精度への影響 | 備考 |
|---|---|---|---|
| FP16/BF16（基準） | 1x | なし | デフォルト |
| FP8 per-tensor | 0.5x | 軽微 | スケールはチェックポイント静的値 |
| INT8 per-token-head | 0.5x | 軽微 | 動的スケール、精度高め |
| KIVI（4bit） | ~0.25x（長列）| やや低下 | residual バッファで補償 |
| NVFP4 | 0.25x | 低下あり | H100/B200 専用 |

---

## 8. その他の最適化技術

### 8-1. FlashAttention との関係

KV Cache の節約とは別の観点で、FlashAttention は **Attention 計算そのもののメモリ効率を改善**する：

```
通常の Attention:
  S = Q·Kᵀ → [seq_len, seq_len] の行列を DRAM に書き出す
  P = softmax(S) → DRAM から読み込み
  O = P·V → DRAM から読み込み

FlashAttention:
  S, P を SRAM（GPU オンチップ）内で完結させる（タイリング）
  DRAM への書き出しなし → メモリバンド幅のボトルネックを解消
```

KV Cache とは**直交する最適化**：FlashAttention は計算時の中間行列のメモリ問題、KV Cache はトークン保存のメモリ問題を解決する。

### 8-2. スライディングウィンドウ / Chunked Attention

vLLM の `SlidingWindowSpec`（`vllm/v1/kv_cache_interface.py` L435–）：

```python
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int  # このサイズ分しか KV を保持しない

# メモリ: 全長 seq_len の代わりに sliding_window 分だけ
# Mistral-7B: sliding_window=4096
```

### 8-3. CPU オフロード / ディスクオフロード

transformers の `Cache` クラスは `offloading=True` で GPU → CPU 転送を非同期ストリームで実行：

```python
# layer i の forward 中に layer i+1 の KV を prefetch
torch.cuda.default_stream().wait_stream(self.prefetch_stream)
self.prefetch(layer_idx + 1)
# forward 後に layer i を CPU に追い出す
self.offload(layer_idx)
```

vLLM v1 では P/D 分離（Prefill / Decode の異なるノードへの分散）と KV 転送も `kv_transfer/` 以下に実装されている。

### 8-4. Speculative Decoding との関係

Draft モデルが生成したトークン候補の KV も同じ KV Cache に乗せる。  
Draft が reject された場合、余分な KV をキャッシュから削除する処理が必要になる。

---

## 9. メモリ計算まとめ

### FullAttentionSpec のメモリ計算（vLLM）

```python
page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_size
# per-token-head スケール付き (+FP32 スケールを含む場合):
page_size_bytes += 2 * block_size * num_kv_heads * 4  # float32 scale
```

### モデル別 KV Cache サイズ（1 トークン・全レイヤー）

| モデル | 手法 | Q/KV heads | head_dim | layers | 1 token KV (BF16) |
|---|---|---|---|---|---|
| LLaMA-3 8B | GQA | 32/8 | 128 | 32 | 32×2×8×128×2 = **131 KB** |
| LLaMA-3 70B | GQA | 64/8 | 128 | 80 | 80×2×8×128×2 = **328 KB** |
| DeepSeek V3 | MLA | 128/1 | 512 | 61 | ~61×584 = **~35 KB** |
| Mistral 7B | GQA+SW | 32/8 | 128 | 32 | ≤131 KB（SW で制限） |

DeepSeek V3 の MLA は LLaMA-3 70B 比で **約 9 倍** のメモリ効率。

---

## 10. 実装の選択指針

| シナリオ | 推奨 |
|---|---|
| 推論サーバー・高スループット | **vLLM**（PagedAttention + Prefix Caching） |
| 研究・ファインチューニング中の推論 | **transformers DynamicCache**（柔軟性重視） |
| 長いシーケンス・メモリ節約 | **KIVI (QuantizedLayer)** or **SlidingWindowLayer** |
| バッチ推論・torch.compile と組み合わせ | **StaticCache** |
| 超長コンテキスト（CPU 補完） | **Cache(offloading=True)** + vLLM prefix caching |
