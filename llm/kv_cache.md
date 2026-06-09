# KV Cache Complete Guide

> Code references:
> - `transformers` (HuggingFace): `src/transformers/cache_utils.py`
> - `vllm` (vLLM project): `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`, `vllm/model_executor/layers/quantization/kv_cache.py`

---

## 1. Why KV Cache is needed

Transformer Self-Attention is computed as follows:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q = X·Wq   shape: [seq_len, d_model]
K = X·Wk   shape: [seq_len, d_model]
V = X·Wv   shape: [seq_len, d_model]
```

In LLM **autoregressive decoding**, one new token is generated per step.
Recomputing Q·Kᵀ over all past tokens at every step is highly inefficient:

```
Generating a new token at step t:
  new Q(t) = x(t) · Wq          (only for the 1 new token)
  K(0..t) = [x(0)..x(t)] · Wk  (all t+1 past tokens → recompute every time?)
  V(0..t) = [x(0)..x(t)] · Wv  (all t+1 past tokens → recompute every time?)
```

**KV Cache** stores K and V in memory once they are computed, skipping recomputation.

```
After step t-1: K(0..t-1), V(0..t-1) are already cached
At step t:
  K(t) = x(t) · Wk  ← compute only for 1 new token
  K(0..t) = concat(K_cache, K(t))  ← append to cache
```

### Computational complexity comparison

| Approach | Computation per step | Total computation over N steps |
|---|---|---|
| Without KV Cache | O(t²·d) | O(N³·d) |
| With KV Cache | O(t·d) | O(N²·d) |

---

## 2. KV Cache memory size

### Basic formula

KV cache size per layer:

```
bytes_per_layer = 2             (K and V)
               × seq_len        (number of tokens)
               × num_kv_heads   (number of KV heads)
               × head_dim       (head dimension)
               × dtype_bytes    (float16=2, bfloat16=2, float32=4)
```

vLLM's `FullAttentionSpec.real_page_size_bytes` is computed as follows
(`vllm/v1/kv_cache_interface.py` L170–175):

```python
return (
    2              # K + V
    * block_size   # number of tokens per block
    * num_kv_heads
    * head_size
    * get_dtype_size(self.dtype)
)
```

### Concrete example: LLaMA-3 70B

| Parameter | Value |
|---|---|
| Number of layers | 80 |
| Number of KV heads (GQA) | 8 |
| Head dimension | 128 |
| dtype | bfloat16 (2 bytes) |

```
Cache size per token:
  2 × 8 × 128 × 2 × 80 = 327,680 bytes ≈ 320 KB

Sequence of 4096 tokens:
  320 KB × 4096 = 1.25 GB / request

Batch size 8:
  1.25 GB × 8 = 10 GB
```

---

## 3. HuggingFace Transformers implementation

`transformers/cache_utils.py` manages the cache with a two-layer structure: `Cache` → `CacheLayerMixin`.

### 3-1. DynamicLayer (default)

A simple implementation that grows dynamically during decoding (L109–188):

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

Because `torch.cat` is called every step, **memory reallocation cost** grows as the sequence gets longer.

### 3-2. StaticLayer (inference optimization)

Pre-allocates the maximum length and updates via slicing (L277–):

```python
class StaticLayer(CacheLayerMixin):
    # Allocates [batch, heads, max_len, head_dim] upfront at initialization
    # On each update: self.keys[:, :, cache_position] = key_states
```

No `torch.cat`, no memory reallocation. Works well with `torch.compile`.

### 3-3. DynamicSlidingWindowLayer

Used with sliding-window Attention in models such as Mistral / Gemma.
Retains only the most recent `sliding_window` tokens and automatically discards older KV entries (L190–275).

### 3-4. QuantizedLayer (following the KIVI paper)

Keeps only the most recent `residual_length` tokens at full precision; earlier tokens are quantized and compressed (L514–587):

```python
class QuantizedLayer(DynamicLayer):
    """residual_length tokens at full precision, earlier tokens are quantized"""
    nbits           = 4      # quantization bit width
    q_group_size    = 64     # group size for group quantization
    residual_length = 128    # number of tokens kept at full precision

    def update(self, key_states, value_states, ...):
        # When the full-precision buffer is full, move it to the quantized cache
        if self.keys.shape[-2] + 1 >= self.residual_length:
            self._quantized_keys   = self._quantize(keys_to_return, axis=self.axis_key)
            self._quantized_values = self._quantize(values_to_return, axis=self.axis_value)
            self.keys   = torch.tensor([])  # reset full-precision buffer
            self.values = torch.tensor([])
```

### 3-5. Cache (container)

A container that bundles all layers and controls CPU offloading (L890–998):

```python
class Cache:
    layers: list[CacheLayerMixin]

    def update(self, key_states, value_states, layer_idx, ...):
        # When offloading, prefetch on a non-default stream
        if self.offloading:
            torch.cuda.default_stream().wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1)

        keys, values = self.layers[layer_idx].update(key_states, value_states)

        if self.offloading:
            self.offload(layer_idx)  # evict from GPU → CPU
        return keys, values
```

---

## 4. vLLM PagedAttention

While `transformers`' `DynamicLayer` allocates contiguous memory per batch,
vLLM **manages physical memory in page (block) units** to eliminate memory fragmentation.

### 4-1. KVCacheBlock (vllm/v1/core/kv_cache_utils.py L116–163)

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 0 to num_gpu_blocks-1
    ref_cnt:  int = 0      # reference count (>1 when shared)
    _block_hash: BlockHashWithGroupId | None = None  # hash for prefix caching

    # Doubly-linked list pointers (used exclusively by FreeKVCacheBlockQueue)
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

### 4-2. FreeKVCacheBlockQueue (vllm/v1/core/kv_cache_utils.py L164–)

Free blocks are managed in a **doubly-linked list that supports O(1) operations**.
Python's `deque` has a C++ implementation but O(N) deletion for middle elements.
This implementation achieves O(1) by storing pointers directly in each block.

```
Eviction order (LRU):
  Head = least recently used block (next candidate for eviction)
  Tail = most recently used block

When returning blocks to the free list:
  Multiple blocks within a request are appended to the tail in reverse order
  → The tail end (head of the block chain) is evicted first
```

### 4-3. KV cache block layout (FlashAttention backend)

`vllm/v1/attention/backends/flash_attn.py` L149:

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    # block_size must be a multiple of 16
    return (num_blocks, 2, block_size, num_kv_heads, head_size)
    #        ^^^^^^^^^^^  ^  ^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^
    #        total blocks K/V  tokens    num heads    head dim
```

To treat physically **non-contiguous blocks** as a single logical sequence,
`block_tables` (a mapping from logical block IDs to physical block IDs) is maintained separately.

### 4-4. Benefits of PagedAttention

```
Conventional cache (contiguous allocation):
  Request A: [K0 K1 K2 K3 ... Kmax] pre-allocated for maximum length
                → short requests waste the tail (internal fragmentation)

PagedAttention (block allocation):
  With block size = 16:
  Request A (23 tokens): [block7][block12][block3(7tokens)]
  Request B (8 tokens):  [block1]
  → Only allocate what is actually used. Free blocks are available for other requests
```

| Metric | Conventional allocation | PagedAttention |
|---|---|---|
| Memory fragmentation | Pre-allocate max length → large waste | Block-level allocation → waste <5% |
| Maximum batch size | Small | Significantly higher (up to 24x throughput in the paper) |
| Implementation cost | Low | High (block_tables management required) |

---

## 5. Prefix Caching

A technique for sharing KV across multiple requests that share the same prefix (e.g., a system prompt).

### 5-1. Match detection via hashing (vllm/v1/core/kv_cache_utils.py L541–570)

```python
def hash_block_tokens(
    hash_function,
    parent_block_hash: BlockHash | None,  # hash of the preceding block (chained)
    curr_block_token_ids: Sequence[int],   # token ID sequence for this block
    extra_keys: tuple | None = None,
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # random seed (collision avoidance)

    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

**Hash chain structure:**

```
Block 0: hash(NONE_HASH, tokens[0:16])      = H0
Block 1: hash(H0,        tokens[16:32])     = H1
Block 2: hash(H1,        tokens[32:48])     = H2
...
```

Because each block's hash **encodes information about the entire preceding token sequence**,
blocks with the same token sequence always produce the same hash value.
(This allows `KVCacheBlock.ref_cnt > 1` so that multiple requests can share a block.)

### 5-2. LRU eviction

When the cache is full, the head of `FreeKVCacheBlockQueue` (LRU) is evicted.
Blocks with `ref_cnt > 0` (in use) are not evicted.

---

## 6. Attention mechanisms and head count optimization

### 6-1. MHA / MQA / GQA comparison

```
MHA (Multi-Head Attention):  num_kv_heads = num_q_heads
  Q heads: [H0 H1 H2 H3]
  K heads: [H0 H1 H2 H3]  ← same count as Q; large KV cache
  V heads: [H0 H1 H2 H3]

MQA (Multi-Query Attention): num_kv_heads = 1
  Q heads: [H0 H1 H2 H3]
  K heads: [H0]             ← all Q heads share a single KV
  V heads: [H0]             ← KV cache is 1/num_heads of MHA

GQA (Grouped-Query Attention): 1 < num_kv_heads < num_q_heads
  Q heads: [H0 H1 H2 H3]
  K heads: [G0     G1   ]   ← shared across 2 groups (LLaMA-3 70B: Q=64, KV=8)
  V heads: [G0     G1   ]
```

| Method | KV size | Accuracy | Adopted by |
|---|---|---|---|
| MHA | baseline (1x) | High | GPT-2, BERT, early LLaMA |
| MQA | 1/num_q_heads | Slightly lower | PaLM, Falcon |
| GQA | 1/group_factor | Close to MHA | LLaMA-2/3, Gemma, Mistral |

### 6-2. MLA (Multi-head Latent Attention) - DeepSeek

`MLAAttentionSpec` in vLLM (`vllm/v1/kv_cache_interface.py` L337–397) represents the innovative architecture adopted in DeepSeek V2/V3:

```
Conventional KV cache:
  K: [seq_len, num_heads, head_dim]
  V: [seq_len, num_heads, head_dim]
  → KV = 2 × num_heads × head_dim per token

MLA KV cache (compressed into a latent vector):
  c_KV: [seq_len, kv_lora_rank]   ← KV represented as a low-rank vector
  k_R:  [seq_len, qk_rope_head_dim]  ← RoPE component stored separately
  → KV ≈ kv_lora_rank + qk_rope_head_dim per token
```

DeepSeek V3 example:

```python
# MLAAttentionSpec in vllm/v1/kv_cache_interface.py:
# fp8_ds_mla format: 584 bytes per token (DeepSeek V4)
# Conventional GQA (head_size=128, num_kv_heads=8):
#   2 × 8 × 128 × 2 bytes = 4096 bytes / token
# MLA: 584 bytes → approximately 7x compression
```

---

## 7. KV Cache quantization

### 7-1. vLLM quantization modes (vllm/v1/kv_cache_interface.py L32–80)

```python
class KVQuantMode(IntEnum):
    NONE              = 0   # FP16/BF16 (default)
    FP8_PER_TENSOR    = 1   # quantize entire tensor to FP8 with a shared scale
    INT8_PER_TOKEN_HEAD = 2  # INT8 with dynamic scale per token×head
    FP8_PER_TOKEN_HEAD  = 3  # FP8 with dynamic scale per token×head
    NVFP4              = 4   # NVIDIA FP4 packed format + FP8 block scale
```

Scale management (`vllm/model_executor/layers/quantization/kv_cache.py`):

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0))  # loaded from checkpoint
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0))

    # In per_token_head mode, scales are computed dynamically inside the kernel
    # In FP8_PER_TENSOR mode, k_scale/v_scale from the checkpoint are used
```

### 7-2. transformers QuantizedLayer (KIVI)

```python
# The most recent residual_length=128 tokens are kept at full precision (BF16)
# Earlier tokens are compressed with 4-bit group quantization

# Memory effect (theoretical):
# Compared to BF16: first 128 tokens = 1x, beyond that ≈ 0.25x
# Significant savings for long sequences
```

### 7-3. Accuracy vs. memory trade-offs from quantization

| Method | Memory reduction | Impact on accuracy | Notes |
|---|---|---|---|
| FP16/BF16 (baseline) | 1x | None | Default |
| FP8 per-tensor | 0.5x | Minor | Static scale from checkpoint |
| INT8 per-token-head | 0.5x | Minor | Dynamic scale, higher accuracy |
| KIVI (4-bit) | ~0.25x (long sequences) | Slight degradation | Residual buffer compensates |
| NVFP4 | 0.25x | Some degradation | H100/B200 only |

---

## 8. Other optimization techniques

### 8-1. Relationship with FlashAttention

Separately from KV Cache savings, FlashAttention **improves memory efficiency of the Attention computation itself**:

```
Conventional Attention:
  S = Q·Kᵀ → writes [seq_len, seq_len] matrix to DRAM
  P = softmax(S) → reads from DRAM
  O = P·V → reads from DRAM

FlashAttention:
  S and P are computed entirely within SRAM (on-chip GPU memory) using tiling
  No DRAM writes → eliminates memory bandwidth bottleneck
```

This is an **orthogonal optimization** to KV Cache: FlashAttention solves the memory problem of intermediate matrices during computation; KV Cache solves the memory problem of storing tokens.

### 8-2. Sliding window / Chunked Attention

vLLM's `SlidingWindowSpec` (`vllm/v1/kv_cache_interface.py` L435–):

```python
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int  # only retain KV for this many tokens

# Memory: only sliding_window tokens instead of the full seq_len
# Mistral-7B: sliding_window=4096
```

### 8-3. CPU offloading / disk offloading

The transformers `Cache` class executes GPU → CPU transfers asynchronously with `offloading=True`:

```python
# Prefetch layer i+1's KV while running the forward pass for layer i
torch.cuda.default_stream().wait_stream(self.prefetch_stream)
self.prefetch(layer_idx + 1)
# After the forward pass, offload layer i to CPU
self.offload(layer_idx)
```

In vLLM v1, P/D disaggregation (distributing Prefill / Decode to different nodes) and KV transfer are also implemented under `kv_transfer/`.

### 8-4. Relationship with Speculative Decoding

KV entries for token candidates generated by the draft model are placed in the same KV Cache.
When a draft token is rejected, the excess KV entries must be removed from the cache.

---

## 9. Memory calculation summary

### FullAttentionSpec memory calculation (vLLM)

```python
page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_size
# With per-token-head scales (when FP32 scales are included):
page_size_bytes += 2 * block_size * num_kv_heads * 4  # float32 scale
```

### KV Cache size per model (1 token, all layers)

| Model | Method | Q/KV heads | head_dim | layers | 1 token KV (BF16) |
|---|---|---|---|---|---|
| LLaMA-3 8B | GQA | 32/8 | 128 | 32 | 32×2×8×128×2 = **131 KB** |
| LLaMA-3 70B | GQA | 64/8 | 128 | 80 | 80×2×8×128×2 = **328 KB** |
| DeepSeek V3 | MLA | 128/1 | 512 | 61 | ~61×584 = **~35 KB** |
| Mistral 7B | GQA+SW | 32/8 | 128 | 32 | ≤131 KB (limited by SW) |

DeepSeek V3's MLA achieves approximately **9x better memory efficiency** than LLaMA-3 70B.

---

## 10. Implementation selection guide

| Scenario | Recommendation |
|---|---|
| Inference server, high throughput | **vLLM** (PagedAttention + Prefix Caching) |
| Research, inference during fine-tuning | **transformers DynamicCache** (flexibility first) |
| Long sequences, memory saving | **KIVI (QuantizedLayer)** or **SlidingWindowLayer** |
| Batch inference, combined with torch.compile | **StaticCache** |
| Ultra-long context (CPU supplement) | **Cache(offloading=True)** + vLLM prefix caching |
