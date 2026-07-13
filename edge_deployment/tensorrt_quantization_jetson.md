---
title: "TensorRT INT8 Quantization on Jetson — A Hands-On Recipe for AD Perception"
description: "Quantize an AD-perception model to INT8 and deploy with TensorRT on Jetson: the affine mapping, PTQ entropy calibration vs QAT, the build/benchmark recipe, and the Thor FP8/FP4 silent-fallback trap."
---

> A verified, device-side recipe to quantize a small autonomous-driving perception model to INT8 and deploy it with TensorRT on NVIDIA Jetson (incl. Jetson Thor / Blackwell). Companion script: [build_int8_engine.py](https://github.com/morimori0456/ML_report/blob/main/edge_deployment/build_int8_engine.py). Like [navsim_hands_on.md](../autonomous_driving/driving_benchmarks/navsim_hands_on.md), this is a hands-on that runs on real hardware; the benchmark tables are templates you fill from your own Jetson run.

On the edge, the model that wins is not the most accurate — it is the most accurate model that fits the latency, memory, and power budget of the SoC. Quantization is the highest-leverage knob for this: moving a perception backbone from FP32/FP16 to INT8 roughly halves memory and, on Tensor-Core hardware, can 2-4x throughput, usually for <1% accuracy loss if calibrated properly. This report is the mechanics of doing that correctly on Jetson with TensorRT — the affine INT8 mapping, PTQ calibration (what `IInt8EntropyCalibrator2` actually computes), when you need QAT, the exact build/benchmark commands, and the Jetson-Thor-specific traps (FP8/FP4, silent FP32 fallback). It fills the "edge deployment" gap in this collection and is the inference-side counterpart to [ml_training_infrastructure.md](../infrastructure/ml_training_infrastructure.md).

---

## Table of Contents
1. [Why Quantize on the Edge: the Precision Ladder](#1-why-quantize-on-the-edge-the-precision-ladder)
2. [INT8 Quantization Theory](#2-int8-quantization-theory)
3. [PTQ: Calibration (What the Calibrator Computes)](#3-ptq-calibration-what-the-calibrator-computes)
4. [QAT: When PTQ Is Not Enough (TensorRT Model Optimizer)](#4-qat-when-ptq-is-not-enough-tensorrt-model-optimizer)
5. [The Jetson Hands-On Recipe](#5-the-jetson-hands-on-recipe)
6. [Benchmark Table (Fill From Your Device)](#6-benchmark-table-fill-from-your-device)
7. [Jetson Thor: FP8 / FP4 and a Silent-Fallback Trap](#7-jetson-thor-fp8--fp4-and-a-silent-fallback-trap)
8. [Common Pitfalls](#8-common-pitfalls)
9. [References](#9-references)

---

## 1. Why Quantize on the Edge: the Precision Ladder

A perception stack (detection, BEV segmentation, occupancy) runs every frame under a hard latency budget. Lower precision buys three things at once: less memory bandwidth (the usual bottleneck), more math throughput (Tensor Cores are faster at low precision), and less energy per inference. TensorRT is the runtime that turns a low-precision *plan* into fused kernels for the specific Jetson GPU.

| Precision | Bits | Relative throughput* | Typical accuracy loss | Notes |
|---|---|---|---|---|
| FP32 | 32 | 1x | baseline | rarely used for deployment |
| FP16 / BF16 | 16 | ~2x | negligible | safe default, almost free on Jetson |
| **INT8** | 8 | ~2-4x | <1% (calibrated) | needs calibration or QAT; the workhorse |
| FP8 (E4M3) | 8 | ~2-4x | small | Ada/Hopper/Blackwell; better dynamic range than INT8 |
| FP4 (NVFP4) | 4 | up to ~8x | model-dependent | Blackwell only (Jetson Thor); needs modern TensorRT |

*Order-of-magnitude, Tensor-Core-bound layers; verify on your device.

### Key insight
> **Try FP16 first; reach for INT8 when FP16 misses the budget.** FP16 is nearly loss-free and needs no calibration data, so it is the baseline. INT8 is the next step when FP16 still exceeds your latency/memory target — but it costs you a calibration set and validation effort. Do not skip straight to INT8 out of habit.

**Why this matters:** the right precision is the *cheapest* one that hits the budget with acceptable accuracy — starting at FP16 saves you from spending calibration effort you may not need.

---

## 2. INT8 Quantization Theory

INT8 quantization maps a floating-point tensor $x$ to 8-bit integers via an **affine** transform with a scale $s$ and (optionally) a zero-point $z$:

$$
x_q = \operatorname{clip}\!\big(\operatorname{round}(x / s) + z,\; q_{\min},\; q_{\max}\big),
\qquad \hat{x} = s\,(x_q - z)
$$

TensorRT uses **symmetric** quantization ($z = 0$, range $[-127, 127]$) for both weights and activations, so the scale is the only parameter to choose:

$$
s = \frac{\alpha}{127}, \qquad \alpha = \text{the clipping threshold (a "max" value) for the tensor}
$$

Everything above $\alpha$ saturates. Choosing $\alpha$ is the entire game — too large wastes INT8 levels on rare outliers (coarse steps for the bulk of values); too small clips useful signal.

| Choice | Options | TensorRT default | Guidance |
|---|---|---|---|
| Symmetric vs asymmetric | sym / asym | **symmetric** | sym is faster (no zero-point term); TensorRT uses it |
| Granularity | per-tensor / per-channel | **per-channel weights**, per-tensor activations | per-channel weights recover most accuracy for conv layers |
| What gets a scale | weights + activations | both | weight scales from the weights directly; activation scales need calibration |

### Key insight
> **Weight scales are free; activation scales are the hard part.** Weights are known at build time, so their (per-channel) scale is just a max over each output channel. Activations depend on the input distribution, which TensorRT cannot see from the graph — so it must *observe* representative data. That observation step is calibration.

**Why this matters:** this split explains the whole PTQ workflow — you supply data solely so TensorRT can estimate the *activation* clipping thresholds; the weight side needs nothing from you.

---

## 3. PTQ: Calibration (What the Calibrator Computes)

Post-Training Quantization (PTQ) takes a trained FP32/FP16 model plus a small **calibration set** (a few hundred representative, preprocessed inputs — no labels) and estimates each activation tensor's clipping threshold $\alpha$. TensorRT runs the network on the calibration data, builds a **histogram** of each activation tensor, and picks $\alpha$ to minimize information loss.

| Calibrator | Threshold rule | Use when |
|---|---|---|
| `IInt8EntropyCalibrator2` (**default**) | minimize KL divergence between FP and quantized distributions | general default; best for most CNNs |
| `IInt8MinMaxCalibrator` | $\alpha = \max\lvert x\rvert$ (no clipping) | transformers / tensors where outliers carry signal |
| percentile (via ModelOpt) | $\alpha$ = 99.9th percentile, etc. | when a fixed clip works better than KL |

The **entropy** calibrator (v2) is what "INT8 calibration" usually means: it searches candidate thresholds over the histogram and keeps the one whose quantized distribution is closest (in KL divergence) to the full-precision one — trading a little clipping for finer steps on the bulk of the mass.

```python
# The calibrator interface TensorRT calls (full working version in build_int8_engine.py):
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def get_batch_size(self):        return 1
    def get_batch(self, names):      ...   # feed one calibration tensor at a time (device ptr)
    def read_calibration_cache(self):...   # reuse a saved cache -> skip recalibration
    def write_calibration_cache(self, c):  # persist scales; portable across builds
        open(self.cache, "wb").write(c)
```

### Key insight
> **The calibration cache is a portable, reusable asset.** Once written, the `.cache` file (per-tensor scales) can be reused to rebuild the engine on another machine or after a TensorRT upgrade *without the calibration dataset*. Version it alongside the model. This is why the entropy/minmax calibrators are preferred — their caches are portable.

**Why this matters:** calibration is the slow, data-dependent step; caching turns it into a one-time cost and decouples engine builds from dataset access.

**Calibration-set rules that actually move accuracy:**
- Use **real, preprocessed** inputs from your distribution (same normalization as inference) — not random noise.
- A few hundred samples (≈100-500) is typically enough; more rarely helps.
- Cover the operating domain (day/night, urban/highway) or you will clip the tails you skipped.

---

## 4. QAT: When PTQ Is Not Enough (TensorRT Model Optimizer)

If PTQ still drops too much accuracy (common for compact backbones, low-bit, or outlier-heavy layers), use **Quantization-Aware Training (QAT)**: insert fake-quantize nodes into the model and fine-tune, so the weights learn to be robust to the rounding. NVIDIA's current tool is **TensorRT Model Optimizer** (`nvidia-modelopt`), which supersedes the older `pytorch-quantization`.

```python
import modelopt.torch.quantization as mtq

# 1. pick a quant config (INT8 per-channel weights + per-tensor activations)
config = mtq.INT8_DEFAULT_CFG

# 2. PTQ calibration in-framework (a forward loop over the calib set)
def forward_loop(model):
    for x in calib_loader:
        model(x)
model = mtq.quantize(model, config, forward_loop)   # inserts + calibrates fake-quant

# 3. (QAT) fine-tune a few epochs at a low LR, then export
#    train(model, ...)  # simulated-quant loss teaches robustness
import modelopt.torch.opt as mto
mto.export(model)                                   # -> ONNX with Q/DQ nodes for TensorRT
```

| | PTQ | QAT |
|---|---|---|
| Data | small calib set (no labels) | full labeled training set |
| Compute | minutes | a fine-tuning run |
| Accuracy recovery | good (usually <1%) | best (can match FP32) |
| When | first attempt | PTQ gap too large, or INT4/FP4 |

### Key insight
> **PTQ first, QAT only if the validation gap justifies the training cost.** QAT can recover near-FP32 accuracy even at 4-bit, but it needs the training pipeline, labels, and GPU-hours. Reach for it when PTQ's accuracy drop exceeds your tolerance — not by default.

**Why this matters:** most perception models pass with PTQ; treating QAT as the fallback (not the starting point) saves days of effort. The QAT path also connects to the NF4/weight-quantization ideas in [lora_qlora.md](../llm/lora_qlora.md).

---

## 5. The Jetson Hands-On Recipe

End-to-end on the device, using the small AD-perception model in [build_int8_engine.py](https://github.com/morimori0456/ML_report/blob/main/edge_deployment/build_int8_engine.py) (a compact camera→BEV segmentation net, output stride 8, ~0.58M params — a stand-in for your real backbone+head). **Verified locally (x86, CPU torch): the model forward pass and the ONNX export.** The INT8 build and benchmark steps require the Jetson.

```bash
# 0. lock clocks so measurements are stable, and check versions
sudo nvpmodel -m 0 && sudo jetson_clocks
python -c "import tensorrt as trt; print('TensorRT', trt.__version__)"

# 1. build a calibration set from REAL preprocessed frames (script writes placeholders;
#    replace calib/*.npy with your normalized dataset frames)
python build_int8_engine.py calib --n 256 --out calib/

# 2. export the perception model to ONNX (opset 17, legacy exporter for clean TRT parsing)
python build_int8_engine.py export --onnx tinybev.onnx

# 3a. baseline FP16 engine
python build_int8_engine.py build --onnx tinybev.onnx --fp16 --engine tinybev_fp16.plan
# 3b. INT8 engine with entropy calibration (writes a portable calib.cache)
python build_int8_engine.py build --onnx tinybev.onnx --int8 --calib calib/ --engine tinybev_int8.plan

# 4. benchmark latency/throughput (clocks locked; average over 1000 iters)
python build_int8_engine.py bench --engine tinybev_fp16.plan
python build_int8_engine.py bench --engine tinybev_int8.plan
```

The fastest path needs no Python for build/bench — `trtexec` does calibration, build, and timing:

```bash
trtexec --onnx=tinybev.onnx --fp16 --saveEngine=tinybev_fp16.plan
trtexec --onnx=tinybev.onnx --int8 --calib=calib.cache --saveEngine=tinybev_int8.plan
trtexec --loadEngine=tinybev_int8.plan --iterations=1000 --avgRuns=1000   # prints mean/median latency
```

### Key insight
> **Lock the clocks before you measure, or your numbers are noise.** Jetson uses dynamic frequency scaling (DVFS); without `nvpmodel -m 0 && jetson_clocks` the same engine can vary 2x run-to-run. Report the power mode alongside every latency number.

**Why this matters:** an unstated or unlocked power mode makes edge benchmarks irreproducible — the single most common reason two people "measure" different latencies for the same engine.

---

## 6. Benchmark Table (Fill From Your Device)

Run the recipe on your Jetson and record the numbers. These are the axes that matter; the cells are intentionally blank — do not trust fabricated edge numbers, measure your own. (For reference, INT8 typically lands ~2-4x FP32 throughput on Tensor-Core-bound layers with <1% accuracy loss, but the exact figures are model- and device-specific.)

| Precision | Latency (ms) | Throughput (inf/s) | Engine size (MB) | mIoU / mAP | Δ vs FP16 |
|---|---|---|---|---|---|
| FP32 | | | | | |
| FP16 | | | | | (baseline) |
| INT8 (PTQ, entropy) | | | | | |
| INT8 (QAT) | | | | | |
| FP8 (Thor) | | | | | |

Record alongside: Jetson model, JetPack + TensorRT version, `nvpmodel` mode, batch size, input resolution, and the calibration-set description. Without these the row is not reproducible.

---

## 7. Jetson Thor: FP8 / FP4 and a Silent-Fallback Trap

Jetson Thor (Blackwell GPU, compute capability 11.0) adds 5th-gen Tensor Cores with a Transformer Engine that switches between **FP8 and FP4 (NVFP4)** — advertised up to ~1035 TFLOPs FP8 (dense) / ~2070 TFLOPs FP4 (sparse). Supported precisions span TF32, BF16, FP16, FP8, FP4, and INT8. FP8's E4M3 format keeps more dynamic range than INT8, so it can be more accuracy-friendly at the same bit width.

> **Trap (verify on your stack):** a reported TensorRT issue (#4590) states that on Thor (CC 11.0), TensorRT 10.13.3.9 *accepts* `BuilderFlag.FP8` / `BuilderFlag.FP4` but **silently builds an FP32 engine** — no error, no warning. Always confirm the layer precisions actually chosen (build with `trt.Logger(VERBOSE)` and inspect the layer-precision log, or compare engine latency/size against FP16). Do not assume a low-precision flag took effect just because the build succeeded.

**Why this matters:** low-precision flags can no-op silently on bleeding-edge hardware/driver combos; a latency that matches FP32 is your signal the precision did not apply. Treat "the flag was accepted" and "the engine is actually low-precision" as two separate facts to verify.

---

## 8. Common Pitfalls

- **Calibrating on random noise.** Scales estimated from noise clip your real activations. Always calibrate on real, identically-preprocessed frames.
- **Not locking clocks.** Report `nvpmodel`/`jetson_clocks` state; unlocked DVFS makes latencies irreproducible.
- **Assuming a precision flag applied.** Especially on Thor (issue #4590), verify actual layer precisions — compare engine size/latency to FP16 or read the verbose build log.
- **INT8 everywhere.** A few sensitive layers (first/last, detection heads) often need FP16. Use mixed precision / per-layer precision rather than forcing the whole graph to INT8.
- **Exporting with the wrong ONNX path.** Use a TensorRT-friendly export (legacy exporter / `dynamo=False`, a supported opset); the new dynamo exporter can emit ops the TRT parser rejects.
- **Ignoring preprocessing mismatch.** Calibration and inference must use identical normalization; a different mean/std silently shifts every scale.
- **Comparing accuracy on the wrong device.** PyTorch cannot run TensorRT INT8 kernels; validate accuracy by running the actual `.plan` engine, not a PyTorch fake-quant stand-in.
- **Forgetting the calibration cache is versioned state.** Rebuilds silently reuse a stale `calib.cache` if present — delete it when you change the calibration set.

---

## 9. References

- TensorRT Developer Guide — Working with Quantized Types: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- "Achieving FP32 Accuracy for INT8 Inference Using QAT with TensorRT" (NVIDIA): https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- NVIDIA TensorRT Model Optimizer (ModelOpt) docs: https://nvidia.github.io/TensorRT-Model-Optimizer/
- ModelOpt PyTorch Quantization guide: https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html
- Jetson Thor / Blackwell overview (NVIDIA): https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/
- TensorRT issue #4590 (Thor FP8/FP4 silent FP32 fallback): https://github.com/NVIDIA/TensorRT/issues/4590
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018. arXiv: https://arxiv.org/abs/1712.05877
- Related in this repo: [lora_qlora.md](../llm/lora_qlora.md) (NF4 weight quantization), [ml_training_infrastructure.md](../infrastructure/ml_training_infrastructure.md)
