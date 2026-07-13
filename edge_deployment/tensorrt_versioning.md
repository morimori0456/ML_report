# TensorRT Version Compatibility — Surviving the Dependency Chain

> A field guide to TensorRT's version hell: what must match (driver / CUDA / cuDNN / TensorRT / the serialized engine), why a `.plan` built yesterday fails to load today, the escape hatches (version- and hardware-compatible engines), and a reproducible-build discipline. Companion diagnostic: [check_trt_env.py](https://github.com/morimori0456/ML_report/blob/main/edge_deployment/check_trt_env.py). Pairs with [tensorrt_quantization_jetson.md](tensorrt_quantization_jetson.md).

TensorRT is the fastest way to run a model on NVIDIA hardware and also the easiest to break, because its behaviour is decided by a *chain* of independently-versioned components. A serialized engine is not a portable model file — it is a build artifact tied to the exact TensorRT patch version, the GPU architecture, the OS, and (transitively) the CUDA it was built against. Change any link and deserialize fails, usually with a terse "engine plan is not compatible" that gives no hint which link moved. This report raises the resolution on that chain: what actually has to match, what only *sometimes* matters, how Jetson/JetPack removes your freedom to choose, and the two build flags that buy portability at a performance cost — so you can set up environments that do not blow up.

---

## Table of Contents
1. [The Dependency Chain (What Talks to What)](#1-the-dependency-chain-what-talks-to-what)
2. [Engine Portability: the Rules That Bite](#2-engine-portability-the-rules-that-bite)
3. [JetPack Pins Everything (the Jetson Reality)](#3-jetpack-pins-everything-the-jetson-reality)
4. [x86 vs Jetson: How TensorRT Is Installed](#4-x86-vs-jetson-how-tensorrt-is-installed)
5. [Escape Hatches: Version- and Hardware-Compatible Engines](#5-escape-hatches-version--and-hardware-compatible-engines)
6. [ONNX Opset and the Export Boundary](#6-onnx-opset-and-the-export-boundary)
7. [A Reproducible-Build Discipline](#7-a-reproducible-build-discipline)
8. [Troubleshooting Decision Tree](#8-troubleshooting-decision-tree)
9. [Common Pitfalls](#9-common-pitfalls)
10. [References](#10-references)

---

## 1. The Dependency Chain (What Talks to What)

TensorRT sits at the end of a stack where each layer constrains the next:

```
GPU driver (nvidia.ko)  ──requires──►  a minimum version for a given CUDA
      │
      ▼
CUDA (toolkit + runtime)  ──TensorRT 10 is CUDA 12.x only──►
      │
      ▼
cuDNN  ──optional in TRT 10 (only some plugins/legacy paths need it)──►
      │
      ▼
TensorRT (libnvinfer.so.X + python binding)  ──builds──►
      │
      ▼
engine .plan  ──tied to: TRT patch version × GPU arch × OS × CUDA──►  runtime
```

| Link | What must match | How strict |
|---|---|---|
| driver ↔ CUDA | driver ≥ CUDA's minimum (e.g. TRT 10 built with CUDA 12.6 needs driver ≥ 525) | hard, but forward-compatible (newer driver runs older CUDA) |
| CUDA ↔ TensorRT | TensorRT 10 → CUDA **12.x only** (single build works across all 12.x) | hard — TRT 10 will not load on CUDA 11 |
| cuDNN ↔ TensorRT | TRT 10 reduced the cuDNN dependency to optional | soft — often irrelevant now |
| TensorRT python ↔ `libnvinfer.so` | same major (and normally same patch) | hard — mixed pip/system installs crash at load |
| engine ↔ runtime | see §2 | hard by default |

### Key insight
> **The engine is the least portable artifact in the whole stack.** Drivers are forward-compatible and TRT 10 spans all CUDA 12.x, so the *runtime* environment is fairly forgiving — but the serialized `.plan` is pinned to the exact TensorRT patch that built it. Treat engines as build outputs to regenerate, never as checked-in model files.

**Why this matters:** most "TensorRT broke" incidents are not driver/CUDA problems — they are someone shipping or caching a `.plan` and loading it under a different TensorRT. Knowing the engine is the brittle link tells you where to look first.

---

## 2. Engine Portability: the Rules That Bite

By default, a serialized engine records the **major.minor.patch.build** of the TensorRT that created it. On deserialize, the runtime compares versions; a mismatch fails hard:

```
The engine plan file is not compatible with this version of TensorRT,
expecting library version X.Y.Z got A.B.C, please rebuild.
```

A `.plan` is, by default, **non-portable across all four of these axes**:

| Axis | Portable by default? | Escape hatch |
|---|---|---|
| TensorRT version | No — exact patch match | Version-compatible build (§5), forward-only |
| GPU architecture (SM/CC) | No — built for the build GPU's compute capability | Hardware-compatible build (§5) |
| OS / platform (Linux ↔ Windows) | No — never | none; rebuild per platform |
| CUDA major | No (10 is 12.x) | none across majors |

### Key insight
> **"Same TensorRT, same GPU model" is still not a guarantee across machines.** Serialized engines are not portable across platforms even with identical TensorRT/CUDA/cuDNN and the same GPU, and are not portable across GPU architectures without hardware-compatibility mode. Build on the deployment target, or build with the compatibility flags and accept the cost.

**Why this matters:** teams routinely build an engine in CI on an A100 and deploy to an Orin/Thor — and it fails. The fix is to build on the target architecture (or in a matched container), not to debug the runtime.

---

## 3. JetPack Pins Everything (the Jetson Reality)

On Jetson you do **not** choose CUDA / cuDNN / TensorRT independently — **JetPack bundles a fixed set**, and TensorRT ships as system packages (`libnvinfer`) tied to that JetPack. Flashing a JetPack version *is* choosing your TensorRT.

| JetPack | CUDA | cuDNN | TensorRT | Board / notes |
|---|---|---|---|---|
| 5.x | 11.4 | 8.x | 8.4–8.6 | Orin, TRT 8 series |
| 6.0 | 12.2 | 8.9 | 8.6 | Orin, still TRT 8 |
| 6.2 / 6.2.1 | 12.6 | 9.3 | 10.3 | Orin, jumps to TRT 10 |
| 7.x | 13.x | 9.x | 10.13–10.16 | **Thor** (Blackwell, CC 11.0), CUDA 13 |

(Verify the exact point-release matrix in the NVIDIA docs — patch levels move.)

### Key insight
> **You cannot `pip install --upgrade tensorrt` your way out of a JetPack on Jetson.** TensorRT on Jetson comes from the JetPack apt repo and is coupled to Jetson Linux (L4T). Upgrading TensorRT means flashing a newer JetPack (JetPack 7 previews an "upgradable compute stack" to loosen this, but treat independent upgrades as unsupported unless the docs say otherwise). Plan the model stack around the JetPack the fleet runs.

**Why this matters:** if your model needs a TensorRT feature (e.g. a newer op, FP4), that is a JetPack/flashing decision for the whole device fleet — not a `pip` change. This constraint should drive your version choice up front.

---

## 4. x86 vs Jetson: How TensorRT Is Installed

The install path — and therefore the failure modes — differ by platform:

| | x86_64 (dev/CI/cloud) | Jetson (aarch64) |
|---|---|---|
| Source | pip wheels (`tensorrt` from NVIDIA PyPI), tar, or `.deb`; or NGC containers | JetPack apt (`libnvinfer*`), pre-installed on the device |
| Upgrade | swap the wheel/container freely | flash a new JetPack |
| Common break | pip `tensorrt` wheel + a different system `libnvinfer.so` on `LD_LIBRARY_PATH` → `undefined symbol` / soname mismatch | mixing a pip `tensorrt` with the JetPack `.so`; wrong `LD_LIBRARY_PATH` |

The single most common x86 break: the Python binding version and the `libnvinfer.so.X` the loader actually finds disagree. `check_trt_env.py` prints both so you can see it at a glance.

### Key insight
> **`import tensorrt` succeeding does not mean the right `libnvinfer.so` is loaded.** The python binding and the C++ library are versioned separately; a stray system install on the library path shadows the wheel's. Always confirm the soname (`ldconfig -p | grep nvinfer`) matches `trt.__version__`.

**Why this matters:** "works in my container, `undefined symbol` on the box" is nearly always this soname split — checking it first saves hours of reinstalling the wrong thing.

---

## 5. Escape Hatches: Version- and Hardware-Compatible Engines

TensorRT 8.6+ offers two build-time flags that trade performance for portability, so you can avoid rebuilding for every environment.

| Flag | Buys you | Direction / scope | Cost |
|---|---|---|---|
| **Version compatibility** (`kVERSION_COMPATIBLE` + embedded **lean runtime**) | engine runs on **newer** TensorRT within the same major | forward-only (TRT10-built runs on TRT11 runtime; **not** the reverse) | typically slower; larger plan (carries a lean runtime) |
| **Hardware compatibility** (`kAMPERE_PLUS` or `kSAME_COMPUTE_CAPABILITY`) | engine runs on **multiple GPU architectures** | `kAMPERE_PLUS` = Ampere and later; `kSAME_COMPUTE_CAPABILITY` = same CC family | some perf loss; `kAMPERE_PLUS` disables newest-HW features |

```python
config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)          # forward-compatible plan
config.hardware_compatibility_level = \
    trt.HardwareCompatibilityLevel.AMPERE_PLUS               # cross-arch (broad, slower)
# or SAME_COMPUTE_CAPABILITY: better perf + keeps FP8/FP4, narrower portability
```

### Key insight
> **Compatibility flags are for fleets and rollouts, not for peak performance.** Version compatibility is forward-only and adds a lean runtime; hardware compatibility (especially `kAMPERE_PLUS`) costs speed and can disable FP8/FP4. `kSAME_COMPUTE_CAPABILITY` keeps low-precision features while still covering a CC family — often the sweet spot. Use them to ship one engine across a mixed fleet, then rebuild per-target when you need the last 10-20%.

**Why this matters:** they turn "rebuild for every device/version" into "build once, run broadly" — the right call for OTA updates and heterogeneous fleets, as long as you knowingly accept the throughput hit.

---

## 6. ONNX Opset and the Export Boundary

Before TensorRT sees your model it is usually ONNX, adding two more version surfaces: the **ONNX opset** and the **exporter**.

- Each TensorRT release supports ONNX up to a certain opset; too-new an opset (or exotic ops) makes the parser reject the graph.
- Prefer the **legacy TorchScript exporter** (`torch.onnx.export(..., dynamo=False)`) with a **supported opset** (opset 17 is a safe modern default) — the newer `dynamo=True` exporter can emit ops the TRT parser does not yet handle.
- Use **`polygraphy`** and **`trtexec --onnx=...`** to validate that a graph parses and builds *before* wiring it into an application.

**Why this matters:** an export/opset mismatch fails at parse time with an unsupported-node error that looks like a TensorRT bug but is really an ONNX-boundary version issue — fixable by lowering the opset or switching exporter, not by touching TensorRT.

---

## 7. A Reproducible-Build Discipline

The way to not get burned is to make the whole stack explicit and rebuildable:

1. **Record the quadruple with every engine:** TensorRT, CUDA, GPU arch (CC), OS. Store it next to the `.plan` (filename or sidecar). `check_trt_env.py` prints exactly this set.
2. **Build in a pinned container.** Use the NGC TensorRT image (or the JetPack-matched L4T container) so CI and the target share a stack. Never build against a floating system CUDA.
3. **Treat `.plan` as a build artifact, not source.** Regenerate it in CI on (or matching) the target arch; do not commit engines or share them across machines.
4. **Pin the ONNX opset and exporter** in the export script; don't let a torch upgrade silently change the graph.
5. **Rebuild engines on every JetPack / TensorRT bump** as a release step — and validate accuracy/latency, since kernels (and thus numerics) can change.

### Key insight
> **Version the environment, ship the ONNX, rebuild the engine on target.** The portable artifacts are your source model and its ONNX (with a pinned opset); the engine is derived. A pipeline that regenerates engines per target in a pinned container makes the version chain a non-event.

**Why this matters:** this single discipline — ONNX is source, engine is derived, environment is pinned — eliminates the entire class of "the plan won't load" failures instead of firefighting them one by one.

---

## 8. Troubleshooting Decision Tree

| Symptom | Most likely cause | Fix |
|---|---|---|
| `engine plan file is not compatible ... expecting X got Y, please rebuild` | engine built with a different TensorRT patch | rebuild on the target TRT, or use a version-compatible build |
| deserialize returns `None`, no clear message | GPU-arch mismatch (built on a different SM) | build on target arch, or hardware-compatibility mode |
| `undefined symbol` / soname error on `import`/load | pip `tensorrt` vs system `libnvinfer.so` split | align versions; fix `LD_LIBRARY_PATH`; check `ldconfig -p` |
| ONNX parse fails: unsupported node/op | opset too new or dynamo-exported op | lower opset, export with `dynamo=False`, check parser support |
| low-precision flag "ignored", latency == FP32 | precision unsupported/silently fell back (e.g. Thor #4590) | verify layer precision (verbose log); newer TRT; different precision |
| TRT 10 fails to load entirely | running on CUDA 11 | TRT 10 needs CUDA 12.x — upgrade CUDA/driver |

Run `python check_trt_env.py --engine your.plan` to dump the stack and test-deserialize the plan in one shot; it flags the driver/CUDA/TRT/soname mismatches and the known Thor trap.

**Why this matters:** almost every TensorRT error maps to one link in the chain; matching the message to the link turns a vague "TensorRT is broken" into a one-line fix.

---

## 9. Common Pitfalls

- **Committing or sharing `.plan` files.** Engines are non-portable build artifacts; regenerate them per target/version instead of caching across machines.
- **Building engines in CI on a different GPU than deployment.** Cross-arch engines don't load without hardware-compat mode — build on (or matching) the target.
- **Assuming Jetson lets you upgrade TensorRT via pip.** It's JetPack/apt-coupled; upgrading means flashing.
- **Mixing a pip `tensorrt` wheel with a system `libnvinfer.so`.** Soname/symbol crashes at load — keep one source of truth on the library path.
- **Trusting `import tensorrt` as proof of a healthy install.** Check the loaded `libnvinfer` soname matches the binding version.
- **Letting the ONNX opset/exporter float.** A torch upgrade can change the exported graph; pin opset and use `dynamo=False` for TRT.
- **Forgetting to rebuild/revalidate after a JetPack or TensorRT bump.** New kernels change numerics and latency; re-run accuracy and benchmarks.
- **Expecting version-compatible engines to be backward-compatible.** They are forward-only (older TRT can't load a newer plan) and slower — not a free lunch.

---

## 10. References

- TensorRT Support Matrix (per-release CUDA/cuDNN/OS): https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- TensorRT Version Compatibility: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/version-compatibility.html
- TensorRT Engine (Hardware) Compatibility: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
- JetPack SDK (bundled component versions): https://developer.nvidia.com/embedded/jetpack
- Torch-TensorRT on JetPack (build notes): https://docs.pytorch.org/TensorRT/getting_started/jetpack.html
- TensorRT issue #4590 (Thor FP8/FP4 silent FP32 fallback): https://github.com/NVIDIA/TensorRT/issues/4590
- Polygraphy (ONNX/TensorRT debugging toolkit): https://github.com/NVIDIA/TensorRT/tree/release/10.0/tools/Polygraphy
- Related in this repo: [tensorrt_quantization_jetson.md](tensorrt_quantization_jetson.md), [build_int8_engine.py](https://github.com/morimori0456/ML_report/blob/main/edge_deployment/build_int8_engine.py)
