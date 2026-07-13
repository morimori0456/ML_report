---
title: "DriveTransformer Complete Guide — Unified Transformer for Scalable End-to-End Autonomous Driving"
description: "A from-first-principles walkthrough of DriveTransformer, which unifies end-to-end autonomous-driving perception, prediction, and planning in a single Transformer."
---

> Paper: Xiaosong Jia, Junqi You, Zhiyuan Zhang, Junchi Yan, *DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving*, ICLR 2025
> arXiv: [2503.07656](https://arxiv.org/abs/2503.07656) / Official implementation: [Thinklab-SJTU/DriveTransformer](https://github.com/Thinklab-SJTU/DriveTransformer)

An approach that unifies End-to-End autonomous driving (E2E-AD) with a **single Transformer**. This document aims to build understanding of "why this design?" by working backward from the limitations of prior methods. For a minimal hands-on implementation, see [drive_transformer_demo.ipynb](drive_transformer_demo.ipynb).

---

## Table of Contents
1. [Background: E2E-AD Lineage and Challenges](#1-background-e2e-ad-lineage-and-challenges)
2. [The Three Pillars of DriveTransformer](#2-the-three-pillars-of-drivetransformer)
3. [Overall Architecture and Data Flow](#3-overall-architecture-and-data-flow)
4. [Task Queries — Agent / Map / Ego](#4-task-queries--agent--map--ego)
5. [Sensor Cross-Attention (Why No BEV)](#5-sensor-cross-attention-why-no-bev)
6. [Task Self-Attention (Task Parallelism)](#6-task-self-attention-task-parallelism)
7. [Temporal Cross-Attention and Streaming FIFO](#7-temporal-cross-attention-and-streaming-fifo)
8. [Task Heads and Loss Functions](#8-task-heads-and-loss-functions)
9. [Hyperparameters and Computational Cost](#9-hyperparameters-and-computational-cost)
10. [Common Misconceptions and Pitfalls](#10-common-misconceptions-and-pitfalls)

---

## 1. Background: E2E-AD Lineage and Challenges

### 1.1 The Conventional Paradigm = Sequential

Conventional E2E-AD systems, typified by UniAD / VAD, mimic the human driving pipeline and arrange

```
Perception → Prediction → Planning
```

in **series**. The output of each stage feeds into the next. This design has two structural weaknesses.

1. **Cumulative error**: Perception errors propagate to prediction and planning with no way to correct them in later stages. Information flows in only one direction — forward.
2. **Training instability**: Gradients from later-stage tasks (planning) struggle to reach the heavy backbone of earlier stages (perception). Since roles are fixed per stage, there is no guarantee that the perception stage learns features useful for planning.

### 1.2 Another Axis = BEV Representation

Many approaches convert multi-view images into a dense **BEV (Bird's-Eye-View) feature** grid (e.g., 200×200×C) before processing. BEV is intuitive, but:

- Construction cost is high (view transformation, voxel pooling)
- Representation is constrained by grid resolution
- It is "optimal for perception" but not necessarily "optimal for planning"

### 1.3 DriveTransformer's Position
> **Discard both sequential pipelines and dense BEV; solve all tasks directly on raw sensor features using Transformer attention alone.**

All relationships — task-to-task, task-to-raw-sensor, task-to-past-history — are learned entirely through attention. This eliminates cumulative error and allows planning gradients to flow directly back to the backbone.

---

## 2. The Three Pillars of DriveTransformer

| Pillar | What It Does | Problem Solved |
|---|---|---|
| **Task Parallelism** | All task queries mutually attend in each block; no explicit hierarchy | Cumulative error; gradient bottleneck in later stages |
| **Sparse Representation** | Task queries directly cross-attend to raw sensor features; no BEV | BEV construction cost; representational rigidity |
| **Streaming Processing** | Past task queries are stored in a FIFO queue; temporal fusion via temporal cross-attention | Temporal efficiency; feature reuse |

All three coexist within a single Transformer block. Each block consists of four operations:

```
Task Self-Attn → Sensor Cross-Attn → Temporal Cross-Attn → FFN
```

stacked **L=12 times**. A task head is attached to each block's output, and losses are computed at every block (deep supervision).

---

## 3. Overall Architecture and Data Flow

```
                  ┌─────────────────── 6 Multi-view Camera Images ──────────────────┐
                  │  CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_L, BACK_R       │
                  └───────────────────────────┬──────────────────────────────────────┘
                                              ▼
                        ResNet50 / EVA / VoVNet backbone
                                              ▼
                  Image features [B, N_cam, H, W, D]  +  3D-ray positional encoding
                                              │  flatten
                                              ▼
                  img_feats [B, N_img_token, D]   img_pos_embed [B, N_img_token, D]
                                              │
   ┌──────────── Task Queries (initialized as learned parameters) ─────────────┐  │
   │ Agent  [B, 900, D]   Map [B, 100, D]   Ego [B, 1, D]                      │  │
   └──────────────────────────┬─────────────────────────────────────────────────┘  │
                              ▼                                                      │
   ╔══════════════════ Decoder Block × 12 ════════════════════════╗                │
   ║  ① Task Self-Attn : [Agent;Map;Ego] mutually attend           ║                │
   ║  ② Sensor Cross-Attn: task queries → img_feats ◄──────────────╫────────────────┘
   ║  ③ Temporal Cross-Attn: task queries → FIFO queue (past L frames) ║◄── history queue
   ║  ④ FFN : separate FFN per Agent/Map/Ego                       ║
   ║  ─ head at each block output: det/motion/map/plan → loss      ║
   ╚══════════════════════════════════════════════════════════════╝
                              ▼
   Detection boxes / Motion prediction / Online map / Ego trajectory (6 modes)
                              ▼
        Winner-Take-All selects 1 mode → Control (CARLA closed-loop)
                              ▼
   Top-K queries pushed into FIFO queue for the next frame
```

Key points:
- **All blocks keep direct access to raw sensor features** (②). Features are not collapsed into BEV and discarded.
- **For temporal fusion, queries — not feature maps — are stored** (③). Being sparse, this is lightweight.
- Performance improves as more blocks are stacked — **scalability** (hence "Scalable" in the title).

---

## 4. Task Queries — Agent / Map / Ego

All state in DriveTransformer is represented as "query tokens." There are three types.

| Query | Count (Large) | Represents | Task | Initialization |
|---|---|---|---|---|
| **Agent query** | 900 | Dynamic objects (vehicles, pedestrians) | 3D detection + motion prediction | Learned parameters + uniform positional encoding |
| **Map query** | 100 | Static elements (lanes, signs) | Online mapping | Learned parameters + uniform positional encoding |
| **Ego query** | 1 | Possible ego vehicle behaviors | Planning (trajectory generation) | CAN bus passed through MLP; positional encoding = 0 |

The three types are concatenated as `query = [Agent; Map; Ego]` (shape `[B, 900+100+1, D]`) and treated as a single sequence. **This concatenation is the physical embodiment of task parallelism.** Applying self-attention causes agent and ego to interact within the same attention matrix.

> Intuition: The reasoning "that vehicle (agent) looks like it will turn right (motion) → ego should decelerate" is learned not as message passing between separate modules, but as the **off-diagonal entries of a single attention matrix**.

---

## 5. Sensor Cross-Attention (Why No BEV)

### 5.1 What It Does
Task queries serve as Q, and image tokens `img_feats` serve as K=V in a cross-attention operation.

```
Q = Linear(LayerNorm(query + pos_embed))  (separate weights cross_w_q[0..2] per agent/map/ego)
K = V = img_feats  (+ key_pos = img_pos_embed)
out = identity + Attention(Q, K, V)        # identity is the residual
```

In the official implementation, map queries are expanded to `map_pts_per_vec` points (polyline vertices) before cross-attention, enabling finer-grained reading of the image.

### 5.2 3D Positional Encoding (PETR-style)
Instead of constructing BEV, the positional encoding communicates **which 3D direction each image patch is looking toward**.

1. From each patch (u,v), cast a **ray** into 3D space using the camera intrinsic and extrinsic matrices
2. Sample K depth values along the ray to produce a 3D point sequence
3. Concatenate these and pass through an MLP to obtain `img_pos_embed`

This embeds geometry like "the patch at the upper-left of the image faces roughly X meters ahead-right of the ego vehicle" into the attention keys. **Camera extrinsic calibration** ([../camera_calibration/extrinsic_calibration.md](../camera_calibration/extrinsic_calibration.md)) is critical here.

### 5.3 Why Sparse Is Better
- Instead of computing all BEV grid cells, **only 900+100+1 queries attend to the pixels they need** → lower computation
- Since there is no intermediate representation (BEV), planning gradients travel in a straight line back to the backbone (consistent with end-to-end optimization)

---

## 6. Task Self-Attention (Task Parallelism)

Simply applying standard self-attention to the concatenated query `[Agent; Map; Ego]`. But the meaning is profound.

- agent ↔ agent: inter-object interactions (overtaking, following)
- agent ↔ map: "is this vehicle following a lane?"
- ego ↔ agent/map: ego query directly absorbs surrounding context needed for planning

**No hierarchy** — this is the key difference from prior work. In UniAD, the order is fixed as "detection → tracking → prediction → occupancy → planning," but here all tasks exchange information bidirectionally in each block as equals. Which relationships matter is decided by the attention during training.

---

## 7. Temporal Cross-Attention and Streaming FIFO

Velocity and intent cannot be inferred from a single frame. Temporal fusion is required. DriveTransformer **does not stack dense BEV features across time**. Instead, it **stores sparse queries**.

### 7.1 FIFO Queue
- Separate queues per task type (agent/map/ego)
- Each frame, only the **top-K highest-confidence** queries (50 per agent/map) are pushed (to avoid redundancy)
- Retains `memory_len_frame = 10` past frames (4 for nuScenes)

### 7.2 Temporal Fusion Procedure
Current queries Q cross-attend to past queries K=V in the queue. Since the coordinate frame shifts over time, corrections are needed:

1. **Ego coordinate transform**: Shift past-frame positional encodings to the current ego coordinate frame using transform matrix `T_{t→t0}`
2. **Motion compensation**: For agents, extrapolate past positions to the current timestep using predicted velocity
3. **Relative timestamp embedding**: Encode the time difference `(t − t0)` and add to keys
4. In the official implementation, a **zero register token** is prepended to each memory sequence to provide a fallback when no past information exists (used together with attn_mask)

> Effect: Attention can establish correspondences like "this agent query from 3 frames ago corresponds to this current query." Directly reusing features (queries) enables **feature reuse**, keeping computation lightweight.

The implementation uses DiT-style adaptive LayerNorm to inject temporal conditioning (scale/shift of normalization vary according to timestep).

---

## 8. Task Heads and Loss Functions

Heads are attached to each decoder block's output (agent/map/ego queries), and **losses are computed at every block** (deep supervision; only the final block is used at inference).

| Task | Head Output | Loss |
|---|---|---|
| **Detection** | 3D box (center, size, orientation, class) | DETR-style Hungarian matching loss |
| **Motion prediction** | Future trajectories per agent (multi-mode) | Winner-Take-All (local agent coordinate frame) |
| **Mapping** | Polyline map elements | MapTR-style Hungarian matching loss |
| **Planning** | Ego trajectory × **6 modes** + confidence | Winner-Take-All (regress to best of 6 modes) + classification |

The total loss is a weighted sum, with each term's scale adjusted to be approximately 1:

```
L = w_det·L_det + w_motion·L_motion + w_map·L_map + w_plan·L_plan
```

### 8.1 The 6 Planning Modes
The ego query has a **mode embedding** `nn.Embedding(6, D)`. The 6 modes roughly correspond to driving intentions such as "go straight / stop / turn left (shallow/sharp) / turn right (shallow/sharp)." Training uses WTA: regression loss is applied only to the 1 predicted trajectory (out of 6) closest to the GT (preventing mode collapse while preserving multimodality). The official implementation produces trajectories in two variants: equal-time-interval (fix_time) and equal-distance-interval (fix_dist).

---

## 9. Hyperparameters and Computational Cost

From the official `drivetransformer_large.py`:

| Item | Value |
|---|---|
| Agent query count | 900 (Top-50/frame/type propagated) |
| Map query count | 100 (Top-50/frame/type propagated) |
| Ego query count | 1 (6 mode embeddings) |
| Decoder layers L | 12 |
| Hidden dimension D (Large) | 768 |
| Memory length | 10 frames (4 for nuScenes) |
| Parameter count (Large) | ~646M |
| Backbone | ResNet50 / VoVNet / EVA02 |

Scaling law: Driving scores on Bench2Drive improve monotonically as layers and dimensions are increased from Tiny → Small → Base → Large. The sparse design without BEV yields good memory and FPS efficiency; even the Large model achieves closed-loop SOTA with a latency of approximately 211ms.

### Evaluation
- **Bench2Drive (CARLA closed-loop)**: Driving score SOTA
- **nuScenes (open-loop)**: Competitive results at high FPS

---

## 10. Common Misconceptions and Pitfalls

1. **"Build BEV once and then discard it" is NOT what happens** — BEV is never built in the first place. Task queries directly attend to raw image features in every block.
2. **"Parallelizing a sequential pipeline" is NOT what happens** — The stages are not arranged in parallel; **the very concept of stages is eliminated**, and all task queries are fed into a single self-attention sequence.
3. **Temporal fusion stores queries, not feature maps** — Being sparse, even 10 frames' worth is lightweight. The cost is orders of magnitude lower than methods that stack 10 dense BEV frames.
4. **Forgetting the temporal coordinate transform will break things** — Past query positions must be transformed to the current ego coordinate frame, and agent positions must be motion-compensated. Neglecting this causes misalignment between past and present object correspondences.
5. **Deep supervision is essential** — Attaching losses to every block ensures gradients flow even in deep blocks, stabilizing task parallelism.
6. **Ego query positional encoding is 0** — Unlike agent/map, the ego vehicle is always at the origin, so uniform PE is unnecessary. Note that it is initialized using CAN bus data (speed, steering angle).
7. **Map queries are expanded into point sequences before sensor cross-attn** — This is done to read polyline map elements at pixel-level granularity. The expansion method differs from that used for detection (agent).

---

## References

- Paper: [arXiv:2503.07656](https://arxiv.org/abs/2503.07656) / [OpenReview](https://openreview.net/forum?id=M42KR4W9P5)
- Official implementation: [github.com/Thinklab-SJTU/DriveTransformer](https://github.com/Thinklab-SJTU/DriveTransformer)
- Related: positional encoding principle from [PETR](https://arxiv.org/abs/2203.05625), maps from [MapTR](https://arxiv.org/abs/2208.14437), detection from [DETR](https://arxiv.org/abs/2005.12872)
- Prerequisites in this repository: [Camera Extrinsic Calibration](../camera_calibration/extrinsic_calibration.md) (geometry of 3D positional encoding), [VAD Dataloader](../VAD/dataloader.md) (nuScenes/Bench2Drive input)

> To solidify understanding through implementation, run [drive_transformer_demo.ipynb](drive_transformer_demo.ipynb). It executes a **minimal pure PyTorch DriveTransformer** (the three attention types, FIFO queue, and 6-mode planning head) on CPU — stripped of ResNet and datasets — so you can inspect each tensor's shape and gradient flow.
