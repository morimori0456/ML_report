---
title: "VAD Ego Trajectory (gt_ego_his_trajs / gt_ego_fut_trajs) Computation Logic"
description: "How VAD computes ego-vehicle past and future trajectories as sequential per-step offsets in the lidar coordinate frame."
---

## Overview

The ego vehicle's past and future trajectories are computed inside `_fill_trainval_infos()` in `vad_nuscenes_converter.py`.
Both are stored as **sequential per-step offsets on the current frame's LiDAR coordinate system (LCF)**.

| Key | Shape | Meaning |
|---|---|---|
| `gt_ego_his_trajs` | `[2, 2]` (= `[his_ts, 2]`) | Sequential displacement (x, y) for past 2 steps |
| `gt_ego_fut_trajs` | `[6, 2]` (= `[fut_ts, 2]`) | Sequential displacement (x, y) for future 6 steps |

---

## Common Foundation: `get_global_sensor_pose()`

```python
# vad_nuscenes_converter.py L541–560
def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    sd_ep = nusc.get('ego_pose',          lidar_sample_data['ego_pose_token'])
    sd_cs = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])

    global_from_ego  = transform_matrix(sd_ep['translation'], Quaternion(sd_ep['rotation']))
    ego_from_sensor  = transform_matrix(sd_cs['translation'], Quaternion(sd_cs['rotation']))
    pose = global_from_ego @ ego_from_sensor   # shape: [4, 4]
    return pose
```

`pose[:3, 3]` gives **the position (xyz) of the LIDAR_TOP sensor in the global coordinate system**.
In mathematical notation:

```
p_global = R_e2g · p_lidar_in_ego + t_e2g
         = R_e2g · (R_l2e · p_lidar + t_l2e) + t_e2g
```

When `p_lidar = (0, 0, 0)` (sensor origin), `pose[:3, 3]` = global position of the LiDAR sensor.

---

## gt_ego_his_trajs Computation Logic

### Step 1: Collecting Global Positions (L402–418)

```python
his_ts = 2
ego_his_trajs      = np.zeros((his_ts+1, 3))  # shape [3, 3]
ego_his_trajs_diff = np.zeros((his_ts+1, 3))  # difference for extrapolation

sample_cur = sample   # start from current frame (t)
for i in range(his_ts, -1, -1):   # i = 2, 1, 0 in order
    if sample_cur is not None:
        pose_mat = get_global_sensor_pose(sample_cur, nusc)
        ego_his_trajs[i] = pose_mat[:3, 3]          # store global position

        # for extrapolation: keep the difference with the next frame
        if sample_cur['next'] != '':
            sample_next = nusc.get('sample', sample_cur['next'])
            pose_mat_next = get_global_sensor_pose(sample_next, nusc)
            ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]

        sample_cur = nusc.get('sample', sample_cur['prev']) if sample_cur['prev'] != '' else None
    else:
        # No past frame at scene start → extrapolate with constant velocity
        ego_his_trajs[i]      = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
        ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]
```

Contents of `ego_his_trajs` after the loop (in chronological order):

| Index | Corresponding Sample | Contents |
|---|---|---|
| `[0]` | `t-2` (2 frames ago) | Global LiDAR position, or constant-velocity extrapolated value |
| `[1]` | `t-1` (1 frame ago) | Global LiDAR position |
| `[2]` | `t` (current) | Global LiDAR position |

**Intent of constant-velocity extrapolation:** At the start of a scene, no past frames exist. To still generate a fixed-length trajectory tensor, the most recent velocity vector (`ego_his_trajs_diff[i+1]`) is used directly to back-calculate a virtual position.

### Step 2: Transform from Global → LCF (Current Frame's LiDAR Coordinates) (L420–427)

```python
# (1) global → ego (current frame's vehicle coordinates)
ego_his_trajs  -= np.array(pose_record['translation'])        # translation
rot_mat         = Quaternion(pose_record['rotation']).inverse.rotation_matrix
ego_his_trajs   = np.dot(rot_mat, ego_his_trajs.T).T          # rotation

# (2) ego → lidar (current frame's LiDAR coordinates)
ego_his_trajs  -= np.array(cs_record['translation'])          # translation
rot_mat         = Quaternion(cs_record['rotation']).inverse.rotation_matrix
ego_his_trajs   = np.dot(rot_mat, ego_his_trajs.T).T          # rotation
```

After transformation, `ego_his_trajs[2]` (the current frame's LiDAR position viewed in LiDAR coordinates) is by definition **(0, 0, 0)**.

### Step 3: Absolute Positions → Sequential Offsets (L428)

```python
ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]
# shape: [3, 3] → [2, 3]
```

Each row becomes a "displacement vector for one step":

| Row Index | Meaning |
|---|---|
| `[0]` | Displacement from `pos(t-2) → pos(t-1)` (LCF) |
| `[1]` | Displacement from `pos(t-1) → pos(t)`   (LCF) |

### Step 4: Saving (L528)

```python
info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
# z component discarded; saved to pkl with shape = [2, 2]
```

---

## gt_ego_fut_trajs Computation Logic

### Step 1: Collecting Global Positions (L431–442)

```python
fut_ts = 6
ego_fut_trajs = np.zeros((fut_ts+1, 3))   # shape [7, 3]
ego_fut_masks = np.zeros((fut_ts+1))

sample_cur = sample   # current frame (t)
for i in range(fut_ts+1):    # i = 0, 1, ..., 6
    pose_mat = get_global_sensor_pose(sample_cur, nusc)
    ego_fut_trajs[i] = pose_mat[:3, 3]
    ego_fut_masks[i] = 1

    if sample_cur['next'] == '':
        ego_fut_trajs[i+1:] = ego_fut_trajs[i]   # fill remaining with final position
        break
    else:
        sample_cur = nusc.get('sample', sample_cur['next'])
```

| Index | Corresponding Sample | Contents |
|---|---|---|
| `[0]` | `t` (current) | Global LiDAR position |
| `[1]` | `t+1` | Global LiDAR position |
| ... | ... | ... |
| `[6]` | `t+6` (3 seconds later) | Global LiDAR position (filled with final position if unavailable) |

**Handling at scene end:** `fut_valid_flag = False` is set, and the planning error for this sample is skipped during evaluation (near `VAD_head.py` L595).

### Step 2: Transform from Global → LCF (L444–450)

Same logic as for past trajectories (`pose_record` / `cs_record` from the current frame are used).

After transformation, `ego_fut_trajs[0]` (the current frame viewed in LCF) is **(0, 0, 0)**.

### Step 3: Driving Command Determination (L452–457)

```python
# Determined from "absolute position on LCF" before converting to sequential offsets
if ego_fut_trajs[-1][0] >= 2:     # x displacement at t+6 is +2m or more
    command = np.array([1, 0, 0]) # Turn Right
elif ego_fut_trajs[-1][0] <= -2:  # x displacement at t+6 is -2m or less
    command = np.array([0, 1, 0]) # Turn Left
else:
    command = np.array([0, 0, 1]) # Go Straight
```

> **On the correspondence between the LCF x-axis and "Turn Right":**
> In nuScenes' LiDAR coordinate system, the x-axis points **to the left of the vehicle's direction of travel** (right-hand system).
> Therefore, `x >= 2` corresponds to "Turn Right" because the LCF used here is derived by applying the inverse transform of `ego2global @ lidar2ego`.
> In the implementation, the steering sign inversion for Singapore (left-hand traffic) is applied only to the Kappa calculation in `ego_lcf_feat`; the command determination is not modified.

### Step 4: Absolute Positions → Sequential Offsets (L459)

```python
ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
# shape: [7, 3] → [6, 3]
```

| Row Index | Meaning |
|---|---|
| `[0]` | Displacement from `pos(t) → pos(t+1)` (LCF) |
| `[1]` | Displacement from `pos(t+1) → pos(t+2)` (LCF) |
| ... | ... |
| `[5]` | Displacement from `pos(t+5) → pos(t+6)` (LCF) |

### Step 5: Saving (L529–530)

```python
info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)  # shape [6, 2]
info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)     # shape [6]
# masks[0]=1 means the t→t+1 offset is valid
```

---

## Coordinate Transform Chain Summary

```
Global coordinates (ENU / UTM)
  │  get_global_sensor_pose() = global_from_ego @ ego_from_sensor
  │  → pose[:3, 3] = LiDAR position in global coordinates
  │
  ▼ − pose_record['translation']         (global → ego translation)
  ▼ × Quaternion(pose_record).inverse    (global → ego rotation)
Ego coordinates (current frame's vehicle center)
  │
  ▼ − cs_record['translation']           (ego → lidar translation)
  ▼ × Quaternion(cs_record).inverse      (ego → lidar rotation)
LiDAR coordinates (current frame = LCF)
  ← take sequential differences here
```

Since all positions are unified to "coordinates with the current frame's LiDAR as the origin,"
the model can learn **ego-centric relative motion**.

---

## Usage in the Model

### gt_ego_his_trajs → Planning Query Initialization

```python
# VAD_head.py L712
if self.ego_his_encoder is not None:
    ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
else:
    ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
ego_query = ego_his_feats   # used as the planner's initial query
```

The displacements for the past 2 steps are encoded by an MLP and injected as the initial state of the planner, representing "how the ego is currently moving." If `ego_his_encoder` is `None`, a fixed learned query is used instead.

### gt_ego_fut_trajs → Supervision Signal and Evaluation for Planning Loss

```python
# VAD.py L421, L425–426 (test time)
ego_fut_trajs = ego_fut_trajs[0, 0]              # [fut_ts, 2]
ego_fut_pred  = ego_fut_preds[ego_fut_cmd_idx]   # select prediction corresponding to command
ego_fut_pred  = ego_fut_pred.cumsum(dim=-2)      # per-step offset → cumulative positions
ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)     # per-step offset → cumulative positions
```

At evaluation time, sequential offsets are converted back to cumulative positions before computing L2 distance (ADE/FDE).

---

## Full Data Flow Diagram

```
nuScenes sample (current frame t)
│
├─ get_global_sensor_pose(t-2, t-1, t)    global LiDAR positions for past 3 frames
│   → transform to LCF
│   → sequential difference [t-2→t-1, t-1→t]
│   → gt_ego_his_trajs: shape [2, 2]     ← saved to pkl
│
└─ get_global_sensor_pose(t, t+1, ..., t+6)  global LiDAR positions for future 7 frames
    → driving command determined from absolute LCF position (x displacement at t+6)
    → transform to LCF
    → sequential difference [t→t+1, ..., t+5→t+6]
    → gt_ego_fut_trajs: shape [6, 2]     ← saved to pkl
    → gt_ego_fut_cmd:   shape [3]        ← saved to pkl (one-hot)
    → gt_ego_fut_masks: shape [6]        ← saved to pkl

                               ↓
VADCustomNuScenesDataset.get_data_info()
  → unpack ego_his_trajs / ego_fut_trajs into input_dict
  → included in batch via CustomCollect3D

                               ↓
VAD_head.forward()
  → ego_his_trajs → ego_his_encoder → Planning Query
  → ego_fut_trajs → Loss computation / evaluation metrics
```

---

## Notes and Pitfalls

| Item | Detail |
|---|---|
| **Constant-velocity extrapolation at scene start** | Virtual positions are generated when past frames do not exist, which may differ from actual motion. Occurs for samples where `frame_idx == 0` |
| **Static padding at scene end** | When future frames are insufficient, the remaining entries are filled with the final frame's position, so later offsets all become (0, 0) |
| **z component discarded** | Saved as `[:, :2]`, so elevation is not included. On overpasses or slopes, the actual travel distance may differ from the horizontal distance |
| **Driving command determination timing** | Determined from "absolute positions" before sequential differencing, so the command is based on the lateral displacement of the cumulative trajectory's final point |
| **Fixed LCF** | All frames' trajectories are transformed to the **current frame (t)'s LiDAR coordinates**, so evaluation and inference share the same origin |
