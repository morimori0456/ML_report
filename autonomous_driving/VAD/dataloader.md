---
title: "VAD Dataloader Implementation Report (nuScenes Format)"
description: "An implementation-level walkthrough of VAD's nuScenes-format dataloader, covering the offline conversion phase and the runtime loading phase."
---

## Overview

VAD uses **`VADCustomNuScenesDataset`**, which inherits from mmdet3d's `NuScenesDataset`. Data loading is structured in two phases: an **offline data conversion phase** and a **runtime loading phase**.

---

## 1. File Structure

| File | Role |
|---|---|
| `tools/data_converter/vad_nuscenes_converter.py` | Converts raw nuScenes data → pkl info files |
| `projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` | Runtime dataset class; HD map loading |
| `projects/mmdet3d_plugin/datasets/pipelines/loading.py` | LiDAR point cloud / multi-sweep loading pipeline |
| `projects/configs/VAD/VAD_base_e2e.py` | Model and data configuration |

---

## 2. Phase 1: Offline Data Conversion (`vad_nuscenes_converter.py`)

`create_nuscenes_infos()` is the main function. After execution, `{prefix}_infos_temporal_train/val.pkl` is generated.

### 2-1. Initialization

```python
nusc = NuScenes(version=version, dataroot=root_path)
nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
```

`NuScenesCanBus` provides low-level sensor data such as odometry and steering.

### 2-2. Information Collected Per Sample (`_fill_trainval_infos()`)

**Sensor information** (L256–315)

```python
info = {
    'lidar_path': ...,            # LIDAR_TOP .bin file path
    'token': sample['token'],
    'cams': dict(),               # Info for 6 cameras
    'sweeps': [...],              # Up to 10 sweeps
    'ego2global_translation/rotation': ...,
    'lidar2ego_translation/rotation': ...,
    'can_bus': can_bus_18d,       # [pos(3), quat(4), ..., 0, 0] = 18 dimensions
    'map_location': ...,          # 4 cities e.g. 'boston-seaport'
    'fut_valid_flag': bool,       # Whether 6 future frames exist
}
```

**Camera information** — loop over 6 cameras (L288–302)

```python
info['cams'] = {
    'CAM_FRONT': {
        'data_path': '...',
        'cam_intrinsic': 3x3 matrix,
        'sensor2ego_rotation/translation': ...,
        'sensor2lidar_rotation/translation': ...,  # computed by obtain_sensor2top()
    },
    # + CAM_FRONT_RIGHT/LEFT, CAM_BACK/BACK_LEFT/BACK_RIGHT
}
```

**Annotation information** — VAD-specific extensions (L317–532)

| Key | Shape | Description |
|---|---|---|
| `gt_boxes` | [N, 7] | x,y,z,w,l,h,yaw (LiDAR coordinates) |
| `gt_names` | [N] | Class names |
| `gt_velocity` | [N, 2] | Velocity (converted to LiDAR coordinates) |
| `gt_agent_fut_trajs` | [N, fut_ts\*2] | Future 6-step trajectory per agent (sequential offsets) |
| `gt_agent_fut_masks` | [N, fut_ts] | Valid step mask |
| `gt_agent_fut_goal` | [N] | Goal direction class (directions 0–8 + stationary=9) |
| `gt_agent_lcf_feat` | [N, 9] | (x, y, yaw, vx, vy, w, l, h, cat_idx) |
| `gt_ego_his_trajs` | [2, 2] | Sequential offsets for past 2 frames of ego vehicle |
| `gt_ego_fut_trajs` | [6, 2] | Sequential offsets for future 6 frames of ego vehicle |
| `gt_ego_fut_cmd` | [3] | Driving command (Turn R / Turn L / Straight) |
| `gt_ego_lcf_feat` | [9] | (vx, vy, ax, ay, yaw angular velocity, length, width, speed, steering curvature) |

**Future trajectory computation logic** (L367–399)

```python
for j in range(fut_ts):  # fut_ts=6
    anno_next = nusc.get('sample_annotation', cur_anno['next'])
    box_next = Box(anno_next['translation'], ...)
    # Transform: global → ego → LiDAR coordinates
    box_next.translate(-pose_record['translation'])
    box_next.rotate(Quaternion(pose_record['rotation']).inverse)
    box_next.translate(-cs_record['translation'])
    box_next.rotate(Quaternion(cs_record['rotation']).inverse)
    gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]  # sequential difference
```

---

## 3. Phase 2: Runtime Loading (`VADCustomNuScenesDataset`)

### 3-1. Class Initialization (`__init__`, L984–1035)

```python
VADCustomNuScenesDataset(
    queue_length=4,                          # Temporal queue length
    fut_ts=6,                                # Future prediction steps
    pc_range=[-15, -30, -2, 15, 30, 2],     # Point cloud / BEV range
    map_classes=['divider', 'ped_crossing', 'boundary'],
    map_fixed_ptsnum_per_line=20,            # Fixed number of sample points per map element
)
```

Creates `VectorizedLocalMap`: pre-loads `NuScenesMap` + `NuScenesMapExplorer` for all 4 cities.

### 3-2. `get_data_info()` — Per-Sample Information Retrieval (L1271–1393)

Reads from pkl and computes the following additional values.

```python
# Projection matrix from LiDAR → camera → image (4x4 matrix per camera)
lidar2img_rt = viewpad @ lidar2cam_rt.T

# Coordinate transform matrix from LiDAR → global
lidar2global = ego2global @ lidar2ego

# Append heading angle to CAN bus
can_bus[-2] = patch_angle_rad
can_bus[-1] = patch_angle_deg
```

### 3-3. `get_ann_info()` — Annotation Retrieval (L1210–1268)

- When `with_attr=True`, loads `gt_agent_fut_trajs` etc. from pkl
- Converts bbox origin from nuScenes format (0.5, 0.5, 0.5) → KITTI format (0.5, 0.5, 0)

### 3-4. `vectormap_pipeline()` — Online HD Map Generation (L1064–1115)

```python
# Compute coordinates for querying the map API via LiDAR → ego → global transform
lidar2global = ego2global @ lidar2ego
map_pose = lidar2global[:2, 3]          # Patch center (global)
patch_angle = quaternion_yaw(rotation)  # Ego vehicle heading

anns_results = vector_map.gen_vectorized_samples(
    location, lidar2global_translation, lidar2global_rotation
)
```

Map elements retrieved by `VectorizedLocalMap.gen_vectorized_samples()` (L520–576):

| Class | Source Layer | Geometry Transform |
|---|---|---|
| `divider` | `road_divider` + `lane_divider` | LineString as-is |
| `ped_crossing` | `ped_crossing` polygon | Outline as LineString |
| `boundary` | Union of `road_segment` + `lane` polygons | Exterior/interior rings as LineString |

All elements are stored in `LiDARInstanceLines`. Each LineString is clipped by `patch_box`, then rotated by `patch_angle` to convert to LiDAR-relative coordinates.

**Point sampling methods in `LiDARInstanceLines`** (L32–464)

| Property | Description |
|---|---|
| `fixed_num_sampled_points` | Equal-distance sampling with `np.linspace(0, length, fixed_num)` → [N, fixed_num, 2] |
| `shift_fixed_num_sampled_points` | Polygons: all-direction shifts; lines: 2 directions (forward and reverse) to handle order ambiguity |

### 3-5. `prepare_train_data()` — Temporal Queue Construction (L1117–1161)

```python
# Temporal augmentation: shuffle past frames and randomly sample a subset
prev_indexs_list = list(range(index - queue_length, index))
random.shuffle(prev_indexs_list)
prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

# Only add frames from the same scene to the queue
if input_dict['scene_token'] == scene_token:
    data_queue.append(example)
```

### 3-6. `union2one()` — Merging Queue into a Single Sample (L1179–1208)

```python
# Images: [queue_length, 6cams, H, W, 3] → stacked tensor
imgs_list = [each['img'].data for each in queue]
queue[-1]['img'] = DC(torch.stack(imgs_list), stack=True)

# CAN bus: convert to relative displacement (relative to the first frame)
metas_map[i]['can_bus'][:3] -= prev_pos    # position difference
metas_map[i]['can_bus'][-1] -= prev_angle  # angle difference
```

---

## 4. Data Loading Pipeline (Camera Mode)

Pipeline configured in `VAD_base_e2e.py` (`input_modality.use_lidar=False`):

```
LoadMultiViewImageFromFiles            # Load 6 camera images
→ RandomScaleImageMultiViewImage       # Scale augmentation
→ PhotoMetricDistortionMultiViewImage  # Color space augmentation
→ NormalizeMultiviewImage              # ImageNet normalization
→ PadMultiViewImage                    # Padding
→ DefaultFormatBundle3D                # Tensor conversion
→ CustomCollect3D                      # Collect required keys
```

Since VAD uses only camera images, the LiDAR pipeline is skipped.

---

## 5. Overall Data Flow

```
nuScenes raw data
       │
       ▼
vad_nuscenes_converter.py
  - Traverse samples via NuScenes API
  - Compute coordinate transform matrices (sensor2lidar, lidar2ego, ego2global)
  - Compute future trajectories, ego trajectories, CAN bus data
  - Save to .pkl file
       │
       ▼
VADCustomNuScenesDataset.__getitem__()
  ├─ get_data_info()        : unpack info from pkl, compute lidar2img etc.
  ├─ get_ann_info()         : gt_boxes + future trajectory annotations
  ├─ pipeline()             : image load, augmentation, normalization
  ├─ vectormap_pipeline()   : online HD map generation from NuScenesMap API
  │     └─ VectorizedLocalMap.gen_vectorized_samples()
  │           → divider / ped_crossing / boundary → LiDARInstanceLines
  └─ union2one()            : merge queue_length=4 frames into one sample
       │
       ▼
Model inputs:
  - img              : [B, T, 6, 3, H, W]
  - img_metas        : CAN bus delta, lidar2img, coordinate transform matrices, etc.
  - gt_bboxes_3d     : LiDARInstance3DBoxes + future trajectories
  - map_gt_bboxes_3d : LiDARInstanceLines (HD map elements)
  - ego_fut_trajs    : ego vehicle future trajectory
  - ego_his_trajs    : ego vehicle past trajectory
```

---

## 6. nuScenes Coordinate Transform Chain

```
Global coordinates
  ↑ ego2global  (pose_record)
Ego coordinates
  ↑ lidar2ego   (cs_record)
LiDAR TOP coordinates  ← VAD's reference coordinate system
  ↑ sensor2lidar  (computed by obtain_sensor2top())
Individual camera / sensor coordinates
```

Map elements: compute the global patch coordinates using `lidar2global`, retrieve via `NuScenesMap.get_patch_coord()`, then convert back to LiDAR coordinates using `affinity.rotate / affine_transform` (L771–864).

---

## 7. Key Points Summary

| Point | Detail |
|---|---|
| **HD map is generated online** | Not saved in pkl; generated from `NuScenesMap` API at every iteration |
| **Temporal 4-frame queue** | Temporal data augmentation via shuffling; relative displacement managed with CAN bus |
| **Agent trajectories are sequential offsets in LiDAR coordinates** | Global trajectories converted to LCF (Local Coordinate Frame) |
| **Camera-only mode** | `use_lidar=False` skips the LiDAR pipeline |
| **Handling order ambiguity** | `shift_fixed_num_sampled_points` absorbs directional ambiguity of polygons and lines |
