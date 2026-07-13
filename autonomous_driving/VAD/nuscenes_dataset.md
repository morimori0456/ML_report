---
title: "nuScenes Dataset Guide"
description: "A practical guide to the nuScenes multimodal autonomous-driving dataset — its sensors, schema, and common tasks."
---

## Overview

nuScenes is a **large-scale multimodal dataset for autonomous driving** built by Motional (formerly nuTonomy). Data was collected using a vehicle equipped with cameras, LiDAR, RADAR, and GPS/IMU, and is widely used for tasks including 3D object detection, tracking, map generation, and behavior prediction.

| Item | Value |
|---|---|
| Number of scenes | 1,000 scenes (20 seconds each) |
| Annotated samples | 40,000 samples (2 Hz) |
| Total frames (all sensors) | ~1.4 million frames |
| 3D bounding boxes | ~1.4 million |
| Recording cities | Boston (USA) and Singapore (Singapore), 2 locations each |
| Data size (full) | ~350 GB |

---

## 1. Sensor Configuration

```
LIDAR_TOP          : Velodyne HDL32E 32-line LiDAR (top center; 20 Hz)
CAM_FRONT          : Front, 70° FOV, 1600×900
CAM_FRONT_RIGHT    : Front-right, 70° FOV
CAM_FRONT_LEFT     : Front-left, 70° FOV
CAM_BACK           : Rear, 110° FOV
CAM_BACK_RIGHT     : Rear-right, 70° FOV
CAM_BACK_LEFT      : Rear-left, 70° FOV
RADAR_FRONT        : Front millimeter-wave radar
RADAR_FRONT_RIGHT  : Front-right millimeter-wave radar
RADAR_FRONT_LEFT   : Front-left millimeter-wave radar
RADAR_BACK_RIGHT   : Rear-right millimeter-wave radar
RADAR_BACK_LEFT    : Rear-left millimeter-wave radar
IMU / GPS          : Acquired via CAN bus
```

The 6 cameras provide **360° horizontal field of view**. Keyframes synchronized with LiDAR samples are labeled at 2 Hz (0.5-second intervals); between each keyframe, up to 10 unlabeled sweeps exist.

---

## 2. Data Hierarchy

nuScenes data is organized in 6 layers.

```
scene
  └─ sample                 ← keyframe (2 Hz, with annotations)
       ├─ sample_data       ← raw data token per sensor
       │    ├─ LIDAR_TOP    ← .bin point cloud file
       │    └─ CAM_*/RADAR_*
       ├─ sample_annotation ← 3D bounding box annotation per instance
       │    └─ instance → category
       └─ ego_pose          ← ego vehicle position and pose (global coordinates)
```

**Key Table Relationships**

| Table | Key Information |
|---|---|
| `scene` | `first_sample_token`, `nbr_samples`, `log_token` |
| `sample` | `timestamp`, `scene_token`, `prev/next` |
| `sample_data` | `filename`, `calibrated_sensor_token`, `ego_pose_token` |
| `calibrated_sensor` | `sensor2ego_translation/rotation`, `camera_intrinsic` |
| `sample_annotation` | `translation`, `size`, `rotation`, `num_lidar_pts`, `prev/next` |
| `instance` | `category_token`, `nbr_annotations` (frame count) |
| `log` | `location` (city name) |

### Coordinate Transform Chain

```
Global coordinates (EPSG:32618 / UTM)
  ↑ ego2global  (ego_pose table: translation + rotation)
Ego coordinates (ego vehicle center)
  ↑ lidar2ego   (calibrated_sensor table)
LIDAR_TOP coordinates  ← VAD's reference coordinate system
  ↑ sensor2lidar  (computed by composing each sensor's sensor2ego)
Individual camera / RADAR coordinates
```

### Localization Accuracy and Generation Method of ego_pose

The values in the `ego_pose` table are **not raw GPS coordinates; they are the result of fused localization post-processed offline by Motional after data collection**.

```
[Sensor inputs during recording]
  GNSS (GPS)                  ← UTM coordinate reference; accuracy ~3m
  IMU (accelerometer/gyro)    ← High frequency; good short-term accuracy; has drift
  LiDAR scan matching         ← Odometry via feature-point matching

         ↓ Offline fusion via EKF (Extended Kalman Filter), etc.

  translation / rotation in ego_pose table
  Accuracy: ~10cm order (urban open environments)
```

#### Handling of ego_pose in VAD Code

VAD's `vad_nuscenes_converter.py` **simply reads this post-processed value directly** via `nusc.get('ego_pose', ...)` and performs no additional filtering in the code.

Furthermore, in `get_data_info()`, the `pos` / `orientation` read from the CAN bus are **overwritten** with the `ego_pose` values:

```python
# nuscenes_vad_dataset.py L1376–1377
can_bus[:3] = translation   # overwrite with ego_pose.translation (raw CAN bus position discarded)
can_bus[3:7] = rotation     # overwrite with ego_pose.rotation
```

The remaining CAN bus fields (acceleration, angular velocity, speed, etc.; `[7:]` onwards) are not overwritten, and raw vehicle sensor values are used as-is.

| Field | Data Source | Accuracy Level |
|---|---|---|
| `can_bus[:3]` position | `ego_pose` (fused localization) | ~10cm |
| `can_bus[3:7]` attitude | `ego_pose` (fused localization) | ~0.1° |
| `can_bus[7:]` acceleration, speed, etc. | Raw CAN bus values | Sensor-dependent |
| `gt_ego_his/fut_trajs` trajectories | Derived from `ego_pose` | ~10cm |

#### Correspondence with Real-World Deployment (Robotaxi)

Since nuScenes `ego_pose` is generated offline after data collection, it is unavailable during inference (online).
In real-world deployment, the output of a real-time localization stack (NDT-matching + EKF, etc.) must serve as the equivalent of this value.
→ See [localization_tech](../localization_tech.md) for details.

---

## 3. Annotations

### 3-1. Detection Categories (10 Classes)

| Class | Description |
|---|---|
| `car` | Passenger car |
| `truck` | Truck |
| `bus` | Bus (large vehicle) |
| `trailer` | Trailer |
| `construction_vehicle` | Construction vehicle |
| `pedestrian` | Pedestrian |
| `motorcycle` | Motorcycle |
| `bicycle` | Bicycle |
| `traffic_cone` | Traffic cone |
| `barrier` | Fence / barricade |

The VAD converter maps original category names (e.g., `vehicle.car`) to these class names through `NuScenesDataset.NameMapping` (`vad_nuscenes_converter.py` L339).

### 3-2. Attributes (9 Types)

| Attribute | Applicable Objects |
|---|---|
| `cycle.with_rider` | bicycle/motorcycle |
| `cycle.without_rider` | bicycle/motorcycle |
| `pedestrian.moving` | pedestrian |
| `pedestrian.standing` | pedestrian |
| `pedestrian.sitting_lying_down` | pedestrian |
| `vehicle.moving` | car/truck/bus, etc. |
| `vehicle.parked` | car/truck/bus, etc. |
| `vehicle.stopped` | car/truck/bus, etc. |
| `None` | traffic_cone/barrier |

### 3-3. 3D Bounding Box Format

```python
gt_boxes: np.ndarray  # shape [N, 7]
# [x, y, z, width, length, height, yaw]
# Coordinates: LIDAR_TOP coordinate system
# yaw: SECOND format = -(nuScenes_yaw + π/2)

gt_velocity: np.ndarray  # shape [N, 2]
# Global velocity → converted to LiDAR coordinates
```

Boxes with `num_lidar_pts` / `num_radar_pts` of 0 can be excluded from evaluation with `valid_flag=False`.

---

## 4. CAN Bus Data

Sensor data provided by the `NuScenesCanBus` API.

```python
can_bus_18d = [
    pos[0], pos[1], pos[2],          # position (3)
    orientation[x,y,z,w],            # quaternion (4)
    # Remaining fields from the pose message (10):
    # accel(3), rotation_rate(3), vel(3), etc.
    0.0, 0.0                         # heading angle (rad, deg) — overwritten later
]  # 18 dimensions total
```

In VAD, this 18-dimensional vector is used as input for the positional encoding of BEVFormer's Temporal Self-Attention. For scenes recorded in Singapore (left-hand traffic), the steering sign is inverted (`vad_nuscenes_converter.py` L492).

---

## 5. Map Data (NuScenesMap)

HD maps for 4 cities are provided.

| Map Name | City |
|---|---|
| `boston-seaport` | Boston, MA |
| `singapore-hollandvillage` | Singapore |
| `singapore-onenorth` | Singapore |
| `singapore-queenstown` | Singapore |

### Map Layer List

| Layer Type | Layer Name | Geometry |
|---|---|---|
| Polygon | `road_segment`, `lane`, `drivable_area`, `ped_crossing`, `walkway`, `stop_line`, `carpark_area` | Polygon |
| Line | `road_divider`, `lane_divider`, `traffic_light` | LineString |

### Layers Used in VAD

```python
# VectorizedLocalMap.CLASS2LABEL
{
    'road_divider':  0,  # → divider class
    'lane_divider':  0,  # → divider class
    'ped_crossing':  1,  # → ped_crossing class
    'contours':      2,  # polygon boundary of road_segment + lane → boundary class
}
```

Maps are not saved in offline pkl files; they are generated at every iteration using `get_patch_coord()` (online generation). Patch size is determined from `pc_range` (in VAD_base: [-51.2, -51.2] → [51.2, 51.2] m = 102.4×102.4 m).

---

## 6. Dataset Splits

| Split | Scenes | Samples |
|---|---|---|
| `v1.0-trainval` train | 700 | 28,130 |
| `v1.0-trainval` val | 150 | 6,019 |
| `v1.0-test` | 150 | No labels |
| `v1.0-mini` train | 8 | 323 |
| `v1.0-mini` val | 2 | 81 |

Scene name lists can be retrieved from the `nuscenes.utils.splits` module via `splits.train`, `splits.val`, etc.

---

## 7. VAD Custom Extended Annotations

Extended information computed in `_fill_trainval_infos()` of `vad_nuscenes_converter.py` that is not part of the standard nuScenes annotations.

### 7-1. Agent Future Trajectories (`fut_ts=6`; 3 seconds ahead)

```python
gt_agent_fut_trajs:  # [N, 6, 2]  sequential offsets (LiDAR coordinates)
gt_agent_fut_masks:  # [N, 6]     valid step flags
gt_agent_fut_yaw:    # [N, 6]     yaw difference per step
gt_agent_fut_goal:   # [N]        goal direction class (0-7: 45° increments; 9: stationary)
```

**Goal direction class determination logic:**

```python
coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
if coord_diff.max() < 1.0:
    gt_fut_goal[i] = 9  # stationary
else:
    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # direction class 0-7
```

### 7-2. Agent LCF Features

```python
gt_agent_lcf_feat:  # [N, 9]
# [x, y, yaw, vx, vy, width, length, height, category_idx]
```

### 7-3. Ego Vehicle Trajectory

```python
gt_ego_his_trajs:  # [2, 2]  sequential offsets for past 2 frames (LiDAR coordinate xy)
gt_ego_fut_trajs:  # [6, 2]  sequential offsets for future 6 frames (LiDAR coordinate xy)
gt_ego_fut_masks:  # [6]
gt_ego_fut_cmd:    # [3]  one-hot: [Turn Right, Turn Left, Go Straight]
```

**Driving command determination (LiDAR coordinates):**

```python
if ego_fut_trajs[-1][0] >= 2:   # x direction (lateral) is positive → turn right
    command = [1, 0, 0]
elif ego_fut_trajs[-1][0] <= -2: # x direction is negative → turn left
    command = [0, 1, 0]
else:
    command = [0, 0, 1]          # go straight
```

### 7-4. Ego LCF Features

```python
gt_ego_lcf_feat:  # [9]
# [vx, vy, ax, ay, yaw angular velocity, length(4.084m), width(1.85m), longitudinal speed, steering curvature]
```

Steering curvature is computed from the bicycle model `v = L * tan(δ)` (`L=2.588` wheelbase).

---

## 8. Evaluation Metrics

### 8-1. nuScenes Detection Score (NDS)

$$\text{NDS} = \frac{1}{10}\left[5 \cdot \text{mAP} + \sum_{\text{TP}} (1 - \min(1, \text{err}))\right]$$

TP metrics (5 types):

| Metric | Description |
|---|---|
| ATE | Average Translation Error (m) |
| ASE | Average Scale Error (1 - IoU) |
| AOE | Average Orientation Error (rad) |
| AVE | Average Velocity Error (m/s) |
| AAE | Average Attribute Error |

### 8-2. VAD Custom Evaluation Configuration (`vad_nusc_detection_cvpr_2019.json`)

Unlike the standard nuScenes evaluation, this uses an asymmetric detection range in the x and y directions.

```json
{
  "class_range_x": {"car": 30, ...},   // longitudinal ±30m
  "class_range_y": {"car": 15, ...},   // lateral ±15m
  "dist_ths": [0.5, 1.0, 2.0, 4.0],
  "dist_th_tp": 2.0
}
```

Standard nuScenes evaluation uses a circle of radius `r`, but VAD evaluates within a rectangular region biased toward the forward field of view.

---

## 9. pkl File Structure Summary

Keys stored per sample in `{prefix}_infos_temporal_train/val.pkl`:

```python
info = {
    # Sensors
    'lidar_path':                str,           # LIDAR_TOP .bin
    'cams':                      dict,          # 6-camera info
    'sweeps':                    list,          # up to 10 sweeps
    # Coordinate transforms
    'lidar2ego_translation':     [3],
    'lidar2ego_rotation':        [4],           # quaternion
    'ego2global_translation':    [3],
    'ego2global_rotation':       [4],
    # Temporal
    'token':                     str,
    'prev':                      str,           # previous frame token
    'next':                      str,           # next frame token
    'scene_token':               str,
    'frame_idx':                 int,           # index within scene
    'timestamp':                 int,           # microseconds
    # Metadata
    'can_bus':                   [18],
    'map_location':              str,           # one of 4 cities
    'fut_valid_flag':            bool,          # whether 6 future frames exist
    # Annotations (standard nuScenes)
    'gt_boxes':                  [N, 7],
    'gt_names':                  [N],
    'gt_velocity':               [N, 2],
    'valid_flag':                [N],
    'num_lidar_pts':             [N],
    # Annotations (VAD custom extensions)
    'gt_agent_fut_trajs':        [N, 6, 2],
    'gt_agent_fut_masks':        [N, 6],
    'gt_agent_fut_yaw':          [N, 6],
    'gt_agent_fut_goal':         [N],
    'gt_agent_lcf_feat':         [N, 9],
    'gt_ego_his_trajs':          [2, 2],
    'gt_ego_fut_trajs':          [6, 2],
    'gt_ego_fut_masks':          [6],
    'gt_ego_fut_cmd':            [3],
    'gt_ego_lcf_feat':           [9],
}
```

---

## 10. nuScenes Python API Key Methods

| Method | Purpose |
|---|---|
| `nusc.get(table, token)` | Retrieve a record from any table |
| `nusc.get_sample_data(token)` | Get path, boxes, and camera intrinsics |
| `nusc.box_velocity(token)` | Compute velocity in global coordinates |
| `NuScenesMap.get_patch_coord(patch_box, patch_angle)` | Shapely Polygon for a specified patch |
| `NuScenesMapExplorer._get_layer_line()` | Retrieve line-type layers |
| `NuScenesMapExplorer._get_layer_polygon()` | Retrieve polygon-type layers |
| `NuScenesCanBus.get_messages(scene, channel)` | Retrieve CAN bus messages |
