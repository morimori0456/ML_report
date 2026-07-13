---
title: "Survey of Localization Technologies for Autonomous Driving"
description: "A survey of the sensing and estimation techniques — GNSS/INS, wheel odometry, SLAM, and map matching — that produce a vehicle's pose for autonomous driving."
---

> Related: `ego_pose` generation pipeline in [nuscenes_dataset](VAD/nuscenes_dataset.md), LCF coordinate system in [ego_trajectory](VAD/ego_trajectory.md)

---

## Overview

Localization in autonomous driving — estimating "where the vehicle is right now" — requires high accuracy, high frequency, and low latency. Common target values are **lateral accuracy within ±10 cm** (needed for lane keeping) and an **update rate of 100 Hz or more** (to match the control loop).

No single sensor provides both accuracy and robustness, making **multi-sensor fusion (Sensor Fusion)** essential.

---

## 1. Sensor Characteristics Comparison

| Sensor | Accuracy | Update Rate | Strengths | Weaknesses |
|---|---|---|---|---|
| **GNSS (GPS)** | ~3m (standalone) / ~1cm (RTK) | 1–10 Hz | Absolute coordinates; no drift | Multipath; unusable in tunnels/indoors |
| **IMU** | High short-term accuracy | 100–1000 Hz | High frequency; no external disturbances | Integration error (drift) accumulates |
| **LiDAR odometry** | ~5cm (environment-dependent) | 10–20 Hz | High accuracy; weather-independent | Affected by dynamic objects |
| **Camera odometry** | ~10cm (environment-dependent) | 30 Hz | Low cost | Degrades with lighting changes or lack of texture |
| **RADAR odometry** | ~20cm | 10–20 Hz | Robust in poor weather | Low resolution |
| **HD map matching** | ~5cm | Depends on map updates | Absolute accuracy | Requires map freshness management |

---

## 2. Sensor Fusion Architectures

### 2-1. Kalman Filter (KF / EKF / UKF)

The most classical and proven fusion framework. Manages state variables (position, velocity, attitude) as probability distributions (mean + covariance), alternating between prediction and update steps.

```
[Prediction step]  Propagate state at high frequency using IMU integration
[Update step]      Correct errors using low-frequency sensors (GPS, LiDAR, etc.)

State vector x = [x, y, z, vx, vy, vz, roll, pitch, yaw, ...]
```

| Type | Characteristics | Use Case |
|---|---|---|
| KF | Linear systems only | Simple GPS+IMU fusion |
| **EKF** (Extended KF) | Approximates nonlinearity with first-order Taylor expansion | Most widely used in autonomous driving |
| UKF (Unscented KF) | Approximates nonlinearity with sigma points (higher accuracy) | Attitude estimation; high-precision applications |
| Error-state KF | Includes IMU bias and scale in the state | High-precision INS/GNSS fusion |

**EKF fundamental equations:**

```
Predict:  x̂⁻ = f(x̂, u)         (state transition function f = motion model)
          P⁻  = F·P·Fᵀ + Q       (covariance propagation; Q = process noise)

Update:   K   = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹   (Kalman gain)
          x̂   = x̂⁻ + K·(z - h(x̂⁻))       (correction with observation z)
          P   = (I - K·H)·P⁻                (covariance update)
```

### 2-2. Particle Filter (Monte Carlo Localization)

A non-parametric method that represents the state distribution as a **set of particles**. Can handle multimodal distributions (situations where the vehicle's location is ambiguous).

```
Each particle = a hypothetical position/attitude + likelihood weight
→ Weights are updated using observations (LiDAR scans, etc.)
→ Resampling removes low-likelihood particles
```

| Strengths | Weaknesses |
|---|---|
| Can represent multimodality and global uncertainty | High computational cost (proportional to particle count) |
| Handles the kidnapped robot problem (sudden position loss) | Rapidly becomes inefficient as dimensionality increases (curse of dimensionality) |

### 2-3. Graph-Based Optimization (Graph SLAM)

Represents sensor observations as a graph of "nodes (positions)" and "edges (relative displacement constraints)," then finds the most consistent trajectory via least-squares optimization.

```
Nodes: vehicle pose at each timestep
Edges: constraints from LiDAR odometry, GPS, loop detection, etc.

Optimization: minimize Σ ||e(xᵢ, xⱼ)||²
              (using libraries such as g2o / GTSAM)
```

**Loop closure:** Detects a cycle in the graph when revisiting a previously visited location, dramatically reducing accumulated error. The core technology in SLAM (Simultaneous Localization And Mapping).

---

## 3. LiDAR-Based Localization

### 3-1. NDT Matching (Normal Distributions Transform)

Divides the LiDAR point cloud into voxels, models each voxel's point distribution as a Gaussian, and finds the transform (RT matrix) that maximizes alignment with a new scan.

```
Per voxel: pre-compute mean μ and covariance Σ of the point cloud
Score function: Σ exp(-d²/2·dᵀ·Σ⁻¹·d)  (d = point displacement)
→ Optimize the RT that maximizes the score using Newton's method, etc.
```

- Standard in **Autoware.AI / Autoware.Universe**
- Lower computational cost than ICP (voxelization compresses the point cloud)
- Requires pre-building an **NDT map (voxelized)** as the prior map

### 3-2. ICP (Iterative Closest Point)

A classical method that iteratively associates nearest-neighbor point pairs between scan and map, and optimizes the transform.

```
1. Associate each scan point with its nearest map point
2. Solve for the RT that minimizes the distance of associated pairs
3. Repeat steps 1–2 until convergence
```

Weaknesses: sensitivity to initialization and high computational cost. Point-to-Plane ICP enables speedup and improved accuracy.

### 3-3. Location-Based LiDAR Localization (Map Matching)

Obtains the absolute position by matching scans acquired while driving against a pre-built HD LiDAR map (full scan or intensity map) in real time.

```
Offline: Build a LiDAR map of the entire area using a survey vehicle
Online:  Match the current scan to the map using NDT / ICP, etc.
         → Obtain global 6DoF pose
```

**Map freshness problem:** Construction or structural changes can make the map stale, causing match failures.

---

## 4. GNSS-Based Localization

### 4-1. Standard GNSS (GPS)

Triangulates position from the travel time of satellite signals. Horizontal accuracy ~3m, update rate 1–10Hz.
Significantly degrades in tunnels, under overpasses, and in dense urban canyons (multipath).

### 4-2. RTK-GNSS (Real-Time Kinematic)

High-precision positioning using **carrier phase differences** relative to a reference station (fixed point).

```
Reference station (known fixed point)
  ↓ Sends differential correction data (RTCM, etc.) via radio/LTE
Rover (ego vehicle)
  ↓ Cancels errors using received correction data
  → Horizontal accuracy ~1–2cm (Fix solution) / ~10cm (Float solution)
```

**Condition for Fix solution:** Integer ambiguity must be resolved. If satellites are blocked, the Fix can degrade to Float or Single solutions.

### 4-3. PPP (Precise Point Positioning)

A newer approach that achieves ~10cm accuracy standalone by receiving precise satellite orbit and clock correction data over the internet, without a reference station. Convergence time (~several minutes) is a limitation. In Japan, the augmentation signals (CLAS/SLAS) from "Michibiki (QZSS)" are available.

---

## 5. Camera-Based Localization

### 5-1. Visual Odometry (VO)

Estimates relative displacement by tracking feature points across consecutive frames (FAST, ORB, etc.).
Can be realized without LiDAR at low cost, but is vulnerable to scale ambiguity (monocular) and lighting changes.

### 5-2. Visual-Inertial Odometry (VIO)

Improves scale estimation and high-frequency motion estimation by tightly coupling camera VO with IMU (Tightly Coupled).

```
Representative implementations: VINS-Mono, OKVIS, ORB-SLAM3
```

### 5-3. Visual Localization (Visual Map Matching)

Obtains a 6DoF pose by matching feature points in the current frame against a pre-built feature point map (Structure from Motion, etc.).
Deep learning-based methods such as `NetVLAD` and `SuperGlue` have significantly improved accuracy and robustness.

---

## 6. Deep Learning-Based Localization

### 6-1. Direct Regression (PoseNet-style)

An end-to-end approach that directly regresses a 6DoF pose from a camera image.

```
PoseNet (2015): directly regresses position and attitude with CNN
MapNet:         adds temporal constraints
AtLoc:          focuses on important regions via Attention
```

Accuracy is inferior to classical methods, but map construction costs are low.

### 6-2. LiDAR Deep Learning Localization

- **PointNetVLAD**: Learns a place descriptor from point clouds (place recognition)
- **L3-Net / DCP**: Deep learning-based scan matching
- **Diff-SLAM**: End-to-end optimization of LiDAR SLAM via gradient-based methods

### 6-3. Neural Field (Neural Radiance Field / Gaussian Splatting)-Based

- **NeRF-SLAM / iNeRF**: Represents the scene as an implicit NeRF; localizes using rendering error
- **3D Gaussian Splatting**: Faster variant of NeRF with improved real-time capability
- High map compressibility and completeness, but computational cost and generalization remain challenges

---

## 7. Multi-Sensor Fusion Implementation Patterns

### 7-1. Loose Coupling

Each sensor **independently outputs a localization result (position, attitude)**, which are fused in an EKF.

```
GNSS    → position (x, y, z) → EKF
LiDAR   → position (x, y, z) → EKF
IMU     → velocity / attitude change → EKF (prediction step)
```

Easy to implement and modularize. Correlation information between sensor outputs is lost.

### 7-2. Tight Coupling

Directly fuses **raw sensor observations** (IMU accelerations, GPS pseudoranges, LiDAR point clouds, etc.). Maximally leverages information.

```
Raw IMU + GPS pseudoranges + LiDAR point cloud → joint optimization
```

Higher accuracy but complex to design and implement. Representative: GTSAM / Ceres Solver-based implementations.

### 7-3. General Fusion Pipeline Architecture (Autoware-style)

```
Sensor inputs
  │
  ├─ IMU (100-1000Hz)
  │     → EKF prediction step (state propagation at high frequency)
  │
  ├─ GNSS (1-10Hz)
  │     → EKF update step (correction with absolute position)
  │
  ├─ LiDAR (10-20Hz)
  │     → NDT matching → relative position
  │     → EKF update step
  │
  └─ EKF filter output (100Hz)
        = Localization result
        * nuScenes ego_pose generation follows a pipeline of this type
```

---

## 8. Accuracy Degradation Scenarios and Countermeasures

| Scenario | Affected Sensor | Countermeasure |
|---|---|---|
| Tunnel (GNSS blockage) | GNSS | IMU dead reckoning + LiDAR map matching |
| Dense urban canyons (multipath) | GNSS | Prioritize LiDAR matching; GNSS confidence scoring |
| Heavy snow / rain | LiDAR, camera | RADAR supplementation; alternative features for map matching |
| Road changes due to construction | HD map matching | Dynamic object removal; map difference detection |
| Sensor startup (cold start) | GNSS Fix, general | GNSS-aided initialization; global localization (particle filter) |
| High-speed driving | Camera (motion blur) | IMU-predictive compensation; shutter speed control |

---

## 9. Implementation Stack Comparison

| Stack | Localization Method | Fusion Approach | Notes |
|---|---|---|---|
| **Autoware.Universe** | NDT + GNSS | EKF (ekf_localizer) | ROS2-based; widely deployed in Japan |
| **Apollo (Baidu)** | NDT + GNSS | EKF | Large-scale deployment track record in China |
| **LOAM / LeGO-LOAM** | LiDAR Odometry + Mapping | Graph optimization | Simultaneous map building |
| **SLAM Toolbox (2D)** | LiDAR SLAM | Particle / Graph | Low-speed; indoor use |
| **Cartographer (Google)** | 2D/3D SLAM | Graph optimization | ROS integration |
| **KISS-ICP** | LiDAR odometry only | Sequential ICP | Extremely simple, fast, zero parameters |

---

## 10. Localization Accuracy Metrics

| Metric | Definition | Notes |
|---|---|---|
| **ATE** (Absolute Trajectory Error) | RMSE of absolute position error between estimated and GT trajectories | Suitable for evaluating loop closure |
| **RPE** (Relative Pose Error) | Relative displacement error over a fixed interval | Suitable for evaluating drift |
| **CEP (Circular Error Probable)** | Radius within which 50% of horizontal errors fall | Commonly used in GNSS accuracy specifications |
| **Availability** | Fraction of time localization is achieved within required accuracy | Important for real-world operational requirements |

---

## 11. Correspondence with nuScenes ego_pose

```
Data collection pipeline for nuScenes:

  Velodyne HDL32E (LiDAR)
    → Scan matching (NDT-like) → LiDAR odometry
  GNSS
    → UTM conversion → absolute position (loose or tight coupling)
  IMU (via CAN bus)
    → EKF prediction step

  → Offline EKF / graph optimization

  → Stored in the ego_pose table (accuracy ~10cm)
```

The accuracy of `gt_ego_his_trajs` / `gt_ego_fut_trajs` generated by VAD from `ego_pose` depends directly on the accuracy of this fusion localization pipeline.
In real-world deployment, a real-time localization stack of equivalent quality is required.

---

## Summary

| Use Case | Recommended Approach |
|---|---|
| Urban / open environments | NDT + RTK-GNSS + IMU (EKF fusion) |
| Japanese expressways with many tunnels | LiDAR odometry + IMU (GNSS supplementary) |
| Low-cost (camera-centric) | VIO (Visual-Inertial Odometry) + GNSS |
| High-precision / all environments | Tight-coupled LiDAR-IMU-GNSS + graph optimization |
| Research / emerging technologies | NeRF-SLAM / 3D Gaussian Splatting |

In localization, **fusion architecture and fallback design for degradation scenarios** — more than the performance of any individual sensor — determine real-world operational quality.
