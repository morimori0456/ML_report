# Camera Extrinsic Calibration Complete Guide (Including Rectification)

> Related: sensor fusion in [[localization_tech]], `calibrated_sensor` in [[nuscenes_dataset]] (extrinsic parameters for each sensor)
> Demo: `extrinsic_calibration_demo.ipynb` (numpy only; no GPU required) / `extrinsic_calibration_opencv.ipynb` (practical OpenCV)

---

## 0. Goal of This Document

To resolve confusion about extrinsic calibration. By the end, you should be able to:

1. Explain what **extrinsic parameters `[R|t]`** represent in terms of coordinate systems
2. Write the projection equation `x = K[R|t]X` from scratch and state the role of each matrix
3. Understand **methods for estimating `[R|t]`** (PnP/DLT; inter-sensor calibration)
4. Describe end-to-end the flow of **epipolar geometry → stereo calibration → rectification**
5. Explain "why rectification is necessary and what it does" (preprocessing for disparity computation)

Most confusion originates from ambiguous coordinate system definitions. Make sure to solidify §1 first.

---

## 1. First, Fully Clarify the Coordinate Systems (This Is 90% of the Battle)

Four coordinate systems appear in calibration:

| Coordinate System | Symbol | Unit | Description |
|---|---|---|---|
| World coordinate system | `X_w = (Xw,Yw,Zw)` | m | Global reference. Checkerboard corners are often taken as the origin |
| Camera coordinate system | `X_c = (Xc,Yc,Zc)` | m | Camera optical center at origin; optical axis is +Z; right is +X; down is +Y |
| Image coordinate system (normalized) | `x_n = (x,y)` | dimensionless | Ideal projection plane obtained by dividing camera coordinates by Z (focal length = 1) |
| Pixel coordinate system | `u = (u,v)` | px | Origin at top-left of image; right is +u; down is +v |

**The decisive difference between intrinsic and extrinsic:**

- **Extrinsic parameters `[R|t]`** = **rigid body transform** from world coordinate system → camera coordinate system (where and in what orientation the camera is placed in the world)
- **Intrinsic parameters `K`** = **projection** from camera coordinate system → pixel coordinate system (focal length, principal point; camera-specific and scene-independent)

```
X_w ──[R|t] (extrinsic)──► X_c ──K (intrinsic)──► u
       camera pose                 lens properties
```

> Note: In the autonomous driving context, "extrinsic parameters" refer to two things:
> - **Camera ↔ World (or ego/vehicle body)**: Transform that places sensors in the vehicle body coordinate system (the `translation`/`rotation` in nuScenes `calibrated_sensor`)
> - **Camera ↔ other sensors** (camera-camera, camera-LiDAR): Relative pose between sensors. Also `[R|t]`.
> Both are mathematically the same "rigid body transform `[R|t]`."

---

## 2. Mathematics of Rigid Body Transform `[R|t]`

Transforming world point `X_w` to camera coordinates `X_c`:

```
X_c = R · X_w + t

  R ∈ SO(3)   3×3 rotation matrix (orthogonal; det=1)
  t ∈ ℝ³      translation vector
```

In homogeneous coordinates:

```
⎡X_c⎤   ⎡ R   t ⎤ ⎡X_w⎤
⎢ 1 ⎥ = ⎢ 0ᵀ  1 ⎥ ⎢ 1 ⎥
⎣   ⎦   ⎣       ⎦ ⎣   ⎦
```

### 2.1 "Camera Position" and t Are Not the Same (The Most Common Misconception)

`t` is **not** the "camera position." Its relationship to camera center `C` (in world coordinates) is:

```
X_c = R(X_w - C)  =  R·X_w - R·C
∴ t = -R·C   ⟺   C = -Rᵀ·t
```

- `t` = the position of the world origin as seen in camera coordinates
- `C = -Rᵀt` = the world coordinates of the camera center (this is the intuitive "position")

This confusion is the typical cause of "I can't understand extrinsic calibration." **`t` is a translation, not a position.**

### 2.2 Inverse Transform (Camera → World)

```
X_w = Rᵀ·X_c - Rᵀ·t = Rᵀ·X_c + C
```

The inverse of `R` is `Rᵀ` (since it is an orthogonal matrix). Frequently used in implementation.

---

## 3. Projection: Combining with Intrinsic `K` to Reach the Image

### 3.1 Pinhole Model

Camera coordinates `X_c=(Xc,Yc,Zc)` → normalized image coordinates:

```
x = Xc / Zc,   y = Yc / Zc      (dividing by Z = perspective projection)
```

→ To pixels (intrinsic matrix `K`):

```
⎡u⎤   ⎡fx  s  cx⎤ ⎡x⎤
⎢v⎥ = ⎢ 0 fy  cy⎥ ⎢y⎥        K = intrinsic parameters
⎣1⎦   ⎣ 0  0   1⎦ ⎣1⎦

  fx, fy : focal lengths (in pixels)
  cx, cy : principal point (near image center)
  s      : skew (typically 0)
```

### 3.2 Full Projection Matrix P

Combining extrinsic and intrinsic:

```
s·⎡u⎤
  ⎢v⎥ = K · [R | t] · ⎡X_w⎤ = P · X_w (homogeneous)
  ⎣1⎦                  ⎢ 1 ⎥
                       ⎣   ⎦

P = K[R|t]   (3×4 projection matrix; scale-ambiguous)
```

- `P` is 3×4 with 11 degrees of freedom (12 elements, scale-ambiguous)
- Can be decomposed into `K` (intrinsic, 5 DoF) and `[R|t]` (extrinsic, 6 DoF = 3 rotation + 3 translation)
- **Calibration means estimating `K` and `[R|t]` from point correspondences**

→ Hands-on: "Projection pipeline" and "how the image moves as R, t are varied" in the `demo`

---

## 4. Estimating Extrinsic Parameters (Single Camera: PnP)

When `K` is known, the problem of finding `[R|t]` from **correspondences between 3D points and their image projections** is called **PnP (Perspective-n-Point)**.

### 4.1 Intuitive Solution via DLT (Direct Linear Transform)

Each correspondence `(X_w, u)` satisfies `s·u = P·X_w`. Eliminating the scale `s` yields 2 linear equations per point:

```
u·(p₃ᵀX) - (p₁ᵀX) = 0
v·(p₃ᵀX) - (p₂ᵀX) = 0      (p₁,p₂,p₃ are rows of P)
```

Stacking `n≥6` points gives `A·p = 0` where `p` is the 12 elements of P. The **right singular vector corresponding to the smallest singular value** of `A` (2n×12) via SVD gives `p`.
This yields `P`, and since `K` is known:

```
[R|t] = K⁻¹ · P
```

However, the resulting `R` may not be a strictly orthogonal matrix due to numerical errors → **orthogonalize with SVD**: `R = U·Vᵀ` (replace Σ with I in `R=UΣVᵀ`). Correct `t` scale to match the scale of `R`.

### 4.2 Practical Methods

- `cv2.solvePnP` (iterative, P3P, EPnP, etc.) + `cv2.solvePnPRansac` (outlier removal)
- Estimate from correspondences between known 3D grids and detected 2D corners using checkerboard / AprilTag / ChArUco
- Accuracy metric: **reprojection error**: mean of `||u_observed - π(K[R|t]X)||` (px). **0.5px or less** is the target

→ Hands-on: "Solve PnP with DLT → compare with ground truth" in the `demo`; `solvePnP` in `opencv`

---

## 5. Two Cameras: Epipolar Geometry

Extrinsic calibration for stereo or multi-camera setups is the problem of finding the **relative pose `[R|t]` between cameras**. The foundation is epipolar geometry.

### 5.1 Essential Matrix E

For the relative rotation `R` and translation `t` between two cameras (cam1→cam2), normalized coordinates `x₁,x₂` satisfy:

```
x₂ᵀ · E · x₁ = 0      (epipolar constraint)

E = [t]× · R          ([t]× is the skew-symmetric matrix of t = cross-product matrix)

       ⎡ 0   -tz   ty⎤
[t]× = ⎢ tz   0  -tx⎥
       ⎣-ty   tx   0 ⎦
```

`E` tells you **which line (epipolar line)** a point in one camera must lie on in the other image. `R, t` (translation is scale-ambiguous) can be decomposed from `E`.

### 5.2 Fundamental Matrix F

Writing the same constraint in pixel coordinates `u₁,u₂`:

```
u₂ᵀ · F · u₁ = 0

F = K₂⁻ᵀ · E · K₁⁻¹
```

- `F` incorporates all intrinsic and extrinsic information and can be estimated from pixel correspondences alone (8-point algorithm, etc.)
- Epipolar lines: `l₂ = F·u₁` (on cam2), `l₁ = Fᵀ·u₂` (on cam1)

### 5.3 Stereo Calibration

Simultaneously estimates both cameras' `K₁, K₂` (and distortion) along with the **inter-camera `R, t`**.
Capture a checkerboard simultaneously with both cameras → `cv2.stereoCalibrate`.
The output `R, t` is "the pose of cam2 as seen from cam1's coordinate system" = extrinsic parameters.

→ Hands-on: "Visualize epipolar lines" in the `demo`; `stereoCalibrate` in `opencv`

---

## 6. Rectification (Image Alignment) — The Main Topic

### 6.1 Why It Is Necessary

To **compute depth** from stereo, you need to find corresponding points between left and right images (correspondence search = matching). In a general configuration, corresponding points can be anywhere in the image, making it a costly and unstable 2D search.

The **epipolar constraint** says corresponding points always lie on epipolar lines. Therefore:

> **Rectification = the process of transforming left and right images so that all epipolar lines become "horizontal lines at the same height."**

This reduces correspondence search to **a 1D search along the same row (same v)**, making disparity computation fast and stable.

```
Before rectification: corresponding points lie on diagonal epipolar lines (2D search)
After rectification:  corresponding points share the same row v (v matches left-right; 1D search)
```

### 6.2 What It Does (Mathematics)

**Virtually rotate** both cameras to a **common orientation `R_rect`**, making their image planes coplanar and parallel.
Since only rotation is applied (camera centers do not move), this can be achieved by applying a **homography (projective transform)** to each camera's image:

```
Rectifying homography:
  H_i = K_new · R_rect_i · K_i⁻¹

New pixel u_rect = H_i · u   (warp the image)
```

How to compute `R_rect` (Bouguet/Fusiello approach):

```
New X axis = baseline direction (vector connecting 2 camera centers)  e₁ = t/||t||
New Y axis = normalize(old optical axis × e₁)                          e₂
New Z axis = e₁ × e₂                                                   e₃
R_rect = [e₁ᵀ; e₂ᵀ; e₃ᵀ]
```

Key insight: **By aligning the new X axis with the baseline**, the two camera centers have no difference along the new Y and Z directions, so corresponding points' vertical coordinate `v` matches between left and right — epipolar lines become horizontal.

### 6.3 From Disparity to Depth

After rectification, the horizontal coordinate difference for the same point in the left and right images is called **disparity d**:

```
d = u_left - u_right   (px; d>0)

Depth Z = f · B / d

  f : focal length after rectification (px; same for both cameras)
  B : baseline length (m) = ||t||
```

- Larger disparity = closer; smaller disparity = farther
- `Z = fB/d` is triangulation itself. Increasing `B` improves far-range accuracy (but increases near-range blind spot)
- In OpenCV, `cv2.stereoRectify` returns reprojection matrix `Q`; `cv2.reprojectImageTo3D(disparity, Q)` reconstructs the full 3D point cloud

### 6.4 OpenCV Processing Pipeline

```
stereoCalibrate        → K1,K2,dist1,dist2,R,t
stereoRectify          → R1,R2 (rotation per camera), P1,P2 (new projection matrices), Q (reprojection matrix)
initUndistortRectifyMap→ remap maps per camera (undistortion + rectification combined)
remap                  → warp actual images (images with row-aligned correspondences)
StereoSGBM.compute     → disparity map
reprojectImageTo3D(Q)  → depth / point cloud
```

→ Hands-on: "Verify that v matches between left and right after rectification" and "disparity → depth" in the `demo`; full `stereoRectify` pipeline in `opencv`

---

## 7. Autonomous Driving Context: Inter-Sensor Extrinsic Calibration

### 7.1 Camera ↔ LiDAR

To fuse LiDAR point clouds with camera images (colorizing point clouds, assisting 3D object detection), the **`[R|t]` from LiDAR coordinate system → camera coordinate system** is required.

```
u = K · [R_lidar→cam | t_lidar→cam] · X_lidar
```

Estimation methods:
- **Target-based**: Observe a checkerboard or special target with both camera and LiDAR; optimize from plane/corner correspondences
- **Targetless**: Match image edges with depth discontinuities in the point cloud (e.g., mutual information maximization). Useful for re-calibration
- **Motion-based (hand-eye)**: Solve `AX=XB` from each sensor's ego-motion `A_i, B_i`

### 7.2 Handling in nuScenes

Each sensor in `calibrated_sensor` holds `translation` (t) and `rotation` (quaternion) **with respect to the ego (vehicle body) coordinate system**.
Camera projection chains rigid body transforms as "global → ego → sensor → image" (see [[nuscenes_dataset]]).
The key point is that **chaining (composing) extrinsic parameters** lets you construct transforms between any coordinate systems:

```
T_global→cam = T_ego→cam · T_global→ego
(each T is a 4×4 [R|t]; composed via matrix multiplication)
```

---

## 8. Evaluation and Pitfalls

| Pitfall | Symptom | Countermeasure |
|---|---|---|
| Misunderstanding `t` as position | Sign / direction inverted | Always keep in mind `C=-Rᵀt` (§2.1) |
| Confusing coordinate axis orientation (+Y down/up, +Z forward/backward) | Upside-down or reversed depth | Document each sensor's convention explicitly |
| R not strictly orthogonal | Residual reprojection error | Orthogonalize with SVD (§4.1) |
| Checkerboard always flat and frontal | Unstable estimation (degenerate configuration) | Capture at diverse distances, angles, and image positions |
| Forgetting distortion correction | Large errors in periphery | Also estimate distortion coefficients in intrinsic calibration; apply correction first |
| Vertical misalignment after rectification | Disparity matching fails | Verify `|v_l - v_r|` after rectification (done in demo) |
| Baseline too short / too long | Insufficient far-range precision / near-range blind spot | Design B according to use case; estimate error with `Z=fB/d` |

The fundamental evaluation metric is **reprojection error (px)**. For stereo, also inspect the **epipolar error after rectification** (vertical coordinate difference between corresponding points).

---

## 9. Summary — One-Page Recap

```
Coordinate systems:  X_w ──[R|t] (extrinsic)──► X_c ──K (intrinsic)──► u
                            t is translation (≠ position). Position is C=-Rᵀt

Single camera:  PnP/DLT   known K + 3D-2D correspondences → [R|t]   (Ap=0 via SVD, K⁻¹P, orthogonalize R)
Two cameras:    E=[t]×R,  F=K₂⁻ᵀEK₁⁻¹,  epipolar constraint x₂ᵀEx₁=0
Stereo:         stereoCalibrate → inter-camera R,t
Rectification:  H_i=K_new R_rect K_i⁻¹ warps images → horizontalizes epipolar lines
                → disparity d → depth Z=fB/d
Autonomous driving: camera-LiDAR / nuScenes calibrated_sensor also use the same chained [R|t]
```

**Learning roadmap:**
1. Experience projection, PnP, epipolar geometry, and rectification in `extrinsic_calibration_demo.ipynb` (numpy; runs in this environment)
2. Run real APIs (`calibrateCamera` / `solvePnP` / `stereoCalibrate` / `stereoRectify`) in `extrinsic_calibration_opencv.ipynb`
3. Measure reprojection error on your own stereo images or nuScenes data

---

## References

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision" (the bible; Chapter 6: camera matrices, Chapter 9: epipolar geometry, Chapter 11: estimation)
- Zhang, "A Flexible New Technique for Camera Calibration" (Zhang's method; the standard for intrinsic calibration)
- Fusiello et al., "A compact algorithm for rectification of stereo pairs"
- OpenCV docs: Camera Calibration and 3D Reconstruction (`calibrateCamera`, `stereoRectify`)
- Lepetit et al., "EPnP: An Accurate O(n) Solution to the PnP Problem"
