# nuScenes データセット解説

## 概要

nuScenes は Motional（旧 nuTonomy）が構築した**自動運転向け大規模マルチモーダルデータセット**。カメラ・LiDAR・RADAR・GPS/IMU を搭載した車両で収集されており、3D 物体検出・追跡・地図生成・行動予測など多数のタスクで広く使われている。

| 項目 | 数値 |
|---|---|
| シーン数 | 1000 シーン（各 20 秒） |
| アノテーション済みサンプル数 | 40,000 サンプル（2Hz） |
| 全フレーム数（全センサー） | 約 140 万フレーム |
| 3D バウンディングボックス数 | 約 140 万 |
| 収録都市 | Boston（米）・Singapore（シンガポール）各2ロケーション |
| データサイズ（フル） | 約 350 GB |

---

## 1. センサー構成

```
LIDAR_TOP          : Velodyne HDL32E 32線LiDAR（上部中央、20Hz）
CAM_FRONT          : 前方 70° FOV、1600×900
CAM_FRONT_RIGHT    : 右前方 70° FOV
CAM_FRONT_LEFT     : 左前方 70° FOV
CAM_BACK           : 後方 110° FOV
CAM_BACK_RIGHT     : 右後方 70° FOV
CAM_BACK_LEFT      : 左後方 70° FOV
RADAR_FRONT        : 前方ミリ波レーダー
RADAR_FRONT_RIGHT  : 右前方ミリ波レーダー
RADAR_FRONT_LEFT   : 左前方ミリ波レーダー
RADAR_BACK_RIGHT   : 右後方ミリ波レーダー
RADAR_BACK_LEFT    : 左後方ミリ波レーダー
IMU / GPS          : CAN バス経由で取得
```

6カメラで**水平360°の視野角**を確保。LiDAR サンプルに同期したキーフレームは 2Hz（0.5秒間隔）でラベル付けされ、各キーフレーム間には非ラベルスイープ（最大10枚）が存在する。

---

## 2. データ階層構造

nuScenes のデータは以下の6層で構成される。

```
scene
  └─ sample                 ← キーフレーム（2Hz、アノテーション付き）
       ├─ sample_data       ← 各センサーの生データトークン
       │    ├─ LIDAR_TOP    ← .bin 点群ファイル
       │    └─ CAM_*/RADAR_*
       ├─ sample_annotation ← 各インスタンスの3Dボックスアノテーション
       │    └─ instance → category
       └─ ego_pose          ← 自車の位置・姿勢（グローバル座標）
```

**主要テーブルの関係**

| テーブル | キー情報 |
|---|---|
| `scene` | `first_sample_token`, `nbr_samples`, `log_token` |
| `sample` | `timestamp`, `scene_token`, `prev/next` |
| `sample_data` | `filename`, `calibrated_sensor_token`, `ego_pose_token` |
| `calibrated_sensor` | `sensor2ego_translation/rotation`, `camera_intrinsic` |
| `sample_annotation` | `translation`, `size`, `rotation`, `num_lidar_pts`, `prev/next` |
| `instance` | `category_token`, `nbr_annotations`（フレーム数） |
| `log` | `location`（都市名） |

### 座標変換チェーン

```
グローバル座標（EPSG:32618 / UTM）
  ↑ ego2global  (ego_pose テーブル: translation + rotation)
ego 座標（自車中心）
  ↑ lidar2ego   (calibrated_sensor テーブル)
LIDAR_TOP 座標  ← VAD の基準座標系
  ↑ sensor2lidar  (各センサーの sensor2ego から合成)
各カメラ / RADAR 座標
```

### ego_pose の測位精度と生成方法

`ego_pose` テーブルの値は **生の GPS 座標ではなく、Motional 社が収録後にオフライン処理した融合測位の結果**である。

```
[収録時センサー入力]
  GNSS (GPS)          ← UTM 座標系の基準、精度 ~3m
  IMU (加速度・ジャイロ) ← 高頻度・短期精度良・ドリフトあり
  LiDAR スキャンマッチング ← 特徴点照合による odometry

         ↓ EKF（拡張カルマンフィルタ）等でオフライン融合

  ego_pose テーブルの translation / rotation
  精度: ~10cm オーダー（都市部開放環境）
```

#### VAD コードにおける ego_pose の扱い

VAD の `vad_nuscenes_converter.py` は `nusc.get('ego_pose', ...)` でこの **後処理済みの値を直接読み取るだけ**であり、コード内に追加のフィルタ処理は存在しない。

さらに `get_data_info()` では、CAN バスから取り出した `pos` / `orientation` を `ego_pose` の値で**上書き**する：

```python
# nuscenes_vad_dataset.py L1376–1377
can_bus[:3] = translation   # ego_pose.translation で上書き（生 CAN bus 位置を破棄）
can_bus[3:7] = rotation     # ego_pose.rotation で上書き
```

CAN バスの残余フィールド（加速度・角速度・速度など `[7:]` 以降）は上書きされず、車両センサーの生値がそのまま使われる。

| フィールド | データソース | 精度レベル |
|---|---|---|
| `can_bus[:3]` 位置 | `ego_pose`（融合測位） | ~10cm |
| `can_bus[3:7]` 姿勢 | `ego_pose`（融合測位） | ~0.1° |
| `can_bus[7:]` 加速度・速度等 | CAN バス生値 | センサー依存 |
| `gt_ego_his/fut_trajs` 軌跡 | `ego_pose` 由来 | ~10cm |

#### 実運用（ロボタクシー）との対応

nuScenes の `ego_pose` は収録後オフラインで生成されるため、推論時（オンライン）には利用不可。  
実運用ではリアルタイム測位スタック（NDT-matching + EKF 等）からの出力をこの値に相当させる必要がある。  
→ 詳細は [[localization_tech]] 参照。

---

## 3. アノテーション

### 3-1. 検出カテゴリ（10クラス）

| クラス | 説明 |
|---|---|
| `car` | 乗用車 |
| `truck` | トラック |
| `bus` | バス（大型） |
| `trailer` | トレーラー |
| `construction_vehicle` | 建設車両 |
| `pedestrian` | 歩行者 |
| `motorcycle` | オートバイ |
| `bicycle` | 自転車 |
| `traffic_cone` | コーン |
| `barrier` | 柵・バリケード |

VADコンバーターでは `NuScenesDataset.NameMapping` を通じて元のカテゴリ名（例: `vehicle.car`）をこれらのクラス名にマッピングする（`vad_nuscenes_converter.py` L339）。

### 3-2. アトリビュート（9種類）

| アトリビュート | 適用対象 |
|---|---|
| `cycle.with_rider` | bicycle/motorcycle |
| `cycle.without_rider` | bicycle/motorcycle |
| `pedestrian.moving` | pedestrian |
| `pedestrian.standing` | pedestrian |
| `pedestrian.sitting_lying_down` | pedestrian |
| `vehicle.moving` | car/truck/bus 等 |
| `vehicle.parked` | car/truck/bus 等 |
| `vehicle.stopped` | car/truck/bus 等 |
| `None` | traffic_cone/barrier |

### 3-3. 3D バウンディングボックス形式

```python
gt_boxes: np.ndarray  # shape [N, 7]
# [x, y, z, width, length, height, yaw]
# 座標: LIDAR_TOP 座標系
# yaw: SECOND 形式 = -(nuScenes_yaw + π/2)

gt_velocity: np.ndarray  # shape [N, 2]
# グローバル速度 → LIDAR 座標に変換済み
```

`num_lidar_pts` / `num_radar_pts` が 0 のボックスは `valid_flag=False` として評価から除外できる。

---

## 4. CAN バスデータ

`NuScenesCanBus` API が提供するセンサーデータ。

```python
can_bus_18d = [
    pos[0], pos[1], pos[2],          # 位置 (3)
    orientation[x,y,z,w],            # 四元数 (4)
    # 以下は pose メッセージの残フィールド群 (10)
    # accel(3), rotation_rate(3), vel(3), ...等
    0.0, 0.0                         # heading角 (rad, deg) — 後で上書き
]  # 計18次元
```

VAD ではこの18次元ベクトルを BEVFormer の Temporal Self-Attention に使うポジションエンコーディングの入力として使用する。シンガポール収録シーンは左側通行のためステアリング符号を反転（`vad_nuscenes_converter.py` L492）。

---

## 5. 地図データ（NuScenesMap）

4都市のHDマップが提供される。

| マップ名 | 都市 |
|---|---|
| `boston-seaport` | Boston, MA |
| `singapore-hollandvillage` | Singapore |
| `singapore-onenorth` | Singapore |
| `singapore-queenstown` | Singapore |

### マップレイヤー一覧

| レイヤー種別 | レイヤー名 | 形状 |
|---|---|---|
| ポリゴン | `road_segment`, `lane`, `drivable_area`, `ped_crossing`, `walkway`, `stop_line`, `carpark_area` | Polygon |
| ライン | `road_divider`, `lane_divider`, `traffic_light` | LineString |

### VAD で使用するレイヤー

```python
# VectorizedLocalMap.CLASS2LABEL
{
    'road_divider':  0,  # → divider クラス
    'lane_divider':  0,  # → divider クラス
    'ped_crossing':  1,  # → ped_crossing クラス
    'contours':      2,  # road_segment + lane のポリゴン境界 → boundary クラス
}
```

マップはオフラインのpklには保存せず、`get_patch_coord()` で毎イテレーション実行時に生成する（オンライン生成）。パッチサイズは `pc_range` から決定される（VAD_base では [-51.2, -51.2] → [51.2, 51.2] m = 102.4×102.4 m）。

---

## 6. データセット分割

| スプリット | シーン数 | サンプル数 |
|---|---|---|
| `v1.0-trainval` train | 700 | 28,130 |
| `v1.0-trainval` val | 150 | 6,019 |
| `v1.0-test` | 150 | ラベルなし |
| `v1.0-mini` train | 8 | 323 |
| `v1.0-mini` val | 2 | 81 |

`nuscenes.utils.splits` モジュールから `splits.train`, `splits.val` 等でシーン名リストを取得できる。

---

## 7. VAD 独自拡張アノテーション

`vad_nuscenes_converter.py` の `_fill_trainval_infos()` で計算される、nuScenes 標準には含まれない拡張情報。

### 7-1. エージェント未来軌跡（`fut_ts=6`、3秒先）

```python
gt_agent_fut_trajs:  # [N, 6, 2]  逐次オフセット（LiDAR座標）
gt_agent_fut_masks:  # [N, 6]     有効ステップフラグ
gt_agent_fut_yaw:    # [N, 6]     各ステップのyaw差分
gt_agent_fut_goal:   # [N]        ゴール方向クラス（0-7: 45°刻み、9: 静止）
```

**ゴール方向クラスの判定ロジック：**

```python
coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
if coord_diff.max() < 1.0:
    gt_fut_goal[i] = 9  # 静止
else:
    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-7の方向クラス
```

### 7-2. エージェント LCF 特徴量

```python
gt_agent_lcf_feat:  # [N, 9]
# [x, y, yaw, vx, vy, width, length, height, category_idx]
```

### 7-3. 自車軌跡

```python
gt_ego_his_trajs:  # [2, 2]  過去2フレームの逐次オフセット（LiDAR座標xy）
gt_ego_fut_trajs:  # [6, 2]  未来6フレームの逐次オフセット（LiDAR座標xy）
gt_ego_fut_masks:  # [6]
gt_ego_fut_cmd:    # [3]  one-hot: [Turn Right, Turn Left, Go Straight]
```

**運転コマンドの判定（LiDAR座標）：**

```python
if ego_fut_trajs[-1][0] >= 2:   # x方向（横）が正 → 右折
    command = [1, 0, 0]
elif ego_fut_trajs[-1][0] <= -2: # x方向が負 → 左折
    command = [0, 1, 0]
else:
    command = [0, 0, 1]          # 直進
```

### 7-4. 自車 LCF 特徴量

```python
gt_ego_lcf_feat:  # [9]
# [vx, vy, ax, ay, yaw角速度, length(4.084m), width(1.85m), 縦速度, ステア曲率]
```

ステア曲率は `v = L * tan(δ)` の自転車モデルから計算（`L=2.588` ホイールベース）。

---

## 8. 評価指標

### 8-1. nuScenes Detection Score (NDS)

$$\text{NDS} = \frac{1}{10}\left[5 \cdot \text{mAP} + \sum_{\text{TP}} (1 - \min(1, \text{err}))\right]$$

TP メトリクス（5種）：

| メトリクス | 説明 |
|---|---|
| ATE | 平均位置誤差 (m) |
| ASE | 平均サイズ誤差 (1 - IoU) |
| AOE | 平均向き誤差 (rad) |
| AVE | 平均速度誤差 (m/s) |
| AAE | 平均アトリビュート誤差 |

### 8-2. VAD カスタム評価設定（`vad_nusc_detection_cvpr_2019.json`）

標準の nuScenes 評価と異なり、xy方向で非対称な検出範囲を使用する。

```json
{
  "class_range_x": {"car": 30, ...},   // 縦方向 ±30m
  "class_range_y": {"car": 15, ...},   // 横方向 ±15m
  "dist_ths": [0.5, 1.0, 2.0, 4.0],
  "dist_th_tp": 2.0
}
```

通常の nuScenes 評価は半径 `r` の円内で評価するが、VAD は前方視野に偏った長方形領域で評価する。

---

## 9. pkl ファイル構造まとめ

`{prefix}_infos_temporal_train/val.pkl` に保存される1サンプルあたりのキー：

```python
info = {
    # センサー
    'lidar_path':                str,           # LIDAR_TOP .bin
    'cams':                      dict,          # 6カメラ情報
    'sweeps':                    list,          # 最大10スイープ
    # 座標変換
    'lidar2ego_translation':     [3],
    'lidar2ego_rotation':        [4],           # quaternion
    'ego2global_translation':    [3],
    'ego2global_rotation':       [4],
    # 時系列
    'token':                     str,
    'prev':                      str,           # 前フレームトークン
    'next':                      str,           # 次フレームトークン
    'scene_token':               str,
    'frame_idx':                 int,           # シーン内インデックス
    'timestamp':                 int,           # μs
    # メタ情報
    'can_bus':                   [18],
    'map_location':              str,           # 4都市のいずれか
    'fut_valid_flag':            bool,          # 未来6フレームの存在確認
    # アノテーション（nuScenes 標準）
    'gt_boxes':                  [N, 7],
    'gt_names':                  [N],
    'gt_velocity':               [N, 2],
    'valid_flag':                [N],
    'num_lidar_pts':             [N],
    # アノテーション（VAD 独自拡張）
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

## 10. nuScenes Python API 主要メソッド

| メソッド | 用途 |
|---|---|
| `nusc.get(table, token)` | 任意テーブルのレコード取得 |
| `nusc.get_sample_data(token)` | パス・ボックス・カメラ内部行列を取得 |
| `nusc.box_velocity(token)` | グローバル座標での速度を計算 |
| `NuScenesMap.get_patch_coord(patch_box, patch_angle)` | 指定パッチの Shapely Polygon |
| `NuScenesMapExplorer._get_layer_line()` | ライン系レイヤーの取得 |
| `NuScenesMapExplorer._get_layer_polygon()` | ポリゴン系レイヤーの取得 |
| `NuScenesCanBus.get_messages(scene, channel)` | CAN バスメッセージ取得 |
