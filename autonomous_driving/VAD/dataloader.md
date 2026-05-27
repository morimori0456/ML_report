# VAD データローダー実装レポート（nuScenes 形式）

## 概要

VAD は mmdet3d の `NuScenesDataset` を継承した **`VADCustomNuScenesDataset`** を使用する。データ読み込みは **オフラインのデータ変換フェーズ** と **実行時の読み込みフェーズ** の2段階で構成される。

---

## 1. ファイル構成

| ファイル | 役割 |
|---|---|
| `tools/data_converter/vad_nuscenes_converter.py` | 生の nuScenes データ → pkl 情報ファイルへの変換 |
| `projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` | 実行時データセットクラス・HDマップ読み込み |
| `projects/mmdet3d_plugin/datasets/pipelines/loading.py` | LiDAR 点群・マルチスイープ読み込みパイプライン |
| `projects/configs/VAD/VAD_base_e2e.py` | モデル・データ設定 |

---

## 2. フェーズ1：オフラインデータ変換（`vad_nuscenes_converter.py`）

`create_nuscenes_infos()` がメイン関数。実行後 `{prefix}_infos_temporal_train/val.pkl` が生成される。

### 2-1. 初期化

```python
nusc = NuScenes(version=version, dataroot=root_path)
nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
```

`NuScenesCanBus` はオドメトリ・ステアリング等の低レベルセンサーデータを提供する。

### 2-2. サンプルごとに収集する情報（`_fill_trainval_infos()`）

**センサー情報**（L256–315）

```python
info = {
    'lidar_path': ...,            # LIDAR_TOP .bin ファイルパス
    'token': sample['token'],
    'cams': dict(),               # 6カメラ分の情報
    'sweeps': [...],              # 最大10スイープ
    'ego2global_translation/rotation': ...,
    'lidar2ego_translation/rotation': ...,
    'can_bus': can_bus_18d,       # [pos(3), quat(4), ..., 0, 0] = 18次元
    'map_location': ...,          # 'boston-seaport' など4都市
    'fut_valid_flag': bool,       # 未来6フレームが存在するか
}
```

**カメラ情報** — 6カメラ分ループ（L288–302）

```python
info['cams'] = {
    'CAM_FRONT': {
        'data_path': '...',
        'cam_intrinsic': 3x3 matrix,
        'sensor2ego_rotation/translation': ...,
        'sensor2lidar_rotation/translation': ...,  # obtain_sensor2top() で計算
    },
    # + CAM_FRONT_RIGHT/LEFT, CAM_BACK/BACK_LEFT/BACK_RIGHT
}
```

**アノテーション情報** — VAD 独自拡張（L317–532）

| キー | 形状 | 説明 |
|---|---|---|
| `gt_boxes` | [N, 7] | x,y,z,w,l,h,yaw（LiDAR 座標） |
| `gt_names` | [N] | クラス名 |
| `gt_velocity` | [N, 2] | 速度（LiDAR 座標に変換済み） |
| `gt_agent_fut_trajs` | [N, fut_ts\*2] | 各エージェントの未来6ステップ軌跡（逐次オフセット） |
| `gt_agent_fut_masks` | [N, fut_ts] | 有効ステップマスク |
| `gt_agent_fut_goal` | [N] | ゴール方向クラス（0–8 方向 + 静止=9） |
| `gt_agent_lcf_feat` | [N, 9] | (x, y, yaw, vx, vy, w, l, h, cat_idx) |
| `gt_ego_his_trajs` | [2, 2] | 自車の過去2フレームの逐次オフセット |
| `gt_ego_fut_trajs` | [6, 2] | 自車の未来6フレームの逐次オフセット |
| `gt_ego_fut_cmd` | [3] | 運転コマンド（Turn R / Turn L / Straight） |
| `gt_ego_lcf_feat` | [9] | (vx, vy, ax, ay, yaw角速度, length, width, 速度, ステア曲率) |

**未来軌跡の計算ロジック**（L367–399）

```python
for j in range(fut_ts):  # fut_ts=6
    anno_next = nusc.get('sample_annotation', cur_anno['next'])
    box_next = Box(anno_next['translation'], ...)
    # グローバル → ego → LiDAR 座標へ変換
    box_next.translate(-pose_record['translation'])
    box_next.rotate(Quaternion(pose_record['rotation']).inverse)
    box_next.translate(-cs_record['translation'])
    box_next.rotate(Quaternion(cs_record['rotation']).inverse)
    gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]  # 逐次差分
```

---

## 3. フェーズ2：実行時読み込み（`VADCustomNuScenesDataset`）

### 3-1. クラス初期化（`__init__`、L984–1035）

```python
VADCustomNuScenesDataset(
    queue_length=4,                          # 時系列キューの長さ
    fut_ts=6,                                # 未来予測ステップ数
    pc_range=[-15, -30, -2, 15, 30, 2],     # 点群/BEV 範囲
    map_classes=['divider', 'ped_crossing', 'boundary'],
    map_fixed_ptsnum_per_line=20,            # マップ要素の固定サンプル点数
)
```

`VectorizedLocalMap` を生成：4都市分の `NuScenesMap` + `NuScenesMapExplorer` をあらかじめロード。

### 3-2. `get_data_info()` — サンプルごとの情報取得（L1271–1393）

pkl から読み出し、以下を追加計算する。

```python
# LiDAR → カメラ → 画像 の投影行列（各カメラの 4x4 行列）
lidar2img_rt = viewpad @ lidar2cam_rt.T

# LiDAR → グローバルの座標変換行列
lidar2global = ego2global @ lidar2ego

# CAN bus に heading 角度を追記
can_bus[-2] = patch_angle_rad
can_bus[-1] = patch_angle_deg
```

### 3-3. `get_ann_info()` — アノテーション取得（L1210–1268）

- `with_attr=True` の場合に `gt_agent_fut_trajs` 等を pkl から読み込む
- bbox の原点を nuScenes 形式 (0.5, 0.5, 0.5) → KITTI 形式 (0.5, 0.5, 0) に変換

### 3-4. `vectormap_pipeline()` — HDマップのオンライン生成（L1064–1115）

```python
# LiDAR → ego → global 変換でマップ API を引く座標を算出
lidar2global = ego2global @ lidar2ego
map_pose = lidar2global[:2, 3]          # パッチ中心 (global)
patch_angle = quaternion_yaw(rotation)  # 自車ヘディング

anns_results = vector_map.gen_vectorized_samples(
    location, lidar2global_translation, lidar2global_rotation
)
```

`VectorizedLocalMap.gen_vectorized_samples()`（L520–576）で取得するマップ要素：

| クラス | ソースレイヤー | ジオメトリ変換 |
|---|---|---|
| `divider` | `road_divider` + `lane_divider` | LineString そのまま |
| `ped_crossing` | `ped_crossing` ポリゴン | 外形の LineString |
| `boundary` | `road_segment` + `lane` ポリゴンのユニオン | 外周・内周の LineString |

全要素を `LiDARInstanceLines` に格納。各 LineString は `patch_box` でクリップし、`patch_angle` で回転してLiDAR相対座標に変換する。

**`LiDARInstanceLines` のポイントサンプリング方法**（L32–464）

| プロパティ | 説明 |
|---|---|
| `fixed_num_sampled_points` | `np.linspace(0, length, fixed_num)` で等距離サンプリング → [N, fixed_num, 2] |
| `shift_fixed_num_sampled_points` | ポリゴンは全方向シフト、ラインは正方向・逆方向の2通り（順序不変性対応） |

### 3-5. `prepare_train_data()` — 時系列キュー構築（L1117–1161）

```python
# temporal augmentation: 過去フレームをシャッフルして一部ランダムサンプリング
prev_indexs_list = list(range(index - queue_length, index))
random.shuffle(prev_indexs_list)
prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

# 同シーン内のフレームのみキューに追加
if input_dict['scene_token'] == scene_token:
    data_queue.append(example)
```

### 3-6. `union2one()` — キューを1サンプルに統合（L1179–1208）

```python
# 画像: [queue_length, 6cams, H, W, 3] → stacked tensor
imgs_list = [each['img'].data for each in queue]
queue[-1]['img'] = DC(torch.stack(imgs_list), stack=True)

# CAN bus: 相対変位に変換（先頭フレームを基準）
metas_map[i]['can_bus'][:3] -= prev_pos    # 位置差分
metas_map[i]['can_bus'][-1] -= prev_angle  # 角度差分
```

---

## 4. データ読み込みパイプライン（カメラモード）

`VAD_base_e2e.py` で設定されるパイプライン（`input_modality.use_lidar=False`）：

```
LoadMultiViewImageFromFiles            # 6カメラ画像を読み込み
→ RandomScaleImageMultiViewImage       # スケール拡張
→ PhotoMetricDistortionMultiViewImage  # 色空間拡張
→ NormalizeMultiviewImage              # ImageNet 正規化
→ PadMultiViewImage                    # パディング
→ DefaultFormatBundle3D                # Tensor 変換
→ CustomCollect3D                      # 必要なキーを収集
```

VAD はカメラ画像のみ使用するため LiDAR パイプラインはスキップされる。

---

## 5. 全体データフロー

```
nuScenes raw data
       │
       ▼
vad_nuscenes_converter.py
  - NuScenes API でサンプルを巡回
  - 座標変換行列を計算 (sensor2lidar, lidar2ego, ego2global)
  - 未来軌跡・自車軌跡・CAN bus を計算
  - .pkl ファイルに保存
       │
       ▼
VADCustomNuScenesDataset.__getitem__()
  ├─ get_data_info()        : pkl から情報展開、lidar2img 等を計算
  ├─ get_ann_info()         : gt_boxes + 未来軌跡アノテーション
  ├─ pipeline()             : 画像ロード・拡張・正規化
  ├─ vectormap_pipeline()   : NuScenesMap API からオンラインで HD マップ生成
  │     └─ VectorizedLocalMap.gen_vectorized_samples()
  │           → divider / ped_crossing / boundary → LiDARInstanceLines
  └─ union2one()            : queue_length=4 フレームを1サンプルに統合
       │
       ▼
モデル入力:
  - img              : [B, T, 6, 3, H, W]
  - img_metas        : CAN bus 差分, lidar2img, 座標変換行列等
  - gt_bboxes_3d     : LiDARInstance3DBoxes + 未来軌跡
  - map_gt_bboxes_3d : LiDARInstanceLines（HD マップ要素）
  - ego_fut_trajs    : 自車の未来軌跡
  - ego_his_trajs    : 自車の過去軌跡
```

---

## 6. nuScenes 座標変換チェーン

```
グローバル座標
  ↑ ego2global  (pose_record)
ego 座標
  ↑ lidar2ego   (cs_record)
LiDAR TOP 座標  ← VAD の基準座標系
  ↑ sensor2lidar  (obtain_sensor2top() で計算)
各カメラ / センサー座標
```

マップ要素は `lidar2global` でグローバルパッチ座標を算出してから `NuScenesMap.get_patch_coord()` で取得し、`affinity.rotate / affine_transform` でLiDAR座標に戻す（L771–864）。

---

## 7. 重要ポイントまとめ

| ポイント | 詳細 |
|---|---|
| **HD マップはオンライン生成** | pkl には保存せず、毎イテレーション `NuScenesMap` API から生成 |
| **時系列4フレームキュー** | シャッフルによる temporal データ拡張あり、CAN bus で相対変位を管理 |
| **エージェント軌跡は LiDAR 座標の逐次オフセット** | グローバル軌跡を LCF（Local Coordinate Frame）に変換 |
| **カメラ only モード** | `use_lidar=False` で LiDAR パイプラインはスキップ |
| **順序不変性への対応** | `shift_fixed_num_sampled_points` でポリゴン・ラインの方向曖昧性を吸収 |
