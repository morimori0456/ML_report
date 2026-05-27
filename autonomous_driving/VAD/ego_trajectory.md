# VAD 自車軌跡（gt_ego_his_trajs / gt_ego_fut_trajs）計算ロジック解説

## 概要

`vad_nuscenes_converter.py` の `_fill_trainval_infos()` 内で計算される自車の過去軌跡・未来軌跡。  
いずれも **現在フレームの LiDAR 座標系（LCF）上での逐次オフセット（per-step offset）** として保存される。

| キー | 形状 | 意味 |
|---|---|---|
| `gt_ego_his_trajs` | `[2, 2]` (=`[his_ts, 2]`) | 過去2ステップの逐次変位 (x, y) |
| `gt_ego_fut_trajs` | `[6, 2]` (=`[fut_ts, 2]`) | 未来6ステップの逐次変位 (x, y) |

---

## 共通の基礎：`get_global_sensor_pose()`

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

`pose[:3, 3]` が **グローバル座標系における LIDAR_TOP センサーの位置（xyz）** になる。  
数式で書くと：

```
p_global = R_e2g · p_lidar_in_ego + t_e2g
         = R_e2g · (R_l2e · p_lidar + t_l2e) + t_e2g
```

ここで `p_lidar = (0, 0, 0)` のとき（センサー原点）、`pose[:3, 3]` = LiDAR センサーのグローバル位置。

---

## gt_ego_his_trajs の計算ロジック

### ステップ 1：グローバル位置の収集（L402–418）

```python
his_ts = 2
ego_his_trajs      = np.zeros((his_ts+1, 3))  # shape [3, 3]
ego_his_trajs_diff = np.zeros((his_ts+1, 3))  # 外挿用の差分

sample_cur = sample   # 現在フレーム（t）から出発
for i in range(his_ts, -1, -1):   # i = 2, 1, 0 の順
    if sample_cur is not None:
        pose_mat = get_global_sensor_pose(sample_cur, nusc)
        ego_his_trajs[i] = pose_mat[:3, 3]          # グローバル位置を格納

        # 外挿用：次フレームとの差分を保持
        if sample_cur['next'] != '':
            sample_next = nusc.get('sample', sample_cur['next'])
            pose_mat_next = get_global_sensor_pose(sample_next, nusc)
            ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]

        sample_cur = nusc.get('sample', sample_cur['prev']) if sample_cur['prev'] != '' else None
    else:
        # シーン先頭で過去フレームが存在しない → 等速運動で外挿
        ego_his_trajs[i]      = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
        ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]
```

ループ終了後の `ego_his_trajs` の内容（時刻の早い順）：

| インデックス | 対応サンプル | 内容 |
|---|---|---|
| `[0]` | `t-2`（2フレーム前） | グローバル LiDAR 位置、または等速外挿値 |
| `[1]` | `t-1`（1フレーム前） | グローバル LiDAR 位置 |
| `[2]` | `t`（現在） | グローバル LiDAR 位置 |

**等速外挿の意図：** シーン先頭サンプルでは過去フレームが存在しない。その場合でも固定長の軌跡テンソルを生成するため、直前の速度ベクトル（`ego_his_trajs_diff[i+1]`）をそのまま使って仮想の位置を逆算する。

### ステップ 2：グローバル → LCF（現在フレームの LiDAR 座標）への変換（L420–427）

```python
# (1) global → ego（現在フレームの自車座標）
ego_his_trajs  -= np.array(pose_record['translation'])        # 並進
rot_mat         = Quaternion(pose_record['rotation']).inverse.rotation_matrix
ego_his_trajs   = np.dot(rot_mat, ego_his_trajs.T).T          # 回転

# (2) ego → lidar（現在フレームの LiDAR 座標）
ego_his_trajs  -= np.array(cs_record['translation'])          # 並進
rot_mat         = Quaternion(cs_record['rotation']).inverse.rotation_matrix
ego_his_trajs   = np.dot(rot_mat, ego_his_trajs.T).T          # 回転
```

変換後、`ego_his_trajs[2]`（現在フレームの LiDAR 位置を LiDAR 座標で見たもの）は定義上 **(0, 0, 0)** になる。

### ステップ 3：絶対位置 → 逐次オフセット（L428）

```python
ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]
# shape: [3, 3] → [2, 3]
```

各行が「1ステップ分の変位ベクトル」になる：

| 行インデックス | 意味 |
|---|---|
| `[0]` | `pos(t-2) → pos(t-1)` の変位（LCF） |
| `[1]` | `pos(t-1) → pos(t)`   の変位（LCF） |

### ステップ 4：保存（L528）

```python
info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
# z 成分を捨て shape = [2, 2] として pkl に保存
```

---

## gt_ego_fut_trajs の計算ロジック

### ステップ 1：グローバル位置の収集（L431–442）

```python
fut_ts = 6
ego_fut_trajs = np.zeros((fut_ts+1, 3))   # shape [7, 3]
ego_fut_masks = np.zeros((fut_ts+1))

sample_cur = sample   # 現在フレーム（t）
for i in range(fut_ts+1):    # i = 0, 1, ..., 6
    pose_mat = get_global_sensor_pose(sample_cur, nusc)
    ego_fut_trajs[i] = pose_mat[:3, 3]
    ego_fut_masks[i] = 1

    if sample_cur['next'] == '':
        ego_fut_trajs[i+1:] = ego_fut_trajs[i]   # 残りを最終位置で埋める
        break
    else:
        sample_cur = nusc.get('sample', sample_cur['next'])
```

| インデックス | 対応サンプル | 内容 |
|---|---|---|
| `[0]` | `t`（現在） | グローバル LiDAR 位置 |
| `[1]` | `t+1` | グローバル LiDAR 位置 |
| ... | ... | ... |
| `[6]` | `t+6`（3秒後） | グローバル LiDAR 位置（なければ最終位置で埋め） |

**シーン末尾の処理：** `fut_valid_flag = False` が立ち、評価時にこのサンプルの計画誤差はスキップされる（`VAD_head.py` L595付近）。

### ステップ 2：グローバル → LCF への変換（L444–450）

過去軌跡と同一のロジック（`pose_record` / `cs_record` は現在フレームのものを使用）。

変換後、`ego_fut_trajs[0]`（現在フレームを LCF で見た値）は **(0, 0, 0)**。

### ステップ 3：運転コマンドの決定（L452–457）

```python
# 逐次変換前の「LCF 上の絶対位置」で判定する
if ego_fut_trajs[-1][0] >= 2:     # t+6 の x 変位が +2m 以上
    command = np.array([1, 0, 0]) # Turn Right
elif ego_fut_trajs[-1][0] <= -2:  # t+6 の x 変位が -2m 以下
    command = np.array([0, 1, 0]) # Turn Left
else:
    command = np.array([0, 0, 1]) # Go Straight
```

> **LCF の x 軸と「右折」の対応について：**  
> nuScenes の LiDAR 座標系は x 軸が**車両進行方向の左**を向いている（右手系）。  
> したがって `x >= 2` が「右折」に対応するのは、ここで使われる LCF が  
> `ego2global @ lidar2ego` の逆変換を適用した座標系であることに由来する。  
> 実装として、シンガポール（左側通行）でもステアリング符号反転は `ego_lcf_feat` の  
> Kappa 計算に対してのみ行われ、コマンド判定は変換しない。

### ステップ 4：絶対位置 → 逐次オフセット（L459）

```python
ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
# shape: [7, 3] → [6, 3]
```

| 行インデックス | 意味 |
|---|---|
| `[0]` | `pos(t) → pos(t+1)` の変位（LCF） |
| `[1]` | `pos(t+1) → pos(t+2)` の変位（LCF） |
| ... | ... |
| `[5]` | `pos(t+5) → pos(t+6)` の変位（LCF） |

### ステップ 5：保存（L529–530）

```python
info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)  # shape [6, 2]
info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)     # shape [6]
# masks[0]=1 の場合 t→t+1 のオフセットが有効
```

---

## 座標変換チェーンの整理

```
グローバル座標（ENU / UTM）
  │  get_global_sensor_pose() = global_from_ego @ ego_from_sensor
  │  → pose[:3, 3] = グローバルの LiDAR 位置
  │
  ▼ − pose_record['translation']         (グローバル → ego 並進)
  ▼ × Quaternion(pose_record).inverse    (グローバル → ego 回転)
ego 座標（現在フレームの自車中心）
  │
  ▼ − cs_record['translation']           (ego → lidar 並進)
  ▼ × Quaternion(cs_record).inverse      (ego → lidar 回転)
LiDAR 座標（現在フレーム = LCF）
  ← ここで逐次差分を取る
```

すべての位置が「現在フレームの LiDAR を原点とした座標」に統一されるため、  
モデルは **自車中心の相対的な運動量** を学習できる。

---

## モデルでの利用

### gt_ego_his_trajs → Planning Query の初期化

```python
# VAD_head.py L712
if self.ego_his_encoder is not None:
    ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
else:
    ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
ego_query = ego_his_feats   # Planner の初期クエリとして使用
```

過去2ステップの変位を MLP でエンコードし、自車の「今どう動いているか」をプランナーの初期状態として注入する。`ego_his_encoder` が `None` の場合は学習済み固定クエリを使用。

### gt_ego_fut_trajs → Planning Loss の教師信号 & 評価

```python
# VAD.py L421, L425–426（テスト時）
ego_fut_trajs = ego_fut_trajs[0, 0]              # [fut_ts, 2]
ego_fut_pred  = ego_fut_preds[ego_fut_cmd_idx]   # コマンドに対応する予測を選択
ego_fut_pred  = ego_fut_pred.cumsum(dim=-2)      # per-step offset → 累積位置
ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)     # per-step offset → 累積位置
```

評価時は「逐次オフセット → 累積位置」に戻してから L2 距離（ADE/FDE）を計算する。

---

## データフロー全体図

```
nuScenes サンプル（現在フレーム t）
│
├─ get_global_sensor_pose(t-2, t-1, t)    過去3フレームのグローバル LiDAR 位置
│   → LCF へ座標変換
│   → 逐次差分 [t-2→t-1, t-1→t]
│   → gt_ego_his_trajs: shape [2, 2]     ← pkl に保存
│
└─ get_global_sensor_pose(t, t+1, ..., t+6)  未来7フレームのグローバル LiDAR 位置
    → LCF の絶対位置で運転コマンド判定（t+6 の x 変位）
    → LCF へ座標変換
    → 逐次差分 [t→t+1, ..., t+5→t+6]
    → gt_ego_fut_trajs: shape [6, 2]     ← pkl に保存
    → gt_ego_fut_cmd:   shape [3]        ← pkl に保存（one-hot）
    → gt_ego_fut_masks: shape [6]        ← pkl に保存

                               ↓
VADCustomNuScenesDataset.get_data_info()
  → ego_his_trajs / ego_fut_trajs として input_dict に展開
  → CustomCollect3D でバッチに含まれる

                               ↓
VAD_head.forward()
  → ego_his_trajs → ego_his_encoder → Planning Query
  → ego_fut_trajs → Loss 計算 / 評価メトリクス
```

---

## 注意点・落とし穴

| 項目 | 内容 |
|---|---|
| **シーン先頭の等速外挿** | 過去フレームが存在しない場合は仮想位置を生成するため、実際の運動とは異なる可能性がある。`frame_idx == 0` のサンプルで発生 |
| **シーン末尾の静止補完** | 未来フレームが足りない場合、最終フレームの位置で残りを埋めるため、後半オフセットがすべて (0, 0) になる |
| **z 成分の破棄** | `[:, :2]` で保存するため高低差は含まれない。立体交差・坂道では実際の移動量と水平距離にズレが生じる |
| **運転コマンド判定タイミング** | 逐次差分化前の「絶対位置」で判定するため、コマンドは累積軌跡の最終点の横方向変位に基づく |
| **LCF の固定** | すべてのフレームの軌跡を **現在フレーム（t）の LiDAR 座標** に変換するため、評価も推論も同じ原点を共有する |
