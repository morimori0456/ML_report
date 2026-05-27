# ML Report

機械学習・自動運転関連の論文・実装の調査レポートをまとめるリポジトリ。

---

## ディレクトリ構成

```
ML_report/
├── llm/
│   └── kv_cache.md             # KV Cache 完全解説（transformers / vLLM コード解析）
└── autonomous_driving/
    ├── localization_tech.md    # 自己位置推定技術サーベイ（センサーフュージョン全般）
    └── VAD/                   # VAD (Vectorized Scene Representation)
        ├── dataloader.md       # nuScenes データローダー実装解説
        ├── nuscenes_dataset.md # nuScenes データセット詳細解説（ego_pose 測位追記）
        └── ego_trajectory.md   # 自車軌跡（gt_ego_his/fut_trajs）計算ロジック
```

---

## レポート一覧

### LLM

| タイトル | トピック | リンク |
|---|---|---|
| KV Cache 完全解説 | 原理・メモリ計算・PagedAttention・Prefix Caching・MLA・量子化（transformers/vLLM コード解析） | [llm/kv_cache.md](llm/kv_cache.md) |

### 自動運転（共通技術）

| タイトル | トピック | リンク |
|---|---|---|
| 自己位置推定技術サーベイ | KF/EKF・NDT・SLAM・VIO・DL測位・センサーフュージョン全般 | [autonomous_driving/localization_tech.md](autonomous_driving/localization_tech.md) |

### 自動運転（VAD）

| タイトル | トピック | リンク |
|---|---|---|
| VAD データローダー実装解説 | nuScenes 形式のデータ読み込み・HDマップ生成・時系列キュー | [autonomous_driving/VAD/dataloader.md](autonomous_driving/VAD/dataloader.md) |
| nuScenes データセット詳細解説 | センサー構成・データ階層・アノテーション・地図・ego_pose 測位精度 | [autonomous_driving/VAD/nuscenes_dataset.md](autonomous_driving/VAD/nuscenes_dataset.md) |
| 自車軌跡計算ロジック解説 | gt_ego_his_trajs / gt_ego_fut_trajs の座標変換・逐次差分・モデル利用 | [autonomous_driving/VAD/ego_trajectory.md](autonomous_driving/VAD/ego_trajectory.md) |

---

## 追加方針

- トピックごとにトップレベルディレクトリを作成（例: `nlp/`, `generative/`）
- 論文・実装ごとにサブディレクトリを作成
- ファイル名はレポートの内容を端的に表す snake_case
