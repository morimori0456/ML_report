# ML Report

機械学習・自動運転関連の論文・実装の調査レポートをまとめるリポジトリ。

---

## ディレクトリ構成

```
ML_report/
└── autonomous_driving/        # 自動運転
    └── VAD/                   # VAD (Vectorized Scene Representation)
        ├── dataloader.md       # nuScenes データローダー実装解説
        ├── nuscenes_dataset.md # nuScenes データセット詳細解説
        └── ego_trajectory.md   # 自車軌跡（gt_ego_his/fut_trajs）計算ロジック
```

---

## レポート一覧

### 自動運転

| タイトル | トピック | リンク |
|---|---|---|
| VAD データローダー実装解説 | nuScenes 形式のデータ読み込み・HDマップ生成・時系列キュー | [autonomous_driving/VAD/dataloader.md](autonomous_driving/VAD/dataloader.md) |
| nuScenes データセット詳細解説 | センサー構成・データ階層・アノテーション・地図・VAD拡張情報 | [autonomous_driving/VAD/nuscenes_dataset.md](autonomous_driving/VAD/nuscenes_dataset.md) |
| 自車軌跡計算ロジック解説 | gt_ego_his_trajs / gt_ego_fut_trajs の座標変換・逐次差分・モデル利用 | [autonomous_driving/VAD/ego_trajectory.md](autonomous_driving/VAD/ego_trajectory.md) |

---

## 追加方針

- トピックごとにトップレベルディレクトリを作成（例: `nlp/`, `generative/`）
- 論文・実装ごとにサブディレクトリを作成
- ファイル名はレポートの内容を端的に表す snake_case
