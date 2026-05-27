# ML Report

機械学習・自動運転関連の論文・実装の調査レポートをまとめるリポジトリ。

---

## ディレクトリ構成

```
ML_report/
└── autonomous_driving/        # 自動運転
    └── VAD/                   # VAD (Vectorized Scene Representation)
        └── dataloader.md      # nuScenes データローダー実装解説
```

---

## レポート一覧

### 自動運転

| タイトル | トピック | リンク |
|---|---|---|
| VAD データローダー実装解説 | nuScenes 形式のデータ読み込み・HDマップ生成・時系列キュー | [autonomous_driving/VAD/dataloader.md](autonomous_driving/VAD/dataloader.md) |

---

## 追加方針

- トピックごとにトップレベルディレクトリを作成（例: `nlp/`, `generative/`）
- 論文・実装ごとにサブディレクトリを作成
- ファイル名はレポートの内容を端的に表す snake_case
