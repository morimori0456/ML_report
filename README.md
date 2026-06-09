# ML Report

機械学習・自動運転関連の論文・実装の調査レポートをまとめるリポジトリ。

---

## セットアップ（uv 共通環境）

全 notebook を動かす共通環境を [uv](https://docs.astral.sh/uv/) で管理しています。

```bash
# uv 未導入の場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# コア環境を構築（GPU 不要の notebook 6本）
uv sync

# Jupyter を起動
uv run jupyter lab
```

`uv sync` で `pyproject.toml` / `uv.lock` から `.venv/` が再現されます（numpy・matplotlib・scipy・opencv-python・jupyterlab）。

### Transformer 実装 notebook（`autonomous_driving/drive_transformer/drive_transformer_demo.ipynb`）

純 PyTorch の最小実装。**CPU の torch で動く**（GPU 不要）。

```bash
uv sync --extra transformer   # torch (CPU) を追加
```

### GPU 微調整 notebook（`llm/lora_qlora_finetune.ipynb`）

この1本だけ CUDA GPU が必要（QLoRA / bitsandbytes）。CUDA 環境または Colab で、追加依存を入れます。

```bash
uv sync --extra llm-gpu   # torch / transformers / peft / trl / bitsandbytes / datasets / accelerate
```

> bitsandbytes は CUDA 前提のため CPU / aarch64 では import に失敗します。コア依存からは分離してあります。

| notebook | 必要な依存 |
|---|---|
| `llm/kv_cache_demo.ipynb` | コアのみ |
| `llm/lora_qlora_demo.ipynb` | コアのみ |
| `autonomous_driving/camera_calibration/extrinsic_calibration_demo.ipynb` | コアのみ |
| `autonomous_driving/camera_calibration/extrinsic_calibration_opencv.ipynb` | コアのみ（opencv） |
| `autonomous_driving/VAD/vad_dataloader_demo.ipynb` | コアのみ |
| `autonomous_driving/VAD/nuscenes_coordinate_transform.ipynb` | コアのみ |
| `autonomous_driving/drive_transformer/drive_transformer_demo.ipynb` | `--extra transformer`（CPU torch） |
| `llm/lora_qlora_finetune.ipynb` | `--extra llm-gpu`（CUDA GPU） |

---

## ディレクトリ構成

```
ML_report/
├── llm/
│   ├── kv_cache.md             # KV Cache 完全解説（transformers / vLLM コード解析）
│   ├── kv_cache_demo.ipynb     # KV Cache デモ（numpy）
│   ├── lora_qlora.md           # LoRA / QLoRA 完全解説（原理・NF4量子化・PEFTコード解析）
│   ├── lora_qlora_demo.ipynb   # 概念デモ（numpyのみ・GPU不要）
│   └── lora_qlora_finetune.ipynb # 実機QLoRA微調整（Colab/GPU・PEFT/trl/bitsandbytes）
└── autonomous_driving/
    ├── localization_tech.md    # 自己位置推定技術サーベイ（センサーフュージョン全般）
    ├── camera_calibration/     # カメラ外部キャリブレーション
    │   ├── extrinsic_calibration.md            # 完全解説（[R|t]・PnP・エピポーラ・レクティフィケーション）
    │   ├── extrinsic_calibration_demo.ipynb    # 概念デモ（numpyのみ・GPU不要）
    │   └── extrinsic_calibration_opencv.ipynb  # OpenCV実践（calibrate/solvePnP/stereoRectify）
    ├── drive_transformer/      # DriveTransformer (ICLR 2025, E2E自動運転)
    │   ├── drive_transformer.md                # 完全解説（タスク並列・疎表現・ストリーミング）
    │   └── drive_transformer_demo.ipynb        # 最小PyTorch実装（3種attention・FIFO・6モード計画）
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
| LoRA / QLoRA 完全解説 | 低ランク分解・α/rスケーリング・NF4量子化・Double Quant・メモリ計算・PEFT/bitsandbytesコード解析・DoRA等派生 | [llm/lora_qlora.md](llm/lora_qlora.md) ＋ [概念デモ](llm/lora_qlora_demo.ipynb) / [実機微調整](llm/lora_qlora_finetune.ipynb) |

### 自動運転（共通技術）

| タイトル | トピック | リンク |
|---|---|---|
| 自己位置推定技術サーベイ | KF/EKF・NDT・SLAM・VIO・DL測位・センサーフュージョン全般 | [autonomous_driving/localization_tech.md](autonomous_driving/localization_tech.md) |
| カメラ外部キャリブレーション完全解説 | 座標系・[R\|t]・射影P=K[R\|t]・PnP/DLT・エピポーラ幾何・ステレオレクティフィケーション・視差→深度・camera-LiDAR | [extrinsic_calibration.md](autonomous_driving/camera_calibration/extrinsic_calibration.md) ＋ [概念デモ](autonomous_driving/camera_calibration/extrinsic_calibration_demo.ipynb) / [OpenCV実践](autonomous_driving/camera_calibration/extrinsic_calibration_opencv.ipynb) |
| DriveTransformer 完全解説 | 統一Transformer型E2E自動運転・タスク並列(Self-Attn)・疎表現(BEVレス Sensor Cross-Attn)・ストリーミングFIFO(Temporal Cross-Attn)・6モード計画WTA | [drive_transformer.md](autonomous_driving/drive_transformer/drive_transformer.md) ＋ [最小実装デモ](autonomous_driving/drive_transformer/drive_transformer_demo.ipynb) |

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
