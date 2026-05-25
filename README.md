# kaggle_06_predicting_f1_pit_stops

Kaggle Playground Series S6E5 — F1 レースにおいて「次のラップでピットインするか」を予測する二値分類の PoC。  
LightGBM × Stratified K-Fold で学習し、Streamlit 上でドライバー・タイヤ条件を動かしながらピット確率をリアルタイム確認できるシミュレーターとして実装した。

予測精度だけでなく、**質の高いデータを再現性高くステークホルダーへ提供できる環境**の構築を目標とした。

- **データ品質の担保**：Pandera によるスキーマバリデーションを学習・推論の両フローに組み込み、入力データの異常を早期に検知できる設計にした
- **実験管理の効率化**：Hydra で設定を一元管理し、MLflow で実験ログを蓄積、Optuna でベイズ最適化を行う3層パイプラインを構築。設定変更からログ確認まで一貫して追跡できる
- **再現性の高い開発環境**：Dev Container により、OS・依存関係を問わず誰でも同一条件で動作確認できる環境を整備した
- **非エンジニアにも届く成果物**：Streamlit でインタラクティブなシミュレーター画面を構築。ドライバーやタイヤ条件をスライダー・セレクトボックスで操作しながらリアルタイムにピット確率を確認できるため、技術的背景を問わずステークホルダーへの説明に活用できる

## Stack

| Tool                                         | Role                         | Design Notes                             |
| -------------------------------------------- | ---------------------------- | ---------------------------------------- |
| [LightGBM](https://lightgbm.readthedocs.io/) | Binary classification model  |                                          |
| [Pandera](https://pandera.readthedocs.io/)   | Input data schema validation | [実装記録](docs/pandera_implementation.md)   |
| [Hydra](https://hydra.cc/)                   | Config management            | [実装記録](docs/hydra_implementation.md)     |
| [MLflow](https://mlflow.org/)                | Experiment tracking          | 〃                                        |
| [Optuna](https://optuna.org/)                | Hyperparameter optimization  | 〃                                        |
| [Streamlit](https://streamlit.io/)           | Interactive POC dashboard    | [実装記録](docs/streamlit_implementation.md) |
| [uv](https://docs.astral.sh/uv/)             | Package management           |                                          |

## Prerequisites

Python 3.12 と [uv](https://docs.astral.sh/uv/getting-started/installation/) が必要です。

```bash
# 依存関係のインストール
uv sync
```

## Getting Started

```bash
# 1. 学習（前処理 + モデル保存）
uv run python scripts/train.py

# 2. アプリ起動
uv run streamlit run src/f1_pit_stops/app/main.py
```

## Directory Structure

```
.
├── conf/                    # Hydra 設定ファイル
│   ├── config.yaml          # メイン設定（データパス・MLflow・Optuna スイーパー）
│   └── experiment/          # 実験プリセット（default / high_leaves / high_lr）
├── data/
│   ├── raw/                 # 生データ（.gitignore 対象）
│   └── processed/           # 前処理済みデータ
├── docs/                    # 設計記録・実装メモ
├── model/                   # 学習済みモデル（.gitignore 対象）
├── notebooks/               # EDA・可視化ノートブック
├── scripts/
│   └── train.py             # 学習エントリーポイント（前処理を内包）
└── src/f1_pit_stops/
    ├── app/
    │   └── main.py          # Streamlit アプリ
    ├── models/              # モデル関連
    └── schema/              # Pandera スキーマ定義
```

## Docs

実装の設計思想・試行錯誤を時系列で記録したドキュメントです。

| Document                                                     | 内容                                                |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [特徴量選択記録](docs/feature_selection.md)                         | EDA → 外れ値処理 → 特徴量エンジニアリング → 重要度分析 → 最終 7 変数への絞り込み |
| [Hydra × MLflow × Optuna 実装記録](docs/hydra_implementation.md) | 実験管理パイプライン（設定・追跡・最適化）の構築過程                        |
| [Pandera 実装記録](docs/pandera_implementation.md)               | 入力データのスキーマ設計と試行錯誤                                 |
| [Streamlit 実装記録](docs/streamlit_implementation.md)           | シミュレーター UI の設計思想と試行錯誤                             |

## Notebook

[`notebooks/Predicting F1 Pit Stops.ipynb`](notebooks/Predicting%20F1%20Pit%20Stops.ipynb)

EDA・可視化を行ったノートブック。SWEETVIZ による Train vs Test 比較、カテゴリ変数の Lift ヒートマップ、相関行列、LightGBM × 4-fold による特徴量重要度分析など、最終的な特徴量選択の根拠となる分析を収録している。詳細は [特徴量選択記録](docs/feature_selection.md) を参照。

## Data Attribution

This project is based on data from two Kaggle datasets:

### Playground Competition Dataset
- **Title**: Predicting F1 Pit Stops (Playground Series - Season 6 Episode 5)
- **Author**: Kaggle
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e5
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Original Dataset
- **Title**: F1 Strategy Dataset | Pit Stop Prediction
- **Author**: Aadit Gupta
- **URL**: https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The Playground dataset was derived from the original F1 Strategy Dataset
with modified feature distributions. This application uses synthetic data
generated based on statistical properties of these datasets.
