# kaggle_06_predicting_f1_pit_stops

Kaggle Playground Series S6E5 — F1 レースにおいて「次のラップでピットインするか」を予測する二値分類の PoC。  
LightGBM × Stratified K-Fold で学習し、Streamlit 上でドライバー・タイヤ条件を動かしながらピット確率をリアルタイム確認できるシミュレーターとして実装した。

予測精度だけでなく、**質の高いデータを再現性高くステークホルダーへ提供できる環境**の構築を目標とした。

- **データ品質の担保**：Pandera によるスキーマバリデーションを学習・推論の両フローに組み込み、入力データの異常を早期に検知できる設計にした
- **実験管理の効率化**：Hydra で設定を一元管理し、MLflow で実験ログを蓄積、Optuna でベイズ最適化を行う3層パイプラインを構築。設定変更からログ確認まで一貫して追跡できる
- **再現性の高い開発環境**：Dev Container により、OS・依存関係を問わず誰でも同一条件で動作確認できる環境を整備した
- **非エンジニアにも届く成果物**：Streamlit でインタラクティブなシミュレーター画面を構築。ドライバーやタイヤ条件をスライダー・セレクトボックスで操作しながらリアルタイムにピット確率を確認できるため、技術的背景を問わずステークホルダーへの説明に活用できる

## 🔮 デモ

デモモデルをもとにインタラクティブな画面を体験できます（学習・環境構築不要）。

👉 **[Streamlit アプリを開く](https://predictingf1pitstopspocview-pcuczhsvgujhkilh9faxt6.streamlit.app/)**

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
# 0. Kaggle からデータを取得して配置
#    https://www.kaggle.com/competitions/playground-series-s6e5
#    → train.csv / test.csv / sample_submission.csv を data/raw/ に置く

# 1. 学習（前処理 + モデル保存）
uv run python scripts/train.py

# 1.5. ハイパーパラメータ探索（Optuna・10 トライアル、任意）
uv run python scripts/train.py -m
#    → learning_rate / num_leaves を自動探索。各トライアルが MLflow に記録される
```

`conf/config.yaml` の `hydra.sweeper.n_trials` や探索範囲を**書き換えずに、1回限りの設定で試したいとき**は CLI で上書きします。たとえば次のような場面です。

- 試走としてトライアル数だけ減らしたい（本番の 10 回はそのまま残したい）
- 前回の MLflow 結果を見て、`learning_rate` の探索範囲を絞り込みたい
- 設定ファイルを触らずに、別の探索条件をすぐ試したい

```bash
uv run python scripts/train.py -m \
  hydra.sweeper.n_trials=5 \
  hydra.sweeper.params.experiment.params.learning_rate=interval(0.01, 0.05)
```

```bash
# 2. 実験結果を MLflow UI で確認
uv run mlflow ui
#    → http://localhost:5000 をブラウザで開く（mlruns/ を自動参照）

# 3. アプリ起動
uv run streamlit run src/f1_pit_stops/app/main.py
```

> **Note** ローカルで学習せずにアプリだけ試したい場合は、上の [デモリンク](#-デモ) からすぐ使えます。実験ログの簡易確認は Streamlit の「実験管理」ページでも可能です。

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

## 参考文献

### 📖 書籍

- 浅野 純季, 木村 真也, 田中 冬馬, 武藤 克大, 栁 泉穂（2025）『先輩データサイエンティストからの指南書 ―実務で生き抜くためのエンジニアリングスキル』技術評論社. ISBN: 978-4-297-15100-3. [→ 公式ページ](https://gihyo.jp/book/2025/978-4-297-15100-3)

### 🐳 Dev Container

- [Get Started with Dev Containers in VS Code（YouTube）](https://youtu.be/b1RavPr_878?si=3sIDOcyZxPuc4djl)
- [Dev Containersとは？Dockerを使った開発環境構築の決定版【図解で完全理解】（Zenn）](https://zenn.dev/yamato_snow/articles/fcb3cf8cf0ad03#discuss)
- [Dev Containerについて基礎を学習する（Qiita）](https://qiita.com/smr1/items/137b912de86c4947ead0)

### ✅ Pandera

- [Panderaをマスターしよう（1. 基本編）（Qiita）](https://qiita.com/GGravitons/items/981f439f687df0dc0be1)
- [Panderaの基本から応用まで（Zenn）](https://zenn.dev/zenn_tkc/articles/e07b34716237b6)
- [Pandera使ってみた（Pandasの型検証）（note）](https://note.com/yuuki_iwasaki/n/n4c15dcade0e9)

### ⚙️ Hydra × MLflow × Optuna

- [設定管理ツール Hydra で内部構造ごと書き換える（Zenn）](https://zenn.dev/gesonanko/articles/417d43669cf2af)
- [【機械学習】Hydraで「再現できない実験」とおさらば（Qiita）](https://qiita.com/shun_hobby/items/eecffd36b1fb827ae7f9)
- [実験管理が簡単に行えるmlflow trackingをローカル環境上で試してみた（Classmethod）](https://dev.classmethod.jp/articles/mlflow-tracking/)
- [MLflowで実験管理入門（フューチャー技術ブログ）](https://future-architect.github.io/articles/20200626/)
- [【13日目】MLflow で実験管理を始めよう（Zenn）](https://zenn.dev/churadata/articles/961bc10fd19ef6)
- [機械学習の煩雑なパラメーター管理の決定版 Hydra・MLflow・Optunaの組み合わせ（ログミー）](https://logmi.jp/main/technology/325087)

### 🖥️ Streamlit

- [Streamlitとは？Pythonで可視化Webアプリを作ろう（AI Academy Media）](https://aiacademy.jp/media/?p=6929)
- [案件先で学んだPythonのStreamlitスキルを公開してみた（note）](https://note.com/cograph_data/n/n167a5bffb6e1)
