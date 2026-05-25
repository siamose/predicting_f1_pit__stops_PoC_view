# kaggle_06_predicting_f1_pit_stops

[Brief description of what this project does — 1-2 sentences]

## Stack

| Tool                                       | Role                                           |
| ------------------------------------------ | ---------------------------------------------- |
| [Pandera](https://pandera.readthedocs.io/) | Input data schema validation                   |
| [Hydra](https://hydra.cc/)                 | Config management and multi-run experiments    |
| [MLflow](https://mlflow.org/)              | Experiment tracking and model artifact storage |
| [Streamlit](https://streamlit.io/)         | Interactive POC dashboard                      |
| [uv](https://docs.astral.sh/uv/)           | Package management                             |

## Getting Started

```bash
# 1. 学習（前処理 + モデル保存）
uv run python scripts/train.py -m

# 2. アプリ起動
uv run streamlit run src/f1_pit_stops/app/main.py
```

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
