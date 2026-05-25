"""LightGBM 学習スクリプト（Hydra + MLflow）

前処理（カラム選択・バリデーション）から学習まで一括で実行する。

    uv run python scripts/train.py                        # 前処理 + 学習・モデル保存

Hydra でパラメータを上書きする場合:
    uv run python scripts/train.py experiment.params.learning_rate=0.01
    uv run python scripts/train.py experiment.params.num_leaves=127

マルチラン（複数設定を一括実行）:
    uv run python scripts/train.py --multirun \\
        experiment.params.learning_rate=0.01,0.05 \\
        experiment.params.num_leaves=63,127
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

# train.py → scripts/ → プロジェクトルート → conf/
_CONF_DIR = str(Path(__file__).parents[1] / "conf")

import hydra
import hydra.utils
from hydra.core.hydra_config import HydraConfig
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ── 定数 ──────────────────────────────────────────────────────────────────────
TARGET = "PitNextLap"
FEATURE_COLS = ["Stint", "Year", "Driver", "Race", "TyreLife", "RaceProgress", "Compound"]
CAT_COLS = ["Driver", "Race", "Compound"]
TRAIN_COLS = ["id"] + FEATURE_COLS + [TARGET]
TEST_COLS = ["id"] + FEATURE_COLS


# ── カテゴリカル処理 ──────────────────────────────────────────────────────────

def apply_categories(
    df: pd.DataFrame,
    categories: dict[str, list],
) -> pd.DataFrame:
    """保存済みのカテゴリ定義を DataFrame に適用する。

    学習時・推論時で同じコードが使われることを保証する。
    未知のカテゴリは NaN になる（Streamlit のセレクトボックスが
    既知の値しか渡さないため、通常は発生しない）。
    """
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


# ── 学習 ──────────────────────────────────────────────────────────────────────

def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    params: DictConfig,
    model_dir: Path,
) -> tuple[np.ndarray, list[float]]:
    """StratifiedKFold × LightGBM で学習し、各 fold のモデルを pickle 保存する。

    Returns
    -------
    test_preds : np.ndarray
        全 fold の予測確率を平均したもの
    oof_auc : list[float]
        各 fold の OOF AUC
    """
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "min_child_samples": params.min_child_samples,
        "feature_fraction": params.feature_fraction,
        "bagging_fraction": params.bagging_fraction,
        "bagging_freq": params.bagging_freq,
        "scale_pos_weight": float((y == 0).sum() / (y == 1).sum()),
        "verbose": -1,
        "random_state": params.random_state,
    }

    skf = StratifiedKFold(
        n_splits=params.n_splits,
        shuffle=True,
        random_state=params.random_state,
    )
    test_preds = np.zeros(len(X_test))
    oof_auc: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.train(
            lgb_params,
            lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS),
            num_boost_round=1000,
            valid_sets=[
                lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_COLS)
            ],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        auc = roc_auc_score(y_val, model.predict(X_val))
        oof_auc.append(auc)
        test_preds += model.predict(X_test) / params.n_splits

        # fold ごとに pickle 保存
        with open(model_dir / f"lgbm_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.log_metric(f"fold{fold}_auc", auc, step=fold)
        print(f"  Fold {fold}: AUC = {auc:.4f}  (best_iter={model.best_iteration})")

    return test_preds, oof_auc


# ── エントリーポイント ────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    root = Path(hydra.utils.get_original_cwd())
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri((root / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    hydra_cfg = HydraConfig.get()
    is_multirun = hydra_cfg.mode.name == "MULTIRUN"
    run_name = (
        f"{cfg.experiment.name}_trial{hydra_cfg.job.num}"
        if is_multirun
        else cfg.experiment.name
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(cfg.experiment.params))

        # ── 1. データ読み込み・カラム選択・バリデーション ──────────────────
        from f1_pit_stops.schema import InferenceSchema, TrainSchema

        train = pd.read_csv(root / cfg.data.train_path, encoding="utf-8-sig")[TRAIN_COLS]
        train[TARGET] = train[TARGET].astype(int)
        TrainSchema.validate(train)

        test = pd.read_csv(root / cfg.data.test_path, encoding="utf-8-sig")[TEST_COLS]
        InferenceSchema.validate(test[FEATURE_COLS])

        print(f"train: {train.shape}, test: {test.shape}")

        # ── 2. カテゴリ定義を作成して保存 ─────────────────────────────────
        # train のカテゴリを基準にする（test にしか存在しない値は unknown 扱い）
        categories: dict[str, list] = {}
        for col in CAT_COLS:
            train[col] = train[col].astype("category")
            categories[col] = train[col].cat.categories.tolist()

        categories_path = model_dir / "categories.json"
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        print(f"categories.json saved → {categories_path}")

        # test にも同じカテゴリ定義を適用
        test = apply_categories(test, categories)

        # ── 3. 学習 ────────────────────────────────────────────────────────
        X = train[FEATURE_COLS]
        y = train[TARGET]
        X_test = test[FEATURE_COLS]

        print(f"\nTraining ({cfg.experiment.params.n_splits}-fold CV)...")
        test_preds, oof_auc = train_cv(X, y, X_test, cfg.experiment.params, model_dir)

        mean_auc = float(np.mean(oof_auc))
        std_auc = float(np.std(oof_auc))
        mlflow.log_metric("oof_auc_mean", mean_auc)
        mlflow.log_metric("oof_auc_std", std_auc)
        print(f"\nOOF AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        # ── 4. submission 保存 ────────────────────────────────────────────
        sample_sub = pd.read_csv(root / "data" / "raw" / "sample_submission.csv")
        submission = sample_sub[["id"]].copy()
        submission[TARGET] = test_preds

        sub_path = root / cfg.data.submission_path
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(sub_path, index=False)

        mlflow.log_artifact(str(sub_path))
        print(f"Submission saved → {sub_path}")

    return mean_auc


if __name__ == "__main__":
    main()
