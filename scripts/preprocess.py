"""前処理スクリプト

data/raw/ から必要な特徴量だけを選んで data/processed/ に保存する。
追加の前処理が必要になったら、各セクションに処理を追記していく。

経緯：
今回は、特徴量重要度、相関係数、およびデータの特性から選択した7つの特徴量を用いて、
POCのインタラクティブなウェブアプリを作成することにしました。
そのため、ここでの前処理は今回のプロジェクト特有の処理として、ディレクトリ内の scripts/preprocess.py に含めることにしました。

実行方法:
    uv run python scripts/preprocess.py
"""

from pathlib import Path

import pandas as pd

# ── パス設定 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

from f1_pit_stops.schema import InferenceSchema, TrainSchema

# ── カラム定義 ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Stint",
    "Year",
    "Driver",
    "Race",
    "TyreLife",
    "RaceProgress",
    "Compound",
]
TRAIN_COLS = ["id"] + FEATURE_COLS + ["PitNextLap"]
TEST_COLS = ["id"] + FEATURE_COLS


# ── 各データの前処理 ──────────────────────────────────────────────────────────

def preprocess_train(raw_path: Path) -> pd.DataFrame:
    """学習データを前処理して返す。

    Steps:
        1. 必要なカラムのみ選択
        2. PitNextLap を int にキャスト
        3. スキーマバリデーション
        # ここに前処理を追記する
    """
    df = pd.read_csv(raw_path, encoding="utf-8-sig")[TRAIN_COLS]

    # --- 型の調整 ---
    df["PitNextLap"] = df["PitNextLap"].astype(int)

    # --- ここに前処理を追記する ---
    # 例: df["TyreLife"] = df["TyreLife"].clip(upper=50)

    # --- バリデーション ---
    TrainSchema.validate(df)

    return df


def preprocess_test(raw_path: Path) -> pd.DataFrame:
    """推論データを前処理して返す。

    Steps:
        1. 必要なカラムのみ選択
        2. スキーマバリデーション（id を除いた 7 特徴量を検証）
        # ここに前処理を追記する
    """
    df = pd.read_csv(raw_path, encoding="utf-8-sig")[TEST_COLS]

    # --- ここに前処理を追記する ---

    # --- バリデーション（id を除いた 7 特徴量のみ対象） ---
    InferenceSchema.validate(df[FEATURE_COLS])

    return df


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    train = preprocess_train(RAW_DIR / "train.csv")
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    print(f"[train] {train.shape} → {PROCESSED_DIR / 'train.csv'}")

    # Test
    test = preprocess_test(RAW_DIR / "test.csv")
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"[test]  {test.shape} → {PROCESSED_DIR / 'test.csv'}")

    print("\n前処理完了。次のステップ: uv run f1-train")


if __name__ == "__main__":
    main()
