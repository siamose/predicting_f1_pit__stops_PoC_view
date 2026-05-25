import pandera.pandas as pa
from pandera.typing import Series


class FeatureSchema(pa.DataFrameModel):
    """モデルで使う 7 特徴量の共通スキーマ。

    TrainSchema と InferenceSchema の基底クラス。
    値の範囲は train + test データの実測値をもとに設定している。

    Notes
    -----
    strict=True のため、バリデーション前に必要なカラムだけを
    DataFrame に残しておく必要がある。
    """

    Stint: Series[int] = pa.Field(ge=1, le=8, description="スティント番号（1〜8）")
    Year: Series[int] = pa.Field(
        isin=[2022, 2023, 2024, 2025],
        description="シーズン年",
    )
    Driver: Series[str] = pa.Field(description="ドライバーコード")
    Race: Series[str] = pa.Field(description="レース名")
    TyreLife: Series[float] = pa.Field(
        ge=1.0, le=77.0,
        description="タイヤ使用ラップ数（1〜77）",
    )
    RaceProgress: Series[float] = pa.Field(
        ge=0.01, le=1.0,
        description="レース進捗率（0.01〜1.0）",
    )
    Compound: Series[str] = pa.Field(
        isin=["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"],
        description="タイヤコンパウンド",
    )

    class Config:
        strict = True
        coerce = True


class TrainSchema(FeatureSchema):
    """学習データ用スキーマ。

    FeatureSchema（7特徴量）に id と目的変数 PitNextLap を追加する。

    使い方::

        TRAIN_COLS = ["id", "Stint", "Year", "Driver", "Race",
                      "TyreLife", "RaceProgress", "Compound", "PitNextLap"]
        df = pd.read_csv(path)[TRAIN_COLS]
        TrainSchema.validate(df)
    """

    id: Series[int] = pa.Field(ge=0, description="行識別子")
    PitNextLap: Series[int] = pa.Field(
        isin=[0, 1],
        description="目的変数：次のラップでピットインするなら 1",
    )

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True


class InferenceSchema(FeatureSchema):
    """推論入力用スキーマ。

    Streamlit のシミュレーターがモデルに渡す DataFrame を検証する。
    id・PitNextLap は含まない。7特徴量のみ。

    使い方::

        df = pd.DataFrame([{
            "Stint": 2, "Year": 2024, "Driver": "HAM",
            "Race": "Monaco Grand Prix", "TyreLife": 15.0,
            "RaceProgress": 0.4, "Compound": "MEDIUM",
        }])
        InferenceSchema.validate(df)
    """

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True
