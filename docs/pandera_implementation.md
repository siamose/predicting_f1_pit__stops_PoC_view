# Pandera 実装記録

> F1 Pit Stop Predictor PoC における Pandera スキーマ設計の思想・試行錯誤を、時系列で記録したドキュメント。

---

## 0. なぜ Pandera を採用したか

機械学習パイプラインで起きやすい問題の一つが「データの型やレンジが暗黙的に変わっていても気づかない」こと。特に：

- 学習時と推論時で異なる前処理が適用される
- CSV の読み込みで意図せず型が変わる（float が object になる等）
- カテゴリカル変数に未知の値が混入する

これらを**コードとして明示的に検証**するために Pandera を導入した。

Pandera を選んだ理由：
- pandas の DataFrame をそのまま検証できる（変換不要）
- クラスベースのスキーマ定義で継承が使える
- 型チェック・値域チェック・カテゴリチェックを1ファイルに集約できる

---

## 1. スキーマ設計の出発点：「3クラス構成」

### 問いの立て方

スキーマを設計するにあたり、最初に「何を検証したいか」を整理した：

1. **学習データ**：特徴量 + `id` + `PitNextLap`（目的変数）
2. **テスト・推論データ**：特徴量 + `id`（目的変数なし）
3. **Streamlit 入力**：特徴量のみ（`id` も不要）

3つで「特徴量の定義」は共通しているため、**継承**で解決するのが自然。

```
FeatureSchema（7特徴量の定義）
    ├── TrainSchema（+ id, PitNextLap）
    └── InferenceSchema（7特徴量のみ）
```

`InferenceSchema` は `FeatureSchema` と構造的には同じだが、**意図を明確にするために別クラスとして定義した**。「推論に使う入力はこの形」というドキュメントとしての役割も持つ。

---

## 2. 特徴量ごとの制約設計

各特徴量の値域は **train + test データの実測値** をもとに設定した。

```python
import pandera.pandas as pa
from pandera.typing import Series


class FeatureSchema(pa.DataFrameModel):

    Stint: Series[int] = pa.Field(ge=1, le=8,
        description="スティント番号（1〜8）")

    Year: Series[int] = pa.Field(isin=[2022, 2023, 2024, 2025],
        description="シーズン年")

    Driver: Series[str] = pa.Field(
        description="ドライバーコード")

    Race: Series[str] = pa.Field(
        description="レース名")

    TyreLife: Series[float] = pa.Field(ge=1.0, le=77.0,
        description="タイヤ使用ラップ数（1〜77）")

    RaceProgress: Series[float] = pa.Field(ge=0.01, le=1.0,
        description="レース進捗率（0.01〜1.0）")

    Compound: Series[str] = pa.Field(
        isin=["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"],
        description="タイヤコンパウンド")

    class Config:
        strict = True
        coerce = True
```

### 各フィールドの設計判断

| フィールド | 制約 | 理由 |
|---|---|---|
| `Stint` | `ge=1, le=8` | データ上の最大スティント数は 8 |
| `Year` | `isin=[2022, 2023, 2024, 2025]` | 連続値ではなく「許可された年のみ」を明示 |
| `Driver` | 制約なし | 887種類のコードを列挙するのは現実的でない。カテゴリ管理は `categories.json` に委ねる |
| `Race` | 制約なし | 同上 |
| `TyreLife` | `ge=1.0, le=77.0` | データの実測最大値 77 を上限に。float 型（ラップ数だが小数あり） |
| `RaceProgress` | `ge=0.01, le=1.0` | 0.0 は「レース開始前」を意味し実質使われないため下限は 0.01 |
| `Compound` | `isin=[...]` | 有限の選択肢なので列挙で制約 |

---

## 3. `strict=True` の意味と使い方

### `strict=True` とは

スキーマに定義されていないカラムが DataFrame に含まれていた場合、**エラーを出す**設定。

デフォルト（`strict=False`）だと余分なカラムがあっても無視されるため、意図しない列が混入しても気づけない。

### 「バリデーション前にカラムを絞る」というパターン

`strict=True` を使う場合、事前に必要なカラムだけに絞り込む必要がある：

```python
# train.py
TRAIN_COLS = ["id"] + FEATURE_COLS + [TARGET]
train = pd.read_csv(path)[TRAIN_COLS]       # ← 必要なカラムだけ選択
TrainSchema.validate(train)                  # ← strict=True でも通る

# テストデータはidを持つが、FeatureSchema で検証するには除外が必要
InferenceSchema.validate(test[FEATURE_COLS])  # ← id を除いた7列だけ渡す
```

コードコメントにも明記している：

```python
class FeatureSchema(pa.DataFrameModel):
    """
    Notes
    -----
    strict=True のため、バリデーション前に必要なカラムだけを
    DataFrame に残しておく必要がある。
    """
```

---

## 4. `coerce=True` の意味

CSV 読み込み時に型が期待と異なるケースへの対策。

例えば `Stint` は `int` を期待しているが、CSV によっては `float`（`1.0`, `2.0`...）として読み込まれることがある。`coerce=True` があれば `1.0 → 1` と自動変換してくれる。

```python
class Config:
    strict = True   # 余分なカラムを許可しない
    coerce = True   # 型が違っても変換して通す
```

---

## 5. 子クラスの定義

### `TrainSchema`

学習データ用。`FeatureSchema` に `id` と目的変数 `PitNextLap` を追加。

```python
class TrainSchema(FeatureSchema):
    """
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
```

`PitNextLap` に `isin=[0, 1]` をかけることで、バイナリ分類のラベルとして正しい値しか入っていないことを保証する。

### `InferenceSchema`

推論入力用。追加フィールドなし。`FeatureSchema` をそのまま継承するが、**意図を名前で示す**ために別クラスとして定義した。

```python
class InferenceSchema(FeatureSchema):
    """
    Streamlit のシミュレーターがモデルに渡す DataFrame を検証する。
    id・PitNextLap は含まない。7特徴量のみ。
    """

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True
```

---

## 6. トラブル：`import pandera as pa` の FutureWarning

### 問題

当初のコード：

```python
import pandera as pa
```

実行時に FutureWarning が出ていた：

```
FutureWarning: The top-level pandera module will be deprecated in a future version.
Please use `import pandera.pandas as pa` instead.
```

### 対処

```python
# Before
import pandera as pa

# After
import pandera.pandas as pa
```

Pandera v0.20 以降、pandas 向けの API は `pandera.pandas` サブモジュールに整理されている。トップレベルの `pandera` は将来的に廃止予定。

---

## 7. スキーマの使われる場所

| ファイル | 使用するスキーマ | タイミング |
|---|---|---|
| `scripts/preprocess.py` | `TrainSchema`, `InferenceSchema` | CSV 読み込み後、processed/ 保存前 |
| `src/f1_pit_stops/models/train.py` | `TrainSchema`, `InferenceSchema` | CSV 読み込み直後（学習前） |

`train.py` に前処理を統合したため、実質的には `train.py` 内で両スキーマが使われる。`scripts/preprocess.py` は参照用として残している。

---

## 8. `pipeline/` モジュールの廃止

### 経緯

当初は `pipeline/` というモジュールを作り、前処理パイプラインとバリデーションを分離する設計を考えていた。しかし実装を進める中で：

- バリデーションだけなら `schema/__init__.py` に書けば十分
- 前処理自体もシンプルな列選択・型変換のみ

という状況になり、`pipeline/` は実質 Pandera を呼ぶだけのラッパーになっていた。

### ユーザーの提案

> 「Pipelineについてですが、実質的にはバリデーションを行うだけの役割になっているのであれば、もうプリプロセスに統合してしまってもいいのではないかと思うのですが、どう思いますか？」

→ 削除に合意。`schema/__init__.py` に全スキーマを集約し、`pipeline/` は廃止。

---

## 9. 最終的なスキーマファイル全体

```python
# src/f1_pit_stops/schema/__init__.py

import pandera.pandas as pa
from pandera.typing import Series


class FeatureSchema(pa.DataFrameModel):
    """モデルで使う 7 特徴量の共通スキーマ。"""

    Stint: Series[int] = pa.Field(ge=1, le=8, description="スティント番号（1〜8）")
    Year: Series[int] = pa.Field(isin=[2022, 2023, 2024, 2025], description="シーズン年")
    Driver: Series[str] = pa.Field(description="ドライバーコード")
    Race: Series[str] = pa.Field(description="レース名")
    TyreLife: Series[float] = pa.Field(ge=1.0, le=77.0, description="タイヤ使用ラップ数（1〜77）")
    RaceProgress: Series[float] = pa.Field(ge=0.01, le=1.0, description="レース進捗率（0.01〜1.0）")
    Compound: Series[str] = pa.Field(
        isin=["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"],
        description="タイヤコンパウンド",
    )

    class Config:
        strict = True
        coerce = True


class TrainSchema(FeatureSchema):
    """学習データ用スキーマ。FeatureSchema + id + PitNextLap。"""

    id: Series[int] = pa.Field(ge=0, description="行識別子")
    PitNextLap: Series[int] = pa.Field(isin=[0, 1], description="目的変数")

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True


class InferenceSchema(FeatureSchema):
    """推論入力用スキーマ。7特徴量のみ。"""

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True
```

---

## 10. 学んだこと・ポイントまとめ

| ポイント | 内容 |
|---|---|
| `import pandera.pandas as pa` | v0.20 以降はサブモジュールを直接インポートする |
| `strict=True` の使い方 | バリデーション前に必要なカラムだけ選択する（`df[COLS]`）がセット |
| `coerce=True` の役割 | CSV 読み込み時の型の揺れを自動変換で吸収する |
| 3クラス継承の意義 | 特徴量定義を1箇所に集約しつつ、用途ごとの意図を名前で示す |
| Driver・Race は制約なし | カテゴリ管理は `categories.json` に委ね、スキーマは「型と構造」だけを見る |
| `pipeline/` 廃止 | ラッパーを作るより `schema/__init__.py` に集約した方がシンプル |
