# Hydra 実装記録

> F1 Pit Stop Predictor PoC における Hydra 導入の試行錯誤と設計思想を、時系列で記録したドキュメント。

---

## 0. なぜ Hydra を採用したか

このプロジェクトの最初の設計判断として、実験管理には **Hydra + MLflow の組み合わせ** を採用した。

- **MLflow**：実験結果（AUC、パラメータ）を記録・可視化
- **Hydra**：実験パラメータを YAML で外部化し、CLI から上書き可能にする

LightGBM のハイパーパラメータ（`learning_rate`、`num_leaves` など）をコード内にハードコードすると、実験のたびにファイルを開いて書き換える必要がある。Hydra を使えば：

```bash
# パラメータをコードを変えずに上書きできる
uv run f1-train experiment.params.learning_rate=0.01
uv run f1-train experiment.params.num_leaves=127

# 複数設定を一括実行（グリッドサーチ）
uv run f1-train --multirun \
    experiment.params.learning_rate=0.01,0.05 \
    experiment.params.num_leaves=63,127
```

---

## 1. 設定ファイルの構造設計

### 1-1. ディレクトリ構成

```
conf/
├── config.yaml           # データパス・MLflow設定（プロジェクト全体の定数）
└── experiment/
    └── default.yaml      # ハイパーパラメータ（実験ごとに差し替える想定）
```

`config.yaml` と `experiment/default.yaml` を分けた理由：
- `config.yaml`：データパスや MLflow の URI などプロジェクト固定の設定
- `experiment/default.yaml`：実験ごとに変えたいパラメータ群

### 1-2. `conf/config.yaml`

```yaml
defaults:
  - experiment: default

data:
  train_path: data/raw/train.csv
  test_path: data/raw/test.csv
  submission_path: data/processed/submission.csv

mlflow:
  tracking_uri: mlruns
  experiment_name: kaggle_06_predicting_f1_pit__stops
```

`tracking_uri` は**相対パス**（`mlruns`）として書いておき、コード側で `Path(hydra.utils.get_original_cwd()) / cfg.mlflow.tracking_uri` として解決する。理由は後述。

### 1-3. `conf/experiment/default.yaml`

```yaml
# @package _global_
experiment:
  name: default
  params:
    learning_rate: 0.05
    num_leaves: 63
    min_child_samples: 20
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 1
    random_state: 42
    n_splits: 4
```

---

## 2. エントリーポイントの設計と `pyproject.toml`

```toml
[project.scripts]
f1-train = "f1_pit_stops.models.train:main"
```

この1行で `uv run f1-train` が使えるようになる。`main()` に `@hydra.main` デコレータを付けることで、CLI 引数の解析と YAML の読み込みが自動化される。

```python
@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    ...
```

---

## 3. トラブル①：パッケージ名が数字始まりだった

### 問題

`uv run f1-train` を実行すると次のエラーが発生した：

```
SyntaxError: invalid decimal literal
```

### 原因

パッケージディレクトリ名が `06_predicting_f1_pit__stops` だった。Python はモジュール名を数字で始めることができないため、`from 06_predicting_f1_pit__stops.models.train import main` という import 文がパースエラーになる。

### ユーザーの気づき

> 「これってもしかして、ファイル名の冒頭に『06』という数字が入っているせいでインポートエラーのようなものが起きているという解釈で合っていますか？だとしたら、ファイル名を変える必要がありますよね。」

正確な理解だった。Python のモジュール識別子は変数名と同じルールに従うため、数字始まりは許可されていない。

### 対処

1. `src/06_predicting_f1_pit__stops/` → `src/f1_pit_stops/` にリネーム
2. `pyproject.toml` を更新：
   ```toml
   [project]
   name = "f1-pit-stops"

   [tool.hatch.build.targets.wheel]
   packages = ["src/f1_pit_stops"]

   [project.scripts]
   f1-train = "f1_pit_stops.models.train:main"
   ```
3. 全 import を `f1_pit_stops.*` に更新
4. `uv sync` で再インストール

---

## 4. トラブル②：Hydra が `conf/` を見つけられない

### 問題

パッケージリネーム後に実行すると：

```
Primary config module 'conf' not found in search path.
```

### 原因

当初のコードは相対パスを使っていた：

```python
@hydra.main(config_path="../../../conf", config_name="config")
```

インタラクティブに `python train.py` を実行する場合、カレントディレクトリ基準で解決される。しかし `uv run f1-train`（インストール済みエントリーポイント）で実行すると、Hydra はスクリプトの **`__file__` の場所** を起点にパスを解決しようとする。パッケージが `site-packages` 内にインストールされると、相対パスでは `conf/` に到達できない。

### ユーザーの理解

> 「SRCのパスが変わったから、正しいパスをここに入れないといけないという話ですよね。じゃあ、そのパスとそのインポートの書き方だけ、明確に修正をお願いします。」

### 対処

`config_path` を `__file__` から絶対パスとして計算する：

```python
# train.py → models/ → f1_pit_stops/ → src/ → プロジェクトルート → conf/
_CONF_DIR = str(Path(__file__).parents[3] / "conf")

@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    ...
```

`Path(__file__).parents[3]` の意味：

| インデックス | パス |
|---|---|
| `__file__` | `src/f1_pit_stops/models/train.py` |
| `parents[0]` | `src/f1_pit_stops/models/` |
| `parents[1]` | `src/f1_pit_stops/` |
| `parents[2]` | `src/` |
| `parents[3]` | プロジェクトルート |

---

## 5. トラブル③：MLflow の URI フォーマット

### 問題

Hydra 修正後、次のエラーが出た：

```
MlflowException: UnsupportedModelRegistryStoreURIException
```

### 原因

最初の実装では：

```python
mlflow.set_tracking_uri(str(root / cfg.mlflow.tracking_uri))
# → "C:\Users\la9ma\...\mlruns"  （Windows パス）
```

MLflow は URI のスキーム（`http:`、`file:` など）を見てストレージバックエンドを判別する。Windows パスの `C:\...` を渡すと、`C` がスキームとして解釈されてしまい、未知のスキームとしてエラーになる。

### ユーザーの問いかけ

> 「これって、MLflowに関するエラーなのはわかるんだけど、URIというか出力結果がないからエラーが起きているということなのかな。そのエラーの解決はいいから、まずは理由だけ教えて。」

→ 「出力結果がない」ではなく「Windows パスが URI として誤解釈される」が原因だった。

### 対処

`.as_uri()` を使って正規の `file://` URI に変換する：

```python
# Before（Windows パスをそのまま渡す）
mlflow.set_tracking_uri(str(root / cfg.mlflow.tracking_uri))

# After（file:// スキーム付きの正規 URI に変換）
mlflow.set_tracking_uri((root / cfg.mlflow.tracking_uri).as_uri())
# → "file:///C:/Users/la9ma/.../mlruns"
```

同じ修正を `app/main.py` にも適用：

```python
# page_experiments() 内
mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
```

### パス記述の統一方針

この修正をきっかけに、パスを外部コンポーネント（MLflow 等）に渡す際の方針を明確にした：

| 渡す先 | 形式 | 理由 |
|---|---|---|
| `pandas.read_csv()` | `Path` オブジェクト | pandas が Path を直接受け付ける |
| `open()` | `Path` オブジェクト | Python 標準ライブラリが Path を直接受け付ける |
| `mlflow.set_tracking_uri()` | `.as_uri()` | `file:///` スキーム必須 |
| `mlflow.log_artifact()` | `str(path)` | str が必要 |

---

## 6. プロジェクトルートの取得：`hydra.utils.get_original_cwd()`

Hydra はデフォルトで **カレントディレクトリを `outputs/` 配下に変更** する（実行ごとにタイムスタンプ付きディレクトリを作成）。これにより、`Path("data/raw/train.csv")` のような相対パスはすべて壊れる。

解決策として `hydra.utils.get_original_cwd()` を使い、**コマンドを実行したディレクトリ**（プロジェクトルート）を取得する：

```python
@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())   # ← 実行時のカレントディレクトリ
    model_dir = root / "model"

    train = pd.read_csv(root / cfg.data.train_path, ...)
    test  = pd.read_csv(root / cfg.data.test_path, ...)
```

`cfg.data.train_path` は `data/raw/train.csv`（相対パス）として YAML に書いておき、コード側で `root /` を付けて絶対パスにするパターンを一貫して使用。

---

## 7. 最終的なコード構造

```python
from pathlib import Path

_CONF_DIR = str(Path(__file__).parents[3] / "conf")

import hydra
import hydra.utils
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(hydra.utils.get_original_cwd())
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    # MLflow URI は .as_uri() で正規化
    mlflow.set_tracking_uri((root / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(dict(cfg.experiment.params))
        # ... 学習処理 ...
```

---

## 8. 実装後の成功確認

```
uv run f1-train
```

> できた！うれしい！

上記のコマンド1行で：
1. `conf/config.yaml` を読み込む
2. `data/raw/train.csv`・`data/raw/test.csv` を読み込む
3. Pandera でスキーマバリデーション
4. カテゴリ定義を `model/categories.json` に保存
5. 4-fold CV で LightGBM を学習
6. 各 fold のモデルを `model/lgbm_fold{1-4}.pkl` に保存
7. MLflow に AUC・パラメータを記録
8. `data/processed/submission.csv` を保存

---

## 9. 学んだこと・ポイントまとめ

| ポイント | 内容 |
|---|---|
| モジュール名は変数名ルールに従う | 数字始まりは使えない |
| Hydra の `config_path` | 相対パスはエントリーポイント実行時に壊れる。`__file__` から絶対パスで計算する |
| `get_original_cwd()` | Hydra が変えるカレントディレクトリを元に戻す唯一の公式手段 |
| MLflow の URI | Windows パスを直接渡すと `C:` がスキームとして誤解釈される。`.as_uri()` 必須 |
| YAML には相対パス | コード側で `root /` を付けることで、YAML が環境に依存しなくなる |

---

## Phase 2：実験管理の拡張と Optuna 統合

> 基本動作が確認できた後、実験管理をより実践的にするための改善フェーズ。

---

## 10. `train.py` の配置を見直した

### 動機

「HydraとMLflowに関係するファイルはどれか」を整理したとき、関係するのは実質 `conf/` 配下と `train.py` だけだとわかった。そこで `train.py` がパッケージ内（`src/f1_pit_stops/models/`）にある必然性がないことに気づき、`scripts/` 直下に移すことにした。

> 「Hydra/MLflowに関係するスクリプトはパッケージ外に置いたほうが自然」

### 移動前に影響範囲を洗い出した

実装前に影響が出る箇所を全部列挙してから進めた。見切り発車しないための習慣。

| ファイル | 変更内容 |
|---|---|
| `train.py` 内の `_CONF_DIR` | `parents[3]` → `parents[1]`（階層が変わるため） |
| `pyproject.toml` | `[project.scripts]` エントリーポイントを削除（パッケージ外になるため使えない） |
| `README.md` | `uv run f1-train` → `uv run python scripts/train.py` |
| `README.md` | Streamlitのパスtypoも合わせて修正（`src/06_predicting...` → `src/f1_pit_stops/...`） |

### パス計算の変化

```python
# 移動前：train.py → models/ → f1_pit_stops/ → src/ → プロジェクトルート
_CONF_DIR = str(Path(__file__).parents[3] / "conf")

# 移動後：train.py → scripts/ → プロジェクトルート
_CONF_DIR = str(Path(__file__).parents[1] / "conf")
```

---

## 11. `config.yaml` と `experiment/default.yaml` が分かれている意味を理解した

最初は「なぜ2つに分かれているのか」が明確ではなかった。整理すると役割が明確に分かれていた。

```
conf/
  config.yaml            ← 変わらない基盤（データパス・MLflow設定）
  experiment/
    default.yaml         ← 差し替えて比較する実験条件
```

`experiment/` ディレクトリが Hydra の **config group** として機能している。この構造のおかげで、`experiment/` 内に新しい YAML を追加するだけで、コードを一切触らずに実験条件をCLIで切り替えられる。

```bash
# default.yaml を使う（何も指定しなければこれ）
uv run python scripts/train.py

# high_lr.yaml に丸ごと切り替える
uv run python scripts/train.py experiment=high_lr
```

---

## 12. 実験設定ファイルを段階的に作った

### 方針：最初から凝ったものを作らない

グリッドサーチやベイズ最適化を入れる前に、まず「手動で差し替えられる実験設定ファイル」を2つ作ることにした。凝った仕組みを最初から入れても理解しにくいという判断。

```
conf/experiment/
  default.yaml       # learning_rate=0.05, num_leaves=63
  high_lr.yaml       # learning_rate=0.1（学習率を大きくしたもの）
  high_leaves.yaml   # num_leaves=127（木の複雑さを上げたもの）
```

変えるのは1パラメータずつ。他はすべて `default.yaml` と同じ値を保つことで、比較が明確になる。

---

## 13. CLIの操作パターンを整理した

実験設定の切り替えには2通りのアプローチがあることを理解した。

| 操作 | コマンド例 | 用途 |
|---|---|---|
| YAML丸ごと切り替え | `experiment=high_lr` | 大きな単位の実験切り替え |
| 個別の値を上書き | `experiment.params.learning_rate=0.01` | defaultベースで微調整 |
| 両方の組み合わせ | `experiment=high_lr experiment.params.learning_rate=0.2` | YAMLをベースにさらに変更 |

「デフォルトのYAMLに対して上書きするのがCLI指定、YAMLごと差し替えるのが `experiment=`」という整理。

---

## 14. PowerShell でのCLI操作でハマった

bash の感覚でコマンドを書いたところ、PowerShell 固有のルールに引っかかった。

### ハマり①：行継続文字

```powershell
# bash の行継続（PowerShell では動かない）
uv run python scripts/train.py -m \
    "experiment.params.learning_rate=0.01,0.05"

# PowerShell の行継続はバッククォート
uv run python scripts/train.py -m `
    "experiment.params.learning_rate=0.01,0.05"
```

### ハマり②：カンマの扱い

PowerShell はカンマを配列の区切り文字として解釈するため、Hydra に渡す前に分解されてしまう。ダブルクォートで囲むことで回避。

```powershell
# NG：PowerShell が配列として解釈する
experiment.params.learning_rate=0.01,0.05

# OK：クォートで囲む
"experiment.params.learning_rate=0.01,0.05"
```

### ハマり③：キー階層の間違い

```powershell
# NG：存在しないキー
model.learning_rate=0.01
modelnum_leaves=20   # ← ドット抜け

# OK：実際の config 構造に合わせる
experiment.params.learning_rate=0.01
experiment.params.num_leaves=20
```

Hydra のキーはドット区切りで YAML の階層をそのまま辿る。config 構造を把握していないと間違いやすい。

---

## 15. Optuna を Hydra Sweeper として統合した

### 動機

グリッドサーチ（手動の総当たり）の次の段階として、ベイズ最適化による自動探索を加えた。Hydra には `hydra-optuna-sweeper` というプラグインがあり、`-m` フラグを付けるだけで Optuna の探索ループに切り替わる仕組みになっている。

### 実装前に留意点を整理した

いきなり実装する前に、ハマりそうな箇所を先に洗い出した。

| 留意点 | 内容 |
|---|---|
| インストールコマンド | `pip install` ではなく `uv add`（環境を合わせる） |
| YAMLのキー | サンプルコードの `model.lr` ではなく `experiment.params.learning_rate` |
| `main()` の戻り値 | `None` のままだと Optuna が次のトライアルを決められない → `float` に変更必須 |
| `direction: maximize` | AUC を最大化する場合は明示的に指定が必要 |
| MLflow のrun名 | 全トライアルが同じ名前になるため、trial番号をサフィックスとして付与 |

### `config.yaml` への追加

```yaml
defaults:
  - experiment: default
  - override hydra/sweeper: optuna   # ← Optuna Sweeper を有効化
  - _self_

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler   # ベイズ最適化（TPE）
    direction: maximize                       # AUC を最大化
    n_trials: 20
    params:
      experiment.params.learning_rate: interval(0.001, 0.1)
      experiment.params.num_leaves: choice(31, 63, 127, 255)
```

### `train.py` への変更

**① 戻り値を `float` に変更**

```python
# Before
@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    ...
    # return なし

# After
@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    ...
    return mean_auc   # Optuna が最大化する値
```

**② MLflow の run 名にトライアル番号を付与**

マルチラン時に全トライアルが同じ run 名になると MLflow UI で見分けがつかなくなる。`HydraConfig` からトライアル番号を取得して区別した。

```python
from hydra.core.hydra_config import HydraConfig

hydra_cfg = HydraConfig.get()
is_multirun = hydra_cfg.mode.name == "MULTIRUN"
run_name = (
    f"{cfg.experiment.name}_trial{hydra_cfg.job.num}"
    if is_multirun
    else cfg.experiment.name
)

with mlflow.start_run(run_name=run_name):
    ...
```

### 実行

```bash
# Optuna が自動で 20 トライアルを探索・各結果を MLflow に記録
uv run python scripts/train.py -m
```

### 3層の役割まとめ

```
Hydra  → 設定・実行層：YAMLで構造化、CLIで上書き・グリッドサーチ
MLflow → 実験ログ層：パラメータ・メトリクス・アーティファクトをGUIで比較
Optuna → 自動最適化層：SweeperプラグインでTPEベイズ探索、結果はMLflowに流入
```

---

## 16. 静的解析ツール（Ruff・mypy）について

今回は個人開発のため、どちらも導入を見送った。

| ツール | 役割 | 今回の判断 |
|---|---|---|
| Ruff | スタイル・構造チェック（未使用変数、import順など） | Claude にコードを見てもらっているため不要と判断 |
| mypy | 型の整合性チェック（引数の型違い、戻り値の不一致） | チーム開発でないためスキップ |

チーム開発や長期メンテの場合は mypy の導入価値が上がる。特に `main()` の戻り値を `None → float` に変えたような修正は、mypy があれば事前に検出できる種類の変更。
