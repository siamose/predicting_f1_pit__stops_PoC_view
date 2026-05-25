"""F1 Pit Stop Predictor — Streamlit アプリ

起動方法:
    uv run streamlit run src/f1_pit_stops/app/main.py

事前に以下を実行しておくこと:
    uv run python scripts/preprocess.py   # raw → processed/
    uv run f1-train                        # 学習・モデル保存
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── パス ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[3]   # src/pkg/app/main.py → 3階層上 = プロジェクトルート
MODEL_DIR = ROOT / "model"         # ローカル学習済みモデル（gitignore 対象）
DEMO_DIR  = ROOT / "model" / "demo"  # デモ用モデル（git にコミット済み）

# ── 定数 ─────────────────────────────────────────────────────────────────────
FEATURE_COLS = ["Stint", "Year", "Driver", "Race", "TyreLife", "RaceProgress", "Compound"]
CAT_COLS = ["Driver", "Race", "Compound"]

COMPOUND_OPTIONS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]


# ── モデル・カテゴリ読み込み ───────────────────────────────────────────────────

@st.cache_resource
def load_resources() -> tuple[list | None, dict | None]:
    """学習済みモデルと categories.json を読み込む。

    優先順位：
      1. model/          ローカルで学習した最新モデル（gitignore 対象）
      2. model/demo/     git にコミットされたデモ用モデル（Streamlit Cloud 用）
    どちらも存在しない場合は (None, None) を返す。
    @st.cache_resource により、アプリ起動中は1回だけ読み込む。
    """
    fold_paths = [MODEL_DIR / f"lgbm_fold{i}.pkl" for i in range(1, 5)]
    cat_path = MODEL_DIR / "categories.json"

    # ローカルモデルが存在しない場合はデモモデルにフォールバック
    if not all(p.exists() for p in fold_paths) or not cat_path.exists():
        fold_paths = [DEMO_DIR / f"lgbm_fold{i}.pkl" for i in range(1, 5)]
        cat_path = DEMO_DIR / "categories.json"

    if not all(p.exists() for p in fold_paths) or not cat_path.exists():
        return None, None

    models = []
    for p in fold_paths:
        with open(p, "rb") as f:
            models.append(pickle.load(f))

    with open(cat_path, encoding="utf-8") as f:
        categories = json.load(f)

    return models, categories


# ── 推論 ─────────────────────────────────────────────────────────────────────

@st.cache_data
def predict(inputs_json: str) -> float:
    """7特徴量の JSON 文字列を受け取り、4 fold 平均の予測確率を返す。

    @st.cache_data により、同じ入力の組み合わせはキャッシュから即返却される。
    LightGBM Booster はハッシュ化できないため引数に取らず、
    load_resources()（@st.cache_resource 済み）を内部で呼ぶ。
    """
    models, categories = load_resources()
    inputs = json.loads(inputs_json)
    df = pd.DataFrame([inputs])
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    X = df[FEATURE_COLS]
    return float(np.mean([m.predict(X)[0] for m in models]))


# ── UI パーツ ─────────────────────────────────────────────────────────────────

@st.cache_data
def gauge_chart(value: float, title: str) -> go.Figure:
    """予測確率をゲージグラフで表示する。確率に応じて色が変わる。

    @st.cache_data により、同じ確率値・タイトルの図はキャッシュから返す。
    """
    if value < 0.3:
        bar_color = "#2ecc71"   # 緑：ピット可能性低
    elif value < 0.7:
        bar_color = "#f39c12"   # 橙：不確実
    else:
        bar_color = "#e74c3c"   # 赤：ピット可能性高

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".3f", "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".1f"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0.0, 0.3], "color": "#d5f5e3"},
                {"range": [0.3, 0.7], "color": "#fef9e7"},
                {"range": [0.7, 1.0], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": 0.5,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def render_inputs(scenario: str, categories: dict) -> dict:
    """1シナリオ分の入力ウィジェットを描画し、入力値を dict で返す。"""
    s = f"_{scenario}"   # session_state のキーサフィックス

    driver = st.selectbox(
        "🏎️ Driver",
        options=categories["Driver"],
        key=f"driver{s}",
        help="名前の一部を入力して絞り込めます",
    )
    race = st.selectbox(
        "🏁 Race",
        options=categories["Race"],
        key=f"race{s}",
    )
    compound = st.selectbox(
        "🔵 Compound",
        options=COMPOUND_OPTIONS,
        key=f"compound{s}",
    )
    stint = st.slider("Stint", min_value=1, max_value=8, key=f"stint{s}")
    year = st.slider("Year", min_value=2022, max_value=2025, step=1, key=f"year{s}")
    tyre_life = st.slider(
        "TyreLife（ラップ数）", min_value=1.0, max_value=77.0, step=1.0, key=f"tyre_life{s}"
    )
    race_progress = st.slider(
        "RaceProgress（0〜1）", min_value=0.01, max_value=1.0, step=0.01, key=f"race_progress{s}"
    )

    return {
        "Driver": driver,
        "Race": race,
        "Compound": compound,
        "Stint": stint,
        "Year": year,
        "TyreLife": tyre_life,
        "RaceProgress": race_progress,
    }


def init_session_state(categories: dict) -> None:
    """セッション変数のデフォルト値を設定する（初回のみ）。"""
    defaults = {
        "driver": categories["Driver"][0],
        "race": categories["Race"][0],
        "compound": "MEDIUM",
        "stint": 2,
        "year": 2024,
        "tyre_life": 15.0,
        "race_progress": 0.3,
    }
    for scenario in ("a", "b"):
        for key, val in defaults.items():
            state_key = f"{key}_{scenario}"
            if state_key not in st.session_state:
                st.session_state[state_key] = val


def copy_a_to_b() -> None:
    """シナリオ A の入力値をシナリオ B にコピーする。"""
    keys = ["driver", "race", "compound", "stint", "year", "tyre_life", "race_progress"]
    for key in keys:
        st.session_state[f"{key}_b"] = st.session_state[f"{key}_a"]


# ── ページ：シミュレーター ────────────────────────────────────────────────────

def page_simulator() -> None:
    st.title("🔮 Pit Stop シミュレーター")
    st.caption("7つの特徴量を指定して、次のラップでピットインする確率を予測します。")

    models, categories = load_resources()  # @st.cache_resource 済み

    if models is None:
        st.warning(
            "モデルが見つかりません。先に以下を実行してください。\n\n"
            "```\nuv run f1-train\n```"
        )
        return

    init_session_state(categories)

    # ── 入力：2カラム ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("シナリオ A")
        inputs_a = render_inputs("a", categories)

    with col_b:
        sub_col, btn_col = st.columns([3, 1])
        sub_col.subheader("シナリオ B")
        if btn_col.button("← A をコピー", use_container_width=True):
            copy_a_to_b()
            st.rerun()
        inputs_b = render_inputs("b", categories)

    # ── 予測・表示 ────────────────────────────────────────────────────────
    st.divider()

    pred_a = predict(json.dumps(inputs_a, ensure_ascii=False))
    pred_b = predict(json.dumps(inputs_b, ensure_ascii=False))
    diff = pred_b - pred_a

    g_col_a, g_col_b, g_col_diff = st.columns([2, 2, 1])

    with g_col_a:
        st.plotly_chart(gauge_chart(pred_a, "シナリオ A"), use_container_width=True)

    with g_col_b:
        st.plotly_chart(gauge_chart(pred_b, "シナリオ B"), use_container_width=True)

    with g_col_diff:
        st.write("")
        st.write("")
        st.metric(
            label="差 (B − A)",
            value=f"{diff:+.3f}",
            delta=None,
        )
        label = "B の方がピット確率が高い" if diff > 0 else "A の方がピット確率が高い"
        st.caption(label)


# ── ページ：実験管理 ──────────────────────────────────────────────────────────

def page_experiments() -> None:
    st.title("📊 実験管理")

    # ── MLflow ────────────────────────────────────────────────────────────
    st.subheader("MLflow 実験結果")
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())

    try:
        runs = mlflow.search_runs(search_all_experiments=True)
        if runs.empty:
            st.info("`uv run f1-train` を実行するとここに結果が表示されます。")
        else:
            metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
            param_cols = [c for c in runs.columns if c.startswith("params.")]
            show_cols = (
                ["tags.mlflow.runName", "status", "start_time"]
                + metric_cols
                + param_cols
            )
            show_cols = [c for c in show_cols if c in runs.columns]
            st.dataframe(
                runs[show_cols].sort_values("start_time", ascending=False),
                use_container_width=True,
            )

            # OOF AUC 推移
            auc_col = "metrics.oof_auc_mean"
            if auc_col in runs.columns and runs[auc_col].notna().any():
                st.subheader("OOF AUC 推移")
                runs_sorted = runs.dropna(subset=[auc_col]).sort_values("start_time")
                fig = go.Figure(go.Scatter(
                    x=runs_sorted["start_time"],
                    y=runs_sorted[auc_col],
                    mode="lines+markers",
                    hovertemplate="AUC: %{y:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="実行日時", yaxis_title="OOF AUC",
                    height=320, paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"MLflow の読み込みに失敗しました: {e}")

    # ── Submission ────────────────────────────────────────────────────────
    st.subheader("Submission スコア分布")
    sub_path = ROOT / "data" / "processed" / "submission.csv"

    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("件数", f"{len(sub):,}")
        c2.metric("平均スコア", f"{sub['PitNextLap'].mean():.4f}")
        c3.metric("中央値", f"{sub['PitNextLap'].median():.4f}")

        fig2 = go.Figure(go.Histogram(
            x=sub["PitNextLap"], nbinsx=100,
            marker_color="#3366cc",
            hovertemplate="スコア: %{x:.3f}<br>件数: %{y}<extra></extra>",
        ))
        fig2.update_layout(
            xaxis_title="PitNextLap（予測確率）", yaxis_title="件数",
            height=320, paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("`uv run f1-train` を実行すると submission.csv が生成されます。")


# ── クレジット ────────────────────────────────────────────────────────────────

def show_credits() -> None:
    st.divider()
    st.caption(
        "**Data Sources** &nbsp;|&nbsp; "
        "Both datasets are licensed under "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
    )
    st.caption(
        "1. **Kaggle Playground Series S6E5** — Predicting F1 Pit Stops &nbsp; "
        "[→ Kaggle](https://www.kaggle.com/competitions/playground-series-s6e5)"
    )
    st.caption(
        "2. **F1 Strategy Dataset** by Aadit Gupta &nbsp; "
        "[→ Kaggle](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction)"
    )
    st.caption("※ 本アプリは合成データを使用しています。")


# ── メイン ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
)

page = st.sidebar.radio(
    "ページ",
    ["🔮 シミュレーター", "📊 実験管理"],
    label_visibility="collapsed",
)

if page == "🔮 シミュレーター":
    page_simulator()
else:
    page_experiments()

show_credits()
