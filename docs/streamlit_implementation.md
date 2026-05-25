# Streamlit 実装記録

> F1 Pit Stop Predictor PoC における Streamlit UI の設計思想・試行錯誤を、時系列で記録したドキュメント。

---

## 0. なぜ Streamlit を採用したか

このプロジェクトの目的は「ポートフォリオとして業務の解像度を示すこと」。  
F1 のピットストップ予測モデルをただ Kaggle に提出するだけでなく、**インタラクティブなシミュレーター**として動かすことで、モデルの振る舞いを直感的に説明できるようにした。

Streamlit を選んだ理由：
- Python だけで動く（フロントエンド知識不要）
- スライダーやセレクトボックスなどの UI 部品が豊富
- リアルタイムで予測結果が更新される UX が作りやすい

---

## 1. アプリの目的と対象ユーザーの整理

### 最初の問い：「何を見せたいのか」

設計初期に、Streamlit アプリで何を実現したいかを整理した。候補は主に2つ：

**案A：EDA ダッシュボード重視**
- データの分布や相関をビジュアルで見せる
- モデルの説明可能性（SHAP等）を前面に出す

**案B：予測シミュレーター重視**
- 特徴量を自由にいじって予測がどう変わるかを体感する
- ポートフォリオとして「業務的に使えるツール」感を出す

### ユーザーの方針決定

> 「想定しているのは、データの特徴量をいじって予測がどう変わるか試せる『シミュレーター』中心。ポートフォリオとして使おうかなと思っているので、業務の解像度という点でどちらが良いか意見も聞かせてもらえれば。」

→ **案B（シミュレーター重視）** を採用。加えて、MLflow の実験ログと Submission スコア分布も見られる「実験管理」ページを追加することで、モデル開発のプロセスも伝えられる構成にした。

---

## 2. シミュレーターのUI設計

### 2-1. 入力特徴量の決定

7つの特徴量をシミュレーターの入力とした。選定の背景：

| 特徴量 | 型 | 入力ウィジェット | 選定理由 |
|---|---|---|---|
| Driver | str | セレクトボックス | ドライバーによる挙動の違いを比較できる |
| Race | str | セレクトボックス | サーキット特性（モナコ vs ハンガロリンク等）が影響 |
| Compound | str | セレクトボックス | 「どのタイヤを選ぶか」はユーザー目線で重要な判断軸 |
| Stint | int | スライダー（1〜8）| ピットの「何回目か」は戦略の文脈 |
| Year | int | スライダー（2022〜2025）| レギュレーション変化の影響を確認できる |
| TyreLife | float | スライダー（1〜77）| タイヤの劣化度合いを直感的に操作できる |
| RaceProgress | float | スライダー（0.01〜1.0）| レースの進捗（序盤・終盤）による戦略の違い |

**Compound についての補足：**
> 「コンパウンドは特徴量としてあまり強くなかったのでどうかなと思ったんですけど、ユーザー目線で立った時に『どのタイヤを選ぶか』という要素は重要なので、入れてもいいかなと思います。」

特徴量重要度は低くても、**UX 上の説明価値**を優先してシミュレーターに含めた判断。

### 2-2. A/B 比較という設計

シミュレーターを**2シナリオ同時比較**（シナリオ A vs シナリオ B）の形にした。

理由：
- 「TyreLife が 10 の場合と 30 の場合ではどれだけピット確率が変わるか」を一画面で確認できる
- 「← A をコピー」ボタンで A の設定をまるごと B にコピーし、1つだけ変えて比較できる

```
[シナリオ A]         [シナリオ B]
Driver: HAM          Driver: HAM
...                  ...
TyreLife: 10    →    TyreLife: 30  ← ここだけ変える
```

### 2-3. ゲージグラフと色分け

予測確率（0〜1）をゲージグラフで表示。確率のレンジに応じて色が変わる：

| 確率 | バーの色 | 意味 |
|---|---|---|
| 0.0〜0.3 | 緑 (#2ecc71) | ピット可能性低 |
| 0.3〜0.7 | 橙 (#f39c12) | 不確実 |
| 0.7〜1.0 | 赤 (#e74c3c) | ピット可能性高 |

0.5 に閾値ラインを引き、「どちら側にいるか」を視覚的に示した。

---

## 3. 「ドライバーが887種類いる問題」の対処

### 問題提起

> 「ドライバー887種から選ばせるのは結構しんどいと思うので、もう少し絞った方がいいかもしれません。」

887人のドライバーをセレクトボックスで選ばせるのはUX上の課題。

### 対処方針

Streamlit の `st.selectbox` は**インクリメンタルサーチ（インライン絞り込み）が標準搭載**されている。  
ユーザーが名前の一部を入力するだけで候補が絞り込まれるため、人工的なリスト制限は不要という判断になった。

```python
driver = st.selectbox(
    "🏎️ Driver",
    options=categories["Driver"],
    key=f"driver{s}",
    help="名前の一部を入力して絞り込めます",  # ← ヘルプテキストで案内
)
```

---

## 4. カテゴリ選択肢の動的生成

### 設計方針

Driver・Race の選択肢はコードにハードコードせず、**学習時に `categories.json` に保存したカテゴリ一覧** から動的に取得する。

```python
@st.cache_resource
def load_resources() -> tuple[list | None, dict | None]:
    with open(cat_path, encoding="utf-8") as f:
        categories = json.load(f)
    return models, categories

# UI側
driver = st.selectbox("🏎️ Driver", options=categories["Driver"], ...)
race   = st.selectbox("🏁 Race",   options=categories["Race"],   ...)
```

メリット：
- 学習データに含まれるドライバー・レースだけが選択肢に現れる（未知カテゴリを入力する心配がない）
- データが更新されても UI 側のコードを変更する必要がない

---

## 5. ワークフローをめぐる設計判断

### 最初の問い：「3ステップに分ける必要はあるか？」

当初のワークフローは：

```
1. uv run python scripts/preprocess.py
2. uv run f1-train
3. uv run streamlit run ...
```

### ユーザーの問題提起

> 「今回『前処理』『学習』『アプリ起動』の手順を分けている理由は何ですか？分けるメリットが自分にはよくわからず、個人的にはコード1行でアプリが立ち上がった方が使い勝手がいいのではないかと思っています。」

正当な指摘。元々は「再利用性」のために分けていたが、**PoC の目的には合っていなかった**。

### キャッシュの活用提案

> 「モデルのロードと推論の段階でキャッシュを入れることで、割と解決できるところがあるんじゃないかなと思うんだけど。他にも入れられそうなところにキャッシュを入れて、なるべくスムーズにUIが動くようにしてほしいんだけど、どうですか？」

Streamlit のキャッシュ機能で解決できる範囲を整理した結果：

| 処理 | キャッシュ種別 | 理由 |
|---|---|---|
| モデル読み込み | `@st.cache_resource` | オブジェクトをセッション間で共有。再起動まで1回のみ読み込む |
| 予測実行 | `@st.cache_data` | 同じ入力の組み合わせはキャッシュから即返却 |
| ゲージグラフ生成 | `@st.cache_data` | 同じ確率値なら図を再生成しない |

ただし「学習自体」はキャッシュでは解決できない（数十秒〜数分かかる）。

### 最終的なワークフロー

> 「とりあえず最初は1ステップで実装して、ストレスを感じたら2ステップにする。」

→ 前処理を `train.py` に統合し、2ステップに削減：

```bash
uv run f1-train          # 学習（初回のみ。モデルが残れば不要）
uv run streamlit run src/f1_pit_stops/app/main.py
```

モデルが存在しない場合のエラーメッセージも整備：

```python
if models is None:
    st.warning(
        "モデルが見つかりません。先に以下を実行してください。\n\n"
        "```\nuv run f1-train\n```"
    )
    return
```

---

## 6. `@st.cache_data` の LightGBM Booster 問題

### 問題

`predict()` 関数に `@st.cache_data` を使う場合、通常はモデルオブジェクトを引数に取る。しかし LightGBM の `Booster` オブジェクトはハッシュ化できないため、`@st.cache_data` の引数に渡せない。

### 解決策

モデルオブジェクトを引数に取らず、**関数内で `load_resources()`（`@st.cache_resource` 済み）を呼ぶ**設計にした。

```python
@st.cache_data
def predict(inputs_json: str) -> float:
    """入力を JSON 文字列として受け取ることでキャッシュキーを文字列化する。"""
    models, categories = load_resources()  # cache_resource から取得（ハッシュ化不要）
    inputs = json.loads(inputs_json)
    df = pd.DataFrame([inputs])
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    X = df[FEATURE_COLS]
    return float(np.mean([m.predict(X)[0] for m in models]))
```

呼び出し側でも dict を JSON 文字列に変換してから渡す：

```python
pred_a = predict(json.dumps(inputs_a, ensure_ascii=False))
```

---

## 7. セッション状態の管理

スライダーとセレクトボックスの**デフォルト値**を明示的に `st.session_state` に設定。これにより：
- アプリ起動直後から「それらしい値」が入っている
- 「← A をコピー」ボタンが機能する（B の state を A の値で上書き後に `st.rerun()`）

```python
def init_session_state(categories: dict) -> None:
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
```

---

## 8. ページ構成：2ページ構成

```python
page = st.sidebar.radio(
    "ページ",
    ["🔮 シミュレーター", "📊 実験管理"],
    label_visibility="collapsed",
)
```

| ページ | 内容 |
|---|---|
| 🔮 シミュレーター | 7特徴量を入力 → 予測確率をゲージで表示。A/B 比較 |
| 📊 実験管理 | MLflow の実験ログ一覧、OOF AUC 推移グラフ、Submission スコア分布 |

### 実験管理ページを追加した理由

> 「MLflowのテーブルが見れていいことって何ですか？」

ポートフォリオとして見せる場合、「どんな実験をして、どう精度が上がったか」のプロセスも伝えられると業務的な解像度の高さを示せる。Streamlit 内にそのまま埋め込むことで、別途 MLflow UI を立ち上げる手間なく確認できる。

---

## 9. クレジット表示の設計

### 要件

> 「使用したデータセットのライセンスに基づいて適切な帰属（attribution）を表示すること。目立ちすぎない場所に配置したい。」

ユーザーからライセンス情報と正確なクレジット文言が提供された：

```
Both datasets are licensed under CC BY 4.0
1. Kaggle Playground Series S6E5 — Predicting F1 Pit Stops
2. F1 Strategy Dataset by Aadit Gupta
```

### 実装

全ページ共通で最下部に表示する `show_credits()` を作成：

```python
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
```

`st.caption()` を使うことで、本文より小さいフォントサイズで目立ちすぎず表示。

---

## 10. 最終的なファイル構造

```
src/f1_pit_stops/app/main.py
```

```
定数・パス設定
  ↓
load_resources()       @st.cache_resource  モデル + categories.json
predict()              @st.cache_data      入力JSON → 予測確率
gauge_chart()          @st.cache_data      確率値 → Plotly ゲージ図
render_inputs()        ─                   1シナリオ分の入力ウィジェット群
init_session_state()   ─                   セッション初期化（初回のみ）
copy_a_to_b()          ─                   A の値を B にコピー
  ↓
page_simulator()       ─                   シミュレーターページ
page_experiments()     ─                   実験管理ページ
show_credits()         ─                   フッタークレジット
  ↓
メイン：ページルーティング
```

---

## 11. 学んだこと・ポイントまとめ

| ポイント | 内容 |
|---|---|
| `@st.cache_resource` vs `@st.cache_data` | リソース（モデル等）は `cache_resource`、データ変換は `cache_data` |
| LightGBM の cache 回避 | Booster はハッシュ化不要。`cache_resource` で持ち、`cache_data` 側は JSON 文字列をキーにする |
| セレクトボックスの絞り込み | Streamlit の `st.selectbox` はインクリメンタルサーチ内蔵。人工的なリスト制限は不要 |
| A/B 比較の UX | 「← コピー」ボタン + `st.rerun()` で自然な操作フロー |
| ワークフロー設計 | PoC は「シンプルに動く」が最優先。ストレスを感じてから分割する |
