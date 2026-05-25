# 特徴量選択記録

> F1 Pit Stop Predictor PoC における特徴量選択の思考プロセス・根拠を時系列で記録したドキュメント。  
> EDA → 外れ値処理 → 特徴量エンジニアリング → 重要度分析 → 最終選択の流れを追う。

---

## 0. 出発点：生データに含まれる変数

`data/raw/train.csv` に含まれていた列：

| 変数名 | 型 | 説明 |
|---|---|---|
| `id` | int | 行識別子 |
| `PitNextLap` | int | **目的変数**（次のラップでピットインするなら 1） |
| `Stint` | int | スティント番号（何回目のピット間か） |
| `Year` | int | シーズン年 |
| `Driver` | str | ドライバーコード |
| `Race` | str | レース名 |
| `TyreLife` | float | タイヤ使用ラップ数 |
| `RaceProgress` | float | レース進捗率（0〜1） |
| `Compound` | str | タイヤコンパウンド |
| `LapTime (s)` | float | ラップタイム（秒） |
| `LapTime_Delta` | float | 直前ラップとのタイム差 |
| `Cumulative_Degradation` | float | 累積タイヤ劣化量 |
| `LapNumber` | int | ラップ番号 |

最終的に使用した特徴量は **7つ**（Stint / Year / Driver / Race / TyreLife / RaceProgress / Compound）。  
残りは EDA・特徴量エンジニアリングの過程で「PoC の文脈では不採用」と判断した。

---

## 1. EDA：カテゴリ変数の分析

### 1-1. SWEETVIZ による俯瞰

最初に SWEETVIZ を使って Train vs Test 比較、および PitNextLap=0 vs 1 比較を実施した。

```python
# train vs test の全変数分布を一括比較
report = sv.compare([train, "Train"], [test, "Test"], target_feat="PitNextLap")
report.show_html("sv_train_vs_test.html")

# 目的変数の 0 / 1 グループを比較
report = sv.compare([train_0, "Train_0"], [train_1, "Train_1"], target_feat="PitNextLap")
report.show_html("sv_0_vs_1.html")
```

目的：変数ごとの分布・欠損・偏りを一度に把握し、「詳しく見るべき変数」を絞る。

### 1-2. Lift ヒートマップ（カテゴリ変数）

Race・Compound について、水準ごとの **Lift**（全体の P(1) に対する倍率）をヒートマップで可視化。

```python
# Lift の定義：P(target=k | 水準) / P(target=k 全体)
# 1.0 = 全体と同じ、>1.0 = そのクラスが濃い、<1.0 = 薄い

pct = pd.crosstab(train[col], train[target], normalize="index")
lift = pd.DataFrame(
    {0: pct[0] / p0_global, 1: pct[1] / p1_global},
    index=pct.index,
)
```

> **Driver を除外した理由**：カーディナリティが 887 と高く、ヒートマップで一覧するには多すぎる。
> 視覚的な分析は Race・Compound に絞り、Driver はモデルに委ねる方針にした。

### 1-3. Plotly 横棒グラフ（P(1|水準)）

ヒートマップだけでは「水準間の大小・順位」が読み取りにくいため、横棒グラフも併用した。

```
横軸：P(PitNextLap=1 | その水準)
参照線：全体の P(1)（破線）
ホバー：n（件数）表示で、件数が少ない水準の「偶然の高さ」に騙されないよう対策
```

**ヒートマップより横棒が優れている点**：
- 水準間の順位付けが棒の長さで直感的にわかる
- 参照線（全体平均）との差がひと目でわかる
- ズーム・パン・件数ホバーで Race のように水準が多い変数でも探索できる

---

## 2. EDA：連続変数の分析

相関行列を可視化し、連続変数間の関係を確認。

```python
Nr = train[num].drop(columns=["id"], errors="ignore")
sns.heatmap(Nr.corr(), annot=True, cmap="coolwarm")
```

ここで確認したのは **多重共線性のリスク**。モデル（LightGBM）自体は多重共線性に比較的ロバストだが、特徴量の解釈性・シミュレーターでの操作性を考慮するうえで相関構造の把握は重要だった。

---

## 3. 外れ値処理：クリッピング

LapTime / LapTime_Delta / Cumulative_Degradation の3変数に外れ値が確認された。

### 処理方針

```python
# 分位数は train のみで算出 → test にも同じ閾値を適用（リーク防止）
q95_lap   = float(train["LapTime (s)"].quantile(0.95))   # 109.7 秒
q5_delta  = float(train["LapTime_Delta"].quantile(0.05)) # -24.8 秒
q95_delta = float(train["LapTime_Delta"].quantile(0.95)) # +19.0 秒
q95_cumdeg = float(train["Cumulative_Degradation"].quantile(0.95)) # 84.4
```

**クリップ後に元列をドロップ**し、`_clipped` 列と `_clip_flag`（0=通常 / 1=クリップ対象だった）の2種類を生成：

```
LapTime (s)_clipped        + LapTime (s)_clip_flag
LapTime_Delta_clipped      + LapTime_Delta_clip_flag
Cumulative_Degradation_clipped + Cumulative_Degradation_clip_flag
```

> **clip_flag を残した理由**：クリップされた行（異常値行）は「セーフティカーが入った」「赤旗中断」などの特別な状況を示している可能性がある。この情報をモデルに与えることで、クリップによる情報損失を補う。

### clip_flag × 目的変数の確認

clip_flag=1 の行と PitNextLap=1 の関連をクロス集計で確認。一部の変数では clip_flag=1 のときに P(1) が有意に変化しており、信号として有効であることを確認した。

---

## 4. 特徴量エンジニアリング

### 4-1. スティント内累積特徴量（expanding）

スティント内で「現ラップまでの統計」をリアルタイムに計算：

```python
# (Driver, Race, Stint) でグルーピング → LapNumber 昇順で expanding
sorted_df["lap_time_cumean"]      = g.expanding().mean()   # 累積平均ラップタイム
sorted_df["lap_time_custd"]       = g.expanding().std()    # 累積標準偏差（タイヤ劣化の不均一性）
sorted_df["laps_in_stint_so_far"] = g.expanding().count()  # スティント内の経過ラップ数
```

> **expanding を選んだ理由**：rolling（固定窓）は「直近 N ラップ」しか見ないが、expanding は「スティント開始からの全ラップ」を使う。ピット判断はそのスティント全体の劣化傾向を見て行われるため、expanding の方が文脈に合っている。
>
> **リーク回避**：`expanding().mean()` は現ラップ自身を含む。「現ラップ時点で見えている情報」として問題ないが、Shift はしていないため未来情報は含まない。

### 4-2. 履歴スティント長（prev_stint_length_mean）

同一（Driver, Race）の**前のスティントの平均長**を特徴量として追加：

```python
# ① train の (Driver, Race, Stint) でスティント長を集計
# ② Stint 昇順に expanding().mean() → shift(1) で「現スティントを除いた過去の平均」
# ③ Stint=1（前スティントなし）の NaN → train 全体の stint_length 中央値で補完
```

> **test のリーク防止**：履歴スティント長のテーブルは **train のみ**で計算し、test にはマージで付与する。test 側には test の正解情報を混ぜない。
>
> **test 未登場の (Driver, Race, Stint) の補完**：グローバル中央値で埋める。未知の組み合わせに対するフォールバック値として中央値は外れ値の影響を受けにくく妥当。

---

## 5. 特徴量重要度の分析

上記の前処理・特徴量エンジニアリングをすべて適用したうえで、LightGBM × 4-fold CV で **gain ベースの特徴量重要度** を計算した。

```python
feature_cols = [c for c in train_enc.columns if c not in ["id", "PitNextLap"]]
# この時点での feature_cols に含まれていた候補：
# Stint, Year, Driver, Race, TyreLife, RaceProgress, Compound
# LapTime (s)_clipped, LapTime_Delta_clipped, Cumulative_Degradation_clipped
# LapTime (s)_clip_flag, LapTime_Delta_clip_flag, Cumulative_Degradation_clip_flag
# lap_time_cumean, lap_time_custd, laps_in_stint_so_far
# prev_stint_length_mean
# LapNumber
```

> **この時点では LabelEncoder を使用**（train+test 全ラベルで fit し、未知ラベル対策）。重要度の算出だけが目的なので「人工的な順序付け」の問題はここでは許容した。最終的な本番モデルでは LightGBM ネイティブカテゴリカルに切り替えた（→ `pandera_implementation.md` 参照）。

### 重要度の上位を占めた変数

EDA・重要度分析を通じて、**TyreLife と RaceProgress が特に強い信号** であることが確認された。これはドメイン的にも直感と一致する：

- **TyreLife**：タイヤが何ラップ使われているか。劣化が進むほどピット確率が上がる
- **RaceProgress**：レースの終盤ほど「最後のピットを入れる」タイミングの判断が入る

**Driver・Race** はカテゴリカルながら重要度が高く、ドライバー特有のピット戦略やサーキット特性がモデルに効いていることが示された。

---

## 6. 最終的な特徴量選択：7変数への絞り込み

### 絞り込みの判断軸

EDA・エンジニアリング・重要度分析を経て、**なぜ7変数に絞ったか** の理由：

#### (1) シミュレーターとして「操作できるか」

このプロジェクトの最終成果物は Streamlit シミュレーター。ユーザーが値を自由に変えて予測の変化を見るためには、**UIで直感的に操作できる変数**である必要がある。

| 変数 | UI 操作のしやすさ | 採否 |
|---|---|---|
| TyreLife | スライダーで 1〜77 ラップ設定できる | ✅ 採用 |
| RaceProgress | スライダーで 0〜1 設定できる | ✅ 採用 |
| Stint | スライダーで 1〜8 設定できる | ✅ 採用 |
| Year | スライダーで 2022〜2025 設定できる | ✅ 採用 |
| Driver / Race / Compound | セレクトボックスで選べる | ✅ 採用 |
| LapTime (s)_clipped | 「今のラップタイムを入力してください」→ UX が難しい | ❌ 不採用 |
| lap_time_cumean | 「スティント内の累積平均タイムは？」→ ユーザーが知らない | ❌ 不採用 |
| prev_stint_length_mean | 「前スティントの平均長さは？」→ 直感的でない | ❌ 不採用 |
| clip_flag | 「異常値フラグを入力してください」→ 意味不明 | ❌ 不採用 |
| LapNumber | RaceProgress と意味が重複、かつ総ラップ数が変わると解釈が変わる | ❌ 不採用 |

#### (2) 解釈しやすいか

「この特徴量が上がるとピット確率がどう変わるか」がユーザーに説明できること。エンジニアリング特徴量（cumean, custd）は説明が複雑になりがちで、PoC の説明性を下げると判断した。

#### (3) 重要度との兼ね合い

LapTime 系変数はエンジニアリング後に一定の重要度を示したが、UI で扱えない以上採用できない。7変数だけでも十分な予測力を持つことを4-fold AUC で確認したうえで絞り込んだ。

### Compound の特殊なケース

> 「コンパウンドは特徴量としてあまり強くなかったのでどうかなと思ったんですけど、ユーザー目線で立った時に『どのタイヤを選ぶか』という要素は重要なので、入れてもいいかなと思います。」

Compound は特徴量重要度が他の変数より低い結果になった。しかし：

- F1 戦略において「どのコンパウンドを履いているか」はピット判断の中心的な要素
- ユーザーが「SOFT と MEDIUM ではピット確率がどう違うか」を試せることはシミュレーターの本質的な価値
- 重要度が低い → 「モデルに効いていない」だが、「UIに不要」とは別の話

**特徴量重要度が低くても、ユーザー体験・ドメイン的文脈として採用する** という判断をした。

---

## 7. 最終特徴量の統計サマリ

train + test（合計 627,305 行）での確認：

| 特徴量 | 型 | ユニーク数 | min | max | mean | median |
|---|---|---|---|---|---|---|
| Stint | int | 8 | 1 | 8 | 1.79 | 2.0 |
| Year | int | 4 | 2022 | 2025 | 2023.5 | 2024.0 |
| Driver | str | 887 | — | — | — | — |
| Race | str | 26 | — | — | — | — |
| TyreLife | float | 78 | 1.0 | 77.0 | 14.2 | 12.0 |
| RaceProgress | float | 2097 | 0.013 | 1.0 | 0.337 | 0.269 |
| Compound | str | 5 | — | — | — | — |

欠損なし（null_count = 0）。この統計値が Pandera スキーマの値域設定の根拠になった。

---

## 8. LabelEncoder から Native Categorical への切り替え

ノートブックの重要度分析では LabelEncoder を使ったが、最終モデルでは LightGBM のネイティブカテゴリカルに切り替えた。

**理由：**

> 「ラベルエンコーダーは必要のない大小関係を生み出してしまうためあまり良くないと思っていました。」

LabelEncoder は `HAM=0, VER=1, LEC=2, ...` のように整数に変換するが、この順序には意味がない。LightGBM にとって「VER は HAM より大きい」という情報は誤りであり、分割基準が歪む。

ネイティブカテゴリカルを使えば、LightGBM は「ラベルのグループ」として処理するため、人工的な順序付けを避けられる。

この切り替えにあわせて `categories.json` を導入し、学習時のカテゴリ定義を推論時に再現できるようにした（→ `pandera_implementation.md` 参照）。

---

## 9. 特徴量選択の全体フロー

```
生データ（13変数）
    ↓
SWEETVIZ / Lift ヒートマップ / 相関行列で EDA
    ↓
LapTime 系 3変数の外れ値クリッピング（+ clip_flag 生成）
    ↓
特徴量エンジニアリング
    ├─ スティント内累積統計（lap_time_cumean, custd, laps_in_stint_so_far）
    └─ 履歴スティント長（prev_stint_length_mean）
    ↓
LightGBM 4-fold で全変数の特徴量重要度を算出
    ↓
「UI で操作できるか」「説明しやすいか」「重要度との兼ね合い」で絞り込み
    ↓
最終 7 特徴量（Stint / Year / Driver / Race / TyreLife / RaceProgress / Compound）
    ↓
Pandera スキーマ定義（値域は統計サマリから設定）
```

---

## 10. 学んだこと・ポイントまとめ

| ポイント | 内容 |
|---|---|
| 特徴量重要度だけで選ばない | Compound のように「重要度は低いが UI・ドメイン的に重要な変数」がある |
| PoC のスコープを意識する | AUC 最大化より「シミュレーターで使えるか」を優先した |
| expanding vs rolling | ピット判断はスティント全体の文脈 → expanding の方がドメインに合う |
| リーク防止の設計 | 外れ値閾値は train のみで計算、履歴特徴量は train 由来テーブルのみから付与 |
| clip_flag を残す | クリップで失った情報（異常値の発生）を別軸で保持する |
| LabelEncoder の問題 | 順序なしカテゴリへの整数エンコードは不要な大小関係を生む |
