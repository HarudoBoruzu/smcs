# Getting Started

このガイドでは、smcsライブラリの基本的な使い方を説明します。

## インストール

```bash
pip install smcs
```

開発用（テスト・リント含む）：

```bash
pip install smcs[dev]
```

## 基本的な使い方

### 1. 高レベルAPIを使う（推奨）

最も簡単な方法は、`Agent`クラスを使うことです。

```python
import jax.numpy as jnp
import pandas as pd
from smcs import LocalLevelAgent, SMCConfig

# サンプルデータを作成
data = jnp.array([[1.0], [1.2], [0.9], [1.1], [1.3], [1.5], [1.4], [1.6]])

# 設定を作成
config = SMCConfig(
    n_particles=1000,  # 粒子数
    seed=42,           # 乱数シード
)

# エージェントを作成してフィット
agent = LocalLevelAgent(config)
agent.fit(data)

# 予測を生成
forecast = agent.predict(horizon=5)

print("予測平均:", forecast.mean)
print("予測標準偏差:", forecast.std)
print("95%信頼区間:", forecast.quantiles[0.05], "〜", forecast.quantiles[0.95])
```

### 2. DataFrameとの連携

pandas DataFrameから直接データを読み込めます。

```python
import pandas as pd
from smcs import LocalLevelAgent, SMCConfig
from smcs.io import from_dataframe, forecast_to_dataframe

# CSVからデータを読み込み
df = pd.read_csv("timeseries.csv", index_col=0, parse_dates=True)

# JAX配列に変換（タイムスタンプも取得）
data, timestamps = from_dataframe(df, columns=["value"])

# フィットと予測
config = SMCConfig(n_particles=1000)
agent = LocalLevelAgent(config)
agent.fit(data, timestamps)

forecast = agent.predict(horizon=10)

# 結果をDataFrameに変換
result_df = forecast_to_dataframe(forecast, column_names=["value"])
print(result_df)
```

### 3. オンライン更新

新しい観測が得られたら、モデルをオンラインで更新できます。

```python
# フィット済みのエージェントに新しい観測を追加
new_observation = jnp.array([1.7])
agent.update(new_observation)

# 更新後に再度予測
new_forecast = agent.predict(horizon=5)
```

### 4. 低レベルAPIを使う

より細かい制御が必要な場合は、低レベル関数を直接使えます。

```python
import jax
import jax.numpy as jnp
from smcs import (
    run_bootstrap_filter,
    LocalLevelModel,
    LocalLevelParams,
)

# モデルとパラメータを定義
model = LocalLevelModel()
params = LocalLevelParams(
    sigma_obs=0.5,     # 観測ノイズの標準偏差
    sigma_level=0.1,   # レベル変動の標準偏差
    m0=0.0,            # 初期状態の平均
    C0=1.0,            # 初期状態の分散
)

# データ
observations = jnp.array([[1.0], [1.2], [0.9], [1.1], [1.3]])

# 粒子フィルタを実行
key = jax.random.PRNGKey(42)
state, info = run_bootstrap_filter(
    key,
    observations,
    model,
    params,
    n_particles=1000,
)

print(f"対数尤度: {state.log_likelihood:.4f}")
print(f"最終ESS: {info.ess[-1]:.1f}")
print(f"フィルタ平均: {state.weighted_mean()}")
```

## 利用可能なモデル

| モデル | 説明 | Agent クラス |
|--------|------|-------------|
| Local Level | ランダムウォーク＋ノイズ | `LocalLevelAgent` |
| Local Linear Trend | レベル＋トレンド | `LocalLinearTrendAgent` |
| ARIMA | 自己回帰和分移動平均 | `ARIMAAgent` |
| GARCH | 条件付き分散変動 | `GARCHAgent` |
| Stochastic Volatility | 確率的ボラティリティ | `SVAgent` |

## 設定オプション

`SMCConfig`で設定できる主なオプション：

```python
from smcs import SMCConfig

config = SMCConfig(
    n_particles=1000,           # 粒子数（多いほど精度↑、計算量↑）
    seed=42,                    # 乱数シード
    ess_threshold=0.5,          # リサンプリング閾値（0〜1）
    resampling_method="systematic",  # リサンプリング手法
    jit_compile=True,           # JITコンパイル有効化
)
```

リサンプリング手法の選択肢：
- `"systematic"`: 系統的リサンプリング（推奨、分散最小）
- `"multinomial"`: 多項リサンプリング（単純だが分散大）
- `"stratified"`: 層化リサンプリング
- `"residual"`: 残差リサンプリング

## 次のステップ

- [API リファレンス](api_reference.md): 全関数・クラスの詳細
- [サンプル集](examples.md): より実践的な使用例
