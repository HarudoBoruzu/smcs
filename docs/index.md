# smcs ドキュメント

**smcs** は、JAXベースのSequential Monte Carlo（逐次モンテカルロ）ライブラリです。
時系列予測のための状態空間モデルとSMCアルゴリズムを提供します。

## 特徴

- **JAXネイティブ**: JITコンパイル、GPU/TPU対応
- **型安全**: jaxtypingによる包括的な型アノテーション
- **豊富なアルゴリズム**: Bootstrap PF, APF, Liu-West, SMC², PMMH など
- **多様なモデル**: DLM, ARIMA, SV, GARCH, Factor, Regime-switching
- **使いやすいAPI**: 高レベルAgentクラスと低レベル関数APIの両方を提供

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [Getting Started](getting_started.md) | インストールと基本的な使い方 |
| [API Reference](api_reference.md) | 全関数・クラスの詳細リファレンス |
| [Examples](examples.md) | 実践的なサンプルコード集 |

## クイックスタート

```python
from smcs import LocalLevelAgent, SMCConfig
import jax.numpy as jnp

# データ
data = jnp.array([[1.0], [1.2], [0.9], [1.1], [1.3]])

# 設定とエージェント
config = SMCConfig(n_particles=1000)
agent = LocalLevelAgent(config)

# フィットと予測
agent.fit(data)
forecast = agent.predict(horizon=5)

print(forecast.mean)
```

## アルゴリズム一覧

| アルゴリズム | 用途 | 計算量 |
|------------|------|--------|
| Bootstrap PF | 基本的なフィルタリング | O(NT) |
| Auxiliary PF | 情報量の多い観測への対応 | O(NT) |
| Liu-West | オンラインパラメータ学習 | O(NT) |
| Storvik | 十分統計量を持つモデル | O(NT) |
| SMC² | 完全なオンラインパラメータ学習 | O(Nθ×Nx×T) |
| PMMH | バッチパラメータ学習 | O(N×MCMC) |
| Waste-Free | 効率的なMCMC利用 | O(NT) |

## モデル一覧

| モデル | 説明 |
|--------|------|
| Local Level | ランダムウォーク＋ノイズ |
| Local Linear Trend | レベル＋スロープ |
| ARIMA | 自己回帰和分移動平均 |
| Stochastic Volatility | 確率的ボラティリティ |
| GARCH | 条件付き異分散 |
| Dynamic Factor | 多変量因子モデル |
| Regime-Switching | マルコフ切り替えモデル |

## ライセンス

MIT License
