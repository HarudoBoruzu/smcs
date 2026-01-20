# サンプル集

## 1. 基本的な時系列予測

### ローカルレベルモデルによる予測

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from smcs import LocalLevelAgent, SMCConfig

# 合成データを生成
key = jax.random.PRNGKey(42)
n = 100
true_level = jnp.cumsum(0.1 * jax.random.normal(key, (n,)))
observations = true_level + 0.5 * jax.random.normal(jax.random.PRNGKey(0), (n,))
observations = observations[:, None]  # [n, 1]の形に

# モデルをフィット
config = SMCConfig(n_particles=2000, seed=42)
agent = LocalLevelAgent(config)
agent.fit(observations)

# 予測
horizon = 20
forecast = agent.predict(horizon=horizon)

# プロット
plt.figure(figsize=(12, 6))
plt.plot(range(n), observations, 'b-', label='観測値')
plt.plot(range(n, n + horizon), forecast.mean, 'r-', label='予測')
plt.fill_between(
    range(n, n + horizon),
    forecast.quantiles[0.05].flatten(),
    forecast.quantiles[0.95].flatten(),
    alpha=0.3,
    color='red',
    label='90% 信頼区間'
)
plt.legend()
plt.xlabel('時間')
plt.ylabel('値')
plt.title('ローカルレベルモデルによる予測')
plt.show()
```

## 2. トレンドのある時系列

### ローカルリニアトレンドモデル

```python
from smcs import LocalLinearTrendAgent, SMCConfig

# トレンドのあるデータ
key = jax.random.PRNGKey(42)
n = 100
t = jnp.arange(n)
trend = 0.5 + 0.02 * t + 0.1 * jnp.cumsum(jax.random.normal(key, (n,)))
observations = trend + 0.3 * jax.random.normal(jax.random.PRNGKey(0), (n,))
observations = observations[:, None]

# フィットと予測
config = SMCConfig(n_particles=2000)
agent = LocalLinearTrendAgent(config)
agent.fit(observations)
forecast = agent.predict(horizon=30)

print("予測されたトレンド（10ステップ先）:", forecast.mean[:10])
```

## 3. ARIMA モデル

### AR(1) による予測

```python
from smcs import ARIMAAgent, SMCConfig

# AR(1) データ: y_t = 0.8 * y_{t-1} + epsilon_t
key = jax.random.PRNGKey(42)
n = 200
phi = 0.8
y = [0.0]
for i in range(n - 1):
    y.append(phi * y[-1] + 0.5 * float(jax.random.normal(jax.random.fold_in(key, i))))
observations = jnp.array(y)[:, None]

# AR(1) モデルでフィット
config = SMCConfig(n_particles=1000)
agent = ARIMAAgent(config, order=(1, 0, 0))
agent.fit(observations)
forecast = agent.predict(horizon=20)

print("AR(1) 予測:")
print(forecast.mean[:5])
```

## 4. ボラティリティモデル

### 確率的ボラティリティモデル

```python
from smcs import SVAgent, SMCConfig

# 金融リターンのようなデータ
key = jax.random.PRNGKey(42)
n = 500

# SVモデルからシミュレート
mu, phi, sigma_eta = -1.0, 0.95, 0.2
h = [mu]
for i in range(1, n):
    h.append(mu + phi * (h[-1] - mu) + sigma_eta * float(jax.random.normal(jax.random.fold_in(key, i))))
h = jnp.array(h)
returns = jnp.exp(h / 2) * jax.random.normal(jax.random.PRNGKey(0), (n,))
observations = returns[:, None]

# SVモデルでフィット
config = SMCConfig(n_particles=2000)
agent = SVAgent(config)
agent.fit(observations)

# ボラティリティ予測
forecast = agent.predict(horizon=30)
print("予測ボラティリティ（標準偏差）:", forecast.std[:5])
```

## 5. オンライン学習

### 逐次更新による予測

```python
from smcs import LocalLevelAgent, SMCConfig

# 初期データでフィット
initial_data = jnp.array([[1.0], [1.1], [0.9], [1.2], [1.0]])
config = SMCConfig(n_particles=1000)
agent = LocalLevelAgent(config)
agent.fit(initial_data)

# 新しい観測が到着するたびに更新
new_observations = [1.3, 1.1, 1.4, 1.2, 1.5]
predictions = []

for obs in new_observations:
    # 予測を保存
    forecast = agent.predict(horizon=1)
    predictions.append(float(forecast.mean[0, 0]))

    # モデルを更新
    agent.update(jnp.array([obs]))

print("1ステップ先予測:", predictions)
print("実際の値:", new_observations)
```

## 6. パラメータ学習

### Liu-West フィルタによるオンラインパラメータ推定

```python
import jax
import jax.numpy as jnp
from smcs import run_liu_west_filter, LocalLevelModel, LocalLevelParams

# モデル
model = LocalLevelModel()

# パラメータベクトル [log(sigma_obs), log(sigma_level)] → ModelParams
def param_to_model_params(param_vec):
    return LocalLevelParams(
        sigma_obs=jnp.exp(param_vec[0]),
        sigma_level=jnp.exp(param_vec[1]),
        m0=0.0,
        C0=1.0,
    )

# 初期状態サンプラー
def initial_state_sampler(key, param_vec):
    params = param_to_model_params(param_vec)
    return jnp.array([params.m0 + jnp.sqrt(params.C0) * jax.random.normal(key)])

# 初期パラメータサンプラー（事前分布）
def initial_param_sampler(key):
    # log(sigma) ~ N(-1, 0.5)
    return -1.0 + 0.5 * jax.random.normal(key, (2,))

# データ
key = jax.random.PRNGKey(42)
true_sigma_obs, true_sigma_level = 0.5, 0.1
n = 200
level = jnp.cumsum(true_sigma_level * jax.random.normal(key, (n,)))
observations = level + true_sigma_obs * jax.random.normal(jax.random.PRNGKey(0), (n,))
observations = observations[:, None]

# Liu-West フィルタを実行
state, info = run_liu_west_filter(
    jax.random.PRNGKey(123),
    observations,
    model,
    param_to_model_params,
    initial_state_sampler,
    initial_param_sampler,
    n_particles=500,
    delta=0.98,
)

# 推定パラメータ
estimated = state.weighted_param_mean()
print(f"真の sigma_obs: {true_sigma_obs:.3f}, 推定: {jnp.exp(estimated[0]):.3f}")
print(f"真の sigma_level: {true_sigma_level:.3f}, 推定: {jnp.exp(estimated[1]):.3f}")
```

## 7. バッチパラメータ推定

### PMMH による事後分布推定

```python
import jax
import jax.numpy as jnp
from smcs import run_pmmh, random_walk_proposal, LocalLevelModel, LocalLevelParams

model = LocalLevelModel()

def param_to_model_params(param_vec):
    return LocalLevelParams(
        sigma_obs=jnp.exp(param_vec[0]),
        sigma_level=jnp.exp(param_vec[1]),
        m0=0.0,
        C0=1.0,
    )

# 対数事前分布
def log_prior(param_vec):
    # log(sigma) ~ N(-1, 1)
    return -0.5 * jnp.sum((param_vec + 1) ** 2)

# データ（上の例と同じ）
observations = ...  # 適切に設定

# PMMH を実行
proposal = random_walk_proposal(scale=0.1)
result = run_pmmh(
    jax.random.PRNGKey(42),
    observations,
    model,
    param_to_model_params,
    log_prior,
    proposal,
    initial_params=jnp.array([-1.0, -2.0]),
    n_samples=2000,
    n_burnin=500,
    n_particles=100,
)

print(f"受容率: {result.acceptance_rate:.2%}")
print(f"sigma_obs の事後平均: {jnp.exp(result.samples[:, 0]).mean():.3f}")
print(f"sigma_level の事後平均: {jnp.exp(result.samples[:, 1]).mean():.3f}")
```

## 8. カスタムモデルの定義

### 独自の状態空間モデル

```python
from smcs import StateSpaceModel, ModelParams, Distribution, Normal
from smcs import run_bootstrap_filter
import chex

@chex.dataclass(frozen=True)
class MyModelParams(ModelParams):
    """カスタムモデルのパラメータ"""
    alpha: float
    sigma_state: float
    sigma_obs: float

class MyModel(StateSpaceModel):
    """カスタム状態空間モデル: 非線形遷移"""

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params):
        return Normal(loc=0.0, scale=1.0)

    def transition_distribution(self, params, state, t=None):
        # 非線形遷移: x_t = alpha * x_{t-1} / (1 + x_{t-1}^2) + noise
        x = state[0]
        mean = params.alpha * x / (1 + x ** 2)
        return Normal(loc=mean, scale=params.sigma_state)

    def emission_distribution(self, params, state, t=None):
        return Normal(loc=state[0], scale=params.sigma_obs)

# 使用例
model = MyModel()
params = MyModelParams(alpha=0.9, sigma_state=0.1, sigma_obs=0.5)

state, info = run_bootstrap_filter(
    jax.random.PRNGKey(42),
    observations,
    model,
    params,
    n_particles=1000,
)
```

## 9. 複数時系列の処理

### vmapによる並列処理

```python
import jax
from jax import vmap
from smcs import run_bootstrap_filter, LocalLevelModel, LocalLevelParams

model = LocalLevelModel()
params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

# 複数の時系列
n_series = 10
n_timesteps = 100
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, n_series)
all_observations = jax.random.normal(key, (n_series, n_timesteps, 1))

# 各時系列に対してフィルタを実行
def filter_one(key, obs):
    state, info = run_bootstrap_filter(key, obs, model, params, n_particles=500)
    return state.log_likelihood

# vmap で並列化
log_likelihoods = vmap(filter_one)(keys, all_observations)
print("各時系列の対数尤度:", log_likelihoods)
```

## 10. 診断とデバッグ

### ESS と対数尤度の監視

```python
from smcs import run_bootstrap_filter, LocalLevelModel, LocalLevelParams

model = LocalLevelModel()
params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

state, info = run_bootstrap_filter(
    jax.random.PRNGKey(42),
    observations,
    model,
    params,
    n_particles=1000,
)

# ESS の推移を確認
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(info.ess)
plt.axhline(y=500, color='r', linestyle='--', label='ESS閾値 (50%)')
plt.xlabel('時間ステップ')
plt.ylabel('ESS')
plt.title('Effective Sample Size の推移')
plt.legend()

# リサンプリング回数
n_resampled = info.resampled.sum()
print(f"リサンプリング回数: {n_resampled} / {len(info.resampled)}")
print(f"最終対数尤度: {state.log_likelihood:.4f}")
```
