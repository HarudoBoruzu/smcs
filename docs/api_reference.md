# API リファレンス

## Core モジュール

### SMCState

粒子フィルタの状態を保持するイミュータブルなデータクラス。

```python
from smcs import SMCState

# 属性
state.particles      # 粒子 [n_particles, state_dim]
state.log_weights    # 対数重み [n_particles]
state.ancestors      # 祖先インデックス [n_particles]
state.log_likelihood # 累積対数尤度
state.step           # 現在のステップ

# プロパティ
state.n_particles    # 粒子数
state.state_dim      # 状態次元

# メソッド
state.normalized_weights()  # 正規化された重み
state.weighted_mean()       # 重み付き平均
state.weighted_cov()        # 重み付き共分散
```

### SMCInfo

SMCステップの診断情報。

```python
from smcs import SMCInfo

info.ess             # Effective Sample Size
info.resampled       # リサンプリング実行フラグ
info.acceptance_rate # MCMC受容率（該当する場合）
```

### リサンプリング関数

```python
from smcs import (
    systematic_resample,   # 系統的（推奨）
    multinomial_resample,  # 多項
    stratified_resample,   # 層化
    residual_resample,     # 残差
    resample,              # 統一インターフェース
)

# 使い方
indices = systematic_resample(key, log_weights)
indices = resample(key, log_weights, method="systematic")
```

### 重み計算

```python
from smcs import compute_ess, normalize_log_weights, log_mean_exp

ess = compute_ess(log_weights)  # ESS計算
normalized = normalize_log_weights(log_weights)  # 正規化
log_avg = log_mean_exp(log_values)  # log(mean(exp(x)))
```

---

## Algorithms モジュール

### Bootstrap Particle Filter

```python
from smcs import run_bootstrap_filter, bootstrap_step, initialize_particles

# 全観測系列に対して実行
state, info = run_bootstrap_filter(
    key,                  # JAX乱数キー
    observations,         # 観測 [n_timesteps, obs_dim]
    model,                # StateSpaceModel
    params,               # ModelParams
    n_particles=1000,     # 粒子数
    ess_threshold=0.5,    # リサンプリング閾値
    resampling_method="systematic",
)

# 1ステップのみ実行
new_state, info = bootstrap_step(
    key, state, observation, model, params, ess_threshold
)

# 粒子の初期化
state = initialize_particles(key, model, params, n_particles)
```

### Auxiliary Particle Filter

```python
from smcs import run_auxiliary_filter, auxiliary_step

state, info = run_auxiliary_filter(
    key, observations, model, params,
    n_particles=1000,
    ess_threshold=0.5,
)
```

### Liu-West Filter（パラメータ学習）

```python
from smcs import run_liu_west_filter, LiuWestState

state, info = run_liu_west_filter(
    key,
    observations,
    model,
    param_to_model_params,   # パラメータベクトル → ModelParams
    initial_state_sampler,   # (key, param) → state
    initial_param_sampler,   # (key,) → param
    n_particles=1000,
    delta=0.98,              # 縮小係数（0.95〜0.99推奨）
)

# 推定パラメータの取得
estimated_params = state.weighted_param_mean()
param_cov = state.weighted_param_cov()
```

### SMC² (オンラインパラメータ学習)

```python
from smcs import run_smc2, SMC2State

state, info = run_smc2(
    key,
    observations,
    model,
    param_to_model_params,
    initial_param_sampler,
    initial_state_sampler,
    n_theta_particles=100,   # パラメータ粒子数
    n_x_particles=100,       # 状態粒子数（各θごと）
)
```

### PMMH (Particle MCMC)

```python
from smcs import run_pmmh, random_walk_proposal, PMMHResult

# 提案分布を作成
proposal = random_walk_proposal(scale=0.1)

result: PMMHResult = run_pmmh(
    key,
    observations,
    model,
    param_to_model_params,
    log_prior_fn,           # パラメータの対数事前分布
    proposal,               # 提案関数
    initial_params,         # 初期パラメータ
    n_samples=5000,
    n_burnin=1000,
    n_particles=100,
)

print(f"受容率: {result.acceptance_rate:.2%}")
print(f"事後平均: {result.samples.mean(axis=0)}")
```

### Waste-Free SMC

```python
from smcs import run_waste_free_smc, metropolis_hastings_kernel

# MCMCカーネルを作成
kernel = metropolis_hastings_kernel(proposal_std=0.1)

state, info = run_waste_free_smc(
    key,
    initial_particles,
    target_sequence,  # [pi_0, pi_1, ..., pi_T]
    kernel,
    n_mcmc_steps=5,
)
```

---

## Models モジュール

### StateSpaceModel Protocol

すべてのモデルが実装するプロトコル：

```python
from smcs import StateSpaceModel

class MyModel(StateSpaceModel):
    @property
    def state_dim(self) -> int: ...

    @property
    def obs_dim(self) -> int: ...

    def initial_distribution(self, params) -> Distribution: ...
    def transition_distribution(self, params, state, t=None) -> Distribution: ...
    def emission_distribution(self, params, state, t=None) -> Distribution: ...

    # オプション: カスタム提案分布
    def proposal_distribution(self, params, state, obs, t=None) -> Distribution: ...
```

### Dynamic Linear Models

```python
from smcs import (
    LocalLevelModel, LocalLevelParams,
    LocalLinearTrendModel, LocalLinearTrendParams,
    DLM, DLMParams,
)

# Local Level（ランダムウォーク＋ノイズ）
model = LocalLevelModel()
params = LocalLevelParams(
    sigma_obs=0.5,    # 観測ノイズ
    sigma_level=0.1,  # レベル変動
    m0=0.0,           # 初期平均
    C0=1.0,           # 初期分散
)

# Local Linear Trend
model = LocalLinearTrendModel()
params = LocalLinearTrendParams(
    sigma_obs=0.5,
    sigma_level=0.1,
    sigma_slope=0.01,
    m0=jnp.array([0.0, 0.0]),
    C0=jnp.eye(2),
)
```

### ARIMA

```python
from smcs import ARIMAModel, ARIMAParams

model = ARIMAModel(order=(1, 0, 0))  # AR(1)
params = ARIMAParams(
    ar_coeffs=jnp.array([0.8]),
    ma_coeffs=jnp.array([]),
    sigma=0.5,
    d=0,
    m0=jnp.zeros(1),
    C0=jnp.eye(1),
)
```

### Stochastic Volatility

```python
from smcs import SVModel, SVParams

model = SVModel()
params = SVParams(
    mu=-1.0,          # 長期平均対数ボラティリティ
    phi=0.95,         # 持続性パラメータ
    sigma_eta=0.2,    # ボラティリティのボラティリティ
    h0=-1.0,          # 初期対数ボラティリティ
    P0=0.5,           # 初期分散
)
```

### GARCH

```python
from smcs import GARCHModel, GARCHParams

model = GARCHModel(order=(1, 1))
params = GARCHParams(
    omega=0.01,
    alpha=jnp.array([0.1]),
    beta=jnp.array([0.8]),
    sigma0=0.2,
)
```

---

## Agents モジュール

### ForecastResult

予測結果を格納するデータクラス：

```python
from smcs import ForecastResult

result.mean        # 点予測 [horizon, obs_dim]
result.std         # 標準偏差 [horizon, obs_dim]
result.quantiles   # 分位点 {q: array}
result.particles   # 粒子サンプル（オプション）
result.timestamps  # タイムスタンプ（オプション）
```

### Agent クラス

```python
from smcs import (
    LocalLevelAgent,
    LocalLinearTrendAgent,
    ARIMAAgent,
    GARCHAgent,
    SVAgent,
)

# 共通インターフェース
agent = LocalLevelAgent(config)
agent.fit(observations, timestamps=None)
forecast = agent.predict(horizon=10, n_samples=1000, quantiles=(0.05, 0.5, 0.95))
agent.update(new_observation)
state = agent.get_filter_state()
```

---

## Config モジュール

### SMCConfig

```python
from smcs import SMCConfig

config = SMCConfig(
    n_particles=1000,
    seed=42,
    ess_threshold=0.5,
    resampling_method="systematic",  # systematic/multinomial/stratified/residual
    liu_west_delta=0.98,
    n_mcmc_samples=5000,
    n_burnin=1000,
    jit_compile=True,
    use_checkpoint=False,
)
```

---

## IO モジュール

```python
from smcs.io import from_dataframe, to_dataframe, forecast_to_dataframe

# DataFrame → JAX array
data, timestamps = from_dataframe(df, columns=["value"], dropna=True)

# JAX array → DataFrame
df = to_dataframe(data, timestamps, column_names=["value"])

# ForecastResult → DataFrame
result_df = forecast_to_dataframe(forecast, column_names=["value"])
```
