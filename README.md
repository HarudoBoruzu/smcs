# smcs

[![CI](https://github.com/HarudoBoruzu/smcs/actions/workflows/ci.yml/badge.svg)](https://github.com/HarudoBoruzu/smcs/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/smcs.svg)](https://badge.fury.io/py/smcs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**smcs** is a JAX-based Sequential Monte Carlo library for time series prediction. It combines the academic rigor of [particles](https://github.com/nchopin/particles), the functional design patterns of [BlackJAX](https://github.com/blackjax-devs/blackjax), and the state-space model abstractions of [Dynamax](https://github.com/probml/dynamax).

## Features

- **JAX-native**: Full JIT compilation and GPU/TPU support
- **Type-safe**: Comprehensive jaxtyping annotations with runtime checking
- **Multiple SMC algorithms**:
  - Bootstrap Particle Filter
  - Auxiliary Particle Filter
  - Liu-West Filter (online parameter learning)
  - Storvik Filter (sufficient statistics)
  - SMC² (nested SMC for parameters)
  - PMMH (Particle MCMC)
  - Waste-Free SMC
- **State-space models**:
  - Dynamic Linear Models (Local Level, Local Linear Trend)
  - ARIMA/SARIMA
  - Stochastic Volatility
  - GARCH family
  - Dynamic Factor Models
  - Regime-switching models
- **High-level forecasting agents** for easy use
- **Pandas DataFrame integration**

## Installation

```bash
pip install smcs
```

For development:

```bash
pip install smcs[dev]
```

## Quick Start

### Using High-Level Agents

```python
import jax.numpy as jnp
from smcs import LocalLevelAgent, SMCConfig
from smcs.io import from_dataframe

# Load data
data, timestamps = from_dataframe(df)

# Create and configure agent
config = SMCConfig(n_particles=1000, seed=42)
agent = LocalLevelAgent(config)

# Fit model
agent.fit(data, timestamps)

# Generate forecasts
forecast = agent.predict(horizon=10)
print(f"Forecast mean: {forecast.mean}")
print(f"95% interval: [{forecast.quantiles[0.05]}, {forecast.quantiles[0.95]}]")

# Online update with new observation
agent.update(jnp.array([new_value]))
```

### Using Low-Level API

```python
import jax
import jax.numpy as jnp
from smcs import (
    run_bootstrap_filter,
    LocalLevelModel,
    LocalLevelParams,
)

# Define model and parameters
model = LocalLevelModel()
params = LocalLevelParams(
    sigma_obs=0.5,
    sigma_level=0.1,
    m0=0.0,
    C0=1.0,
)

# Generate synthetic data
key = jax.random.PRNGKey(42)
observations = jax.random.normal(key, shape=(100, 1))

# Run particle filter
filter_key = jax.random.PRNGKey(123)
state, info = run_bootstrap_filter(
    filter_key,
    observations,
    model,
    params,
    n_particles=1000,
)

print(f"Log-likelihood: {state.log_likelihood:.4f}")
print(f"Final ESS: {info.ess[-1]:.1f}")
```

### Parameter Learning with Liu-West Filter

```python
from smcs import run_liu_west_filter, LocalLevelModel

model = LocalLevelModel()

# Define parameter conversion
def param_to_model_params(param_vec):
    return LocalLevelParams(
        sigma_obs=jnp.exp(param_vec[0]),
        sigma_level=jnp.exp(param_vec[1]),
        m0=0.0,
        C0=1.0,
    )

# Run Liu-West filter
state, info = run_liu_west_filter(
    key,
    observations,
    model,
    param_to_model_params,
    initial_state_sampler,
    initial_param_sampler,
    n_particles=1000,
    delta=0.98,
)

# Get estimated parameters
estimated_params = state.weighted_param_mean()
```

## Architecture

```
smcs/
├── src/smcs/
│   ├── core/           # Particles, resampling, ESS computation
│   ├── algorithms/     # SMC algorithm implementations
│   ├── models/         # State space model definitions
│   ├── agents/         # High-level forecasting agents
│   ├── config/         # Pydantic configuration
│   └── io/             # DataFrame utilities
├── tests/
└── docs/
```

## SMC Algorithms

| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| Bootstrap PF | Basic filtering | O(NT) |
| Auxiliary PF | Informative observations | O(NT) |
| Liu-West | Online parameter learning | O(NT) |
| Storvik | Models with sufficient statistics | O(NT) |
| SMC² | Full online parameter learning | O(Nθ×Nx×T) |
| PMMH | Batch parameter learning | O(N×MCMC) |
| Waste-Free | Efficient MCMC utilization | O(NT) |

## State Space Models

| Model | Description |
|-------|-------------|
| Local Level | Random walk + noise |
| Local Linear Trend | Level + slope dynamics |
| ARIMA | Autoregressive integrated moving average |
| Stochastic Volatility | Time-varying volatility |
| GARCH | Deterministic volatility dynamics |
| Dynamic Factor | Multivariate with latent factors |
| Regime-Switching | Markov-switching dynamics |

## Configuration

```python
from smcs import SMCConfig

config = SMCConfig(
    n_particles=1000,          # Number of particles
    seed=42,                   # Random seed
    ess_threshold=0.5,         # ESS/N ratio for resampling
    resampling_method="systematic",  # Resampling algorithm
    liu_west_delta=0.98,       # Liu-West discount factor
    jit_compile=True,          # JIT compilation
)
```

## Development

```bash
# Clone repository
git clone https://github.com/HarudoBoruzu/smcs.git
cd smcs

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src tests

# Type checking
mypy src
```

## References

1. Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*.
2. Pitt, M. K., & Shephard, N. (1999). Filtering via simulation: Auxiliary particle filters. *JASA*.
3. Liu, J., & West, M. (2001). Combined parameter and state estimation in simulation-based filtering.
4. Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC²: an efficient algorithm for sequential analysis of state space models. *JRSS-B*.
5. Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. *JRSS-B*.
6. Dau, H. D., & Chopin, N. (2022). Waste-free sequential Monte Carlo. *JRSS-B*.

## License

MIT License - see [LICENSE](LICENSE) for details.
