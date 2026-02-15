# smcs

[![CI](https://github.com/HarudoBoruzus/smcs/actions/workflows/ci.yml/badge.svg)](https://github.com/HarudoBoruzus/smcs/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/smcs.svg)](https://badge.fury.io/py/smcs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**smcs** (Sequential Monte Carlo Samplers) is a comprehensive JAX-based Monte Carlo library. It provides:

- **Sequential Monte Carlo (SMC)** algorithms for state-space models and time series
- **Bootstrap resampling** methods for statistical inference
- **MCMC samplers** (Metropolis-Hastings, HMC, NUTS, Gibbs, Slice)
- **Importance sampling** methods
- **Quasi-Monte Carlo** sequences for low-discrepancy sampling

Designed with the academic rigor of [particles](https://github.com/nchopin/particles), the functional patterns of [BlackJAX](https://github.com/blackjax-devs/blackjax), and state-space abstractions of [Dynamax](https://github.com/probml/dynamax).

## Features

- **JAX-native**: Full JIT compilation and GPU/TPU support
- **Type-safe**: Comprehensive jaxtyping annotations with runtime checking
- **159 tests** ensuring correctness across all methods

### Sequential Monte Carlo

- Bootstrap Particle Filter
- Auxiliary Particle Filter
- Liu-West Filter (online parameter learning)
- Storvik Filter, SMC², PMMH, Waste-Free SMC
- Guided, Regularized, Unscented Particle Filters
- Ensemble Kalman Filter (EnKF)
- Particle Gibbs, PGAS, ABC-SMC
- Smoothing algorithms (FFBS, Two-Filter)

### Monte Carlo Methods

- **Bootstrap**: Ordinary, Block, Moving Block, Circular, Stationary, Wild, Residual, Parametric
- **MCMC**: Slice Sampling, Metropolis-Hastings, Random Walk MH, Gibbs, HMC, NUTS
- **Importance Sampling**: Basic IS, Self-Normalized, Multiple IS, Adaptive IS
- **Quasi-Monte Carlo**: Halton, Sobol, Latin Hypercube (LHS)

### State-Space Models

- Local Level, Local Linear Trend, ARIMA
- Stochastic Volatility, GARCH
- Dynamic Factor Models, Regime-Switching

## Installation

```bash
pip install smcs
```

For development:

```bash
pip install smcs[dev]
```

## Quick Start

### Bootstrap Resampling

```python
import jax
import jax.numpy as jnp
from smcs import ordinary_bootstrap, bootstrap_ci

# Sample data
data = jnp.array([1.2, 2.3, 1.8, 2.1, 1.9, 2.5, 2.0])

# Generate bootstrap samples
key = jax.random.PRNGKey(42)
bootstrap_samples = ordinary_bootstrap(key, data, n_bootstrap=1000)

# Compute bootstrap means
bootstrap_means = jnp.mean(bootstrap_samples, axis=1)

# Get 95% confidence interval
lower, upper = bootstrap_ci(bootstrap_means, confidence=0.95)
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

### Block Bootstrap for Time Series

```python
from smcs import block_bootstrap, stationary_bootstrap

# Time series data with autocorrelation
time_series = jnp.array([...])

# Block bootstrap (fixed block size)
samples = block_bootstrap(key, time_series, block_size=10, n_bootstrap=500)

# Stationary bootstrap (random block sizes)
samples = stationary_bootstrap(key, time_series, mean_block_size=10.0, n_bootstrap=500)
```

### MCMC Sampling

```python
from smcs import run_hmc, run_nuts, run_slice_sampler

# Define log probability (e.g., Gaussian)
def log_prob(x):
    return -0.5 * jnp.sum(x ** 2)

# HMC sampling
initial = jnp.array([5.0, 5.0])
samples, acceptance_rate = run_hmc(
    key, initial, log_prob,
    step_size=0.1, n_leapfrog=20,
    n_samples=1000, n_burnin=200
)

# NUTS sampling (auto-tuned trajectory length)
samples = run_nuts(key, initial, log_prob, step_size=0.1, n_samples=1000)

# Slice sampling
samples = run_slice_sampler(key, initial, log_prob, n_samples=1000)
```

### Importance Sampling

```python
from smcs import importance_sample, self_normalized_is, compute_ess_is

def proposal_sample(k):
    return jax.random.normal(k, shape=(2,))

def log_target(x):
    return -0.5 * jnp.sum((x - 2.0) ** 2)  # N(2, I)

def log_proposal(x):
    return -0.5 * jnp.sum(x ** 2)  # N(0, I)

# Self-normalized importance sampling
samples, weights = self_normalized_is(
    key, proposal_sample, log_target, log_proposal, n_samples=5000
)

# Estimate mean
estimated_mean = jnp.sum(weights[:, None] * samples, axis=0)

# Check effective sample size
ess = compute_ess_is(jnp.log(weights))
```

### Quasi-Monte Carlo

```python
from smcs import halton_sequence, sobol_sequence, latin_hypercube

# Low-discrepancy sequences for numerical integration
halton_pts = halton_sequence(n_samples=1000, dim=3)
sobol_pts = sobol_sequence(n_samples=1000, dim=3)

# Latin Hypercube Sampling (stratified)
lhs_pts = latin_hypercube(key, n_samples=100, dim=5)

# Integrate f(x) = sin(pi*x) over [0,1]
integral = jnp.mean(jnp.sin(jnp.pi * halton_pts[:, 0]))
```

### Particle Filtering

```python
from smcs import run_bootstrap_filter, LocalLevelModel, LocalLevelParams

# Define model
model = LocalLevelModel()
params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

# Run particle filter
state, info = run_bootstrap_filter(
    key, observations, model, params, n_particles=1000
)

print(f"Log-likelihood: {state.log_likelihood:.4f}")
print(f"Final ESS: {info.ess[-1]:.1f}")
```

### High-Level Forecasting Agents

```python
from smcs import LocalLevelAgent, SMCConfig
from smcs.io import from_dataframe

# Load data
data, timestamps = from_dataframe(df)

# Configure and fit
config = SMCConfig(n_particles=1000, seed=42)
agent = LocalLevelAgent(config)
agent.fit(data, timestamps)

# Forecast
forecast = agent.predict(horizon=10)
print(f"Forecast: {forecast.mean}")
```

## API Reference

### Bootstrap Methods

| Function | Description |
|----------|-------------|
| `ordinary_bootstrap` | Standard i.i.d. bootstrap |
| `block_bootstrap` | Non-overlapping block bootstrap |
| `moving_block_bootstrap` | Overlapping block bootstrap |
| `circular_block_bootstrap` | Circular block bootstrap |
| `stationary_bootstrap` | Random block lengths (geometric) |
| `wild_bootstrap` | For heteroscedastic data |
| `residual_bootstrap` | Resample regression residuals |
| `parametric_bootstrap` | Sample from fitted distribution |
| `bootstrap_ci` | Confidence intervals (percentile, basic, BCa) |
| `jackknife` | Leave-one-out resampling |

### MCMC Methods

| Function | Description |
|----------|-------------|
| `slice_sample` / `run_slice_sampler` | Slice sampling |
| `metropolis_hastings` / `run_metropolis_hastings` | General M-H |
| `random_walk_metropolis` / `run_random_walk_metropolis` | Random walk M-H |
| `gibbs_sample` / `run_gibbs_sampler` | Gibbs sampling |
| `hmc_step` / `run_hmc` | Hamiltonian Monte Carlo |
| `nuts_step` / `run_nuts` | No-U-Turn Sampler |

### Importance Sampling

| Function | Description |
|----------|-------------|
| `importance_sample` | Basic importance sampling |
| `self_normalized_is` | Self-normalized IS |
| `multiple_importance_sampling` | Multiple proposal MIS |
| `adaptive_importance_sampling` | Adaptive Gaussian proposal |
| `compute_ess_is` | Effective sample size |
| `compute_is_diagnostics` | Full diagnostics |

### Quasi-Monte Carlo

| Function | Description |
|----------|-------------|
| `halton_sequence` | Halton low-discrepancy sequence |
| `sobol_sequence` | Sobol sequence |
| `latin_hypercube` | Latin Hypercube Sampling |
| `randomized_halton` | Randomized Halton |
| `randomized_lhs` | Randomized LHS |

### SMC Algorithms

| Algorithm | Use Case |
|-----------|----------|
| Bootstrap PF | Basic filtering |
| Auxiliary PF | Informative observations |
| Liu-West | Online parameter learning |
| Storvik | Sufficient statistics models |
| SMC² | Full parameter learning |
| PMMH | Batch parameter estimation |

## Architecture

```
smcs/
├── src/smcs/
│   ├── core/           # Resampling, ESS, weights
│   ├── algorithms/     # SMC algorithms (20+)
│   ├── models/         # State-space models
│   ├── montecarlo/     # Bootstrap, MCMC, IS, QMC
│   ├── agents/         # High-level forecasting
│   └── io/             # DataFrame utilities
└── tests/              # 159 tests
```

## Development

```bash
# Clone and install
git clone https://github.com/smcs-authors/smcs.git
cd smcs
pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check src tests
```

## References

1. Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation.
2. Neal, R. M. (2003). Slice sampling. *Annals of Statistics*.
3. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler.
4. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*.
5. Chopin, N., & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*.

## License

MIT License - see [LICENSE](LICENSE) for details.
