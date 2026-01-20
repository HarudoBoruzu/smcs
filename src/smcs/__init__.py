"""smcs: JAX-based Sequential Monte Carlo library for time series prediction.

This library provides a comprehensive set of tools for Sequential Monte Carlo
(SMC) methods applied to state-space models for time series analysis.

Key Features
------------
- JAX-based implementation with JIT compilation and GPU support
- Type-safe with jaxtyping annotations
- Multiple SMC algorithms (Bootstrap PF, APF, Liu-West, SMC², PMMH)
- Various state-space models (DLM, ARIMA, SV, GARCH)
- High-level forecasting agents for easy use
- Pandas DataFrame integration

Quick Start
-----------
>>> from smcs import LocalLevelAgent, SMCConfig
>>> from smcs.io import from_dataframe
>>>
>>> # Load data
>>> data, timestamps = from_dataframe(df)
>>>
>>> # Create and fit agent
>>> config = SMCConfig(n_particles=1000)
>>> agent = LocalLevelAgent(config)
>>> agent.fit(data, timestamps)
>>>
>>> # Generate forecasts
>>> forecast = agent.predict(horizon=10)
>>> print(forecast.mean)

Example with custom model:

>>> from smcs import run_bootstrap_filter
>>> from smcs.models import LocalLevelModel, LocalLevelParams
>>>
>>> model = LocalLevelModel()
>>> params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)
>>>
>>> state, info = run_bootstrap_filter(
...     key, observations, model, params, n_particles=1000
... )
>>> print(f"Log-likelihood: {state.log_likelihood:.4f}")
"""

from smcs._typing import (
    Ancestors,
    Array,
    Bool,
    CovarianceMatrix,
    Float,
    Int,
    LogWeights,
    Matrix,
    Observations,
    Particles,
    PRNGKeyArray,
    Scalar,
    SingleObservation,
    StateVector,
    Vector,
)
from smcs.agents import (
    ARIMAAgent,
    BaseAgent,
    ForecastingAgent,
    ForecastResult,
    GARCHAgent,
    LocalLevelAgent,
    LocalLinearTrendAgent,
    SVAgent,
)
from smcs.algorithms import (
    LiuWestState,
    PMMHResult,
    PMMHState,
    SMC2State,
    StorvikState,
    WasteFreeState,
    auxiliary_step,
    bootstrap_step,
    initialize_particles,
    liu_west_step,
    metropolis_hastings_kernel,
    pmmh_step,
    random_walk_proposal,
    run_auxiliary_filter,
    run_bootstrap_filter,
    run_liu_west_filter,
    run_pmmh,
    run_smc2,
    run_storvik_filter,
    run_waste_free_smc,
    smc2_step,
    storvik_step,
    waste_free_step,
)
from smcs.config import AgentConfig, SMCConfig
from smcs.core import (
    ResamplingMethod,
    SMCInfo,
    SMCState,
    compute_ess,
    log_mean_exp,
    multinomial_resample,
    normalize_log_weights,
    resample,
    residual_resample,
    stratified_resample,
    systematic_resample,
)
from smcs.io import forecast_to_dataframe, from_dataframe, to_dataframe
from smcs.models import (
    ARIMAModel,
    ARIMAParams,
    Categorical,
    Distribution,
    DLM,
    DLMParams,
    DynamicFactorModel,
    DynamicFactorParams,
    GARCHModel,
    GARCHParams,
    GJRGARCHModel,
    GJRGARCHParams,
    LocalLevelModel,
    LocalLevelParams,
    LocalLinearTrendModel,
    LocalLinearTrendParams,
    MarkovSwitchingModel,
    MarkovSwitchingParams,
    ModelParams,
    MSLocalLevelModel,
    MSLocalLevelParams,
    MultivariateNormal,
    Normal,
    StateSpaceModel,
    SVLeverageModel,
    SVLeverageParams,
    SVModel,
    SVParams,
    Uniform,
    estimate_factor_loadings,
    partial_autocorr_to_ar,
)

__version__ = "0.1.0"
__all__ = [
    # Version
    "__version__",
    # Types
    "Scalar",
    "Vector",
    "Matrix",
    "Particles",
    "LogWeights",
    "Ancestors",
    "Observations",
    "SingleObservation",
    "StateVector",
    "CovarianceMatrix",
    "Array",
    "Float",
    "Int",
    "Bool",
    "PRNGKeyArray",
    # Core
    "SMCState",
    "SMCInfo",
    "compute_ess",
    "normalize_log_weights",
    "log_mean_exp",
    "systematic_resample",
    "multinomial_resample",
    "stratified_resample",
    "residual_resample",
    "resample",
    "ResamplingMethod",
    # Algorithms
    "bootstrap_step",
    "run_bootstrap_filter",
    "initialize_particles",
    "auxiliary_step",
    "run_auxiliary_filter",
    "LiuWestState",
    "liu_west_step",
    "run_liu_west_filter",
    "StorvikState",
    "storvik_step",
    "run_storvik_filter",
    "SMC2State",
    "smc2_step",
    "run_smc2",
    "PMMHState",
    "PMMHResult",
    "pmmh_step",
    "run_pmmh",
    "random_walk_proposal",
    "WasteFreeState",
    "waste_free_step",
    "run_waste_free_smc",
    "metropolis_hastings_kernel",
    # Models
    "StateSpaceModel",
    "ModelParams",
    "Distribution",
    "Normal",
    "MultivariateNormal",
    "Uniform",
    "Categorical",
    "DLM",
    "DLMParams",
    "LocalLevelModel",
    "LocalLevelParams",
    "LocalLinearTrendModel",
    "LocalLinearTrendParams",
    "ARIMAModel",
    "ARIMAParams",
    "partial_autocorr_to_ar",
    "SVModel",
    "SVParams",
    "SVLeverageModel",
    "SVLeverageParams",
    "GARCHModel",
    "GARCHParams",
    "GJRGARCHModel",
    "GJRGARCHParams",
    "DynamicFactorModel",
    "DynamicFactorParams",
    "estimate_factor_loadings",
    "MarkovSwitchingModel",
    "MarkovSwitchingParams",
    "MSLocalLevelModel",
    "MSLocalLevelParams",
    # Agents
    "ForecastResult",
    "ForecastingAgent",
    "BaseAgent",
    "LocalLevelAgent",
    "LocalLinearTrendAgent",
    "ARIMAAgent",
    "GARCHAgent",
    "SVAgent",
    # Config
    "SMCConfig",
    "AgentConfig",
    # IO
    "from_dataframe",
    "to_dataframe",
    "forecast_to_dataframe",
]
