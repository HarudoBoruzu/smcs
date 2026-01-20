"""State space models for SMC.

This module provides various state space model implementations:
- Dynamic Linear Models (DLM)
- ARIMA/SARIMA
- Stochastic Volatility (SV)
- GARCH family
- Dynamic Factor Models
- Regime-switching models
"""

from smcs.models.arima import ARIMAModel, ARIMAParams, partial_autocorr_to_ar
from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import (
    Categorical,
    Distribution,
    MultivariateNormal,
    Normal,
    Uniform,
)
from smcs.models.dlm import (
    DLM,
    DLMParams,
    LocalLevelModel,
    LocalLevelParams,
    LocalLinearTrendModel,
    LocalLinearTrendParams,
)
from smcs.models.factor import DynamicFactorModel, DynamicFactorParams, estimate_factor_loadings
from smcs.models.garch import GARCHModel, GARCHParams, GJRGARCHModel, GJRGARCHParams
from smcs.models.regime import (
    MarkovSwitchingModel,
    MarkovSwitchingParams,
    MSLocalLevelModel,
    MSLocalLevelParams,
)
from smcs.models.sv import SVLeverageModel, SVLeverageParams, SVModel, SVParams

__all__ = [
    # Base
    "StateSpaceModel",
    "ModelParams",
    # Distributions
    "Distribution",
    "Normal",
    "MultivariateNormal",
    "Uniform",
    "Categorical",
    # DLM
    "DLM",
    "DLMParams",
    "LocalLevelModel",
    "LocalLevelParams",
    "LocalLinearTrendModel",
    "LocalLinearTrendParams",
    # ARIMA
    "ARIMAModel",
    "ARIMAParams",
    "partial_autocorr_to_ar",
    # Stochastic Volatility
    "SVModel",
    "SVParams",
    "SVLeverageModel",
    "SVLeverageParams",
    # GARCH
    "GARCHModel",
    "GARCHParams",
    "GJRGARCHModel",
    "GJRGARCHParams",
    # Factor Models
    "DynamicFactorModel",
    "DynamicFactorParams",
    "estimate_factor_loadings",
    # Regime Switching
    "MarkovSwitchingModel",
    "MarkovSwitchingParams",
    "MSLocalLevelModel",
    "MSLocalLevelParams",
]
