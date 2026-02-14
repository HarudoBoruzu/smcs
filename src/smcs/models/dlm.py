"""Dynamic Linear Models (DLM) for SMC.

This module provides implementations of various DLMs:
- Local Level Model (random walk + noise)
- Local Linear Trend Model
- Seasonal Model
- General DLM with custom matrices
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Distribution, MultivariateNormal, Normal

__all__ = [
    "DLMParams",
    "LocalLevelParams",
    "LocalLinearTrendParams",
    "DLM",
    "LocalLevelModel",
    "LocalLinearTrendModel",
]


@chex.dataclass(frozen=True)
class DLMParams(ModelParams):
    """Parameters for a general Dynamic Linear Model.

    The DLM is defined as:
        y_t = F' @ theta_t + nu_t,  nu_t ~ N(0, V)
        theta_t = G @ theta_{t-1} + omega_t,  omega_t ~ N(0, W)

    Attributes
    ----------
    F : Array
        Observation matrix [state_dim, obs_dim].
    G : Array
        State transition matrix [state_dim, state_dim].
    V : Array
        Observation noise covariance [obs_dim, obs_dim].
    W : Array
        State noise covariance [state_dim, state_dim].
    m0 : Array
        Initial state mean [state_dim].
    C0 : Array
        Initial state covariance [state_dim, state_dim].
    """

    F: Float[Array, "state_dim obs_dim"]
    G: Float[Array, "state_dim state_dim"]
    V: Float[Array, "obs_dim obs_dim"]
    W: Float[Array, "state_dim state_dim"]
    m0: Float[Array, " state_dim"]
    C0: Float[Array, "state_dim state_dim"]


@chex.dataclass(frozen=True)
class LocalLevelParams(ModelParams):
    """Parameters for the Local Level Model.

    The local level model is:
        y_t = mu_t + nu_t,  nu_t ~ N(0, sigma_obs^2)
        mu_t = mu_{t-1} + omega_t,  omega_t ~ N(0, sigma_level^2)

    Attributes
    ----------
    sigma_obs : float
        Observation noise standard deviation.
    sigma_level : float
        Level innovation standard deviation.
    m0 : float
        Initial level mean.
    C0 : float
        Initial level variance.
    """

    sigma_obs: Float[Array, ""]
    sigma_level: Float[Array, ""]
    m0: Float[Array, ""]
    C0: Float[Array, ""]


@chex.dataclass(frozen=True)
class LocalLinearTrendParams(ModelParams):
    """Parameters for the Local Linear Trend Model.

    The local linear trend model is:
        y_t = mu_t + nu_t,  nu_t ~ N(0, sigma_obs^2)
        mu_t = mu_{t-1} + beta_{t-1} + omega_mu_t,  omega_mu_t ~ N(0, sigma_level^2)
        beta_t = beta_{t-1} + omega_beta_t,  omega_beta_t ~ N(0, sigma_slope^2)

    Attributes
    ----------
    sigma_obs : float
        Observation noise standard deviation.
    sigma_level : float
        Level innovation standard deviation.
    sigma_slope : float
        Slope innovation standard deviation.
    m0 : Array
        Initial state mean [level, slope].
    C0 : Array
        Initial state covariance [2, 2].
    """

    sigma_obs: Float[Array, ""]
    sigma_level: Float[Array, ""]
    sigma_slope: Float[Array, ""]
    m0: Float[Array, " 2"]
    C0: Float[Array, "2 2"]


class DLM(StateSpaceModel):
    """General Dynamic Linear Model.

    Implements the full DLM with arbitrary F, G, V, W matrices.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        """Initialize DLM.

        Parameters
        ----------
        state_dim : int
            Dimension of the state vector.
        obs_dim : int
            Dimension of the observation vector.
        """
        self._state_dim = state_dim
        self._obs_dim = obs_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def initial_distribution(self, params: DLMParams) -> Distribution:
        """Return initial state distribution."""
        return MultivariateNormal(loc=params.m0, covariance_matrix=params.C0)

    def transition_distribution(
        self,
        params: DLMParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution p(x_t | x_{t-1})."""
        mean = params.G @ state
        return MultivariateNormal(loc=mean, covariance_matrix=params.W)

    def emission_distribution(
        self,
        params: DLMParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution p(y_t | x_t)."""
        mean = params.F.T @ state
        return MultivariateNormal(loc=mean, covariance_matrix=params.V)


class LocalLevelModel(StateSpaceModel):
    """Local Level Model (Random Walk + Noise).

    The simplest DLM with 1D state and 1D observation.
    """

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: LocalLevelParams) -> Distribution:
        """Return initial state distribution."""
        return Normal(loc=params.m0, scale=jnp.sqrt(params.C0))

    def transition_distribution(
        self,
        params: LocalLevelParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution."""
        return Normal(loc=state[0], scale=params.sigma_level)

    def emission_distribution(
        self,
        params: LocalLevelParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        return Normal(loc=state[0], scale=params.sigma_obs)


class LocalLinearTrendModel(StateSpaceModel):
    """Local Linear Trend Model.

    State: [level, slope]
    Observation: level + noise
    """

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: LocalLinearTrendParams) -> Distribution:
        """Return initial state distribution."""
        return MultivariateNormal(loc=params.m0, covariance_matrix=params.C0)

    def transition_distribution(
        self,
        params: LocalLinearTrendParams,
        state: Float[Array, " 2"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution."""
        G = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        mean = G @ state
        W = jnp.diag(jnp.array([params.sigma_level**2, params.sigma_slope**2]))
        return MultivariateNormal(loc=mean, covariance_matrix=W)

    def emission_distribution(
        self,
        params: LocalLinearTrendParams,
        state: Float[Array, " 2"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        return Normal(loc=state[0], scale=params.sigma_obs)
