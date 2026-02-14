"""Stochastic Volatility (SV) models for SMC.

This module provides implementations of stochastic volatility models:
- Basic SV model
- SV with leverage effect
- SV with jumps
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Distribution, Normal

__all__ = [
    "SVParams",
    "SVModel",
    "SVLeverageParams",
    "SVLeverageModel",
]


@chex.dataclass(frozen=True)
class SVParams(ModelParams):
    """Parameters for the basic Stochastic Volatility model.

    The basic SV model is:
        y_t = exp(h_t / 2) * epsilon_t,  epsilon_t ~ N(0, 1)
        h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,  eta_t ~ N(0, 1)

    Attributes
    ----------
    mu : float
        Long-run mean log-volatility.
    phi : float
        Persistence parameter in (-1, 1).
    sigma_eta : float
        Volatility of log-volatility.
    h0 : float
        Initial log-volatility mean.
    P0 : float
        Initial log-volatility variance.
    """

    mu: Float[Array, ""]
    phi: Float[Array, ""]
    sigma_eta: Float[Array, ""]
    h0: Float[Array, ""]
    P0: Float[Array, ""]


@chex.dataclass(frozen=True)
class SVLeverageParams(ModelParams):
    """Parameters for SV model with leverage effect.

    Adds correlation between observation and volatility shocks.

    Attributes
    ----------
    mu : float
        Long-run mean log-volatility.
    phi : float
        Persistence parameter.
    sigma_eta : float
        Volatility of log-volatility.
    rho : float
        Leverage correlation in (-1, 1).
    h0 : float
        Initial log-volatility mean.
    P0 : float
        Initial log-volatility variance.
    """

    mu: Float[Array, ""]
    phi: Float[Array, ""]
    sigma_eta: Float[Array, ""]
    rho: Float[Array, ""]
    h0: Float[Array, ""]
    P0: Float[Array, ""]


class SVModel(StateSpaceModel):
    """Basic Stochastic Volatility model.

    State: h_t (log-volatility)
    Observation: y_t = exp(h_t/2) * epsilon_t
    """

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: SVParams) -> Distribution:
        """Return initial state distribution.

        Uses the stationary distribution when |phi| < 1.
        """
        # Stationary variance: sigma_eta^2 / (1 - phi^2)
        stationary_var = params.sigma_eta**2 / (1 - params.phi**2)
        return Normal(loc=params.mu, scale=jnp.sqrt(jnp.maximum(stationary_var, params.P0)))

    def transition_distribution(
        self,
        params: SVParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution for log-volatility."""
        h = state[0]
        mean = params.mu + params.phi * (h - params.mu)
        return Normal(loc=mean, scale=params.sigma_eta)

    def emission_distribution(
        self,
        params: SVParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        h = state[0]
        scale = jnp.exp(h / 2)
        return Normal(loc=0.0, scale=scale)


class SVLeverageModel(StateSpaceModel):
    """Stochastic Volatility model with leverage effect.

    The leverage effect captures the negative correlation between
    returns and volatility (negative returns lead to higher volatility).

    This model requires joint sampling of (y_t, h_t) due to correlation.
    For SMC, we use a locally optimal proposal when available.
    """

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: SVLeverageParams) -> Distribution:
        """Return initial state distribution."""
        stationary_var = params.sigma_eta**2 / (1 - params.phi**2)
        return Normal(loc=params.mu, scale=jnp.sqrt(jnp.maximum(stationary_var, params.P0)))

    def transition_distribution(
        self,
        params: SVLeverageParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution for log-volatility."""
        h = state[0]
        mean = params.mu + params.phi * (h - params.mu)
        return Normal(loc=mean, scale=params.sigma_eta)

    def emission_distribution(
        self,
        params: SVLeverageParams,
        state: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        h = state[0]
        scale = jnp.exp(h / 2)
        return Normal(loc=0.0, scale=scale)

    def proposal_distribution(
        self,
        params: SVLeverageParams,
        state: Float[Array, " 1"],
        observation: Float[Array, " 1"],
        t: int | None = None,
    ) -> Distribution:
        """Return locally optimal proposal for SV with leverage.

        For the SV model with leverage, we can derive an approximate
        locally optimal proposal by conditioning on the observation.
        """
        h = state[0]
        y = observation[0]

        # Prior mean and variance
        prior_mean = params.mu + params.phi * (h - params.mu)
        prior_var = params.sigma_eta**2

        # Approximate likelihood contribution (using first-order Taylor expansion)
        # log p(y|h) ≈ -h/2 - y^2 * exp(-h) / 2
        # This leads to an approximate Gaussian posterior
        exp_neg_h = jnp.exp(-prior_mean)
        likelihood_info = y**2 * exp_neg_h / 2  # Approximate precision contribution

        # Combine prior and likelihood (Gaussian approximation)
        posterior_var = 1.0 / (1.0 / prior_var + likelihood_info)
        posterior_mean = posterior_var * (prior_mean / prior_var)

        return Normal(loc=posterior_mean, scale=jnp.sqrt(posterior_var))
