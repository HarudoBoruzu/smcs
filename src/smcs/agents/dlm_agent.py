"""DLM-based forecasting agents.

This module provides agents for Dynamic Linear Models including
Local Level and Local Linear Trend models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger

from smcs.agents.base import BaseAgent
from smcs.models.dlm import (
    LocalLevelModel,
    LocalLevelParams,
    LocalLinearTrendModel,
    LocalLinearTrendParams,
)

if TYPE_CHECKING:
    from smcs.config.settings import SMCConfig

__all__ = [
    "LocalLevelAgent",
    "LocalLinearTrendAgent",
]


class LocalLevelAgent(BaseAgent):
    """Agent for the Local Level Model (Random Walk + Noise).

    The local level model is:
        y_t = mu_t + nu_t,  nu_t ~ N(0, sigma_obs^2)
        mu_t = mu_{t-1} + omega_t,  omega_t ~ N(0, sigma_level^2)

    This is the simplest DLM and is suitable for data with no trend
    but with a slowly-changing level.
    """

    def __init__(self, config: SMCConfig):
        """Initialize Local Level Agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        """
        model = LocalLevelModel()
        super().__init__(config, model)

    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> LocalLevelParams:
        """Estimate parameters using method of moments.

        Uses the relationship:
        Var(Delta y_t) = 2*sigma_level^2 + sigma_obs^2

        Parameters
        ----------
        observations : Array
            Observation sequence.

        Returns
        -------
        params : LocalLevelParams
            Estimated parameters.
        """
        # Flatten to 1D for univariate model
        y = observations.flatten()

        # Method of moments estimation
        diff = jnp.diff(y)
        var_diff = jnp.var(diff)
        var_y = jnp.var(y)

        # Simple allocation (can be improved with more sophisticated methods)
        # Assume: Var(Δy) ≈ 2*sigma_level^2 + 2*sigma_obs^2 for random walk + noise
        # We use a heuristic split
        sigma_total = jnp.sqrt(var_diff / 2)
        sigma_level = sigma_total * 0.7  # 70% to level noise
        sigma_obs = sigma_total * 0.7  # Allow some overlap

        # Ensure positive values
        sigma_obs = jnp.maximum(sigma_obs, 0.01)
        sigma_level = jnp.maximum(sigma_level, 0.01)

        logger.debug(f"Estimated sigma_obs={sigma_obs:.4f}, sigma_level={sigma_level:.4f}")

        return LocalLevelParams(
            sigma_obs=jnp.asarray(sigma_obs),
            sigma_level=jnp.asarray(sigma_level),
            m0=jnp.asarray(y[0]),
            C0=jnp.asarray(var_y),
        )


class LocalLinearTrendAgent(BaseAgent):
    """Agent for the Local Linear Trend Model.

    The local linear trend model is:
        y_t = mu_t + nu_t,  nu_t ~ N(0, sigma_obs^2)
        mu_t = mu_{t-1} + beta_{t-1} + omega_mu,  omega_mu ~ N(0, sigma_level^2)
        beta_t = beta_{t-1} + omega_beta,  omega_beta ~ N(0, sigma_slope^2)

    This model captures data with a time-varying level and slope.
    """

    def __init__(self, config: SMCConfig):
        """Initialize Local Linear Trend Agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        """
        model = LocalLinearTrendModel()
        super().__init__(config, model)

    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> LocalLinearTrendParams:
        """Estimate parameters.

        Parameters
        ----------
        observations : Array
            Observation sequence.

        Returns
        -------
        params : LocalLinearTrendParams
            Estimated parameters.
        """
        y = observations.flatten()
        n = len(y)

        # First difference for level estimation
        diff1 = jnp.diff(y)
        var_diff1 = jnp.var(diff1)

        # Second difference for slope estimation
        diff2 = jnp.diff(diff1)
        var_diff2 = jnp.var(diff2)

        # Heuristic parameter estimation
        # Var(Δy) ≈ sigma_obs^2 + sigma_level^2 + sigma_slope^2
        # Var(Δ²y) ≈ 2*sigma_obs^2 + 2*sigma_level^2 + sigma_slope^2

        # Simple allocation
        sigma_slope = jnp.sqrt(jnp.maximum(var_diff2 / 6, 0.001))
        sigma_level = jnp.sqrt(jnp.maximum((var_diff1 - var_diff2 / 2) / 2, 0.001))
        sigma_obs = jnp.sqrt(jnp.maximum(var_diff1 / 3, 0.001))

        # Initial state estimation using simple linear regression
        slope_init = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
        level_init = y[0]

        logger.debug(
            f"Estimated sigma_obs={sigma_obs:.4f}, "
            f"sigma_level={sigma_level:.4f}, sigma_slope={sigma_slope:.4f}"
        )

        return LocalLinearTrendParams(
            sigma_obs=jnp.asarray(sigma_obs),
            sigma_level=jnp.asarray(sigma_level),
            sigma_slope=jnp.asarray(sigma_slope),
            m0=jnp.array([level_init, slope_init]),
            C0=jnp.diag(jnp.array([jnp.var(y), sigma_slope**2 * 10])),
        )
