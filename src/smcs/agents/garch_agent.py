"""GARCH-based forecasting agents.

This module provides agents for GARCH volatility models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger

from smcs.agents.base import BaseAgent
from smcs.models.garch import GARCHModel, GARCHParams

if TYPE_CHECKING:
    from smcs.config.settings import SMCConfig

__all__ = [
    "GARCHAgent",
]


class GARCHAgent(BaseAgent):
    """Agent for GARCH(p, q) models.

    GARCH models capture time-varying volatility (heteroskedasticity)
    in financial time series.

    Note: GARCH has deterministic volatility dynamics, so SMC is mainly
    useful for missing data handling and parameter learning.
    """

    def __init__(
        self,
        config: "SMCConfig",
        order: tuple[int, int] = (1, 1),
    ):
        """Initialize GARCH Agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        order : tuple
            GARCH(p, q) order.
        """
        self._order = order
        model = GARCHModel(order)
        super().__init__(config, model)

    @property
    def order(self) -> tuple[int, int]:
        """GARCH order (p, q)."""
        return self._order

    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> GARCHParams:
        """Estimate GARCH parameters.

        Uses moment-based estimation as initialization.
        For better estimates, use PMMH or external optimization.

        Parameters
        ----------
        observations : Array
            Observation sequence (returns).

        Returns
        -------
        params : GARCHParams
            Estimated parameters.
        """
        y = observations.flatten()
        p, q = self._order

        # Unconditional variance
        var_y = jnp.var(y)
        sigma2 = jnp.maximum(var_y, 1e-6)

        # Persistence estimation from autocorrelation of squared returns
        y2 = y**2
        mean_y2 = jnp.mean(y2)

        # Autocorrelation of squared returns
        centered_y2 = y2 - mean_y2
        if len(y2) > 1:
            acf1_y2 = jnp.sum(centered_y2[:-1] * centered_y2[1:]) / jnp.sum(centered_y2**2)
        else:
            acf1_y2 = 0.0

        # For GARCH(1,1): alpha + beta ≈ acf(1) of squared returns
        persistence = jnp.clip(acf1_y2, 0.0, 0.98)

        # Heuristic allocation between alpha and beta
        # Typically beta > alpha in financial data
        alpha = jnp.array([persistence * 0.15])  # ARCH effect
        beta = jnp.array([persistence * 0.8])  # GARCH effect

        # Pad if needed
        if q > 1:
            alpha = jnp.concatenate([alpha, jnp.zeros(q - 1)])
        if p > 1:
            beta = jnp.concatenate([beta, jnp.zeros(p - 1)])

        # omega from unconditional variance: sigma2 = omega / (1 - alpha - beta)
        omega = sigma2 * (1 - jnp.sum(alpha) - jnp.sum(beta))
        omega = jnp.maximum(omega, 1e-8)

        # Initial volatility (unconditional)
        sigma0 = jnp.sqrt(sigma2)

        logger.debug(
            f"Estimated GARCH params: omega={omega:.6f}, "
            f"alpha={alpha}, beta={beta}"
        )

        return GARCHParams(
            omega=float(omega),
            alpha=alpha,
            beta=beta,
            sigma0=float(sigma0),
        )
