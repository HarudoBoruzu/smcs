"""ARIMA-based forecasting agents.

This module provides agents for ARIMA models using SMC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger

from smcs.agents.base import BaseAgent
from smcs.models.arima import ARIMAModel, ARIMAParams

if TYPE_CHECKING:
    from smcs.config.settings import SMCConfig

__all__ = [
    "ARIMAAgent",
]


class ARIMAAgent(BaseAgent):
    """Agent for ARIMA(p, d, q) models.

    Uses Harvey's state space representation for ARIMA models,
    enabling particle filtering for state estimation.
    """

    def __init__(
        self,
        config: "SMCConfig",
        order: tuple[int, int, int] = (1, 0, 0),
    ):
        """Initialize ARIMA Agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        order : tuple
            ARIMA(p, d, q) order.
        """
        self._order = order
        model = ARIMAModel(order)
        super().__init__(config, model)

    @property
    def order(self) -> tuple[int, int, int]:
        """ARIMA order (p, d, q)."""
        return self._order

    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> ARIMAParams:
        """Estimate ARIMA parameters.

        Uses simple moment-based estimation for AR parameters.
        For more accurate estimation, consider using Liu-West filter
        or PMMH.

        Parameters
        ----------
        observations : Array
            Observation sequence.

        Returns
        -------
        params : ARIMAParams
            Estimated parameters.
        """
        y = observations.flatten()
        p, d, q = self._order

        # Apply differencing if needed
        y_diff = y
        for _ in range(d):
            y_diff = jnp.diff(y_diff)

        n = len(y_diff)

        # Estimate AR coefficients using Yule-Walker
        if p > 0:
            # Compute autocorrelations
            mean_y = jnp.mean(y_diff)
            centered = y_diff - mean_y
            var_y = jnp.var(y_diff)

            acf = jnp.zeros(p + 1)
            acf = acf.at[0].set(1.0)
            for k in range(1, p + 1):
                if n > k:
                    acf = acf.at[k].set(
                        jnp.sum(centered[:-k] * centered[k:]) / ((n - k) * var_y)
                    )

            # Yule-Walker equations (simple version)
            # For AR(1): phi_1 = acf[1]
            # For higher orders, we'd need to solve the Yule-Walker system
            if p == 1:
                ar_coeffs = jnp.array([jnp.clip(acf[1], -0.99, 0.99)])
            else:
                # Simple approximation for higher orders
                ar_coeffs = jnp.clip(acf[1 : p + 1], -0.99, 0.99)
        else:
            ar_coeffs = jnp.array([])

        # MA coefficients (simple initialization)
        if q > 0:
            # Initialize MA coefficients to small values
            ma_coeffs = jnp.zeros(q)
        else:
            ma_coeffs = jnp.array([])

        # Innovation variance
        if p > 0:
            residual_var = var_y * (1 - jnp.sum(ar_coeffs * acf[1 : p + 1]))
        else:
            residual_var = jnp.var(y_diff)
        sigma = jnp.sqrt(jnp.maximum(residual_var, 0.001))

        # State dimension
        state_dim = max(p + d, q + 1) if (p > 0 or q > 0) else 1

        # Initial state
        m0 = jnp.zeros(state_dim)
        if n > 0:
            m0 = m0.at[0].set(y_diff[-1] if len(y_diff) > 0 else 0.0)

        # Initial covariance (stationary variance approximation)
        if p > 0 and jnp.sum(ar_coeffs**2) < 1:
            stationary_var = float(sigma**2 / (1 - jnp.sum(ar_coeffs**2)))
        else:
            stationary_var = float(sigma**2)
        C0 = jnp.eye(state_dim) * stationary_var

        logger.debug(f"Estimated AR coeffs: {ar_coeffs}, sigma: {sigma:.4f}")

        return ARIMAParams(
            ar_coeffs=ar_coeffs,
            ma_coeffs=ma_coeffs,
            sigma=float(sigma),
            d=d,
            m0=m0,
            C0=C0,
        )
