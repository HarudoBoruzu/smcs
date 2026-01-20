"""Stochastic Volatility forecasting agents.

This module provides agents for Stochastic Volatility models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float
from loguru import logger

from smcs.agents.base import BaseAgent
from smcs.models.sv import SVModel, SVParams

if TYPE_CHECKING:
    from smcs.config.settings import SMCConfig

__all__ = [
    "SVAgent",
]


class SVAgent(BaseAgent):
    """Agent for Stochastic Volatility models.

    The SV model is:
        y_t = exp(h_t / 2) * epsilon_t,  epsilon_t ~ N(0, 1)
        h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,  eta_t ~ N(0, 1)

    Unlike GARCH, SV models have stochastic volatility dynamics,
    making SMC particularly well-suited for inference.
    """

    def __init__(self, config: "SMCConfig"):
        """Initialize SV Agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        """
        model = SVModel()
        super().__init__(config, model)

    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> SVParams:
        """Estimate SV parameters.

        Uses moment-based estimation. For better estimates,
        consider using PMMH or SMC².

        Parameters
        ----------
        observations : Array
            Observation sequence (returns).

        Returns
        -------
        params : SVParams
            Estimated parameters.
        """
        y = observations.flatten()

        # Use log absolute returns as proxy for log-volatility
        abs_y = jnp.abs(y)
        # Add small constant to avoid log(0)
        log_abs_y = jnp.log(abs_y + 1e-8)

        # Estimate mu (long-run mean log-volatility)
        # E[log|y_t|] ≈ mu/2 + E[log|epsilon|] where E[log|epsilon|] ≈ -0.635 for N(0,1)
        mean_log_abs_y = jnp.mean(log_abs_y)
        mu = 2 * (mean_log_abs_y + 0.635)

        # Estimate phi from autocorrelation of log|y|
        centered = log_abs_y - mean_log_abs_y
        var_centered = jnp.var(log_abs_y)
        if len(log_abs_y) > 1:
            acf1 = jnp.sum(centered[:-1] * centered[1:]) / (len(centered) - 1) / var_centered
        else:
            acf1 = 0.9

        # Autocorrelation of log|y| is related to phi
        # For SV: acf_log|y|(1) ≈ phi * Var(h) / (Var(h) + pi^2/2)
        # Rough approximation: phi ≈ acf1 * 1.5 (accounting for measurement noise)
        phi = jnp.clip(acf1 * 1.5, 0.8, 0.999)

        # Estimate sigma_eta from variance of log|y|
        # Var(log|y|) ≈ Var(h)/4 + pi^2/8
        # Var(h) = sigma_eta^2 / (1 - phi^2)
        var_log_abs_y = jnp.var(log_abs_y)
        var_h = jnp.maximum((var_log_abs_y - jnp.pi**2 / 8) * 4, 0.01)
        sigma_eta = jnp.sqrt(var_h * (1 - phi**2))
        sigma_eta = jnp.clip(sigma_eta, 0.01, 1.0)

        # Initial conditions
        h0 = mu  # Start at long-run mean
        P0 = var_h  # Stationary variance

        logger.debug(
            f"Estimated SV params: mu={mu:.4f}, phi={phi:.4f}, sigma_eta={sigma_eta:.4f}"
        )

        return SVParams(
            mu=float(mu),
            phi=float(phi),
            sigma_eta=float(sigma_eta),
            h0=float(h0),
            P0=float(P0),
        )
