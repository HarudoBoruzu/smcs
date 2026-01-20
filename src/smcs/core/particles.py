"""Particle state management for SMC algorithms.

This module provides data structures for managing particle states in SMC.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = [
    "SMCState",
    "SMCInfo",
]


@chex.dataclass(frozen=True)
class SMCState:
    """Immutable SMC state container.

    Attributes
    ----------
    particles : Array
        Particle states with shape [n_particles, state_dim].
    log_weights : Array
        Log-weights for each particle with shape [n_particles].
    ancestors : Array
        Ancestor indices from resampling with shape [n_particles].
    log_likelihood : float
        Cumulative log marginal likelihood estimate.
    step : int
        Current time step.
    """

    particles: Float[Array, "n_particles state_dim"]
    log_weights: Float[Array, " n_particles"]
    ancestors: Int[Array, " n_particles"]
    log_likelihood: float
    step: int

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.particles.shape[0]

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.particles.shape[1]

    def normalized_weights(self) -> Float[Array, " n_particles"]:
        """Return normalized weights (not log)."""
        import jax.scipy.special

        return jnp.exp(self.log_weights - jax.scipy.special.logsumexp(self.log_weights))

    def weighted_mean(self) -> Float[Array, " state_dim"]:
        """Compute weighted mean of particles."""
        weights = self.normalized_weights()
        return jnp.sum(self.particles * weights[:, None], axis=0)

    def weighted_cov(self) -> Float[Array, "state_dim state_dim"]:
        """Compute weighted covariance of particles."""
        weights = self.normalized_weights()
        mean = self.weighted_mean()
        centered = self.particles - mean
        return jnp.einsum("i,ij,ik->jk", weights, centered, centered)


@chex.dataclass(frozen=True)
class SMCInfo:
    """Diagnostic information from an SMC step.

    Attributes
    ----------
    ess : float
        Effective Sample Size.
    resampled : bool
        Whether resampling was performed.
    acceptance_rate : float | None
        MCMC acceptance rate (if applicable).
    """

    ess: float
    resampled: bool
    acceptance_rate: float | None = None
