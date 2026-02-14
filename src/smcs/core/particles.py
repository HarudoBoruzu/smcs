"""Particle state management for SMC algorithms.

This module provides data structures for managing particle states in SMC.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

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
    log_likelihood : Array
        Cumulative log marginal likelihood estimate (scalar).
    step : Array
        Current time step (scalar integer).
    """

    particles: Float[Array, "n_particles state_dim"]
    log_weights: Float[Array, " n_particles"]
    ancestors: Int[Array, " n_particles"]
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]

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
    ess : Array
        Effective Sample Size (scalar).
    resampled : Array
        Whether resampling was performed (scalar boolean).
    acceptance_rate : Array | None
        MCMC acceptance rate (if applicable, scalar).
    """

    ess: Float[Array, ""]
    resampled: Bool[Array, ""]
    acceptance_rate: Float[Array, ""] | None = None
