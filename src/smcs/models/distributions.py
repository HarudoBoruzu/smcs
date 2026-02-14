"""Probability distribution wrappers for SMC.

This module provides a unified interface for probability distributions
that wraps JAX's distribution implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

__all__ = [
    "Distribution",
    "Normal",
    "MultivariateNormal",
    "Uniform",
    "Categorical",
]


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Float[Array, ...]:
        """Draw a sample from the distribution.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.

        Returns
        -------
        sample : Array
            A sample from the distribution.
        """
        ...

    @abstractmethod
    def log_prob(self, value: Float[Array, ...]) -> Float[Array, ""]:
        """Compute log probability of a value.

        Parameters
        ----------
        value : Array
            Value to evaluate.

        Returns
        -------
        log_prob : Array
            Log probability (scalar JAX array).
        """
        ...

    def prob(self, value: Float[Array, ...]) -> Float[Array, ""]:
        """Compute probability of a value.

        Parameters
        ----------
        value : Array
            Value to evaluate.

        Returns
        -------
        prob : Array
            Probability (scalar JAX array).
        """
        return jnp.exp(self.log_prob(value))


class Normal(Distribution):
    """Univariate normal distribution."""

    def __init__(self, loc: float | Float[Array, ""], scale: float | Float[Array, ""]):
        """Initialize normal distribution.

        Parameters
        ----------
        loc : float
            Mean (location).
        scale : float
            Standard deviation (scale).
        """
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)

    def sample(self, key: PRNGKeyArray) -> Float[Array, ""]:
        """Draw a sample."""
        return self.loc + self.scale * jax.random.normal(key)

    def log_prob(self, value: Float[Array, ""]) -> Float[Array, ""]:
        """Compute log probability."""
        z = (value - self.loc) / self.scale
        return -0.5 * z**2 - jnp.log(self.scale) - 0.5 * jnp.log(2 * jnp.pi)


class MultivariateNormal(Distribution):
    """Multivariate normal distribution."""

    def __init__(
        self,
        loc: Float[Array, " dim"],
        covariance_matrix: Float[Array, "dim dim"] | None = None,
        scale_tril: Float[Array, "dim dim"] | None = None,
    ):
        """Initialize multivariate normal distribution.

        Parameters
        ----------
        loc : Array
            Mean vector.
        covariance_matrix : Array, optional
            Covariance matrix (will compute Cholesky).
        scale_tril : Array, optional
            Lower triangular Cholesky factor of covariance.
            Provide either covariance_matrix or scale_tril, not both.
        """
        self.loc = jnp.asarray(loc)

        if scale_tril is not None:
            self.scale_tril = jnp.asarray(scale_tril)
        elif covariance_matrix is not None:
            self.scale_tril = jnp.linalg.cholesky(covariance_matrix)
        else:
            raise ValueError("Must provide either covariance_matrix or scale_tril")

        self._dim = self.loc.shape[0]

    def sample(self, key: PRNGKeyArray) -> Float[Array, " dim"]:
        """Draw a sample."""
        z = jax.random.normal(key, shape=(self._dim,))
        return self.loc + self.scale_tril @ z

    def log_prob(self, value: Float[Array, " dim"]) -> Float[Array, ""]:
        """Compute log probability."""
        diff = value - self.loc
        # Solve L @ y = diff for y, then compute y^T @ y
        y = jax.scipy.linalg.solve_triangular(self.scale_tril, diff, lower=True)
        mahalanobis = jnp.dot(y, y)

        log_det = 2 * jnp.sum(jnp.log(jnp.diag(self.scale_tril)))
        return -0.5 * (self._dim * jnp.log(2 * jnp.pi) + log_det + mahalanobis)


class Uniform(Distribution):
    """Uniform distribution."""

    def __init__(
        self,
        low: float | Float[Array, ...] = 0.0,
        high: float | Float[Array, ...] = 1.0,
    ):
        """Initialize uniform distribution.

        Parameters
        ----------
        low : float or Array
            Lower bound.
        high : float or Array
            Upper bound.
        """
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)

    def sample(self, key: PRNGKeyArray) -> Float[Array, ...]:
        """Draw a sample."""
        return jax.random.uniform(key, shape=self.low.shape, minval=self.low, maxval=self.high)

    def log_prob(self, value: Float[Array, ...]) -> Float[Array, ""]:
        """Compute log probability."""
        in_support = jnp.all((value >= self.low) & (value <= self.high))
        log_p = -jnp.sum(jnp.log(self.high - self.low))
        return jnp.where(in_support, log_p, -jnp.inf)


class Categorical(Distribution):
    """Categorical distribution."""

    def __init__(
        self,
        logits: Float[Array, " n_categories"] | None = None,
        probs: Float[Array, " n_categories"] | None = None,
    ):
        """Initialize categorical distribution.

        Parameters
        ----------
        logits : Array, optional
            Unnormalized log probabilities.
        probs : Array, optional
            Probabilities (will convert to logits).
        """
        if logits is not None:
            self.logits = jnp.asarray(logits)
        elif probs is not None:
            self.logits = jnp.log(jnp.asarray(probs))
        else:
            raise ValueError("Must provide either logits or probs")

    def sample(self, key: PRNGKeyArray) -> Int[Array, ""]:
        """Draw a sample."""
        return jax.random.categorical(key, self.logits)

    def log_prob(self, value: Int[Array, ""] | int) -> Float[Array, ""]:
        """Compute log probability."""
        log_probs = self.logits - jax.scipy.special.logsumexp(self.logits)
        return log_probs[value]
