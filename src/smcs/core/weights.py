"""Weight computation utilities for SMC algorithms.

This module provides functions for computing ESS and normalizing log-weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

__all__ = [
    "compute_ess",
    "normalize_log_weights",
    "log_mean_exp",
]


@jaxtyped(typechecker=beartype)
def compute_ess(log_weights: Float[Array, " n_particles"]) -> Float[Array, ""]:
    """Compute Effective Sample Size (ESS).

    ESS = 1 / sum(w_i^2) = exp(-logsumexp(2 * log_w_normalized))

    Parameters
    ----------
    log_weights : Array
        Log-weights (not necessarily normalized).

    Returns
    -------
    ess : Array
        Effective sample size (scalar), in range (0, n_particles].
    """
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jnp.exp(-jax.scipy.special.logsumexp(2 * log_weights_normalized))


@jaxtyped(typechecker=beartype)
def normalize_log_weights(
    log_weights: Float[Array, " n_particles"],
) -> Float[Array, " n_particles"]:
    """Normalize log-weights so they sum to 1 (in probability space).

    Parameters
    ----------
    log_weights : Array
        Unnormalized log-weights.

    Returns
    -------
    normalized : Array
        Normalized log-weights where exp(log_weights).sum() == 1.
    """
    return log_weights - jax.scipy.special.logsumexp(log_weights)


@jaxtyped(typechecker=beartype)
def log_mean_exp(log_values: Float[Array, " n"]) -> Float[Array, ""]:
    """Compute log of the mean of exp(log_values).

    log(mean(exp(x))) = logsumexp(x) - log(n)

    This is useful for computing marginal likelihood estimates.

    Parameters
    ----------
    log_values : Array
        Log-values to average.

    Returns
    -------
    result : Array
        Log of the mean of the exponentiated values (scalar).
    """
    n = log_values.shape[0]
    return jax.scipy.special.logsumexp(log_values) - jnp.log(n)
