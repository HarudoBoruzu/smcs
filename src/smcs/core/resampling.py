"""Resampling algorithms for SMC.

This module provides various resampling schemes:
- Systematic resampling (recommended, O(N), lowest variance)
- Multinomial resampling (simple, O(N log N))
- Stratified resampling (O(N), good variance properties)
- Residual resampling (O(N), minimum variance for uniform weights)
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

__all__ = [
    "systematic_resample",
    "multinomial_resample",
    "stratified_resample",
    "residual_resample",
    "resample",
]


@jaxtyped(typechecker=beartype)
def systematic_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Systematic resampling (O(N), lowest variance).

    Uses a single uniform random number to generate all samples,
    resulting in the lowest variance among resampling methods.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    log_weights : Array
        Log-weights (not necessarily normalized).

    Returns
    -------
    indices : Array
        Resampled particle indices.
    """
    n_particles = log_weights.shape[0]

    # Normalize weights and compute cumulative sum
    weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
    cumsum = jnp.cumsum(weights)

    # Generate systematic positions
    u0 = jax.random.uniform(key) / n_particles
    positions = u0 + jnp.arange(n_particles) / n_particles

    # Find indices using searchsorted
    return jnp.searchsorted(cumsum, positions)


@jaxtyped(typechecker=beartype)
def multinomial_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Multinomial resampling (simple but higher variance).

    Samples independently from the categorical distribution defined by weights.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    log_weights : Array
        Log-weights (not necessarily normalized).

    Returns
    -------
    indices : Array
        Resampled particle indices.
    """
    n_particles = log_weights.shape[0]
    log_probs = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jax.random.categorical(key, log_probs, shape=(n_particles,))


@jaxtyped(typechecker=beartype)
def stratified_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Stratified resampling (O(N), good variance properties).

    Divides [0,1] into N strata and samples one point from each stratum.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    log_weights : Array
        Log-weights (not necessarily normalized).

    Returns
    -------
    indices : Array
        Resampled particle indices.
    """
    n_particles = log_weights.shape[0]

    # Normalize weights and compute cumulative sum
    weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
    cumsum = jnp.cumsum(weights)

    # Generate stratified positions
    u = jax.random.uniform(key, shape=(n_particles,))
    positions = (jnp.arange(n_particles) + u) / n_particles

    return jnp.searchsorted(cumsum, positions)


@jaxtyped(typechecker=beartype)
def residual_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Residual resampling.

    First deterministically copies floor(N*w_i) copies of particle i,
    then multinomial samples the remaining particles from residual weights.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    log_weights : Array
        Log-weights (not necessarily normalized).

    Returns
    -------
    indices : Array
        Resampled particle indices.
    """
    n_particles = log_weights.shape[0]

    # Normalize weights
    weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
    scaled_weights = n_particles * weights

    # Deterministic part: floor(N * w_i) copies
    counts = jnp.floor(scaled_weights).astype(jnp.int32)
    n_deterministic = jnp.sum(counts)

    # Residual weights for stochastic part
    residuals = scaled_weights - counts
    residuals = residuals / jnp.sum(residuals)

    # Create deterministic indices
    det_indices = jnp.repeat(jnp.arange(n_particles), counts, total_repeat_length=n_particles)

    # Stochastic part (n_stochastic = n_particles - n_deterministic)
    stoch_indices = jax.random.choice(
        key, n_particles, shape=(n_particles,), p=residuals, replace=True
    )

    # Combine: use deterministic where available, else stochastic
    idx = jnp.arange(n_particles)
    return jnp.where(idx < n_deterministic, det_indices, stoch_indices)


ResamplingMethod = Literal["systematic", "multinomial", "stratified", "residual"]


def resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
    method: ResamplingMethod = "systematic",
) -> Int[Array, " n_particles"]:
    """Resample particles according to specified method.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    log_weights : Array
        Log-weights (not necessarily normalized).
    method : str
        Resampling method: "systematic", "multinomial", "stratified", or "residual".

    Returns
    -------
    indices : Array
        Resampled particle indices.
    """
    methods = {
        "systematic": systematic_resample,
        "multinomial": multinomial_resample,
        "stratified": stratified_resample,
        "residual": residual_resample,
    }
    return methods[method](key, log_weights)
