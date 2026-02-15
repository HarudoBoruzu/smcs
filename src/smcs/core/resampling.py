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
    "killing_resample",
    "ssp_resample",
    "optimal_transport_resample",
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


@jaxtyped(typechecker=beartype)
def killing_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Killing resampling (branching process).

    Uses a branching process interpretation where particles are killed
    or duplicated based on their weights.

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

    # Expected number of offspring
    floor_counts = jnp.floor(scaled_weights).astype(jnp.int32)
    remainders = scaled_weights - floor_counts

    # Stochastic rounding for remainders
    key, round_key = jax.random.split(key)
    extra = (jax.random.uniform(round_key, shape=(n_particles,)) < remainders).astype(
        jnp.int32
    )
    counts = floor_counts + extra

    # Adjust counts to sum to n_particles
    total = jnp.sum(counts)
    n_particles - total

    # If diff > 0, add to random particles; if diff < 0, remove from random
    key, adjust_key = jax.random.split(key)

    def adjust_counts(counts, diff, key):
        # Simple adjustment: add/remove from highest/lowest weight particles
        sorted_idx = jnp.argsort(weights)
        if diff > 0:
            # Add to highest weight particles
            for i in range(diff):
                idx = sorted_idx[-(i % n_particles) - 1]
                counts = counts.at[idx].add(1)
        return counts

    # Use repeat to create indices
    indices = jnp.repeat(jnp.arange(n_particles), counts, total_repeat_length=n_particles)
    return indices


@jaxtyped(typechecker=beartype)
def ssp_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Srinivasan Sampling Process (SSP) resampling.

    A low-variance resampling method that guarantees the number of
    copies of each particle differs by at most 1 from the expected value.

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

    # Floor counts
    floor_counts = jnp.floor(scaled_weights).astype(jnp.int32)
    remainders = scaled_weights - floor_counts

    # Sort by remainders in descending order
    sorted_idx = jnp.argsort(-remainders)
    sorted_remainders = remainders[sorted_idx]

    # Cumulative sum of remainders
    cumsum = jnp.cumsum(sorted_remainders)
    n_extra = jnp.int32(jnp.round(cumsum[-1]))

    # Select particles to get extra copy
    u = jax.random.uniform(key)
    positions = (jnp.arange(n_extra) + u) / n_extra * cumsum[-1]
    extra_sorted = jnp.searchsorted(cumsum, positions)

    # Add extra counts
    extra_counts = jnp.zeros(n_particles, dtype=jnp.int32)
    extra_counts = extra_counts.at[sorted_idx].add(
        jnp.bincount(extra_sorted, length=n_particles)
    )

    counts = floor_counts + extra_counts

    # Create indices
    return jnp.repeat(jnp.arange(n_particles), counts, total_repeat_length=n_particles)


@jaxtyped(typechecker=beartype)
def optimal_transport_resample(
    key: PRNGKeyArray,
    log_weights: Float[Array, " n_particles"],
) -> Int[Array, " n_particles"]:
    """Optimal transport resampling.

    Minimizes the expected squared distance between resampled and
    original particle positions by solving an optimal transport problem.

    This is a simplified 1D version using the quantile coupling.

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

    # Compute cumulative distribution
    cumsum = jnp.cumsum(weights)

    # Target uniform quantiles
    uniform_quantiles = (jnp.arange(n_particles) + 0.5) / n_particles

    # Add small random perturbation for tie-breaking
    u = jax.random.uniform(key, shape=(n_particles,)) * 0.5 / n_particles
    perturbed_quantiles = uniform_quantiles + u - 0.25 / n_particles

    # Map to particle indices via inverse CDF
    indices = jnp.searchsorted(cumsum, perturbed_quantiles)
    indices = jnp.clip(indices, 0, n_particles - 1)

    return indices


ResamplingMethod = Literal[
    "systematic", "multinomial", "stratified", "residual",
    "killing", "ssp", "optimal_transport"
]


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
        Resampling method: "systematic", "multinomial", "stratified",
        "residual", "killing", "ssp", or "optimal_transport".

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
        "killing": killing_resample,
        "ssp": ssp_resample,
        "optimal_transport": optimal_transport_resample,
    }
    return methods[method](key, log_weights)
