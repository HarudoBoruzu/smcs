"""Bootstrap resampling methods.

Classical bootstrap methods for statistical inference including:
- Ordinary (IID) bootstrap
- Block bootstrap (for time series)
- Moving block bootstrap
- Circular block bootstrap
- Stationary bootstrap
- Wild bootstrap (for heteroscedasticity)
- Residual bootstrap
- Parametric bootstrap
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

__all__ = [
    "ordinary_bootstrap",
    "block_bootstrap",
    "moving_block_bootstrap",
    "circular_block_bootstrap",
    "stationary_bootstrap",
    "wild_bootstrap",
    "residual_bootstrap",
    "parametric_bootstrap",
    "bootstrap_ci",
    "jackknife",
]


@jaxtyped(typechecker=beartype)
def ordinary_bootstrap(
    key: PRNGKeyArray,
    data: Float[Array, "n ..."],
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Ordinary (IID) bootstrap resampling.

    Resample with replacement from the original data.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    data : Array
        Original data of shape (n, ...).
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples of shape (n_bootstrap, n, ...).
    """
    n = data.shape[0]

    def resample_once(key_i):
        indices = jax.random.randint(key_i, shape=(n,), minval=0, maxval=n)
        return data[indices]

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def block_bootstrap(
    key: PRNGKeyArray,
    data: Float[Array, "n ..."],
    block_size: int,
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Non-overlapping block bootstrap for time series.

    Resamples blocks of consecutive observations to preserve
    temporal dependence structure.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    data : Array
        Time series data of shape (n, ...).
    block_size : int
        Size of each block.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples of shape (n_bootstrap, n, ...).
    """
    n = data.shape[0]
    n_blocks = n // block_size
    remainder = n % block_size

    def resample_once(key_i):
        # Sample block indices
        block_indices = jax.random.randint(
            key_i, shape=(n_blocks + 1,), minval=0, maxval=n_blocks
        )

        # Build resampled series from blocks
        def get_block(idx):
            start = idx * block_size
            return jax.lax.dynamic_slice(
                data, (start,) + (0,) * (data.ndim - 1),
                (block_size,) + data.shape[1:]
            )

        blocks = jax.vmap(get_block)(block_indices[:n_blocks])
        resampled = blocks.reshape((-1,) + data.shape[1:])

        # Handle remainder
        if remainder > 0:
            extra_block = get_block(block_indices[-1])
            resampled = jnp.concatenate([resampled, extra_block[:remainder]], axis=0)

        return resampled[:n]

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def moving_block_bootstrap(
    key: PRNGKeyArray,
    data: Float[Array, "n ..."],
    block_size: int,
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Moving block bootstrap (overlapping blocks).

    Uses overlapping blocks starting from any position in the series.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    data : Array
        Time series data of shape (n, ...).
    block_size : int
        Size of each block.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples of shape (n_bootstrap, n, ...).
    """
    n = data.shape[0]
    n_possible_blocks = n - block_size + 1
    n_blocks_needed = (n + block_size - 1) // block_size

    def resample_once(key_i):
        # Sample starting positions for blocks
        starts = jax.random.randint(
            key_i, shape=(n_blocks_needed,), minval=0, maxval=n_possible_blocks
        )

        def get_block(start):
            return jax.lax.dynamic_slice(
                data, (start,) + (0,) * (data.ndim - 1),
                (block_size,) + data.shape[1:]
            )

        blocks = jax.vmap(get_block)(starts)
        resampled = blocks.reshape((-1,) + data.shape[1:])
        return resampled[:n]

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def circular_block_bootstrap(
    key: PRNGKeyArray,
    data: Float[Array, "n ..."],
    block_size: int,
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Circular block bootstrap.

    Treats the series as circular, allowing blocks to wrap around.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    data : Array
        Time series data of shape (n, ...).
    block_size : int
        Size of each block.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples of shape (n_bootstrap, n, ...).
    """
    n = data.shape[0]
    n_blocks_needed = (n + block_size - 1) // block_size

    # Extend data circularly
    extended_data = jnp.concatenate([data, data[:block_size - 1]], axis=0)

    def resample_once(key_i):
        # Sample starting positions (can start anywhere)
        starts = jax.random.randint(
            key_i, shape=(n_blocks_needed,), minval=0, maxval=n
        )

        def get_block(start):
            return jax.lax.dynamic_slice(
                extended_data, (start,) + (0,) * (data.ndim - 1),
                (block_size,) + data.shape[1:]
            )

        blocks = jax.vmap(get_block)(starts)
        resampled = blocks.reshape((-1,) + data.shape[1:])
        return resampled[:n]

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def stationary_bootstrap(
    key: PRNGKeyArray,
    data: Float[Array, "n ..."],
    mean_block_size: float,
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Stationary bootstrap with random block lengths.

    Block lengths follow a geometric distribution, making the
    resampled series stationary.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    data : Array
        Time series data of shape (n, ...).
    mean_block_size : float
        Expected block length (1/p where p is geometric parameter).
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples of shape (n_bootstrap, n, ...).
    """
    n = data.shape[0]
    p = 1.0 / mean_block_size  # Probability of ending block

    def resample_once(key_i):
        key_start, key_geom = jax.random.split(key_i)

        # Generate starting points and continuation indicators
        uniforms = jax.random.uniform(key_geom, shape=(n,))
        continue_block = uniforms > p  # Continue with prob (1-p)

        # Random starting points for new blocks
        new_starts = jax.random.randint(key_start, shape=(n,), minval=0, maxval=n)

        # Build resampled indices
        def scan_fn(carry, inputs):
            current_idx = carry
            cont, new_start = inputs

            # Either continue current block or start new one
            next_idx = jnp.where(cont, (current_idx + 1) % n, new_start)
            return next_idx, current_idx

        initial_idx = new_starts[0]
        _, indices = jax.lax.scan(scan_fn, initial_idx, (continue_block, new_starts))

        return data[indices]

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def wild_bootstrap(
    key: PRNGKeyArray,
    residuals: Float[Array, "n ..."],
    fitted_values: Float[Array, "n ..."],
    n_bootstrap: int = 1000,
    distribution: str = "rademacher",
) -> Float[Array, "n_bootstrap n ..."]:
    """Wild bootstrap for heteroscedastic errors.

    Multiplies residuals by random weights to preserve heteroscedasticity.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    residuals : Array
        Residuals from model fit.
    fitted_values : Array
        Fitted values from model.
    n_bootstrap : int
        Number of bootstrap samples.
    distribution : str
        Distribution for weights: "rademacher", "mammen", or "normal".

    Returns
    -------
    samples : Array
        Bootstrap samples (fitted + weighted residuals).
    """
    n = residuals.shape[0]

    def get_weights(key_i):
        if distribution == "rademacher":
            # +1 or -1 with equal probability
            return 2.0 * jax.random.bernoulli(key_i, 0.5, shape=(n,)) - 1.0
        elif distribution == "mammen":
            # Two-point distribution: (1-sqrt(5))/2 and (1+sqrt(5))/2
            sqrt5 = jnp.sqrt(5.0)
            p = (sqrt5 + 1) / (2 * sqrt5)
            u = jax.random.bernoulli(key_i, p, shape=(n,))
            return jnp.where(u, (1 - sqrt5) / 2, (1 + sqrt5) / 2)
        else:  # normal
            return jax.random.normal(key_i, shape=(n,))

    def resample_once(key_i):
        weights = get_weights(key_i)
        if residuals.ndim > 1:
            weighted_residuals = residuals * weights[..., None]
        else:
            weighted_residuals = residuals * weights
        return fitted_values + weighted_residuals

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def residual_bootstrap(
    key: PRNGKeyArray,
    residuals: Float[Array, "n ..."],
    fitted_values: Float[Array, "n ..."],
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap n ..."]:
    """Residual bootstrap for regression.

    Resamples residuals and adds to fitted values.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    residuals : Array
        Residuals from model fit (centered).
    fitted_values : Array
        Fitted values from model.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples.
    """
    n = residuals.shape[0]

    # Center residuals
    centered_residuals = residuals - jnp.mean(residuals, axis=0)

    def resample_once(key_i):
        indices = jax.random.randint(key_i, shape=(n,), minval=0, maxval=n)
        resampled_residuals = centered_residuals[indices]
        return fitted_values + resampled_residuals

    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(resample_once)(keys)


@jaxtyped(typechecker=beartype)
def parametric_bootstrap(
    key: PRNGKeyArray,
    sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    n_bootstrap: int = 1000,
) -> Float[Array, "n_bootstrap ..."]:
    """Parametric bootstrap.

    Samples from a parametric distribution (e.g., fitted model).

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    sample_fn : Callable
        Function (key) -> sample that generates one bootstrap sample.
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    samples : Array
        Bootstrap samples.
    """
    keys = jax.random.split(key, n_bootstrap)
    return jax.vmap(sample_fn)(keys)


@jaxtyped(typechecker=beartype)
def bootstrap_ci(
    bootstrap_estimates: Float[Array, " n_bootstrap"],
    confidence: float = 0.95,
    method: str = "percentile",
    original_estimate: Float[Array, ""] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute bootstrap confidence interval.

    Parameters
    ----------
    bootstrap_estimates : Array
        Bootstrap estimates of the statistic.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    method : str
        Method: "percentile", "basic", or "bca".
    original_estimate : float, optional
        Original estimate (required for "basic" method).

    Returns
    -------
    lower, upper : float
        Lower and upper bounds of confidence interval.
    """
    alpha = 1 - confidence

    if method == "percentile":
        lower = jnp.percentile(bootstrap_estimates, 100 * alpha / 2)
        upper = jnp.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    elif method == "basic":
        if original_estimate is None:
            raise ValueError("original_estimate required for basic method")
        q_low = jnp.percentile(bootstrap_estimates, 100 * alpha / 2)
        q_high = jnp.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        lower = 2 * original_estimate - q_high
        upper = 2 * original_estimate - q_low

    else:  # bca (simplified without acceleration)
        lower = jnp.percentile(bootstrap_estimates, 100 * alpha / 2)
        upper = jnp.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return lower, upper


@jaxtyped(typechecker=beartype)
def jackknife(
    data: Float[Array, "n ..."],
    statistic_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " n"]]:
    """Jackknife resampling for bias and variance estimation.

    Parameters
    ----------
    data : Array
        Original data of shape (n, ...).
    statistic_fn : Callable
        Function to compute statistic from data.

    Returns
    -------
    estimate : float
        Bias-corrected jackknife estimate.
    std_error : float
        Jackknife standard error.
    leave_one_out : Array
        Leave-one-out estimates.
    """
    n = data.shape[0]
    original_stat = statistic_fn(data)

    # Compute leave-one-out estimates using a loop (JAX compatible)
    def compute_leave_one_out(carry, i):
        # Create indices excluding i
        indices = jnp.where(jnp.arange(n) < i, jnp.arange(n), jnp.arange(n) + 1)
        indices = indices[:n - 1]
        reduced_data = data[indices]
        stat = statistic_fn(reduced_data)
        return carry, stat

    _, leave_one_out = jax.lax.scan(compute_leave_one_out, None, jnp.arange(n))

    # Jackknife estimate and standard error
    jackknife_mean = jnp.mean(leave_one_out)
    bias = (n - 1) * (jackknife_mean - original_stat)
    estimate = original_stat - bias

    # Variance
    variance = ((n - 1) / n) * jnp.sum((leave_one_out - jackknife_mean) ** 2)
    std_error = jnp.sqrt(variance)

    return estimate, std_error, leave_one_out
