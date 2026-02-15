"""Importance Sampling methods.

Classical importance sampling techniques including:
- Basic importance sampling
- Self-normalized importance sampling
- Multiple importance sampling
- Adaptive importance sampling
- Effective sample size computation
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

__all__ = [
    "importance_sample",
    "self_normalized_is",
    "multiple_importance_sampling",
    "adaptive_importance_sampling",
    "compute_ess_is",
    "compute_is_diagnostics",
]


@jaxtyped(typechecker=beartype)
def importance_sample(
    key: PRNGKeyArray,
    proposal_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_target_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    log_proposal_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    n_samples: int = 1000,
) -> tuple[Float[Array, "n_samples ..."], Float[Array, " n_samples"]]:
    """Basic importance sampling.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    proposal_sample_fn : Callable
        Function to sample from proposal distribution.
    log_target_fn : Callable
        Log probability of target distribution.
    log_proposal_fn : Callable
        Log probability of proposal distribution.
    n_samples : int
        Number of samples.

    Returns
    -------
    samples : Array
        Importance samples.
    log_weights : Array
        Unnormalized log importance weights.
    """
    keys = jax.random.split(key, n_samples)
    samples = jax.vmap(proposal_sample_fn)(keys)

    log_target = jax.vmap(log_target_fn)(samples)
    log_proposal = jax.vmap(log_proposal_fn)(samples)
    log_weights = log_target - log_proposal

    return samples, log_weights


@jaxtyped(typechecker=beartype)
def self_normalized_is(
    key: PRNGKeyArray,
    proposal_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_target_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    log_proposal_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    n_samples: int = 1000,
) -> tuple[Float[Array, "n_samples ..."], Float[Array, " n_samples"]]:
    """Self-normalized importance sampling.

    Normalizes weights to sum to 1, useful when target is known
    only up to a constant.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    proposal_sample_fn : Callable
        Proposal sampler.
    log_target_fn : Callable
        Log unnormalized target.
    log_proposal_fn : Callable
        Log proposal.
    n_samples : int
        Number of samples.

    Returns
    -------
    samples : Array
        Importance samples.
    normalized_weights : Array
        Normalized importance weights (sum to 1).
    """
    samples, log_weights = importance_sample(
        key, proposal_sample_fn, log_target_fn, log_proposal_fn, n_samples
    )

    # Normalize weights in log space for numerical stability
    log_sum_weights = jax.scipy.special.logsumexp(log_weights)
    normalized_weights = jnp.exp(log_weights - log_sum_weights)

    return samples, normalized_weights


@jaxtyped(typechecker=beartype)
def multiple_importance_sampling(
    key: PRNGKeyArray,
    proposal_sample_fns: list[Callable[[PRNGKeyArray], Float[Array, ...]]],
    log_target_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    log_proposal_fns: list[Callable[[Float[Array, ...]], Float[Array, ""]]],
    n_samples_per_proposal: int = 100,
    weighting: str = "balance",
) -> tuple[Float[Array, "n_total ..."], Float[Array, " n_total"]]:
    """Multiple importance sampling (MIS).

    Combines samples from multiple proposals using optimal weighting.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    proposal_sample_fns : list[Callable]
        List of proposal samplers.
    log_target_fn : Callable
        Log target density.
    log_proposal_fns : list[Callable]
        List of log proposal densities.
    n_samples_per_proposal : int
        Samples from each proposal.
    weighting : str
        "balance" (recommended) or "maximum".

    Returns
    -------
    samples : Array
        Combined samples.
    weights : Array
        MIS weights.
    """
    n_proposals = len(proposal_sample_fns)
    keys = jax.random.split(key, n_proposals)

    all_samples = []
    all_log_target = []
    all_log_proposals = []

    for _i, (sample_fn, key_i) in enumerate(zip(proposal_sample_fns, keys, strict=False)):
        sample_keys = jax.random.split(key_i, n_samples_per_proposal)
        samples_i = jax.vmap(sample_fn)(sample_keys)
        all_samples.append(samples_i)

        log_target_i = jax.vmap(log_target_fn)(samples_i)
        all_log_target.append(log_target_i)

        # Evaluate all proposals at these samples
        log_props_i = []
        for log_prop_fn in log_proposal_fns:
            log_props_i.append(jax.vmap(log_prop_fn)(samples_i))
        all_log_proposals.append(jnp.stack(log_props_i, axis=0))

    # Concatenate
    samples = jnp.concatenate(all_samples, axis=0)
    log_target = jnp.concatenate(all_log_target, axis=0)
    log_proposals = jnp.concatenate(all_log_proposals, axis=1)  # (n_proposals, n_total)

    samples.shape[0]

    # Compute MIS weights
    if weighting == "balance":
        # Balance heuristic: w_i = p(x) / (sum_j n_j q_j(x))
        log_mixture = jax.scipy.special.logsumexp(
            log_proposals + jnp.log(n_samples_per_proposal), axis=0
        )
        log_weights = log_target - log_mixture
    else:  # maximum
        # Maximum heuristic: w_i = p(x) / max_j q_j(x)
        log_max_proposal = jnp.max(log_proposals, axis=0)
        log_weights = log_target - log_max_proposal

    # Normalize
    weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

    return samples, weights


@jaxtyped(typechecker=beartype)
def adaptive_importance_sampling(
    key: PRNGKeyArray,
    log_target_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    initial_mean: Float[Array, " dim"],
    initial_cov: Float[Array, "dim dim"],
    n_samples_per_iter: int = 100,
    n_iterations: int = 10,
    adaptation_rate: float = 0.5,
) -> tuple[Float[Array, "n_total dim"], Float[Array, " n_total"]]:
    """Adaptive importance sampling with Gaussian proposal.

    Iteratively adapts proposal mean and covariance based on
    weighted samples.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    log_target_fn : Callable
        Log target density.
    initial_mean : Array
        Initial proposal mean.
    initial_cov : Array
        Initial proposal covariance.
    n_samples_per_iter : int
        Samples per iteration.
    n_iterations : int
        Number of adaptation iterations.
    adaptation_rate : float
        Rate of adaptation (0 to 1).

    Returns
    -------
    samples : Array
        All samples across iterations.
    weights : Array
        Normalized importance weights.
    """
    dim = initial_mean.shape[0]

    def iteration(carry, key_i):
        mean, cov = carry

        # Sample from current proposal
        samples = jax.random.multivariate_normal(
            key_i, mean, cov, shape=(n_samples_per_iter,)
        )

        # Compute weights
        log_target = jax.vmap(log_target_fn)(samples)

        def log_mvn(x):
            diff = x - mean
            sign, logdet = jnp.linalg.slogdet(cov)
            mahal = jnp.dot(diff, jnp.linalg.solve(cov, diff))
            return -0.5 * (dim * jnp.log(2 * jnp.pi) + logdet + mahal)

        log_proposal = jax.vmap(log_mvn)(samples)
        log_weights = log_target - log_proposal

        # Normalize weights
        weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

        # Update proposal parameters
        new_mean = (1 - adaptation_rate) * mean + adaptation_rate * jnp.sum(
            weights[:, None] * samples, axis=0
        )
        diff = samples - new_mean
        new_cov = (1 - adaptation_rate) * cov + adaptation_rate * jnp.sum(
            weights[:, None, None] * jnp.einsum("ij,ik->ijk", diff, diff), axis=0
        )
        # Ensure positive definite
        new_cov = 0.5 * (new_cov + new_cov.T) + 1e-6 * jnp.eye(dim)

        return (new_mean, new_cov), (samples, log_weights)

    keys = jax.random.split(key, n_iterations)
    _, (all_samples, all_log_weights) = jax.lax.scan(
        iteration, (initial_mean, initial_cov), keys
    )

    # Flatten
    samples = all_samples.reshape(-1, dim)
    log_weights = all_log_weights.reshape(-1)
    weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))

    return samples, weights


@jaxtyped(typechecker=beartype)
def compute_ess_is(
    log_weights: Float[Array, " n"],
) -> Float[Array, ""]:
    """Compute effective sample size for importance sampling.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights.

    Parameters
    ----------
    log_weights : Array
        Log importance weights (unnormalized).

    Returns
    -------
    ess : float
        Effective sample size.
    """
    log_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jnp.exp(-jax.scipy.special.logsumexp(2 * log_normalized))


@jaxtyped(typechecker=beartype)
def compute_is_diagnostics(
    log_weights: Float[Array, " n"],
) -> dict:
    """Compute importance sampling diagnostics.

    Parameters
    ----------
    log_weights : Array
        Log importance weights.

    Returns
    -------
    diagnostics : dict
        Dictionary with ESS, max weight, entropy, etc.
    """
    n = log_weights.shape[0]
    log_sum = jax.scipy.special.logsumexp(log_weights)
    log_normalized = log_weights - log_sum

    # Normalized weights
    weights = jnp.exp(log_normalized)

    # ESS
    ess = jnp.exp(-jax.scipy.special.logsumexp(2 * log_normalized))

    # Maximum weight
    max_weight = jnp.max(weights)

    # Entropy (normalized by max entropy)
    entropy = -jnp.sum(weights * log_normalized)
    max_entropy = jnp.log(n)
    normalized_entropy = entropy / max_entropy

    # Coefficient of variation of weights
    weight_mean = 1.0 / n
    weight_std = jnp.std(weights)
    cv = weight_std / weight_mean

    return {
        "ess": ess,
        "ess_ratio": ess / n,
        "max_weight": max_weight,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "cv": cv,
        "log_normalizing_constant": log_sum - jnp.log(n),
    }
