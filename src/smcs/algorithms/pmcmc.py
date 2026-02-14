"""Particle MCMC algorithms.

This module implements Particle Marginal Metropolis-Hastings (PMMH)
and Particle Gibbs (PG) for batch parameter learning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, Bool, Float, PRNGKeyArray, jaxtyped

from smcs.algorithms.bootstrap import run_bootstrap_filter

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "PMMHState",
    "PMMHResult",
    "pmmh_step",
    "run_pmmh",
]


@chex.dataclass(frozen=True)
class PMMHState:
    """State for PMMH algorithm.

    Attributes
    ----------
    params : Array
        Current parameter vector.
    log_likelihood : float
        Log marginal likelihood at current params.
    log_prior : float
        Log prior at current params.
    """

    params: Float[Array, " param_dim"]
    log_likelihood: Float[Array, ""]
    log_prior: Float[Array, ""]


class PMMHResult(NamedTuple):
    """Result of PMMH run.

    Attributes
    ----------
    samples : Array
        Parameter samples [n_samples, param_dim].
    log_likelihoods : Array
        Log likelihoods at each sample.
    acceptance_rate : float
        Overall acceptance rate.
    """

    samples: Float[Array, "n_samples param_dim"]
    log_likelihoods: Float[Array, " n_samples"]
    acceptance_rate: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def pmmh_step(
    key: PRNGKeyArray,
    state: PMMHState,
    observations: Float[Array, "n_timesteps ..."],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    log_prior_fn: Callable,
    proposal_fn: Callable,
    n_particles: int = 100,
) -> tuple[PMMHState, Bool[Array, ""]]:
    """Perform one step of Particle Marginal Metropolis-Hastings.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : PMMHState
        Current PMMH state.
    observations : Array
        Full observation sequence.
    model : StateSpaceModel
        State space model.
    param_to_model_params : callable
        Converts parameter vector to ModelParams.
    log_prior_fn : callable
        Function (params) -> log prior.
    proposal_fn : callable
        Function (key, current_params) -> proposed_params.
    n_particles : int
        Number of particles for likelihood estimation.

    Returns
    -------
    new_state : PMMHState
        Updated PMMH state.
    accepted : bool
        Whether the proposal was accepted.
    """
    key, proposal_key, pf_key, accept_key = jax.random.split(key, 4)

    # Propose new parameters
    proposed_params = proposal_fn(proposal_key, state.params)
    proposed_log_prior = log_prior_fn(proposed_params)

    # Run particle filter for proposed parameters
    proposed_model_params = param_to_model_params(proposed_params)
    pf_result, _ = run_bootstrap_filter(
        pf_key,
        observations,
        model,
        proposed_model_params,
        n_particles=n_particles,
    )
    proposed_log_likelihood = pf_result.log_likelihood

    # Compute acceptance probability
    # log alpha = log p(y|theta') + log p(theta') - log p(y|theta) - log p(theta)
    log_alpha = (
        proposed_log_likelihood
        + proposed_log_prior
        - state.log_likelihood
        - state.log_prior
    )

    # Accept/reject
    log_u = jnp.log(jax.random.uniform(accept_key))
    accepted = log_u < log_alpha

    new_state = lax.cond(
        accepted,
        lambda _: PMMHState(
            params=proposed_params,
            log_likelihood=proposed_log_likelihood,
            log_prior=proposed_log_prior,
        ),
        lambda _: state,
        None,
    )

    return new_state, accepted


@jaxtyped(typechecker=beartype)
def run_pmmh(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    log_prior_fn: Callable,
    proposal_fn: Callable,
    initial_params: Float[Array, " param_dim"],
    n_samples: int = 1000,
    n_burnin: int = 100,
    n_particles: int = 100,
) -> PMMHResult:
    """Run Particle Marginal Metropolis-Hastings.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    observations : Array
        Observation sequence.
    model : StateSpaceModel
        State space model.
    param_to_model_params : callable
        Converts parameter vector to ModelParams.
    log_prior_fn : callable
        Function (params) -> log prior.
    proposal_fn : callable
        Function (key, current_params) -> proposed_params.
    initial_params : Array
        Initial parameter values.
    n_samples : int
        Number of samples to collect after burn-in.
    n_burnin : int
        Number of burn-in iterations.
    n_particles : int
        Number of particles for likelihood estimation.

    Returns
    -------
    result : PMMHResult
        PMMH samples and diagnostics.
    """
    key, init_pf_key = jax.random.split(key)

    # Initialize: run particle filter for initial params
    initial_model_params = param_to_model_params(initial_params)
    init_pf_result, _ = run_bootstrap_filter(
        init_pf_key,
        observations,
        model,
        initial_model_params,
        n_particles=n_particles,
    )

    init_state = PMMHState(
        params=initial_params,
        log_likelihood=init_pf_result.log_likelihood,
        log_prior=log_prior_fn(initial_params),
    )

    total_iterations = n_burnin + n_samples

    # Scan function
    def scan_fn(carry, _):
        state, step_key, n_accepted = carry
        step_key, next_key = jax.random.split(step_key)

        new_state, accepted = pmmh_step(
            step_key,
            state,
            observations,
            model,
            param_to_model_params,
            log_prior_fn,
            proposal_fn,
            n_particles,
        )
        n_accepted = n_accepted + accepted.astype(jnp.int32)

        return (new_state, next_key, n_accepted), (new_state.params, new_state.log_likelihood)

    step_keys = jax.random.split(key, total_iterations + 1)
    (final_state, _, total_accepted), (all_samples, all_log_liks) = lax.scan(
        scan_fn,
        (init_state, step_keys[0], 0),
        None,
        length=total_iterations,
    )

    # Remove burn-in
    samples = all_samples[n_burnin:]
    log_likelihoods = all_log_liks[n_burnin:]

    acceptance_rate = total_accepted / total_iterations

    return PMMHResult(
        samples=samples,
        log_likelihoods=log_likelihoods,
        acceptance_rate=jnp.asarray(acceptance_rate, dtype=jnp.float32),
    )


def random_walk_proposal(
    scale: Float[Array, " param_dim"] | float,
) -> Callable:
    """Create a random walk proposal function.

    Parameters
    ----------
    scale : Array or float
        Standard deviation(s) for the random walk.

    Returns
    -------
    proposal_fn : callable
        Function (key, params) -> proposed_params.
    """
    scale = jnp.atleast_1d(scale)

    def proposal_fn(
        key: PRNGKeyArray, params: Float[Array, " param_dim"]
    ) -> Float[Array, " param_dim"]:
        noise = jax.random.normal(key, shape=params.shape)
        return params + scale * noise

    return proposal_fn
