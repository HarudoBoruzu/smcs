"""Annealed Importance Sampling (AIS).

AIS uses a sequence of intermediate distributions to estimate
normalizing constants and sample from complex distributions.
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.weights import compute_ess, log_mean_exp

__all__ = [
    "AISState",
    "ais_step",
    "run_ais",
    "estimate_log_normalizing_constant",
]


@chex.dataclass(frozen=True)
class AISState(SMCState):
    """State for Annealed Importance Sampling.

    Attributes
    ----------
    particles : Array
        Sample particles.
    log_weights : Array
        Log importance weights.
    temperature : Array
        Current temperature.
    ancestors : Array
        Not used in AIS (no resampling).
    log_likelihood : Array
        Accumulated log-weight sum for normalizing constant.
    step : Array
        Current step.
    """

    temperature: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def ais_step(
    key: PRNGKeyArray,
    state: AISState,
    next_temperature: Float[Array, ""],
    log_prior_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    transition_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        Float[Array, ...],
    ],
) -> tuple[AISState, SMCInfo]:
    """Perform one step of Annealed Importance Sampling.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : AISState
        Current state.
    next_temperature : float
        Next temperature.
    log_prior_fn : Callable
        Log-prior function.
    log_likelihood_fn : Callable
        Log-likelihood function.
    transition_kernel : Callable
        MCMC transition kernel that leaves the bridging distribution invariant.

    Returns
    -------
    new_state : AISState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]

    # Compute incremental weights
    log_liks = jax.vmap(log_likelihood_fn)(state.particles)
    delta_temp = next_temperature - state.temperature
    log_increments = delta_temp * log_liks
    new_log_weights = state.log_weights + log_increments

    # Apply MCMC transition at new temperature
    def apply_mcmc(key_i, particle):
        return transition_kernel(key_i, particle, next_temperature)

    transition_keys = jax.random.split(key, n_particles)
    new_particles = jax.vmap(apply_mcmc)(transition_keys, state.particles)

    # ESS (for monitoring)
    ess = compute_ess(new_log_weights)

    new_state = AISState(
        particles=new_particles,
        log_weights=new_log_weights,
        temperature=next_temperature,
        ancestors=jnp.arange(n_particles),
        log_likelihood=state.log_likelihood + jnp.mean(log_increments),
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=jnp.array(False),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_ais(
    key: PRNGKeyArray,
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_prior_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    transition_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        Float[Array, ...],
    ],
    n_particles: int = 1000,
    temperatures: Float[Array, " n_temps"] | None = None,
    n_temperatures: int = 100,
) -> tuple[AISState, SMCInfo]:
    """Run Annealed Importance Sampling.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    prior_sample_fn : Callable
        Function to sample from prior.
    log_prior_fn : Callable
        Log-prior function.
    log_likelihood_fn : Callable
        Log-likelihood function.
    transition_kernel : Callable
        MCMC transition kernel.
    n_particles : int
        Number of particles (chains).
    temperatures : Array, optional
        Temperature schedule from 0 to 1.
    n_temperatures : int
        Number of temperatures if schedule not provided.

    Returns
    -------
    final_state : AISState
        Final state with importance-weighted samples.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize from prior
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(prior_sample_fn)(init_keys)

    # Default temperature schedule (geometric spacing often works better)
    if temperatures is None:
        # Use geometric schedule: beta_k = (k/K)^5
        temperatures = jnp.power(
            jnp.linspace(0.0, 1.0, n_temperatures + 1)[1:], 5.0
        )

    initial_state = AISState(
        particles=particles,
        log_weights=jnp.zeros(n_particles),
        temperature=jnp.array(0.0),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        temp, step_key = inputs
        new_state, info = ais_step(
            step_key,
            state,
            temp,
            log_prior_fn,
            log_likelihood_fn,
            transition_kernel,
        )
        return (new_state, key), info

    step_keys = jax.random.split(key, len(temperatures))

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        (temperatures, step_keys),
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=None,
    )

    return final_state, combined_info


@jaxtyped(typechecker=beartype)
def estimate_log_normalizing_constant(
    state: AISState,
) -> Float[Array, ""]:
    """Estimate log normalizing constant from AIS state.

    Z_target / Z_prior = E[w] where w are importance weights.

    Parameters
    ----------
    state : AISState
        Final AIS state.

    Returns
    -------
    log_z : float
        Estimated log normalizing constant ratio.
    """
    return log_mean_exp(state.log_weights)
