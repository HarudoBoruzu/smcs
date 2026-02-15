"""Adaptive Sequential Monte Carlo (ASMC).

Adaptive SMC automatically tunes algorithm parameters such as
the tempering schedule and mutation kernel parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess

if TYPE_CHECKING:
    pass

__all__ = [
    "AdaptiveState",
    "adaptive_step",
    "run_adaptive_smc",
    "find_next_temperature",
]


@chex.dataclass(frozen=True)
class AdaptiveState(SMCState):
    """State for Adaptive SMC.

    Attributes
    ----------
    particles : Array
        Particle positions.
    log_weights : Array
        Log-weights.
    temperature : Array
        Current temperature (0 to 1).
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Accumulated log-likelihood.
    step : Array
        Current step.
    """

    temperature: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def find_next_temperature(
    log_likelihood_diffs: Float[Array, " n_particles"],
    current_temp: Float[Array, ""],
    target_ess_ratio: float = 0.5,
    max_temp: float = 1.0,
) -> Float[Array, ""]:
    """Find the next temperature using bisection to achieve target ESS.

    Parameters
    ----------
    log_likelihood_diffs : Array
        Log-likelihood differences for each particle.
    current_temp : float
        Current temperature.
    target_ess_ratio : float
        Target ESS ratio (0 to 1).
    max_temp : float
        Maximum temperature.

    Returns
    -------
    next_temp : float
        Next temperature.
    """
    n_particles = log_likelihood_diffs.shape[0]
    target_ess = target_ess_ratio * n_particles

    def compute_ess_at_temp(delta_temp):
        log_weights = delta_temp * log_likelihood_diffs
        log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
        return jnp.exp(-jax.scipy.special.logsumexp(2 * log_weights_norm))

    # Bisection search
    def bisect_step(carry, _):
        low, high = carry
        mid = (low + high) / 2
        ess_mid = compute_ess_at_temp(mid - current_temp)
        new_low = jnp.where(ess_mid > target_ess, mid, low)
        new_high = jnp.where(ess_mid > target_ess, high, mid)
        return (new_low, new_high), None

    (final_low, final_high), _ = jax.lax.scan(
        bisect_step, (current_temp, max_temp), jnp.arange(50)
    )

    next_temp = (final_low + final_high) / 2
    return jnp.minimum(next_temp, max_temp)


@jaxtyped(typechecker=beartype)
def adaptive_step(
    key: PRNGKeyArray,
    state: AdaptiveState,
    log_target_fn: Callable[[Float[Array, " state_dim"]], Float[Array, ""]],
    log_prior_fn: Callable[[Float[Array, " state_dim"]], Float[Array, ""]],
    mutation_fn: Callable[[PRNGKeyArray, Float[Array, " state_dim"]], Float[Array, " state_dim"]],
    target_ess_ratio: float = 0.5,
    n_mcmc_steps: int = 5,
) -> tuple[AdaptiveState, SMCInfo]:
    """Perform one step of Adaptive SMC.

    Automatically adapts the tempering schedule based on ESS.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : AdaptiveState
        Current state.
    log_target_fn : Callable
        Log-target density function (e.g., posterior).
    log_prior_fn : Callable
        Log-prior density function.
    mutation_fn : Callable
        MCMC mutation kernel.
    target_ess_ratio : float
        Target ESS ratio for temperature adaptation.
    n_mcmc_steps : int
        Number of MCMC mutation steps.

    Returns
    -------
    new_state : AdaptiveState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, mutate_key = jax.random.split(key, 3)

    # Compute log-likelihoods (target - prior)
    log_likelihoods = jax.vmap(lambda x: log_target_fn(x) - log_prior_fn(x))(
        state.particles
    )

    # Find next temperature adaptively
    next_temp = find_next_temperature(
        log_likelihoods,
        state.temperature,
        target_ess_ratio,
    )

    # Compute incremental weights
    delta_temp = next_temp - state.temperature
    log_increments = delta_temp * log_likelihoods
    new_log_weights = state.log_weights + log_increments

    # Compute ESS
    ess = compute_ess(new_log_weights)

    # Resample
    ancestors = resample(resample_key, new_log_weights, method="systematic")
    resampled_particles = state.particles[ancestors]

    # MCMC mutation
    def mutate_particle(key_i, particle):
        def mcmc_step(carry, _):
            x, k = carry
            k, step_key = jax.random.split(k)
            x_new = mutation_fn(step_key, x)

            # Accept/reject
            log_alpha = (
                next_temp * (log_target_fn(x_new) - log_target_fn(x))
                + (1 - next_temp) * (log_prior_fn(x_new) - log_prior_fn(x))
            )
            k, accept_key = jax.random.split(k)
            accept = jnp.log(jax.random.uniform(accept_key)) < log_alpha
            x_out = jnp.where(accept, x_new, x)
            return (x_out, k), accept

        (final_x, _), accepts = jax.lax.scan(
            mcmc_step, (particle, key_i), jnp.arange(n_mcmc_steps)
        )
        return final_x, jnp.mean(accepts)

    mutate_keys = jax.random.split(mutate_key, n_particles)
    mutated_particles, acceptance_rates = jax.vmap(mutate_particle)(
        mutate_keys, resampled_particles
    )

    # Log-likelihood increment
    log_likelihood_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(state.log_weights)

    new_state = AdaptiveState(
        particles=mutated_particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        temperature=next_temp,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=jnp.array(True),
        acceptance_rate=jnp.mean(acceptance_rates),
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_adaptive_smc(
    key: PRNGKeyArray,
    log_target_fn: Callable[[Float[Array, " state_dim"]], Float[Array, ""]],
    log_prior_fn: Callable[[Float[Array, " state_dim"]], Float[Array, ""]],
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, " state_dim"]],
    mutation_fn: Callable[[PRNGKeyArray, Float[Array, " state_dim"]], Float[Array, " state_dim"]],
    n_particles: int = 1000,
    target_ess_ratio: float = 0.5,
    n_mcmc_steps: int = 5,
    max_steps: int = 100,
) -> tuple[AdaptiveState, SMCInfo]:
    """Run Adaptive SMC until temperature reaches 1.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    log_target_fn : Callable
        Log-target (posterior) density.
    log_prior_fn : Callable
        Log-prior density.
    prior_sample_fn : Callable
        Function to sample from prior.
    mutation_fn : Callable
        MCMC mutation kernel.
    n_particles : int
        Number of particles.
    target_ess_ratio : float
        Target ESS ratio.
    n_mcmc_steps : int
        MCMC steps per SMC step.
    max_steps : int
        Maximum number of SMC steps.

    Returns
    -------
    final_state : AdaptiveState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize from prior
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(prior_sample_fn)(init_keys)

    particles.shape[1]

    initial_state = AdaptiveState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        temperature=jnp.array(0.0),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def cond_fn(carry):
        state, _, step = carry
        return (state.temperature < 1.0) & (step < max_steps)

    def body_fn(carry):
        state, key, step = carry
        key, step_key = jax.random.split(key)
        new_state, info = adaptive_step(
            step_key,
            state,
            log_target_fn,
            log_prior_fn,
            mutation_fn,
            target_ess_ratio,
            n_mcmc_steps,
        )
        return new_state, key, step + 1

    final_state, _, n_steps = jax.lax.while_loop(
        cond_fn, body_fn, (initial_state, key, 0)
    )

    # Create combined info (simplified for while_loop)
    combined_info = SMCInfo(
        ess=compute_ess(final_state.log_weights),
        resampled=jnp.array(True),
        acceptance_rate=None,
    )

    return final_state, combined_info
