"""SMC Samplers for static parameter estimation.

SMC Samplers use a sequence of bridging distributions to sample
from complex target distributions, useful for Bayesian inference.
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, normalize_log_weights

__all__ = [
    "SMCSamplersState",
    "smc_samplers_step",
    "run_smc_samplers",
]


@chex.dataclass(frozen=True)
class SMCSamplersState(SMCState):
    """State for SMC Samplers.

    Attributes
    ----------
    particles : Array
        Parameter particles.
    log_weights : Array
        Log-weights.
    temperature : Array
        Current temperature in [0, 1].
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Accumulated log normalizing constant.
    step : Array
        Current step.
    """

    temperature: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def smc_samplers_step(
    key: PRNGKeyArray,
    state: SMCSamplersState,
    next_temperature: Float[Array, ""],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    mcmc_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        tuple[Float[Array, ...], Float[Array, ""]],
    ],
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[SMCSamplersState, SMCInfo]:
    """Perform one step of SMC Samplers.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : SMCSamplersState
        Current state.
    next_temperature : float
        Next temperature.
    log_likelihood_fn : Callable
        Function computing log-likelihood for a particle.
    mcmc_kernel : Callable
        MCMC transition kernel (key, particle, temperature) -> (new_particle, accept_rate).
    n_mcmc_steps : int
        Number of MCMC steps.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : SMCSamplersState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, mcmc_key = jax.random.split(key, 3)

    # Compute log-likelihoods
    log_liks = jax.vmap(log_likelihood_fn)(state.particles)

    # Compute incremental weights
    delta_temp = next_temperature - state.temperature
    log_increments = delta_temp * log_liks
    new_log_weights = state.log_weights + log_increments

    # Compute ESS and check if resampling needed
    ess = compute_ess(new_log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    # Resample if needed
    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, new_log_weights, method="systematic"),
        lambda: jnp.arange(n_particles),
    )
    resampled_particles = state.particles[ancestors]

    # Reset weights after resampling
    post_resample_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        normalize_log_weights(new_log_weights),
    )

    # MCMC mutation at current temperature
    def mutate_one(key_i, particle):
        def mcmc_step(carry, _):
            x, k, total_accept = carry
            k, step_key = jax.random.split(k)
            x_new, accept = mcmc_kernel(step_key, x, next_temperature)
            return (x_new, k, total_accept + accept), None

        init_carry = (particle, key_i, jnp.array(0.0))
        (final_x, _, total_accept), _ = jax.lax.scan(
            mcmc_step, init_carry, jnp.arange(n_mcmc_steps)
        )
        return final_x, total_accept / n_mcmc_steps

    mcmc_keys = jax.random.split(mcmc_key, n_particles)
    mutated_particles, acceptance_rates = jax.vmap(mutate_one)(
        mcmc_keys, resampled_particles
    )

    # Log normalizing constant increment
    log_nc_increment = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(
        n_particles
    )

    new_state = SMCSamplersState(
        particles=mutated_particles,
        log_weights=post_resample_weights,
        temperature=next_temperature,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_nc_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=jnp.mean(acceptance_rates),
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_smc_samplers(
    key: PRNGKeyArray,
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    mcmc_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        tuple[Float[Array, ...], Float[Array, ""]],
    ],
    n_particles: int = 1000,
    temperatures: Float[Array, " n_temps"] | None = None,
    n_temperatures: int = 10,
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[SMCSamplersState, SMCInfo]:
    """Run SMC Samplers with a tempering schedule.

    Samples from pi(theta) ∝ prior(theta) * likelihood(theta)^temperature
    with temperature going from 0 to 1.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    prior_sample_fn : Callable
        Function to sample from prior.
    log_likelihood_fn : Callable
        Log-likelihood function.
    mcmc_kernel : Callable
        MCMC transition kernel.
    n_particles : int
        Number of particles.
    temperatures : Array, optional
        Temperature schedule. If None, uses linear schedule.
    n_temperatures : int
        Number of temperatures if schedule not provided.
    n_mcmc_steps : int
        MCMC steps per temperature.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    final_state : SMCSamplersState
        Final state with posterior samples.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize from prior
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(prior_sample_fn)(init_keys)

    # Default temperature schedule
    if temperatures is None:
        temperatures = jnp.linspace(0.0, 1.0, n_temperatures + 1)[1:]

    initial_state = SMCSamplersState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        temperature=jnp.array(0.0),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        temp, step_key = inputs
        new_state, info = smc_samplers_step(
            step_key,
            state,
            temp,
            log_likelihood_fn,
            mcmc_kernel,
            n_mcmc_steps,
            ess_threshold,
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
        acceptance_rate=infos.acceptance_rate,
    )

    return final_state, combined_info
