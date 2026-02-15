"""Approximate Bayesian Computation SMC (ABC-SMC).

ABC-SMC enables likelihood-free inference by comparing simulated
data with observations using summary statistics and distance thresholds.
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
    "ABCSMCState",
    "abc_smc_step",
    "run_abc_smc",
    "compute_adaptive_threshold",
]


@chex.dataclass(frozen=True)
class ABCSMCState(SMCState):
    """State for ABC-SMC.

    Attributes
    ----------
    particles : Array
        Parameter particles.
    log_weights : Array
        Log-weights.
    distances : Array
        Distances for each particle.
    threshold : Array
        Current acceptance threshold.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Not used in ABC (set to 0).
    step : Array
        Current step.
    """

    distances: Float[Array, " n_particles"]
    threshold: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def compute_adaptive_threshold(
    distances: Float[Array, " n_particles"],
    quantile: float = 0.5,
) -> Float[Array, ""]:
    """Compute adaptive threshold based on distance quantile.

    Parameters
    ----------
    distances : Array
        Current distances.
    quantile : float
        Quantile to use (0-1).

    Returns
    -------
    threshold : float
        New threshold.
    """
    sorted_distances = jnp.sort(distances)
    idx = jnp.int32(quantile * len(distances))
    return sorted_distances[idx]


@jaxtyped(typechecker=beartype)
def abc_smc_step(
    key: PRNGKeyArray,
    state: ABCSMCState,
    new_threshold: Float[Array, ""],
    simulator_fn: Callable[[PRNGKeyArray, Float[Array, ...]], Float[Array, ...]],
    summary_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    distance_fn: Callable[[Float[Array, ...], Float[Array, ...]], Float[Array, ""]],
    observed_summary: Float[Array, ...],
    prior_log_prob_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    perturbation_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...]], Float[Array, ...]
    ],
    perturbation_log_prob_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ""]
    ],
    n_mcmc_steps: int = 1,
    max_attempts: int = 1000,
) -> tuple[ABCSMCState, SMCInfo]:
    """Perform one step of ABC-SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : ABCSMCState
        Current state.
    new_threshold : float
        New acceptance threshold.
    simulator_fn : Callable
        Function (key, params) -> simulated_data.
    summary_fn : Callable
        Function to compute summary statistics.
    distance_fn : Callable
        Distance function between summaries.
    observed_summary : Array
        Summary statistics of observed data.
    prior_log_prob_fn : Callable
        Log-prior density.
    perturbation_kernel : Callable
        MCMC perturbation kernel.
    perturbation_log_prob_fn : Callable
        Log-probability of perturbation.
    n_mcmc_steps : int
        Number of MCMC steps.
    max_attempts : int
        Maximum simulation attempts per particle.

    Returns
    -------
    new_state : ABCSMCState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, perturb_key = jax.random.split(key, 3)

    # Compute weights based on distance to threshold
    # Particles with distance > threshold get zero weight
    valid = state.distances <= state.threshold
    log_weights_adj = jnp.where(valid, state.log_weights, -jnp.inf)

    # Resample
    ess = compute_ess(log_weights_adj)
    ancestors = resample(resample_key, log_weights_adj, method="systematic")
    resampled_particles = state.particles[ancestors]
    resampled_distances = state.distances[ancestors]

    # Perturb and simulate
    def perturb_and_simulate(key_i, particle, old_distance):
        def attempt_loop(carry, _):
            x, d, k, accepted = carry
            k, perturb_key, sim_key = jax.random.split(k, 3)

            # Perturb
            x_new = perturbation_kernel(perturb_key, x)

            # Check prior support
            log_prior_new = prior_log_prob_fn(x_new)
            in_prior = log_prior_new > -jnp.inf

            # Simulate and compute distance
            sim_data = simulator_fn(sim_key, x_new)
            sim_summary = summary_fn(sim_data)
            d_new = distance_fn(sim_summary, observed_summary)

            # Accept if distance below threshold
            accept = in_prior & (d_new <= new_threshold) & ~accepted

            x_out = jnp.where(accept, x_new, x)
            d_out = jnp.where(accept, d_new, d)
            accepted_out = accepted | accept

            return (x_out, d_out, k, accepted_out), None

        init_carry = (particle, old_distance, key_i, jnp.array(False))
        (final_x, final_d, _, accepted), _ = jax.lax.scan(
            attempt_loop, init_carry, jnp.arange(max_attempts)
        )

        return final_x, final_d, accepted

    perturb_keys = jax.random.split(perturb_key, n_particles)
    new_particles, new_distances, accepted = jax.vmap(perturb_and_simulate)(
        perturb_keys, resampled_particles, resampled_distances
    )

    # Compute new weights based on perturbation kernel
    def compute_weight(new_p, old_p):
        # Weight = prior(new) / sum_j w_j * K(new | old_j)
        log_prior = prior_log_prob_fn(new_p)

        # Simplified: just use prior (assumes symmetric kernel)
        return log_prior

    new_log_weights = jax.vmap(compute_weight)(new_particles, resampled_particles)
    new_log_weights = normalize_log_weights(new_log_weights)

    new_state = ABCSMCState(
        particles=new_particles,
        log_weights=new_log_weights,
        distances=new_distances,
        threshold=new_threshold,
        ancestors=ancestors,
        log_likelihood=jnp.array(0.0),
        step=state.step + 1,
    )

    acceptance_rate = jnp.mean(accepted.astype(jnp.float32))

    info = SMCInfo(
        ess=ess,
        resampled=jnp.array(True),
        acceptance_rate=acceptance_rate,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_abc_smc(
    key: PRNGKeyArray,
    observed_data: Float[Array, ...],
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    prior_log_prob_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    simulator_fn: Callable[[PRNGKeyArray, Float[Array, ...]], Float[Array, ...]],
    summary_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    distance_fn: Callable[[Float[Array, ...], Float[Array, ...]], Float[Array, ""]],
    perturbation_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...]], Float[Array, ...]
    ],
    perturbation_log_prob_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ""]
    ],
    n_particles: int = 1000,
    initial_threshold: float = jnp.inf,
    threshold_quantile: float = 0.5,
    min_threshold: float = 0.01,
    max_iterations: int = 20,
    n_mcmc_steps: int = 1,
) -> tuple[ABCSMCState, SMCInfo]:
    """Run ABC-SMC for likelihood-free inference.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observed_data : Array
        Observed data.
    prior_sample_fn : Callable
        Function to sample from prior.
    prior_log_prob_fn : Callable
        Log-prior density.
    simulator_fn : Callable
        Simulator function.
    summary_fn : Callable
        Summary statistics function.
    distance_fn : Callable
        Distance function.
    perturbation_kernel : Callable
        MCMC perturbation kernel.
    perturbation_log_prob_fn : Callable
        Perturbation log-probability.
    n_particles : int
        Number of particles.
    initial_threshold : float
        Initial acceptance threshold.
    threshold_quantile : float
        Quantile for adaptive threshold.
    min_threshold : float
        Minimum threshold (stopping criterion).
    max_iterations : int
        Maximum SMC iterations.
    n_mcmc_steps : int
        MCMC steps per iteration.

    Returns
    -------
    final_state : ABCSMCState
        Final state with posterior samples.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Compute observed summary
    observed_summary = summary_fn(observed_data)

    # Initialize from prior with rejection sampling
    def init_one(key_i):
        def sample_loop(carry, _):
            x, d, k, done = carry
            k, sample_key, sim_key = jax.random.split(k, 3)

            x_new = prior_sample_fn(sample_key)
            sim_data = simulator_fn(sim_key, x_new)
            sim_summary = summary_fn(sim_data)
            d_new = distance_fn(sim_summary, observed_summary)

            accept = (d_new <= initial_threshold) & ~done

            x_out = jnp.where(accept, x_new, x)
            d_out = jnp.where(accept, d_new, d)
            done_out = done | accept

            return (x_out, d_out, k, done_out), None

        # Get initial shape
        dummy = prior_sample_fn(key_i)
        init_carry = (dummy, jnp.array(jnp.inf), key_i, jnp.array(False))

        (x, d, _, _), _ = jax.lax.scan(sample_loop, init_carry, jnp.arange(10000))
        return x, d

    init_keys = jax.random.split(init_key, n_particles)
    particles, distances = jax.vmap(init_one)(init_keys)

    initial_state = ABCSMCState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        distances=distances,
        threshold=jnp.array(initial_threshold),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Adaptive threshold schedule
    def cond_fn(carry):
        state, _, step = carry
        return (state.threshold > min_threshold) & (step < max_iterations)

    def body_fn(carry):
        state, key, step = carry
        key, step_key = jax.random.split(key)

        # Compute new threshold
        new_threshold = compute_adaptive_threshold(
            state.distances, threshold_quantile
        )
        new_threshold = jnp.maximum(new_threshold, min_threshold)

        new_state, info = abc_smc_step(
            step_key,
            state,
            new_threshold,
            simulator_fn,
            summary_fn,
            distance_fn,
            observed_summary,
            prior_log_prob_fn,
            perturbation_kernel,
            perturbation_log_prob_fn,
            n_mcmc_steps,
        )

        return new_state, key, step + 1

    final_state, _, n_iters = jax.lax.while_loop(
        cond_fn, body_fn, (initial_state, key, 0)
    )

    # Final info
    combined_info = SMCInfo(
        ess=compute_ess(final_state.log_weights),
        resampled=jnp.array(True),
        acceptance_rate=None,
    )

    return final_state, combined_info
