"""Regularized Particle Filter (Kernel Smoothing).

The Regularized Particle Filter applies kernel smoothing to particles
after resampling to reduce particle degeneracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "RegularizedState",
    "regularized_step",
    "run_regularized_filter",
    "compute_kernel_bandwidth",
]


@chex.dataclass(frozen=True)
class RegularizedState(SMCState):
    """State for Regularized Particle Filter.

    Inherits all fields from SMCState.
    """

    pass


@jaxtyped(typechecker=beartype)
def compute_kernel_bandwidth(
    particles: Float[Array, "n_particles state_dim"],
    log_weights: Float[Array, " n_particles"],
) -> Float[Array, " state_dim"]:
    """Compute optimal kernel bandwidth using Silverman's rule.

    h = (4 / (d + 2))^(1/(d+4)) * n^(-1/(d+4)) * sigma

    Parameters
    ----------
    particles : Array
        Particle positions.
    log_weights : Array
        Log-weights.

    Returns
    -------
    bandwidth : Array
        Bandwidth for each dimension.
    """
    n_particles, state_dim = particles.shape
    weights = jnp.exp(normalize_log_weights(log_weights))

    # Weighted mean and std
    mean = jnp.sum(weights[:, None] * particles, axis=0)
    var = jnp.sum(weights[:, None] * (particles - mean) ** 2, axis=0)
    std = jnp.sqrt(var + 1e-10)

    # Silverman's rule of thumb
    factor = (4.0 / (state_dim + 2)) ** (1.0 / (state_dim + 4))
    bandwidth = factor * (n_particles ** (-1.0 / (state_dim + 4))) * std

    return bandwidth


@jaxtyped(typechecker=beartype)
def regularized_step(
    key: PRNGKeyArray,
    state: RegularizedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    ess_threshold: float = 0.5,
    kernel_scale: float = 1.0,
) -> tuple[RegularizedState, SMCInfo]:
    """Perform one step of the Regularized Particle Filter.

    After resampling, particles are jittered using a Gaussian kernel
    to prevent particle collapse.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : RegularizedState
        Current filter state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    ess_threshold : float
        ESS threshold for resampling.
    kernel_scale : float
        Scaling factor for kernel bandwidth.

    Returns
    -------
    new_state : RegularizedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    state_dim = state.particles.shape[1]
    key, resample_key, jitter_key, propagate_key = jax.random.split(key, 4)

    # Compute ESS and resample if needed
    ess = compute_ess(state.log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    log_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        state.log_weights,
    )

    # Compute bandwidth before resampling
    bandwidth = compute_kernel_bandwidth(state.particles, state.log_weights)
    bandwidth = kernel_scale * bandwidth

    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, state.log_weights, method="systematic"),
        lambda: jnp.arange(n_particles),
    )
    resampled_particles = state.particles[ancestors]

    # Apply kernel jittering after resampling
    jitter = jax.random.normal(jitter_key, (n_particles, state_dim))
    jittered_particles = jax.lax.cond(
        do_resample,
        lambda: resampled_particles + jitter * bandwidth,
        lambda: resampled_particles,
    )

    # Propagate through transition
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    new_particles = jax.vmap(propagate_one)(propagate_keys, jittered_particles)

    # Compute weights
    def compute_log_likelihood(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return jnp.squeeze(emit_dist.log_prob(observation))

    log_increments = jax.vmap(compute_log_likelihood)(new_particles)
    new_log_weights = log_weights + log_increments

    # Update log-likelihood
    log_likelihood_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)
    new_log_likelihood = state.log_likelihood + log_likelihood_increment

    new_state = RegularizedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        ancestors=ancestors,
        log_likelihood=new_log_likelihood,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_regularized_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    kernel_scale: float = 1.0,
) -> tuple[RegularizedState, SMCInfo]:
    """Run the Regularized Particle Filter on observations.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations of shape (n_steps, obs_dim).
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold for resampling.
    kernel_scale : float
        Scaling factor for kernel bandwidth.

    Returns
    -------
    final_state : RegularizedState
        Final filter state.
    info : SMCInfo
        Combined information from all steps.
    """
    key, init_key = jax.random.split(key)

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    initial_state = RegularizedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = regularized_step(
            step_key, state, obs, model, params, ess_threshold, kernel_scale
        )
        return (new_state, step_key), info

    n_steps = observations.shape[0]
    step_keys = jax.random.split(key, n_steps)

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        (observations, step_keys),
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=None,
    )

    return final_state, combined_info
