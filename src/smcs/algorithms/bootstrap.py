"""Bootstrap Particle Filter implementation.

The Bootstrap Particle Filter (Gordon et al., 1993) is the foundational
SMC algorithm using the transition distribution as the proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax, vmap
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import ResamplingMethod, resample
from smcs.core.weights import compute_ess

if TYPE_CHECKING:
    from smcs.models.base import ModelParams, StateSpaceModel

__all__ = [
    "bootstrap_step",
    "run_bootstrap_filter",
    "initialize_particles",
]


@jaxtyped(typechecker=beartype)
def initialize_particles(
    key: PRNGKeyArray,
    model: "StateSpaceModel",
    params: "ModelParams",
    n_particles: int,
) -> SMCState:
    """Initialize particles from the initial distribution.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    model : StateSpaceModel
        State space model.
    params : ModelParams
        Model parameters.
    n_particles : int
        Number of particles.

    Returns
    -------
    state : SMCState
        Initial SMC state.
    """
    keys = jax.random.split(key, n_particles)
    init_dist = model.initial_distribution(params)

    # Sample initial particles
    particles = vmap(init_dist.sample)(keys)

    # Reshape if needed (ensure 2D)
    if particles.ndim == 1:
        particles = particles[:, None]

    return SMCState(
        particles=particles,
        log_weights=jnp.zeros(n_particles),
        ancestors=jnp.arange(n_particles),
        log_likelihood=0.0,
        step=0,
    )


@jaxtyped(typechecker=beartype)
def bootstrap_step(
    key: PRNGKeyArray,
    state: SMCState,
    observation: Float[Array, "..."],
    model: "StateSpaceModel",
    params: "ModelParams",
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[SMCState, SMCInfo]:
    """Perform one step of the Bootstrap Particle Filter.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : SMCState
        Current SMC state.
    observation : Array
        Current observation y_t.
    model : StateSpaceModel
        State space model.
    params : ModelParams
        Model parameters.
    ess_threshold : float
        ESS/N ratio threshold for resampling (default 0.5).
    resampling_method : str
        Resampling method to use.

    Returns
    -------
    new_state : SMCState
        Updated SMC state.
    info : SMCInfo
        Diagnostic information.
    """
    n_particles = state.n_particles
    key, resample_key, propagate_key = jax.random.split(key, 3)

    # Ensure observation is array
    observation = jnp.atleast_1d(observation)

    # Compute ESS and decide on resampling
    ess = compute_ess(state.log_weights)
    should_resample = ess < ess_threshold * n_particles

    # Conditional resampling
    def do_resample(args):
        particles, log_weights, rkey = args
        indices = resample(rkey, log_weights, resampling_method)
        return particles[indices], jnp.zeros(n_particles), indices

    def no_resample(args):
        particles, log_weights, _ = args
        return particles, log_weights, jnp.arange(n_particles)

    particles, log_weights, ancestors = lax.cond(
        should_resample,
        do_resample,
        no_resample,
        (state.particles, state.log_weights, resample_key),
    )

    # Propagate particles through transition
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_single(key_particle):
        key, particle = key_particle
        trans_dist = model.transition_distribution(params, particle, state.step)
        new_particle = trans_dist.sample(key)
        # Ensure 1D output
        return jnp.atleast_1d(new_particle)

    new_particles = vmap(propagate_single)((propagate_keys, particles))

    # Update weights with observation likelihood
    def compute_log_likelihood(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return emit_dist.log_prob(observation)

    log_likelihoods = vmap(compute_log_likelihood)(new_particles)
    new_log_weights = log_weights + log_likelihoods

    # Compute marginal likelihood increment
    log_likelihood_increment = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(n_particles)

    new_state = SMCState(
        particles=new_particles,
        log_weights=new_log_weights,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=should_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_bootstrap_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: "StateSpaceModel",
    params: "ModelParams",
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[SMCState, SMCInfo]:
    """Run the Bootstrap Particle Filter on a sequence of observations.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    observations : Array
        Observation sequence [n_timesteps, obs_dim].
    model : StateSpaceModel
        State space model.
    params : ModelParams
        Model parameters.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS/N ratio threshold for resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    final_state : SMCState
        Final SMC state after processing all observations.
    final_info : SMCInfo
        Aggregated diagnostic information.
    """
    key, init_key = jax.random.split(key)

    # Initialize particles
    state = initialize_particles(init_key, model, params, n_particles)

    # Scan function
    def scan_fn(carry, inputs):
        current_state, step_key = carry
        obs = inputs
        step_key, next_key = jax.random.split(step_key)

        new_state, info = bootstrap_step(
            step_key,
            current_state,
            obs,
            model,
            params,
            ess_threshold,
            resampling_method,
        )
        return (new_state, next_key), info

    # Run scan over observations
    step_keys = jax.random.split(key, observations.shape[0] + 1)
    (final_state, _), info_history = lax.scan(
        scan_fn,
        (state, step_keys[0]),
        observations,
    )

    # Aggregate info (take last ESS, count resamples)
    # Note: info_history is a pytree of stacked SMCInfo
    return final_state, info_history
