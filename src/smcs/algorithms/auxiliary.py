"""Auxiliary Particle Filter implementation.

The Auxiliary Particle Filter (Pitt & Shephard, 1999) improves on the
Bootstrap filter by using information from the current observation to
guide particle propagation.
"""

from __future__ import annotations

from collections.abc import Callable
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
    "auxiliary_step",
    "run_auxiliary_filter",
]


@jaxtyped(typechecker=beartype)
def auxiliary_step(
    key: PRNGKeyArray,
    state: SMCState,
    observation: Float[Array, ...],
    model: StateSpaceModel,
    params: ModelParams,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
    predictive_likelihood_fn: Callable | None = None,
) -> tuple[SMCState, SMCInfo]:
    """Perform one step of the Auxiliary Particle Filter.

    The APF uses a two-stage resampling:
    1. First-stage weights based on predicted observation likelihood
    2. Second-stage adjustment after propagation

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
        ESS/N ratio threshold for resampling.
    resampling_method : str
        Resampling method to use.
    predictive_likelihood_fn : callable, optional
        Function to compute predictive likelihood p(y_t | x_{t-1}).
        If None, uses the emission distribution at the transition mean.

    Returns
    -------
    new_state : SMCState
        Updated SMC state.
    info : SMCInfo
        Diagnostic information.
    """
    n_particles = state.n_particles
    key, first_resample_key, propagate_key = jax.random.split(key, 3)

    observation = jnp.atleast_1d(observation)

    # Compute predictive likelihoods (first-stage weights)
    def compute_predictive_likelihood(particle):
        # Use transition mean as the predictive point
        trans_dist = model.transition_distribution(params, particle, state.step)
        # Get the mean of the transition distribution
        # For Gaussian, this is the loc parameter
        if hasattr(trans_dist, "loc"):
            mu = trans_dist.loc
        else:
            # Fallback: use current particle
            mu = particle

        mu = jnp.atleast_1d(mu)
        emit_dist = model.emission_distribution(params, mu, state.step + 1)
        return jnp.squeeze(emit_dist.log_prob(observation))

    if predictive_likelihood_fn is not None:
        log_predictive = vmap(lambda p: predictive_likelihood_fn(params, p, observation))(
            state.particles
        )
    else:
        log_predictive = vmap(compute_predictive_likelihood)(state.particles)

    # First-stage weights
    first_stage_log_weights = state.log_weights + log_predictive

    # ESS-based resampling
    ess = compute_ess(first_stage_log_weights)
    should_resample = ess < ess_threshold * n_particles

    def do_resample(args):
        particles, log_weights, log_pred, rkey = args
        indices = resample(rkey, log_weights + log_pred, resampling_method)
        return (
            particles[indices],
            jnp.zeros(n_particles),
            log_pred[indices],
            indices,
        )

    def no_resample(args):
        particles, log_weights, log_pred, _ = args
        return particles, log_weights, log_pred, jnp.arange(n_particles)

    particles, log_weights, resampled_log_pred, ancestors = lax.cond(
        should_resample,
        do_resample,
        no_resample,
        (state.particles, state.log_weights, log_predictive, first_resample_key),
    )

    # Propagate particles
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_single(key_particle):
        key, particle = key_particle
        trans_dist = model.transition_distribution(params, particle, state.step)
        new_particle = trans_dist.sample(key)
        return jnp.atleast_1d(new_particle)

    new_particles = vmap(propagate_single)((propagate_keys, particles))

    # Second-stage weight adjustment
    def compute_actual_likelihood(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return jnp.squeeze(emit_dist.log_prob(observation))

    log_actual = vmap(compute_actual_likelihood)(new_particles)

    # Adjustment: w_t ∝ p(y_t | x_t) / p(y_t | μ_t)
    new_log_weights = log_weights + log_actual - resampled_log_pred

    # Marginal likelihood increment
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
def run_auxiliary_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: StateSpaceModel,
    params: ModelParams,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[SMCState, SMCInfo]:
    """Run the Auxiliary Particle Filter on a sequence of observations.

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
    info_history : SMCInfo
        Stacked diagnostic information for all steps.
    """
    from smcs.algorithms.bootstrap import initialize_particles

    key, init_key = jax.random.split(key)

    # Initialize particles
    state = initialize_particles(init_key, model, params, n_particles)

    # Scan function
    def scan_fn(carry, inputs):
        current_state, step_key = carry
        obs = inputs
        step_key, next_key = jax.random.split(step_key)

        new_state, info = auxiliary_step(
            step_key,
            current_state,
            obs,
            model,
            params,
            ess_threshold,
            resampling_method,
        )
        return (new_state, next_key), info

    # Run scan
    step_keys = jax.random.split(key, observations.shape[0] + 1)
    (final_state, _), info_history = lax.scan(
        scan_fn,
        (state, step_keys[0]),
        observations,
    )

    return final_state, info_history
