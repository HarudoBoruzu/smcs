"""Storvik Filter for models with sufficient statistics.

The Storvik Filter (Storvik, 2002) enables efficient parameter learning
for models where the posterior has a known form given sufficient statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax, vmap
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo
from smcs.core.resampling import ResamplingMethod, resample
from smcs.core.weights import compute_ess

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "StorvikState",
    "storvik_step",
    "run_storvik_filter",
]


@chex.dataclass(frozen=True)
class StorvikState:
    """State for Storvik Filter with sufficient statistics.

    Attributes
    ----------
    state_particles : Array
        State particles [n_particles, state_dim].
    sufficient_stats : Array
        Sufficient statistics particles [n_particles, stats_dim].
    log_weights : Array
        Log-weights [n_particles].
    log_likelihood : float
        Cumulative log marginal likelihood.
    step : int
        Current time step.
    """

    state_particles: Float[Array, "n_particles state_dim"]
    sufficient_stats: Float[Array, "n_particles stats_dim"]
    log_weights: Float[Array, " n_particles"]
    log_likelihood: float
    step: int

    @property
    def n_particles(self) -> int:
        return self.state_particles.shape[0]


@jaxtyped(typechecker=beartype)
def storvik_step(
    key: PRNGKeyArray,
    state: StorvikState,
    observation: Float[Array, "..."],
    model: "StateSpaceModel",
    update_sufficient_stats: Callable,
    sample_params_from_stats: Callable,
    stats_to_model_params: Callable,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[StorvikState, SMCInfo]:
    """Perform one step of the Storvik Filter.

    The Storvik filter maintains sufficient statistics for the parameter
    posterior, allowing exact sampling of parameters given states.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : StorvikState
        Current Storvik state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State space model.
    update_sufficient_stats : callable
        Function (stats, state, obs) -> new_stats.
    sample_params_from_stats : callable
        Function (key, stats) -> param_sample.
    stats_to_model_params : callable
        Function (stats) -> ModelParams for transition/emission.
    ess_threshold : float
        ESS/N threshold for resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    new_state : StorvikState
        Updated Storvik state.
    info : SMCInfo
        Diagnostic information.
    """
    n_particles = state.n_particles
    key, resample_key, param_key, prop_key = jax.random.split(key, 4)

    observation = jnp.atleast_1d(observation)

    # ESS and resampling
    ess = compute_ess(state.log_weights)
    should_resample = ess < ess_threshold * n_particles

    def do_resample(args):
        s_parts, stats, log_w, rkey = args
        indices = resample(rkey, log_w, resampling_method)
        return s_parts[indices], stats[indices], jnp.zeros(n_particles)

    def no_resample(args):
        s_parts, stats, log_w, _ = args
        return s_parts, stats, log_w

    state_particles, sufficient_stats, log_weights = lax.cond(
        should_resample,
        do_resample,
        no_resample,
        (state.state_particles, state.sufficient_stats, state.log_weights, resample_key),
    )

    # Sample parameters from sufficient statistics
    param_keys = jax.random.split(param_key, n_particles)
    param_samples = vmap(sample_params_from_stats)(param_keys, sufficient_stats)

    # Propagate state particles
    prop_keys = jax.random.split(prop_key, n_particles)

    def propagate_single(inputs):
        key, state_particle, param_sample = inputs
        model_params = stats_to_model_params(param_sample)
        trans_dist = model.transition_distribution(model_params, state_particle, state.step)
        return jnp.atleast_1d(trans_dist.sample(key))

    new_state_particles = vmap(propagate_single)((prop_keys, state_particles, param_samples))

    # Update sufficient statistics with new state and observation
    def update_stats(stats, new_state, param_sample):
        return update_sufficient_stats(stats, new_state, observation, param_sample)

    new_sufficient_stats = vmap(update_stats)(sufficient_stats, new_state_particles, param_samples)

    # Update weights with observation likelihood
    def compute_log_likelihood(state_particle, param_sample):
        model_params = stats_to_model_params(param_sample)
        emit_dist = model.emission_distribution(model_params, state_particle, state.step + 1)
        return emit_dist.log_prob(observation)

    log_likelihoods = vmap(compute_log_likelihood)(new_state_particles, param_samples)
    new_log_weights = log_weights + log_likelihoods

    # Marginal likelihood increment
    log_likelihood_increment = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(n_particles)

    new_state = StorvikState(
        state_particles=new_state_particles,
        sufficient_stats=new_sufficient_stats,
        log_weights=new_log_weights,
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
def run_storvik_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: "StateSpaceModel",
    update_sufficient_stats: Callable,
    sample_params_from_stats: Callable,
    stats_to_model_params: Callable,
    initial_state_sampler: Callable,
    initial_stats: Float[Array, " stats_dim"],
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[StorvikState, SMCInfo]:
    """Run the Storvik Filter for joint state and parameter estimation.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    observations : Array
        Observation sequence.
    model : StateSpaceModel
        State space model.
    update_sufficient_stats : callable
        Updates sufficient statistics given new state/observation.
    sample_params_from_stats : callable
        Samples parameters from posterior given sufficient statistics.
    stats_to_model_params : callable
        Converts sufficient statistics to model parameters.
    initial_state_sampler : callable
        Function (key,) -> initial state sample.
    initial_stats : Array
        Initial sufficient statistics (same for all particles).
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS/N threshold for resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    final_state : StorvikState
        Final Storvik state.
    info_history : SMCInfo
        Stacked diagnostic information.
    """
    key, init_key = jax.random.split(key)
    state_keys = jax.random.split(init_key, n_particles)

    # Initialize state particles
    initial_states = vmap(initial_state_sampler)(state_keys)
    if initial_states.ndim == 1:
        initial_states = initial_states[:, None]

    # Initialize sufficient statistics (replicated)
    initial_sufficient_stats = jnp.tile(initial_stats[None, :], (n_particles, 1))

    init_state = StorvikState(
        state_particles=initial_states,
        sufficient_stats=initial_sufficient_stats,
        log_weights=jnp.zeros(n_particles),
        log_likelihood=0.0,
        step=0,
    )

    # Scan function
    def scan_fn(carry, obs):
        current_state, step_key = carry
        step_key, next_key = jax.random.split(step_key)

        new_state, info = storvik_step(
            step_key,
            current_state,
            obs,
            model,
            update_sufficient_stats,
            sample_params_from_stats,
            stats_to_model_params,
            ess_threshold,
            resampling_method,
        )
        return (new_state, next_key), info

    step_keys = jax.random.split(key, observations.shape[0] + 1)
    (final_state, _), info_history = lax.scan(
        scan_fn,
        (init_state, step_keys[0]),
        observations,
    )

    return final_state, info_history
