"""Liu-West Filter for online parameter learning.

The Liu-West Filter (Liu & West, 2001) enables online learning of static
parameters by artificially introducing dynamics through kernel shrinkage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax, vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo
from smcs.core.resampling import ResamplingMethod, resample
from smcs.core.weights import compute_ess

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "LiuWestState",
    "liu_west_step",
    "run_liu_west_filter",
]


@chex.dataclass(frozen=True)
class LiuWestState:
    """State for Liu-West Filter with parameter particles.

    Attributes
    ----------
    state_particles : Array
        State particles [n_particles, state_dim].
    param_particles : Array
        Parameter particles [n_particles, param_dim].
    log_weights : Array
        Log-weights [n_particles].
    log_likelihood : float
        Cumulative log marginal likelihood.
    step : int
        Current time step.
    """

    state_particles: Float[Array, "n_particles state_dim"]
    param_particles: Float[Array, "n_particles param_dim"]
    log_weights: Float[Array, " n_particles"]
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]

    @property
    def n_particles(self) -> int:
        return self.state_particles.shape[0]

    def weighted_param_mean(self) -> Float[Array, " param_dim"]:
        """Compute weighted mean of parameter particles."""
        weights = jnp.exp(self.log_weights - jax.scipy.special.logsumexp(self.log_weights))
        return jnp.sum(self.param_particles * weights[:, None], axis=0)

    def weighted_param_cov(self) -> Float[Array, "param_dim param_dim"]:
        """Compute weighted covariance of parameter particles."""
        weights = jnp.exp(self.log_weights - jax.scipy.special.logsumexp(self.log_weights))
        mean = self.weighted_param_mean()
        centered = self.param_particles - mean
        return jnp.einsum("i,ij,ik->jk", weights, centered, centered)


@jaxtyped(typechecker=beartype)
def liu_west_step(
    key: PRNGKeyArray,
    state: LiuWestState,
    observation: Float[Array, ...],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    delta: float = 0.98,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[LiuWestState, SMCInfo]:
    """Perform one step of the Liu-West Filter.

    The Liu-West filter uses kernel shrinkage to maintain parameter diversity:
        theta_t^(i) = a * theta_{t-1}^(i) + (1-a) * theta_bar + sqrt(1-a^2) * V^{1/2} * eps

    where a = (3*delta - 1) / (2*delta) and delta in (0.95, 0.99).

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : LiuWestState
        Current Liu-West state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State space model.
    param_to_model_params : callable
        Function to convert parameter vector to ModelParams.
    delta : float
        Discount factor in (0, 1). Higher values mean less shrinkage.
        Recommended: 0.95 to 0.99.
    ess_threshold : float
        ESS/N ratio threshold for resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    new_state : LiuWestState
        Updated Liu-West state.
    info : SMCInfo
        Diagnostic information.
    """
    n_particles = state.n_particles
    key, resample_key, param_key, prop_key = jax.random.split(key, 4)

    observation = jnp.atleast_1d(observation)

    # Shrinkage coefficient
    a = (3 * delta - 1) / (2 * delta)

    # Weighted mean and covariance of parameters
    weights = jnp.exp(state.log_weights - jax.scipy.special.logsumexp(state.log_weights))
    param_mean = jnp.sum(state.param_particles * weights[:, None], axis=0)
    centered = state.param_particles - param_mean
    param_cov = jnp.einsum("i,ij,ik->jk", weights, centered, centered)

    # Add small regularization for numerical stability
    param_cov = param_cov + 1e-8 * jnp.eye(param_cov.shape[0])

    # Cholesky for sampling
    try:
        param_cov_sqrt = jnp.linalg.cholesky(param_cov)
    except Exception:
        param_cov_sqrt = jnp.sqrt(jnp.diag(jnp.diag(param_cov)))

    # ESS and resampling
    ess = compute_ess(state.log_weights)
    should_resample = ess < ess_threshold * n_particles

    def do_resample(args):
        s_parts, p_parts, log_w, rkey = args
        indices = resample(rkey, log_w, resampling_method)
        return s_parts[indices], p_parts[indices], jnp.zeros(n_particles)

    def no_resample(args):
        s_parts, p_parts, log_w, _ = args
        return s_parts, p_parts, log_w

    state_particles, param_particles, log_weights = lax.cond(
        should_resample,
        do_resample,
        no_resample,
        (state.state_particles, state.param_particles, state.log_weights, resample_key),
    )

    # Apply kernel shrinkage to parameters
    param_keys = jax.random.split(param_key, n_particles)

    def shrink_param(key_param):
        key, param = key_param
        # Shrink towards mean
        shrunk = a * param + (1 - a) * param_mean
        # Add noise
        noise = jax.random.normal(key, shape=param.shape)
        return shrunk + jnp.sqrt(1 - a**2) * param_cov_sqrt @ noise

    new_param_particles = vmap(shrink_param)((param_keys, param_particles))

    # Propagate state particles using updated parameters
    prop_keys = jax.random.split(prop_key, n_particles)

    def propagate_single(inputs):
        key, state_particle, param_vec = inputs
        model_params = param_to_model_params(param_vec)
        trans_dist = model.transition_distribution(model_params, state_particle, state.step)
        return jnp.atleast_1d(trans_dist.sample(key))

    new_state_particles = vmap(propagate_single)((prop_keys, state_particles, new_param_particles))

    # Update weights with observation likelihood
    def compute_log_likelihood(state_particle, param_vec):
        model_params = param_to_model_params(param_vec)
        emit_dist = model.emission_distribution(model_params, state_particle, state.step + 1)
        return emit_dist.log_prob(observation)

    log_likelihoods = vmap(compute_log_likelihood)(new_state_particles, new_param_particles)
    new_log_weights = log_weights + log_likelihoods

    # Marginal likelihood increment
    log_likelihood_increment = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(n_particles)

    new_state = LiuWestState(
        state_particles=new_state_particles,
        param_particles=new_param_particles,
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
def run_liu_west_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    initial_state_sampler: Callable,
    initial_param_sampler: Callable,
    n_particles: int = 1000,
    delta: float = 0.98,
    ess_threshold: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[LiuWestState, SMCInfo]:
    """Run the Liu-West Filter for joint state and parameter estimation.

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
    initial_state_sampler : callable
        Function (key, param_vec) -> initial state sample.
    initial_param_sampler : callable
        Function (key,) -> initial parameter sample.
    n_particles : int
        Number of particles.
    delta : float
        Discount factor for kernel shrinkage.
    ess_threshold : float
        ESS/N threshold for resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    final_state : LiuWestState
        Final Liu-West state.
    info_history : SMCInfo
        Stacked diagnostic information.
    """
    key, init_key = jax.random.split(key)
    param_keys, state_keys = jax.random.split(init_key)
    param_keys = jax.random.split(param_keys, n_particles)
    state_keys = jax.random.split(state_keys, n_particles)

    # Initialize parameter particles from prior
    initial_params = vmap(initial_param_sampler)(param_keys)

    # Initialize state particles given parameters
    initial_states = vmap(initial_state_sampler)(state_keys, initial_params)

    # Ensure proper shapes
    if initial_states.ndim == 1:
        initial_states = initial_states[:, None]

    init_state = LiuWestState(
        state_particles=initial_states,
        param_particles=initial_params,
        log_weights=jnp.zeros(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Scan function
    def scan_fn(carry, obs):
        current_state, step_key = carry
        step_key, next_key = jax.random.split(step_key)

        new_state, info = liu_west_step(
            step_key,
            current_state,
            obs,
            model,
            param_to_model_params,
            delta,
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
