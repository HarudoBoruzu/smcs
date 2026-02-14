"""SMC² (SMC-squared) for online parameter learning.

SMC² (Chopin et al., 2013) is a nested SMC algorithm where:
- Outer SMC: samples over parameter space
- Inner SMC: particle filters for state estimation given each parameter
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
    "SMC2State",
    "smc2_step",
    "run_smc2",
]


@chex.dataclass(frozen=True)
class SMC2State:
    """State for SMC² algorithm.

    Attributes
    ----------
    param_particles : Array
        Parameter particles [n_theta_particles, param_dim].
    state_particles : Array
        State particles for each theta [n_theta_particles, n_x_particles, state_dim].
    state_log_weights : Array
        State particle weights [n_theta_particles, n_x_particles].
    param_log_weights : Array
        Parameter particle weights [n_theta_particles].
    log_likelihood : float
        Cumulative log marginal likelihood.
    step : int
        Current time step.
    """

    param_particles: Float[Array, "n_theta state_dim"]
    state_particles: Float[Array, "n_theta n_x state_dim"]
    state_log_weights: Float[Array, "n_theta n_x"]
    param_log_weights: Float[Array, " n_theta"]
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]

    @property
    def n_theta_particles(self) -> int:
        return self.param_particles.shape[0]

    @property
    def n_x_particles(self) -> int:
        return self.state_particles.shape[1]


@jaxtyped(typechecker=beartype)
def smc2_step(
    key: PRNGKeyArray,
    state: SMC2State,
    observation: Float[Array, ...],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    ess_threshold_theta: float = 0.5,
    ess_threshold_x: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
    rejuvenate_fn: Callable | None = None,
) -> tuple[SMC2State, SMCInfo]:
    """Perform one step of SMC².

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : SMC2State
        Current SMC² state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State space model.
    param_to_model_params : callable
        Converts parameter vector to ModelParams.
    ess_threshold_theta : float
        ESS/N threshold for parameter resampling.
    ess_threshold_x : float
        ESS/N threshold for state resampling.
    resampling_method : str
        Resampling method.
    rejuvenate_fn : callable, optional
        MCMC move to rejuvenate parameter particles after resampling.

    Returns
    -------
    new_state : SMC2State
        Updated SMC² state.
    info : SMCInfo
        Diagnostic information.
    """
    n_theta = state.n_theta_particles
    n_x = state.n_x_particles
    key, theta_resamp_key, x_keys_key, prop_key = jax.random.split(key, 4)

    observation = jnp.atleast_1d(observation)

    # Run one step of particle filter for each theta particle
    x_step_keys = jax.random.split(x_keys_key, n_theta)

    def inner_pf_step(inputs):
        step_key, theta, x_particles, x_log_weights = inputs
        model_params = param_to_model_params(theta)

        # Resample state particles if needed
        x_ess = compute_ess(x_log_weights)
        should_resample_x = x_ess < ess_threshold_x * n_x

        resamp_key, prop_key = jax.random.split(step_key)

        def do_resample_x(args):
            parts, log_w, rkey = args
            indices = resample(rkey, log_w, resampling_method)
            return parts[indices], jnp.zeros(n_x)

        def no_resample_x(args):
            parts, log_w, _ = args
            return parts, log_w

        x_particles, x_log_weights = lax.cond(
            should_resample_x,
            do_resample_x,
            no_resample_x,
            (x_particles, x_log_weights, resamp_key),
        )

        # Propagate state particles
        prop_keys = jax.random.split(prop_key, n_x)

        def propagate_x(key_x):
            key, x = key_x
            trans_dist = model.transition_distribution(model_params, x, state.step)
            return jnp.atleast_1d(trans_dist.sample(key))

        new_x_particles = vmap(propagate_x)((prop_keys, x_particles))

        # Update state weights with observation likelihood
        def compute_x_log_lik(x):
            emit_dist = model.emission_distribution(model_params, x, state.step + 1)
            return emit_dist.log_prob(observation)

        x_log_liks = vmap(compute_x_log_lik)(new_x_particles)
        new_x_log_weights = x_log_weights + x_log_liks

        # Log marginal likelihood for this theta particle
        log_marginal = jax.scipy.special.logsumexp(new_x_log_weights) - jnp.log(n_x)

        return new_x_particles, new_x_log_weights, log_marginal

    results = vmap(inner_pf_step)(
        (x_step_keys, state.param_particles, state.state_particles, state.state_log_weights)
    )
    new_state_particles, new_state_log_weights, log_marginals = results

    # Update parameter weights
    new_param_log_weights = state.param_log_weights + log_marginals

    # Resample parameter particles if ESS is low
    theta_ess = compute_ess(new_param_log_weights)
    should_resample_theta = theta_ess < ess_threshold_theta * n_theta

    def do_resample_theta(args):
        thetas, x_parts, x_log_w, theta_log_w, rkey = args
        indices = resample(rkey, theta_log_w, resampling_method)
        return (
            thetas[indices],
            x_parts[indices],
            x_log_w[indices],
            jnp.zeros(n_theta),
        )

    def no_resample_theta(args):
        thetas, x_parts, x_log_w, theta_log_w, _ = args
        return thetas, x_parts, x_log_w, theta_log_w

    (
        new_param_particles,
        new_state_particles,
        new_state_log_weights,
        new_param_log_weights,
    ) = lax.cond(
        should_resample_theta,
        do_resample_theta,
        no_resample_theta,
        (
            state.param_particles,
            new_state_particles,
            new_state_log_weights,
            new_param_log_weights,
            theta_resamp_key,
        ),
    )

    # Note: rejuvenate_fn support is reserved for future implementation.
    # It would require lax.cond to handle should_resample_theta (a traced value).

    # Marginal likelihood increment
    log_likelihood_increment = (
        jax.scipy.special.logsumexp(new_param_log_weights) - jnp.log(n_theta)
    )

    new_state = SMC2State(
        param_particles=new_param_particles,
        state_particles=new_state_particles,
        state_log_weights=new_state_log_weights,
        param_log_weights=new_param_log_weights,
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=theta_ess,
        resampled=should_resample_theta,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_smc2(
    key: PRNGKeyArray,
    observations: Float[Array, "n_timesteps ..."],
    model: StateSpaceModel,
    param_to_model_params: Callable,
    initial_param_sampler: Callable,
    initial_state_sampler: Callable,
    n_theta_particles: int = 100,
    n_x_particles: int = 100,
    ess_threshold_theta: float = 0.5,
    ess_threshold_x: float = 0.5,
    resampling_method: ResamplingMethod = "systematic",
) -> tuple[SMC2State, SMCInfo]:
    """Run SMC² for joint state and parameter estimation.

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
    initial_param_sampler : callable
        Function (key,) -> initial parameter sample.
    initial_state_sampler : callable
        Function (key, param_vec) -> initial state sample.
    n_theta_particles : int
        Number of parameter particles.
    n_x_particles : int
        Number of state particles per parameter.
    ess_threshold_theta : float
        ESS/N threshold for parameter resampling.
    ess_threshold_x : float
        ESS/N threshold for state resampling.
    resampling_method : str
        Resampling method.

    Returns
    -------
    final_state : SMC2State
        Final SMC² state.
    info_history : SMCInfo
        Stacked diagnostic information.
    """
    key, theta_init_key, x_init_key = jax.random.split(key, 3)

    # Initialize parameter particles from prior
    theta_keys = jax.random.split(theta_init_key, n_theta_particles)
    initial_params = vmap(initial_param_sampler)(theta_keys)

    # Initialize state particles for each theta
    x_keys = jax.random.split(x_init_key, n_theta_particles * n_x_particles).reshape(
        n_theta_particles, n_x_particles, -1
    )

    def init_x_for_theta(keys_theta):
        keys, theta = keys_theta
        return vmap(lambda k: jnp.atleast_1d(initial_state_sampler(k, theta)))(keys)

    initial_states = vmap(init_x_for_theta)(
        (x_keys[:, :, 0], initial_params)
    )  # Shape: [n_theta, n_x, state_dim]

    init_state = SMC2State(
        param_particles=initial_params,
        state_particles=initial_states,
        state_log_weights=jnp.zeros((n_theta_particles, n_x_particles)),
        param_log_weights=jnp.zeros(n_theta_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Scan function
    def scan_fn(carry, obs):
        current_state, step_key = carry
        step_key, next_key = jax.random.split(step_key)

        new_state, info = smc2_step(
            step_key,
            current_state,
            obs,
            model,
            param_to_model_params,
            ess_threshold_theta,
            ess_threshold_x,
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
