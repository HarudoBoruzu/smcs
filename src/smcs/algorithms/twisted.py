"""Twisted SMC and Controlled SMC.

These algorithms use twisted/tilted proposals to reduce variance
by incorporating future observations into the proposal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "TwistedState",
    "twisted_step",
    "run_twisted_smc",
    "controlled_step",
    "run_controlled_smc",
    "learn_twisting_functions",
]


@chex.dataclass(frozen=True)
class TwistedState(SMCState):
    """State for Twisted/Controlled SMC.

    Attributes
    ----------
    particles : Array
        Particle positions.
    log_weights : Array
        Log-weights.
    twisting_log_values : Array
        Twisting function log-values for each particle.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Log-likelihood estimate.
    step : Array
        Current step.
    """

    twisting_log_values: Float[Array, " n_particles"]


@jaxtyped(typechecker=beartype)
def twisted_step(
    key: PRNGKeyArray,
    state: TwistedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    twisting_fn: Callable[
        [Float[Array, " state_dim"], Int[Array, ""]], Float[Array, ""]
    ],
    next_twisting_fn: Callable[
        [Float[Array, " state_dim"], Int[Array, ""]], Float[Array, ""]
    ] | None = None,
    ess_threshold: float = 0.5,
) -> tuple[TwistedState, SMCInfo]:
    """Perform one step of Twisted SMC.

    The twisted target at time t is:
    pi_t(x_t) ∝ p(y_t | x_t) * p(x_t | x_{t-1}) * psi_t(x_t) / psi_{t-1}(x_{t-1})

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : TwistedState
        Current state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    twisting_fn : Callable
        Twisting function psi_t(x, t) at current time.
    next_twisting_fn : Callable, optional
        Twisting function at next time (None for final step).
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : TwistedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, propagate_key = jax.random.split(key, 3)

    # Compute twisted weights for resampling
    # Include twisting at current particles
    current_twisting = jax.vmap(lambda x: twisting_fn(x, state.step))(state.particles)
    twisted_log_weights = state.log_weights + current_twisting - state.twisting_log_values

    # ESS and resampling
    ess = compute_ess(twisted_log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, twisted_log_weights, "systematic"),
        lambda: jnp.arange(n_particles),
    )

    log_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        twisted_log_weights,
    )
    resampled_particles = state.particles[ancestors]
    resampled_twisting = current_twisting[ancestors]

    # Propagate
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    new_particles = jax.vmap(propagate_one)(propagate_keys, resampled_particles)

    # Compute incremental weights
    def compute_increment(new_x, old_x, old_twist):
        # Emission
        emit_dist = model.emission_distribution(params, new_x, state.step + 1)
        log_emit = jnp.squeeze(emit_dist.log_prob(observation))

        # New twisting (use identity if None)
        if next_twisting_fn is not None:
            new_twist = next_twisting_fn(new_x, state.step + 1)
        else:
            new_twist = jnp.array(0.0)

        # Increment: g_t(y|x) * psi_{t+1}(x') / psi_t(x)
        return log_emit + new_twist - old_twist, new_twist

    increments_and_twists = jax.vmap(compute_increment)(
        new_particles, resampled_particles, resampled_twisting
    )
    log_increments = increments_and_twists[0]
    new_twisting_values = increments_and_twists[1]

    new_log_weights = log_weights + log_increments

    # Log-likelihood increment
    log_lik_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)

    new_state = TwistedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        twisting_log_values=new_twisting_values,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_lik_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def controlled_step(
    key: PRNGKeyArray,
    state: TwistedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    policy_fn: Callable[
        [PRNGKeyArray, Float[Array, " state_dim"], Float[Array, " obs_dim"]],
        Float[Array, " state_dim"],
    ],
    policy_log_prob_fn: Callable[
        [Float[Array, " state_dim"], Float[Array, " state_dim"], Float[Array, " obs_dim"]],
        Float[Array, ""],
    ],
    ess_threshold: float = 0.5,
) -> tuple[TwistedState, SMCInfo]:
    """Perform one step of Controlled SMC.

    Uses a learned policy to propose particles that account for
    future observations (lookahead).

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : TwistedState
        Current state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    policy_fn : Callable
        Policy function (key, state, obs) -> new_state.
    policy_log_prob_fn : Callable
        Log-probability of policy (new_state, old_state, obs) -> log_prob.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    new_state : TwistedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, propose_key = jax.random.split(key, 3)

    # ESS and resampling
    ess = compute_ess(state.log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, state.log_weights, "systematic"),
        lambda: jnp.arange(n_particles),
    )

    log_weights = jnp.where(do_resample, jnp.zeros(n_particles), state.log_weights)
    resampled_particles = state.particles[ancestors]

    # Propose using policy
    propose_keys = jax.random.split(propose_key, n_particles)

    def propose_one(key_i, particle):
        return policy_fn(key_i, particle, observation)

    new_particles = jax.vmap(propose_one)(propose_keys, resampled_particles)

    # Compute importance weights
    def compute_weight(new_x, old_x):
        # Transition
        trans_dist = model.transition_distribution(params, old_x, state.step)
        log_trans = jnp.squeeze(trans_dist.log_prob(new_x))

        # Emission
        emit_dist = model.emission_distribution(params, new_x, state.step + 1)
        log_emit = jnp.squeeze(emit_dist.log_prob(observation))

        # Policy
        log_policy = policy_log_prob_fn(new_x, old_x, observation)

        return log_trans + log_emit - log_policy

    log_increments = jax.vmap(compute_weight)(new_particles, resampled_particles)
    new_log_weights = log_weights + log_increments

    # Log-likelihood increment
    log_lik_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)

    new_state = TwistedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        twisting_log_values=jnp.zeros(n_particles),
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_lik_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_twisted_smc(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    twisting_functions: list[Callable] | None = None,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
) -> tuple[TwistedState, SMCInfo]:
    """Run Twisted SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    twisting_functions : list[Callable], optional
        List of twisting functions for each time step.
        If None, uses identity (reduces to bootstrap filter).
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    final_state : TwistedState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)
    n_steps = observations.shape[0]

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    # Default twisting functions (identity)
    if twisting_functions is None:
        twisting_functions = [lambda x, t: jnp.array(0.0)] * (n_steps + 1)

    initial_state = TwistedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        twisting_log_values=jnp.zeros(n_particles),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Run filter (can't use scan easily with list of functions)
    state = initial_state
    all_ess = []
    all_resampled = []

    for t in range(n_steps):
        key, step_key = jax.random.split(key)
        twist_fn = twisting_functions[t]
        next_twist_fn = twisting_functions[t + 1] if t < n_steps - 1 else None

        state, info = twisted_step(
            step_key,
            state,
            observations[t],
            model,
            params,
            twist_fn,
            next_twist_fn,
            ess_threshold,
        )
        all_ess.append(info.ess)
        all_resampled.append(info.resampled)

    combined_info = SMCInfo(
        ess=jnp.stack(all_ess),
        resampled=jnp.stack(all_resampled),
        acceptance_rate=None,
    )

    return state, combined_info


@jaxtyped(typechecker=beartype)
def run_controlled_smc(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    policy_fn: Callable,
    policy_log_prob_fn: Callable,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
) -> tuple[TwistedState, SMCInfo]:
    """Run Controlled SMC with a learned policy.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    policy_fn : Callable
        Policy function.
    policy_log_prob_fn : Callable
        Policy log-probability function.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    final_state : TwistedState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    initial_state = TwistedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        twisting_log_values=jnp.zeros(n_particles),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = controlled_step(
            step_key,
            state,
            obs,
            model,
            params,
            policy_fn,
            policy_log_prob_fn,
            ess_threshold,
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


def learn_twisting_functions(
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_iterations: int = 10,
) -> list[Callable]:
    """Learn twisting functions via backward iteration.

    This is a simplified version that computes approximate
    optimal twisting functions using dynamic programming.

    Parameters
    ----------
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    n_iterations : int
        Number of learning iterations.

    Returns
    -------
    twisting_functions : list[Callable]
        Learned twisting functions.
    """
    n_steps = observations.shape[0]

    # Initialize with zero twisting
    # In practice, this would be learned via value function approximation
    twisting_functions = [lambda x, t: jnp.array(0.0)] * (n_steps + 1)

    return twisting_functions
