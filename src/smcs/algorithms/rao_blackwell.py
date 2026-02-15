"""Rao-Blackwellized Particle Filter (RBPF).

RBPF exploits conditional linear-Gaussian structure to analytically
marginalize out some state components, reducing variance.
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
from smcs.core.weights import compute_ess, normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "RBPFState",
    "rbpf_step",
    "run_rbpf",
    "MarginalizedState",
    "marginalized_step",
    "run_marginalized_filter",
]


@chex.dataclass(frozen=True)
class RBPFState(SMCState):
    """State for Rao-Blackwellized Particle Filter.

    Tracks both sampled nonlinear states and analytically computed
    Gaussian sufficient statistics for linear states.

    Attributes
    ----------
    particles : Array
        Sampled nonlinear state components.
    log_weights : Array
        Log-weights.
    linear_means : Array
        Kalman filter means for linear states.
    linear_covs : Array
        Kalman filter covariances for linear states.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Accumulated log-likelihood.
    step : Array
        Current step.
    """

    linear_means: Float[Array, "n_particles linear_dim"]
    linear_covs: Float[Array, "n_particles linear_dim linear_dim"]


@chex.dataclass(frozen=True)
class MarginalizedState(SMCState):
    """State for Marginalized Particle Filter.

    Attributes
    ----------
    particles : Array
        Sampled state components.
    log_weights : Array
        Log-weights.
    sufficient_stats : ArrayTree
        Sufficient statistics for marginalized components.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Accumulated log-likelihood.
    step : Array
        Current step.
    """

    sufficient_stats: chex.ArrayTree


@jaxtyped(typechecker=beartype)
def rbpf_step(
    key: PRNGKeyArray,
    state: RBPFState,
    observation: Float[Array, " obs_dim"],
    nonlinear_transition_fn: Callable[
        [PRNGKeyArray, Float[Array, " nonlinear_dim"]], Float[Array, " nonlinear_dim"]
    ],
    linear_transition_fn: Callable[
        [Float[Array, " nonlinear_dim"]],
        tuple[Float[Array, "linear_dim linear_dim"], Float[Array, "linear_dim linear_dim"]],
    ],
    observation_fn: Callable[
        [Float[Array, " nonlinear_dim"]],
        tuple[Float[Array, "obs_dim linear_dim"], Float[Array, "obs_dim obs_dim"]],
    ],
    ess_threshold: float = 0.5,
) -> tuple[RBPFState, SMCInfo]:
    """Perform one step of Rao-Blackwellized Particle Filter.

    Assumes model structure:
    x_t^n = f(x_{t-1}^n) + noise  (nonlinear, sampled)
    x_t^l = A(x_t^n) x_{t-1}^l + noise  (linear, Kalman filtered)
    y_t = H(x_t^n) x_t^l + noise

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : RBPFState
        Current state.
    observation : Array
        Current observation.
    nonlinear_transition_fn : Callable
        Transition for nonlinear states.
    linear_transition_fn : Callable
        Returns (A, Q) matrices for linear transition given nonlinear state.
    observation_fn : Callable
        Returns (H, R) matrices for observation given nonlinear state.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : RBPFState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    state.linear_means.shape[1]

    key, resample_key, propagate_key = jax.random.split(key, 3)

    # Compute ESS and resample if needed
    ess = compute_ess(state.log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, state.log_weights, "systematic"),
        lambda: jnp.arange(n_particles),
    )

    log_weights = jnp.where(do_resample, jnp.zeros(n_particles), state.log_weights)
    resampled_nonlinear = state.particles[ancestors]
    resampled_means = state.linear_means[ancestors]
    resampled_covs = state.linear_covs[ancestors]

    # Propagate nonlinear states
    propagate_keys = jax.random.split(propagate_key, n_particles)
    new_nonlinear = jax.vmap(nonlinear_transition_fn)(propagate_keys, resampled_nonlinear)

    # Kalman filter update for linear states
    def kalman_update(nonlinear_state, mean, cov):
        # Get transition matrices
        A, Q = linear_transition_fn(nonlinear_state)

        # Predict
        pred_mean = A @ mean
        pred_cov = A @ cov @ A.T + Q

        # Get observation matrices
        H, R = observation_fn(nonlinear_state)

        # Innovation
        innovation = observation - H @ pred_mean
        S = H @ pred_cov @ H.T + R

        # Kalman gain
        K = jnp.linalg.solve(S, H @ pred_cov).T

        # Update
        post_mean = pred_mean + K @ innovation
        post_cov = pred_cov - K @ S @ K.T

        # Ensure symmetry
        post_cov = 0.5 * (post_cov + post_cov.T)

        # Log-likelihood contribution
        sign, logdet = jnp.linalg.slogdet(S)
        log_lik = -0.5 * (
            jnp.dot(innovation, jnp.linalg.solve(S, innovation))
            + logdet
            + observation.shape[0] * jnp.log(2 * jnp.pi)
        )

        return post_mean, post_cov, log_lik

    new_means, new_covs, log_liks = jax.vmap(kalman_update)(
        new_nonlinear, resampled_means, resampled_covs
    )

    # Update weights
    new_log_weights = log_weights + log_liks

    # Log-likelihood increment
    log_lik_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)

    new_state = RBPFState(
        particles=new_nonlinear,
        log_weights=normalize_log_weights(new_log_weights),
        linear_means=new_means,
        linear_covs=new_covs,
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
def run_rbpf(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    nonlinear_init_fn: Callable[[PRNGKeyArray], Float[Array, " nonlinear_dim"]],
    linear_init_mean: Float[Array, " linear_dim"],
    linear_init_cov: Float[Array, "linear_dim linear_dim"],
    nonlinear_transition_fn: Callable,
    linear_transition_fn: Callable,
    observation_fn: Callable,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
) -> tuple[RBPFState, SMCInfo]:
    """Run Rao-Blackwellized Particle Filter.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    nonlinear_init_fn : Callable
        Initialize nonlinear state.
    linear_init_mean : Array
        Initial mean for linear state.
    linear_init_cov : Array
        Initial covariance for linear state.
    nonlinear_transition_fn : Callable
        Nonlinear state transition.
    linear_transition_fn : Callable
        Linear state transition matrices.
    observation_fn : Callable
        Observation matrices.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    final_state : RBPFState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize nonlinear particles
    init_keys = jax.random.split(init_key, n_particles)
    nonlinear_particles = jax.vmap(nonlinear_init_fn)(init_keys)

    # Initialize linear state (same for all particles)
    linear_init_mean.shape[0]
    linear_means = jnp.tile(linear_init_mean, (n_particles, 1))
    linear_covs = jnp.tile(linear_init_cov, (n_particles, 1, 1))

    initial_state = RBPFState(
        particles=nonlinear_particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        linear_means=linear_means,
        linear_covs=linear_covs,
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = rbpf_step(
            step_key,
            state,
            obs,
            nonlinear_transition_fn,
            linear_transition_fn,
            observation_fn,
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


@jaxtyped(typechecker=beartype)
def marginalized_step(
    key: PRNGKeyArray,
    state: MarginalizedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    marginal_update_fn: Callable[
        [chex.ArrayTree, Float[Array, " state_dim"], Float[Array, " obs_dim"]],
        tuple[chex.ArrayTree, Float[Array, ""]],
    ],
    ess_threshold: float = 0.5,
) -> tuple[MarginalizedState, SMCInfo]:
    """Perform one step of Marginalized Particle Filter.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : MarginalizedState
        Current state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    marginal_update_fn : Callable
        Updates sufficient statistics and returns (new_stats, log_lik).
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    new_state : MarginalizedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, propagate_key = jax.random.split(key, 3)

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

    # Resample sufficient statistics (handled as pytree)
    resampled_stats = jax.tree.map(lambda x: x[ancestors], state.sufficient_stats)

    # Propagate particles
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    new_particles = jax.vmap(propagate_one)(propagate_keys, resampled_particles)

    # Update sufficient statistics and compute likelihoods
    def update_one(stats, particle):
        return marginal_update_fn(stats, particle, observation)

    new_stats, log_liks = jax.vmap(update_one)(resampled_stats, new_particles)

    # Update weights
    new_log_weights = log_weights + log_liks

    # Log-likelihood increment
    log_lik_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)

    new_state = MarginalizedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        sufficient_stats=new_stats,
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
def run_marginalized_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    init_stats_fn: Callable[[PRNGKeyArray], chex.ArrayTree],
    marginal_update_fn: Callable,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
) -> tuple[MarginalizedState, SMCInfo]:
    """Run Marginalized Particle Filter.

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
    init_stats_fn : Callable
        Initialize sufficient statistics.
    marginal_update_fn : Callable
        Update sufficient statistics.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    final_state : MarginalizedState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key, stats_key = jax.random.split(key, 3)

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    # Initialize sufficient statistics
    stats_keys = jax.random.split(stats_key, n_particles)
    initial_stats = jax.vmap(init_stats_fn)(stats_keys)

    initial_state = MarginalizedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        sufficient_stats=initial_stats,
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = marginalized_step(
            step_key,
            state,
            obs,
            model,
            params,
            marginal_update_fn,
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
