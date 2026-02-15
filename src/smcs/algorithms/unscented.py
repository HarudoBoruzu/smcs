"""Unscented Particle Filter.

Combines the Unscented Kalman Filter with particle filtering for
improved performance in nonlinear systems.
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
    "UnscentedState",
    "unscented_step",
    "run_unscented_filter",
    "compute_sigma_points",
    "unscented_transform",
]


@chex.dataclass(frozen=True)
class UnscentedState(SMCState):
    """State for Unscented Particle Filter.

    Attributes
    ----------
    particles : Array
        Particle positions (means from UKF).
    log_weights : Array
        Log-weights.
    covariances : Array
        Covariance matrices for each particle.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Accumulated log-likelihood.
    step : Array
        Current step.
    """

    covariances: Float[Array, "n_particles state_dim state_dim"]


@jaxtyped(typechecker=beartype)
def compute_sigma_points(
    mean: Float[Array, " state_dim"],
    cov: Float[Array, "state_dim state_dim"],
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[
    Float[Array, "n_sigma state_dim"],
    Float[Array, " n_sigma"],
    Float[Array, " n_sigma"],
]:
    """Compute sigma points and weights for unscented transform.

    Parameters
    ----------
    mean : Array
        Mean vector.
    cov : Array
        Covariance matrix.
    alpha : float
        Spread of sigma points (typically 1e-4 to 1).
    beta : float
        Prior knowledge about distribution (2 is optimal for Gaussian).
    kappa : float
        Secondary scaling parameter.

    Returns
    -------
    sigma_points : Array
        Sigma points of shape (2*n + 1, n).
    weights_mean : Array
        Weights for mean computation.
    weights_cov : Array
        Weights for covariance computation.
    """
    n = mean.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

    # Compute matrix square root
    sqrt_cov = jnp.linalg.cholesky((n + lambda_) * cov + 1e-6 * jnp.eye(n))

    # Generate sigma points
    sigma_points = jnp.zeros((2 * n + 1, n))
    sigma_points = sigma_points.at[0].set(mean)

    for i in range(n):
        sigma_points = sigma_points.at[i + 1].set(mean + sqrt_cov[:, i])
        sigma_points = sigma_points.at[n + i + 1].set(mean - sqrt_cov[:, i])

    # Compute weights
    w0_mean = lambda_ / (n + lambda_)
    w0_cov = w0_mean + (1 - alpha**2 + beta)
    wi = 1.0 / (2 * (n + lambda_))

    weights_mean = jnp.full(2 * n + 1, wi)
    weights_mean = weights_mean.at[0].set(w0_mean)

    weights_cov = jnp.full(2 * n + 1, wi)
    weights_cov = weights_cov.at[0].set(w0_cov)

    return sigma_points, weights_mean, weights_cov


@jaxtyped(typechecker=beartype)
def unscented_transform(
    sigma_points: Float[Array, "n_sigma dim_in"],
    weights_mean: Float[Array, " n_sigma"],
    weights_cov: Float[Array, " n_sigma"],
    transform_fn: Callable,
) -> tuple[Float[Array, " dim_out"], Float[Array, "dim_out dim_out"]]:
    """Apply unscented transform to sigma points.

    Parameters
    ----------
    sigma_points : Array
        Sigma points.
    weights_mean : Array
        Weights for mean.
    weights_cov : Array
        Weights for covariance.
    transform_fn : callable
        Nonlinear transformation function.

    Returns
    -------
    mean : Array
        Transformed mean.
    cov : Array
        Transformed covariance.
    """
    # Transform sigma points
    transformed = jax.vmap(transform_fn)(sigma_points)

    # Compute weighted mean
    mean = jnp.sum(weights_mean[:, None] * transformed, axis=0)

    # Compute weighted covariance
    diff = transformed - mean
    cov = jnp.sum(
        weights_cov[:, None, None] * jnp.einsum("ij,ik->ijk", diff, diff), axis=0
    )

    return mean, cov


@jaxtyped(typechecker=beartype)
def unscented_step(
    key: PRNGKeyArray,
    state: UnscentedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    process_noise_cov: Float[Array, "state_dim state_dim"] | None = None,
    obs_noise_cov: Float[Array, "obs_dim obs_dim"] | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    ess_threshold: float = 0.5,
) -> tuple[UnscentedState, SMCInfo]:
    """Perform one step of the Unscented Particle Filter.

    Combines UKF proposal with particle filter weights.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : UnscentedState
        Current filter state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    process_noise_cov : Array, optional
        Process noise covariance.
    obs_noise_cov : Array, optional
        Observation noise covariance.
    alpha, beta, kappa : float
        UKF parameters.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : UnscentedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    state_dim = state.particles.shape[1]
    obs_dim = observation.shape[0]

    key, resample_key, sample_key = jax.random.split(key, 3)

    # Default noise covariances
    if process_noise_cov is None:
        process_noise_cov = jnp.eye(state_dim) * 0.01
    if obs_noise_cov is None:
        obs_noise_cov = jnp.eye(obs_dim) * 0.1

    # Compute ESS and resample if needed
    ess = compute_ess(state.log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    log_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        state.log_weights,
    )
    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, state.log_weights, method="systematic"),
        lambda: jnp.arange(n_particles),
    )
    particles = state.particles[ancestors]
    covariances = state.covariances[ancestors]

    def ukf_update(mean_cov):
        mean, cov = mean_cov

        # Prediction step with sigma points
        sigma_pts, w_m, w_c = compute_sigma_points(mean, cov, alpha, beta, kappa)

        # Transform through dynamics
        def transition_mean(x):
            trans_dist = model.transition_distribution(params, x, state.step)
            return trans_dist.mean if hasattr(trans_dist, "mean") else x

        pred_mean, pred_cov = unscented_transform(sigma_pts, w_m, w_c, transition_mean)
        pred_cov = pred_cov + process_noise_cov

        # Generate new sigma points for observation
        pred_sigma, w_m2, w_c2 = compute_sigma_points(
            pred_mean, pred_cov, alpha, beta, kappa
        )

        # Transform through observation model
        def obs_fn(x):
            emit_dist = model.emission_distribution(params, x, state.step + 1)
            return emit_dist.mean if hasattr(emit_dist, "mean") else x[:obs_dim]

        obs_mean, obs_cov = unscented_transform(pred_sigma, w_m2, w_c2, obs_fn)
        obs_cov = obs_cov + obs_noise_cov

        # Cross-covariance
        pred_diff = pred_sigma - pred_mean
        obs_diff = jax.vmap(obs_fn)(pred_sigma) - obs_mean
        cross_cov = jnp.sum(
            w_c2[:, None, None] * jnp.einsum("ij,ik->ijk", pred_diff, obs_diff), axis=0
        )

        # Kalman gain and update
        K = jnp.linalg.solve(obs_cov.T, cross_cov.T).T
        innovation = observation - obs_mean
        post_mean = pred_mean + K @ innovation
        post_cov = pred_cov - K @ obs_cov @ K.T

        # Ensure positive definiteness
        post_cov = 0.5 * (post_cov + post_cov.T) + 1e-6 * jnp.eye(state_dim)

        # Log-likelihood for this particle
        log_lik = -0.5 * (
            jnp.dot(innovation, jnp.linalg.solve(obs_cov, innovation))
            + jnp.log(jnp.linalg.det(obs_cov) + 1e-10)
            + obs_dim * jnp.log(2 * jnp.pi)
        )

        return post_mean, post_cov, log_lik

    # Apply UKF update to each particle
    results = jax.vmap(lambda mc: ukf_update(mc))(
        (particles, covariances)
    )
    new_means, new_covs, log_liks = results

    # Sample from posterior Gaussians
    sample_keys = jax.random.split(sample_key, n_particles)

    def sample_posterior(key_i, mean, cov):
        return jax.random.multivariate_normal(key_i, mean, cov)

    new_particles = jax.vmap(sample_posterior)(sample_keys, new_means, new_covs)

    # Update weights
    new_log_weights = log_weights + log_liks

    # Log-likelihood increment
    log_likelihood_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)

    new_state = UnscentedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        covariances=new_covs,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_unscented_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_particles: int = 100,
    initial_cov: Float[Array, "state_dim state_dim"] | None = None,
    process_noise_cov: Float[Array, "state_dim state_dim"] | None = None,
    obs_noise_cov: Float[Array, "obs_dim obs_dim"] | None = None,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    ess_threshold: float = 0.5,
) -> tuple[UnscentedState, SMCInfo]:
    """Run the Unscented Particle Filter.

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
    initial_cov : Array, optional
        Initial covariance for each particle.
    process_noise_cov : Array, optional
        Process noise covariance.
    obs_noise_cov : Array, optional
        Observation noise covariance.
    alpha, beta, kappa : float
        UKF parameters.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    final_state : UnscentedState
        Final filter state.
    info : SMCInfo
        Combined information from all steps.
    """
    key, init_key = jax.random.split(key)

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    state_dim = particles.shape[1]
    obs_dim = observations.shape[1]

    # Default covariances
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)
    if process_noise_cov is None:
        process_noise_cov = jnp.eye(state_dim) * 0.01
    if obs_noise_cov is None:
        obs_noise_cov = jnp.eye(obs_dim) * 0.1

    initial_state = UnscentedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        covariances=jnp.tile(initial_cov, (n_particles, 1, 1)),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = unscented_step(
            step_key,
            state,
            obs,
            model,
            params,
            process_noise_cov,
            obs_noise_cov,
            alpha,
            beta,
            kappa,
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
