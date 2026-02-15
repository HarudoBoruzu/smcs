"""Ensemble Kalman Filter (EnKF).

The Ensemble Kalman Filter is a Monte Carlo approximation to the Kalman filter
that works with nonlinear models by using ensemble statistics.
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

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "EnKFState",
    "enkf_step",
    "run_enkf",
    "eakf_step",
    "run_eakf",
]


@chex.dataclass(frozen=True)
class EnKFState(SMCState):
    """State for Ensemble Kalman Filter.

    Inherits all fields from SMCState. In EnKF, particles are called
    ensemble members and have equal weights.
    """

    pass


@jaxtyped(typechecker=beartype)
def enkf_step(
    key: PRNGKeyArray,
    state: EnKFState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    obs_noise_cov: Float[Array, "obs_dim obs_dim"] | None = None,
    inflation: float = 1.0,
    localization_fn: Callable | None = None,
) -> tuple[EnKFState, SMCInfo]:
    """Perform one step of the Ensemble Kalman Filter.

    Uses the stochastic EnKF formulation with perturbed observations.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : EnKFState
        Current filter state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model (must have linear or linearizable observation).
    params : ArrayTree
        Model parameters.
    obs_noise_cov : Array, optional
        Observation noise covariance. If None, estimated from model.
    inflation : float
        Covariance inflation factor (>= 1.0).
    localization_fn : callable, optional
        Function to localize covariance updates.

    Returns
    -------
    new_state : EnKFState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_ensemble = state.particles.shape[0]
    state.particles.shape[1]
    obs_dim = observation.shape[0]

    key, forecast_key, perturb_key = jax.random.split(key, 3)

    # Forecast step: propagate ensemble through transition
    forecast_keys = jax.random.split(forecast_key, n_ensemble)

    def forecast_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    forecast_ensemble = jax.vmap(forecast_one)(forecast_keys, state.particles)

    # Apply covariance inflation
    ensemble_mean = jnp.mean(forecast_ensemble, axis=0)
    forecast_ensemble = ensemble_mean + inflation * (forecast_ensemble - ensemble_mean)

    # Compute predicted observations for each ensemble member
    def predict_obs(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return emit_dist.mean if hasattr(emit_dist, "mean") else particle[:obs_dim]

    predicted_obs = jax.vmap(predict_obs)(forecast_ensemble)
    obs_mean = jnp.mean(predicted_obs, axis=0)

    # Compute covariances
    state_anomalies = forecast_ensemble - ensemble_mean
    obs_anomalies = predicted_obs - obs_mean

    # Cross-covariance P_xy
    P_xy = jnp.dot(state_anomalies.T, obs_anomalies) / (n_ensemble - 1)

    # Observation covariance P_yy
    P_yy = jnp.dot(obs_anomalies.T, obs_anomalies) / (n_ensemble - 1)

    # Add observation noise
    if obs_noise_cov is None:
        # Estimate from model emission distribution
        emit_dist = model.emission_distribution(
            params, ensemble_mean, state.step + 1
        )
        if hasattr(emit_dist, "scale"):
            obs_noise_cov = jnp.eye(obs_dim) * emit_dist.scale**2
        else:
            obs_noise_cov = jnp.eye(obs_dim) * 0.1

    P_yy = P_yy + obs_noise_cov

    # Apply localization if provided
    if localization_fn is not None:
        P_xy = localization_fn(P_xy)

    # Kalman gain
    K = jnp.linalg.solve(P_yy.T, P_xy.T).T

    # Perturb observations for stochastic EnKF
    obs_perturbations = jax.random.multivariate_normal(
        perturb_key,
        jnp.zeros(obs_dim),
        obs_noise_cov,
        shape=(n_ensemble,),
    )
    perturbed_obs = observation + obs_perturbations

    # Analysis update
    innovations = perturbed_obs - predicted_obs
    analysis_ensemble = forecast_ensemble + jnp.dot(innovations, K.T)

    # Compute approximate log-likelihood
    innovation_mean = observation - obs_mean
    log_likelihood_increment = -0.5 * (
        jnp.dot(innovation_mean, jnp.linalg.solve(P_yy, innovation_mean))
        + jnp.log(jnp.linalg.det(P_yy) + 1e-10)
        + obs_dim * jnp.log(2 * jnp.pi)
    )

    new_state = EnKFState(
        particles=analysis_ensemble,
        log_weights=jnp.full(n_ensemble, -jnp.log(n_ensemble)),
        ancestors=jnp.arange(n_ensemble),
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    # ESS is always n_ensemble for EnKF (equal weights)
    info = SMCInfo(
        ess=jnp.array(float(n_ensemble)),
        resampled=jnp.array(False),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def eakf_step(
    key: PRNGKeyArray,
    state: EnKFState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    obs_noise_var: Float[Array, " obs_dim"] | None = None,
    inflation: float = 1.0,
) -> tuple[EnKFState, SMCInfo]:
    """Perform one step of the Ensemble Adjustment Kalman Filter.

    EAKF is a deterministic variant that adjusts ensemble members without
    random perturbations, preserving ensemble variance exactly.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key (for forecast step).
    state : EnKFState
        Current filter state.
    observation : Array
        Current observation (processed one dimension at a time).
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    obs_noise_var : Array, optional
        Observation noise variance per dimension.
    inflation : float
        Covariance inflation factor.

    Returns
    -------
    new_state : EnKFState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_ensemble = state.particles.shape[0]
    obs_dim = observation.shape[0]

    key, forecast_key = jax.random.split(key)

    # Forecast step
    forecast_keys = jax.random.split(forecast_key, n_ensemble)

    def forecast_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    forecast_ensemble = jax.vmap(forecast_one)(forecast_keys, state.particles)

    # Covariance inflation
    ensemble_mean = jnp.mean(forecast_ensemble, axis=0)
    forecast_ensemble = ensemble_mean + inflation * (forecast_ensemble - ensemble_mean)

    # Get predicted observations
    def predict_obs(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return emit_dist.mean if hasattr(emit_dist, "mean") else particle[:obs_dim]

    predicted_obs = jax.vmap(predict_obs)(forecast_ensemble)

    # Default observation noise
    if obs_noise_var is None:
        obs_noise_var = jnp.ones(obs_dim) * 0.1

    # Process each observation dimension sequentially
    def update_one_obs(ensemble_and_pred, obs_idx):
        ensemble, pred_obs = ensemble_and_pred
        y = observation[obs_idx]
        y_pred = pred_obs[:, obs_idx]
        r = obs_noise_var[obs_idx]

        # Prior statistics
        y_mean = jnp.mean(y_pred)
        y_var = jnp.var(y_pred)

        # Posterior variance and mean
        post_var = 1.0 / (1.0 / y_var + 1.0 / r)
        post_mean = post_var * (y_mean / y_var + y / r)

        # Adjustment factor
        alpha = jnp.sqrt(post_var / y_var)

        # Update predicted observations
        new_y_pred = post_mean + alpha * (y_pred - y_mean)

        # Update state ensemble using regression
        state_obs_cov = jnp.sum(
            (ensemble - jnp.mean(ensemble, axis=0)) * (y_pred - y_mean)[:, None],
            axis=0,
        ) / (n_ensemble - 1)
        regression = state_obs_cov / (y_var + 1e-10)

        new_ensemble = ensemble + jnp.outer(new_y_pred - y_pred, regression)

        return (new_ensemble, pred_obs.at[:, obs_idx].set(new_y_pred)), None

    (analysis_ensemble, _), _ = jax.lax.scan(
        update_one_obs,
        (forecast_ensemble, predicted_obs),
        jnp.arange(obs_dim),
    )

    # Approximate log-likelihood
    obs_mean = jnp.mean(predicted_obs, axis=0)
    obs_var = jnp.var(predicted_obs, axis=0) + obs_noise_var
    innovation = observation - obs_mean
    log_likelihood_increment = -0.5 * jnp.sum(
        innovation**2 / obs_var + jnp.log(obs_var) + jnp.log(2 * jnp.pi)
    )

    new_state = EnKFState(
        particles=analysis_ensemble,
        log_weights=jnp.full(n_ensemble, -jnp.log(n_ensemble)),
        ancestors=jnp.arange(n_ensemble),
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=jnp.array(float(n_ensemble)),
        resampled=jnp.array(False),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_enkf(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_ensemble: int = 100,
    inflation: float = 1.0,
    obs_noise_cov: Float[Array, "obs_dim obs_dim"] | None = None,
) -> tuple[EnKFState, SMCInfo]:
    """Run the Ensemble Kalman Filter.

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
    n_ensemble : int
        Number of ensemble members.
    inflation : float
        Covariance inflation factor.
    obs_noise_cov : Array, optional
        Observation noise covariance.

    Returns
    -------
    final_state : EnKFState
        Final filter state.
    info : SMCInfo
        Combined information from all steps.
    """
    key, init_key = jax.random.split(key)

    # Initialize ensemble
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_ensemble)
    particles = jax.vmap(init_dist.sample)(init_keys)

    initial_state = EnKFState(
        particles=particles,
        log_weights=jnp.full(n_ensemble, -jnp.log(n_ensemble)),
        ancestors=jnp.arange(n_ensemble),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = enkf_step(
            step_key, state, obs, model, params, obs_noise_cov, inflation
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
def run_eakf(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_ensemble: int = 100,
    inflation: float = 1.0,
    obs_noise_var: Float[Array, " obs_dim"] | None = None,
) -> tuple[EnKFState, SMCInfo]:
    """Run the Ensemble Adjustment Kalman Filter.

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
    n_ensemble : int
        Number of ensemble members.
    inflation : float
        Covariance inflation factor.
    obs_noise_var : Array, optional
        Observation noise variance per dimension.

    Returns
    -------
    final_state : EnKFState
        Final filter state.
    info : SMCInfo
        Combined information from all steps.
    """
    key, init_key = jax.random.split(key)

    # Initialize ensemble
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_ensemble)
    particles = jax.vmap(init_dist.sample)(init_keys)

    initial_state = EnKFState(
        particles=particles,
        log_weights=jnp.full(n_ensemble, -jnp.log(n_ensemble)),
        ancestors=jnp.arange(n_ensemble),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = eakf_step(
            step_key, state, obs, model, params, obs_noise_var, inflation
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
