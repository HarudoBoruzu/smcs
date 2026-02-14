"""Base classes and protocols for forecasting agents.

This module provides the foundational abstractions for time series
forecasting using SMC methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import pandas as pd
from jax import vmap
from jaxtyping import Array, Float, PRNGKeyArray
from loguru import logger

from smcs.algorithms.bootstrap import bootstrap_step, run_bootstrap_filter
from smcs.core.particles import SMCState
from smcs.core.resampling import systematic_resample

if TYPE_CHECKING:
    from smcs.config.settings import SMCConfig
    from smcs.models.base import ModelParams, StateSpaceModel

__all__ = [
    "ForecastResult",
    "ForecastingAgent",
    "BaseAgent",
]


@dataclass(frozen=True)
class ForecastResult:
    """Container for forecast results.

    Attributes
    ----------
    mean : Array
        Point forecast (mean) [horizon, obs_dim].
    std : Array
        Standard deviation [horizon, obs_dim].
    quantiles : dict
        Quantile forecasts {q: Array[horizon, obs_dim]}.
    particles : Array or None
        Particle samples [n_samples, horizon, obs_dim].
    timestamps : DatetimeIndex or None
        Future timestamps.
    """

    mean: Float[Array, "horizon obs_dim"]
    std: Float[Array, "horizon obs_dim"]
    quantiles: dict[float, Float[Array, "horizon obs_dim"]]
    particles: Float[Array, "n_samples horizon obs_dim"] | None
    timestamps: pd.DatetimeIndex | None


@runtime_checkable
class ForecastingAgent(Protocol):
    """Protocol defining the interface for forecasting agents."""

    @property
    def name(self) -> str:
        """Agent name."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the agent has been fitted."""
        ...

    def fit(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
        timestamps: pd.DatetimeIndex | None = None,
    ) -> ForecastingAgent:
        """Fit the model to observations.

        Parameters
        ----------
        observations : Array
            Observation sequence.
        timestamps : DatetimeIndex, optional
            Timestamps for observations.

        Returns
        -------
        self : ForecastingAgent
            Fitted agent (for method chaining).
        """
        ...

    def predict(
        self,
        horizon: int,
        n_samples: int = 1000,
        quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> ForecastResult:
        """Generate forecasts.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        n_samples : int
            Number of forecast samples.
        quantiles : tuple
            Quantiles to compute.

        Returns
        -------
        result : ForecastResult
            Forecast results.
        """
        ...

    def update(
        self,
        observation: Float[Array, " obs_dim"],
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """Online update with a single observation.

        Parameters
        ----------
        observation : Array
            New observation.
        timestamp : Timestamp, optional
            Timestamp of observation.
        """
        ...

    def get_filter_state(self) -> SMCState:
        """Get current filter state."""
        ...


class BaseAgent(ABC):
    """Base class for forecasting agents.

    Provides common functionality for SMC-based time series forecasting.
    """

    def __init__(
        self,
        config: SMCConfig,
        model: StateSpaceModel,
    ):
        """Initialize agent.

        Parameters
        ----------
        config : SMCConfig
            SMC configuration.
        model : StateSpaceModel
            State space model.
        """
        self._config = config
        self._model = model
        self._filter_state: SMCState | None = None
        self._params: ModelParams | None = None
        self._is_fitted = False
        self._key = jax.random.PRNGKey(config.seed)
        self._timestamps: pd.DatetimeIndex | None = None

        logger.info(f"Initialized {self.name}")

    @property
    def name(self) -> str:
        """Agent name."""
        return self.__class__.__name__

    @property
    def is_fitted(self) -> bool:
        """Whether the agent has been fitted."""
        return self._is_fitted

    @property
    def config(self) -> SMCConfig:
        """SMC configuration."""
        return self._config

    @property
    def model(self) -> StateSpaceModel:
        """State space model."""
        return self._model

    @abstractmethod
    def _estimate_params(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
    ) -> ModelParams:
        """Estimate model parameters from observations.

        Subclasses must implement this method.

        Parameters
        ----------
        observations : Array
            Observation sequence.

        Returns
        -------
        params : ModelParams
            Estimated parameters.
        """
        ...

    def fit(
        self,
        observations: Float[Array, "n_timesteps obs_dim"],
        timestamps: pd.DatetimeIndex | None = None,
    ) -> BaseAgent:
        """Fit the model to observations.

        Parameters
        ----------
        observations : Array
            Observation sequence.
        timestamps : DatetimeIndex, optional
            Timestamps.

        Returns
        -------
        self : BaseAgent
            Fitted agent.
        """
        logger.info(f"Fitting {self.name} on {observations.shape[0]} observations")

        self._timestamps = timestamps
        self._key, param_key, filter_key = jax.random.split(self._key, 3)

        # Ensure observations are properly shaped
        observations = jnp.atleast_2d(observations)
        if observations.shape[1] > observations.shape[0]:
            observations = observations.T

        # Estimate parameters
        self._params = self._estimate_params(observations)
        logger.debug(f"Estimated parameters: {self._params}")

        # Run particle filter
        self._filter_state, info_history = run_bootstrap_filter(
            filter_key,
            observations,
            self._model,
            self._params,
            n_particles=self._config.n_particles,
            ess_threshold=self._config.ess_threshold,
            resampling_method=self._config.resampling_method,
        )

        # Log diagnostics
        logger.info(
            f"Filtering complete: final ESS={info_history.ess[-1]:.1f}, "
            f"log-likelihood={self._filter_state.log_likelihood:.4f}"
        )

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        n_samples: int = 1000,
        quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> ForecastResult:
        """Generate forecasts.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        n_samples : int
            Number of forecast samples.
        quantiles : tuple
            Quantiles to compute.

        Returns
        -------
        result : ForecastResult
            Forecast results.
        """
        if not self._is_fitted:
            raise RuntimeError("Agent must be fitted before prediction")

        logger.info(f"Generating {horizon}-step forecast with {n_samples} samples")

        self._key, forecast_key = jax.random.split(self._key)

        # Propagate particles forward
        forecast_particles = self._propagate_particles(forecast_key, horizon, n_samples)

        # Compute statistics
        mean = jnp.mean(forecast_particles, axis=0)
        std = jnp.std(forecast_particles, axis=0)
        quantile_dict = {
            q: jnp.percentile(forecast_particles, q * 100, axis=0) for q in quantiles
        }

        # Generate future timestamps
        future_timestamps = None
        if self._timestamps is not None:
            try:
                freq = pd.infer_freq(self._timestamps)
                if freq:
                    future_timestamps = pd.date_range(
                        start=self._timestamps[-1],
                        periods=horizon + 1,
                        freq=freq,
                    )[1:]
            except Exception:
                pass

        return ForecastResult(
            mean=mean,
            std=std,
            quantiles=quantile_dict,
            particles=forecast_particles,
            timestamps=future_timestamps,
        )

    def _propagate_particles(
        self,
        key: PRNGKeyArray,
        horizon: int,
        n_samples: int,
    ) -> Float[Array, "n_samples horizon obs_dim"]:
        """Propagate particles forward to generate forecast samples.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key.
        horizon : int
            Forecast horizon.
        n_samples : int
            Number of samples.

        Returns
        -------
        forecast : Array
            Forecast samples [n_samples, horizon, obs_dim].
        """
        resample_key, propagate_key = jax.random.split(key)

        # Resample to get representative particles
        indices = systematic_resample(resample_key, self._filter_state.log_weights)
        selected_indices = jax.random.choice(
            resample_key, indices, shape=(n_samples,), replace=True
        )
        current_states = self._filter_state.particles[selected_indices]

        # Propagate forward
        def propagate_one_step(carry, t):
            states, step_key = carry
            step_key, next_key = jax.random.split(step_key)

            # State transition
            trans_keys = jax.random.split(step_key, n_samples)

            def sample_transition(inputs):
                k, state = inputs
                trans_dist = self._model.transition_distribution(
                    self._params, state, self._filter_state.step + t
                )
                return jnp.atleast_1d(trans_dist.sample(k))

            new_states = vmap(sample_transition)((trans_keys, states))

            # Observation sampling
            obs_keys = jax.random.split(next_key, n_samples)

            def sample_observation(inputs):
                k, state = inputs
                emit_dist = self._model.emission_distribution(
                    self._params, state, self._filter_state.step + t + 1
                )
                return jnp.atleast_1d(emit_dist.sample(k))

            observations = vmap(sample_observation)((obs_keys, new_states))

            return (new_states, next_key), observations

        from jax import lax

        _, forecast_obs = lax.scan(
            propagate_one_step,
            (current_states, propagate_key),
            jnp.arange(horizon),
        )

        # Transpose: [horizon, n_samples, obs_dim] -> [n_samples, horizon, obs_dim]
        return jnp.transpose(forecast_obs, (1, 0, 2))

    def update(
        self,
        observation: Float[Array, " obs_dim"],
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """Online update with a single observation.

        Parameters
        ----------
        observation : Array
            New observation.
        timestamp : Timestamp, optional
            Timestamp.
        """
        if not self._is_fitted:
            raise RuntimeError("Agent must be fitted before online update")

        self._key, update_key = jax.random.split(self._key)

        observation = jnp.atleast_1d(observation)

        self._filter_state, info = bootstrap_step(
            update_key,
            self._filter_state,
            observation,
            self._model,
            self._params,
            self._config.ess_threshold,
            self._config.resampling_method,
        )

        if timestamp is not None and self._timestamps is not None:
            self._timestamps = self._timestamps.append(pd.DatetimeIndex([timestamp]))

        logger.debug(f"Online update: ESS={info.ess:.1f}, resampled={info.resampled}")

    def get_filter_state(self) -> SMCState:
        """Get current filter state.

        Returns
        -------
        state : SMCState
            Current SMC state.
        """
        if self._filter_state is None:
            raise RuntimeError("No filter state available")
        return self._filter_state
