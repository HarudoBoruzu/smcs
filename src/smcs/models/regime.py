"""Regime-switching models for SMC.

This module provides Markov-switching state space models.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Categorical, Distribution, MultivariateNormal, Normal

__all__ = [
    "MarkovSwitchingParams",
    "MarkovSwitchingModel",
    "MSLocalLevelParams",
    "MSLocalLevelModel",
]


@chex.dataclass(frozen=True)
class MarkovSwitchingParams(ModelParams):
    """Parameters for Markov-Switching models.

    Attributes
    ----------
    P : Array
        Transition probability matrix [n_regimes, n_regimes].
        P[i, j] = P(s_t = j | s_{t-1} = i)
    pi0 : Array
        Initial regime distribution [n_regimes].
    regime_params : tuple
        Tuple of regime-specific parameters.
    """

    P: Float[Array, "n_regimes n_regimes"]
    pi0: Float[Array, " n_regimes"]
    regime_params: tuple


@chex.dataclass(frozen=True)
class MSLocalLevelParams(ModelParams):
    """Parameters for Markov-Switching Local Level model.

    Attributes
    ----------
    P : Array
        Regime transition matrix.
    pi0 : Array
        Initial regime probabilities.
    sigma_obs : Array
        Observation noise std by regime [n_regimes].
    sigma_level : Array
        Level innovation std by regime [n_regimes].
    m0 : float
        Initial level mean.
    C0 : float
        Initial level variance.
    """

    P: Float[Array, "n_regimes n_regimes"]
    pi0: Float[Array, " n_regimes"]
    sigma_obs: Float[Array, " n_regimes"]
    sigma_level: Float[Array, " n_regimes"]
    m0: float
    C0: float


class MarkovSwitchingModel(StateSpaceModel):
    """Base class for Markov-Switching state space models.

    The state includes both the continuous state and the discrete regime.
    For SMC, we augment the particle state with regime information.
    """

    def __init__(self, n_regimes: int, continuous_state_dim: int, obs_dim: int):
        """Initialize Markov-Switching model.

        Parameters
        ----------
        n_regimes : int
            Number of regimes.
        continuous_state_dim : int
            Dimension of continuous state.
        obs_dim : int
            Dimension of observations.
        """
        self._n_regimes = n_regimes
        self._continuous_state_dim = continuous_state_dim
        self._obs_dim = obs_dim
        # State: [continuous_state, regime_indicator (one-hot)]
        self._state_dim = continuous_state_dim + n_regimes

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_regimes(self) -> int:
        return self._n_regimes

    @property
    def continuous_state_dim(self) -> int:
        return self._continuous_state_dim

    def _get_regime(self, state: Float[Array, " state_dim"]) -> int:
        """Extract regime index from state vector."""
        regime_probs = state[self._continuous_state_dim :]
        return jnp.argmax(regime_probs)

    def _get_continuous_state(
        self, state: Float[Array, " state_dim"]
    ) -> Float[Array, " continuous_dim"]:
        """Extract continuous state from state vector."""
        return state[: self._continuous_state_dim]

    def _build_state(
        self, continuous: Float[Array, " continuous_dim"], regime: int
    ) -> Float[Array, " state_dim"]:
        """Build full state from continuous state and regime."""
        regime_onehot = jnp.zeros(self._n_regimes).at[regime].set(1.0)
        return jnp.concatenate([continuous, regime_onehot])

    def regime_transition_distribution(
        self, params: MarkovSwitchingParams, current_regime: int
    ) -> Distribution:
        """Return distribution over next regime."""
        return Categorical(logits=jnp.log(params.P[current_regime] + 1e-10))


class MSLocalLevelModel(MarkovSwitchingModel):
    """Markov-Switching Local Level Model.

    Each regime has different observation and level noise variances.
    """

    def __init__(self, n_regimes: int = 2):
        """Initialize MS Local Level model.

        Parameters
        ----------
        n_regimes : int
            Number of regimes.
        """
        super().__init__(
            n_regimes=n_regimes,
            continuous_state_dim=1,
            obs_dim=1,
        )

    def initial_distribution(self, params: MSLocalLevelParams) -> Distribution:
        """Return initial state distribution."""
        # Sample initial regime
        # For simplicity, we'll use a composite distribution
        return _MSInitialDistribution(params, self._n_regimes)

    def transition_distribution(
        self,
        params: MSLocalLevelParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution."""
        continuous = self._get_continuous_state(state)
        regime = self._get_regime(state)
        return _MSTransitionDistribution(params, continuous, regime, self._n_regimes)

    def emission_distribution(
        self,
        params: MSLocalLevelParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        continuous = self._get_continuous_state(state)
        regime = self._get_regime(state)
        sigma = params.sigma_obs[regime]
        return Normal(loc=continuous[0], scale=sigma)


class _MSInitialDistribution(Distribution):
    """Initial distribution for MS models (samples regime + state)."""

    def __init__(self, params: MSLocalLevelParams, n_regimes: int):
        self.params = params
        self.n_regimes = n_regimes

    def sample(self, key):
        import jax

        key1, key2 = jax.random.split(key)

        # Sample regime
        regime = jax.random.categorical(key1, jnp.log(self.params.pi0 + 1e-10))

        # Sample continuous state
        level = self.params.m0 + jnp.sqrt(self.params.C0) * jax.random.normal(key2)

        # Build full state
        regime_onehot = jnp.zeros(self.n_regimes).at[regime].set(1.0)
        return jnp.concatenate([jnp.array([level]), regime_onehot])

    def log_prob(self, value):
        # Extract components
        level = value[0]
        regime_probs = value[1:]
        regime = jnp.argmax(regime_probs)

        # Log prob of regime
        log_p_regime = jnp.log(self.params.pi0[regime] + 1e-10)

        # Log prob of level
        z = (level - self.params.m0) / jnp.sqrt(self.params.C0)
        log_p_level = -0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi * self.params.C0)

        return log_p_regime + log_p_level


class _MSTransitionDistribution(Distribution):
    """Transition distribution for MS models."""

    def __init__(
        self,
        params: MSLocalLevelParams,
        continuous: Float[Array, " 1"],
        current_regime: int,
        n_regimes: int,
    ):
        self.params = params
        self.continuous = continuous
        self.current_regime = current_regime
        self.n_regimes = n_regimes

    def sample(self, key):
        import jax

        key1, key2 = jax.random.split(key)

        # Sample next regime
        next_regime = jax.random.categorical(
            key1, jnp.log(self.params.P[self.current_regime] + 1e-10)
        )

        # Sample continuous state transition given new regime
        sigma_level = self.params.sigma_level[next_regime]
        new_level = self.continuous[0] + sigma_level * jax.random.normal(key2)

        # Build full state
        regime_onehot = jnp.zeros(self.n_regimes).at[next_regime].set(1.0)
        return jnp.concatenate([jnp.array([new_level]), regime_onehot])

    def log_prob(self, value):
        # Extract components
        new_level = value[0]
        regime_probs = value[1:]
        next_regime = jnp.argmax(regime_probs)

        # Log prob of regime transition
        log_p_regime = jnp.log(self.params.P[self.current_regime, next_regime] + 1e-10)

        # Log prob of level transition
        sigma_level = self.params.sigma_level[next_regime]
        z = (new_level - self.continuous[0]) / sigma_level
        log_p_level = -0.5 * z**2 - jnp.log(sigma_level) - 0.5 * jnp.log(2 * jnp.pi)

        return log_p_regime + log_p_level
