"""GARCH models for SMC.

This module provides GARCH family models in state space form.
While GARCH models have deterministic volatility dynamics,
SMC can be useful for parameter learning and missing data handling.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Distribution, Normal

__all__ = [
    "GARCHParams",
    "GARCHModel",
    "GJRGARCHParams",
    "GJRGARCHModel",
]


@chex.dataclass(frozen=True)
class GARCHParams(ModelParams):
    """Parameters for GARCH(p, q) model.

    The GARCH(1,1) model is:
        y_t = sigma_t * epsilon_t,  epsilon_t ~ N(0, 1)
        sigma_t^2 = omega + alpha * y_{t-1}^2 + beta * sigma_{t-1}^2

    Constraints:
        omega > 0
        alpha >= 0, beta >= 0
        alpha + beta < 1 (stationarity)

    Attributes
    ----------
    omega : float
        Constant term in variance equation.
    alpha : Array
        ARCH coefficients.
    beta : Array
        GARCH coefficients.
    sigma0 : float
        Initial volatility.
    """

    omega: float
    alpha: Float[Array, " q"]
    beta: Float[Array, " p"]
    sigma0: float


@chex.dataclass(frozen=True)
class GJRGARCHParams(ModelParams):
    """Parameters for GJR-GARCH model (asymmetric GARCH).

    The GJR-GARCH(1,1) model is:
        y_t = sigma_t * epsilon_t,  epsilon_t ~ N(0, 1)
        sigma_t^2 = omega + alpha * y_{t-1}^2 + gamma * y_{t-1}^2 * I(y_{t-1} < 0) + beta * sigma_{t-1}^2

    The gamma term captures the leverage effect.

    Attributes
    ----------
    omega : float
        Constant term.
    alpha : float
        ARCH coefficient.
    gamma : float
        Asymmetry coefficient (leverage effect).
    beta : float
        GARCH coefficient.
    sigma0 : float
        Initial volatility.
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    sigma0: float


class GARCHModel(StateSpaceModel):
    """GARCH(p, q) model in state space form.

    State: [sigma_t^2, y_{t-1}^2, ..., y_{t-q+1}^2, sigma_{t-1}^2, ..., sigma_{t-p+1}^2]

    For GARCH(1,1), state is simply [sigma_t^2, y_{t-1}^2].
    """

    def __init__(self, order: tuple[int, int] = (1, 1)):
        """Initialize GARCH model.

        Parameters
        ----------
        order : tuple
            (p, q) where p=GARCH order, q=ARCH order.
        """
        self.p, self.q = order
        # State: [current_var, past_y^2 (q), past_var (p-1)]
        self._state_dim = 1 + self.q + max(0, self.p - 1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: GARCHParams) -> Distribution:
        """Return initial state distribution.

        For GARCH, the initial state is deterministic given sigma0.
        We use a very small variance to make it nearly deterministic.
        """
        # Initial state: [sigma0^2, 0, ..., 0] (no past observations)
        m0 = jnp.zeros(self._state_dim)
        m0 = m0.at[0].set(params.sigma0**2)

        # Very small variance (essentially deterministic)
        scale = 1e-8 * jnp.ones(self._state_dim)
        return _DiagonalNormal(loc=m0, scale=scale)

    def transition_distribution(
        self,
        params: GARCHParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution.

        Note: GARCH has deterministic volatility dynamics.
        The stochasticity comes from the observation equation.
        """
        # This will be updated after observing y_t
        # For now, return current state (will be modified in the filter)
        return _DiagonalNormal(loc=state, scale=1e-8 * jnp.ones(self._state_dim))

    def emission_distribution(
        self,
        params: GARCHParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        sigma_sq = state[0]
        sigma = jnp.sqrt(jnp.maximum(sigma_sq, 1e-8))
        return Normal(loc=0.0, scale=sigma)

    def update_state(
        self,
        params: GARCHParams,
        state: Float[Array, " state_dim"],
        observation: Float[Array, " 1"],
    ) -> Float[Array, " state_dim"]:
        """Update state after observing y_t.

        This is the GARCH-specific update for the variance equation.
        """
        y = observation[0]
        y_sq = y**2

        # Current variance components
        sigma_sq = state[0]

        # Compute new variance
        new_sigma_sq = params.omega

        # ARCH terms
        new_sigma_sq = new_sigma_sq + params.alpha[0] * y_sq
        for i in range(1, self.q):
            if i < self._state_dim - 1:
                new_sigma_sq = new_sigma_sq + params.alpha[i] * state[1 + i - 1]

        # GARCH terms
        new_sigma_sq = new_sigma_sq + params.beta[0] * sigma_sq
        for i in range(1, self.p):
            if self.q + i < self._state_dim:
                new_sigma_sq = new_sigma_sq + params.beta[i] * state[self.q + i]

        # Build new state
        new_state = jnp.zeros(self._state_dim)
        new_state = new_state.at[0].set(new_sigma_sq)

        # Shift past y^2
        if self.q > 0:
            new_state = new_state.at[1].set(y_sq)
            for i in range(1, self.q):
                if 1 + i < self._state_dim and i < self._state_dim - 1:
                    new_state = new_state.at[1 + i].set(state[i])

        # Shift past sigma^2
        if self.p > 1:
            for i in range(self.p - 1):
                idx = 1 + self.q + i
                if idx < self._state_dim:
                    src_idx = self.q + i if i == 0 else idx - 1
                    new_state = new_state.at[idx].set(state[0] if i == 0 else state[src_idx])

        return new_state


class GJRGARCHModel(StateSpaceModel):
    """GJR-GARCH(1,1) model with leverage effect.

    State: [sigma_t^2, y_{t-1}, sigma_{t-1}^2]
    We keep y_{t-1} (not squared) to compute the indicator.
    """

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: GJRGARCHParams) -> Distribution:
        """Return initial state distribution."""
        m0 = jnp.array([params.sigma0**2, 0.0, params.sigma0**2])
        return _DiagonalNormal(loc=m0, scale=1e-8 * jnp.ones(3))

    def transition_distribution(
        self,
        params: GJRGARCHParams,
        state: Float[Array, " 3"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution."""
        return _DiagonalNormal(loc=state, scale=1e-8 * jnp.ones(3))

    def emission_distribution(
        self,
        params: GJRGARCHParams,
        state: Float[Array, " 3"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        sigma_sq = state[0]
        sigma = jnp.sqrt(jnp.maximum(sigma_sq, 1e-8))
        return Normal(loc=0.0, scale=sigma)

    def update_state(
        self,
        params: GJRGARCHParams,
        state: Float[Array, " 3"],
        observation: Float[Array, " 1"],
    ) -> Float[Array, " 3"]:
        """Update state with asymmetric GARCH dynamics."""
        y = observation[0]
        y_sq = y**2
        sigma_sq = state[0]

        # Indicator for negative returns
        indicator = jnp.where(y < 0, 1.0, 0.0)

        # GJR-GARCH variance equation
        new_sigma_sq = (
            params.omega
            + params.alpha * y_sq
            + params.gamma * y_sq * indicator
            + params.beta * sigma_sq
        )

        return jnp.array([new_sigma_sq, y, sigma_sq])


class _DiagonalNormal(Distribution):
    """Diagonal multivariate normal (independent components)."""

    def __init__(self, loc: Float[Array, " dim"], scale: Float[Array, " dim"]):
        self.loc = loc
        self.scale = scale
        self._dim = loc.shape[0]

    def sample(self, key):
        import jax

        z = jax.random.normal(key, shape=(self._dim,))
        return self.loc + self.scale * z

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return jnp.sum(-0.5 * z**2 - jnp.log(self.scale) - 0.5 * jnp.log(2 * jnp.pi))
