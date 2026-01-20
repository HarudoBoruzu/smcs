"""ARIMA/SARIMA models in state space form.

This module provides ARIMA models using Harvey's state space representation.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Distribution, MultivariateNormal, Normal

__all__ = [
    "ARIMAParams",
    "ARIMAModel",
]


@chex.dataclass(frozen=True)
class ARIMAParams(ModelParams):
    """Parameters for ARIMA(p, d, q) model.

    Attributes
    ----------
    ar_coeffs : Array
        AR coefficients [phi_1, ..., phi_p].
    ma_coeffs : Array
        MA coefficients [theta_1, ..., theta_q].
    sigma : float
        Innovation standard deviation.
    d : int
        Differencing order.
    m0 : Array
        Initial state mean.
    C0 : Array
        Initial state covariance.
    """

    ar_coeffs: Float[Array, " p"]
    ma_coeffs: Float[Array, " q"]
    sigma: float
    d: int
    m0: Float[Array, " state_dim"]
    C0: Float[Array, "state_dim state_dim"]


def _build_companion_matrix(
    ar_coeffs: Float[Array, " p"],
    ma_coeffs: Float[Array, " q"],
) -> Float[Array, "r r"]:
    """Build companion form transition matrix for ARIMA.

    Parameters
    ----------
    ar_coeffs : Array
        AR coefficients.
    ma_coeffs : Array
        MA coefficients.

    Returns
    -------
    G : Array
        Companion form transition matrix.
    """
    p = ar_coeffs.shape[0] if ar_coeffs.size > 0 else 0
    q = ma_coeffs.shape[0] if ma_coeffs.size > 0 else 0
    r = max(p, q + 1) if (p > 0 or q > 0) else 1

    G = jnp.zeros((r, r))

    # First column: AR coefficients (padded with zeros)
    ar_padded = jnp.zeros(r)
    if p > 0:
        ar_padded = ar_padded.at[:p].set(ar_coeffs)
    G = G.at[:, 0].set(ar_padded)

    # Shift matrix in remaining columns
    if r > 1:
        G = G.at[:-1, 1:].set(jnp.eye(r - 1))

    return G


def _build_noise_vector(
    ma_coeffs: Float[Array, " q"],
    state_dim: int,
) -> Float[Array, " state_dim"]:
    """Build noise loading vector for ARIMA.

    Parameters
    ----------
    ma_coeffs : Array
        MA coefficients.
    state_dim : int
        State dimension.

    Returns
    -------
    R : Array
        Noise loading vector.
    """
    q = ma_coeffs.shape[0] if ma_coeffs.size > 0 else 0
    R = jnp.zeros(state_dim)
    R = R.at[0].set(1.0)

    if q > 0:
        R = R.at[1 : q + 1].set(ma_coeffs)

    return R


class ARIMAModel(StateSpaceModel):
    """ARIMA(p, d, q) model in state space form.

    Uses Harvey's state space representation where the state vector
    has dimension r = max(p+d, q+1).
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)):
        """Initialize ARIMA model.

        Parameters
        ----------
        order : tuple
            (p, d, q) where p=AR order, d=differencing, q=MA order.
        """
        self.p, self.d, self.q = order
        self._state_dim = max(self.p + self.d, self.q + 1) if (self.p > 0 or self.q > 0) else 1

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return 1

    def initial_distribution(self, params: ARIMAParams) -> Distribution:
        """Return initial state distribution."""
        return MultivariateNormal(loc=params.m0, covariance_matrix=params.C0)

    def transition_distribution(
        self,
        params: ARIMAParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return transition distribution."""
        G = _build_companion_matrix(params.ar_coeffs, params.ma_coeffs)
        R = _build_noise_vector(params.ma_coeffs, self._state_dim)

        mean = G @ state
        # Noise covariance: sigma^2 * R @ R'
        W = params.sigma**2 * jnp.outer(R, R)
        # Add small regularization for numerical stability
        W = W + 1e-8 * jnp.eye(self._state_dim)

        return MultivariateNormal(loc=mean, covariance_matrix=W)

    def emission_distribution(
        self,
        params: ARIMAParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> Distribution:
        """Return emission distribution."""
        # Observation is the first state element
        return Normal(loc=state[0], scale=1e-8)  # Essentially deterministic


def partial_autocorr_to_ar(
    pacf: Float[Array, " p"],
) -> Float[Array, " p"]:
    """Convert partial autocorrelations to AR coefficients.

    Uses the Durbin-Levinson algorithm to ensure stationarity.
    PACF values in (-1, 1) guarantee stationary AR coefficients.

    Parameters
    ----------
    pacf : Array
        Partial autocorrelation coefficients in (-1, 1).

    Returns
    -------
    ar_coeffs : Array
        Stationary AR coefficients.
    """
    p = pacf.shape[0]
    if p == 0:
        return jnp.array([])

    phi = jnp.zeros((p, p))

    # Initialize
    phi = phi.at[0, 0].set(pacf[0])

    # Durbin-Levinson recursion
    for k in range(1, p):
        phi = phi.at[k, k].set(pacf[k])
        for j in range(k):
            phi = phi.at[k, j].set(phi[k - 1, j] - pacf[k] * phi[k - 1, k - 1 - j])

    return phi[p - 1, :p]
