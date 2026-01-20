"""Factor models for SMC.

This module provides dynamic factor models for multivariate time series.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from smcs.models.base import ModelParams, StateSpaceModel
from smcs.models.distributions import Distribution, MultivariateNormal

__all__ = [
    "DynamicFactorParams",
    "DynamicFactorModel",
]


@chex.dataclass(frozen=True)
class DynamicFactorParams(ModelParams):
    """Parameters for Dynamic Factor Model.

    The dynamic factor model is:
        y_t = Lambda @ f_t + epsilon_t,  epsilon_t ~ N(0, Psi)
        f_t = Phi @ f_{t-1} + eta_t,  eta_t ~ N(0, Q)

    Where:
        y_t: n_obs dimensional observation
        f_t: n_factors dimensional latent factors
        Lambda: factor loadings matrix [n_obs, n_factors]
        Psi: idiosyncratic variance (diagonal) [n_obs, n_obs]
        Phi: factor dynamics [n_factors, n_factors]
        Q: factor innovation covariance [n_factors, n_factors]

    Attributes
    ----------
    Lambda : Array
        Factor loadings matrix.
    Psi : Array
        Idiosyncratic variance matrix (typically diagonal).
    Phi : Array
        Factor dynamics matrix.
    Q : Array
        Factor innovation covariance.
    f0 : Array
        Initial factor mean.
    P0 : Array
        Initial factor covariance.
    """

    Lambda: Float[Array, "n_obs n_factors"]
    Psi: Float[Array, "n_obs n_obs"]
    Phi: Float[Array, "n_factors n_factors"]
    Q: Float[Array, "n_factors n_factors"]
    f0: Float[Array, " n_factors"]
    P0: Float[Array, "n_factors n_factors"]


class DynamicFactorModel(StateSpaceModel):
    """Dynamic Factor Model.

    Factors follow VAR(1) dynamics and observations are linear
    combinations of factors plus idiosyncratic noise.
    """

    def __init__(self, n_factors: int, n_obs: int):
        """Initialize Dynamic Factor Model.

        Parameters
        ----------
        n_factors : int
            Number of latent factors.
        n_obs : int
            Number of observed variables.
        """
        self._n_factors = n_factors
        self._n_obs = n_obs

    @property
    def state_dim(self) -> int:
        return self._n_factors

    @property
    def obs_dim(self) -> int:
        return self._n_obs

    def initial_distribution(self, params: DynamicFactorParams) -> Distribution:
        """Return initial factor distribution."""
        return MultivariateNormal(loc=params.f0, covariance_matrix=params.P0)

    def transition_distribution(
        self,
        params: DynamicFactorParams,
        state: Float[Array, " n_factors"],
        t: int | None = None,
    ) -> Distribution:
        """Return factor transition distribution."""
        mean = params.Phi @ state
        return MultivariateNormal(loc=mean, covariance_matrix=params.Q)

    def emission_distribution(
        self,
        params: DynamicFactorParams,
        state: Float[Array, " n_factors"],
        t: int | None = None,
    ) -> Distribution:
        """Return observation distribution."""
        mean = params.Lambda @ state
        return MultivariateNormal(loc=mean, covariance_matrix=params.Psi)


def estimate_factor_loadings(
    data: Float[Array, "n_obs n_timesteps"],
    n_factors: int,
) -> tuple[Float[Array, "n_obs n_factors"], Float[Array, " n_factors"]]:
    """Estimate factor loadings using PCA.

    Parameters
    ----------
    data : Array
        Observation data [n_obs, n_timesteps].
    n_factors : int
        Number of factors to extract.

    Returns
    -------
    loadings : Array
        Factor loadings [n_obs, n_factors].
    explained_variance : Array
        Variance explained by each factor.
    """
    # Demean
    data_centered = data - jnp.mean(data, axis=1, keepdims=True)

    # Covariance matrix
    cov = data_centered @ data_centered.T / (data.shape[1] - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Sort by decreasing eigenvalue
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract top n_factors
    loadings = eigenvectors[:, :n_factors] * jnp.sqrt(eigenvalues[:n_factors])
    explained_variance = eigenvalues[:n_factors]

    return loadings, explained_variance
