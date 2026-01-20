"""Base protocol and types for state space models.

This module defines the StateSpaceModel protocol that all models must implement.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import chex
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from smcs.models.distributions import Distribution

__all__ = [
    "StateSpaceModel",
    "ModelParams",
]


@chex.dataclass(frozen=True)
class ModelParams:
    """Base class for model parameters.

    All model-specific parameter classes should inherit from this.
    """

    pass


@runtime_checkable
class StateSpaceModel(Protocol):
    """Protocol defining the interface for state space models.

    A state space model consists of:
    - Initial distribution: p(x_0)
    - Transition distribution: p(x_t | x_{t-1})
    - Emission distribution: p(y_t | x_t)
    - (Optional) Proposal distribution: q(x_t | x_{t-1}, y_t)
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        ...

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of the observation vector."""
        ...

    @abstractmethod
    def initial_distribution(self, params: ModelParams) -> "Distribution":
        """Return the initial state distribution p(x_0).

        Parameters
        ----------
        params : ModelParams
            Model parameters.

        Returns
        -------
        dist : Distribution
            Initial state distribution.
        """
        ...

    @abstractmethod
    def transition_distribution(
        self,
        params: ModelParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> "Distribution":
        """Return the transition distribution p(x_t | x_{t-1}).

        Parameters
        ----------
        params : ModelParams
            Model parameters.
        state : Array
            Current state x_{t-1}.
        t : int, optional
            Time step (for time-varying models).

        Returns
        -------
        dist : Distribution
            Transition distribution.
        """
        ...

    @abstractmethod
    def emission_distribution(
        self,
        params: ModelParams,
        state: Float[Array, " state_dim"],
        t: int | None = None,
    ) -> "Distribution":
        """Return the emission distribution p(y_t | x_t).

        Parameters
        ----------
        params : ModelParams
            Model parameters.
        state : Array
            Current state x_t.
        t : int, optional
            Time step (for time-varying models).

        Returns
        -------
        dist : Distribution
            Emission distribution.
        """
        ...

    def proposal_distribution(
        self,
        params: ModelParams,
        state: Float[Array, " state_dim"],
        observation: Float[Array, " obs_dim"],
        t: int | None = None,
    ) -> "Distribution":
        """Return the proposal distribution q(x_t | x_{t-1}, y_t).

        By default, returns the transition distribution (bootstrap proposal).
        Override for locally optimal or other informed proposals.

        Parameters
        ----------
        params : ModelParams
            Model parameters.
        state : Array
            Current state x_{t-1}.
        observation : Array
            Current observation y_t.
        t : int, optional
            Time step.

        Returns
        -------
        dist : Distribution
            Proposal distribution.
        """
        return self.transition_distribution(params, state, t)
