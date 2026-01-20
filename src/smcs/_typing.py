"""Type definitions for smcs library.

This module provides type aliases using jaxtyping for type-safe SMC operations.
"""

from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

__all__ = [
    # Basic types
    "Scalar",
    "Vector",
    "Matrix",
    # SMC types
    "Particles",
    "LogWeights",
    "Ancestors",
    # Observation types
    "Observations",
    "SingleObservation",
    # State types
    "StateVector",
    "CovarianceMatrix",
    # Re-exports
    "Array",
    "Float",
    "Int",
    "Bool",
    "PRNGKeyArray",
]

# Basic type aliases
Scalar: TypeAlias = Float[Array, ""]
Vector: TypeAlias = Float[Array, " dim"]
Matrix: TypeAlias = Float[Array, "rows cols"]

# SMC-specific types
Particles: TypeAlias = Float[Array, "n_particles state_dim"]
LogWeights: TypeAlias = Float[Array, " n_particles"]
Ancestors: TypeAlias = Int[Array, " n_particles"]

# Observation data types
Observations: TypeAlias = Float[Array, "n_timesteps obs_dim"]
SingleObservation: TypeAlias = Float[Array, " obs_dim"]

# Parameter types (model-dependent)
StateVector: TypeAlias = Float[Array, " state_dim"]
CovarianceMatrix: TypeAlias = Float[Array, "dim dim"]
