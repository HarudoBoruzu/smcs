"""Configuration models for SMC algorithms.

This module provides Pydantic-based configuration classes for
SMC algorithms and agents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "SMCConfig",
    "AgentConfig",
]


class SMCConfig(BaseModel):
    """Configuration for SMC algorithms.

    Attributes
    ----------
    n_particles : int
        Number of particles (default 1000).
    seed : int
        Random seed for reproducibility.
    ess_threshold : float
        ESS/N ratio below which to resample (default 0.5).
    resampling_method : str
        Resampling algorithm to use.
    liu_west_delta : float
        Discount factor for Liu-West filter (default 0.98).
    n_mcmc_samples : int
        Number of MCMC samples for PMMH/SMC².
    n_burnin : int
        MCMC burn-in period.
    jit_compile : bool
        Whether to JIT compile SMC operations.
    use_checkpoint : bool
        Use gradient checkpointing for memory efficiency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Basic settings
    n_particles: int = Field(default=1000, gt=10, description="Number of particles")
    seed: int = Field(default=42, description="Random seed")

    # Resampling settings
    ess_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="ESS/N ratio threshold for resampling",
    )
    resampling_method: Literal["systematic", "multinomial", "stratified", "residual"] = Field(
        default="systematic",
        description="Resampling method",
    )

    # Liu-West filter settings
    liu_west_delta: float = Field(
        default=0.98,
        gt=0.9,
        lt=1.0,
        description="Liu-West discount factor",
    )

    # MCMC settings (for PMMH, SMC²)
    n_mcmc_samples: int = Field(default=5000, gt=100, description="MCMC samples")
    n_burnin: int = Field(default=1000, ge=0, description="Burn-in period")

    # Computation settings
    jit_compile: bool = Field(default=True, description="JIT compile operations")
    use_checkpoint: bool = Field(default=False, description="Use gradient checkpointing")

    @field_validator("n_burnin")
    @classmethod
    def burnin_less_than_samples(cls, v: int, info) -> int:
        """Validate that burn-in is less than total samples."""
        if "n_mcmc_samples" in info.data and v >= info.data["n_mcmc_samples"]:
            raise ValueError("n_burnin must be less than n_mcmc_samples")
        return v


class AgentConfig(BaseModel):
    """Configuration for forecasting agents.

    Attributes
    ----------
    smc : SMCConfig
        SMC algorithm configuration.
    model_type : str
        Type of state space model.
    arima_order : tuple
        ARIMA(p, d, q) order.
    garch_order : tuple
        GARCH(p, q) order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    smc: SMCConfig = Field(default_factory=SMCConfig)

    # Model-specific settings
    model_type: Literal["local_level", "local_trend", "arima", "garch", "sv"] = Field(
        default="local_level",
        description="State space model type",
    )

    # ARIMA settings
    arima_order: tuple[int, int, int] = Field(
        default=(1, 0, 0),
        description="ARIMA(p, d, q) order",
    )

    # GARCH settings
    garch_order: tuple[int, int] = Field(
        default=(1, 1),
        description="GARCH(p, q) order",
    )
