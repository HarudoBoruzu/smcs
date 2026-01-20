"""Core SMC components: particles, resampling, and weight utilities."""

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import (
    ResamplingMethod,
    multinomial_resample,
    resample,
    residual_resample,
    stratified_resample,
    systematic_resample,
)
from smcs.core.weights import compute_ess, log_mean_exp, normalize_log_weights

__all__ = [
    # Particles
    "SMCState",
    "SMCInfo",
    # Resampling
    "systematic_resample",
    "multinomial_resample",
    "stratified_resample",
    "residual_resample",
    "resample",
    "ResamplingMethod",
    # Weights
    "compute_ess",
    "normalize_log_weights",
    "log_mean_exp",
]
