"""Monte Carlo methods.

This module provides classical Monte Carlo techniques including:
- Bootstrap resampling methods
- MCMC samplers (Slice, MH, HMC, NUTS)
- Importance sampling
- Quasi-Monte Carlo sequences
"""

from __future__ import annotations

from smcs.montecarlo.bootstrap import (
    block_bootstrap,
    bootstrap_ci,
    circular_block_bootstrap,
    jackknife,
    moving_block_bootstrap,
    ordinary_bootstrap,
    parametric_bootstrap,
    residual_bootstrap,
    stationary_bootstrap,
    wild_bootstrap,
)
from smcs.montecarlo.importance import (
    adaptive_importance_sampling,
    compute_ess_is,
    compute_is_diagnostics,
    importance_sample,
    multiple_importance_sampling,
    self_normalized_is,
)
from smcs.montecarlo.mcmc import (
    gibbs_sample,
    hmc_step,
    metropolis_hastings,
    nuts_step,
    random_walk_metropolis,
    run_gibbs_sampler,
    run_hmc,
    run_metropolis_hastings,
    run_nuts,
    run_random_walk_metropolis,
    run_slice_sampler,
    slice_sample,
)
from smcs.montecarlo.quasi import (
    halton_sequence,
    latin_hypercube,
    randomized_halton,
    randomized_lhs,
    sobol_sequence,
)

__all__ = [
    # Bootstrap methods
    "ordinary_bootstrap",
    "block_bootstrap",
    "moving_block_bootstrap",
    "circular_block_bootstrap",
    "stationary_bootstrap",
    "wild_bootstrap",
    "residual_bootstrap",
    "parametric_bootstrap",
    "bootstrap_ci",
    "jackknife",
    # MCMC methods
    "slice_sample",
    "run_slice_sampler",
    "metropolis_hastings",
    "run_metropolis_hastings",
    "random_walk_metropolis",
    "run_random_walk_metropolis",
    "gibbs_sample",
    "run_gibbs_sampler",
    "hmc_step",
    "run_hmc",
    "nuts_step",
    "run_nuts",
    # Importance sampling
    "importance_sample",
    "self_normalized_is",
    "multiple_importance_sampling",
    "adaptive_importance_sampling",
    "compute_ess_is",
    "compute_is_diagnostics",
    # Quasi-Monte Carlo
    "halton_sequence",
    "sobol_sequence",
    "latin_hypercube",
    "randomized_halton",
    "randomized_lhs",
]
