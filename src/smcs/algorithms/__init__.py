"""SMC algorithm implementations.

This module provides various SMC algorithms:
- Bootstrap Particle Filter
- Auxiliary Particle Filter
- Liu-West Filter (online parameter learning)
- Storvik Filter (sufficient statistics)
- SMC² (nested SMC for parameters)
- PMMH (Particle MCMC)
- Waste-Free SMC
"""

from smcs.algorithms.auxiliary import auxiliary_step, run_auxiliary_filter
from smcs.algorithms.bootstrap import (
    bootstrap_step,
    initialize_particles,
    run_bootstrap_filter,
)
from smcs.algorithms.liu_west import LiuWestState, liu_west_step, run_liu_west_filter
from smcs.algorithms.pmcmc import PMMHResult, PMMHState, pmmh_step, random_walk_proposal, run_pmmh
from smcs.algorithms.smc2 import SMC2State, run_smc2, smc2_step
from smcs.algorithms.storvik import StorvikState, run_storvik_filter, storvik_step
from smcs.algorithms.waste_free import (
    WasteFreeState,
    metropolis_hastings_kernel,
    run_waste_free_smc,
    waste_free_step,
)

__all__ = [
    # Bootstrap
    "bootstrap_step",
    "run_bootstrap_filter",
    "initialize_particles",
    # Auxiliary
    "auxiliary_step",
    "run_auxiliary_filter",
    # Liu-West
    "LiuWestState",
    "liu_west_step",
    "run_liu_west_filter",
    # Storvik
    "StorvikState",
    "storvik_step",
    "run_storvik_filter",
    # SMC²
    "SMC2State",
    "smc2_step",
    "run_smc2",
    # PMCMC
    "PMMHState",
    "PMMHResult",
    "pmmh_step",
    "run_pmmh",
    "random_walk_proposal",
    # Waste-Free
    "WasteFreeState",
    "waste_free_step",
    "run_waste_free_smc",
    "metropolis_hastings_kernel",
]
