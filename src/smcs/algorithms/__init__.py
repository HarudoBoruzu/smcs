"""SMC algorithm implementations.

This module provides a comprehensive collection of Sequential Monte Carlo algorithms:

Filtering Algorithms:
- Bootstrap Particle Filter - Standard particle filter
- Auxiliary Particle Filter - Look-ahead proposal
- Guided/Optimal Particle Filter - Observation-dependent proposals
- Regularized Particle Filter - Kernel smoothing to prevent degeneracy
- Unscented Particle Filter - UKF-based proposals
- Ensemble Kalman Filter (EnKF/EAKF) - Ensemble-based filtering
- Rao-Blackwellized Particle Filter - Marginalized PF
- Island Particle Filter - Parallel islands

Parameter Learning:
- Liu-West Filter - Kernel density estimation
- Storvik Filter - Sufficient statistics
- SMC² - Nested SMC for parameters
- PMMH - Particle Marginal Metropolis-Hastings

Smoothing:
- Forward-Backward Smoothing (FFBS)
- Backward Simulation
- Two-Filter Smoother

Advanced Methods:
- Adaptive SMC - Automatic tempering
- SMC Samplers - Static parameter inference
- Annealed Importance Sampling (AIS)
- Tempered SMC - Tempering for multimodal targets
- Population Monte Carlo (PMC)
- Particle Gibbs / PGAS
- Nested SMC
- Twisted/Controlled SMC
- ABC-SMC - Likelihood-free inference
- Waste-Free SMC
"""

# Bootstrap
# ABC-SMC
from smcs.algorithms.abc_smc import (
    ABCSMCState,
    abc_smc_step,
    compute_adaptive_threshold,
    run_abc_smc,
)

# Adaptive SMC
from smcs.algorithms.adaptive import (
    AdaptiveState,
    adaptive_step,
    find_next_temperature,
    run_adaptive_smc,
)

# Annealed Importance Sampling
from smcs.algorithms.ais import (
    AISState,
    ais_step,
    estimate_log_normalizing_constant,
    run_ais,
)

# Auxiliary
from smcs.algorithms.auxiliary import auxiliary_step, run_auxiliary_filter
from smcs.algorithms.bootstrap import (
    bootstrap_step,
    initialize_particles,
    run_bootstrap_filter,
)

# Ensemble Kalman Filter
from smcs.algorithms.enkf import (
    EnKFState,
    eakf_step,
    enkf_step,
    run_eakf,
    run_enkf,
)

# Guided/Optimal
from smcs.algorithms.guided import (
    GuidedState,
    guided_step,
    run_guided_filter,
)

# Liu-West
from smcs.algorithms.liu_west import LiuWestState, liu_west_step, run_liu_west_filter

# Nested SMC / Island
from smcs.algorithms.nested import (
    IslandState,
    NestedSMCState,
    island_step,
    nested_smc_step,
    run_island_filter,
    run_nested_smc,
)

# Particle Gibbs
from smcs.algorithms.particle_gibbs import (
    PGState,
    conditional_smc_step,
    particle_gibbs_step,
    run_conditional_smc,
    run_particle_gibbs,
    run_pgas,
)

# PMCMC
from smcs.algorithms.pmcmc import (
    PMMHResult,
    PMMHState,
    pmmh_step,
    random_walk_proposal,
    run_pmmh,
)

# Rao-Blackwellized
from smcs.algorithms.rao_blackwell import (
    MarginalizedState,
    RBPFState,
    marginalized_step,
    rbpf_step,
    run_marginalized_filter,
    run_rbpf,
)

# Regularized
from smcs.algorithms.regularized import (
    RegularizedState,
    compute_kernel_bandwidth,
    regularized_step,
    run_regularized_filter,
)

# SMC²
from smcs.algorithms.smc2 import SMC2State, run_smc2, smc2_step

# SMC Samplers
from smcs.algorithms.smc_samplers import (
    SMCSamplersState,
    run_smc_samplers,
    smc_samplers_step,
)

# Smoothing
from smcs.algorithms.smoothing import (
    FilterHistory,
    SmoothingState,
    backward_sampling,
    forward_filter,
    run_backward_simulation,
    run_ffbs,
    run_two_filter_smoother,
)

# Storvik
from smcs.algorithms.storvik import StorvikState, run_storvik_filter, storvik_step

# Tempered SMC / PMC
from smcs.algorithms.tempered import (
    PMCState,
    TemperedState,
    pmc_step,
    run_pmc,
    run_tempered_smc,
    tempered_smc_step,
)

# Twisted / Controlled SMC
from smcs.algorithms.twisted import (
    TwistedState,
    controlled_step,
    learn_twisting_functions,
    run_controlled_smc,
    run_twisted_smc,
    twisted_step,
)

# Unscented Particle Filter
from smcs.algorithms.unscented import (
    UnscentedState,
    compute_sigma_points,
    run_unscented_filter,
    unscented_step,
    unscented_transform,
)

# Waste-Free
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
    # Guided
    "GuidedState",
    "guided_step",
    "run_guided_filter",
    # Regularized
    "RegularizedState",
    "regularized_step",
    "run_regularized_filter",
    "compute_kernel_bandwidth",
    # EnKF
    "EnKFState",
    "enkf_step",
    "run_enkf",
    "eakf_step",
    "run_eakf",
    # Unscented
    "UnscentedState",
    "unscented_step",
    "run_unscented_filter",
    "compute_sigma_points",
    "unscented_transform",
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
    # Adaptive
    "AdaptiveState",
    "adaptive_step",
    "run_adaptive_smc",
    "find_next_temperature",
    # SMC Samplers
    "SMCSamplersState",
    "smc_samplers_step",
    "run_smc_samplers",
    # AIS
    "AISState",
    "ais_step",
    "run_ais",
    "estimate_log_normalizing_constant",
    # Particle Gibbs
    "PGState",
    "conditional_smc_step",
    "particle_gibbs_step",
    "run_conditional_smc",
    "run_particle_gibbs",
    "run_pgas",
    # Smoothing
    "SmoothingState",
    "FilterHistory",
    "forward_filter",
    "backward_sampling",
    "run_ffbs",
    "run_backward_simulation",
    "run_two_filter_smoother",
    # ABC-SMC
    "ABCSMCState",
    "abc_smc_step",
    "run_abc_smc",
    "compute_adaptive_threshold",
    # Tempered / PMC
    "TemperedState",
    "PMCState",
    "tempered_smc_step",
    "run_tempered_smc",
    "pmc_step",
    "run_pmc",
    # Nested / Island
    "NestedSMCState",
    "IslandState",
    "nested_smc_step",
    "run_nested_smc",
    "island_step",
    "run_island_filter",
    # Twisted / Controlled
    "TwistedState",
    "twisted_step",
    "run_twisted_smc",
    "controlled_step",
    "run_controlled_smc",
    "learn_twisting_functions",
    # Rao-Blackwellized
    "RBPFState",
    "MarginalizedState",
    "rbpf_step",
    "run_rbpf",
    "marginalized_step",
    "run_marginalized_filter",
]
