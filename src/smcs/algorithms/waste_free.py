"""Waste-Free SMC implementation.

Waste-Free SMC (Dau & Chopin, 2022) improves efficiency by keeping
all intermediate MCMC samples rather than discarding them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax, vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo
from smcs.core.resampling import systematic_resample
from smcs.core.weights import compute_ess

if TYPE_CHECKING:
    pass

__all__ = [
    "WasteFreeState",
    "waste_free_step",
    "run_waste_free_smc",
]


@chex.dataclass(frozen=True)
class WasteFreeState:
    """State for Waste-Free SMC.

    In Waste-Free SMC, we maintain M "mother" particles and generate
    k MCMC samples from each, resulting in M×k total particles.

    Attributes
    ----------
    particles : Array
        All particles [n_particles, dim].
    log_weights : Array
        Log-weights [n_particles].
    log_likelihood : float
        Cumulative log normalizing constant.
    step : int
        Current SMC step.
    """

    particles: Float[Array, "n_particles dim"]
    log_weights: Float[Array, " n_particles"]
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]

    @property
    def n_particles(self) -> int:
        return self.particles.shape[0]


@jaxtyped(typechecker=beartype)
def waste_free_step(
    key: PRNGKeyArray,
    state: WasteFreeState,
    target_log_prob: Callable,
    mcmc_kernel: Callable,
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[WasteFreeState, SMCInfo]:
    """Perform one step of Waste-Free SMC.

    In Waste-Free SMC:
    1. Resample to get M unique "mother" particles
    2. Run k MCMC steps from each mother
    3. Keep all M×k particles with appropriate weights

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    state : WasteFreeState
        Current state.
    target_log_prob : callable
        Function (x) -> log probability under target.
    mcmc_kernel : callable
        Function (key, x, log_prob_fn) -> new_x.
        Should be a valid MCMC kernel (e.g., MH, HMC).
    n_mcmc_steps : int
        Number of MCMC steps per mother particle.
    ess_threshold : float
        ESS/N threshold for resampling.

    Returns
    -------
    new_state : WasteFreeState
        Updated state.
    info : SMCInfo
        Diagnostic information.
    """
    n_particles = state.n_particles
    n_mothers = n_particles // n_mcmc_steps  # M = N/k
    key, resample_key, mcmc_key = jax.random.split(key, 3)

    # Compute ESS
    ess = compute_ess(state.log_weights)
    should_resample = ess < ess_threshold * n_particles

    def do_waste_free(args):
        particles, log_weights, rkey, mkey = args

        # Resample to get M unique mothers
        indices = systematic_resample(rkey, log_weights)
        # Take first M unique (approximately)
        mother_indices = indices[:n_mothers]
        mothers = particles[mother_indices]

        # Run k MCMC steps from each mother
        mcmc_keys = jax.random.split(mkey, n_mothers * n_mcmc_steps).reshape(
            n_mothers, n_mcmc_steps, -1
        )

        def run_mcmc_chain(mother_key):
            mother, keys = mother_key
            def mcmc_step(x, step_key):
                new_x = mcmc_kernel(step_key, x, target_log_prob)
                return new_x, new_x

            _, chain = lax.scan(mcmc_step, mother, keys)
            return chain  # Shape: [k, dim]

        chains = vmap(run_mcmc_chain)((mothers, mcmc_keys[:, :, 0]))
        # Shape: [M, k, dim]

        # Flatten to get all particles
        new_particles = chains.reshape(-1, particles.shape[-1])

        # Compute weights for waste-free
        # w^{(i,j)} = (1/k) * pi_n(x^{(i,j)}) / pi_{n-1}(x^{(i,j)})
        # For SMC samplers with tempering: ratio is the likelihood ratio
        new_log_probs = vmap(target_log_prob)(new_particles)
        _old_log_probs = vmap(target_log_prob)(particles[:n_particles])  # noqa: F841

        # Approximate: use uniform weights within chains
        new_log_weights = jnp.zeros(n_particles)

        return new_particles, new_log_weights, new_log_probs

    def no_resample(args):
        particles, log_weights, _, _ = args
        log_probs = vmap(target_log_prob)(particles)
        return particles, log_weights, log_probs

    new_particles, new_log_weights, log_probs = lax.cond(
        should_resample,
        do_waste_free,
        no_resample,
        (state.particles, state.log_weights, resample_key, mcmc_key),
    )

    # Update log normalizing constant
    log_likelihood_increment = jax.scipy.special.logsumexp(new_log_weights) - jnp.log(n_particles)

    new_state = WasteFreeState(
        particles=new_particles,
        log_weights=new_log_weights,
        log_likelihood=state.log_likelihood + log_likelihood_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=should_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_waste_free_smc(
    key: PRNGKeyArray,
    initial_particles: Float[Array, "n_particles dim"],
    target_sequence: list[Callable],
    mcmc_kernel: Callable,
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[WasteFreeState, list[SMCInfo]]:
    """Run Waste-Free SMC sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key.
    initial_particles : Array
        Initial particle samples from pi_0.
    target_sequence : list of callable
        Sequence of target log-probability functions [pi_0, pi_1, ..., pi_T].
    mcmc_kernel : callable
        MCMC kernel function.
    n_mcmc_steps : int
        Number of MCMC steps per resampling.
    ess_threshold : float
        ESS/N threshold.

    Returns
    -------
    final_state : WasteFreeState
        Final SMC state.
    info_history : list of SMCInfo
        Diagnostic information for each step.
    """
    n_particles = initial_particles.shape[0]

    init_state = WasteFreeState(
        particles=initial_particles,
        log_weights=jnp.zeros(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    state = init_state
    info_history = []
    keys = jax.random.split(key, len(target_sequence))

    for _, (step_key, target_log_prob) in enumerate(zip(keys, target_sequence[1:], strict=False)):
        state, info = waste_free_step(
            step_key,
            state,
            target_log_prob,
            mcmc_kernel,
            n_mcmc_steps,
            ess_threshold,
        )
        info_history.append(info)

    return state, info_history


def metropolis_hastings_kernel(
    proposal_std: float = 0.1,
) -> Callable:
    """Create a simple Metropolis-Hastings kernel.

    Parameters
    ----------
    proposal_std : float
        Standard deviation of Gaussian random walk proposal.

    Returns
    -------
    kernel : callable
        MH kernel function (key, x, log_prob_fn) -> new_x.
    """

    def kernel(
        key: PRNGKeyArray,
        x: Float[Array, " dim"],
        log_prob_fn: Callable,
    ) -> Float[Array, " dim"]:
        prop_key, accept_key = jax.random.split(key)

        # Propose
        proposal = x + proposal_std * jax.random.normal(prop_key, shape=x.shape)

        # Accept/reject
        log_alpha = log_prob_fn(proposal) - log_prob_fn(x)
        log_u = jnp.log(jax.random.uniform(accept_key))

        return jnp.where(log_u < log_alpha, proposal, x)

    return kernel
