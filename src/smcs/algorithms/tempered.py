"""Tempered SMC and Population Monte Carlo (PMC).

These algorithms use tempering to bridge between prior and posterior,
enabling efficient sampling from complex multimodal distributions.
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, normalize_log_weights

__all__ = [
    "TemperedState",
    "PMCState",
    "tempered_smc_step",
    "run_tempered_smc",
    "pmc_step",
    "run_pmc",
]


@chex.dataclass(frozen=True)
class TemperedState(SMCState):
    """State for Tempered SMC.

    Attributes
    ----------
    particles : Array
        Parameter particles.
    log_weights : Array
        Log-weights.
    temperature : Array
        Current temperature.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Log normalizing constant estimate.
    step : Array
        Current step.
    """

    temperature: Float[Array, ""]


@chex.dataclass(frozen=True)
class PMCState(SMCState):
    """State for Population Monte Carlo.

    Attributes
    ----------
    particles : Array
        Parameter particles.
    log_weights : Array
        Log-weights.
    proposal_means : Array
        Proposal means for each particle.
    proposal_covs : Array
        Proposal covariances.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Log normalizing constant estimate.
    step : Array
        Current iteration.
    """

    proposal_means: Float[Array, "n_particles param_dim"]
    proposal_covs: Float[Array, "n_particles param_dim param_dim"]


@jaxtyped(typechecker=beartype)
def tempered_smc_step(
    key: PRNGKeyArray,
    state: TemperedState,
    next_temperature: Float[Array, ""],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    mcmc_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        tuple[Float[Array, ...], Float[Array, ""]],
    ],
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[TemperedState, SMCInfo]:
    """Perform one step of Tempered SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : TemperedState
        Current state.
    next_temperature : float
        Next temperature.
    log_likelihood_fn : Callable
        Log-likelihood function.
    mcmc_kernel : Callable
        MCMC kernel (key, particle, temp) -> (new_particle, accept).
    n_mcmc_steps : int
        Number of MCMC steps.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : TemperedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, mcmc_key = jax.random.split(key, 3)

    # Compute incremental weights
    log_liks = jax.vmap(log_likelihood_fn)(state.particles)
    delta_temp = next_temperature - state.temperature
    log_increments = delta_temp * log_liks
    new_log_weights = state.log_weights + log_increments

    # Normalizing constant contribution
    log_nc_increment = (
        jax.scipy.special.logsumexp(new_log_weights)
        - jax.scipy.special.logsumexp(state.log_weights)
    )

    # ESS and conditional resampling
    ess = compute_ess(new_log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, new_log_weights, method="systematic"),
        lambda: jnp.arange(n_particles),
    )

    resampled_particles = state.particles[ancestors]
    post_resample_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        normalize_log_weights(new_log_weights),
    )

    # MCMC mutation
    def mutate_one(key_i, particle):
        def mcmc_step(carry, _):
            x, k, total_accept = carry
            k, step_key = jax.random.split(k)
            x_new, accept = mcmc_kernel(step_key, x, next_temperature)
            return (x_new, k, total_accept + accept), None

        (final_x, _, total_accept), _ = jax.lax.scan(
            mcmc_step, (particle, key_i, jnp.array(0.0)), jnp.arange(n_mcmc_steps)
        )
        return final_x, total_accept / n_mcmc_steps

    mcmc_keys = jax.random.split(mcmc_key, n_particles)
    mutated_particles, acceptance_rates = jax.vmap(mutate_one)(
        mcmc_keys, resampled_particles
    )

    new_state = TemperedState(
        particles=mutated_particles,
        log_weights=post_resample_weights,
        temperature=next_temperature,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood + log_nc_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=jnp.mean(acceptance_rates),
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_tempered_smc(
    key: PRNGKeyArray,
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_likelihood_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    mcmc_kernel: Callable[
        [PRNGKeyArray, Float[Array, ...], Float[Array, ""]],
        tuple[Float[Array, ...], Float[Array, ""]],
    ],
    n_particles: int = 1000,
    temperatures: Float[Array, " n_temps"] | None = None,
    n_temperatures: int = 20,
    n_mcmc_steps: int = 5,
    ess_threshold: float = 0.5,
) -> tuple[TemperedState, SMCInfo]:
    """Run Tempered SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    prior_sample_fn : Callable
        Function to sample from prior.
    log_likelihood_fn : Callable
        Log-likelihood function.
    mcmc_kernel : Callable
        MCMC kernel.
    n_particles : int
        Number of particles.
    temperatures : Array, optional
        Temperature schedule (0 to 1).
    n_temperatures : int
        Number of temperatures if not provided.
    n_mcmc_steps : int
        MCMC steps per temperature.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    final_state : TemperedState
        Final state with posterior samples.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize from prior
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(prior_sample_fn)(init_keys)

    # Default temperature schedule
    if temperatures is None:
        temperatures = jnp.linspace(0.0, 1.0, n_temperatures + 1)[1:]

    initial_state = TemperedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        temperature=jnp.array(0.0),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        temp, step_key = inputs
        new_state, info = tempered_smc_step(
            step_key,
            state,
            temp,
            log_likelihood_fn,
            mcmc_kernel,
            n_mcmc_steps,
            ess_threshold,
        )
        return (new_state, key), info

    step_keys = jax.random.split(key, len(temperatures))

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        (temperatures, step_keys),
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=infos.acceptance_rate,
    )

    return final_state, combined_info


@jaxtyped(typechecker=beartype)
def pmc_step(
    key: PRNGKeyArray,
    state: PMCState,
    log_target_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    adapt_proposals: bool = True,
) -> tuple[PMCState, SMCInfo]:
    """Perform one iteration of Population Monte Carlo.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : PMCState
        Current state.
    log_target_fn : Callable
        Log-target density function.
    adapt_proposals : bool
        Whether to adapt proposal distributions.

    Returns
    -------
    new_state : PMCState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    param_dim = state.particles.shape[1]
    key, sample_key, resample_key = jax.random.split(key, 3)

    # Sample from proposal mixtures
    def sample_one(key_i, mean, cov):
        return jax.random.multivariate_normal(key_i, mean, cov)

    sample_keys = jax.random.split(sample_key, n_particles)
    new_particles = jax.vmap(sample_one)(
        sample_keys, state.proposal_means, state.proposal_covs
    )

    # Compute importance weights
    def compute_weight(particle):
        log_target = log_target_fn(particle)

        # Proposal is mixture of Gaussians
        def log_proposal_component(mean, cov):
            diff = particle - mean
            sign, logdet = jnp.linalg.slogdet(cov)
            mahal = jnp.dot(diff, jnp.linalg.solve(cov, diff))
            return -0.5 * (param_dim * jnp.log(2 * jnp.pi) + logdet + mahal)

        log_proposals = jax.vmap(log_proposal_component)(
            state.proposal_means, state.proposal_covs
        )
        log_mixture = jax.scipy.special.logsumexp(
            log_proposals + state.log_weights
        )

        return log_target - log_mixture

    new_log_weights = jax.vmap(compute_weight)(new_particles)
    ess = compute_ess(new_log_weights)

    # Resample
    ancestors = resample(resample_key, new_log_weights, method="systematic")

    # Adapt proposals if requested
    if adapt_proposals:
        # Use resampled particles to update proposals
        weights = jax.nn.softmax(new_log_weights)

        # Global adapted proposal (simplified: same for all particles)
        adapted_mean = jnp.sum(weights[:, None] * new_particles, axis=0)
        diff = new_particles - adapted_mean
        adapted_cov = jnp.sum(
            weights[:, None, None] * jnp.einsum("ij,ik->ijk", diff, diff), axis=0
        )
        # Add regularization
        adapted_cov = adapted_cov + 0.01 * jnp.eye(param_dim)

        new_proposal_means = jnp.tile(adapted_mean, (n_particles, 1))
        new_proposal_covs = jnp.tile(adapted_cov, (n_particles, 1, 1))
    else:
        new_proposal_means = state.proposal_means[ancestors]
        new_proposal_covs = state.proposal_covs[ancestors]

    new_state = PMCState(
        particles=new_particles[ancestors],
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        proposal_means=new_proposal_means,
        proposal_covs=new_proposal_covs,
        ancestors=ancestors,
        log_likelihood=state.log_likelihood
        + jax.scipy.special.logsumexp(new_log_weights)
        - jnp.log(n_particles),
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=jnp.array(True),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_pmc(
    key: PRNGKeyArray,
    prior_sample_fn: Callable[[PRNGKeyArray], Float[Array, ...]],
    log_target_fn: Callable[[Float[Array, ...]], Float[Array, ""]],
    n_particles: int = 1000,
    n_iterations: int = 10,
    initial_proposal_cov: Float[Array, "param_dim param_dim"] | None = None,
    adapt_proposals: bool = True,
) -> tuple[PMCState, SMCInfo]:
    """Run Population Monte Carlo.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    prior_sample_fn : Callable
        Function to sample from prior.
    log_target_fn : Callable
        Log-target density.
    n_particles : int
        Number of particles.
    n_iterations : int
        Number of PMC iterations.
    initial_proposal_cov : Array, optional
        Initial proposal covariance.
    adapt_proposals : bool
        Whether to adapt proposals.

    Returns
    -------
    final_state : PMCState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize from prior
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(prior_sample_fn)(init_keys)
    param_dim = particles.shape[1]

    # Initialize proposals centered at particles
    if initial_proposal_cov is None:
        # Use empirical covariance from prior samples
        mean = jnp.mean(particles, axis=0)
        diff = particles - mean
        initial_proposal_cov = jnp.cov(diff.T) + 0.01 * jnp.eye(param_dim)

    proposal_means = particles
    proposal_covs = jnp.tile(initial_proposal_cov, (n_particles, 1, 1))

    initial_state = PMCState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        proposal_means=proposal_means,
        proposal_covs=proposal_covs,
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        step_key = inputs
        new_state, info = pmc_step(step_key, state, log_target_fn, adapt_proposals)
        return (new_state, key), info

    step_keys = jax.random.split(key, n_iterations)

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        step_keys,
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=None,
    )

    return final_state, combined_info
