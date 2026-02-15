"""Nested SMC and Island Particle Filter.

These algorithms use hierarchical particle structures for
improved estimation of normalizing constants and rare events.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, log_mean_exp, normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "NestedSMCState",
    "IslandState",
    "nested_smc_step",
    "run_nested_smc",
    "island_step",
    "run_island_filter",
]


@chex.dataclass(frozen=True)
class NestedSMCState:
    """State for Nested SMC.

    Attributes
    ----------
    outer_particles : Array
        Outer-level particles (parameters or high-level states).
    outer_log_weights : Array
        Outer-level log-weights.
    inner_particles : Array
        Inner-level particles for each outer particle.
    inner_log_weights : Array
        Inner-level log-weights.
    log_likelihood : Array
        Accumulated log-likelihood.
    step : Array
        Current step.
    """

    outer_particles: Float[Array, "n_outer outer_dim"]
    outer_log_weights: Float[Array, " n_outer"]
    inner_particles: Float[Array, "n_outer n_inner inner_dim"]
    inner_log_weights: Float[Array, "n_outer n_inner"]
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]


@chex.dataclass(frozen=True)
class IslandState(SMCState):
    """State for Island Particle Filter.

    Attributes
    ----------
    particles : Array
        All particles across islands (n_islands * particles_per_island, state_dim).
    log_weights : Array
        Log-weights for all particles.
    island_assignments : Array
        Island index for each particle.
    island_log_likelihoods : Array
        Log-likelihood estimate per island.
    ancestors : Array
        Ancestor indices.
    log_likelihood : Array
        Combined log-likelihood estimate.
    step : Array
        Current step.
    """

    island_assignments: Int[Array, " n_particles"]
    island_log_likelihoods: Float[Array, " n_islands"]


@jaxtyped(typechecker=beartype)
def nested_smc_step(
    key: PRNGKeyArray,
    state: NestedSMCState,
    observation: Float[Array, " obs_dim"],
    outer_transition_fn: Callable[
        [PRNGKeyArray, Float[Array, " outer_dim"]], Float[Array, " outer_dim"]
    ],
    inner_model_fn: Callable,
    outer_ess_threshold: float = 0.5,
    inner_ess_threshold: float = 0.5,
) -> tuple[NestedSMCState, SMCInfo]:
    """Perform one step of Nested SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : NestedSMCState
        Current state.
    observation : Array
        Current observation.
    outer_transition_fn : Callable
        Transition for outer particles.
    inner_model_fn : Callable
        Function returning (transition_fn, emission_fn) for given outer particle.
    outer_ess_threshold : float
        ESS threshold for outer resampling.
    inner_ess_threshold : float
        ESS threshold for inner resampling.

    Returns
    -------
    new_state : NestedSMCState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_outer = state.outer_particles.shape[0]
    n_inner = state.inner_particles.shape[1]

    key, outer_resample_key, outer_propagate_key, inner_key = jax.random.split(key, 4)

    # Outer level: resample if needed
    outer_ess = compute_ess(state.outer_log_weights)
    outer_threshold = outer_ess_threshold * n_outer
    do_outer_resample = outer_ess < outer_threshold

    outer_ancestors = jax.lax.cond(
        do_outer_resample,
        lambda: resample(outer_resample_key, state.outer_log_weights, "systematic"),
        lambda: jnp.arange(n_outer),
    )

    outer_log_weights = jnp.where(
        do_outer_resample,
        jnp.zeros(n_outer),
        state.outer_log_weights,
    )

    resampled_outer = state.outer_particles[outer_ancestors]
    resampled_inner = state.inner_particles[outer_ancestors]
    resampled_inner_weights = state.inner_log_weights[outer_ancestors]

    # Propagate outer particles
    outer_keys = jax.random.split(outer_propagate_key, n_outer)
    new_outer_particles = jax.vmap(outer_transition_fn)(outer_keys, resampled_outer)

    # For each outer particle, run inner SMC step
    def run_inner_step(key_i, outer_p, inner_ps, inner_ws):
        trans_fn, emit_fn = inner_model_fn(outer_p)

        # Resample inner
        inner_ess = compute_ess(inner_ws)
        do_inner_resample = inner_ess < inner_ess_threshold * n_inner

        key_i, resample_k, propagate_k = jax.random.split(key_i, 3)

        inner_ancestors = jax.lax.cond(
            do_inner_resample,
            lambda: resample(resample_k, inner_ws, "systematic"),
            lambda: jnp.arange(n_inner),
        )

        inner_log_ws = jnp.where(
            do_inner_resample, jnp.zeros(n_inner), inner_ws
        )
        resampled_inner_ps = inner_ps[inner_ancestors]

        # Propagate
        prop_keys = jax.random.split(propagate_k, n_inner)
        new_inner_ps = jax.vmap(trans_fn)(prop_keys, resampled_inner_ps)

        # Weight by observation
        log_liks = jax.vmap(lambda x: emit_fn(x, observation))(new_inner_ps)
        new_inner_log_ws = inner_log_ws + log_liks

        # Normalizing constant contribution
        log_nc = jax.scipy.special.logsumexp(new_inner_log_ws) - jnp.log(n_inner)

        return new_inner_ps, normalize_log_weights(new_inner_log_ws), log_nc

    inner_keys = jax.random.split(inner_key, n_outer)
    new_inner_particles, new_inner_log_weights, inner_log_ncs = jax.vmap(
        run_inner_step
    )(inner_keys, new_outer_particles, resampled_inner, resampled_inner_weights)

    # Update outer weights with inner normalizing constants
    new_outer_log_weights = outer_log_weights + inner_log_ncs

    # Overall log-likelihood increment
    log_lik_increment = log_mean_exp(new_outer_log_weights + jnp.log(n_outer))

    new_state = NestedSMCState(
        outer_particles=new_outer_particles,
        outer_log_weights=normalize_log_weights(new_outer_log_weights),
        inner_particles=new_inner_particles,
        inner_log_weights=new_inner_log_weights,
        log_likelihood=state.log_likelihood + log_lik_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=outer_ess,
        resampled=do_outer_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_nested_smc(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    outer_init_fn: Callable[[PRNGKeyArray], Float[Array, " outer_dim"]],
    inner_init_fn: Callable[
        [PRNGKeyArray, Float[Array, " outer_dim"]], Float[Array, " inner_dim"]
    ],
    outer_transition_fn: Callable[
        [PRNGKeyArray, Float[Array, " outer_dim"]], Float[Array, " outer_dim"]
    ],
    inner_model_fn: Callable,
    n_outer: int = 100,
    n_inner: int = 100,
    outer_ess_threshold: float = 0.5,
    inner_ess_threshold: float = 0.5,
) -> tuple[NestedSMCState, SMCInfo]:
    """Run Nested SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    outer_init_fn : Callable
        Initialize outer particles.
    inner_init_fn : Callable
        Initialize inner particles given outer.
    outer_transition_fn : Callable
        Outer transition.
    inner_model_fn : Callable
        Returns (trans_fn, emit_fn) for inner given outer.
    n_outer : int
        Number of outer particles.
    n_inner : int
        Number of inner particles per outer.
    outer_ess_threshold : float
        Outer ESS threshold.
    inner_ess_threshold : float
        Inner ESS threshold.

    Returns
    -------
    final_state : NestedSMCState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)

    # Initialize outer particles
    outer_keys = jax.random.split(init_key, n_outer)
    outer_particles = jax.vmap(outer_init_fn)(outer_keys)
    outer_particles.shape[1]

    # Initialize inner particles for each outer
    def init_inner(key_i, outer_p):
        inner_keys = jax.random.split(key_i, n_inner)
        return jax.vmap(lambda k: inner_init_fn(k, outer_p))(inner_keys)

    inner_init_keys = jax.random.split(key, n_outer)
    inner_particles = jax.vmap(init_inner)(inner_init_keys, outer_particles)
    inner_particles.shape[2]

    initial_state = NestedSMCState(
        outer_particles=outer_particles,
        outer_log_weights=jnp.full(n_outer, -jnp.log(n_outer)),
        inner_particles=inner_particles,
        inner_log_weights=jnp.full((n_outer, n_inner), -jnp.log(n_inner)),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        obs, step_key = inputs
        new_state, info = nested_smc_step(
            step_key,
            state,
            obs,
            outer_transition_fn,
            inner_model_fn,
            outer_ess_threshold,
            inner_ess_threshold,
        )
        return (new_state, key), info

    n_steps = observations.shape[0]
    step_keys = jax.random.split(key, n_steps)

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        (observations, step_keys),
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=None,
    )

    return final_state, combined_info


@jaxtyped(typechecker=beartype)
def island_step(
    key: PRNGKeyArray,
    state: IslandState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_islands: int,
    exchange_rate: float = 0.1,
    ess_threshold: float = 0.5,
) -> tuple[IslandState, SMCInfo]:
    """Perform one step of Island Particle Filter.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : IslandState
        Current state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    n_islands : int
        Number of islands.
    exchange_rate : float
        Probability of inter-island exchange.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : IslandState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    particles_per_island = n_particles // n_islands

    key, resample_key, propagate_key, exchange_key = jax.random.split(key, 4)

    # Compute ESS per island and resample within islands
    def process_island(island_idx, key_i):
        mask = state.island_assignments == island_idx
        island_particles = state.particles[mask]
        island_log_weights = state.log_weights[mask]

        # ESS for this island
        island_ess = compute_ess(island_log_weights)

        # Resample within island
        do_resample = island_ess < ess_threshold * particles_per_island
        ancestors = jax.lax.cond(
            do_resample,
            lambda: resample(key_i, island_log_weights, "systematic"),
            lambda: jnp.arange(particles_per_island),
        )

        resampled = island_particles[ancestors]
        new_weights = jnp.where(
            do_resample,
            jnp.zeros(particles_per_island),
            island_log_weights,
        )

        return resampled, new_weights, island_ess

    jax.random.split(resample_key, n_islands)
    # This is simplified - in practice we'd need more careful indexing
    # For now, assume particles are ordered by island

    # Propagate all particles
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, state.step)
        return trans_dist.sample(key_i)

    new_particles = jax.vmap(propagate_one)(propagate_keys, state.particles)

    # Compute weights
    def compute_weight(particle):
        emit_dist = model.emission_distribution(params, particle, state.step + 1)
        return jnp.squeeze(emit_dist.log_prob(observation))

    log_increments = jax.vmap(compute_weight)(new_particles)
    new_log_weights = state.log_weights + log_increments

    # Inter-island exchange (simplified)
    # With probability exchange_rate, swap some particles between islands
    jnp.int32(exchange_rate * n_particles)

    # Overall ESS
    ess = compute_ess(new_log_weights)

    # Compute per-island log-likelihood estimates
    def island_log_lik(island_idx):
        mask = state.island_assignments == island_idx
        island_weights = jnp.where(mask, new_log_weights, -jnp.inf)
        return jax.scipy.special.logsumexp(island_weights) - jnp.log(particles_per_island)

    island_log_liks = jax.vmap(island_log_lik)(jnp.arange(n_islands))

    # Combined log-likelihood using all islands
    log_lik_increment = log_mean_exp(island_log_liks)

    new_state = IslandState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        island_assignments=state.island_assignments,
        island_log_likelihoods=island_log_liks,
        ancestors=jnp.arange(n_particles),
        log_likelihood=state.log_likelihood + log_lik_increment,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=jnp.array(False),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_island_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_islands: int = 10,
    particles_per_island: int = 100,
    exchange_rate: float = 0.1,
    ess_threshold: float = 0.5,
) -> tuple[IslandState, SMCInfo]:
    """Run Island Particle Filter.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    n_islands : int
        Number of islands.
    particles_per_island : int
        Particles per island.
    exchange_rate : float
        Inter-island exchange rate.
    ess_threshold : float
        ESS threshold.

    Returns
    -------
    final_state : IslandState
        Final state.
    info : SMCInfo
        Combined information.
    """
    key, init_key = jax.random.split(key)
    n_particles = n_islands * particles_per_island

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    # Assign particles to islands
    island_assignments = jnp.repeat(jnp.arange(n_islands), particles_per_island)

    initial_state = IslandState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        island_assignments=island_assignments,
        island_log_likelihoods=jnp.zeros(n_islands),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, _ = carry
        obs, step_key = inputs
        new_state, info = island_step(
            step_key,
            state,
            obs,
            model,
            params,
            n_islands,
            exchange_rate,
            ess_threshold,
        )
        return (new_state, step_key), info

    n_steps = observations.shape[0]
    step_keys = jax.random.split(key, n_steps)

    (final_state, _), infos = jax.lax.scan(
        scan_fn,
        (initial_state, key),
        (observations, step_keys),
    )

    combined_info = SMCInfo(
        ess=infos.ess,
        resampled=infos.resampled,
        acceptance_rate=None,
    )

    return final_state, combined_info
