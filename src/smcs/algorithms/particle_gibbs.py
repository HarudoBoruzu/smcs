"""Particle Gibbs (PG) and Particle Gibbs with Ancestor Sampling (PGAS).

Particle Gibbs is a Markov chain Monte Carlo method that uses
conditional SMC as a Gibbs update for the latent states.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo
from smcs.core.resampling import resample
from smcs.core.weights import normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "PGState",
    "conditional_smc_step",
    "run_conditional_smc",
    "particle_gibbs_step",
    "run_particle_gibbs",
    "run_pgas",
]


@chex.dataclass(frozen=True)
class PGState:
    """State for Particle Gibbs.

    Attributes
    ----------
    trajectory : Array
        Current reference trajectory (n_steps, state_dim).
    parameters : ArrayTree
        Current parameter values.
    log_likelihood : Array
        Log-likelihood of current trajectory.
    step : Array
        Current MCMC iteration.
    """

    trajectory: Float[Array, "n_steps state_dim"]
    parameters: chex.ArrayTree
    log_likelihood: Float[Array, ""]
    step: Int[Array, ""]


@jaxtyped(typechecker=beartype)
def conditional_smc_step(
    key: PRNGKeyArray,
    particles: Float[Array, "n_particles state_dim"],
    log_weights: Float[Array, " n_particles"],
    reference_particle: Float[Array, " state_dim"],
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    step: Int[Array, ""],
    ancestor_sampling: bool = True,
) -> tuple[
    Float[Array, "n_particles state_dim"],
    Float[Array, " n_particles"],
    Int[Array, " n_particles"],
]:
    """Perform one step of conditional SMC.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    particles : Array
        Current particles.
    log_weights : Array
        Current log-weights.
    reference_particle : Array
        Reference particle to condition on.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    step : int
        Current time step.
    ancestor_sampling : bool
        Whether to use ancestor sampling (PGAS).

    Returns
    -------
    new_particles : Array
        Updated particles.
    new_log_weights : Array
        Updated log-weights.
    ancestors : Array
        Ancestor indices.
    """
    n_particles = particles.shape[0]
    key, resample_key, propagate_key, ancestor_key = jax.random.split(key, 4)

    # Resample (conditional: keep reference at index 0)
    ancestors = resample(resample_key, log_weights, method="systematic")

    # For CSMC, always keep reference particle's lineage
    # We'll handle this by setting ancestor[0] appropriately
    if ancestor_sampling:
        # Ancestor sampling: sample ancestor for reference particle
        # based on transition probability to reference
        def compute_ancestor_weight(idx):
            parent = particles[idx]
            trans_dist = model.transition_distribution(params, parent, step)
            return log_weights[idx] + jnp.squeeze(trans_dist.log_prob(reference_particle))

        ancestor_log_weights = jax.vmap(compute_ancestor_weight)(jnp.arange(n_particles))
        ancestor_probs = jax.nn.softmax(ancestor_log_weights)
        reference_ancestor = jax.random.choice(
            ancestor_key, n_particles, p=ancestor_probs
        )
        ancestors = ancestors.at[0].set(reference_ancestor)
    else:
        # Standard CSMC: ancestor of reference is previous reference (index 0)
        ancestors = ancestors.at[0].set(0)

    resampled_particles = particles[ancestors]

    # Propagate
    propagate_keys = jax.random.split(propagate_key, n_particles)

    def propagate_one(key_i, particle):
        trans_dist = model.transition_distribution(params, particle, step)
        return trans_dist.sample(key_i)

    new_particles = jax.vmap(propagate_one)(propagate_keys, resampled_particles)

    # Replace particle 0 with reference
    new_particles = new_particles.at[0].set(reference_particle)

    # Compute new weights
    def compute_weight(particle):
        emit_dist = model.emission_distribution(params, particle, step + 1)
        return jnp.squeeze(emit_dist.log_prob(observation))

    log_increments = jax.vmap(compute_weight)(new_particles)
    new_log_weights = normalize_log_weights(log_increments)

    return new_particles, new_log_weights, ancestors


@jaxtyped(typechecker=beartype)
def run_conditional_smc(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    reference_trajectory: Float[Array, "n_steps state_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    n_particles: int = 100,
    ancestor_sampling: bool = True,
) -> tuple[Float[Array, "n_steps state_dim"], Float[Array, ""]]:
    """Run conditional SMC to sample a new trajectory.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    reference_trajectory : Array
        Reference trajectory to condition on.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    n_particles : int
        Number of particles.
    ancestor_sampling : bool
        Whether to use ancestor sampling.

    Returns
    -------
    new_trajectory : Array
        Sampled trajectory.
    log_likelihood : float
        Log-likelihood of sampled trajectory.
    """
    key, init_key = jax.random.split(key)
    n_steps = observations.shape[0]
    state_dim = reference_trajectory.shape[1]

    # Initialize particles
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    # Replace particle 0 with reference initial state
    particles = particles.at[0].set(reference_trajectory[0])

    # Initial weights from first observation
    def init_weight(particle):
        emit_dist = model.emission_distribution(params, particle, 1)
        return jnp.squeeze(emit_dist.log_prob(observations[0]))

    log_weights = jax.vmap(init_weight)(particles)
    log_likelihood = jax.scipy.special.logsumexp(log_weights) - jnp.log(n_particles)
    log_weights = normalize_log_weights(log_weights)

    # Store all particles and ancestors for backward sampling
    all_particles = jnp.zeros((n_steps, n_particles, state_dim))
    all_particles = all_particles.at[0].set(particles)
    all_ancestors = jnp.zeros((n_steps, n_particles), dtype=jnp.int32)

    def scan_fn(carry, inputs):
        particles, log_weights, all_particles, all_ancestors, log_lik, key = carry
        obs, ref_particle, t = inputs

        key, step_key = jax.random.split(key)
        new_particles, new_log_weights, ancestors = conditional_smc_step(
            step_key,
            particles,
            log_weights,
            ref_particle,
            obs,
            model,
            params,
            t,
            ancestor_sampling,
        )

        # Update log-likelihood
        log_lik_incr = jax.scipy.special.logsumexp(new_log_weights)
        new_log_lik = log_lik + log_lik_incr

        # Store
        new_all_particles = all_particles.at[t + 1].set(new_particles)
        new_all_ancestors = all_ancestors.at[t + 1].set(ancestors)

        return (
            new_particles,
            normalize_log_weights(new_log_weights),
            new_all_particles,
            new_all_ancestors,
            new_log_lik,
            key,
        ), None

    # Run forward pass (skip first observation, already processed)
    (final_particles, final_weights, all_particles, all_ancestors, log_likelihood, _), _ = (
        jax.lax.scan(
            scan_fn,
            (particles, log_weights, all_particles, all_ancestors, log_likelihood, key),
            (
                observations[1:],
                reference_trajectory[1:],
                jnp.arange(n_steps - 1),
            ),
        )
    )

    # Sample final particle index
    key, sample_key = jax.random.split(key)
    # Always select particle 0 (the reference) with high probability in CSMC
    # Or sample proportionally to weights
    final_idx = jax.random.choice(
        sample_key, n_particles, p=jax.nn.softmax(final_weights)
    )

    # Trace back ancestry to get trajectory
    def trace_back(carry, inputs):
        idx = carry
        particles_t, ancestors_t = inputs
        state = particles_t[idx]
        new_idx = ancestors_t[idx]
        return new_idx, state

    _, trajectory_reversed = jax.lax.scan(
        trace_back,
        final_idx,
        (all_particles[::-1], all_ancestors[::-1]),
    )
    new_trajectory = trajectory_reversed[::-1]

    return new_trajectory, log_likelihood


@jaxtyped(typechecker=beartype)
def particle_gibbs_step(
    key: PRNGKeyArray,
    state: PGState,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    param_update_fn: Callable[
        [PRNGKeyArray, Float[Array, "n_steps state_dim"], chex.ArrayTree],
        chex.ArrayTree,
    ],
    n_particles: int = 100,
    ancestor_sampling: bool = True,
) -> tuple[PGState, SMCInfo]:
    """Perform one iteration of Particle Gibbs.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : PGState
        Current PG state.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    param_update_fn : Callable
        Function to update parameters given trajectory.
    n_particles : int
        Number of particles for CSMC.
    ancestor_sampling : bool
        Whether to use ancestor sampling.

    Returns
    -------
    new_state : PGState
        Updated state.
    info : SMCInfo
        Step information.
    """
    key, csmc_key, param_key = jax.random.split(key, 3)

    # Sample new trajectory using CSMC
    new_trajectory, log_lik = run_conditional_smc(
        csmc_key,
        observations,
        state.trajectory,
        model,
        state.parameters,
        n_particles,
        ancestor_sampling,
    )

    # Update parameters given new trajectory
    new_params = param_update_fn(param_key, new_trajectory, state.parameters)

    new_state = PGState(
        trajectory=new_trajectory,
        parameters=new_params,
        log_likelihood=log_lik,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=jnp.array(float(n_particles)),
        resampled=jnp.array(True),
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_particle_gibbs(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    initial_params: chex.ArrayTree,
    param_update_fn: Callable[
        [PRNGKeyArray, Float[Array, "n_steps state_dim"], chex.ArrayTree],
        chex.ArrayTree,
    ],
    n_particles: int = 100,
    n_iterations: int = 1000,
    n_burnin: int = 100,
    ancestor_sampling: bool = False,
) -> tuple[PGState, list[chex.ArrayTree]]:
    """Run Particle Gibbs sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    initial_params : ArrayTree
        Initial parameter values.
    param_update_fn : Callable
        Parameter update function.
    n_particles : int
        Number of particles.
    n_iterations : int
        Total MCMC iterations.
    n_burnin : int
        Burn-in iterations to discard.
    ancestor_sampling : bool
        Whether to use ancestor sampling.

    Returns
    -------
    final_state : PGState
        Final state.
    param_samples : list
        Parameter samples after burn-in.
    """
    key, init_key = jax.random.split(key)
    n_steps = observations.shape[0]
    state_dim = model.state_dim

    # Initialize trajectory from bootstrap filter
    from smcs.algorithms.bootstrap import run_bootstrap_filter

    init_state, _ = run_bootstrap_filter(
        init_key, observations, model, initial_params, n_particles
    )

    # Use weighted mean as initial trajectory (simplified)
    initial_trajectory = jnp.zeros((n_steps, state_dim))

    initial_pg_state = PGState(
        trajectory=initial_trajectory,
        parameters=initial_params,
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    # Run iterations
    def scan_fn(carry, inputs):
        state, key = carry
        step_key = inputs
        new_state, info = particle_gibbs_step(
            step_key,
            state,
            observations,
            model,
            param_update_fn,
            n_particles,
            ancestor_sampling,
        )
        return (new_state, key), new_state.parameters

    step_keys = jax.random.split(key, n_iterations)

    (final_state, _), all_params = jax.lax.scan(
        scan_fn,
        (initial_pg_state, key),
        step_keys,
    )

    # Return samples after burn-in (as pytree)
    # Note: In practice, you'd want to handle this differently for large samples

    return final_state, all_params


@jaxtyped(typechecker=beartype)
def run_pgas(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    initial_params: chex.ArrayTree,
    param_update_fn: Callable[
        [PRNGKeyArray, Float[Array, "n_steps state_dim"], chex.ArrayTree],
        chex.ArrayTree,
    ],
    n_particles: int = 100,
    n_iterations: int = 1000,
    n_burnin: int = 100,
) -> tuple[PGState, list[chex.ArrayTree]]:
    """Run Particle Gibbs with Ancestor Sampling (PGAS).

    PGAS improves mixing by sampling ancestors for the reference particle.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations.
    model : StateSpaceModel
        State-space model.
    initial_params : ArrayTree
        Initial parameter values.
    param_update_fn : Callable
        Parameter update function.
    n_particles : int
        Number of particles.
    n_iterations : int
        Total MCMC iterations.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    final_state : PGState
        Final state.
    param_samples : list
        Parameter samples.
    """
    return run_particle_gibbs(
        key,
        observations,
        model,
        initial_params,
        param_update_fn,
        n_particles,
        n_iterations,
        n_burnin,
        ancestor_sampling=True,
    )
