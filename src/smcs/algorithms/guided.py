"""Guided (Optimal) Particle Filter.

The Guided Particle Filter uses proposal distributions that incorporate
the current observation, leading to more efficient sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from smcs.core.particles import SMCInfo, SMCState
from smcs.core.resampling import resample
from smcs.core.weights import compute_ess, normalize_log_weights

if TYPE_CHECKING:
    from smcs.models.base import StateSpaceModel

__all__ = [
    "GuidedState",
    "guided_step",
    "run_guided_filter",
]


@chex.dataclass(frozen=True)
class GuidedState(SMCState):
    """State for Guided Particle Filter.

    Inherits all fields from SMCState.
    """

    pass


@jaxtyped(typechecker=beartype)
def guided_step(
    key: PRNGKeyArray,
    state: GuidedState,
    observation: Float[Array, " obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    proposal_fn: Callable,
    proposal_log_prob_fn: Callable,
    ess_threshold: float = 0.5,
) -> tuple[GuidedState, SMCInfo]:
    """Perform one step of the Guided Particle Filter.

    The guided filter uses a proposal q(x_t | x_{t-1}, y_t) that depends
    on the current observation, rather than the prior p(x_t | x_{t-1}).

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    state : GuidedState
        Current filter state.
    observation : Array
        Current observation.
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    proposal_fn : Callable
        Function (key, particles, observation, params, step) -> new_particles.
    proposal_log_prob_fn : Callable
        Function (new_particles, old_particles, observation, params, step) -> log_probs.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    new_state : GuidedState
        Updated state.
    info : SMCInfo
        Step information.
    """
    n_particles = state.particles.shape[0]
    key, resample_key, propose_key = jax.random.split(key, 3)

    # Compute ESS and resample if needed
    ess = compute_ess(state.log_weights)
    threshold = ess_threshold * n_particles
    do_resample = ess < threshold

    log_weights = jnp.where(
        do_resample,
        jnp.zeros(n_particles),
        state.log_weights,
    )
    ancestors = jax.lax.cond(
        do_resample,
        lambda: resample(resample_key, state.log_weights, method="systematic"),
        lambda: jnp.arange(n_particles),
    )
    particles = state.particles[ancestors]

    # Propose new particles using observation-dependent proposal
    new_particles = proposal_fn(
        propose_key, particles, observation, params, state.step
    )

    # Compute importance weights: w = p(y|x) * p(x|x') / q(x|x',y)
    def compute_weight(new_x, old_x):
        # Transition probability
        trans_dist = model.transition_distribution(params, old_x, state.step)
        log_trans = jnp.squeeze(trans_dist.log_prob(new_x))

        # Emission probability
        emit_dist = model.emission_distribution(params, new_x, state.step + 1)
        log_emit = jnp.squeeze(emit_dist.log_prob(observation))

        # Proposal probability
        log_proposal = proposal_log_prob_fn(
            new_x, old_x, observation, params, state.step
        )

        return log_emit + log_trans - log_proposal

    log_increments = jax.vmap(compute_weight)(new_particles, particles)
    new_log_weights = log_weights + log_increments

    # Update log-likelihood
    log_likelihood_increment = jax.scipy.special.logsumexp(
        new_log_weights
    ) - jax.scipy.special.logsumexp(log_weights)
    new_log_likelihood = state.log_likelihood + log_likelihood_increment

    new_state = GuidedState(
        particles=new_particles,
        log_weights=normalize_log_weights(new_log_weights),
        ancestors=ancestors,
        log_likelihood=new_log_likelihood,
        step=state.step + 1,
    )

    info = SMCInfo(
        ess=ess,
        resampled=do_resample,
        acceptance_rate=None,
    )

    return new_state, info


@jaxtyped(typechecker=beartype)
def run_guided_filter(
    key: PRNGKeyArray,
    observations: Float[Array, "n_steps obs_dim"],
    model: StateSpaceModel,
    params: chex.ArrayTree,
    proposal_fn: Callable,
    proposal_log_prob_fn: Callable,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
) -> tuple[GuidedState, SMCInfo]:
    """Run the Guided Particle Filter on a sequence of observations.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    observations : Array
        Observations of shape (n_steps, obs_dim).
    model : StateSpaceModel
        State-space model.
    params : ArrayTree
        Model parameters.
    proposal_fn : Callable
        Proposal sampling function.
    proposal_log_prob_fn : Callable
        Proposal log-probability function.
    n_particles : int
        Number of particles.
    ess_threshold : float
        ESS threshold for resampling.

    Returns
    -------
    final_state : GuidedState
        Final filter state.
    info : SMCInfo
        Combined information from all steps.
    """
    key, init_key = jax.random.split(key)

    # Initialize particles from prior
    init_dist = model.initial_distribution(params)
    init_keys = jax.random.split(init_key, n_particles)
    particles = jax.vmap(init_dist.sample)(init_keys)

    initial_state = GuidedState(
        particles=particles,
        log_weights=jnp.full(n_particles, -jnp.log(n_particles)),
        ancestors=jnp.arange(n_particles),
        log_likelihood=jnp.array(0.0),
        step=jnp.array(0, dtype=jnp.int32),
    )

    def scan_fn(carry, inputs):
        state, key = carry
        obs, step_key = inputs
        new_state, info = guided_step(
            step_key,
            state,
            obs,
            model,
            params,
            proposal_fn,
            proposal_log_prob_fn,
            ess_threshold,
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
