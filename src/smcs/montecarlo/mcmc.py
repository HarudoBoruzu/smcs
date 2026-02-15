"""Markov Chain Monte Carlo (MCMC) methods.

Classical MCMC algorithms including:
- Slice sampling
- Metropolis-Hastings
- Random walk Metropolis
- Gibbs sampling
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS)
"""

from __future__ import annotations

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

__all__ = [
    "MCMCState",
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
]


@chex.dataclass(frozen=True)
class MCMCState:
    """State for MCMC samplers.

    Attributes
    ----------
    position : Array
        Current position in parameter space.
    log_prob : Array
        Log probability at current position.
    step : Array
        Current iteration.
    accepted : Array
        Number of accepted proposals.
    """

    position: Float[Array, " dim"]
    log_prob: Float[Array, ""]
    step: jax.Array
    accepted: jax.Array


# =============================================================================
# Slice Sampling
# =============================================================================


@jaxtyped(typechecker=beartype)
def slice_sample(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    width: float = 1.0,
    max_steps: int = 100,
) -> Float[Array, " dim"]:
    """Perform one iteration of slice sampling.

    Slice sampling automatically adapts to the local scale of the
    distribution without tuning.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    log_prob_fn : Callable
        Log probability function.
    width : float
        Initial bracket width.
    max_steps : int
        Maximum stepping-out iterations.

    Returns
    -------
    new_position : Array
        New position after slice sampling.
    """
    dim = position.shape[0]
    log_y = log_prob_fn(position) + jnp.log(jax.random.uniform(key))

    def sample_one_dim(carry, dim_idx):
        pos, key = carry
        key, step_key, shrink_key = jax.random.split(key, 3)

        # Current value in this dimension
        x0 = pos[dim_idx]

        # Stepping out to find bracket
        u = jax.random.uniform(step_key)
        L = x0 - width * u
        R = L + width

        # Step out left
        def step_out_left(state):
            left, k, _ = state
            k, uk = jax.random.split(k)
            new_pos = pos.at[dim_idx].set(left)
            return left - width, k, log_prob_fn(new_pos) > log_y

        L, _, _ = jax.lax.while_loop(
            lambda s: s[2],
            step_out_left,
            (L, step_key, True)
        )

        # Step out right
        def step_out_right(state):
            r, k, _ = state
            k, uk = jax.random.split(k)
            new_pos = pos.at[dim_idx].set(r)
            return r + width, k, log_prob_fn(new_pos) > log_y

        R, _, _ = jax.lax.while_loop(
            lambda s: s[2],
            step_out_right,
            (R, step_key, True)
        )

        # Shrinking to find new point
        def shrink_step(state):
            left, right, k, x, done = state
            k, uk = jax.random.split(k)
            x_new = left + jax.random.uniform(uk) * (right - left)
            new_pos = pos.at[dim_idx].set(x_new)
            accept = log_prob_fn(new_pos) > log_y

            new_left = jnp.where(x_new < x0, x_new, left)
            new_right = jnp.where(x_new >= x0, x_new, right)

            return new_left, new_right, k, jnp.where(accept, x_new, x), accept

        _, _, _, x_final, _ = jax.lax.while_loop(
            lambda s: ~s[4],
            shrink_step,
            (L, R, shrink_key, x0, False)
        )

        new_pos = pos.at[dim_idx].set(x_final)
        return (new_pos, key), None

    (new_position, _), _ = jax.lax.scan(
        sample_one_dim, (position, key), jnp.arange(dim)
    )

    return new_position


@jaxtyped(typechecker=beartype)
def run_slice_sampler(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    n_samples: int = 1000,
    n_burnin: int = 100,
    width: float = 1.0,
) -> Float[Array, "n_samples dim"]:
    """Run slice sampler to generate samples.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    log_prob_fn : Callable
        Log probability function.
    n_samples : int
        Number of samples to generate.
    n_burnin : int
        Number of burn-in iterations.
    width : float
        Initial bracket width.

    Returns
    -------
    samples : Array
        MCMC samples of shape (n_samples, dim).
    """
    total_iterations = n_samples + n_burnin

    def scan_fn(position, key_i):
        new_position = slice_sample(key_i, position, log_prob_fn, width)
        return new_position, new_position

    keys = jax.random.split(key, total_iterations)
    _, all_samples = jax.lax.scan(scan_fn, initial_position, keys)

    return all_samples[n_burnin:]


# =============================================================================
# Metropolis-Hastings
# =============================================================================


@jaxtyped(typechecker=beartype)
def metropolis_hastings(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    proposal_fn: Callable[[PRNGKeyArray, Float[Array, " dim"]], Float[Array, " dim"]],
    proposal_log_prob_fn: Callable[
        [Float[Array, " dim"], Float[Array, " dim"]], Float[Array, ""]
    ] | None = None,
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """Perform one Metropolis-Hastings step.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    log_prob_fn : Callable
        Log probability of target distribution.
    proposal_fn : Callable
        Proposal distribution sampler (key, current) -> proposed.
    proposal_log_prob_fn : Callable, optional
        Log probability of proposal (new, old) -> log_prob.
        If None, assumes symmetric proposal.

    Returns
    -------
    new_position : Array
        New position (may be same as old if rejected).
    accepted : float
        1.0 if accepted, 0.0 if rejected.
    """
    key, proposal_key, accept_key = jax.random.split(key, 3)

    # Generate proposal
    proposed = proposal_fn(proposal_key, position)

    # Compute log acceptance ratio
    log_prob_current = log_prob_fn(position)
    log_prob_proposed = log_prob_fn(proposed)
    log_alpha = log_prob_proposed - log_prob_current

    # Add proposal ratio if asymmetric
    if proposal_log_prob_fn is not None:
        log_alpha += (
            proposal_log_prob_fn(position, proposed)
            - proposal_log_prob_fn(proposed, position)
        )

    # Accept/reject
    log_u = jnp.log(jax.random.uniform(accept_key))
    accept = log_u < log_alpha

    new_position = jnp.where(accept, proposed, position)
    return new_position, accept.astype(jnp.float32)


@jaxtyped(typechecker=beartype)
def run_metropolis_hastings(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    proposal_fn: Callable[[PRNGKeyArray, Float[Array, " dim"]], Float[Array, " dim"]],
    proposal_log_prob_fn: Callable | None = None,
    n_samples: int = 1000,
    n_burnin: int = 100,
) -> tuple[Float[Array, "n_samples dim"], Float[Array, ""]]:
    """Run Metropolis-Hastings sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    log_prob_fn : Callable
        Log probability function.
    proposal_fn : Callable
        Proposal distribution sampler.
    proposal_log_prob_fn : Callable, optional
        Proposal log probability.
    n_samples : int
        Number of samples.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    samples : Array
        MCMC samples.
    acceptance_rate : float
        Overall acceptance rate.
    """
    total_iterations = n_samples + n_burnin

    def scan_fn(carry, key_i):
        position, total_accepted = carry
        new_position, accepted = metropolis_hastings(
            key_i, position, log_prob_fn, proposal_fn, proposal_log_prob_fn
        )
        return (new_position, total_accepted + accepted), new_position

    keys = jax.random.split(key, total_iterations)
    (_, total_accepted), all_samples = jax.lax.scan(
        scan_fn, (initial_position, jnp.array(0.0)), keys
    )

    acceptance_rate = total_accepted / total_iterations
    return all_samples[n_burnin:], acceptance_rate


@jaxtyped(typechecker=beartype)
def random_walk_metropolis(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float | Float[Array, " dim"] = 1.0,
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """Random walk Metropolis-Hastings step.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    log_prob_fn : Callable
        Log probability function.
    step_size : float or Array
        Step size (scalar or per-dimension).

    Returns
    -------
    new_position : Array
        New position.
    accepted : float
        Whether proposal was accepted.
    """
    def proposal_fn(key_i, pos):
        noise = jax.random.normal(key_i, shape=pos.shape)
        return pos + step_size * noise

    return metropolis_hastings(key, position, log_prob_fn, proposal_fn)


@jaxtyped(typechecker=beartype)
def run_random_walk_metropolis(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float | Float[Array, " dim"] = 1.0,
    n_samples: int = 1000,
    n_burnin: int = 100,
) -> tuple[Float[Array, "n_samples dim"], Float[Array, ""]]:
    """Run random walk Metropolis sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    log_prob_fn : Callable
        Log probability function.
    step_size : float or Array
        Step size.
    n_samples : int
        Number of samples.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    samples : Array
        MCMC samples.
    acceptance_rate : float
        Acceptance rate.
    """
    def proposal_fn(key_i, pos):
        noise = jax.random.normal(key_i, shape=pos.shape)
        return pos + step_size * noise

    return run_metropolis_hastings(
        key, initial_position, log_prob_fn, proposal_fn, None, n_samples, n_burnin
    )


# =============================================================================
# Gibbs Sampling
# =============================================================================


@jaxtyped(typechecker=beartype)
def gibbs_sample(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    conditional_samplers: list[Callable],
) -> Float[Array, " dim"]:
    """Perform one Gibbs sampling iteration.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    conditional_samplers : list[Callable]
        List of functions (key, position) -> value that sample
        each coordinate from its full conditional.

    Returns
    -------
    new_position : Array
        New position after updating all coordinates.
    """
    n_dims = len(conditional_samplers)
    keys = jax.random.split(key, n_dims)

    for i, (sampler, key_i) in enumerate(zip(conditional_samplers, keys, strict=False)):
        new_value = sampler(key_i, position)
        position = position.at[i].set(new_value)

    return position


@jaxtyped(typechecker=beartype)
def run_gibbs_sampler(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    conditional_samplers: list[Callable],
    n_samples: int = 1000,
    n_burnin: int = 100,
) -> Float[Array, "n_samples dim"]:
    """Run Gibbs sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    conditional_samplers : list[Callable]
        Conditional sampling functions for each dimension.
    n_samples : int
        Number of samples.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    samples : Array
        MCMC samples.
    """
    total_iterations = n_samples + n_burnin

    def scan_fn(position, key_i):
        new_position = gibbs_sample(key_i, position, conditional_samplers)
        return new_position, new_position

    keys = jax.random.split(key, total_iterations)
    _, all_samples = jax.lax.scan(scan_fn, initial_position, keys)

    return all_samples[n_burnin:]


# =============================================================================
# Hamiltonian Monte Carlo
# =============================================================================


@jaxtyped(typechecker=beartype)
def hmc_step(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float = 0.1,
    n_leapfrog: int = 10,
    mass_matrix: Float[Array, "dim dim"] | None = None,
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """Perform one Hamiltonian Monte Carlo step.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    log_prob_fn : Callable
        Log probability function (potential energy = -log_prob).
    step_size : float
        Leapfrog step size.
    n_leapfrog : int
        Number of leapfrog steps.
    mass_matrix : Array, optional
        Mass matrix (default: identity).

    Returns
    -------
    new_position : Array
        New position.
    accepted : float
        Whether proposal was accepted.
    """
    dim = position.shape[0]
    key, momentum_key, accept_key = jax.random.split(key, 3)

    # Mass matrix
    if mass_matrix is None:
        mass_matrix = jnp.eye(dim)
    mass_inv = jnp.linalg.inv(mass_matrix)

    # Gradient of log probability
    grad_log_prob = jax.grad(log_prob_fn)

    # Sample momentum
    momentum = jax.random.multivariate_normal(
        momentum_key, jnp.zeros(dim), mass_matrix
    )

    # Initial Hamiltonian
    def kinetic_energy(p):
        return 0.5 * jnp.dot(p, mass_inv @ p)

    initial_H = -log_prob_fn(position) + kinetic_energy(momentum)

    # Leapfrog integration
    def leapfrog_step(carry, _):
        q, p = carry
        # Half step momentum
        p = p + 0.5 * step_size * grad_log_prob(q)
        # Full step position
        q = q + step_size * (mass_inv @ p)
        # Half step momentum
        p = p + 0.5 * step_size * grad_log_prob(q)
        return (q, p), None

    (proposed_q, proposed_p), _ = jax.lax.scan(
        leapfrog_step, (position, momentum), jnp.arange(n_leapfrog)
    )

    # Final Hamiltonian
    final_H = -log_prob_fn(proposed_q) + kinetic_energy(proposed_p)

    # Accept/reject
    log_accept_prob = initial_H - final_H
    log_u = jnp.log(jax.random.uniform(accept_key))
    accept = log_u < log_accept_prob

    new_position = jnp.where(accept, proposed_q, position)
    return new_position, accept.astype(jnp.float32)


@jaxtyped(typechecker=beartype)
def run_hmc(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float = 0.1,
    n_leapfrog: int = 10,
    mass_matrix: Float[Array, "dim dim"] | None = None,
    n_samples: int = 1000,
    n_burnin: int = 100,
) -> tuple[Float[Array, "n_samples dim"], Float[Array, ""]]:
    """Run Hamiltonian Monte Carlo sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    log_prob_fn : Callable
        Log probability function.
    step_size : float
        Leapfrog step size.
    n_leapfrog : int
        Number of leapfrog steps.
    mass_matrix : Array, optional
        Mass matrix.
    n_samples : int
        Number of samples.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    samples : Array
        HMC samples.
    acceptance_rate : float
        Acceptance rate.
    """
    total_iterations = n_samples + n_burnin

    def scan_fn(carry, key_i):
        position, total_accepted = carry
        new_position, accepted = hmc_step(
            key_i, position, log_prob_fn, step_size, n_leapfrog, mass_matrix
        )
        return (new_position, total_accepted + accepted), new_position

    keys = jax.random.split(key, total_iterations)
    (_, total_accepted), all_samples = jax.lax.scan(
        scan_fn, (initial_position, jnp.array(0.0)), keys
    )

    acceptance_rate = total_accepted / total_iterations
    return all_samples[n_burnin:], acceptance_rate


@jaxtyped(typechecker=beartype)
def nuts_step(
    key: PRNGKeyArray,
    position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float = 0.1,
    max_depth: int = 10,
) -> tuple[Float[Array, " dim"], Float[Array, ""]]:
    """Perform one No-U-Turn Sampler (NUTS) step.

    Simplified NUTS without full tree doubling.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    position : Array
        Current position.
    log_prob_fn : Callable
        Log probability function.
    step_size : float
        Leapfrog step size.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    new_position : Array
        New position.
    accepted : float
        Acceptance indicator.
    """
    dim = position.shape[0]
    key, momentum_key = jax.random.split(key)

    grad_log_prob = jax.grad(log_prob_fn)

    # Sample momentum
    momentum = jax.random.normal(momentum_key, shape=(dim,))

    # Initial energy
    initial_H = -log_prob_fn(position) + 0.5 * jnp.dot(momentum, momentum)

    # Simple NUTS: do leapfrog until U-turn detected
    def no_uturn(q_minus, q_plus, p_minus, p_plus):
        dq = q_plus - q_minus
        return (jnp.dot(dq, p_minus) >= 0) & (jnp.dot(dq, p_plus) >= 0)

    def leapfrog_step(q, p):
        p = p + 0.5 * step_size * grad_log_prob(q)
        q = q + step_size * p
        p = p + 0.5 * step_size * grad_log_prob(q)
        return q, p

    # Build trajectory
    def build_trajectory(carry, _):
        q_minus, q_plus, p_minus, p_plus, q_sample, n_valid, key, continue_ = carry

        key, direction_key, sample_key = jax.random.split(key, 3)
        direction = 2 * jax.random.bernoulli(direction_key) - 1

        # Extend in direction
        q_new, p_new = jax.lax.cond(
            direction > 0,
            lambda: leapfrog_step(q_plus, p_plus),
            lambda: leapfrog_step(q_minus, -p_minus),
        )
        p_new = direction * p_new

        # Update bounds
        q_minus_new = jnp.where(direction < 0, q_new, q_minus)
        q_plus_new = jnp.where(direction > 0, q_new, q_plus)
        p_minus_new = jnp.where(direction < 0, p_new, p_minus)
        p_plus_new = jnp.where(direction > 0, p_new, p_plus)

        # Check U-turn
        uturn = no_uturn(q_minus_new, q_plus_new, p_minus_new, p_plus_new)

        # Sample from trajectory
        H_new = -log_prob_fn(q_new) + 0.5 * jnp.dot(p_new, p_new)
        accept_prob = jnp.exp(jnp.minimum(0.0, initial_H - H_new))
        n_valid_new = n_valid + 1
        update_sample = jax.random.uniform(sample_key) < accept_prob / n_valid_new
        q_sample_new = jnp.where(update_sample, q_new, q_sample)

        continue_new = continue_ & uturn

        return (
            q_minus_new, q_plus_new, p_minus_new, p_plus_new,
            q_sample_new, n_valid_new, key, continue_new
        ), None

    initial_carry = (
        position, position, momentum, momentum,
        position, jnp.array(1), key, jnp.array(True)
    )

    (_, _, _, _, final_sample, _, _, _), _ = jax.lax.scan(
        build_trajectory, initial_carry, jnp.arange(max_depth)
    )

    return final_sample, jnp.array(1.0)


@jaxtyped(typechecker=beartype)
def run_nuts(
    key: PRNGKeyArray,
    initial_position: Float[Array, " dim"],
    log_prob_fn: Callable[[Float[Array, " dim"]], Float[Array, ""]],
    step_size: float = 0.1,
    max_depth: int = 10,
    n_samples: int = 1000,
    n_burnin: int = 100,
) -> Float[Array, "n_samples dim"]:
    """Run No-U-Turn Sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    initial_position : Array
        Starting position.
    log_prob_fn : Callable
        Log probability function.
    step_size : float
        Leapfrog step size.
    max_depth : int
        Maximum tree depth.
    n_samples : int
        Number of samples.
    n_burnin : int
        Burn-in iterations.

    Returns
    -------
    samples : Array
        NUTS samples.
    """
    total_iterations = n_samples + n_burnin

    def scan_fn(position, key_i):
        new_position, _ = nuts_step(key_i, position, log_prob_fn, step_size, max_depth)
        return new_position, new_position

    keys = jax.random.split(key, total_iterations)
    _, all_samples = jax.lax.scan(scan_fn, initial_position, keys)

    return all_samples[n_burnin:]
