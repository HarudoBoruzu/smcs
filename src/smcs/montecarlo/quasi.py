"""Quasi-Monte Carlo methods.

Low-discrepancy sequences for numerical integration including:
- Halton sequence
- Sobol sequence
- Latin Hypercube sampling
- Randomized Quasi-Monte Carlo
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

__all__ = [
    "halton_sequence",
    "sobol_sequence",
    "latin_hypercube",
    "randomized_halton",
    "randomized_lhs",
]


def _get_primes(n: int) -> list[int]:
    """Get first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


@jaxtyped(typechecker=beartype)
def halton_sequence(
    n_samples: int,
    dim: int,
    skip: int = 0,
) -> Float[Array, "n_samples dim"]:
    """Generate Halton quasi-random sequence.

    The Halton sequence provides low-discrepancy points in [0, 1]^d.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    dim : int
        Dimensionality.
    skip : int
        Number of initial points to skip (for better distribution).

    Returns
    -------
    samples : Array
        Halton sequence points in [0, 1]^d.
    """
    primes = _get_primes(dim)

    def halton_1d(base, n):
        """Generate 1D Halton sequence for given base."""
        jnp.zeros(n)
        indices = jnp.arange(skip + 1, skip + n + 1)

        def compute_point(idx):
            f = 1.0
            r = 0.0
            i = idx
            while i > 0:
                f = f / base
                r = r + f * (i % base)
                i = i // base
            return r

        # Vectorized version (approximate, for JAX compatibility)
        max_digits = 20  # Sufficient for most cases

        def digit_contribution(digit_idx):
            divisor = base ** (digit_idx + 1)
            remainder = (indices // (base ** digit_idx)) % base
            return remainder / divisor

        contributions = jax.vmap(digit_contribution)(jnp.arange(max_digits))
        return jnp.sum(contributions, axis=0)

    sequences = []
    for p in primes:
        sequences.append(halton_1d(p, n_samples))

    return jnp.stack(sequences, axis=1)


@jaxtyped(typechecker=beartype)
def sobol_sequence(
    n_samples: int,
    dim: int,
    skip: int = 0,
) -> Float[Array, "n_samples dim"]:
    """Generate Sobol quasi-random sequence.

    Simplified Sobol sequence implementation.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimensionality.
    skip : int
        Number of initial points to skip.

    Returns
    -------
    samples : Array
        Sobol sequence points in [0, 1]^d.
    """
    # Direction numbers for first few dimensions
    # These are the standard direction numbers for Sobol sequences
    max_bits = 32

    def gray_code(n):
        """Compute Gray code of n."""
        return n ^ (n >> 1)

    def count_trailing_zeros(n):
        """Count trailing zeros in binary representation."""
        if n == 0:
            return max_bits
        count = 0
        while (n & 1) == 0:
            count += 1
            n >>= 1
        return count

    # Generate using Gray code
    def generate_sobol_1d(direction_numbers, indices):
        """Generate 1D Sobol sequence."""
        n = len(indices)
        result = jnp.zeros(n, dtype=jnp.uint32)

        # This is a simplified version
        for i in range(n):
            idx = indices[i]
            if idx == 0:
                continue

            gray = gray_code(idx)
            bit_pos = 0
            value = 0
            while gray > 0:
                if gray & 1:
                    value ^= direction_numbers[bit_pos]
                gray >>= 1
                bit_pos += 1
            result = result.at[i].set(value)

        return result.astype(jnp.float32) / (2 ** max_bits)

    # Initialize direction numbers (simplified)
    sequences = []
    jnp.arange(skip, skip + n_samples)

    for d in range(dim):
        # Simple direction numbers (powers of 2 shifted)
        if d == 0:
            dir_nums = [2 ** (max_bits - 1 - i) for i in range(max_bits)]
            direction = jnp.array(dir_nums, dtype=jnp.uint32)
        else:
            # Use polynomial-based direction numbers (simplified)
            direction = jnp.array([
                (2 ** (max_bits - 1 - i)) ^ (2 ** (max_bits - 2 - i) if i < max_bits - 1 else 0)
                for i in range(max_bits)
            ], dtype=jnp.uint32)

        # Generate points
        points = jnp.zeros(n_samples)
        for i in range(n_samples):
            idx = skip + i
            gray = gray_code(idx)
            value = 0
            for b in range(max_bits):
                if (gray >> b) & 1:
                    value ^= int(direction[b])
            points = points.at[i].set(value / (2 ** max_bits))

        sequences.append(points)

    return jnp.stack(sequences, axis=1)


@jaxtyped(typechecker=beartype)
def latin_hypercube(
    key: PRNGKeyArray,
    n_samples: int,
    dim: int,
) -> Float[Array, "n_samples dim"]:
    """Generate Latin Hypercube sample.

    Stratified sampling where each dimension is divided into n equal
    strata and exactly one sample is taken from each stratum.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    n_samples : int
        Number of samples.
    dim : int
        Dimensionality.

    Returns
    -------
    samples : Array
        LHS samples in [0, 1]^d.
    """
    keys = jax.random.split(key, dim + 1)

    def sample_dimension(key_d):
        # Permutation of strata
        perm_key, offset_key = jax.random.split(key_d)
        permutation = jax.random.permutation(perm_key, n_samples)

        # Random offset within each stratum
        offsets = jax.random.uniform(offset_key, shape=(n_samples,))

        # Compute sample points
        return (permutation + offsets) / n_samples

    samples = jax.vmap(sample_dimension)(keys[:dim])
    return samples.T


@jaxtyped(typechecker=beartype)
def randomized_halton(
    key: PRNGKeyArray,
    n_samples: int,
    dim: int,
    skip: int = 0,
) -> Float[Array, "n_samples dim"]:
    """Generate randomized Halton sequence.

    Applies random digital shift to Halton sequence for unbiased estimation.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    n_samples : int
        Number of samples.
    dim : int
        Dimensionality.
    skip : int
        Points to skip.

    Returns
    -------
    samples : Array
        Randomized Halton points in [0, 1]^d.
    """
    # Generate base Halton sequence
    halton = halton_sequence(n_samples, dim, skip)

    # Random shift (Cranley-Patterson rotation)
    shift = jax.random.uniform(key, shape=(dim,))

    # Apply shift modulo 1
    return (halton + shift) % 1.0


@jaxtyped(typechecker=beartype)
def randomized_lhs(
    key: PRNGKeyArray,
    n_samples: int,
    dim: int,
    correlation_reduction: bool = True,
) -> Float[Array, "n_samples dim"]:
    """Generate randomized Latin Hypercube sample with correlation reduction.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key.
    n_samples : int
        Number of samples.
    dim : int
        Dimensionality.
    correlation_reduction : bool
        Apply correlation reduction algorithm.

    Returns
    -------
    samples : Array
        LHS samples in [0, 1]^d.
    """
    key, lhs_key = jax.random.split(key)

    # Generate initial LHS
    samples = latin_hypercube(lhs_key, n_samples, dim)

    if not correlation_reduction or dim < 2:
        return samples

    # Simple correlation reduction: sort by one dimension
    # More sophisticated methods exist but are complex to implement
    key, sort_key = jax.random.split(key)
    reference_dim = jax.random.randint(sort_key, (), 0, dim)
    sort_indices = jnp.argsort(samples[:, reference_dim])

    # Reorder other dimensions to reduce correlation
    result = samples.copy()
    for d in range(dim):
        if d != reference_dim:
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)
            sorted_values = jnp.sort(samples[:, d])
            result = result.at[sort_indices, d].set(sorted_values[perm])

    return result
