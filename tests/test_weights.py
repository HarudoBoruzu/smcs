"""Tests for weight computation utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from smcs.core.weights import compute_ess, log_mean_exp, normalize_log_weights


class TestComputeESS:
    """Tests for ESS computation."""

    def test_uniform_weights_max_ess(self):
        """Uniform weights should give ESS = N."""
        n_particles = 100
        log_weights = jnp.zeros(n_particles)

        ess = compute_ess(log_weights)

        np.testing.assert_allclose(ess, n_particles, rtol=1e-5)

    def test_degenerate_weights_min_ess(self):
        """Degenerate weights (all mass on one) should give ESS ≈ 1."""
        n_particles = 100
        log_weights = jnp.full(n_particles, -jnp.inf)
        log_weights = log_weights.at[0].set(0.0)

        ess = compute_ess(log_weights)

        np.testing.assert_allclose(ess, 1.0, rtol=1e-5)

    def test_ess_range(self):
        """ESS should be in range (0, N]."""
        key = jax.random.PRNGKey(42)
        n_particles = 100

        for _ in range(10):
            key, subkey = jax.random.split(key)
            log_weights = jax.random.normal(subkey, shape=(n_particles,))

            ess = compute_ess(log_weights)

            assert 0 < ess <= n_particles


class TestNormalizeLogWeights:
    """Tests for log weight normalization."""

    def test_normalized_sum_to_one(self):
        """Normalized weights should sum to 1."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        normalized = normalize_log_weights(log_weights)
        weights = jnp.exp(normalized)

        np.testing.assert_allclose(jnp.sum(weights), 1.0, rtol=1e-5)

    def test_preserves_ratios(self):
        """Normalization should preserve weight ratios."""
        log_weights = jnp.array([0.0, 1.0, 2.0])

        normalized = normalize_log_weights(log_weights)

        # Ratios should be preserved
        original_ratio = jnp.exp(log_weights[1] - log_weights[0])
        normalized_ratio = jnp.exp(normalized[1] - normalized[0])

        np.testing.assert_allclose(original_ratio, normalized_ratio, rtol=1e-5)


class TestLogMeanExp:
    """Tests for log mean exp computation."""

    def test_equal_values(self):
        """log(mean(exp(x))) = x when all x are equal."""
        n = 100
        x = 2.0
        log_values = jnp.full(n, x)

        result = log_mean_exp(log_values)

        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_matches_direct_computation(self):
        """Should match direct computation for small values."""
        log_values = jnp.array([0.0, 1.0, 2.0])

        result = log_mean_exp(log_values)
        expected = jnp.log(jnp.mean(jnp.exp(log_values)))

        np.testing.assert_allclose(result, expected, rtol=1e-5)
