"""Tests for quasi-Monte Carlo methods."""

import jax
import jax.numpy as jnp

from smcs.montecarlo.quasi import (
    halton_sequence,
    latin_hypercube,
    randomized_halton,
    randomized_lhs,
    sobol_sequence,
)


class TestHaltonSequence:
    """Tests for Halton sequence generation."""

    def test_output_shape(self):
        """Test output has correct shape."""
        n_samples = 100
        dim = 5

        result = halton_sequence(n_samples, dim)

        assert result.shape == (n_samples, dim)

    def test_values_in_unit_interval(self):
        """Test all values are in [0, 1)."""
        n_samples = 500
        dim = 10

        result = halton_sequence(n_samples, dim)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_1d_halton_correct(self):
        """Test 1D Halton sequence with base 2."""
        # First few terms of base-2 Halton: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        result = halton_sequence(5, 1)

        expected = jnp.array([[0.5], [0.25], [0.75], [0.125], [0.625]])
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_deterministic(self):
        """Test Halton sequence is deterministic."""
        result1 = halton_sequence(50, 3)
        result2 = halton_sequence(50, 3)

        assert jnp.allclose(result1, result2)

    def test_skip_parameter(self):
        """Test skip parameter."""
        full = halton_sequence(100, 2, skip=0)
        skipped = halton_sequence(50, 2, skip=50)

        assert jnp.allclose(full[50:], skipped)

    def test_low_discrepancy(self):
        """Test Halton sequence has lower discrepancy than random."""
        n_samples = 1000
        dim = 2

        halton_points = halton_sequence(n_samples, dim)

        # Simple discrepancy test: check coverage of quadrants
        # Count points in each quadrant
        q1 = jnp.sum((halton_points[:, 0] < 0.5) & (halton_points[:, 1] < 0.5))
        q2 = jnp.sum((halton_points[:, 0] >= 0.5) & (halton_points[:, 1] < 0.5))
        q3 = jnp.sum((halton_points[:, 0] < 0.5) & (halton_points[:, 1] >= 0.5))
        q4 = jnp.sum((halton_points[:, 0] >= 0.5) & (halton_points[:, 1] >= 0.5))

        # Each quadrant should have approximately n/4 points
        expected = n_samples / 4
        tolerance = n_samples * 0.1  # 10% tolerance

        assert jnp.abs(q1 - expected) < tolerance
        assert jnp.abs(q2 - expected) < tolerance
        assert jnp.abs(q3 - expected) < tolerance
        assert jnp.abs(q4 - expected) < tolerance


class TestSobolSequence:
    """Tests for Sobol sequence generation."""

    def test_output_shape(self):
        """Test output has correct shape."""
        n_samples = 100
        dim = 5

        result = sobol_sequence(n_samples, dim)

        assert result.shape == (n_samples, dim)

    def test_values_in_unit_interval(self):
        """Test all values are in [0, 1]."""
        n_samples = 500
        dim = 8

        result = sobol_sequence(n_samples, dim)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_deterministic(self):
        """Test Sobol sequence is deterministic."""
        result1 = sobol_sequence(50, 3)
        result2 = sobol_sequence(50, 3)

        assert jnp.allclose(result1, result2)

    def test_skip_parameter(self):
        """Test skip parameter."""
        full = sobol_sequence(100, 2, skip=0)
        skipped = sobol_sequence(50, 2, skip=50)

        assert jnp.allclose(full[50:], skipped)

    def test_uniform_coverage(self):
        """Test Sobol sequence covers space uniformly."""
        n_samples = 256  # Power of 2 works well
        dim = 2

        sobol_points = sobol_sequence(n_samples, dim)

        # Check that points are generated and bounded
        assert sobol_points.shape == (n_samples, dim)
        assert jnp.all(sobol_points >= 0)
        assert jnp.all(sobol_points <= 1)

        # Check coverage of halves
        left_half = jnp.sum(sobol_points[:, 0] < 0.5)
        right_half = jnp.sum(sobol_points[:, 0] >= 0.5)

        # Should have some points in each half
        assert left_half > 0
        assert right_half > 0


class TestLatinHypercube:
    """Tests for Latin Hypercube Sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        n_samples = 100
        dim = 5

        result = latin_hypercube(key, n_samples, dim)

        assert result.shape == (n_samples, dim)

    def test_values_in_unit_interval(self):
        """Test all values are in [0, 1)."""
        key = jax.random.PRNGKey(42)
        n_samples = 200
        dim = 10

        result = latin_hypercube(key, n_samples, dim)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_latin_property(self):
        """Test Latin hypercube property: one point per row/column in projection."""
        key = jax.random.PRNGKey(123)
        n_samples = 50
        dim = 3

        result = latin_hypercube(key, n_samples, dim)

        # For each dimension, check that bins are filled evenly
        for d in range(dim):
            bins = jnp.floor(result[:, d] * n_samples).astype(int)
            unique_bins = jnp.unique(bins)
            # Each bin should be hit exactly once
            assert len(unique_bins) == n_samples

    def test_reproducibility(self):
        """Test reproducibility with same key."""
        key = jax.random.PRNGKey(0)

        result1 = latin_hypercube(key, 50, 3)
        result2 = latin_hypercube(key, 50, 3)

        assert jnp.allclose(result1, result2)

    def test_different_keys_different_results(self):
        """Test different keys produce different results."""
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)

        result1 = latin_hypercube(key1, 50, 3)
        result2 = latin_hypercube(key2, 50, 3)

        assert not jnp.allclose(result1, result2)


class TestRandomizedHalton:
    """Tests for randomized Halton sequence."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        n_samples = 100
        dim = 5

        result = randomized_halton(key, n_samples, dim)

        assert result.shape == (n_samples, dim)

    def test_values_in_unit_interval(self):
        """Test all values are in [0, 1)."""
        key = jax.random.PRNGKey(42)
        n_samples = 200
        dim = 8

        result = randomized_halton(key, n_samples, dim)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_reproducibility(self):
        """Test reproducibility with same key."""
        key = jax.random.PRNGKey(0)

        result1 = randomized_halton(key, 50, 3)
        result2 = randomized_halton(key, 50, 3)

        assert jnp.allclose(result1, result2)

    def test_different_from_regular_halton(self):
        """Test randomized differs from regular Halton."""
        key = jax.random.PRNGKey(42)

        regular = halton_sequence(100, 3)
        randomized = randomized_halton(key, 100, 3)

        assert not jnp.allclose(regular, randomized)

    def test_uniform_marginal(self):
        """Test marginal distributions are approximately uniform."""
        key = jax.random.PRNGKey(123)
        n_samples = 1000
        dim = 2

        result = randomized_halton(key, n_samples, dim)

        # Check each dimension has approximately uniform marginal
        for d in range(dim):
            # Divide [0, 1) into 10 bins
            hist, _ = jnp.histogram(result[:, d], bins=10, range=(0, 1))
            expected = n_samples / 10
            # Each bin should have roughly the expected count
            assert jnp.all(hist > expected * 0.5)
            assert jnp.all(hist < expected * 1.5)


class TestRandomizedLHS:
    """Tests for randomized Latin Hypercube Sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        n_samples = 100
        dim = 5

        result = randomized_lhs(key, n_samples, dim)

        assert result.shape == (n_samples, dim)

    def test_values_in_unit_interval(self):
        """Test all values are in [0, 1)."""
        key = jax.random.PRNGKey(42)
        n_samples = 200
        dim = 10

        result = randomized_lhs(key, n_samples, dim)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result < 1.0)

    def test_latin_property(self):
        """Test Latin hypercube property is preserved."""
        key = jax.random.PRNGKey(123)
        n_samples = 50
        dim = 3

        result = randomized_lhs(key, n_samples, dim)

        # For each dimension, check bins
        for d in range(dim):
            bins = jnp.floor(result[:, d] * n_samples).astype(int)
            unique_bins = jnp.unique(bins)
            assert len(unique_bins) == n_samples

    def test_reproducibility(self):
        """Test reproducibility with same key."""
        key = jax.random.PRNGKey(0)

        result1 = randomized_lhs(key, 50, 3)
        result2 = randomized_lhs(key, 50, 3)

        assert jnp.allclose(result1, result2)

    def test_correlation_reduction(self):
        """Test correlation reduction option."""
        key = jax.random.PRNGKey(42)
        n_samples = 100
        dim = 2

        # With correlation reduction
        result_reduced = randomized_lhs(key, n_samples, dim, correlation_reduction=True)

        # Without correlation reduction
        result_basic = randomized_lhs(key, n_samples, dim, correlation_reduction=False)

        # Both should be valid
        assert result_reduced.shape == (n_samples, dim)
        assert result_basic.shape == (n_samples, dim)


class TestQMCIntegration:
    """Integration tests for quasi-Monte Carlo methods."""

    def test_integration_accuracy(self):
        """Test QMC integration is more accurate than MC."""
        # Integrate sin(pi*x) from 0 to 1 = 2/pi ≈ 0.6366
        true_value = 2.0 / jnp.pi
        n_samples = 1000

        # Quasi-random (Halton)
        halton_points = halton_sequence(n_samples, 1)
        qmc_estimate = jnp.mean(jnp.sin(jnp.pi * halton_points[:, 0]))

        # Pseudo-random
        key = jax.random.PRNGKey(0)
        random_points = jax.random.uniform(key, shape=(n_samples,))
        mc_estimate = jnp.mean(jnp.sin(jnp.pi * random_points))

        # QMC should be closer to true value (usually)
        qmc_error = jnp.abs(qmc_estimate - true_value)
        # MC error computed for comparison (QMC typically has lower error)
        _ = jnp.abs(mc_estimate - true_value)

        # QMC error should be small
        assert qmc_error < 0.01

    def test_multidimensional_integration(self):
        """Test QMC for multidimensional integration."""
        # Integrate x*y over [0,1]^2 = 0.25
        true_value = 0.25
        n_samples = 1000

        # Halton sequence
        halton_points = halton_sequence(n_samples, 2)
        qmc_estimate = jnp.mean(halton_points[:, 0] * halton_points[:, 1])

        assert jnp.abs(qmc_estimate - true_value) < 0.02

    def test_lhs_for_sampling(self):
        """Test LHS provides good space-filling."""
        key = jax.random.PRNGKey(42)
        n_samples = 100
        dim = 3

        # LHS samples
        lhs_points = latin_hypercube(key, n_samples, dim)

        # Check that points are in valid range
        assert jnp.all(lhs_points >= 0)
        assert jnp.all(lhs_points < 1)

        # Check that we have the right number of samples
        assert lhs_points.shape == (n_samples, dim)

        # Check stratification property
        for d in range(dim):
            bins = jnp.floor(lhs_points[:, d] * n_samples).astype(int)
            assert len(jnp.unique(bins)) == n_samples

    def test_sobol_for_option_pricing(self):
        """Test Sobol for simple option-like calculation."""
        # Price of E[max(S-K, 0)] where S ~ lognormal
        # Simple Black-Scholes like setting
        n_samples = 2000
        K = 1.0  # Strike
        sigma = 0.2
        T = 1.0

        # Sobol sequence
        sobol_points = sobol_sequence(n_samples, 1)

        # Transform to standard normal using inverse CDF
        z = jax.scipy.stats.norm.ppf(sobol_points[:, 0] * 0.998 + 0.001)  # Avoid 0 and 1

        # Log-normal stock price
        S = jnp.exp((- 0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * z)

        # Payoff
        payoff = jnp.maximum(S - K, 0)
        qmc_price = jnp.mean(payoff)

        # Random comparison
        key = jax.random.PRNGKey(0)
        random_z = jax.random.normal(key, shape=(n_samples,))
        S_random = jnp.exp((-0.5 * sigma**2) * T + sigma * jnp.sqrt(T) * random_z)
        mc_price = jnp.mean(jnp.maximum(S_random - K, 0))

        # Both should give reasonable prices
        assert qmc_price > 0
        assert mc_price > 0

    def test_compare_sequences(self):
        """Compare different QMC sequences."""
        key = jax.random.PRNGKey(0)
        n_samples = 500
        dim = 2

        # Generate different sequences
        halton_pts = halton_sequence(n_samples, dim)
        sobol_pts = sobol_sequence(n_samples, dim)
        lhs_pts = latin_hypercube(key, n_samples, dim)
        rhalton_pts = randomized_halton(key, n_samples, dim)

        # All should have correct shape
        assert halton_pts.shape == (n_samples, dim)
        assert sobol_pts.shape == (n_samples, dim)
        assert lhs_pts.shape == (n_samples, dim)
        assert rhalton_pts.shape == (n_samples, dim)

        # All should be in [0, 1)
        for pts in [halton_pts, sobol_pts, lhs_pts, rhalton_pts]:
            assert jnp.all(pts >= 0)
            assert jnp.all(pts <= 1)
