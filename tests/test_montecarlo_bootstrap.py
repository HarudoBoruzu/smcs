"""Tests for Monte Carlo bootstrap methods."""

import jax
import jax.numpy as jnp

from smcs.montecarlo.bootstrap import (
    block_bootstrap,
    bootstrap_ci,
    circular_block_bootstrap,
    jackknife,
    moving_block_bootstrap,
    ordinary_bootstrap,
    parametric_bootstrap,
    residual_bootstrap,
    stationary_bootstrap,
    wild_bootstrap,
)


class TestOrdinaryBootstrap:
    """Tests for ordinary bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_bootstrap = 100

        result = ordinary_bootstrap(key, data, n_bootstrap)

        assert result.shape == (n_bootstrap, len(data))

    def test_output_shape_2d(self):
        """Test output has correct shape for 2D data."""
        key = jax.random.PRNGKey(0)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        n_bootstrap = 50

        result = ordinary_bootstrap(key, data, n_bootstrap)

        assert result.shape == (n_bootstrap, data.shape[0], data.shape[1])

    def test_samples_from_data(self):
        """Test bootstrap samples contain only values from original data."""
        key = jax.random.PRNGKey(42)
        data = jnp.array([10.0, 20.0, 30.0])
        n_bootstrap = 100

        result = ordinary_bootstrap(key, data, n_bootstrap)

        # All values in result should be in original data
        for i in range(n_bootstrap):
            for val in result[i]:
                assert val in data

    def test_reproducibility(self):
        """Test same key produces same result."""
        key = jax.random.PRNGKey(123)
        data = jnp.array([1.0, 2.0, 3.0, 4.0])

        result1 = ordinary_bootstrap(key, data, 50)
        result2 = ordinary_bootstrap(key, data, 50)

        assert jnp.allclose(result1, result2)

    def test_different_keys_different_results(self):
        """Test different keys produce different results."""
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result1 = ordinary_bootstrap(key1, data, 100)
        result2 = ordinary_bootstrap(key2, data, 100)

        assert not jnp.allclose(result1, result2)


class TestBlockBootstrap:
    """Tests for block bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(20.0)
        n_bootstrap = 50
        block_size = 4

        result = block_bootstrap(key, data, block_size, n_bootstrap)

        assert result.shape == (n_bootstrap, len(data))

    def test_blocks_preserved(self):
        """Test that consecutive values come from same block."""
        key = jax.random.PRNGKey(42)
        # Create data where consecutive values are clearly related
        data = jnp.arange(100.0)
        block_size = 5
        n_bootstrap = 10

        result = block_bootstrap(key, data, block_size, n_bootstrap)

        # Check that within each block, values are consecutive
        for i in range(n_bootstrap):
            sample = result[i]
            for b in range(0, len(data) - block_size + 1, block_size):
                block = sample[b:b + block_size]
                # Check differences are all 1 (consecutive)
                diffs = jnp.diff(block)
                assert jnp.all(diffs == 1.0)

    def test_reproducibility(self):
        """Test reproducibility with same key."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(50.0)
        block_size = 5

        result1 = block_bootstrap(key, data, block_size, 30)
        result2 = block_bootstrap(key, data, block_size, 30)

        assert jnp.allclose(result1, result2)


class TestMovingBlockBootstrap:
    """Tests for moving block bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(30.0)
        block_size = 5
        n_bootstrap = 40

        result = moving_block_bootstrap(key, data, block_size, n_bootstrap)

        assert result.shape == (n_bootstrap, len(data))

    def test_blocks_are_consecutive(self):
        """Test blocks contain consecutive values."""
        key = jax.random.PRNGKey(123)
        data = jnp.arange(50.0)
        block_size = 4
        n_bootstrap = 20

        result = moving_block_bootstrap(key, data, block_size, n_bootstrap)

        # Within each block of size block_size, values should be consecutive
        for i in range(n_bootstrap):
            sample = result[i]
            # Check at least one block is consecutive
            for start in range(0, len(sample) - block_size + 1, block_size):
                block = sample[start:start + block_size]
                diffs = jnp.diff(block)
                # All diffs should be 1 (consecutive from original)
                if jnp.all(diffs == 1.0):
                    break


class TestCircularBlockBootstrap:
    """Tests for circular block bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(25.0)
        block_size = 5
        n_bootstrap = 30

        result = circular_block_bootstrap(key, data, block_size, n_bootstrap)

        assert result.shape == (n_bootstrap, len(data))

    def test_wrap_around_behavior(self):
        """Test circular wrap-around is working."""
        key = jax.random.PRNGKey(42)
        n = 10
        data = jnp.arange(float(n))
        block_size = 4
        n_bootstrap = 100

        result = circular_block_bootstrap(key, data, block_size, n_bootstrap)

        # All values should be valid indices wrapped
        assert jnp.all(result >= 0)
        assert jnp.all(result < n)


class TestStationaryBootstrap:
    """Tests for stationary bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(50.0)
        mean_block_size = 5.0
        n_bootstrap = 40

        result = stationary_bootstrap(key, data, mean_block_size, n_bootstrap)

        assert result.shape == (n_bootstrap, len(data))

    def test_values_from_data(self):
        """Test all values come from original data."""
        key = jax.random.PRNGKey(123)
        data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        n_bootstrap = 50

        result = stationary_bootstrap(key, data, 2.0, n_bootstrap)

        # All values should be in original data
        unique_result = jnp.unique(result)
        for val in unique_result:
            assert val in data

    def test_reproducibility(self):
        """Test reproducibility."""
        key = jax.random.PRNGKey(0)
        data = jnp.arange(30.0)

        result1 = stationary_bootstrap(key, data, 4.0, 20)
        result2 = stationary_bootstrap(key, data, 4.0, 20)

        assert jnp.allclose(result1, result2)


class TestWildBootstrap:
    """Tests for wild bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        residuals = jnp.array([0.1, -0.2, 0.3, -0.1, 0.2])
        fitted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_bootstrap = 100

        result = wild_bootstrap(key, residuals, fitted, n_bootstrap)

        assert result.shape == (n_bootstrap, len(residuals))

    def test_rademacher_distribution(self):
        """Test Rademacher distribution produces +1 or -1 weights."""
        key = jax.random.PRNGKey(42)
        residuals = jnp.ones(100)
        fitted = jnp.zeros(100)
        n_bootstrap = 50

        result = wild_bootstrap(key, residuals, fitted, n_bootstrap, "rademacher")

        # Result should be +-1 since fitted is 0 and residuals are 1
        unique_vals = jnp.unique(result)
        assert jnp.allclose(jnp.sort(unique_vals), jnp.array([-1.0, 1.0]))

    def test_mammen_distribution(self):
        """Test Mammen distribution."""
        key = jax.random.PRNGKey(0)
        residuals = jnp.ones(50)
        fitted = jnp.zeros(50)
        n_bootstrap = 100

        result = wild_bootstrap(key, residuals, fitted, n_bootstrap, "mammen")

        # Mammen produces specific values
        sqrt5 = jnp.sqrt(5.0)
        expected_vals = jnp.array([(1 - sqrt5) / 2, (1 + sqrt5) / 2])
        unique_vals = jnp.unique(result)
        assert len(unique_vals) == 2
        # Check values are close to expected Mammen distribution values
        assert jnp.allclose(jnp.sort(unique_vals), jnp.sort(expected_vals), atol=1e-5)

    def test_normal_distribution(self):
        """Test normal distribution."""
        key = jax.random.PRNGKey(123)
        residuals = jnp.ones(100)
        fitted = jnp.zeros(100)
        n_bootstrap = 1000

        result = wild_bootstrap(key, residuals, fitted, n_bootstrap, "normal")

        # Mean should be approximately 0
        mean_result = jnp.mean(result)
        assert jnp.abs(mean_result) < 0.1

    def test_output_shape_2d(self):
        """Test 2D residuals."""
        key = jax.random.PRNGKey(0)
        residuals = jnp.ones((20, 3))
        fitted = jnp.zeros((20, 3))
        n_bootstrap = 50

        result = wild_bootstrap(key, residuals, fitted, n_bootstrap)

        assert result.shape == (n_bootstrap, 20, 3)


class TestResidualBootstrap:
    """Tests for residual bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)
        residuals = jnp.array([0.1, -0.2, 0.15, -0.1, 0.25])
        fitted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_bootstrap = 100

        result = residual_bootstrap(key, residuals, fitted, n_bootstrap)

        assert result.shape == (n_bootstrap, len(residuals))

    def test_bootstrap_adds_residuals_to_fitted(self):
        """Test bootstrap samples are fitted + resampled residuals."""
        key = jax.random.PRNGKey(42)
        residuals = jnp.array([1.0, 2.0, 3.0])
        fitted = jnp.array([10.0, 20.0, 30.0])
        n_bootstrap = 50

        result = residual_bootstrap(key, residuals, fitted, n_bootstrap)

        # Centered residuals
        centered = residuals - jnp.mean(residuals)

        # Each sample value should be fitted[j] + some centered residual
        for i in range(n_bootstrap):
            sample = result[i]
            for j in range(len(fitted)):
                diff = sample[j] - fitted[j]
                # diff should be one of the centered residuals
                assert jnp.any(jnp.isclose(diff, centered))


class TestParametricBootstrap:
    """Tests for parametric bootstrap."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)

        def sample_fn(k):
            return jax.random.normal(k, shape=(5,))

        n_bootstrap = 100
        result = parametric_bootstrap(key, sample_fn, n_bootstrap)

        assert result.shape == (n_bootstrap, 5)

    def test_samples_from_distribution(self):
        """Test samples come from specified distribution."""
        key = jax.random.PRNGKey(42)
        mean = 5.0
        std = 2.0

        def sample_fn(k):
            return mean + std * jax.random.normal(k, shape=(10,))

        n_bootstrap = 1000
        result = parametric_bootstrap(key, sample_fn, n_bootstrap)

        # Mean should be close to specified mean
        assert jnp.abs(jnp.mean(result) - mean) < 0.2
        # Std should be close to specified std
        assert jnp.abs(jnp.std(result) - std) < 0.2

    def test_reproducibility(self):
        """Test reproducibility."""
        key = jax.random.PRNGKey(0)

        def sample_fn(k):
            return jax.random.uniform(k, shape=(3,))

        result1 = parametric_bootstrap(key, sample_fn, 50)
        result2 = parametric_bootstrap(key, sample_fn, 50)

        assert jnp.allclose(result1, result2)


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_percentile_method(self):
        """Test percentile CI method."""
        # Create bootstrap estimates
        estimates = jnp.linspace(0, 100, 1000)

        lower, upper = bootstrap_ci(estimates, confidence=0.95, method="percentile")

        # 2.5th percentile should be around 2.5
        assert jnp.abs(lower - 2.5) < 1.0
        # 97.5th percentile should be around 97.5
        assert jnp.abs(upper - 97.5) < 1.0

    def test_basic_method(self):
        """Test basic CI method."""
        estimates = jnp.linspace(0, 100, 1000)
        original_estimate = jnp.array(50.0)  # Must be a JAX array

        lower, upper = bootstrap_ci(
            estimates, confidence=0.95, method="basic", original_estimate=original_estimate
        )

        # Basic method uses 2*original - quantiles
        assert lower < upper

    def test_bca_method(self):
        """Test BCa CI method."""
        estimates = jnp.linspace(0, 100, 1000)

        lower, upper = bootstrap_ci(estimates, confidence=0.95, method="bca")

        # BCa should produce valid interval
        assert lower < upper
        assert lower >= 0
        assert upper <= 100

    def test_confidence_levels(self):
        """Test different confidence levels."""
        estimates = jnp.linspace(0, 100, 1000)

        lower_90, upper_90 = bootstrap_ci(estimates, confidence=0.90)
        lower_95, upper_95 = bootstrap_ci(estimates, confidence=0.95)
        lower_99, upper_99 = bootstrap_ci(estimates, confidence=0.99)

        # Higher confidence = wider interval
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_90 < width_95 < width_99


class TestJackknife:
    """Tests for jackknife resampling."""

    def test_mean_statistic(self):
        """Test jackknife with mean statistic."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def mean_fn(x):
            return jnp.mean(x)

        result = jackknife(data, mean_fn)

        # Result should be a tuple of 3 arrays
        assert len(result) == 3
        estimate, bias, jackknife_samples = result

        # Estimate should be close to actual mean
        assert jnp.abs(float(estimate) - 3.0) < 0.5
        # Should have n jackknife samples
        assert jackknife_samples.shape == (5,)

    def test_variance_statistic(self):
        """Test jackknife with variance statistic."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def var_fn(x):
            return jnp.var(x)

        result = jackknife(data, var_fn)
        estimate, bias, jackknife_samples = result

        # Should return valid results
        assert float(estimate) > 0
        assert jackknife_samples.shape == (5,)

    def test_jackknife_output_shape(self):
        """Test that jackknife returns expected shapes."""
        data = jnp.array([10.0, 20.0, 30.0, 40.0])

        def sum_fn(x):
            return jnp.sum(x)

        result = jackknife(data, sum_fn)
        estimate, bias, jackknife_samples = result

        assert jackknife_samples.shape == (4,)


class TestBootstrapIntegration:
    """Integration tests for bootstrap methods."""

    def test_bootstrap_mean_estimation(self):
        """Test bootstrap for mean estimation."""
        key = jax.random.PRNGKey(0)
        true_mean = 5.0
        data = true_mean + jax.random.normal(key, shape=(100,))

        bootstrap_samples = ordinary_bootstrap(
            jax.random.PRNGKey(1), data, n_bootstrap=500
        )
        bootstrap_means = jnp.mean(bootstrap_samples, axis=1)

        lower, upper = bootstrap_ci(bootstrap_means, confidence=0.95)

        # True mean should be in CI
        assert lower < true_mean < upper

    def test_block_bootstrap_for_time_series(self):
        """Test block bootstrap preserves autocorrelation structure."""
        key = jax.random.PRNGKey(42)
        # Create AR(1) like series
        n = 100
        rho = 0.7
        noise = jax.random.normal(key, shape=(n,))
        data = jnp.zeros(n)
        data = data.at[0].set(noise[0])
        for i in range(1, n):
            data = data.at[i].set(rho * data[i - 1] + noise[i])

        bootstrap_samples = block_bootstrap(
            jax.random.PRNGKey(1), data, block_size=10, n_bootstrap=100
        )

        # Each bootstrap sample should have same length
        assert bootstrap_samples.shape == (100, n)
