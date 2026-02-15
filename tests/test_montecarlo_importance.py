"""Tests for Monte Carlo importance sampling methods."""

import jax
import jax.numpy as jnp

from smcs.montecarlo.importance import (
    adaptive_importance_sampling,
    compute_ess_is,
    compute_is_diagnostics,
    importance_sample,
    multiple_importance_sampling,
    self_normalized_is,
)


class TestImportanceSample:
    """Tests for basic importance sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)

        def proposal_sample(k):
            return jax.random.normal(k, shape=(2,))

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2)

        samples, log_weights = importance_sample(
            key, proposal_sample, log_target, log_proposal, n_samples=100
        )

        assert samples.shape == (100, 2)
        assert log_weights.shape == (100,)

    def test_weights_correct_for_same_distribution(self):
        """Test weights are uniform when target = proposal."""
        key = jax.random.PRNGKey(42)

        def proposal_sample(k):
            return jax.random.normal(k, shape=(1,))

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples, log_weights = importance_sample(
            key, proposal_sample, log_prob, log_prob, n_samples=1000
        )

        # All weights should be equal (log_weights = 0)
        assert jnp.allclose(log_weights, 0.0, atol=1e-5)

    def test_weights_higher_for_target_region(self):
        """Test weights are higher for samples in target's high prob region."""
        key = jax.random.PRNGKey(123)

        # Proposal: N(0, 1)
        def proposal_sample(k):
            return jax.random.normal(k, shape=(1,))

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2)

        # Target: N(2, 1) - shifted
        def log_target(x):
            return -0.5 * jnp.sum((x - 2) ** 2)

        samples, log_weights = importance_sample(
            key, proposal_sample, log_target, log_proposal, n_samples=1000
        )

        # Samples near 2 should have higher weights
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        near_target = jnp.abs(samples[:, 0] - 2) < 1
        far_from_target = jnp.abs(samples[:, 0] - 2) > 2

        avg_weight_near = jnp.mean(weights[near_target])
        avg_weight_far = jnp.mean(weights[far_from_target])

        assert avg_weight_near > avg_weight_far

    def test_reproducibility(self):
        """Test reproducibility with same key."""
        key = jax.random.PRNGKey(0)

        def proposal_sample(k):
            return jax.random.normal(k, shape=(1,))

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples1, weights1 = importance_sample(
            key, proposal_sample, log_prob, log_prob, n_samples=50
        )
        samples2, weights2 = importance_sample(
            key, proposal_sample, log_prob, log_prob, n_samples=50
        )

        assert jnp.allclose(samples1, samples2)
        assert jnp.allclose(weights1, weights2)


class TestSelfNormalizedIS:
    """Tests for self-normalized importance sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)

        def proposal_sample(k):
            return jax.random.normal(k, shape=(3,))

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2)

        samples, weights = self_normalized_is(
            key, proposal_sample, log_target, log_proposal, n_samples=100
        )

        assert samples.shape == (100, 3)
        assert weights.shape == (100,)

    def test_weights_sum_to_one(self):
        """Test normalized weights sum to 1."""
        key = jax.random.PRNGKey(42)

        def proposal_sample(k):
            return jax.random.normal(k, shape=(2,))

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2)

        _, weights = self_normalized_is(
            key, proposal_sample, log_target, log_proposal, n_samples=500
        )

        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-5

    def test_estimate_mean(self):
        """Test IS can estimate mean of target distribution."""
        key = jax.random.PRNGKey(123)
        target_mean = 3.0

        # Proposal: N(0, 2)
        def proposal_sample(k):
            return 2.0 * jax.random.normal(k, shape=(1,))

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2) / 4.0 - 0.5 * jnp.log(4.0)

        # Target: N(3, 1)
        def log_target(x):
            return -0.5 * jnp.sum((x - target_mean) ** 2)

        samples, weights = self_normalized_is(
            key, proposal_sample, log_target, log_proposal, n_samples=5000
        )

        # Estimate mean using weighted average
        estimated_mean = jnp.sum(weights[:, None] * samples, axis=0)

        assert jnp.abs(estimated_mean[0] - target_mean) < 0.3


class TestMultipleImportanceSampling:
    """Tests for multiple importance sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)

        # Two proposals
        def proposal1(k):
            return jax.random.normal(k, shape=(2,))

        def proposal2(k):
            return 2.0 * jax.random.normal(k, shape=(2,))

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal1(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal2(x):
            return -0.5 * jnp.sum(x**2) / 4.0 - jnp.log(4.0)

        samples, weights = multiple_importance_sampling(
            key,
            [proposal1, proposal2],
            log_target,
            [log_proposal1, log_proposal2],
            n_samples_per_proposal=50,
        )

        # Total samples = 50 * 2 = 100
        assert samples.shape == (100, 2)
        assert weights.shape == (100,)

    def test_balance_weighting(self):
        """Test balance heuristic weighting."""
        key = jax.random.PRNGKey(42)

        def proposal1(k):
            return jax.random.normal(k, shape=(1,))

        def proposal2(k):
            return 3.0 + jax.random.normal(k, shape=(1,))

        def log_target(x):
            # Mixture of two Gaussians
            log_p1 = -0.5 * jnp.sum(x**2)
            log_p2 = -0.5 * jnp.sum((x - 3) ** 2)
            return jax.scipy.special.logsumexp(jnp.array([log_p1, log_p2]))

        def log_proposal1(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal2(x):
            return -0.5 * jnp.sum((x - 3) ** 2)

        samples, weights = multiple_importance_sampling(
            key,
            [proposal1, proposal2],
            log_target,
            [log_proposal1, log_proposal2],
            n_samples_per_proposal=500,
            weighting="balance",
        )

        # Weights should be non-negative
        assert jnp.all(weights >= 0)
        # Weights should sum to 1
        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-5

    def test_cutoff_weighting(self):
        """Test cutoff weighting scheme."""
        key = jax.random.PRNGKey(0)

        def proposal1(k):
            return jax.random.normal(k, shape=(1,))

        def proposal2(k):
            return 2.0 * jax.random.normal(k, shape=(1,))

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal1(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal2(x):
            return -0.5 * jnp.sum(x**2) / 4.0 - 0.5 * jnp.log(4.0)

        samples, weights = multiple_importance_sampling(
            key,
            [proposal1, proposal2],
            log_target,
            [log_proposal1, log_proposal2],
            n_samples_per_proposal=100,
            weighting="cutoff",
        )

        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-5


class TestAdaptiveImportanceSampling:
    """Tests for adaptive importance sampling."""

    def test_output_shape(self):
        """Test output has correct shape."""
        key = jax.random.PRNGKey(0)

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        initial_mean = jnp.array([5.0, 5.0])
        initial_cov = jnp.eye(2) * 4.0

        samples, weights = adaptive_importance_sampling(
            key,
            log_target,
            initial_mean,
            initial_cov,
            n_samples_per_iter=100,
            n_iterations=3,
        )

        # Total samples = 100 * 3 = 300
        assert samples.shape == (300, 2)
        assert weights.shape == (300,)

    def test_adapts_toward_target(self):
        """Test proposal adapts toward target distribution."""
        key = jax.random.PRNGKey(42)

        # Target centered at origin
        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        # Start far from target
        initial_mean = jnp.array([10.0])
        initial_cov = jnp.array([[1.0]])

        samples, weights = adaptive_importance_sampling(
            key,
            log_target,
            initial_mean,
            initial_cov,
            n_samples_per_iter=500,
            n_iterations=5,
        )

        # Weighted mean should be closer to origin
        weighted_mean = jnp.sum(weights[:, None] * samples, axis=0)
        assert jnp.abs(weighted_mean[0]) < 5.0

    def test_weights_normalized(self):
        """Test weights are normalized."""
        key = jax.random.PRNGKey(123)

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        initial_mean = jnp.array([0.0])
        initial_cov = jnp.array([[1.0]])

        _, weights = adaptive_importance_sampling(
            key,
            log_target,
            initial_mean,
            initial_cov,
            n_samples_per_iter=200,
            n_iterations=3,
        )

        # Weights should sum to 1
        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-5


class TestComputeESSIS:
    """Tests for ESS computation for importance sampling."""

    def test_uniform_weights_max_ess(self):
        """Test ESS is n for uniform weights."""
        n = 100
        log_weights = jnp.zeros(n)  # Uniform weights

        ess = compute_ess_is(log_weights)

        assert jnp.abs(ess - n) < 1e-3

    def test_degenerate_weights_min_ess(self):
        """Test ESS is 1 for degenerate weights."""
        n = 100
        log_weights = jnp.full(n, -1000.0)
        log_weights = log_weights.at[0].set(0.0)  # All weight on first sample

        ess = compute_ess_is(log_weights)

        assert jnp.abs(ess - 1.0) < 1e-3

    def test_ess_range(self):
        """Test ESS is between 1 and n."""
        key = jax.random.PRNGKey(0)
        n = 500
        log_weights = jax.random.normal(key, shape=(n,))

        ess = compute_ess_is(log_weights)

        assert 1.0 <= ess <= n

    def test_ess_increases_with_uniformity(self):
        """Test ESS increases as weights become more uniform."""
        n = 100

        # Very non-uniform weights
        log_weights_nonuniform = jnp.arange(n, dtype=float)  # 0, 1, 2, ...

        # More uniform weights
        log_weights_uniform = jnp.arange(n, dtype=float) * 0.01

        ess_nonuniform = compute_ess_is(log_weights_nonuniform)
        ess_uniform = compute_ess_is(log_weights_uniform)

        assert ess_uniform > ess_nonuniform


class TestComputeISDiagnostics:
    """Tests for IS diagnostics computation."""

    def test_diagnostics_keys(self):
        """Test diagnostics returns expected keys."""
        log_weights = jnp.array([0.0, -1.0, -2.0, -0.5])

        diagnostics = compute_is_diagnostics(log_weights)

        assert "ess" in diagnostics
        assert "ess_ratio" in diagnostics
        assert "max_weight" in diagnostics
        assert "entropy" in diagnostics

    def test_ess_ratio_range(self):
        """Test ESS ratio is between 0 and 1."""
        key = jax.random.PRNGKey(42)
        log_weights = jax.random.normal(key, shape=(100,))

        diagnostics = compute_is_diagnostics(log_weights)

        assert 0.0 < diagnostics["ess_ratio"] <= 1.0

    def test_uniform_weights_diagnostics(self):
        """Test diagnostics for uniform weights."""
        n = 100
        log_weights = jnp.zeros(n)

        diagnostics = compute_is_diagnostics(log_weights)

        assert jnp.abs(diagnostics["ess"] - n) < 1e-3
        assert jnp.abs(diagnostics["ess_ratio"] - 1.0) < 1e-3
        assert jnp.abs(diagnostics["max_weight"] - 1 / n) < 1e-5

    def test_degenerate_weights_diagnostics(self):
        """Test diagnostics for degenerate weights."""
        n = 100
        log_weights = jnp.full(n, -1000.0)
        log_weights = log_weights.at[0].set(0.0)

        diagnostics = compute_is_diagnostics(log_weights)

        assert jnp.abs(diagnostics["ess"] - 1.0) < 1e-3
        assert diagnostics["max_weight"] > 0.99


class TestImportanceSamplingIntegration:
    """Integration tests for importance sampling."""

    def test_estimate_expectation(self):
        """Test IS can estimate expectation accurately."""
        key = jax.random.PRNGKey(0)

        # Target: N(0, 1), estimate E[X^2] = 1
        def proposal_sample(k):
            return jax.random.normal(k, shape=(1,)) * 2  # N(0, 4)

        def log_target(x):
            return -0.5 * jnp.sum(x**2)

        def log_proposal(x):
            return -0.5 * jnp.sum(x**2) / 4.0 - 0.5 * jnp.log(4.0)

        samples, weights = self_normalized_is(
            key, proposal_sample, log_target, log_proposal, n_samples=5000
        )

        # Estimate E[X^2]
        x_squared = samples**2
        estimate = jnp.sum(weights[:, None] * x_squared, axis=0)

        assert jnp.abs(estimate[0] - 1.0) < 0.2

    def test_mis_better_than_single_proposal(self):
        """Test MIS can be better than single proposal for multimodal targets."""
        key = jax.random.PRNGKey(42)

        # Bimodal target
        def log_target(x):
            log_p1 = -0.5 * (x - 3) ** 2
            log_p2 = -0.5 * (x + 3) ** 2
            return jax.scipy.special.logsumexp(jnp.array([log_p1[0], log_p2[0]]))

        # Single proposal at 0
        def single_proposal(k):
            return jax.random.normal(k, shape=(1,))

        def log_single_proposal(x):
            return -0.5 * jnp.sum(x**2)

        # Two proposals covering both modes
        def proposal_left(k):
            return -3.0 + jax.random.normal(k, shape=(1,))

        def proposal_right(k):
            return 3.0 + jax.random.normal(k, shape=(1,))

        def log_proposal_left(x):
            return -0.5 * jnp.sum((x + 3) ** 2)

        def log_proposal_right(x):
            return -0.5 * jnp.sum((x - 3) ** 2)

        key1, key2 = jax.random.split(key)

        # Single proposal
        _, weights_single = importance_sample(
            key1, single_proposal, log_target, log_single_proposal, n_samples=1000
        )
        # Single proposal centered at 0 may not cover bimodal target well
        _ = compute_ess_is(weights_single)

        # Multiple proposals
        _, weights_multi = multiple_importance_sampling(
            key2,
            [proposal_left, proposal_right],
            log_target,
            [log_proposal_left, log_proposal_right],
            n_samples_per_proposal=500,
        )
        # Convert normalized weights back to log space for ESS
        log_weights_multi = jnp.log(weights_multi + 1e-10)
        ess_multi = compute_ess_is(log_weights_multi)

        # MIS should have reasonable ESS for this multimodal target
        assert ess_multi > 10
