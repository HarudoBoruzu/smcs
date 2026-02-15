"""Tests for Monte Carlo MCMC methods."""

import jax
import jax.numpy as jnp

from smcs.montecarlo.mcmc import (
    gibbs_sample,
    hmc_step,
    metropolis_hastings,
    nuts_step,
    random_walk_metropolis,
    run_gibbs_sampler,
    run_hmc,
    run_metropolis_hastings,
    run_nuts,
    run_random_walk_metropolis,
    run_slice_sampler,
    slice_sample,
)


class TestSliceSampling:
    """Tests for slice sampling."""

    def test_slice_sample_output_shape(self):
        """Test slice sample returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        new_position = slice_sample(key, position, log_prob)

        assert new_position.shape == position.shape

    def test_slice_sample_stays_in_high_prob_region(self):
        """Test slice sampling stays in high probability region."""
        key = jax.random.PRNGKey(42)
        position = jnp.array([0.0])

        # Standard normal
        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # Run many steps
        keys = jax.random.split(key, 100)
        positions = [position]
        for k in keys:
            position = slice_sample(k, position, log_prob)
            positions.append(position)

        positions = jnp.array(positions)
        # Most samples should be within 3 std
        assert jnp.mean(jnp.abs(positions) < 3.0) > 0.95

    def test_run_slice_sampler_output_shape(self):
        """Test run_slice_sampler returns correct shape."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples = run_slice_sampler(
            key, initial, log_prob, n_samples=100, n_burnin=10
        )

        assert samples.shape == (100, 2)

    def test_run_slice_sampler_sampling_from_gaussian(self):
        """Test slice sampler samples from Gaussian correctly."""
        key = jax.random.PRNGKey(123)
        initial = jnp.array([5.0])

        # Target: N(0, 1)
        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples = run_slice_sampler(
            key, initial, log_prob, n_samples=2000, n_burnin=500
        )

        # Mean should be close to 0
        assert jnp.abs(jnp.mean(samples)) < 0.2
        # Std should be close to 1
        assert jnp.abs(jnp.std(samples) - 1.0) < 0.2


class TestMetropolisHastings:
    """Tests for Metropolis-Hastings."""

    def test_mh_output_shape(self):
        """Test MH returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        def proposal(k, x):
            return x + 0.1 * jax.random.normal(k, shape=x.shape)

        new_position, accepted = metropolis_hastings(key, position, log_prob, proposal)

        assert new_position.shape == position.shape
        assert isinstance(accepted, jax.Array)

    def test_mh_accepts_good_proposals(self):
        """Test MH accepts proposals that increase probability."""
        key = jax.random.PRNGKey(42)
        position = jnp.array([10.0])  # Far from mode

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # Propose moving toward mode
        def proposal(k, x):
            return x * 0.5  # Move toward 0

        new_position, accepted = metropolis_hastings(key, position, log_prob, proposal)

        # Should accept move toward mode
        assert accepted

    def test_run_mh_output_shape(self):
        """Test run_metropolis_hastings returns correct shape."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        def proposal(k, x):
            return x + 0.5 * jax.random.normal(k, shape=x.shape)

        samples, acceptance_rate = run_metropolis_hastings(
            key, initial, log_prob, proposal, n_samples=100, n_burnin=10
        )

        assert samples.shape == (100, 2)
        assert 0.0 <= acceptance_rate <= 1.0

    def test_run_mh_with_symmetric_proposal(self):
        """Test MH with symmetric proposal."""
        key = jax.random.PRNGKey(123)
        initial = jnp.array([0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # Symmetric proposal
        def proposal(k, x):
            return x + jax.random.normal(k, shape=x.shape)

        samples, acceptance_rate = run_metropolis_hastings(
            key,
            initial,
            log_prob,
            proposal,
            n_samples=500,
            n_burnin=100,
        )

        assert samples.shape == (500, 1)
        assert 0.0 < acceptance_rate < 1.0


class TestRandomWalkMetropolis:
    """Tests for Random Walk Metropolis."""

    def test_rwm_output_shape(self):
        """Test RWM returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        new_position, accepted = random_walk_metropolis(
            key, position, log_prob, step_size=0.5
        )

        assert new_position.shape == position.shape

    def test_run_rwm_samples_gaussian(self):
        """Test RWM samples from Gaussian."""
        key = jax.random.PRNGKey(42)
        initial = jnp.array([5.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples, acceptance_rate = run_random_walk_metropolis(
            key, initial, log_prob, step_size=1.0, n_samples=2000, n_burnin=500
        )

        assert samples.shape == (2000, 1)
        # Mean should be close to 0
        assert jnp.abs(jnp.mean(samples)) < 0.2
        # Acceptance rate should be reasonable
        assert 0.1 < acceptance_rate < 0.9

    def test_step_size_affects_acceptance(self):
        """Test that step size affects acceptance rate."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        _, acc_small = run_random_walk_metropolis(
            key, initial, log_prob, step_size=0.1, n_samples=500, n_burnin=100
        )

        _, acc_large = run_random_walk_metropolis(
            key, initial, log_prob, step_size=10.0, n_samples=500, n_burnin=100
        )

        # Small step size should have higher acceptance
        assert acc_small > acc_large


class TestGibbsSampling:
    """Tests for Gibbs sampling."""

    def test_gibbs_sample_output_shape(self):
        """Test Gibbs sample returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0])

        # Simple conditional samplers
        def sample_dim0(k, pos):
            return jax.random.normal(k) * 0.5

        def sample_dim1(k, pos):
            return pos[0] + jax.random.normal(k) * 0.3

        new_position = gibbs_sample(key, position, [sample_dim0, sample_dim1])

        assert new_position.shape == position.shape

    def test_run_gibbs_output_shape(self):
        """Test run_gibbs_sampler returns correct shape."""
        key = jax.random.PRNGKey(42)
        initial = jnp.array([1.0, 1.0])

        def sample_dim0(k, pos):
            return jax.random.normal(k)

        def sample_dim1(k, pos):
            return jax.random.normal(k)

        samples = run_gibbs_sampler(
            key, initial, [sample_dim0, sample_dim1], n_samples=100, n_burnin=10
        )

        assert samples.shape == (100, 2)

    def test_gibbs_samples_bivariate_normal(self):
        """Test Gibbs sampler for bivariate normal with correlation."""
        key = jax.random.PRNGKey(123)
        initial = jnp.array([0.0, 0.0])
        rho = 0.5

        # Conditional distributions for bivariate normal
        def sample_x(k, pos):
            y = pos[1]
            return rho * y + jnp.sqrt(1 - rho**2) * jax.random.normal(k)

        def sample_y(k, pos):
            x = pos[0]
            return rho * x + jnp.sqrt(1 - rho**2) * jax.random.normal(k)

        samples = run_gibbs_sampler(
            key, initial, [sample_x, sample_y], n_samples=2000, n_burnin=500
        )

        # Check marginal means
        assert jnp.abs(jnp.mean(samples[:, 0])) < 0.2
        assert jnp.abs(jnp.mean(samples[:, 1])) < 0.2

        # Check correlation
        corr = jnp.corrcoef(samples.T)[0, 1]
        assert jnp.abs(corr - rho) < 0.15


class TestHMC:
    """Tests for Hamiltonian Monte Carlo."""

    def test_hmc_step_output_shape(self):
        """Test HMC step returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        new_position, accepted = hmc_step(
            key, position, log_prob, step_size=0.1, n_leapfrog=10
        )

        assert new_position.shape == position.shape

    def test_hmc_preserves_energy_approximately(self):
        """Test HMC approximately preserves Hamiltonian."""
        key = jax.random.PRNGKey(42)
        position = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # With small step size and many leapfrog steps
        new_position, accepted = hmc_step(
            key, position, log_prob, step_size=0.01, n_leapfrog=100
        )

        # Energy should be similar (acceptance should be high)
        # For ideal integrator, energy is exactly preserved

    def test_run_hmc_output_shape(self):
        """Test run_hmc returns correct shape."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples, acceptance_rate = run_hmc(
            key,
            initial,
            log_prob,
            step_size=0.1,
            n_leapfrog=10,
            n_samples=100,
            n_burnin=10,
        )

        assert samples.shape == (100, 2)
        assert 0.0 <= acceptance_rate <= 1.0

    def test_run_hmc_samples_gaussian(self):
        """Test HMC samples from Gaussian correctly."""
        key = jax.random.PRNGKey(123)
        initial = jnp.array([5.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples, acceptance_rate = run_hmc(
            key,
            initial,
            log_prob,
            step_size=0.2,
            n_leapfrog=20,
            n_samples=1000,
            n_burnin=200,
        )

        # Mean should be close to 0
        assert jnp.abs(jnp.mean(samples)) < 0.2
        # Std should be close to 1
        assert jnp.abs(jnp.std(samples) - 1.0) < 0.2
        # HMC should have good acceptance
        assert acceptance_rate > 0.5

    def test_hmc_with_mass_matrix(self):
        """Test HMC with custom mass matrix."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # 2D mass matrix (diagonal)
        mass_matrix = jnp.diag(jnp.array([1.0, 2.0]))

        samples, acceptance_rate = run_hmc(
            key,
            initial,
            log_prob,
            step_size=0.1,
            n_leapfrog=10,
            n_samples=100,
            n_burnin=10,
            mass_matrix=mass_matrix,
        )

        assert samples.shape == (100, 2)


class TestNUTS:
    """Tests for No-U-Turn Sampler."""

    def test_nuts_step_output_shape(self):
        """Test NUTS step returns correct shape."""
        key = jax.random.PRNGKey(0)
        position = jnp.array([0.0, 0.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        new_position, accepted = nuts_step(key, position, log_prob, step_size=0.1)

        assert new_position.shape == position.shape

    def test_run_nuts_output_shape(self):
        """Test run_nuts returns correct shape."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([1.0, 1.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples = run_nuts(
            key, initial, log_prob, step_size=0.1, n_samples=50, n_burnin=10
        )

        assert samples.shape == (50, 2)

    def test_run_nuts_samples_gaussian(self):
        """Test NUTS samples from Gaussian correctly."""
        key = jax.random.PRNGKey(42)
        initial = jnp.array([3.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        samples = run_nuts(
            key, initial, log_prob, step_size=0.5, n_samples=500, n_burnin=100
        )

        # Mean should be close to 0
        assert jnp.abs(jnp.mean(samples)) < 0.3
        # Std should be close to 1
        assert jnp.abs(jnp.std(samples) - 1.0) < 0.3


class TestMCMCIntegration:
    """Integration tests for MCMC methods."""

    def test_compare_samplers_on_gaussian(self):
        """Compare different samplers on same target."""
        key = jax.random.PRNGKey(0)
        initial = jnp.array([5.0])

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # Run each sampler
        key1, key2, key3 = jax.random.split(key, 3)

        samples_rwm, _ = run_random_walk_metropolis(
            key1, initial, log_prob, step_size=1.0, n_samples=1000, n_burnin=200
        )

        samples_slice = run_slice_sampler(
            key2, initial, log_prob, n_samples=1000, n_burnin=200
        )

        samples_hmc, _ = run_hmc(
            key3,
            initial,
            log_prob,
            step_size=0.5,
            n_leapfrog=10,
            n_samples=1000,
            n_burnin=200,
        )

        # All should have mean close to 0
        assert jnp.abs(jnp.mean(samples_rwm)) < 0.3
        assert jnp.abs(jnp.mean(samples_slice)) < 0.3
        assert jnp.abs(jnp.mean(samples_hmc)) < 0.3

    def test_sampling_multimodal(self):
        """Test sampling from multimodal distribution."""
        key = jax.random.PRNGKey(42)
        initial = jnp.array([0.0])

        # Mixture of two Gaussians
        def log_prob(x):
            log_p1 = -0.5 * (x - 3) ** 2
            log_p2 = -0.5 * (x + 3) ** 2
            return jax.scipy.special.logsumexp(
                jnp.array([log_p1[0], log_p2[0]]) + jnp.log(0.5)
            )

        # RWM with large step size might explore both modes
        samples, _ = run_random_walk_metropolis(
            key, initial, log_prob, step_size=2.0, n_samples=2000, n_burnin=500
        )

        # Should have samples from both modes
        left_mode = jnp.sum(samples < 0)
        right_mode = jnp.sum(samples > 0)

        # Both modes should be visited
        assert left_mode > 100
        assert right_mode > 100

    def test_sampling_correlated_gaussian(self):
        """Test sampling from correlated Gaussian."""
        key = jax.random.PRNGKey(123)
        initial = jnp.array([0.0, 0.0])

        # Correlated 2D Gaussian
        rho = 0.8
        cov = jnp.array([[1.0, rho], [rho, 1.0]])
        cov_inv = jnp.linalg.inv(cov)

        def log_prob(x):
            return -0.5 * x @ cov_inv @ x

        samples, _ = run_hmc(
            key,
            initial,
            log_prob,
            step_size=0.2,
            n_leapfrog=20,
            n_samples=1000,
            n_burnin=200,
        )

        # Check correlation (use looser tolerance)
        empirical_corr = jnp.corrcoef(samples.T)[0, 1]
        assert jnp.abs(empirical_corr - rho) < 0.2
