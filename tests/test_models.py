"""Tests for state space models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smcs.models.dlm import (
    LocalLevelModel,
    LocalLevelParams,
    LocalLinearTrendModel,
    LocalLinearTrendParams,
)
from smcs.models.sv import SVModel, SVParams
from smcs.models.distributions import Normal, MultivariateNormal


class TestNormalDistribution:
    """Tests for Normal distribution."""

    def test_sample_shape(self):
        """Sample should return scalar."""
        key = jax.random.PRNGKey(42)
        dist = Normal(loc=0.0, scale=1.0)

        sample = dist.sample(key)

        assert sample.shape == ()

    def test_log_prob_standard_normal(self):
        """Log prob at mean should be -0.5*log(2*pi)."""
        dist = Normal(loc=0.0, scale=1.0)

        log_p = dist.log_prob(jnp.array(0.0))

        expected = -0.5 * jnp.log(2 * jnp.pi)
        np.testing.assert_allclose(log_p, expected, rtol=1e-5)


class TestMultivariateNormal:
    """Tests for MultivariateNormal distribution."""

    def test_sample_shape(self):
        """Sample should have correct dimension."""
        key = jax.random.PRNGKey(42)
        dim = 3
        dist = MultivariateNormal(
            loc=jnp.zeros(dim),
            covariance_matrix=jnp.eye(dim)
        )

        sample = dist.sample(key)

        assert sample.shape == (dim,)

    def test_log_prob_at_mean(self):
        """Log prob at mean should be maximum."""
        dim = 2
        mean = jnp.array([1.0, 2.0])
        dist = MultivariateNormal(
            loc=mean,
            covariance_matrix=jnp.eye(dim)
        )

        log_p_mean = dist.log_prob(mean)
        log_p_away = dist.log_prob(mean + 1.0)

        assert log_p_mean > log_p_away


class TestLocalLevelModel:
    """Tests for Local Level Model."""

    def test_state_dim(self):
        """State dimension should be 1."""
        model = LocalLevelModel()
        assert model.state_dim == 1

    def test_obs_dim(self):
        """Observation dimension should be 1."""
        model = LocalLevelModel()
        assert model.obs_dim == 1

    def test_initial_distribution(self):
        """Initial distribution should sample correctly."""
        key = jax.random.PRNGKey(42)
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=1.0, C0=0.5)

        dist = model.initial_distribution(params)
        sample = dist.sample(key)

        assert jnp.isfinite(sample)

    def test_transition_distribution(self):
        """Transition should be centered at current state."""
        key = jax.random.PRNGKey(42)
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)
        state = jnp.array([5.0])

        dist = model.transition_distribution(params, state)

        # Mean should be current state
        assert dist.loc == state[0]

    def test_emission_distribution(self):
        """Emission should be centered at state."""
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)
        state = jnp.array([3.0])

        dist = model.emission_distribution(params, state)

        assert dist.loc == state[0]


class TestLocalLinearTrendModel:
    """Tests for Local Linear Trend Model."""

    def test_state_dim(self):
        """State dimension should be 2 (level, slope)."""
        model = LocalLinearTrendModel()
        assert model.state_dim == 2

    def test_obs_dim(self):
        """Observation dimension should be 1."""
        model = LocalLinearTrendModel()
        assert model.obs_dim == 1


class TestSVModel:
    """Tests for Stochastic Volatility Model."""

    def test_state_dim(self):
        """State dimension should be 1 (log-volatility)."""
        model = SVModel()
        assert model.state_dim == 1

    def test_emission_scale_depends_on_state(self):
        """Emission scale should depend on exp(h/2)."""
        model = SVModel()
        params = SVParams(mu=0.0, phi=0.95, sigma_eta=0.1, h0=0.0, P0=0.1)

        state_low = jnp.array([-2.0])  # Low volatility
        state_high = jnp.array([2.0])  # High volatility

        dist_low = model.emission_distribution(params, state_low)
        dist_high = model.emission_distribution(params, state_high)

        assert dist_low.scale < dist_high.scale
