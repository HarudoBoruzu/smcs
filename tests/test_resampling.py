"""Tests for resampling algorithms."""

import jax
import jax.numpy as jnp
import pytest

from smcs.core.resampling import (
    multinomial_resample,
    resample,
    residual_resample,
    stratified_resample,
    systematic_resample,
)


class TestSystematicResampling:
    """Tests for systematic resampling."""

    def test_preserves_count(self):
        """Resampling should preserve particle count."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        indices = systematic_resample(key, log_weights)

        assert len(indices) == n_particles
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_particles)

    def test_concentrates_on_high_weight(self):
        """High weight particles should be selected more often."""
        key = jax.random.PRNGKey(42)
        n_particles = 1000
        log_weights = jnp.zeros(n_particles)
        log_weights = log_weights.at[0].set(10.0)  # High weight for particle 0

        indices = systematic_resample(key, log_weights)

        count_0 = jnp.sum(indices == 0)
        # Particle 0 should be selected many times
        assert count_0 > n_particles * 0.9

    def test_uniform_weights_spread(self):
        """With uniform weights, selections should be spread evenly."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jnp.zeros(n_particles)  # Uniform weights

        indices = systematic_resample(key, log_weights)

        # Each particle should be selected roughly once
        counts = jnp.bincount(indices, length=n_particles)
        assert jnp.all(counts >= 0)
        assert jnp.all(counts <= 2)  # At most 2 for uniform


class TestMultinomialResampling:
    """Tests for multinomial resampling."""

    def test_preserves_count(self):
        """Resampling should preserve particle count."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        indices = multinomial_resample(key, log_weights)

        assert len(indices) == n_particles
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_particles)


class TestStratifiedResampling:
    """Tests for stratified resampling."""

    def test_preserves_count(self):
        """Resampling should preserve particle count."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        indices = stratified_resample(key, log_weights)

        assert len(indices) == n_particles
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_particles)


class TestResidualResampling:
    """Tests for residual resampling."""

    def test_preserves_count(self):
        """Resampling should preserve particle count."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        indices = residual_resample(key, log_weights)

        assert len(indices) == n_particles
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < n_particles)


class TestResampleDispatch:
    """Tests for the resample dispatch function."""

    @pytest.mark.parametrize(
        "method",
        ["systematic", "multinomial", "stratified", "residual"],
    )
    def test_all_methods(self, method):
        """All resampling methods should work."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        log_weights = jax.random.normal(key, shape=(n_particles,))

        indices = resample(key, log_weights, method)

        assert len(indices) == n_particles
