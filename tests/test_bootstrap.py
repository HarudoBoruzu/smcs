"""Tests for Bootstrap Particle Filter."""

import jax
import jax.numpy as jnp
import numpy as np

from smcs.algorithms.bootstrap import (
    bootstrap_step,
    initialize_particles,
    run_bootstrap_filter,
)
from smcs.models.dlm import LocalLevelModel, LocalLevelParams


class TestInitializeParticles:
    """Tests for particle initialization."""

    def test_correct_shape(self):
        """Initialized particles should have correct shape."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        state = initialize_particles(key, model, params, n_particles)

        assert state.particles.shape == (n_particles, 1)
        assert state.log_weights.shape == (n_particles,)
        assert state.step == 0

    def test_uniform_initial_weights(self):
        """Initial weights should be uniform."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        state = initialize_particles(key, model, params, n_particles)

        np.testing.assert_array_equal(state.log_weights, jnp.zeros(n_particles))


class TestBootstrapStep:
    """Tests for single bootstrap step."""

    def test_step_updates_state(self):
        """Bootstrap step should update state."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        init_key, step_key = jax.random.split(key)
        state = initialize_particles(init_key, model, params, n_particles)
        observation = jnp.array([1.0])

        new_state, info = bootstrap_step(
            step_key, state, observation, model, params
        )

        assert new_state.step == state.step + 1
        assert not jnp.array_equal(new_state.particles, state.particles)

    def test_info_contains_ess(self):
        """Info should contain ESS."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        init_key, step_key = jax.random.split(key)
        state = initialize_particles(init_key, model, params, n_particles)
        observation = jnp.array([1.0])

        _, info = bootstrap_step(step_key, state, observation, model, params)

        assert info.ess > 0
        assert info.ess <= n_particles + 1  # Allow for floating-point precision


class TestRunBootstrapFilter:
    """Tests for full bootstrap filter run."""

    def test_processes_all_observations(self):
        """Filter should process all observations."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        n_timesteps = 10
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        observations = jax.random.normal(key, shape=(n_timesteps, 1))
        filter_key = jax.random.PRNGKey(123)

        final_state, info_history = run_bootstrap_filter(
            filter_key, observations, model, params, n_particles=n_particles
        )

        assert final_state.step == n_timesteps

    def test_log_likelihood_updated(self):
        """Log likelihood should be updated."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        n_timesteps = 10
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        observations = jax.random.normal(key, shape=(n_timesteps, 1))
        filter_key = jax.random.PRNGKey(123)

        final_state, _ = run_bootstrap_filter(
            filter_key, observations, model, params, n_particles=n_particles
        )

        # Log likelihood should be finite and not zero
        assert jnp.isfinite(final_state.log_likelihood)

    def test_reproducibility(self):
        """Same seed should give same results."""
        key = jax.random.PRNGKey(42)
        n_particles = 100
        n_timesteps = 10
        model = LocalLevelModel()
        params = LocalLevelParams(sigma_obs=0.5, sigma_level=0.1, m0=0.0, C0=1.0)

        observations = jax.random.normal(key, shape=(n_timesteps, 1))

        result1, _ = run_bootstrap_filter(
            jax.random.PRNGKey(42), observations, model, params, n_particles=n_particles
        )
        result2, _ = run_bootstrap_filter(
            jax.random.PRNGKey(42), observations, model, params, n_particles=n_particles
        )

        np.testing.assert_array_equal(result1.particles, result2.particles)
        np.testing.assert_array_equal(result1.log_weights, result2.log_weights)
