"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from smcs.config import SMCConfig, AgentConfig


class TestSMCConfig:
    """Tests for SMCConfig."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = SMCConfig()

        assert config.n_particles == 1000
        assert config.seed == 42
        assert config.ess_threshold == 0.5
        assert config.resampling_method == "systematic"

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = SMCConfig(
            n_particles=500,
            seed=123,
            ess_threshold=0.7,
        )

        assert config.n_particles == 500
        assert config.seed == 123
        assert config.ess_threshold == 0.7

    def test_immutable(self):
        """Config should be frozen."""
        config = SMCConfig()

        with pytest.raises(ValidationError):
            config.n_particles = 500

    def test_n_particles_validation(self):
        """n_particles must be > 10."""
        with pytest.raises(ValidationError):
            SMCConfig(n_particles=5)

    def test_ess_threshold_range(self):
        """ess_threshold must be in [0, 1]."""
        with pytest.raises(ValidationError):
            SMCConfig(ess_threshold=1.5)

        with pytest.raises(ValidationError):
            SMCConfig(ess_threshold=-0.1)

    def test_burnin_less_than_samples(self):
        """n_burnin must be less than n_mcmc_samples."""
        with pytest.raises(ValidationError):
            SMCConfig(n_mcmc_samples=1000, n_burnin=1500)


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = AgentConfig()

        assert config.model_type == "local_level"
        assert config.arima_order == (1, 0, 0)
        assert config.garch_order == (1, 1)

    def test_nested_smc_config(self):
        """Should have nested SMCConfig."""
        config = AgentConfig()

        assert isinstance(config.smc, SMCConfig)
        assert config.smc.n_particles == 1000

    def test_custom_nested_config(self):
        """Should accept custom nested config."""
        config = AgentConfig(
            smc=SMCConfig(n_particles=500),
            model_type="arima",
            arima_order=(2, 1, 1),
        )

        assert config.smc.n_particles == 500
        assert config.model_type == "arima"
        assert config.arima_order == (2, 1, 1)
