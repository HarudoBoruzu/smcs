"""Forecasting agents for time series prediction.

This module provides high-level agents that wrap SMC algorithms
for easy-to-use time series forecasting.
"""

from smcs.agents.arima_agent import ARIMAAgent
from smcs.agents.base import BaseAgent, ForecastingAgent, ForecastResult
from smcs.agents.dlm_agent import LocalLevelAgent, LocalLinearTrendAgent
from smcs.agents.garch_agent import GARCHAgent
from smcs.agents.sv_agent import SVAgent

__all__ = [
    # Base
    "ForecastResult",
    "ForecastingAgent",
    "BaseAgent",
    # DLM agents
    "LocalLevelAgent",
    "LocalLinearTrendAgent",
    # ARIMA
    "ARIMAAgent",
    # Volatility models
    "GARCHAgent",
    "SVAgent",
]
