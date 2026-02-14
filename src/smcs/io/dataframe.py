"""DataFrame input/output utilities.

This module provides functions for converting between pandas DataFrames
and JAX arrays, preserving timestamp information.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from loguru import logger

from smcs.agents.base import ForecastResult

__all__ = [
    "from_dataframe",
    "to_dataframe",
    "forecast_to_dataframe",
]


def from_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    dropna: bool = True,
) -> tuple[Float[Array, "n_timesteps n_features"], pd.DatetimeIndex | None]:
    """Convert DataFrame to JAX array with timestamps.

    Parameters
    ----------
    df : DataFrame
        Input data.
    columns : list of str, optional
        Columns to use (default: all numeric columns).
    dropna : bool
        Whether to drop rows with missing values.

    Returns
    -------
    data : Array
        JAX array [n_timesteps, n_features].
    timestamps : DatetimeIndex or None
        Timestamps if index is DatetimeIndex.

    Examples
    --------
    >>> df = pd.DataFrame({"y": [1.0, 2.0, 3.0]},
    ...                   index=pd.date_range("2024-01-01", periods=3))
    >>> data, timestamps = from_dataframe(df)
    >>> data.shape
    (3, 1)
    """
    # Extract timestamps
    timestamps = None
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.copy()
        logger.debug(f"Extracted timestamps: {timestamps[0]} to {timestamps[-1]}")

    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.debug(f"Auto-selected numeric columns: {columns}")

    if not columns:
        raise ValueError("No numeric columns found in DataFrame")

    data = df[columns].copy()

    # Handle missing values
    if dropna:
        mask = ~data.isna().any(axis=1)
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} rows with missing values")
            data = data[mask]
            if timestamps is not None:
                timestamps = timestamps[mask]

    return jnp.array(data.values), timestamps


def to_dataframe(
    data: Float[Array, "n_timesteps n_features"],
    timestamps: pd.DatetimeIndex | None = None,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert JAX array to DataFrame.

    Parameters
    ----------
    data : Array
        Data array [n_timesteps, n_features].
    timestamps : DatetimeIndex, optional
        Index for DataFrame.
    column_names : list of str, optional
        Column names.

    Returns
    -------
    df : DataFrame
        Converted DataFrame.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_timesteps, n_features = data.shape

    if column_names is None:
        column_names = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=column_names)

    if timestamps is not None:
        df.index = timestamps

    return df


def forecast_to_dataframe(
    result: ForecastResult,
    column_names: list[str] | None = None,
    include_particles: bool = False,
) -> pd.DataFrame:
    """Convert ForecastResult to DataFrame.

    Parameters
    ----------
    result : ForecastResult
        Forecast result from agent.
    column_names : list of str, optional
        Base column names.
    include_particles : bool
        Whether to include particle samples.

    Returns
    -------
    df : DataFrame
        DataFrame with forecast statistics.

    Examples
    --------
    >>> df = forecast_to_dataframe(result)
    >>> df.columns
    Index(['y_mean', 'y_std', 'y_q05', 'y_q25', 'y_q50', 'y_q75', 'y_q95'], ...)
    """
    mean = np.asarray(result.mean)
    std = np.asarray(result.std)

    if mean.ndim == 1:
        mean = mean[:, np.newaxis]
        std = std[:, np.newaxis]

    horizon, n_features = mean.shape

    if column_names is None:
        column_names = ["y" if n_features == 1 else f"y_{i}" for i in range(n_features)]

    data = {}

    # Mean and std
    for i, col in enumerate(column_names):
        data[f"{col}_mean"] = mean[:, i]
        data[f"{col}_std"] = std[:, i]

    # Quantiles
    for q, values in result.quantiles.items():
        q_array = np.asarray(values)
        if q_array.ndim == 1:
            q_array = q_array[:, np.newaxis]
        for i, col in enumerate(column_names):
            data[f"{col}_q{int(q*100):02d}"] = q_array[:, i]

    df = pd.DataFrame(data)

    if result.timestamps is not None:
        df.index = result.timestamps

    return df
