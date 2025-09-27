"""Technical indicator helpers."""

from __future__ import annotations

from typing import Union

import pandas as pd


def calculate_sma(series: pd.Series, period: int) -> pd.DataFrame:
    """Return a DataFrame containing the simple moving average for ``series``."""
    if period <= 0:
        raise ValueError("SMA period must be a positive integer.")
    sma = series.rolling(window=period, min_periods=period).mean()
    column_name = f"sma_{period}"
    return pd.DataFrame({column_name: sma}, index=series.index)


def calculate_ema(series: pd.Series, period: int) -> pd.DataFrame:
    """Return a DataFrame containing the exponential moving average for ``series``."""
    if period <= 0:
        raise ValueError("EMA period must be a positive integer.")
    ema = series.ewm(span=period, adjust=False, min_periods=period).mean()
    column_name = f"ema_{period}"
    return pd.DataFrame({column_name: ema}, index=series.index)


def generate_crossover_signals(
    fast_ma: Union[pd.Series, pd.DataFrame],
    slow_ma: Union[pd.Series, pd.DataFrame],
) -> pd.DataFrame:
    """Generate crossover signals from fast and slow moving averages."""
    fast_series = fast_ma.squeeze()
    slow_series = slow_ma.squeeze()

    common_index = fast_series.index.union(slow_series.index)
    fast_series = fast_series.reindex(common_index).ffill()
    slow_series = slow_series.reindex(common_index).ffill()

    signal_raw = (fast_series > slow_series).astype(int)
    signal = signal_raw.diff().fillna(0).clip(lower=-1, upper=1)

    return pd.DataFrame({
        "signal_raw": signal_raw,
        "signal": signal,
    }, index=common_index)
