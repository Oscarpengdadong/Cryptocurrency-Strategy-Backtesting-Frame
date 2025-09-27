"""Trading strategy definitions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from .indicators import calculate_sma, generate_crossover_signals


logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    name: str

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with at least a ``signal`` column."""


@dataclass
class SMACrossoverStrategy(Strategy):
    """Simple moving average crossover strategy."""

    fast_period: int
    slow_period: int
    name: str = "sma_crossover"

    def __post_init__(self) -> None:
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise ValueError("Moving average periods must be positive integers.")
        if self.fast_period >= self.slow_period:
            logger.warning(
                "Fast period %s is not smaller than slow period %s. Strategy may behave unexpectedly.",
                self.fast_period,
                self.slow_period,
            )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("Input dataframe must contain a 'close' column.")

        working_df = df.copy()
        sma_fast = calculate_sma(working_df["close"], self.fast_period)
        sma_slow = calculate_sma(working_df["close"], self.slow_period)

        working_df = pd.concat([working_df, sma_fast, sma_slow], axis=1)

        crossover = generate_crossover_signals(
            working_df[f"sma_{self.fast_period}"],
            working_df[f"sma_{self.slow_period}"],
        )
        working_df = pd.concat([working_df, crossover], axis=1)
        working_df["signal"] = working_df["signal"].fillna(0)
        logger.debug("Generated signals for %d rows", len(working_df))
        return working_df
