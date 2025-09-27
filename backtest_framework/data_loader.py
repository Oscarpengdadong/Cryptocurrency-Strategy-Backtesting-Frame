"""Data loading utilities for the backtesting framework."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict

import duckdb
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DataLoader:
    """Load OHLCV data from multiple sources for backtesting.

    The loader supports DuckDB databases, CSV files, and synthetic data generation.
    All returned DataFrames are indexed by a DatetimeIndex and sorted ascending.
    """

    db_path: Optional[Union[str, Path]] = None
    connection: Optional[duckdb.DuckDBPyConnection] = None

    def connect_to_db(self, db_path: Optional[Union[str, Path]] = None) -> duckdb.DuckDBPyConnection:
        """Connect to a DuckDB database located at ``db_path``.

        Parameters
        ----------
        db_path: Optional[Union[str, Path]]
            Path to the DuckDB database. If omitted, ``self.db_path`` is used.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active database connection that can be reused for subsequent queries.

        Raises
        ------
        ValueError
            If no database path is supplied.
        duckdb.Error
            If the connection attempt fails.
        """
        path = Path(db_path or self.db_path or "")
        if not path:
            raise ValueError("Database path must be provided for DuckDB connection.")

        try:
            self.connection = duckdb.connect(str(path))
            logger.info("Connected to DuckDB database at %s", path)
        except duckdb.Error as exc:  # pragma: no cover - difficult to trigger in tests
            logger.exception("Failed to connect to DuckDB database: %s", exc)
            raise
        self.db_path = path
        return self.connection

    def fetch_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        source: str = "db",
        csv_path: Optional[Union[str, Path]] = None,
        table: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data and resample to the requested timeframe.

        Parameters
        ----------
        symbol: str
            Market symbol (e.g., ``"ETHUSDT"``).
        timeframe: str
            Resampling rule understood by ``pandas.DataFrame.resample`` (e.g., ``"1D"``).
        start_date, end_date: Optional[Union[str, datetime]]
            Inclusive date filters. Accepts ISO date strings or ``datetime`` objects.
        source: str
            Data source identifier: ``"db"``, ``"csv"``, or ``"synthetic"``.
        csv_path: Optional[Union[str, Path]]
            Path to a CSV file containing OHLCV data (required when ``source="csv"``).
        table: Optional[str]
            Database table name. Defaults to ``main.ohlcv_1m`` when unspecified.

        Returns
        -------
        pandas.DataFrame
            Cleaned OHLCV data indexed by timestamp.
        """
        logger.info(
            "Fetching OHLCV data for symbol=%s timeframe=%s source=%s", symbol, timeframe, source
        )

        if source == "synthetic":
            return self.create_synthetic_data(symbol=symbol, timeframe=timeframe)

        if source == "csv":
            if not csv_path:
                raise ValueError("csv_path must be provided when source='csv'.")
            return self._load_from_csv(csv_path, timeframe, start_date, end_date)

        # Default to database source
        if not self.connection:
            self.connect_to_db()
        assert self.connection is not None  # for type checkers

        table_name = table or "main.ohlcv_1m"
        filters = ["symbol = ?"]
        params = [symbol]

        # Normalize date inputs into pandas Timestamps for logging and filtering.
        start_ts = pd.to_datetime(start_date) if start_date is not None else None
        end_ts = pd.to_datetime(end_date) if end_date is not None else None

        if start_ts is not None:
            filters.append("open_standard >= ?")
            params.append(start_ts)
        if end_ts is not None:
            filters.append("open_standard <= ?")
            params.append(end_ts)

        where_clause = " AND ".join(filters)
        query = f"SELECT * FROM {table_name} WHERE {where_clause} ORDER BY open_standard"
        logger.debug("Executing DuckDB query: %s", query)

        try:
            raw_df = self.connection.execute(query, params).fetchdf()
        except duckdb.Error as exc:  # pragma: no cover - depends on external DB
            logger.exception("DuckDB query failed: %s", exc)
            raise

        if raw_df.empty:
            raise ValueError(f"No data returned for {symbol} between {start_ts} and {end_ts}.")

        return self._prepare_dataframe(raw_df, timeframe)

    def create_synthetic_data(
        self,
        n_days: int = 1000,
        start_price: float = 100.0,
        symbol: str = "SYNTH",
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing or demonstration purposes."""
        logger.info(
            "Creating synthetic OHLCV data: n_days=%s start_price=%s symbol=%s", n_days, start_price, symbol
        )
        np.random.seed(42)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        drift = 0.0002
        volatility = 0.02
        returns = np.random.normal(loc=drift, scale=volatility, size=n_days)
        price = start_price * np.exp(np.cumsum(returns))

        highs = price * (1 + np.abs(np.random.normal(0, 0.003, size=n_days)))
        lows = price * (1 - np.abs(np.random.normal(0, 0.003, size=n_days)))
        opens = price * (1 + np.random.normal(0, 0.001, size=n_days))
        closes = price
        volume = np.random.randint(100, 1_000, size=n_days)

        df = pd.DataFrame(
            {
                "symbol": symbol,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "timestamp"
        resampled = self._resample(df, timeframe)
        logger.debug("Synthetic data generated with %d rows", len(resampled))
        return resampled

    def _load_from_csv(
        self,
        csv_path: Union[str, Path],
        timeframe: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
    ) -> pd.DataFrame:
        """Load OHLCV data from a CSV file and resample to the desired timeframe."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info("Loading CSV data from %s", path)
        df = pd.read_csv(path)

        timestamp_col = self._resolve_timestamp_column(df)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)

        if start_date is not None:
            df = df[df[timestamp_col] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df[timestamp_col] <= pd.to_datetime(end_date)]

        df = df.set_index(timestamp_col)
        resampled = self._resample(df, timeframe)
        logger.debug("Loaded %d rows from CSV after resampling", len(resampled))
        return resampled

    def _prepare_dataframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Clean DuckDB data and resample to the requested timeframe."""
        timestamp_col = self._resolve_timestamp_column(df)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)

        # Drop known auxiliary columns if they exist.
        extra_columns = {"open_time", "close_time"}
        drop_cols = [col for col in extra_columns if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        df = df.set_index(timestamp_col)
        resampled = self._resample(df, timeframe)
        logger.debug("Prepared DuckDB dataframe with %d rows", len(resampled))
        return resampled

    @staticmethod
    def _resolve_timestamp_column(df: pd.DataFrame) -> str:
        """Infer the timestamp column name from common alternatives."""
        for candidate in ("timestamp", "open_standard", "datetime", "date"):
            if candidate in df.columns:
                return candidate
        raise ValueError("Unable to locate a timestamp column in the provided data.")

    @staticmethod
    def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample an OHLCV dataframe to a different timeframe."""
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required OHLC columns: {missing}")

        agg_map: Dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else "mean",
        }

        resampled = df.resample(timeframe).agg(agg_map)
        resampled = resampled.dropna(how="any")
        resampled.index.name = "timestamp"
        return resampled
