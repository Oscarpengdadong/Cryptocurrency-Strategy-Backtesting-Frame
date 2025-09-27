"""Application-wide configuration settings for the backtesting framework."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

# Default database configuration. Update the path if your DuckDB file lives elsewhere.
DATABASE_CONFIG: Dict[str, Any] = {
    "path": "/Users/xiaohan/Working/trading/crypto/systematic/data/crypto_data_aggregated/binance_merged.db",
    "default_symbol": "ETHUSDT",
    "default_timeframe": "1D",
    "default_table": "main.ohlcv_1m",
}

# Default backtest parameters used by the Backtester class.
BACKTEST_CONFIG: Dict[str, float] = {
    "initial_cash": 100_000.0,
    "fee_per_trade": 1.0,
    "slippage": 0.0005,
    "position_size_pct": 0.01,
}

# Default strategy parameters keyed by strategy identifier.
STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {
    "sma_crossover": {
        "fast_period": 20,
        "slow_period": 50,
    }
}

# Default directory for saving generated figures and reports.
DEFAULT_OUTPUT_DIR: Path = Path("kline_backtest_outputs")
