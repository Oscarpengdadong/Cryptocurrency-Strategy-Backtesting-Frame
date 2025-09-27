"""Command line runner for the modular backtesting framework."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import config
from .backtester import Backtester
from .data_loader import DataLoader
from .metrics import PerformanceAnalyzer
from .strategy import SMACrossoverStrategy, Strategy
from .visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a modular SMA crossover backtest.")
    parser.add_argument("--symbol", default=config.DATABASE_CONFIG["default_symbol"], help="Trading symbol")
    parser.add_argument(
        "--timeframe", default=config.DATABASE_CONFIG["default_timeframe"], help="Resample timeframe (e.g., 1D)"
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        default="sma_crossover",
        choices=list(config.STRATEGY_PARAMS.keys()),
        help="Strategy identifier",
    )
    parser.add_argument(
        "--data-source",
        choices=["db", "csv", "synthetic"],
        default="db",
        help="Data source to use",
    )
    parser.add_argument("--csv-path", help="Path to CSV file when using data-source=csv")
    parser.add_argument("--synthetic-days", type=int, default=1000, help="Number of days for synthetic data")
    parser.add_argument(
        "--synthetic-start-price", type=float, default=100.0, help="Starting price for synthetic data"
    )
    parser.add_argument("--initial-cash", type=float, default=config.BACKTEST_CONFIG["initial_cash"], help="Initial cash")
    parser.add_argument(
        "--fee-per-trade", type=float, default=config.BACKTEST_CONFIG["fee_per_trade"], help="Flat fee per trade"
    )
    parser.add_argument(
        "--slippage", type=float, default=config.BACKTEST_CONFIG["slippage"], help="Fractional slippage per trade"
    )
    parser.add_argument(
        "--position-size-pct",
        type=float,
        default=config.BACKTEST_CONFIG["position_size_pct"],
        help="Fraction of cash allocated per trade",
    )
    parser.add_argument("--fast-period", type=int, help="Override fast MA period for SMA crossover")
    parser.add_argument("--slow-period", type=int, help="Override slow MA period for SMA crossover")
    parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def select_strategy(args: argparse.Namespace) -> Strategy:
    params = config.STRATEGY_PARAMS.get(args.strategy, {}).copy()
    if args.fast_period is not None:
        params["fast_period"] = args.fast_period
    if args.slow_period is not None:
        params["slow_period"] = args.slow_period

    if args.strategy == "sma_crossover":
        return SMACrossoverStrategy(
            fast_period=int(params.get("fast_period", 20)),
            slow_period=int(params.get("slow_period", 50)),
        )

    raise NotImplementedError(f"Strategy '{args.strategy}' is not implemented.")


def load_data(args: argparse.Namespace, loader: DataLoader) -> pd.DataFrame:
    if args.data_source == "synthetic":
        return loader.create_synthetic_data(
            n_days=args.synthetic_days,
            start_price=args.synthetic_start_price,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

    if args.data_source == "csv":
        if not args.csv_path:
            raise ValueError("--csv-path is required when data-source=csv")
        return loader.fetch_ohlcv_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            source="csv",
            csv_path=args.csv_path,
        )

    # Default: database
    loader.connect_to_db(config.DATABASE_CONFIG["path"])
    return loader.fetch_ohlcv_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        source="db",
        table=config.DATABASE_CONFIG.get("default_table"),
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    start_time = time.time()

    loader = DataLoader(db_path=config.DATABASE_CONFIG.get("path"))
    price_df = load_data(args, loader)

    strategy = select_strategy(args)
    backtester = Backtester(
        initial_cash=args.initial_cash,
        fee_per_trade=args.fee_per_trade,
        slippage=args.slippage,
        position_size_pct=args.position_size_pct,
    )

    backtest_results = backtester.run(price_df, strategy)

    analyzer = PerformanceAnalyzer()
    metrics = analyzer.generate_full_report(backtest_results)
    analyzer.print_summary(metrics)

    visualizer = Visualizer()
    output_dir = Path(args.output_dir)
    visualizer.create_dashboard(
        {
            "price_df": backtest_results["strategy_output"],
            "trades_df": backtest_results["trade_log"],
            "equity_df": backtest_results["equity_curve"],
        },
        save_dir=output_dir,
    )

    elapsed = time.time() - start_time
    logging.getLogger(__name__).info("Backtest completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
