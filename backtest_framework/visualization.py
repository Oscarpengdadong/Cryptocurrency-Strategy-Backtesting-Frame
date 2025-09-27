"""Visualization utilities for backtest results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-darkgrid")


class Visualizer:
    """Create plots for prices, signals, equity curves, and drawdowns."""

    def plot_price_with_signals(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot closing prices and overlay buy/sell markers."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df["close"], label="Close Price", color="steelblue")

        if not trades_df.empty:
            buys = trades_df[trades_df["type"] == "BUY"]
            sells = trades_df[trades_df["type"] == "SELL"]
            ax.scatter(buys["timestamp"], buys["price"], marker="^", color="green", s=80, label="Buy")
            ax.scatter(sells["timestamp"], sells["price"], marker="v", color="red", s=80, label="Sell")

        ax.set_title("Price with Trade Signals")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Saved price plot to %s", save_path)
        return fig

    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot the equity curve over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_df.index, equity_df["equity"], label="Equity", color="purple")
        ax.set_title("Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Saved equity curve to %s", save_path)
        return fig

    def plot_drawdown(
        self,
        equity_df: pd.DataFrame,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot the drawdown curve derived from the equity curve."""
        equity = equity_df["equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown.index, drawdown, color="salmon", step="mid")
        ax.set_title("Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            logger.info("Saved drawdown plot to %s", save_path)
        return fig

    def create_dashboard(self, all_data: Dict[str, Any], save_dir: Path | str = "outputs/") -> None:
        """Generate all plots and optionally save them to ``save_dir``."""
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        price_df = all_data.get("price_df")
        trades_df = all_data.get("trades_df", pd.DataFrame())
        equity_df = all_data.get("equity_df")

        if price_df is None or equity_df is None:
            raise ValueError("price_df and equity_df must be provided in all_data.")

        figures = [
            self.plot_price_with_signals(price_df, trades_df, save_dir_path / "price_with_signals.png"),
            self.plot_equity_curve(equity_df, save_dir_path / "equity_curve.png"),
            self.plot_drawdown(equity_df, save_dir_path / "drawdown.png"),
        ]

        for fig in figures:
            plt.close(fig)
