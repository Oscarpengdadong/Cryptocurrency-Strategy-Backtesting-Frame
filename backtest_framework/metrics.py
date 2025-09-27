"""Performance metrics for backtest results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PerformanceAnalyzer:
    """Compute common performance metrics for trading strategies."""

    periods_per_year: int = 252

    def calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """Compute percentage returns from an equity curve."""
        if "equity" not in equity_curve.columns:
            raise ValueError("Equity curve must contain an 'equity' column.")
        returns = equity_curve["equity"].pct_change().fillna(0)
        return returns

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate the annualised Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(self.periods_per_year)
        return float(sharpe)

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.DataFrame) -> float:
        """Calculate the maximum drawdown from an equity curve."""
        equity = equity_curve["equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def calculate_annualized_return(total_return: float, years: float) -> float:
        """Convert cumulative return into annualised return."""
        if years <= 0:
            return 0.0
        return float((1 + total_return) ** (1 / years) - 1)

    def generate_full_report(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive metrics dictionary."""
        equity_curve = backtest_results["equity_curve"]
        returns = self.calculate_returns(equity_curve)
        total_return = backtest_results["final_metrics"]["total_return"]
        years = max(len(equity_curve) / self.periods_per_year, 1e-9)
        annualized_return = self.calculate_annualized_return(total_return, years)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(equity_curve)

        report = {
            "final_equity": backtest_results["final_metrics"]["final_equity"],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": backtest_results["final_metrics"]["num_trades"],
        }
        logger.debug("Performance report generated: %s", report)
        return report

    @staticmethod
    def print_summary(metrics: Dict[str, Any]) -> None:
        """Pretty-print selected performance metrics to stdout."""
        lines = [
            f"Final Equity: {metrics['final_equity']:.2f}",
            f"Total Return: {metrics['total_return']:.2%}",
            f"Annualized Return: {metrics['annualized_return']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"Number of Trades: {metrics['num_trades']}",
        ]
        print("\n".join(lines))
