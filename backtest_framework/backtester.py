"""Core backtesting engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .strategy import Strategy


logger = logging.getLogger(__name__)


@dataclass
class Backtester:
    """Vectorized backtesting engine with simple position management."""

    initial_cash: float = 100_000.0
    fee_per_trade: float = 1.0
    slippage: float = 0.0005
    position_size_pct: float = 0.01

    def run(self, df: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """Run a backtest using ``strategy`` on ``df`` and return results."""
        if df.empty:
            raise ValueError("Input dataframe is empty. Cannot run backtest.")
        if "close" not in df.columns:
            raise ValueError("Input dataframe must contain a 'close' column.")

        enriched = strategy.generate_signals(df)
        if "signal" not in enriched.columns:
            raise ValueError("Strategy output missing required 'signal' column.")

        cash = self.initial_cash
        position = 0.0
        trade_log: List[Dict[str, Any]] = []
        equity_records: List[Dict[str, Any]] = []

        for timestamp, row in enriched.iterrows():
            price = float(row["close"])
            signal = float(row["signal"])

            cash, position, trade = self.execute_trade(signal, price, timestamp, cash, position)
            if trade is not None:
                trade_log.append(trade)

            equity = cash + position * price
            equity_records.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": cash,
                    "position": position,
                    "price": price,
                }
            )

        equity_df = pd.DataFrame(equity_records).set_index("timestamp")
        trades_df = pd.DataFrame(trade_log)

        final_equity = float(equity_df["equity"].iloc[-1])
        total_return = (final_equity / self.initial_cash) - 1.0

        final_metrics = {
            "final_equity": final_equity,
            "total_return": total_return,
            "cash": cash,
            "open_position": position,
            "num_trades": len(trades_df),
        }

        logger.info(
            "Backtest complete: final_equity=%s total_return=%.2f%% trades=%d",
            final_equity,
            total_return * 100,
            len(trades_df),
        )

        return {
            "equity_curve": equity_df,
            "trade_log": trades_df,
            "final_metrics": final_metrics,
            "strategy_output": enriched,
        }

    def execute_trade(
        self,
        signal: float,
        price: float,
        timestamp: pd.Timestamp,
        cash: float,
        position: float,
    ) -> Tuple[float, float, Dict[str, Any] | None]:
        """Execute trades based on the provided signal and return updated state."""
        trade_record: Dict[str, Any] | None = None

        if signal > 0 and position <= 0:
            # Enter long position sized by available cash.
            effective_price = price * (1 + self.slippage)
            budget = cash * self.position_size_pct
            if budget <= 0:
                return cash, position, None
            shares = int(budget // effective_price)
            if shares <= 0:
                return cash, position, None
            cost = shares * effective_price + self.fee_per_trade
            if cost > cash:
                return cash, position, None
            cash -= cost
            position += shares
            trade_record = {
                "timestamp": timestamp,
                "type": "BUY",
                "price": effective_price,
                "shares": shares,
                "cash_balance": cash,
            }
            logger.debug("Executed BUY of %s shares at %s", shares, effective_price)

        elif signal < 0 and position > 0:
            # Exit long position.
            effective_price = price * (1 - self.slippage)
            proceeds = position * effective_price - self.fee_per_trade
            cash += proceeds
            trade_record = {
                "timestamp": timestamp,
                "type": "SELL",
                "price": effective_price,
                "shares": position,
                "cash_balance": cash,
            }
            logger.debug("Executed SELL of %s shares at %s", position, effective_price)
            position = 0.0

        return cash, position, trade_record
