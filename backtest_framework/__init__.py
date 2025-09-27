"""Modular backtesting framework for cryptocurrency and equities."""

from .data_loader import DataLoader
from .indicators import calculate_sma, calculate_ema, generate_crossover_signals
from .strategy import Strategy, SMACrossoverStrategy
from .backtester import Backtester
from .metrics import PerformanceAnalyzer
from .visualization import Visualizer
from . import config

__all__ = [
    "DataLoader",
    "calculate_sma",
    "calculate_ema",
    "generate_crossover_signals",
    "Strategy",
    "SMACrossoverStrategy",
    "Backtester",
    "PerformanceAnalyzer",
    "Visualizer",
    "config",
]
