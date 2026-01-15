"""
Parallel Backtest Tool for Freqtrade.

This tool enables parallel execution of backtests across multiple trading pairs,
significantly reducing total backtest time by utilizing multiple CPU cores.

Features:
- Parallel execution by trading pairs
- Automatic time-based splitting when pairs < CPU cores
- Monthly granularity for time splitting
"""

__version__ = "0.2.0"
__author__ = "Freqtrade User"

from parallel_backtest.models import (
    BacktestConfig,
    WorkerConfig,
    WorkerResult,
    MergedResult,
)
from parallel_backtest.timerange import (
    TimeChunk,
    split_by_month,
    split_timerange_monthly,
)

__all__ = [
    "BacktestConfig",
    "WorkerConfig",
    "WorkerResult",
    "MergedResult",
    "TimeChunk",
    "split_by_month",
    "split_timerange_monthly",
]
