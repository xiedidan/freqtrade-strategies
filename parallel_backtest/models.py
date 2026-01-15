"""
Data models for the parallel backtest tool.

This module defines all dataclasses used throughout the parallel backtest tool,
including configuration, worker state, and result structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BacktestConfig:
    """
    Configuration for the parallel backtest execution.
    
    This dataclass holds all configuration options parsed from CLI arguments
    and used to coordinate the parallel backtest process.
    
    Attributes:
        config_path: Path to the base Freqtrade configuration file
        strategy: Name of the trading strategy to backtest
        pairs: List of trading pairs to backtest (e.g., ["BTC/USDT", "ETH/USDT"])
        timerange: Optional time range for backtest (e.g., "20240101-20241231")
        max_workers: Maximum number of concurrent worker processes
        output_dir: Directory path for output files
        timeout: Timeout in seconds for each individual backtest task (None for no timeout)
        debug: Debug mode flag - when True, temporary files are preserved
        extra_args: Additional arguments to pass through to Freqtrade CLI
    """
    config_path: str
    strategy: str
    pairs: List[str]
    timerange: Optional[str] = None
    max_workers: int = 1
    output_dir: str = "user_data/backtest_results"
    timeout: Optional[int] = None
    debug: bool = False
    extra_args: List[str] = field(default_factory=list)


@dataclass
class WorkerConfig:
    """
    Configuration for an individual backtest worker.
    
    Each worker gets isolated resources to prevent conflicts during
    parallel execution. This includes separate config files, output
    directories, and log files.
    
    Attributes:
        pair: Trading pair assigned to this worker (e.g., "BTC/USDT")
        config_path: Path to the worker's isolated temporary config file
        result_dir: Path to the worker's isolated result directory
        log_file: Path to the worker's isolated log file
        worker_id: Unique identifier for this worker
        time_chunk: Optional time chunk label (e.g., "2024-01") for split tasks
    """
    pair: str
    config_path: str
    result_dir: str
    log_file: str
    worker_id: int
    time_chunk: Optional[str] = None


@dataclass
class WorkerResult:
    """
    Result from an individual backtest worker execution.
    
    Contains the outcome of a single pair backtest, including success status,
    result file location, and summary statistics.
    
    Attributes:
        pair: Trading pair that was backtested
        success: Whether the backtest completed successfully
        result_file: Path to the result JSON file (None if failed)
        error_message: Error description if backtest failed (None if success)
        duration: Execution time in seconds
        trades_count: Number of trades executed during backtest
        profit_ratio: Total profit ratio from the backtest
        time_chunk: Optional time chunk label for split tasks
    """
    pair: str
    success: bool
    result_file: Optional[str] = None
    error_message: Optional[str] = None
    duration: float = 0.0
    trades_count: int = 0
    profit_ratio: float = 0.0
    time_chunk: Optional[str] = None
    
    @property
    def task_key(self) -> str:
        """Get unique task identifier (pair + time_chunk if split)."""
        if self.time_chunk:
            return f"{self.pair}:{self.time_chunk}"
        return self.pair


@dataclass
class MergedResult:
    """
    Summary of merged backtest results.
    
    Contains aggregate statistics from merging multiple individual
    backtest results into a unified result file.
    
    Attributes:
        output_file: Path to the merged result JSON file
        total_pairs: Total number of trading pairs processed
        successful_pairs: Number of pairs that completed successfully
        failed_pairs: Number of pairs that failed
        total_trades: Total number of trades across all pairs
        total_profit_abs: Absolute total profit across all pairs
        total_profit_ratio: Overall profit ratio
        winrate: Win rate across all trades (wins / total_trades)
        max_drawdown: Maximum drawdown observed
        execution_time: Total execution time in seconds
        speedup_ratio: Speedup compared to sequential execution
        failed_pair_names: List of pair names that failed
        results_per_pair: Dictionary mapping pair names to their statistics
    """
    output_file: str
    total_pairs: int
    successful_pairs: int
    failed_pairs: int
    total_trades: int = 0
    total_profit_abs: float = 0.0
    total_profit_ratio: float = 0.0
    winrate: float = 0.0
    max_drawdown: float = 0.0
    execution_time: float = 0.0
    speedup_ratio: float = 1.0
    failed_pair_names: List[str] = field(default_factory=list)
    results_per_pair: Dict[str, Dict[str, Any]] = field(default_factory=dict)
