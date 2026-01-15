"""
Command Line Interface parser for the parallel backtest tool.

This module handles parsing of command line arguments and validation
of user input for the parallel backtest tool.
"""

import argparse
import os
import json
from typing import List, Optional, Tuple

from .models import BacktestConfig


def get_cpu_count() -> int:
    """
    Get the number of logical CPU cores available.
    
    Returns:
        Number of logical CPU cores, minimum 1
    """
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def get_default_workers() -> int:
    """
    Get the default number of worker processes.
    
    Uses all logical CPU cores for maximum parallelization.
    
    Returns:
        Number of logical CPU cores
    """
    return get_cpu_count()


def parse_pairs(pairs_arg: Optional[List[str]]) -> List[str]:
    """
    Parse trading pairs from command line argument.
    
    Supports multiple formats:
    - Space-separated: --pairs BTC/USDT ETH/USDT
    - Comma-separated: --pairs "BTC/USDT,ETH/USDT"
    - Mixed: --pairs BTC/USDT "ETH/USDT,SOL/USDT"
    
    Args:
        pairs_arg: List of pair arguments from argparse
        
    Returns:
        Flattened list of unique trading pairs
    """
    if not pairs_arg:
        return []
    
    result = []
    for item in pairs_arg:
        # Handle comma-separated pairs within a single argument
        if ',' in item:
            parts = [p.strip() for p in item.split(',') if p.strip()]
            result.extend(parts)
        else:
            stripped = item.strip()
            if stripped:
                result.append(stripped)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for pair in result:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    
    return unique_pairs


def load_pairs_from_config(config_path: str) -> List[str]:
    """
    Load trading pairs from Freqtrade config file.
    
    Reads the pair_whitelist from the exchange section of the config.
    
    Args:
        config_path: Path to Freqtrade config JSON file
        
    Returns:
        List of trading pairs from config
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        KeyError: If pair_whitelist is not found in config
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Try to get pairs from exchange.pair_whitelist
    if 'exchange' in config and 'pair_whitelist' in config['exchange']:
        return config['exchange']['pair_whitelist']
    
    # Fallback to top-level pair_whitelist
    if 'pair_whitelist' in config:
        return config['pair_whitelist']
    
    raise KeyError("No pair_whitelist found in config file")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='parallel_backtest',
        description='Parallel backtest tool for Freqtrade - run backtests for multiple pairs concurrently',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with config file pairs
  python -m parallel_backtest --config configs/config.json --strategy MyStrategy

  # Specify trading pairs
  python -m parallel_backtest --config configs/config.json --strategy MyStrategy \\
      --pairs BTC/USDT ETH/USDT SOL/USDT

  # With time range and worker count
  python -m parallel_backtest --config configs/config.json --strategy MyStrategy \\
      --timerange 20240101-20241231 --workers 4

  # Pass extra arguments to Freqtrade
  python -m parallel_backtest --config configs/config.json --strategy MyStrategy \\
      -- --cache none --enable-protections
'''
    )
    
    # Required arguments
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to Freqtrade configuration file (required)'
    )
    
    parser.add_argument(
        '-s', '--strategy',
        type=str,
        required=True,
        help='Name of the trading strategy to backtest (required)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--timerange',
        type=str,
        default=None,
        help='Time range for backtest (e.g., 20240101-20241231)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help=f'Maximum number of concurrent workers (default: CPU cores - 1 = {get_default_workers()})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='user_data/backtest_results',
        help='Output directory for merged results (default: user_data/backtest_results)'
    )
    
    parser.add_argument(
        '-p', '--pairs',
        type=str,
        nargs='*',
        default=None,
        help='Trading pairs to backtest (overrides config file). '
             'Format: --pairs BTC/USDT ETH/USDT or --pairs "BTC/USDT,ETH/USDT"'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for each backtest task (default: no timeout)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Debug mode - preserve temporary files for inspection'
    )
    
    return parser


def validate_config_file(config_path: str) -> None:
    """
    Validate that the config file exists and is readable.
    
    Args:
        config_path: Path to config file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config path is not a file: {config_path}")
    
    # Try to parse as JSON to validate format
    with open(config_path, 'r', encoding='utf-8') as f:
        json.load(f)


def validate_workers(workers: int) -> None:
    """
    Validate worker count is positive.
    
    Args:
        workers: Number of workers
        
    Raises:
        ValueError: If workers is less than 1
    """
    if workers < 1:
        raise ValueError(f"Workers must be at least 1, got: {workers}")


def validate_timeout(timeout: Optional[int]) -> None:
    """
    Validate timeout is positive if specified.
    
    Args:
        timeout: Timeout in seconds (None for no timeout)
        
    Raises:
        ValueError: If timeout is less than 1
    """
    if timeout is not None and timeout < 1:
        raise ValueError(f"Timeout must be at least 1 second, got: {timeout}")


def parse_args(args: Optional[List[str]] = None) -> Tuple[BacktestConfig, List[str]]:
    """
    Parse command line arguments and return configuration.
    
    Handles the '--' separator for extra Freqtrade arguments.
    
    Args:
        args: Command line arguments (None for sys.argv)
        
    Returns:
        Tuple of (BacktestConfig, extra_args list)
        
    Raises:
        SystemExit: If required arguments are missing
        FileNotFoundError: If config file doesn't exist
        ValueError: If argument values are invalid
    """
    # Split args at '--' to separate our args from Freqtrade passthrough args
    extra_args = []
    our_args = args
    
    if args is not None and '--' in args:
        separator_idx = args.index('--')
        our_args = args[:separator_idx]
        extra_args = args[separator_idx + 1:]
    elif args is None:
        import sys
        if '--' in sys.argv:
            separator_idx = sys.argv.index('--')
            our_args = sys.argv[1:separator_idx]
            extra_args = sys.argv[separator_idx + 1:]
    
    parser = create_parser()
    parsed = parser.parse_args(our_args)
    
    # Validate config file
    validate_config_file(parsed.config)
    
    # Determine worker count
    workers = parsed.workers if parsed.workers is not None else get_default_workers()
    validate_workers(workers)
    
    # Validate timeout
    validate_timeout(parsed.timeout)
    
    # Determine pairs - CLI takes precedence over config file
    if parsed.pairs is not None:
        pairs = parse_pairs(parsed.pairs)
    else:
        pairs = load_pairs_from_config(parsed.config)
    
    if not pairs:
        raise ValueError("No trading pairs specified. Use --pairs or add pair_whitelist to config file.")
    
    config = BacktestConfig(
        config_path=parsed.config,
        strategy=parsed.strategy,
        pairs=pairs,
        timerange=parsed.timerange,
        max_workers=workers,
        output_dir=parsed.output,
        timeout=parsed.timeout,
        debug=parsed.debug,
        extra_args=extra_args
    )
    
    return config, extra_args


class CLIParser:
    """
    CLI Parser class for the parallel backtest tool.
    
    Provides an object-oriented interface for parsing command line arguments.
    """
    
    def __init__(self):
        """Initialize the CLI parser."""
        self._parser = create_parser()
    
    def parse_args(self, args: Optional[List[str]] = None) -> BacktestConfig:
        """
        Parse CLI arguments and return configuration.
        
        Args:
            args: Command line arguments (None for sys.argv)
            
        Returns:
            BacktestConfig with all parsed options
        """
        config, _ = parse_args(args)
        return config
