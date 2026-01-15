"""
Utility functions for the parallel backtest tool.

This module provides helper functions for file operations, logging,
time formatting, and other common tasks used throughout the tool.
"""

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from typing import Optional


def setup_logging(
    debug: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the parallel backtest tool.
    
    Args:
        debug: If True, set log level to DEBUG; otherwise INFO
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('parallel_backtest')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def cleanup_temp_files(temp_dir: str, force: bool = False) -> bool:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_dir: Path to temporary directory to remove
        force: If True, ignore errors during removal
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    if not temp_dir or not os.path.exists(temp_dir):
        return True
    
    try:
        shutil.rmtree(temp_dir, ignore_errors=force)
        return True
    except Exception as e:
        if not force:
            logging.getLogger('parallel_backtest').warning(
                f"Failed to cleanup temp directory {temp_dir}: {e}"
            )
        return False


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 23m 45s" or "45.2s")
    """
    if seconds < 0:
        return "0s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    # Add days if applicable
    if td.days > 0:
        hours += td.days * 24
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_timestamp(timestamp_ms: int) -> str:
    """
    Format millisecond timestamp to ISO format string.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        
    Returns:
        ISO format datetime string
    """
    if timestamp_ms <= 0:
        return ""
    
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError):
        return ""


def format_profit(profit_ratio: float, profit_abs: float, currency: str = "USDT") -> str:
    """
    Format profit values for display.
    
    Args:
        profit_ratio: Profit as ratio (e.g., 0.05 for 5%)
        profit_abs: Absolute profit value
        currency: Currency symbol
        
    Returns:
        Formatted profit string (e.g., "+5.00% (+50.00 USDT)")
    """
    sign = "+" if profit_ratio >= 0 else ""
    percent = profit_ratio * 100
    return f"{sign}{percent:.2f}% ({sign}{profit_abs:.2f} {currency})"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def calculate_speedup(
    parallel_time: float,
    sequential_time: float
) -> float:
    """
    Calculate speedup ratio compared to sequential execution.
    
    Args:
        parallel_time: Time taken for parallel execution
        sequential_time: Estimated time for sequential execution
        
    Returns:
        Speedup ratio (sequential_time / parallel_time)
    """
    if parallel_time <= 0:
        return 1.0
    
    return sequential_time / parallel_time


def estimate_sequential_time(durations: list) -> float:
    """
    Estimate total sequential execution time from individual durations.
    
    Args:
        durations: List of individual task durations in seconds
        
    Returns:
        Sum of all durations (estimated sequential time)
    """
    return sum(d for d in durations if d > 0)


def generate_output_filename(
    output_dir: str,
    strategy_name: str,
    extension: str = "json"
) -> str:
    """
    Generate a unique output filename with timestamp.
    
    Args:
        output_dir: Output directory path
        strategy_name: Strategy name for filename
        extension: File extension (default: "json")
        
    Returns:
        Full path to output file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"backtest-result-{timestamp}.{extension}"
    return os.path.join(output_dir, filename)


def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False


def get_terminal_width() -> int:
    """
    Get the terminal width for formatting output.
    
    Returns:
        Terminal width in characters (default: 80)
    """
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length with suffix.
    
    Args:
        s: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(suffix)] + suffix
