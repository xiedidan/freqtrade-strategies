"""
Time range splitting utilities for parallel backtest.

This module provides functionality to split a time range into smaller chunks,
enabling finer-grained parallelization when the number of trading pairs
is less than available CPU cores.
"""

from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass
class TimeChunk:
    """
    Represents a time chunk for parallel execution.
    
    Attributes:
        start: Start date string (YYYYMMDD format)
        end: End date string (YYYYMMDD format)
        chunk_id: Unique identifier for this chunk
        label: Human-readable label (e.g., "2024-01")
    """
    start: str
    end: str
    chunk_id: int
    label: str = ""
    
    @property
    def timerange(self) -> str:
        """Get Freqtrade-compatible timerange string."""
        return f"{self.start}-{self.end}"


def parse_timerange(timerange: str) -> Tuple[datetime, datetime]:
    """
    Parse a Freqtrade timerange string into datetime objects.
    
    Supports formats:
    - YYYYMMDD-YYYYMMDD (full range)
    - YYYYMMDD- (from date to now)
    - -YYYYMMDD (from beginning to date)
    
    Args:
        timerange: Freqtrade timerange string
        
    Returns:
        Tuple of (start_date, end_date)
        
    Raises:
        ValueError: If timerange format is invalid
    """
    if not timerange or '-' not in timerange:
        raise ValueError(f"Invalid timerange format: {timerange}")
    
    parts = timerange.split('-')
    
    if len(parts) != 2:
        raise ValueError(f"Invalid timerange format: {timerange}")
    
    start_str, end_str = parts
    
    # Parse start date
    if start_str:
        try:
            start_date = datetime.strptime(start_str, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid start date format: {start_str}")
    else:
        # Default to 1 year ago if no start date
        start_date = datetime.now() - timedelta(days=365)
    
    # Parse end date
    if end_str:
        try:
            end_date = datetime.strptime(end_str, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid end date format: {end_str}")
    else:
        # Default to today if no end date
        end_date = datetime.now()
    
    if start_date >= end_date:
        raise ValueError(f"Start date must be before end date: {timerange}")
    
    return start_date, end_date


def split_by_month(timerange: str) -> List[TimeChunk]:
    """
    Split a time range by calendar months.
    
    Each chunk represents one calendar month (or partial month at boundaries).
    
    Args:
        timerange: Freqtrade timerange string (YYYYMMDD-YYYYMMDD)
        
    Returns:
        List of TimeChunk objects, one per month
        
    Raises:
        ValueError: If timerange is invalid
    """
    start_date, end_date = parse_timerange(timerange)
    
    chunks = []
    chunk_id = 0
    current = start_date
    
    while current < end_date:
        # Get the last day of current month
        _, last_day = monthrange(current.year, current.month)
        month_end = datetime(current.year, current.month, last_day)
        
        # Chunk end is either end of month or end_date, whichever is earlier
        chunk_end = min(month_end, end_date)
        
        # Create chunk
        label = current.strftime("%Y-%m")
        chunks.append(TimeChunk(
            start=current.strftime("%Y%m%d"),
            end=chunk_end.strftime("%Y%m%d"),
            chunk_id=chunk_id,
            label=label
        ))
        
        chunk_id += 1
        
        # Move to first day of next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return chunks


def calculate_optimal_chunks(
    num_pairs: int,
    num_cpus: int,
    min_chunk_days: int = 30
) -> int:
    """
    Calculate optimal number of time chunks based on available resources.
    
    The goal is to create enough tasks to fully utilize all CPU cores.
    
    Args:
        num_pairs: Number of trading pairs
        num_cpus: Number of available CPU cores
        min_chunk_days: Minimum days per chunk (to avoid too small chunks)
        
    Returns:
        Optimal number of chunks per pair
    """
    if num_pairs >= num_cpus:
        # Already have enough parallelism from pairs alone
        return 1
    
    # Calculate how many chunks needed to fill all CPUs
    # We want: num_pairs * num_chunks >= num_cpus
    optimal_chunks = (num_cpus + num_pairs - 1) // num_pairs
    
    # Cap at reasonable maximum to avoid too many small tasks
    max_chunks = 12  # Maximum 12 chunks (e.g., monthly for a year)
    
    return min(optimal_chunks, max_chunks)


def split_timerange(
    timerange: str,
    num_chunks: int
) -> List[TimeChunk]:
    """
    Split a time range into equal-sized chunks.
    
    Args:
        timerange: Freqtrade timerange string (YYYYMMDD-YYYYMMDD)
        num_chunks: Number of chunks to create
        
    Returns:
        List of TimeChunk objects
        
    Raises:
        ValueError: If timerange is invalid or num_chunks < 1
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")
    
    if num_chunks == 1:
        # No splitting needed
        start_date, end_date = parse_timerange(timerange)
        return [TimeChunk(
            start=start_date.strftime("%Y%m%d"),
            end=end_date.strftime("%Y%m%d"),
            chunk_id=0
        )]
    
    start_date, end_date = parse_timerange(timerange)
    total_days = (end_date - start_date).days
    
    if total_days < num_chunks:
        # Not enough days to split into requested chunks
        # Fall back to one chunk per day or fewer chunks
        num_chunks = max(1, total_days)
    
    chunks = []
    days_per_chunk = total_days / num_chunks
    
    for i in range(num_chunks):
        chunk_start = start_date + timedelta(days=int(i * days_per_chunk))
        
        if i == num_chunks - 1:
            # Last chunk goes to the end
            chunk_end = end_date
        else:
            chunk_end = start_date + timedelta(days=int((i + 1) * days_per_chunk))
        
        chunks.append(TimeChunk(
            start=chunk_start.strftime("%Y%m%d"),
            end=chunk_end.strftime("%Y%m%d"),
            chunk_id=i
        ))
    
    return chunks


def should_split_timerange(
    timerange: Optional[str],
    num_pairs: int,
    num_cpus: int
) -> bool:
    """
    Determine if time splitting would be beneficial.
    
    Time splitting is beneficial when:
    1. We have fewer pairs than CPUs (obvious case)
    2. Pairs don't divide evenly into CPUs (e.g., 5 pairs on 4 CPUs)
    
    Args:
        timerange: Freqtrade timerange string (can be None)
        num_pairs: Number of trading pairs
        num_cpus: Number of available CPU cores
        
    Returns:
        True if splitting would improve CPU utilization
    """
    if timerange is None:
        return False
    
    if num_cpus <= 1:
        return False
    
    # Always split if pairs < CPUs
    if num_pairs < num_cpus:
        return True
    
    # Also split if pairs don't divide evenly (causes idle CPUs at the end)
    # e.g., 5 pairs on 4 CPUs: first batch runs 4, second batch runs 1 (3 idle)
    # With splitting: 60 tasks on 4 CPUs distributes much more evenly
    if num_pairs % num_cpus != 0:
        return True
    
    return False


def split_timerange_monthly(
    timerange: Optional[str],
    num_pairs: int,
    num_cpus: int
) -> Tuple[List[TimeChunk], bool]:
    """
    Split timerange by month if beneficial for parallelization.
    
    This is the main entry point for monthly time splitting.
    
    Args:
        timerange: Freqtrade timerange string (can be None)
        num_pairs: Number of trading pairs
        num_cpus: Number of available CPU cores
        
    Returns:
        Tuple of (list of TimeChunk, whether splitting was applied)
    """
    if not should_split_timerange(timerange, num_pairs, num_cpus):
        # No splitting needed
        if timerange:
            start_date, end_date = parse_timerange(timerange)
            return [TimeChunk(
                start=start_date.strftime("%Y%m%d"),
                end=end_date.strftime("%Y%m%d"),
                chunk_id=0,
                label="full"
            )], False
        return [], False
    
    try:
        chunks = split_by_month(timerange)
        # Only consider it "split" if we got more than one chunk
        return chunks, len(chunks) > 1
    except ValueError:
        return [], False
