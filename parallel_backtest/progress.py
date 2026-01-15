"""
Progress monitoring for parallel backtest workers.

This module provides real-time progress monitoring by watching
worker log files and parsing Freqtrade output.
"""

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime


@dataclass
class WorkerStatus:
    """
    Status of a single backtest worker.
    
    Attributes:
        pair: Trading pair being processed
        status: Current status (starting, loading, calculating, backtesting, saving, done, failed)
        progress_pct: Progress percentage (0-100)
        message: Current status message
        start_time: When the worker started
        elapsed_seconds: Elapsed time in seconds
        log_file: Path to the worker's log file
    """
    pair: str
    status: str = "starting"
    progress_pct: int = 0
    message: str = "Initializing..."
    start_time: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    log_file: str = ""
    
    def update_elapsed(self) -> None:
        """Update elapsed time."""
        self.elapsed_seconds = time.time() - self.start_time


class ProgressMonitor:
    """
    Monitor progress of multiple backtest workers.
    
    Watches log files and parses Freqtrade output to track
    the progress of each worker in real-time.
    Only shows tasks that have actually started (log file exists).
    """
    
    def __init__(
        self,
        log_files: Dict[str, str],
        update_callback: Optional[Callable[[Dict[str, WorkerStatus]], None]] = None,
        update_interval: float = 1.0
    ):
        """
        Initialize the progress monitor.
        
        Args:
            log_files: Dictionary mapping task keys to log file paths
            update_callback: Callback function called on each update
            update_interval: Seconds between updates
        """
        self.log_files = log_files
        self.update_callback = update_callback
        self.update_interval = update_interval
        
        # Track all registered tasks and their statuses
        self._all_tasks: Dict[str, WorkerStatus] = {}
        self.completed_keys: set = set()
        
        # File position tracking for incremental reading
        self._file_positions: Dict[str, int] = {}
        
        # Threading
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the progress monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the progress monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def register_task(self, task_key: str, log_file: str) -> None:
        """
        Register a task (will only show when log file appears).
        
        Args:
            task_key: Unique task identifier
            log_file: Path to the task's log file
        """
        if task_key not in self.completed_keys and task_key not in self._all_tasks:
            self._all_tasks[task_key] = WorkerStatus(
                pair=task_key,
                log_file=log_file
            )
            self._file_positions[task_key] = 0
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            self._update_all_statuses()
            
            if self.update_callback:
                try:
                    # Only pass tasks that have actually started (log file exists)
                    active = self._get_running_tasks()
                    self.update_callback(active)
                except Exception:
                    pass  # Don't let callback errors stop monitoring
            
            self._stop_event.wait(self.update_interval)
    
    def _get_running_tasks(self) -> Dict[str, WorkerStatus]:
        """Get tasks that are actually running (log file exists and not completed)."""
        running = {}
        for task_key, status in self._all_tasks.items():
            if task_key in self.completed_keys:
                continue
            # Only include if log file exists (task has actually started)
            if status.log_file and os.path.exists(status.log_file):
                running[task_key] = status
        return running
    
    def _update_all_statuses(self) -> None:
        """Update status for all active workers."""
        for task_key in list(self._all_tasks.keys()):
            if task_key not in self.completed_keys:
                self._update_worker_status(task_key)
    
    def _update_worker_status(self, task_key: str) -> None:
        """
        Update status for a single worker by reading its log file.
        
        Args:
            task_key: Task key to update
        """
        if task_key not in self._all_tasks:
            return
            
        status = self._all_tasks[task_key]
        status.update_elapsed()
        
        log_file = status.log_file
        if not log_file or not os.path.exists(log_file):
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last known position
                f.seek(self._file_positions.get(task_key, 0))
                new_content = f.read()
                self._file_positions[task_key] = f.tell()
            
            if new_content:
                self._parse_log_content(status, new_content)
                
        except Exception:
            pass  # Ignore file read errors
    
    def _parse_log_content(self, status: WorkerStatus, content: str) -> None:
        """
        Parse log content and update worker status.
        
        Args:
            status: WorkerStatus to update
            content: New log content to parse
        """
        lines = content.strip().split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect different stages
            if 'loading data' in line_lower:
                status.status = "loading"
                status.progress_pct = 10
                # Try to extract date range
                match = re.search(r'from (\d{4}-\d{2}-\d{2})', line)
                if match:
                    status.message = f"Loading data from {match.group(1)}..."
                else:
                    status.message = "Loading historical data..."
                    
            elif 'dataload complete' in line_lower:
                status.status = "calculating"
                status.progress_pct = 30
                status.message = "Calculating indicators..."
                
            elif 'running backtesting' in line_lower:
                status.status = "backtesting"
                status.progress_pct = 40
                status.message = "Running backtest..."
                
            elif 'backtesting with data' in line_lower:
                status.status = "backtesting"
                status.progress_pct = 50
                # Extract date range and days
                match = re.search(r'from (\d{4}-\d{2}-\d{2}).*to (\d{4}-\d{2}-\d{2}).*\((\d+) days\)', line)
                if match:
                    status.message = f"Backtesting {match.group(1)} to {match.group(2)} ({match.group(3)} days)..."
                else:
                    status.message = "Processing trades..."
                    
            elif 'dumping json' in line_lower:
                status.status = "saving"
                status.progress_pct = 95
                status.message = "Saving results..."
                
            elif 'result for' in line_lower:
                status.status = "done"
                status.progress_pct = 100
                status.message = "Complete"
                
            elif 'entry signal' in line_lower:
                # Count entry signals as progress indicator
                status.status = "backtesting"
                if status.progress_pct < 90:
                    status.progress_pct = min(90, status.progress_pct + 1)
                status.message = "Processing signals..."
                
            elif 'error' in line_lower or 'exception' in line_lower:
                status.status = "error"
                # Extract error message
                status.message = line.strip()[-80:]  # Last 80 chars
    
    def get_status(self, task_key: str) -> Optional[WorkerStatus]:
        """Get status for a specific task."""
        return self._all_tasks.get(task_key)
    
    def get_running_tasks(self) -> Dict[str, WorkerStatus]:
        """Get all running (non-completed) worker statuses."""
        return self._get_running_tasks()
    
    def mark_completed(self, task_key: str, success: bool, message: str = "") -> None:
        """
        Mark a worker as completed and remove from active tracking.
        
        Args:
            task_key: Task key
            success: Whether the backtest succeeded
            message: Optional completion message
        """
        self.completed_keys.add(task_key)
        # Remove from tracking to free memory
        if task_key in self._all_tasks:
            del self._all_tasks[task_key]
        if task_key in self._file_positions:
            del self._file_positions[task_key]


def format_worker_progress(statuses: Dict[str, WorkerStatus], max_width: int = 80) -> str:
    """
    Format worker progress for display.
    
    Args:
        statuses: Dictionary of worker statuses
        max_width: Maximum line width
        
    Returns:
        Formatted progress string
    """
    lines = []
    
    for pair, status in statuses.items():
        # Format elapsed time
        elapsed = status.elapsed_seconds
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}m"
        else:
            time_str = f"{elapsed/3600:.1f}h"
        
        # Status indicator
        if status.status == "done":
            indicator = "✓"
        elif status.status == "failed" or status.status == "error":
            indicator = "✗"
        else:
            indicator = "⋯"
        
        # Progress bar
        bar_width = 20
        filled = int(bar_width * status.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Truncate message if needed
        msg_max = max_width - 50
        message = status.message[:msg_max] if len(status.message) > msg_max else status.message
        
        line = f"  {indicator} {pair:12} [{bar}] {status.progress_pct:3}% {time_str:>6} {message}"
        lines.append(line)
    
    return "\n".join(lines)


def format_single_line_progress(statuses: Dict[str, WorkerStatus], max_width: int = 100) -> str:
    """
    Format all worker progress, one line per worker.
    
    Args:
        statuses: Dictionary of worker statuses
        max_width: Maximum line width (for truncating messages)
        
    Returns:
        Multi-line progress string
    """
    if not statuses:
        return ""
    
    lines = []
    for pair, status in statuses.items():
        # Format elapsed time
        elapsed = status.elapsed_seconds
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}m"
        else:
            time_str = f"{elapsed/3600:.1f}h"
        
        # Status indicator
        if status.status == "done":
            indicator = "✓"
        elif status.status == "failed" or status.status == "error":
            indicator = "✗"
        else:
            indicator = "⋯"
        
        # Short status/step name
        step_map = {
            "starting": "启动中",
            "loading": "加载数据",
            "calculating": "计算指标",
            "backtesting": "回测中",
            "saving": "保存结果",
            "done": "完成",
            "failed": "失败",
            "error": "错误"
        }
        step = step_map.get(status.status, status.status)
        
        # Progress bar
        bar_width = 15
        filled = int(bar_width * status.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        line = f"  {indicator} {pair:12} [{bar}] {status.progress_pct:3}% {time_str:>6}  {step}"
        lines.append(line)
    
    return "\n".join(lines)


class ProgressDisplay:
    """
    Handle multi-line progress display with ANSI escape codes.
    
    Shows only running workers + a summary line for overall progress.
    """
    
    def __init__(self, num_workers: int, total_tasks: int = 0):
        """
        Initialize progress display.
        
        Args:
            num_workers: Number of concurrent workers
            total_tasks: Total number of tasks to complete
        """
        self.num_workers = num_workers
        self.total_tasks = total_tasks if total_tasks > 0 else num_workers
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.last_line_count = 0
        self.is_tty = self._check_tty()
        self.last_print_time = 0
        self.print_interval = 10  # Print every 10 seconds if not TTY
        self.initialized = False
        self.start_time = time.time()
    
    def _check_tty(self) -> bool:
        """Check if stdout is a TTY (supports ANSI escape codes)."""
        import sys
        try:
            return sys.stdout.isatty()
        except Exception:
            return False
    
    def mark_task_complete(self, success: bool) -> None:
        """
        Mark a task as completed.
        
        Args:
            success: Whether the task succeeded
        """
        self.completed_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
    
    def _format_summary_line(self) -> str:
        """Format the summary progress line."""
        elapsed = time.time() - self.start_time
        
        # Format elapsed time
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            time_str = f"{int(elapsed//60)}m{int(elapsed%60)}s"
        else:
            time_str = f"{elapsed/3600:.1f}h"
        
        # Progress percentage
        pct = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # ETA calculation
        if self.completed_tasks > 0 and self.completed_tasks < self.total_tasks:
            avg_time = elapsed / self.completed_tasks
            remaining = (self.total_tasks - self.completed_tasks) * avg_time
            if remaining < 60:
                eta_str = f"~{remaining:.0f}s"
            elif remaining < 3600:
                eta_str = f"~{int(remaining//60)}m"
            else:
                eta_str = f"~{remaining/3600:.1f}h"
        else:
            eta_str = ""
        
        # Build summary line
        status_parts = [f"✓{self.successful_tasks}"]
        if self.failed_tasks > 0:
            status_parts.append(f"✗{self.failed_tasks}")
        status_str = " ".join(status_parts)
        
        summary = f"  Progress: [{bar}] {self.completed_tasks}/{self.total_tasks} ({pct:.0f}%) {status_str}  {time_str} {eta_str}"
        return summary
    
    def update(self, statuses: Dict[str, WorkerStatus]) -> None:
        """
        Update the progress display.
        
        Shows only running workers + summary line.
        
        Args:
            statuses: Dictionary of worker statuses (only running ones)
        """
        import sys
        
        # Filter to only show running workers (not done/failed)
        running = {k: v for k, v in statuses.items() 
                   if v.status not in ("done", "failed")}
        
        lines = []
        
        # Add running worker lines
        for pair, status in running.items():
            # Format elapsed time
            elapsed = status.elapsed_seconds
            if elapsed < 60:
                time_str = f"{elapsed:.0f}s"
            elif elapsed < 3600:
                time_str = f"{elapsed/60:.1f}m"
            else:
                time_str = f"{elapsed/3600:.1f}h"
            
            # Status indicator
            indicator = "⋯"
            
            # Short status/step name
            step_map = {
                "starting": "启动中",
                "loading": "加载数据",
                "calculating": "计算指标",
                "backtesting": "回测中",
                "saving": "保存结果",
            }
            step = step_map.get(status.status, status.status)
            
            # Progress bar (shorter for worker lines)
            bar_width = 10
            filled = int(bar_width * status.progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            line = f"    {indicator} {pair:20} [{bar}] {status.progress_pct:3}% {time_str:>6}  {step}"
            lines.append(line)
        
        # Add summary line at the end
        lines.append(self._format_summary_line())
        
        progress_str = "\n".join(lines)
        current_line_count = len(lines)
        
        if self.is_tty:
            # TTY mode: update in place using ANSI escape codes
            if self.initialized and self.last_line_count > 0:
                # Move cursor up to overwrite previous output
                sys.stdout.write(f"\033[{self.last_line_count}A")
                # Clear each line
                for _ in range(self.last_line_count):
                    sys.stdout.write("\033[2K\n")
                # Move back up
                sys.stdout.write(f"\033[{self.last_line_count}A")
            
            # Print new progress
            sys.stdout.write(progress_str + "\n")
            sys.stdout.flush()
            
            self.last_line_count = current_line_count
            self.initialized = True
        else:
            # Non-TTY mode: print periodically
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                print(progress_str)
                self.last_print_time = current_time
    
    def finish(self) -> None:
        """
        Finish progress display - clear the progress lines.
        """
        import sys
        if self.is_tty and self.initialized and self.last_line_count > 0:
            # Move cursor up and clear all progress lines
            sys.stdout.write(f"\033[{self.last_line_count}A")
            for _ in range(self.last_line_count):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{self.last_line_count}A")
            sys.stdout.flush()
        self.last_line_count = 0
        self.initialized = False
