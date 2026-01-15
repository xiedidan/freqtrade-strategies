"""
Task executor for parallel backtest execution.

This module implements the TaskExecutor class that manages parallel
execution of backtest tasks using ProcessPoolExecutor.
"""

import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import List, Callable, Optional, Any
from dataclasses import dataclass

from .models import WorkerConfig, WorkerResult, BacktestConfig
from .worker import run_backtest_task


@dataclass
class BacktestTask:
    """
    Represents a single backtest task to be executed.
    
    Attributes:
        worker_config: Isolated worker configuration
        strategy: Strategy name to backtest
        timerange: Optional time range for backtest
        extra_args: Additional Freqtrade CLI arguments
        timeout: Timeout in seconds for this task (None for no timeout)
        time_chunk: Optional time chunk label for display
    """
    worker_config: WorkerConfig
    strategy: str
    timerange: Optional[str] = None
    extra_args: Optional[List[str]] = None
    timeout: Optional[int] = None
    time_chunk: Optional[str] = None


class TaskExecutor:
    """
    Manage parallel execution of backtest tasks.
    
    This class handles:
    - Parallel execution using ProcessPoolExecutor
    - Progress callbacks for monitoring
    - Graceful shutdown on interrupt signals
    - Timeout management for individual tasks (optional)
    
    Attributes:
        max_workers: Maximum number of concurrent worker processes
        timeout: Default timeout in seconds for each task (None for no timeout)
        _executor: The ProcessPoolExecutor instance
        _futures: Dictionary mapping futures to their tasks
        _shutdown_requested: Flag indicating shutdown was requested
        _results: List of completed results
    """
    
    def __init__(self, max_workers: int, timeout: Optional[int] = None):
        """
        Initialize the task executor.
        
        Args:
            max_workers: Maximum number of concurrent worker processes
            timeout: Default timeout per task in seconds (None for no timeout)
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        self.max_workers = max_workers
        self.timeout = timeout
        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures: dict = {}
        self._shutdown_requested = False
        self._results: List[WorkerResult] = []
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # Store original handlers
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        
        # Only handle SIGTERM on non-Windows platforms
        if sys.platform != 'win32':
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        
        def shutdown_handler(signum, frame):
            """Handle shutdown signals gracefully."""
            print("\n[Executor] Received interrupt signal, initiating graceful shutdown...")
            self._shutdown_requested = True
            self.shutdown(wait=False)
            
            # Re-raise the signal to allow proper cleanup
            if signum == signal.SIGINT and self._original_sigint_handler:
                if callable(self._original_sigint_handler):
                    self._original_sigint_handler(signum, frame)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, shutdown_handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        
        if sys.platform != 'win32' and self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
    
    def execute_all(
        self,
        tasks: List[BacktestTask],
        progress_callback: Optional[Callable[[WorkerResult, int, int], None]] = None,
        task_started_callback: Optional[Callable[[BacktestTask], None]] = None
    ) -> List[WorkerResult]:
        """
        Execute all backtest tasks in parallel.
        
        Args:
            tasks: List of backtest tasks to execute
            progress_callback: Optional callback function called after each task completes.
                              Signature: callback(result, completed_count, total_count)
            task_started_callback: Optional callback when a task starts execution.
                              Signature: callback(task)
        
        Returns:
            List of WorkerResult objects for all tasks
        """
        if not tasks:
            return []
        
        self._results = []
        self._shutdown_requested = False
        total_tasks = len(tasks)
        completed_count = 0
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        try:
            # Create executor with specified max_workers
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            self._futures = {}
            
            # Submit all tasks
            for task in tasks:
                if self._shutdown_requested:
                    break
                
                # Notify that task is starting
                if task_started_callback:
                    try:
                        task_started_callback(task)
                    except Exception:
                        pass
                
                future = self._executor.submit(
                    run_backtest_task,
                    worker_config=task.worker_config,
                    strategy=task.strategy,
                    timerange=task.timerange,
                    extra_args=task.extra_args,
                    timeout=task.timeout if task.timeout is not None else self.timeout
                )
                self._futures[future] = task
            
            # Process completed tasks
            for future in as_completed(self._futures.keys()):
                if self._shutdown_requested:
                    # Mark remaining tasks as failed due to shutdown
                    break
                
                task = self._futures[future]
                
                try:
                    result = future.result()
                    # Attach time_chunk info to result
                    if task.time_chunk:
                        result.time_chunk = task.time_chunk
                except Exception as e:
                    # Handle any unexpected exceptions
                    result = WorkerResult(
                        pair=task.worker_config.pair,
                        success=False,
                        error_message=f"Task execution failed: {str(e)}",
                        duration=0.0
                    )
                
                self._results.append(result)
                completed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(result, completed_count, total_tasks)
                    except Exception:
                        # Don't let callback errors affect execution
                        pass
            
            # Handle shutdown - mark incomplete tasks as failed
            if self._shutdown_requested:
                for future, task in self._futures.items():
                    if not future.done():
                        future.cancel()
                        self._results.append(WorkerResult(
                            pair=task.worker_config.pair,
                            success=False,
                            error_message="Task cancelled due to shutdown",
                            duration=0.0
                        ))
        
        finally:
            # Cleanup
            self._restore_signal_handlers()
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
        
        return self._results
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor and cancel pending tasks.
        
        Args:
            wait: If True, wait for running tasks to complete.
                  If False, attempt to cancel all pending tasks immediately.
        """
        self._shutdown_requested = True
        
        if self._executor:
            # Cancel pending futures
            for future in self._futures.keys():
                if not future.done():
                    future.cancel()
            
            # Shutdown executor
            self._executor.shutdown(wait=wait, cancel_futures=True)
    
    @property
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    def get_partial_results(self) -> List[WorkerResult]:
        """
        Get results collected so far.
        
        Useful for retrieving partial results after a shutdown.
        
        Returns:
            List of WorkerResult objects for completed tasks
        """
        return list(self._results)


def create_tasks_from_config(
    backtest_config: BacktestConfig,
    worker_configs: List[WorkerConfig]
) -> List[BacktestTask]:
    """
    Create BacktestTask objects from configuration.
    
    Helper function to create task objects from the main backtest
    configuration and generated worker configurations.
    
    Args:
        backtest_config: Main backtest configuration
        worker_configs: List of worker configurations (one per pair)
    
    Returns:
        List of BacktestTask objects ready for execution
    """
    tasks = []
    
    for worker_config in worker_configs:
        task = BacktestTask(
            worker_config=worker_config,
            strategy=backtest_config.strategy,
            timerange=backtest_config.timerange,
            extra_args=backtest_config.extra_args,
            timeout=backtest_config.timeout
        )
        tasks.append(task)
    
    return tasks
