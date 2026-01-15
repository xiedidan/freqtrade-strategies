"""
Main entry point for the parallel backtest tool.

This module serves as the entry point when running the tool as a module:
    python -m parallel_backtest

It integrates all components:
- CLI argument parsing
- Configuration generation
- Parallel task execution
- Result merging
- Progress display
"""

import sys
import time
from datetime import datetime
from typing import List, Optional

from .cli import parse_args, get_cpu_count
from .config import ConfigGenerator
from .executor import TaskExecutor, BacktestTask
from .merger import ResultMerger
from .models import BacktestConfig, WorkerResult, MergedResult
from .progress import ProgressMonitor, ProgressDisplay
from .timerange import split_timerange_monthly, TimeChunk
from .utils import (
    setup_logging,
    cleanup_temp_files,
    format_duration,
    format_profit,
    calculate_speedup,
    estimate_sequential_time,
    generate_output_filename,
    ensure_directory,
)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def print_header(config: BacktestConfig, time_split: bool = False, num_chunks: int = 0) -> None:
    """
    Print the startup header with configuration summary.
    
    Args:
        config: Backtest configuration
        time_split: Whether time splitting is enabled
        num_chunks: Number of time chunks if splitting
    """
    print("\n" + "=" * 60)
    print("  Parallel Backtest Tool for Freqtrade")
    print("=" * 60)
    print(f"  Strategy:    {config.strategy}")
    print(f"  Config:      {config.config_path}")
    print(f"  Pairs:       {len(config.pairs)} pairs")
    print(f"  Workers:     {config.max_workers}")
    if config.timeout:
        print(f"  Timeout:     {config.timeout}s per task")
    else:
        print(f"  Timeout:     No limit")
    if config.timerange:
        print(f"  Timerange:   {config.timerange}")
    if time_split:
        total_tasks = len(config.pairs) * num_chunks
        print(f"  Time Split:  {num_chunks} months × {len(config.pairs)} pairs = {total_tasks} tasks")
    print(f"  Output:      {config.output_dir}")
    if config.debug:
        print(f"  Debug:       Enabled (temp files preserved)")
    print("=" * 60 + "\n")


def print_task_result(result: WorkerResult, completed: int, total: int) -> None:
    """
    Print result summary for a completed task (non-tqdm mode).
    
    Args:
        result: Worker result
        completed: Number of completed tasks
        total: Total number of tasks
    """
    status = "✓" if result.success else "✗"
    duration = format_duration(result.duration)
    
    # Include time chunk in display if present
    task_name = result.task_key
    
    if result.success:
        profit_str = f"{result.profit_ratio * 100:+.2f}%"
        trades_str = f"{result.trades_count} trades"
        print(f"  [{completed}/{total}] {status} {task_name}: {profit_str}, {trades_str} ({duration})")
    else:
        error = result.error_message or "Unknown error"
        # Truncate long error messages
        if len(error) > 50:
            error = error[:47] + "..."
        print(f"  [{completed}/{total}] {status} {task_name}: FAILED - {error} ({duration})")


def create_progress_callback(pbar: Optional['tqdm'], results_list: List[WorkerResult]):
    """
    Create a progress callback function.
    
    Args:
        pbar: Optional tqdm progress bar
        results_list: List to store results for summary
        
    Returns:
        Callback function for progress updates
    """
    def callback(result: WorkerResult, completed: int, total: int) -> None:
        results_list.append(result)
        
        if pbar is not None:
            # Update tqdm progress bar
            pbar.update(1)
            
            # Update description with current pair status
            status = "✓" if result.success else "✗"
            if result.success:
                desc = f"{status} {result.pair}: {result.profit_ratio * 100:+.2f}%"
            else:
                # Show brief error reason
                error_brief = result.error_message[:50] if result.error_message else "Unknown error"
                desc = f"{status} {result.pair}: {error_brief}"
            pbar.set_postfix_str(desc)
        else:
            # Print result without tqdm
            print_task_result(result, completed, total)
    
    return callback


def print_summary(
    results: List[WorkerResult],
    merged_result: Optional[MergedResult],
    total_time: float,
    config: BacktestConfig
) -> None:
    """
    Print final execution summary.
    
    Args:
        results: List of all worker results
        merged_result: Merged result (if successful)
        total_time: Total execution time in seconds
        config: Backtest configuration
    """
    print("\n" + "=" * 60)
    print("  Execution Summary")
    print("=" * 60)
    
    # Count results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"  Total pairs:     {len(results)}")
    print(f"  Successful:      {successful}")
    print(f"  Failed:          {failed}")
    
    # Print failed pairs if any
    if failed > 0:
        failed_pairs = [r.pair for r in results if not r.success]
        print(f"  Failed pairs:    {', '.join(failed_pairs)}")
        # Print failure reasons
        print("  Failure details:")
        for r in results:
            if not r.success:
                print(f"    - {r.pair}: {r.error_message}")
    
    print("-" * 60)
    
    # Timing information
    print(f"  Execution time:  {format_duration(total_time)}")
    
    # Calculate speedup
    sequential_time = estimate_sequential_time([r.duration for r in results])
    if sequential_time > 0:
        speedup = calculate_speedup(total_time, sequential_time)
        print(f"  Sequential est:  {format_duration(sequential_time)}")
        print(f"  Speedup:         {speedup:.2f}x")
    
    # Print merged result statistics
    if merged_result and merged_result.total_trades > 0:
        print("-" * 60)
        print("  Merged Results:")
        print(f"    Total trades:  {merged_result.total_trades}")
        print(f"    Total profit:  {format_profit(merged_result.total_profit_ratio, merged_result.total_profit_abs)}")
        print(f"    Win rate:      {merged_result.winrate * 100:.2f}%")
        print(f"    Max drawdown:  {merged_result.max_drawdown:.2f}")
        print(f"    Output file:   {merged_result.output_file}")
    
    print("=" * 60 + "\n")


def run_parallel_backtest(config: BacktestConfig) -> int:
    """
    Run the parallel backtest with the given configuration.
    
    Args:
        config: Backtest configuration
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = setup_logging(debug=config.debug)
    start_time = time.time()
    config_generator = None
    results: List[WorkerResult] = []
    merged_result: Optional[MergedResult] = None
    progress_monitor = None
    
    try:
        # Check if time splitting is beneficial
        num_cpus = get_cpu_count()
        time_chunks, time_split_enabled = split_timerange_monthly(
            config.timerange,
            len(config.pairs),
            num_cpus
        )
        
        # Print header with time split info
        print_header(config, time_split_enabled, len(time_chunks) if time_split_enabled else 0)
        
        # Ensure output directory exists
        if not ensure_directory(config.output_dir):
            print(f"Error: Cannot create output directory: {config.output_dir}")
            return 1
        
        # Generate worker configurations
        print("Generating worker configurations...")
        config_generator = ConfigGenerator(config.config_path)
        
        # Create tasks based on whether time splitting is enabled
        tasks: List[BacktestTask] = []
        log_files = {}
        
        if time_split_enabled and time_chunks:
            # Create tasks for each pair × time_chunk combination
            worker_id = 0
            for pair in config.pairs:
                for chunk in time_chunks:
                    worker_config = config_generator.generate_worker_config(
                        pair, worker_id, time_chunk=chunk.label
                    )
                    task = BacktestTask(
                        worker_config=worker_config,
                        strategy=config.strategy,
                        timerange=chunk.timerange,
                        extra_args=config.extra_args,
                        timeout=config.timeout,
                        time_chunk=chunk.label
                    )
                    tasks.append(task)
                    log_files[f"{pair}:{chunk.label}"] = worker_config.log_file
                    worker_id += 1
        else:
            # Original behavior: one task per pair
            worker_configs = config_generator.generate_all_worker_configs(config.pairs)
            for wc in worker_configs:
                task = BacktestTask(
                    worker_config=wc,
                    strategy=config.strategy,
                    timerange=config.timerange,
                    extra_args=config.extra_args,
                    timeout=config.timeout
                )
                tasks.append(task)
                log_files[wc.pair] = wc.log_file
        
        logger.debug(f"Generated {len(tasks)} tasks")
        
        print(f"Starting parallel backtest with {len(tasks)} tasks using {config.max_workers} workers...\n")
        
        # Create executor
        executor = TaskExecutor(
            max_workers=config.max_workers,
            timeout=config.timeout
        )
        
        # Create progress monitor (without pre-registering all tasks)
        progress_monitor = ProgressMonitor(
            log_files=log_files,
            update_interval=2.0
        )
        
        # Pass total_tasks to progress display for accurate summary
        progress_display = ProgressDisplay(
            num_workers=config.max_workers,
            total_tasks=len(tasks)
        )
        
        # Track completed tasks
        completed_tasks = set()
        
        def progress_display_callback(statuses):
            """Update progress display with only active workers."""
            progress_display.update(statuses)
        
        # Start progress monitoring
        progress_monitor.update_callback = progress_display_callback
        progress_monitor.start()
        
        # Execute with progress tracking
        results_for_callback: List[WorkerResult] = []
        
        def task_started_callback(task: BacktestTask) -> None:
            """Register task with progress monitor when it starts."""
            task_key = f"{task.worker_config.pair}:{task.time_chunk}" if task.time_chunk else task.worker_config.pair
            progress_monitor.register_task(task_key, task.worker_config.log_file)
        
        def task_complete_callback(result: WorkerResult, completed: int, total: int) -> None:
            """Handle task completion."""
            results_for_callback.append(result)
            completed_tasks.add(result.task_key)
            
            # Update progress display counters
            progress_display.mark_task_complete(result.success)
            
            # Update progress monitor (removes from active tracking)
            if progress_monitor:
                progress_monitor.mark_completed(
                    result.task_key, 
                    result.success,
                    f"{result.profit_ratio*100:+.2f}%" if result.success else result.error_message or "Failed"
                )
            
            # Finish current progress display before printing result
            progress_display.finish()
            
            # Print completion message
            print_task_result(result, completed, total)
        
        # Execute all tasks
        results = executor.execute_all(
            tasks, 
            progress_callback=task_complete_callback,
            task_started_callback=task_started_callback
        )
        
        # Stop progress monitor and finish display
        if progress_monitor:
            progress_monitor.stop()
        progress_display.finish()
        
        print()  # Ensure we're on a new line
        
        # Check if we have any successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            print("\nError: All backtests failed. No results to merge.")
            total_time = time.time() - start_time
            print_summary(results, None, total_time, config)
            return 1
        
        # Merge results
        print("\nMerging results...")
        
        # Generate output filename
        output_file = generate_output_filename(
            config.output_dir,
            config.strategy,
            extension="zip"
        )
        
        # Get starting balance from config if available
        try:
            import json
            with open(config.config_path, 'r') as f:
                base_config = json.load(f)
            starting_balance = base_config.get('dry_run_wallet', 1000.0)
            stake_currency = base_config.get('stake_currency', 'USDT')
        except Exception:
            starting_balance = 1000.0
            stake_currency = 'USDT'
        
        merger = ResultMerger(
            starting_balance=starting_balance,
            stake_currency=stake_currency
        )
        
        merged_result = merger.merge(
            results=results,
            output_path=output_file,
            strategy_name=config.strategy,
            timerange=config.timerange
        )
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Update merged result with execution time and speedup
        sequential_time = estimate_sequential_time([r.duration for r in results])
        merged_result.execution_time = total_time
        merged_result.speedup_ratio = calculate_speedup(total_time, sequential_time)
        
        # Print summary
        print_summary(results, merged_result, total_time, config)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        if progress_monitor:
            progress_monitor.stop()
        total_time = time.time() - start_time
        print_summary(results, merged_result, total_time, config)
        return 130  # Standard exit code for SIGINT
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
        
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1
        
    except Exception as e:
        logger.exception("Unexpected error during execution")
        print(f"\nUnexpected error: {e}")
        return 1
        
    finally:
        # Cleanup temporary files unless in debug mode
        if config_generator and not config.debug:
            print("Cleaning up temporary files...")
            cleanup_temp_files(config_generator.temp_dir)
        elif config_generator and config.debug:
            print(f"Debug mode: Temporary files preserved at {config_generator.temp_dir}")


def main() -> int:
    """
    Main entry point for the parallel backtest tool.
    
    Returns:
        Exit code
    """
    try:
        # Parse command line arguments
        config, extra_args = parse_args()
        
        # Run parallel backtest
        return run_parallel_backtest(config)
        
    except SystemExit as e:
        # argparse calls sys.exit on error
        return e.code if isinstance(e.code, int) else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
