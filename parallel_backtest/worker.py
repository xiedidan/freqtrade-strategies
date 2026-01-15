"""
Backtest worker implementation.

This module implements the BacktestWorker class that executes individual
backtests using the Freqtrade CLI via subprocess with real-time progress monitoring.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from glob import glob
from typing import List, Optional, Dict, Any, Callable

from .models import WorkerConfig, WorkerResult


class WorkerProgress:
    """
    Track progress of a backtest worker.
    
    Attributes:
        pair: Trading pair being processed
        status: Current status (loading, calculating, backtesting, done)
        progress_pct: Progress percentage (0-100)
        message: Current status message
        start_time: When the worker started
    """
    
    def __init__(self, pair: str):
        self.pair = pair
        self.status = "starting"
        self.progress_pct = 0
        self.message = "Initializing..."
        self.start_time = time.time()
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


class BacktestWorker:
    """
    Execute single pair backtest using Freqtrade CLI with progress monitoring.
    
    This class handles:
    - Building the Freqtrade CLI command
    - Executing the backtest via subprocess with real-time output
    - Parsing progress from Freqtrade logs
    - Parsing result files
    - Error handling (no timeout by default)
    
    Attributes:
        strategy: Name of the trading strategy
        timerange: Optional time range for backtest
        extra_args: Additional Freqtrade CLI arguments
        timeout: Timeout in seconds (None for no timeout)
    """
    
    def __init__(
        self,
        strategy: str,
        timerange: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize the backtest worker.
        
        Args:
            strategy: Name of the trading strategy to backtest
            timerange: Optional time range (e.g., "20240101-20241231")
            extra_args: Additional arguments to pass to Freqtrade
            timeout: Timeout in seconds (None for no timeout, default)
        """
        self.strategy = strategy
        self.timerange = timerange
        self.extra_args = extra_args or []
        self.timeout = timeout
    
    def build_command(self, worker_config: WorkerConfig) -> List[str]:
        """
        Build the Freqtrade CLI command for backtest execution.
        
        Args:
            worker_config: Worker configuration with isolated paths
            
        Returns:
            List of command arguments for subprocess
        """
        # Generate export filename (just the filename, not full path)
        export_filename = f"backtest-result-{worker_config.pair.replace('/', '_')}.json"
        
        # Use Python module invocation for better cross-platform compatibility
        cmd = [
            sys.executable,  # Use current Python interpreter
            "-m", "freqtrade",
            "backtesting",
            "--config", worker_config.config_path,
            "--strategy", self.strategy,
            "--export", "trades",
            "--export-directory", worker_config.result_dir,
            "--export-filename", export_filename,
        ]
        
        # Add timerange if specified
        if self.timerange:
            cmd.extend(["--timerange", self.timerange])
        
        # Add log file path
        cmd.extend(["--logfile", worker_config.log_file])
        
        # Add extra arguments
        if self.extra_args:
            cmd.extend(self.extra_args)
        
        return cmd
    
    def run(
        self, 
        worker_config: WorkerConfig,
        progress_callback: Optional[Callable[[WorkerProgress], None]] = None
    ) -> WorkerResult:
        """
        Run backtest for a single trading pair with progress monitoring.
        
        Executes the Freqtrade CLI command and monitors progress in real-time.
        
        Args:
            worker_config: Isolated worker configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            WorkerResult with backtest outcome
        """
        start_time = time.time()
        pair = worker_config.pair
        progress = WorkerProgress(pair)
        
        try:
            # Build command
            cmd = self.build_command(worker_config)
            
            # Execute backtest with real-time output monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd()
            )
            
            # Monitor output and update progress
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                output_lines.append(line)
                
                # Parse progress from output
                self._update_progress(progress, line)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress)
            
            # Wait for process to complete
            return_code = process.wait(timeout=self.timeout)
            
            duration = time.time() - start_time
            
            # Check for execution errors
            if return_code != 0:
                error_msg = self._extract_error_from_output(output_lines)
                return WorkerResult(
                    pair=pair,
                    success=False,
                    error_message=error_msg,
                    duration=duration
                )
            
            # Parse result file
            result_file = self._find_result_file(worker_config.result_dir, pair)
            
            if result_file is None:
                return WorkerResult(
                    pair=pair,
                    success=False,
                    error_message="Result file not found after backtest",
                    duration=duration
                )
            
            # Extract statistics from result
            trades_count, profit_ratio = self._parse_result_stats(
                result_file, self.strategy
            )
            
            return WorkerResult(
                pair=pair,
                success=True,
                result_file=result_file,
                duration=duration,
                trades_count=trades_count,
                profit_ratio=profit_ratio
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            duration = time.time() - start_time
            return WorkerResult(
                pair=pair,
                success=False,
                error_message=f"Backtest timed out after {self.timeout} seconds",
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return WorkerResult(
                pair=pair,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                duration=duration
            )
    
    def _update_progress(self, progress: WorkerProgress, line: str) -> None:
        """
        Update progress based on Freqtrade output line.
        
        Args:
            progress: WorkerProgress object to update
            line: Output line from Freqtrade
        """
        line_lower = line.lower()
        
        # Detect different stages
        if 'loading data' in line_lower:
            progress.status = "loading"
            progress.progress_pct = 10
            # Try to extract date range
            match = re.search(r'from (\d{4}-\d{2}-\d{2})', line)
            if match:
                progress.message = f"Loading data from {match.group(1)}..."
            else:
                progress.message = "Loading historical data..."
                
        elif 'dataload complete' in line_lower:
            progress.status = "calculating"
            progress.progress_pct = 30
            progress.message = "Calculating indicators..."
            
        elif 'running backtesting' in line_lower:
            progress.status = "backtesting"
            progress.progress_pct = 40
            progress.message = "Running backtest..."
            
        elif 'backtesting with data' in line_lower:
            progress.status = "backtesting"
            progress.progress_pct = 50
            # Extract date range
            match = re.search(r'from (\d{4}-\d{2}-\d{2}).*to (\d{4}-\d{2}-\d{2})', line)
            if match:
                progress.message = f"Backtesting {match.group(1)} to {match.group(2)}..."
            else:
                progress.message = "Processing trades..."
                
        elif 'dumping json' in line_lower or 'result for' in line_lower:
            progress.status = "saving"
            progress.progress_pct = 90
            progress.message = "Saving results..."
            
        elif 'entry signal' in line_lower:
            # Count entry signals as progress indicator
            progress.status = "backtesting"
            if progress.progress_pct < 85:
                progress.progress_pct = min(85, progress.progress_pct + 1)
            progress.message = "Processing signals..."
    
    def _extract_error_from_output(self, output_lines: List[str]) -> str:
        """
        Extract error message from output lines.
        
        Args:
            output_lines: List of output lines
            
        Returns:
            Extracted error message
        """
        # Look for error lines
        for line in reversed(output_lines):
            line_lower = line.lower()
            if 'error' in line_lower or 'exception' in line_lower:
                return line.strip()
        
        # Return last non-empty line
        for line in reversed(output_lines):
            if line.strip():
                return line.strip()[:200]
        
        return "Backtest failed with no error message"
    
    def _find_result_file(
        self, 
        result_dir: str, 
        pair: str
    ) -> Optional[str]:
        """
        Find the result JSON or ZIP file in the result directory.
        
        Args:
            result_dir: Directory containing result files
            pair: Trading pair name
            
        Returns:
            Path to result file or None if not found
        """
        # Look for the specific result file (JSON or ZIP)
        pair_safe = pair.replace('/', '_')
        
        # Try JSON first
        expected_json = os.path.join(
            result_dir, 
            f"backtest-result-{pair_safe}.json"
        )
        if os.path.exists(expected_json):
            return expected_json
        
        # Try ZIP
        expected_zip = os.path.join(
            result_dir, 
            f"backtest-result-{pair_safe}.zip"
        )
        if os.path.exists(expected_zip):
            return expected_zip
        
        # Fallback: look for any result file in result directory
        # First try ZIP files (Freqtrade default)
        zip_files = glob(os.path.join(result_dir, "*.zip"))
        if zip_files:
            return zip_files[0]
        
        # Then try JSON files
        json_files = glob(os.path.join(result_dir, "*.json"))
        # Filter out meta files and .last_result.json
        json_files = [
            f for f in json_files 
            if not f.endswith('.meta.json') and not f.endswith('.last_result.json')
        ]
        
        if json_files:
            return json_files[0]
        
        return None
    
    def _parse_result_stats(
        self, 
        result_file: str, 
        strategy: str
    ) -> tuple:
        """
        Parse statistics from the result JSON or ZIP file.
        
        Args:
            result_file: Path to the result JSON or ZIP file
            strategy: Strategy name to look up in results
            
        Returns:
            Tuple of (trades_count, profit_ratio)
        """
        import zipfile
        
        try:
            # Handle ZIP files
            if result_file.endswith('.zip'):
                with zipfile.ZipFile(result_file, 'r') as zf:
                    # Find the JSON file inside the ZIP
                    json_files = [n for n in zf.namelist() if n.endswith('.json')]
                    if not json_files:
                        return 0, 0.0
                    
                    # Read the first JSON file
                    content = zf.read(json_files[0])
                    data = json.loads(content.decode('utf-8'))
            else:
                # Handle regular JSON files
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Navigate to strategy results
            strategy_data = data.get('strategy', {}).get(strategy, {})
            
            if not strategy_data:
                # Try alternative structure
                if strategy in data:
                    strategy_data = data[strategy]
                else:
                    return 0, 0.0
            
            trades_count = strategy_data.get('total_trades', 0)
            profit_ratio = strategy_data.get('profit_total', 0.0)
            
            return trades_count, profit_ratio
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0, 0.0


def run_backtest_task(
    worker_config: WorkerConfig,
    strategy: str,
    timerange: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None
) -> WorkerResult:
    """
    Convenience function to run a single backtest task.
    
    This function is designed to be used with ProcessPoolExecutor
    for parallel execution.
    
    Args:
        worker_config: Isolated worker configuration
        strategy: Strategy name
        timerange: Optional time range
        extra_args: Additional Freqtrade arguments
        timeout: Timeout in seconds (None for no timeout)
        
    Returns:
        WorkerResult with backtest outcome
    """
    worker = BacktestWorker(
        strategy=strategy,
        timerange=timerange,
        extra_args=extra_args,
        timeout=timeout
    )
    return worker.run(worker_config)
