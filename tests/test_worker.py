"""
Unit tests for BacktestWorker.

Tests the backtest worker functionality including command building,
result parsing, and error handling.

**Validates: Requirements 4.1, 4.2**
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from typing import List

from parallel_backtest.worker import BacktestWorker, run_backtest_task
from parallel_backtest.models import WorkerConfig, WorkerResult


def create_temp_dir() -> str:
    """Create a temporary directory and return its path."""
    return tempfile.mkdtemp(prefix='test_worker_')


def cleanup_temp_dir(path: str) -> None:
    """Safely cleanup a temporary directory."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except PermissionError:
        pass


def create_worker_config(
    pair: str = 'BTC/USDT',
    worker_id: int = 0,
    temp_dir: str = None
) -> WorkerConfig:
    """Create a WorkerConfig for testing."""
    if temp_dir is None:
        temp_dir = create_temp_dir()
    
    worker_dir = os.path.join(temp_dir, f'worker_{worker_id}')
    os.makedirs(worker_dir, exist_ok=True)
    
    result_dir = os.path.join(worker_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    config_path = os.path.join(worker_dir, 'config.json')
    # Create a minimal config file
    with open(config_path, 'w') as f:
        json.dump({'exchange': {'pair_whitelist': [pair]}}, f)
    
    log_file = os.path.join(worker_dir, 'freqtrade.log')
    
    return WorkerConfig(
        pair=pair,
        config_path=config_path,
        result_dir=result_dir,
        log_file=log_file,
        worker_id=worker_id
    )


def create_mock_result_file(
    result_dir: str,
    pair: str,
    strategy: str,
    trades_count: int = 10,
    profit_total: float = 0.05
) -> str:
    """Create a mock Freqtrade result file."""
    pair_safe = pair.replace('/', '_')
    result_file = os.path.join(result_dir, f'backtest-result-{pair_safe}.json')
    
    result_data = {
        'strategy': {
            strategy: {
                'trades': [{'pair': pair} for _ in range(trades_count)],
                'total_trades': trades_count,
                'profit_total': profit_total,
                'profit_total_abs': profit_total * 1000,
                'wins': trades_count // 2,
                'losses': trades_count // 2,
                'draws': 0
            }
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f)
    
    return result_file


class TestBacktestWorkerInit:
    """Tests for BacktestWorker initialization"""
    
    def test_init_with_defaults(self):
        """Test initialization with default values"""
        worker = BacktestWorker(strategy='TestStrategy')
        
        assert worker.strategy == 'TestStrategy'
        assert worker.timerange is None
        assert worker.extra_args == []
        assert worker.timeout == 3600
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters"""
        worker = BacktestWorker(
            strategy='TestStrategy',
            timerange='20240101-20241231',
            extra_args=['--cache', 'none'],
            timeout=1800
        )
        
        assert worker.strategy == 'TestStrategy'
        assert worker.timerange == '20240101-20241231'
        assert worker.extra_args == ['--cache', 'none']
        assert worker.timeout == 1800


class TestBuildCommand:
    """Tests for command building"""
    
    def test_build_basic_command(self):
        """Test building basic command without optional params"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(
                pair='BTC/USDT',
                worker_id=0,
                temp_dir=temp_dir
            )
            worker = BacktestWorker(strategy='TestStrategy')
            
            cmd = worker.build_command(worker_config)
            
            # Check required elements
            assert 'freqtrade' in cmd
            assert 'backtesting' in cmd
            assert '--config' in cmd
            assert worker_config.config_path in cmd
            assert '--strategy' in cmd
            assert 'TestStrategy' in cmd
            assert '--export' in cmd
            assert 'trades' in cmd
            assert '--logfile' in cmd
            assert worker_config.log_file in cmd
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_build_command_with_timerange(self):
        """Test building command with timerange"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(
                strategy='TestStrategy',
                timerange='20240101-20241231'
            )
            
            cmd = worker.build_command(worker_config)
            
            assert '--timerange' in cmd
            assert '20240101-20241231' in cmd
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_build_command_with_extra_args(self):
        """Test building command with extra arguments"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(
                strategy='TestStrategy',
                extra_args=['--cache', 'none', '--enable-protections']
            )
            
            cmd = worker.build_command(worker_config)
            
            assert '--cache' in cmd
            assert 'none' in cmd
            assert '--enable-protections' in cmd
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_build_command_export_filename(self):
        """Test that export filename and directory are correctly set"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(
                pair='ETH/USDT',
                temp_dir=temp_dir
            )
            worker = BacktestWorker(strategy='TestStrategy')
            
            cmd = worker.build_command(worker_config)
            
            # Find the export filename
            export_filename_idx = cmd.index('--export-filename')
            export_filename = cmd[export_filename_idx + 1]
            
            # Find the export directory
            export_dir_idx = cmd.index('--export-directory')
            export_dir = cmd[export_dir_idx + 1]
            
            assert 'ETH_USDT' in export_filename
            assert export_dir == worker_config.result_dir
        finally:
            cleanup_temp_dir(temp_dir)


class TestResultParsing:
    """Tests for result file parsing"""
    
    def test_find_result_file_exact_match(self):
        """Test finding result file with exact name match"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(
                pair='BTC/USDT',
                temp_dir=temp_dir
            )
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create the expected result file
            result_file = create_mock_result_file(
                worker_config.result_dir,
                'BTC/USDT',
                'TestStrategy'
            )
            
            found_file = worker._find_result_file(
                worker_config.result_dir,
                'BTC/USDT'
            )
            
            assert found_file == result_file
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_find_result_file_fallback(self):
        """Test finding result file with fallback to any JSON"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create a result file with different name
            result_file = os.path.join(
                worker_config.result_dir,
                'some-other-result.json'
            )
            with open(result_file, 'w') as f:
                json.dump({'strategy': {}}, f)
            
            found_file = worker._find_result_file(
                worker_config.result_dir,
                'BTC/USDT'
            )
            
            assert found_file == result_file
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_find_result_file_ignores_meta(self):
        """Test that .meta.json files are ignored"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create only a meta file
            meta_file = os.path.join(
                worker_config.result_dir,
                'result.meta.json'
            )
            with open(meta_file, 'w') as f:
                json.dump({}, f)
            
            found_file = worker._find_result_file(
                worker_config.result_dir,
                'BTC/USDT'
            )
            
            assert found_file is None
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_find_result_file_not_found(self):
        """Test when no result file exists"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            found_file = worker._find_result_file(
                worker_config.result_dir,
                'BTC/USDT'
            )
            
            assert found_file is None
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_parse_result_stats_success(self):
        """Test parsing statistics from result file"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            result_file = create_mock_result_file(
                worker_config.result_dir,
                'BTC/USDT',
                'TestStrategy',
                trades_count=25,
                profit_total=0.15
            )
            
            trades_count, profit_ratio = worker._parse_result_stats(
                result_file,
                'TestStrategy'
            )
            
            assert trades_count == 25
            assert profit_ratio == 0.15
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_parse_result_stats_invalid_json(self):
        """Test parsing with invalid JSON file"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create invalid JSON file
            result_file = os.path.join(
                worker_config.result_dir,
                'invalid.json'
            )
            with open(result_file, 'w') as f:
                f.write('not valid json')
            
            trades_count, profit_ratio = worker._parse_result_stats(
                result_file,
                'TestStrategy'
            )
            
            # Should return defaults on error
            assert trades_count == 0
            assert profit_ratio == 0.0
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_parse_result_stats_missing_strategy(self):
        """Test parsing when strategy not in results"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create result file with different strategy
            result_file = create_mock_result_file(
                worker_config.result_dir,
                'BTC/USDT',
                'OtherStrategy'
            )
            
            trades_count, profit_ratio = worker._parse_result_stats(
                result_file,
                'TestStrategy'
            )
            
            # Should return defaults when strategy not found
            assert trades_count == 0
            assert profit_ratio == 0.0
        finally:
            cleanup_temp_dir(temp_dir)


class TestErrorExtraction:
    """Tests for error message extraction"""
    
    def test_extract_error_from_stderr(self):
        """Test extracting error from stderr"""
        worker = BacktestWorker(strategy='TestStrategy')
        
        stderr = "Some warning\nError: Configuration file not found\nMore text"
        stdout = ""
        
        error = worker._extract_error_message(stderr, stdout)
        
        assert 'Configuration file not found' in error
    
    def test_extract_error_from_stdout_fallback(self):
        """Test extracting error from stdout when stderr is empty"""
        worker = BacktestWorker(strategy='TestStrategy')
        
        stderr = ""
        stdout = "Processing...\nFailed to load strategy"
        
        error = worker._extract_error_message(stderr, stdout)
        
        assert 'Failed to load strategy' in error
    
    def test_extract_error_empty_output(self):
        """Test error extraction with empty output"""
        worker = BacktestWorker(strategy='TestStrategy')
        
        error = worker._extract_error_message("", "")
        
        assert 'no error message' in error.lower()


class TestRunBacktest:
    """Tests for running backtest with mocked subprocess"""
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_success(self, mock_run):
        """Test successful backtest run"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Create mock result file
            create_mock_result_file(
                worker_config.result_dir,
                'BTC/USDT',
                'TestStrategy',
                trades_count=15,
                profit_total=0.08
            )
            
            # Mock successful subprocess run
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='Backtest complete',
                stderr=''
            )
            
            result = worker.run(worker_config)
            
            assert result.success is True
            assert result.pair == 'BTC/USDT'
            assert result.trades_count == 15
            assert result.profit_ratio == 0.08
            assert result.result_file is not None
            assert result.error_message is None
            assert result.duration >= 0  # Duration can be 0 with mocked subprocess
        finally:
            cleanup_temp_dir(temp_dir)
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_failure_nonzero_return(self, mock_run):
        """Test backtest run with non-zero return code"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Mock failed subprocess run
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='Error: Strategy not found'
            )
            
            result = worker.run(worker_config)
            
            assert result.success is False
            assert result.pair == 'BTC/USDT'
            assert 'Strategy not found' in result.error_message
            assert result.result_file is None
        finally:
            cleanup_temp_dir(temp_dir)
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_timeout(self, mock_run):
        """Test backtest run with timeout"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy', timeout=10)
            
            # Mock timeout
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd='freqtrade',
                timeout=10
            )
            
            result = worker.run(worker_config)
            
            assert result.success is False
            assert 'timed out' in result.error_message.lower()
        finally:
            cleanup_temp_dir(temp_dir)
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_unexpected_exception(self, mock_run):
        """Test backtest run with unexpected exception"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Mock unexpected exception
            mock_run.side_effect = Exception('Unexpected error occurred')
            
            result = worker.run(worker_config)
            
            assert result.success is False
            assert 'Unexpected error' in result.error_message
        finally:
            cleanup_temp_dir(temp_dir)
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_no_result_file(self, mock_run):
        """Test backtest run when result file is not created"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            worker = BacktestWorker(strategy='TestStrategy')
            
            # Mock successful run but no result file
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='Backtest complete',
                stderr=''
            )
            
            result = worker.run(worker_config)
            
            assert result.success is False
            assert 'Result file not found' in result.error_message
        finally:
            cleanup_temp_dir(temp_dir)


class TestRunBacktestTask:
    """Tests for the convenience function run_backtest_task"""
    
    @patch('parallel_backtest.worker.subprocess.run')
    def test_run_backtest_task_function(self, mock_run):
        """Test the run_backtest_task convenience function"""
        temp_dir = create_temp_dir()
        try:
            worker_config = create_worker_config(temp_dir=temp_dir)
            
            # Create mock result file
            create_mock_result_file(
                worker_config.result_dir,
                'BTC/USDT',
                'TestStrategy'
            )
            
            # Mock successful subprocess run
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='',
                stderr=''
            )
            
            result = run_backtest_task(
                worker_config=worker_config,
                strategy='TestStrategy',
                timerange='20240101-20241231',
                extra_args=['--cache', 'none'],
                timeout=1800
            )
            
            assert isinstance(result, WorkerResult)
            assert result.pair == 'BTC/USDT'
        finally:
            cleanup_temp_dir(temp_dir)


class TestWorkerResultDataclass:
    """Tests for WorkerResult dataclass"""
    
    def test_worker_result_success(self):
        """Test creating successful WorkerResult"""
        result = WorkerResult(
            pair='BTC/USDT',
            success=True,
            result_file='/path/to/result.json',
            duration=120.5,
            trades_count=50,
            profit_ratio=0.12
        )
        
        assert result.pair == 'BTC/USDT'
        assert result.success is True
        assert result.result_file == '/path/to/result.json'
        assert result.error_message is None
        assert result.duration == 120.5
        assert result.trades_count == 50
        assert result.profit_ratio == 0.12
    
    def test_worker_result_failure(self):
        """Test creating failed WorkerResult"""
        result = WorkerResult(
            pair='ETH/USDT',
            success=False,
            error_message='Strategy not found',
            duration=5.0
        )
        
        assert result.pair == 'ETH/USDT'
        assert result.success is False
        assert result.result_file is None
        assert result.error_message == 'Strategy not found'
        assert result.trades_count == 0
        assert result.profit_ratio == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
