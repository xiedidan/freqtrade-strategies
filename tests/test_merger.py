"""
Property-based tests for ResultMerger.

Tests the result merging functionality using Hypothesis
for property-based testing.

**Property 4: 交易记录完整性**
**Property 5: 统计数据一致性**
**Property 6: 部分失败容错**
**Validates: Requirements 3.1, 3.2, 3.3, 3.5**
"""

import pytest
import os
import json
import tempfile
import shutil
from typing import List, Dict, Any
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from parallel_backtest.merger import ResultMerger
from parallel_backtest.models import WorkerResult, MergedResult


# Custom strategies for generating test data
@st.composite
def valid_pair_strategy(draw):
    """Generate valid trading pair strings like BTC/USDT"""
    base_currencies = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'DOT']
    quote_currencies = ['USDT', 'BUSD', 'USD']
    
    base = draw(st.sampled_from(base_currencies))
    quote = draw(st.sampled_from(quote_currencies))
    assume(base != quote)
    return f"{base}/{quote}"


@st.composite
def trade_strategy(draw, pair: str = None):
    """Generate a valid trade record"""
    if pair is None:
        pair = draw(valid_pair_strategy())
    
    # Generate timestamps (ensure close > open)
    open_ts = draw(st.integers(min_value=1700000000000, max_value=1800000000000))
    duration_minutes = draw(st.integers(min_value=1, max_value=60))
    close_ts = open_ts + duration_minutes * 60 * 1000
    
    # Generate profit (can be positive, negative, or zero)
    profit_ratio = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False))
    stake_amount = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False))
    profit_abs = stake_amount * profit_ratio
    
    return {
        "pair": pair,
        "stake_amount": stake_amount,
        "amount": draw(st.floats(min_value=0.001, max_value=10.0, allow_nan=False)),
        "open_date": "2024-01-01 00:00:00+00:00",
        "close_date": "2024-01-01 01:00:00+00:00",
        "open_rate": draw(st.floats(min_value=1.0, max_value=100000.0, allow_nan=False)),
        "close_rate": draw(st.floats(min_value=1.0, max_value=100000.0, allow_nan=False)),
        "fee_open": 0.001,
        "fee_close": 0.001,
        "trade_duration": duration_minutes,
        "profit_ratio": profit_ratio,
        "profit_abs": profit_abs,
        "exit_reason": draw(st.sampled_from(["roi", "stop_loss", "trailing_stop", "exit_signal"])),
        "is_open": False,
        "enter_tag": draw(st.sampled_from(["", "signal_1", "signal_2"])),
        "leverage": 1.0,
        "is_short": draw(st.booleans()),
        "open_timestamp": open_ts,
        "close_timestamp": close_ts
    }


@st.composite
def trades_list_strategy(draw, pair: str = None, min_trades: int = 0, max_trades: int = 20):
    """Generate a list of trades for a single pair"""
    num_trades = draw(st.integers(min_value=min_trades, max_value=max_trades))
    trades = []
    for _ in range(num_trades):
        trade = draw(trade_strategy(pair=pair))
        trades.append(trade)
    return trades


@st.composite
def worker_result_strategy(draw, success: bool = None):
    """Generate a WorkerResult with optional success override"""
    pair = draw(valid_pair_strategy())
    
    if success is None:
        success = draw(st.booleans())
    
    if success:
        trades = draw(trades_list_strategy(pair=pair, min_trades=1, max_trades=10))
        trades_count = len(trades)
        profit_ratio = sum(t["profit_ratio"] for t in trades) / len(trades) if trades else 0
        
        return WorkerResult(
            pair=pair,
            success=True,
            result_file=None,  # Will be set when creating temp files
            error_message=None,
            duration=draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False)),
            trades_count=trades_count,
            profit_ratio=profit_ratio
        ), trades
    else:
        return WorkerResult(
            pair=pair,
            success=False,
            result_file=None,
            error_message="Simulated failure",
            duration=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False)),
            trades_count=0,
            profit_ratio=0.0
        ), []


def create_temp_result_file(trades: List[Dict], strategy_name: str, pair: str) -> str:
    """Create a temporary result file with trades data"""
    fd, path = tempfile.mkstemp(suffix='.json')
    
    # Build per-pair results
    wins = sum(1 for t in trades if t.get("profit_ratio", 0) > 0)
    losses = sum(1 for t in trades if t.get("profit_ratio", 0) < 0)
    draws = len(trades) - wins - losses
    profit_total_abs = sum(t.get("profit_abs", 0) for t in trades)
    
    results_per_pair = [{
        "key": pair,
        "trades": len(trades),
        "profit_mean": sum(t.get("profit_ratio", 0) for t in trades) / len(trades) if trades else 0,
        "profit_total_abs": profit_total_abs,
        "profit_total": profit_total_abs / 1000,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate": wins / len(trades) if trades else 0
    }]
    
    result_data = {
        "strategy": {
            strategy_name: {
                "trades": trades,
                "locks": [],
                "results_per_pair": results_per_pair,
                "total_trades": len(trades),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "profit_total_abs": profit_total_abs,
                "starting_balance": 1000,
                "stake_currency": "USDT"
            }
        }
    }
    
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(result_data, f)
    except:
        os.close(fd)
        raise
    
    return path


def cleanup_temp_file(path: str) -> None:
    """Safely cleanup a temporary file"""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass


class TestResultMergerBasic:
    """Basic unit tests for ResultMerger"""
    
    def test_init_default_values(self):
        """Test default initialization values"""
        merger = ResultMerger()
        assert merger.starting_balance == 1000.0
        assert merger.stake_currency == "USDT"
    
    def test_init_custom_values(self):
        """Test custom initialization values"""
        merger = ResultMerger(starting_balance=5000.0, stake_currency="BTC")
        assert merger.starting_balance == 5000.0
        assert merger.stake_currency == "BTC"
    
    def test_merge_empty_results(self):
        """Test merging empty results list"""
        merger = ResultMerger()
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            result = merger.merge([], output_path, "TestStrategy")
            
            assert result.total_pairs == 0
            assert result.successful_pairs == 0
            assert result.failed_pairs == 0
            assert result.total_trades == 0
        finally:
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))
    
    def test_merge_single_successful_result(self):
        """Test merging a single successful result"""
        merger = ResultMerger()
        
        # Create test trades
        trades = [
            {"pair": "BTC/USDT", "profit_ratio": 0.05, "profit_abs": 50.0,
             "open_timestamp": 1700000000000, "close_timestamp": 1700001000000,
             "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
             "exit_reason": "roi", "enter_tag": ""}
        ]
        
        result_file = create_temp_result_file(trades, "TestStrategy", "BTC/USDT")
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            worker_result = WorkerResult(
                pair="BTC/USDT",
                success=True,
                result_file=result_file,
                trades_count=1,
                profit_ratio=0.05
            )
            
            result = merger.merge([worker_result], output_path, "TestStrategy")
            
            assert result.total_pairs == 1
            assert result.successful_pairs == 1
            assert result.failed_pairs == 0
            assert result.total_trades == 1
            assert os.path.exists(output_path)
        finally:
            cleanup_temp_file(result_file)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))


class TestPropertyTradeRecordCompleteness:
    """
    Property 4: 交易记录完整性
    
    For any list of successful backtest results, the merged trades list length
    should equal the sum of all individual result trades list lengths.
    
    **Validates: Requirements 3.1, 3.2**
    """
    
    @given(
        num_pairs=st.integers(min_value=1, max_value=5),
        trades_per_pair=st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_trade_record_completeness(self, num_pairs: int, trades_per_pair: List[int]):
        """
        **Feature: parallel-backtest-tool, Property 4: 交易记录完整性**
        
        For any list of successful backtest results, the merged trades list
        length should equal the sum of all individual result trades list lengths.
        
        **Validates: Requirements 3.1, 3.2**
        """
        # Ensure we have enough trades_per_pair entries
        while len(trades_per_pair) < num_pairs:
            trades_per_pair.append(1)
        trades_per_pair = trades_per_pair[:num_pairs]
        
        merger = ResultMerger()
        strategy_name = "TestStrategy"
        
        # Generate unique pairs
        base_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
        pairs = base_pairs[:num_pairs]
        
        # Create worker results with trades
        worker_results = []
        all_trades_count = 0
        temp_files = []
        
        for i, pair in enumerate(pairs):
            num_trades = trades_per_pair[i]
            trades = []
            
            for j in range(num_trades):
                trade = {
                    "pair": pair,
                    "profit_ratio": 0.01 * (j + 1),
                    "profit_abs": 10.0 * (j + 1),
                    "open_timestamp": 1700000000000 + i * 1000000 + j * 1000,
                    "close_timestamp": 1700000100000 + i * 1000000 + j * 1000,
                    "trade_duration": 15,
                    "is_short": False,
                    "stake_amount": 100.0,
                    "exit_reason": "roi",
                    "enter_tag": ""
                }
                trades.append(trade)
            
            all_trades_count += len(trades)
            
            result_file = create_temp_result_file(trades, strategy_name, pair)
            temp_files.append(result_file)
            
            worker_results.append(WorkerResult(
                pair=pair,
                success=True,
                result_file=result_file,
                trades_count=len(trades),
                profit_ratio=0.05
            ))
        
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            result = merger.merge(worker_results, output_path, strategy_name)
            
            # Property: merged trades count equals sum of individual trades
            assert result.total_trades == all_trades_count, \
                f"Expected {all_trades_count} trades, got {result.total_trades}"
            
            # Verify by reading the output file
            with open(output_path, 'r') as f:
                merged_data = json.load(f)
            
            merged_trades = merged_data["strategy"][strategy_name]["trades"]
            assert len(merged_trades) == all_trades_count, \
                f"Output file has {len(merged_trades)} trades, expected {all_trades_count}"
            
        finally:
            for f in temp_files:
                cleanup_temp_file(f)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))


class TestPropertyStatisticsConsistency:
    """
    Property 5: 统计数据一致性
    
    For any merged result, total_trades should equal the length of the trades
    list, and wins + losses + draws should equal total_trades.
    
    **Validates: Requirements 3.3**
    """
    
    @given(
        num_pairs=st.integers(min_value=1, max_value=4),
        trades_per_pair=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_statistics_consistency(self, num_pairs: int, trades_per_pair: int):
        """
        **Feature: parallel-backtest-tool, Property 5: 统计数据一致性**
        
        For any merged result, total_trades should equal the length of the
        trades list, and wins + losses + draws should equal total_trades.
        
        **Validates: Requirements 3.3**
        """
        merger = ResultMerger()
        strategy_name = "TestStrategy"
        
        # Generate unique pairs
        base_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        pairs = base_pairs[:num_pairs]
        
        # Create worker results with trades
        worker_results = []
        temp_files = []
        
        for i, pair in enumerate(pairs):
            trades = []
            
            for j in range(trades_per_pair):
                # Vary profit to get wins, losses, and draws
                if j % 3 == 0:
                    profit_ratio = 0.05  # Win
                elif j % 3 == 1:
                    profit_ratio = -0.03  # Loss
                else:
                    profit_ratio = 0.0  # Draw
                
                trade = {
                    "pair": pair,
                    "profit_ratio": profit_ratio,
                    "profit_abs": 100.0 * profit_ratio,
                    "open_timestamp": 1700000000000 + i * 1000000 + j * 1000,
                    "close_timestamp": 1700000100000 + i * 1000000 + j * 1000,
                    "trade_duration": 15,
                    "is_short": False,
                    "stake_amount": 100.0,
                    "exit_reason": "roi",
                    "enter_tag": ""
                }
                trades.append(trade)
            
            result_file = create_temp_result_file(trades, strategy_name, pair)
            temp_files.append(result_file)
            
            worker_results.append(WorkerResult(
                pair=pair,
                success=True,
                result_file=result_file,
                trades_count=len(trades),
                profit_ratio=0.01
            ))
        
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            result = merger.merge(worker_results, output_path, strategy_name)
            
            # Read the output file to verify statistics
            with open(output_path, 'r') as f:
                merged_data = json.load(f)
            
            strategy_data = merged_data["strategy"][strategy_name]
            trades = strategy_data["trades"]
            
            # Property 1: total_trades equals trades list length
            assert strategy_data["total_trades"] == len(trades), \
                f"total_trades ({strategy_data['total_trades']}) != len(trades) ({len(trades)})"
            
            # Property 2: wins + losses + draws equals total_trades
            wins = strategy_data["wins"]
            losses = strategy_data["losses"]
            draws = strategy_data["draws"]
            total = strategy_data["total_trades"]
            
            assert wins + losses + draws == total, \
                f"wins ({wins}) + losses ({losses}) + draws ({draws}) != total ({total})"
            
            # Property 3: winrate is correctly calculated
            expected_winrate = wins / total if total > 0 else 0
            assert abs(strategy_data["winrate"] - expected_winrate) < 0.0001, \
                f"winrate ({strategy_data['winrate']}) != expected ({expected_winrate})"
            
        finally:
            for f in temp_files:
                cleanup_temp_file(f)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))


class TestPropertyPartialFailureTolerance:
    """
    Property 6: 部分失败容错
    
    For any result list containing failures, the merge operation should
    succeed, and merged trades should only contain trades from successful results.
    
    **Validates: Requirements 3.5, 4.1**
    """
    
    @given(
        num_successful=st.integers(min_value=1, max_value=3),
        num_failed=st.integers(min_value=1, max_value=3),
        trades_per_pair=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_partial_failure_tolerance(
        self, 
        num_successful: int, 
        num_failed: int, 
        trades_per_pair: int
    ):
        """
        **Feature: parallel-backtest-tool, Property 6: 部分失败容错**
        
        For any result list containing failures, the merge operation should
        succeed, and merged trades should only contain trades from successful
        results.
        
        **Validates: Requirements 3.5, 4.1**
        """
        merger = ResultMerger()
        strategy_name = "TestStrategy"
        
        # Generate unique pairs for successful and failed
        all_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT"]
        successful_pairs = all_pairs[:num_successful]
        failed_pairs = all_pairs[num_successful:num_successful + num_failed]
        
        worker_results = []
        temp_files = []
        expected_trades_count = 0
        
        # Create successful results
        for i, pair in enumerate(successful_pairs):
            trades = []
            for j in range(trades_per_pair):
                trade = {
                    "pair": pair,
                    "profit_ratio": 0.02,
                    "profit_abs": 20.0,
                    "open_timestamp": 1700000000000 + i * 1000000 + j * 1000,
                    "close_timestamp": 1700000100000 + i * 1000000 + j * 1000,
                    "trade_duration": 15,
                    "is_short": False,
                    "stake_amount": 100.0,
                    "exit_reason": "roi",
                    "enter_tag": ""
                }
                trades.append(trade)
            
            expected_trades_count += len(trades)
            
            result_file = create_temp_result_file(trades, strategy_name, pair)
            temp_files.append(result_file)
            
            worker_results.append(WorkerResult(
                pair=pair,
                success=True,
                result_file=result_file,
                trades_count=len(trades),
                profit_ratio=0.02
            ))
        
        # Create failed results
        for pair in failed_pairs:
            worker_results.append(WorkerResult(
                pair=pair,
                success=False,
                result_file=None,
                error_message="Simulated failure",
                trades_count=0,
                profit_ratio=0.0
            ))
        
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            # Merge should succeed despite failures
            result = merger.merge(worker_results, output_path, strategy_name)
            
            # Property 1: Merge operation succeeds
            assert result is not None
            assert os.path.exists(output_path)
            
            # Property 2: Correct counts
            assert result.successful_pairs == num_successful
            assert result.failed_pairs == num_failed
            assert result.total_pairs == num_successful + num_failed
            
            # Property 3: Only successful trades are included
            assert result.total_trades == expected_trades_count
            
            # Property 4: Failed pairs are recorded
            assert len(result.failed_pair_names) == num_failed
            for pair in failed_pairs:
                assert pair in result.failed_pair_names
            
            # Property 5: Verify output file content
            with open(output_path, 'r') as f:
                merged_data = json.load(f)
            
            merged_trades = merged_data["strategy"][strategy_name]["trades"]
            
            # All trades should be from successful pairs only
            for trade in merged_trades:
                assert trade["pair"] in successful_pairs, \
                    f"Trade from failed pair {trade['pair']} found in merged result"
            
        finally:
            for f in temp_files:
                cleanup_temp_file(f)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))


class TestTradesSorting:
    """Test that trades are sorted by timestamp after merging"""
    
    def test_trades_sorted_by_timestamp(self):
        """Test that merged trades are sorted by open_timestamp"""
        merger = ResultMerger()
        strategy_name = "TestStrategy"
        
        # Create trades with different timestamps
        trades1 = [
            {"pair": "BTC/USDT", "profit_ratio": 0.01, "profit_abs": 10.0,
             "open_timestamp": 1700000300000, "close_timestamp": 1700000400000,
             "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
             "exit_reason": "roi", "enter_tag": ""}
        ]
        trades2 = [
            {"pair": "ETH/USDT", "profit_ratio": 0.02, "profit_abs": 20.0,
             "open_timestamp": 1700000100000, "close_timestamp": 1700000200000,
             "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
             "exit_reason": "roi", "enter_tag": ""}
        ]
        
        file1 = create_temp_result_file(trades1, strategy_name, "BTC/USDT")
        file2 = create_temp_result_file(trades2, strategy_name, "ETH/USDT")
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            worker_results = [
                WorkerResult(pair="BTC/USDT", success=True, result_file=file1,
                           trades_count=1, profit_ratio=0.01),
                WorkerResult(pair="ETH/USDT", success=True, result_file=file2,
                           trades_count=1, profit_ratio=0.02)
            ]
            
            merger.merge(worker_results, output_path, strategy_name)
            
            with open(output_path, 'r') as f:
                merged_data = json.load(f)
            
            trades = merged_data["strategy"][strategy_name]["trades"]
            
            # Verify trades are sorted by open_timestamp
            assert len(trades) == 2
            assert trades[0]["open_timestamp"] < trades[1]["open_timestamp"]
            assert trades[0]["pair"] == "ETH/USDT"  # Earlier timestamp
            assert trades[1]["pair"] == "BTC/USDT"  # Later timestamp
            
        finally:
            cleanup_temp_file(file1)
            cleanup_temp_file(file2)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))


class TestOutputFileFormats:
    """Test different output file formats"""
    
    def test_json_output(self):
        """Test JSON output format"""
        merger = ResultMerger()
        
        trades = [{"pair": "BTC/USDT", "profit_ratio": 0.01, "profit_abs": 10.0,
                  "open_timestamp": 1700000000000, "close_timestamp": 1700000100000,
                  "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
                  "exit_reason": "roi", "enter_tag": ""}]
        
        result_file = create_temp_result_file(trades, "TestStrategy", "BTC/USDT")
        output_path = tempfile.mktemp(suffix='.json')
        
        try:
            worker_result = WorkerResult(pair="BTC/USDT", success=True,
                                        result_file=result_file, trades_count=1)
            
            merger.merge([worker_result], output_path, "TestStrategy")
            
            assert os.path.exists(output_path)
            assert output_path.endswith('.json')
            
            # Verify it's valid JSON
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert "strategy" in data
            
        finally:
            cleanup_temp_file(result_file)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.json', '.meta.json'))
    
    def test_zip_output(self):
        """Test ZIP output format"""
        import zipfile
        
        merger = ResultMerger()
        
        trades = [{"pair": "BTC/USDT", "profit_ratio": 0.01, "profit_abs": 10.0,
                  "open_timestamp": 1700000000000, "close_timestamp": 1700000100000,
                  "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
                  "exit_reason": "roi", "enter_tag": ""}]
        
        result_file = create_temp_result_file(trades, "TestStrategy", "BTC/USDT")
        output_path = tempfile.mktemp(suffix='.zip')
        
        try:
            worker_result = WorkerResult(pair="BTC/USDT", success=True,
                                        result_file=result_file, trades_count=1)
            
            merger.merge([worker_result], output_path, "TestStrategy")
            
            assert os.path.exists(output_path)
            
            # Verify it's a valid ZIP
            with zipfile.ZipFile(output_path, 'r') as zf:
                names = zf.namelist()
                assert len(names) == 1
                assert names[0].endswith('.json')
                
                # Verify content is valid JSON
                content = zf.read(names[0])
                data = json.loads(content)
                assert "strategy" in data
            
        finally:
            cleanup_temp_file(result_file)
            cleanup_temp_file(output_path)
            cleanup_temp_file(output_path.replace('.zip', '.meta.json'))
    
    def test_meta_json_created(self):
        """Test that .meta.json file is created"""
        merger = ResultMerger()
        
        trades = [{"pair": "BTC/USDT", "profit_ratio": 0.01, "profit_abs": 10.0,
                  "open_timestamp": 1700000000000, "close_timestamp": 1700000100000,
                  "trade_duration": 15, "is_short": False, "stake_amount": 100.0,
                  "exit_reason": "roi", "enter_tag": ""}]
        
        result_file = create_temp_result_file(trades, "TestStrategy", "BTC/USDT")
        output_path = tempfile.mktemp(suffix='.json')
        meta_path = output_path.replace('.json', '.meta.json')
        
        try:
            worker_result = WorkerResult(pair="BTC/USDT", success=True,
                                        result_file=result_file, trades_count=1)
            
            merger.merge([worker_result], output_path, "TestStrategy")
            
            assert os.path.exists(meta_path)
            
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            assert "TestStrategy" in meta_data
            assert "run_id" in meta_data["TestStrategy"]
            assert "backtest_start_time" in meta_data["TestStrategy"]
            
        finally:
            cleanup_temp_file(result_file)
            cleanup_temp_file(output_path)
            cleanup_temp_file(meta_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
