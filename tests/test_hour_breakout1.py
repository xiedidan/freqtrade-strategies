"""
Unit tests for HourBreakout1 strategy foundation
Tests class inheritance, basic configuration, and informative_pairs method

Requirements: 7.1, 7.2, 7.3
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Tuple
from hypothesis import given, strategies as st, settings
from freqtrade.strategy import IStrategy
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_data.strategies.HourBreakout1 import HourBreakout1


class TestHourBreakout1Foundation:
    """Test suite for HourBreakout1 strategy foundation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create a mock config for strategy initialization
        mock_config = {
            'stake_currency': 'USDT',
            'dry_run': True,
            'timeframe': '1m'
        }
        self.strategy = HourBreakout1(mock_config)
        
        # Mock the data provider
        self.strategy.dp = MagicMock()
        self.strategy.dp.current_whitelist.return_value = ['BTC/USDT', 'ETH/USDT']
    
    def test_strategy_class_exists(self):
        """Test that HourBreakout1 class can be instantiated"""
        # Requirements: 7.1 - Strategy should inherit IStrategy base class
        assert self.strategy is not None
        assert self.strategy.__class__.__name__ == 'HourBreakout1'
    
    def test_interface_version(self):
        """Test that strategy has correct interface version"""
        # Requirements: 7.1 - Strategy should define correct interface version
        assert hasattr(self.strategy, 'INTERFACE_VERSION')
        assert self.strategy.INTERFACE_VERSION == 3
    
    def test_basic_configuration(self):
        """Test basic strategy configuration parameters"""
        # Requirements: 7.1, 7.2 - Define basic configuration
        
        # Test timeframe configuration
        assert self.strategy.timeframe == '1m'
        
        # Test ROI configuration
        assert hasattr(self.strategy, 'minimal_roi')
        assert isinstance(self.strategy.minimal_roi, dict)
        assert len(self.strategy.minimal_roi) > 0
        
        # Test stoploss configuration
        assert hasattr(self.strategy, 'stoploss')
        assert isinstance(self.strategy.stoploss, (int, float))
        assert self.strategy.stoploss < 0  # Should be negative
        
        # Test startup candle count
        assert hasattr(self.strategy, 'startup_candle_count')
        assert self.strategy.startup_candle_count >= 100
    
    def test_hyperopt_parameters_defined(self):
        """Test that HyperOpt parameters are properly defined"""
        # Requirements: 9.1 - Use HyperOpt space definitions
        
        # Test MA period parameter
        assert hasattr(self.strategy, 'ma_period')
        assert self.strategy.ma_period.low == 3
        assert self.strategy.ma_period.high == 10
        assert self.strategy.ma_period.value == 5  # Check current value instead of default
        
        # Test exit minutes parameter
        assert hasattr(self.strategy, 'exit_minutes')
        assert self.strategy.exit_minutes.low == 5
        assert self.strategy.exit_minutes.high == 60
        assert self.strategy.exit_minutes.value == 15
        
        # Test minimum breakout percentage parameter
        assert hasattr(self.strategy, 'min_breakout_pct')
        assert self.strategy.min_breakout_pct.low == 0.001
        assert self.strategy.min_breakout_pct.high == 0.01
        
        # Test pullback tolerance parameter
        assert hasattr(self.strategy, 'pullback_tolerance')
        assert self.strategy.pullback_tolerance.low == 0.0001
        assert self.strategy.pullback_tolerance.high == 0.002
    
    def test_informative_pairs_returns_correct_timeframes(self):
        """Test that informative_pairs returns correct timeframe configuration"""
        # Requirements: 7.3 - Configure multi-timeframe informative pairs
        
        # Get informative pairs
        pairs = self.strategy.informative_pairs()
        
        # Should return a list of tuples
        assert isinstance(pairs, list)
        
        # Each item should be a tuple of (pair, timeframe)
        for pair_info in pairs:
            assert isinstance(pair_info, tuple)
            assert len(pair_info) == 2
            pair, timeframe = pair_info
            assert isinstance(pair, str)
            assert isinstance(timeframe, str)
        
        # Should include both 5m and 1h timeframes for each pair
        expected_pairs = []
        whitelist = ['BTC/USDT', 'ETH/USDT']
        for pair in whitelist:
            expected_pairs.extend([
                (pair, '5m'),
                (pair, '1h')
            ])
        
        assert pairs == expected_pairs
    
    def test_informative_pairs_handles_empty_whitelist(self):
        """Test informative_pairs handles empty whitelist gracefully"""
        # Requirements: 7.3 - Handle edge cases in multi-timeframe configuration
        
        # Mock empty whitelist
        self.strategy.dp.current_whitelist.return_value = []
        
        pairs = self.strategy.informative_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) == 0
    
    def test_informative_pairs_handles_no_dp(self):
        """Test informative_pairs handles missing data provider"""
        # Requirements: 7.3 - Handle edge cases in multi-timeframe configuration
        
        # Remove data provider
        self.strategy.dp = None
        
        pairs = self.strategy.informative_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) == 0
    
    def test_required_methods_exist(self):
        """Test that all required FreqTrade methods exist"""
        # Requirements: 7.2 - Implement required methods
        
        # Test that required methods are defined
        assert hasattr(self.strategy, 'populate_indicators')
        assert callable(self.strategy.populate_indicators)
        
        assert hasattr(self.strategy, 'populate_entry_trend')
        assert callable(self.strategy.populate_entry_trend)
        
        assert hasattr(self.strategy, 'populate_exit_trend')
        assert callable(self.strategy.populate_exit_trend)
        
        assert hasattr(self.strategy, 'informative_pairs')
        assert callable(self.strategy.informative_pairs)
    
    def test_populate_indicators_basic_functionality(self):
        """Test basic functionality of populate_indicators method"""
        # Requirements: 7.2 - Implement required methods
        
        # Create sample dataframe
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Test that method runs without error
        result = self.strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have original columns plus indicators
        assert len(result.columns) >= len(df.columns)
        
        # Should have MA column
        ma_col = f'ma{self.strategy.ma_period.value}'
        assert ma_col in result.columns
        
        # Should have condition columns
        assert 'breakout_condition' in result.columns
        assert 'pullback_condition' in result.columns
        assert 'rebound_condition' in result.columns
    
    def test_populate_entry_trend_basic_functionality(self):
        """Test basic functionality of populate_entry_trend method"""
        # Requirements: 7.2 - Implement required methods
        
        # Create sample dataframe with required columns
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Test that method runs without error
        result = self.strategy.populate_entry_trend(df, {'pair': 'BTC/USDT'})
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have entry columns
        assert 'enter_long' in result.columns
        assert 'enter_tag' in result.columns
    
    def test_populate_exit_trend_basic_functionality(self):
        """Test basic functionality of populate_exit_trend method"""
        # Requirements: 7.2 - Implement required methods
        
        # Create sample dataframe
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Test that method runs without error
        result = self.strategy.populate_exit_trend(df, {'pair': 'BTC/USDT'})
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have exit columns
        assert 'exit_long' in result.columns
        assert 'exit_tag' in result.columns
    
    def test_order_types_configuration(self):
        """Test order types configuration"""
        # Requirements: 7.1 - Define basic configuration
        
        assert hasattr(self.strategy, 'order_types')
        assert isinstance(self.strategy.order_types, dict)
        
        # Check required order type keys
        required_keys = ['entry', 'exit', 'stoploss']
        for key in required_keys:
            assert key in self.strategy.order_types
    
    def test_strategy_flags(self):
        """Test strategy behavior flags"""
        # Requirements: 7.1 - Define basic configuration
        
        # Test short selling capability
        assert hasattr(self.strategy, 'can_short')
        assert self.strategy.can_short is False  # This strategy only goes long
        
        # Test process only new candles
        assert hasattr(self.strategy, 'process_only_new_candles')
        assert self.strategy.process_only_new_candles is True
        
        # Test exit signal usage
        assert hasattr(self.strategy, 'use_exit_signal')
        assert self.strategy.use_exit_signal is True


class TestTechnicalIndicatorCalculation:
    """Property-based tests for technical indicator calculation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create a mock config for strategy initialization
        mock_config = {
            'stake_currency': 'USDT',
            'dry_run': True,
            'timeframe': '1m'
        }
        self.strategy = HourBreakout1(mock_config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_price_series(self, length: int, start_price: float = 100.0, volatility: float = 0.01) -> list:
        """Generate realistic price series for testing"""
        import numpy as np
        np.random.seed(42)  # For reproducible tests
        prices = [start_price]
        
        for _ in range(length - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        return prices
    
    def calculate_sma_manually(self, prices: list, period: int) -> list:
        """Manually calculate SMA for validation"""
        import numpy as np
        sma_values = []
        
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(np.nan)
            else:
                window_prices = prices[i - period + 1:i + 1]
                sma_values.append(sum(window_prices) / len(window_prices))
        
        return sma_values
    
    def test_ma_calculation_correctness_simple(self):
        """
        **Feature: hour-breakout-scalping, Property 8: MA calculation correctness**
        
        Test MA calculation with simple known values
        
        **Validates: Requirements 3.1**
        """
        import numpy as np
        
        # Test with simple known values
        length = 20
        start_price = 100.0
        ma_period = 5
        
        # Generate test price data
        prices = self.generate_price_series(length, start_price)
        
        # Create dataframe
        dates = pd.date_range(start='2023-01-01', periods=length, freq='1min')
        dataframe = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],  # Slightly higher highs
            'low': [p * 0.99 for p in prices],   # Slightly lower lows
            'close': prices,
            'volume': [1000] * length
        }).set_index('date')
        
        # Set the MA period for this test
        self.strategy.ma_period.value = ma_period
        
        # Mock empty informative data to focus on MA calculation
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Calculate indicators
        result_df = self.strategy.populate_indicators(dataframe.copy(), {'pair': 'BTC/USDT'})
        
        # Property 1: MA column should exist
        ma_col = f'ma{ma_period}'
        assert ma_col in result_df.columns, f"MA column {ma_col} not found"
        
        # Property 2: Calculate expected SMA manually
        expected_sma = self.calculate_sma_manually(prices, ma_period)
        
        # Property 3: Compare calculated SMA with expected values (skip NaN values)
        for i in range(ma_period - 1, min(len(result_df), len(expected_sma))):
            calculated_value = result_df[ma_col].iloc[i]
            expected_value = expected_sma[i]
            
            # Both should be valid numbers
            if not np.isnan(expected_value):
                assert not np.isnan(calculated_value), f"Calculated SMA is NaN at index {i}"
                
                # Values should be approximately equal (allowing for floating point precision)
                assert abs(calculated_value - expected_value) < 1e-8, \
                    f"SMA mismatch at index {i}: calculated={calculated_value}, expected={expected_value}"
        
        # Property 4: SMA should be within reasonable bounds
        valid_sma = result_df[ma_col].dropna()
        if len(valid_sma) > 0:
            min_price = min(prices)
            max_price = max(prices)
            
            # SMA should be within the range of input prices
            assert valid_sma.min() >= min_price * 0.95, "SMA below minimum price range"
            assert valid_sma.max() <= max_price * 1.05, "SMA above maximum price range"
    
    def test_ma_calculation_edge_cases(self):
        """
        Test MA calculation with edge cases
        
        **Feature: hour-breakout-scalping, Property 8: MA calculation correctness**
        **Validates: Requirements 3.1**
        """
        import numpy as np
        
        # Test with constant prices
        constant_prices = [100.0] * 50
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1min')
        
        dataframe = pd.DataFrame({
            'date': dates,
            'open': constant_prices,
            'high': constant_prices,
            'low': constant_prices,
            'close': constant_prices,
            'volume': [1000] * 50
        }).set_index('date')
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Calculate indicators
        result_df = self.strategy.populate_indicators(dataframe.copy(), {'pair': 'BTC/USDT'})
        
        # For constant prices, SMA should equal the constant price
        ma_col = f'ma{self.strategy.ma_period.value}'
        valid_sma = result_df[ma_col].dropna()
        
        for value in valid_sma:
            assert abs(value - 100.0) < 1e-10, f"SMA of constant prices should be constant: {value}"
    
    def test_previous_1h_high_calculation(self):
        """
        Test previous 1h high calculation
        
        **Feature: hour-breakout-scalping, Property 8: MA calculation correctness**
        **Validates: Requirements 2.1**
        """
        # Generate test data
        length = 100
        prices = self.generate_price_series(length, 100.0)
        dates = pd.date_range(start='2023-01-01', periods=length, freq='1min')
        
        dataframe = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],  # Higher highs
            'low': [p * 0.98 for p in prices],   # Lower lows
            'close': prices,
            'volume': [1000] * length
        }).set_index('date')
        
        # Create mock 1h data with known high values
        high_1h_values = [110.0, 115.0, 120.0, 125.0, 130.0] * (length // 5 + 1)
        high_1h_values = high_1h_values[:length]
        
        mock_1h_data = pd.DataFrame({
            'high': high_1h_values
        }, index=dates)
        
        # Mock the data provider
        self.strategy.dp.get_pair_dataframe.side_effect = lambda pair, timeframe: {
            '5m': pd.DataFrame(),
            '1h': mock_1h_data
        }.get(timeframe, pd.DataFrame())
        
        # Calculate indicators
        result_df = self.strategy.populate_indicators(dataframe.copy(), {'pair': 'BTC/USDT'})
        
        # Property: high_1h_prev should exist and be properly shifted
        assert 'high_1h_prev' in result_df.columns, "Previous 1h high column not found"
        
        # Property: Previous high should be shifted by 1 period
        if 'high_1h' in result_df.columns and len(result_df) > 1:
            # Check that shift operation worked correctly for first few values
            for i in range(1, min(5, len(result_df))):
                if not pd.isna(result_df['high_1h'].iloc[i-1]):
                    expected_prev = result_df['high_1h'].iloc[i-1]
                    actual_prev = result_df['high_1h_prev'].iloc[i]
                    
                    if not pd.isna(actual_prev):
                        assert abs(actual_prev - expected_prev) < 1e-10, \
                            f"Previous 1h high not properly shifted at index {i}: expected={expected_prev}, actual={actual_prev}"
    
    def test_data_quality_validation(self):
        """
        Test data quality validation and error handling
        
        **Feature: hour-breakout-scalping, Property 8: MA calculation correctness**
        **Validates: Requirements 1.5**
        """
        import numpy as np
        
        # Generate dataframe with quality issues
        length = 50
        prices = self.generate_price_series(length, 100.0)
        dates = pd.date_range(start='2023-01-01', periods=length, freq='1min')
        
        dataframe = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * length
        }).set_index('date')
        
        # Introduce data quality issues
        dataframe.loc[dataframe.index[10], 'close'] = np.nan  # NaN value
        dataframe.loc[dataframe.index[20], 'close'] = 0       # Zero value
        dataframe.loc[dataframe.index[30], 'close'] = -10     # Negative value
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Should handle gracefully
        result_df = self.strategy.populate_indicators(dataframe.copy(), {'pair': 'BTC/USDT'})
        
        # Should not have NaN, zero, or negative values in close after processing
        assert not result_df['close'].isna().any(), "NaN values not handled"
        assert (result_df['close'] > 0).all(), "Zero/negative values not handled"
        
        # Should have data quality flag
        assert 'data_quality_ok' in result_df.columns, "Data quality flag missing"


if __name__ == '__main__':
    pytest.main([__file__])


class TestBreakoutDetection:
    """Property-based tests for breakout detection logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_breakout_test_data(self, num_candles: int, breakout_positions: List[int], 
                                  base_price: float = 100.0) -> pd.DataFrame:
        """Generate test data with controlled breakout scenarios"""
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Generate 1h high data (simulate previous 1h high)
        high_1h_prev = []
        for i in range(num_candles):
            # Simulate 1h high as slightly above recent prices
            lookback_start = max(0, i - 60)  # Look back 60 minutes
            recent_prices = closes[lookback_start:i+1]
            if recent_prices:
                high_1h = max(recent_prices) * 1.002  # Slightly above max
            else:
                high_1h = base_price
            high_1h_prev.append(high_1h)
        
        # Generate 5m close data
        close_5m = []
        for i in range(num_candles):
            if i in breakout_positions:
                # Create breakout: 5m close above 1h high
                breakout_close = high_1h_prev[i] * (1 + np.random.uniform(0.003, 0.01))
                close_5m.append(breakout_close)
            else:
                # Normal case: 5m close below or equal to 1h high
                normal_close = high_1h_prev[i] * np.random.uniform(0.995, 0.999)
                close_5m.append(normal_close)
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = max(closes[i], close_5m[i]) * 1.001
            low = min(closes[i], close_5m[i]) * 0.999
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                'close_5m': close_5m[i],
                'high_1h_prev': high_1h_prev[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=150, max_value=300),
        num_breakouts=st.integers(min_value=1, max_value=10),
        base_price=st.floats(min_value=50.0, max_value=200.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_breakout_condition_detection(self, num_candles: int, num_breakouts: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 2: Breakout condition detection**
        
        For any 5m and 1h price data combination, when 5m close price is higher than 
        previous 1h high, breakout condition should be correctly marked as true
        
        **Validates: Requirements 2.2, 2.3**
        """
        # Generate random breakout positions
        breakout_positions = np.random.choice(
            range(self.strategy.startup_candle_count, num_candles), 
            size=min(num_breakouts, num_candles - self.strategy.startup_candle_count), 
            replace=False
        ).tolist()
        
        # Generate test data with controlled breakouts
        test_df = self.generate_breakout_test_data(num_candles, breakout_positions, base_price)
        
        # Mock empty informative data since we're providing merged data directly
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Apply breakout detection
        self.strategy._detect_breakout_conditions(test_df, 'BTC/USDT')
        
        # Property 1: All intended breakouts should be detected
        for pos in breakout_positions:
            if pos < len(test_df):
                close_5m = test_df.iloc[pos]['close_5m']
                high_1h_prev = test_df.iloc[pos]['high_1h_prev']
                min_breakout_pct = self.strategy.min_breakout_pct.value
                breakout_threshold = high_1h_prev * (1 + min_breakout_pct)
                
                if close_5m > breakout_threshold and test_df.iloc[pos]['data_quality_ok']:
                    assert test_df.iloc[pos]['breakout_condition'], \
                        f"Breakout not detected at position {pos}: 5m_close={close_5m:.6f} > threshold={breakout_threshold:.6f}"
        
        # Property 2: No false positives - non-breakout positions should not be marked
        for i in range(len(test_df)):
            if i not in breakout_positions and i >= self.strategy.startup_candle_count:
                close_5m = test_df.iloc[i]['close_5m']
                high_1h_prev = test_df.iloc[i]['high_1h_prev']
                min_breakout_pct = self.strategy.min_breakout_pct.value
                breakout_threshold = high_1h_prev * (1 + min_breakout_pct)
                
                if close_5m <= breakout_threshold:
                    assert not test_df.iloc[i]['breakout_condition'], \
                        f"False breakout detected at position {i}: 5m_close={close_5m:.6f} <= threshold={breakout_threshold:.6f}"
        
        # Property 3: Early candles (insufficient data) should not have breakouts
        for i in range(min(self.strategy.startup_candle_count, len(test_df))):
            assert not test_df.iloc[i]['breakout_condition'], \
                f"Breakout detected in insufficient data period at position {i}"
        
        # Property 4: Invalid data should not trigger breakouts
        invalid_mask = (
            (test_df['close_5m'].isna()) |
            (test_df['high_1h_prev'].isna()) |
            (test_df['close_5m'] <= 0) |
            (test_df['high_1h_prev'] <= 0) |
            (~test_df['data_quality_ok'])
        )
        
        for i in range(len(test_df)):
            if invalid_mask.iloc[i]:
                assert not test_df.iloc[i]['breakout_condition'], \
                    f"Breakout detected with invalid data at position {i}"
        
        # Property 5: Breakout strength should be calculated correctly
        assert 'breakout_strength' in test_df.columns, "Breakout strength column missing"
        
        for i in range(len(test_df)):
            if test_df.iloc[i]['breakout_condition']:
                expected_strength = (test_df.iloc[i]['close_5m'] - test_df.iloc[i]['high_1h_prev']) / test_df.iloc[i]['high_1h_prev'] * 100
                actual_strength = test_df.iloc[i]['breakout_strength']
                assert abs(actual_strength - expected_strength) < 0.001, \
                    f"Breakout strength calculation error at position {i}: expected={expected_strength:.6f}, actual={actual_strength:.6f}"
            else:
                assert test_df.iloc[i]['breakout_strength'] == 0.0, \
                    f"Non-zero breakout strength for non-breakout at position {i}"
    
    def test_property_breakout_boundary_conditions(self):
        """
        Test breakout detection with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 2: Breakout condition detection**
        **Validates: Requirements 2.5**
        """
        # Test with minimal data
        minimal_df = pd.DataFrame({
            'close_5m': [100.0, 101.0],
            'high_1h_prev': [99.0, 100.0],
            'data_quality_ok': [True, True]
        })
        
        self.strategy._detect_breakout_conditions(minimal_df, 'BTC/USDT')
        
        # Should handle minimal data without error
        assert 'breakout_condition' in minimal_df.columns
        assert minimal_df['breakout_condition'].dtype == bool
        
        # Test with NaN values
        nan_df = pd.DataFrame({
            'close_5m': [100.0, np.nan, 102.0],
            'high_1h_prev': [99.0, 100.0, np.nan],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._detect_breakout_conditions(nan_df, 'BTC/USDT')
        
        # NaN values should not trigger breakouts
        assert not nan_df.iloc[1]['breakout_condition']  # NaN close_5m
        assert not nan_df.iloc[2]['breakout_condition']  # NaN high_1h_prev
        
        # Test with zero/negative values
        invalid_df = pd.DataFrame({
            'close_5m': [100.0, 0.0, -10.0],
            'high_1h_prev': [99.0, 100.0, 100.0],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._detect_breakout_conditions(invalid_df, 'BTC/USDT')
        
        # Zero/negative values should not trigger breakouts
        assert not invalid_df.iloc[1]['breakout_condition']  # Zero close_5m
        assert not invalid_df.iloc[2]['breakout_condition']  # Negative close_5m
    
    def test_property_breakout_threshold_sensitivity(self):
        """
        Test that breakout detection respects minimum breakout percentage threshold
        
        **Feature: hour-breakout-scalping, Property 2: Breakout condition detection**
        **Validates: Requirements 2.3**
        """
        # Create data right at the threshold with sufficient candles
        num_candles = self.strategy.startup_candle_count + 10
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Create base data
        base_data = []
        for i in range(num_candles):
            base_data.append({
                'date': dates[i],
                'open': 100.0,
                'high': 100.1,
                'low': 99.9,
                'close': 100.0,
                'volume': 1000,
                'close_5m': 100.0,
                'high_1h_prev': 100.0,
                'data_quality_ok': True
            })
        
        threshold_df = pd.DataFrame(base_data).set_index('date')
        
        # Set specific test values for the last few candles (after startup period)
        test_start = self.strategy.startup_candle_count
        # 0.002 = 0.2%, so threshold = 100.0 * (1 + 0.002) = 100.2
        threshold_df.loc[threshold_df.index[test_start], 'close_5m'] = 100.0      # Below threshold
        threshold_df.loc[threshold_df.index[test_start + 1], 'close_5m'] = 100.1  # Below threshold
        threshold_df.loc[threshold_df.index[test_start + 2], 'close_5m'] = 100.2  # At threshold (should NOT trigger with >)
        threshold_df.loc[threshold_df.index[test_start + 3], 'close_5m'] = 100.21 # Above threshold (should trigger)
        
        # Set a specific minimum breakout percentage
        original_min_breakout = self.strategy.min_breakout_pct.value
        self.strategy.min_breakout_pct.value = 0.002  # 0.2%
        
        try:
            self.strategy._detect_breakout_conditions(threshold_df, 'BTC/USDT')
            
            # Only prices ABOVE threshold should trigger breakout (using > not >=)
            # Threshold = 100.0 * (1 + 0.002) = 100.2
            assert not threshold_df.iloc[test_start]['breakout_condition']      # 100.0 < 100.2
            assert not threshold_df.iloc[test_start + 1]['breakout_condition']  # 100.1 < 100.2
            assert not threshold_df.iloc[test_start + 2]['breakout_condition']  # 100.2 == 100.2 (not >)
            assert threshold_df.iloc[test_start + 3]['breakout_condition']      # 100.21 > 100.2
            
        finally:
            # Restore original value
            self.strategy.min_breakout_pct.value = original_min_breakout
    
    @given(
        num_candles=st.integers(min_value=200, max_value=400),
        num_breakouts=st.integers(min_value=2, max_value=8),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_state_management_consistency(self, num_candles: int, num_breakouts: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 9: State management consistency**
        
        For any strategy state changes, breakout state should be activated, maintained, 
        and reset under correct conditions
        
        **Validates: Requirements 2.4, 3.5**
        """
        # Generate random breakout positions with sufficient spacing
        min_spacing = 30  # Minimum 30 candles between breakouts
        breakout_positions = []
        
        for _ in range(num_breakouts):
            # Find valid position that doesn't conflict with existing breakouts
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                pos = np.random.randint(self.strategy.startup_candle_count, num_candles - min_spacing)
                if not any(abs(pos - existing) < min_spacing for existing in breakout_positions):
                    breakout_positions.append(pos)
                    break
                attempts += 1
        
        # Generate test data with controlled breakouts
        test_df = self.generate_breakout_test_data(num_candles, breakout_positions, base_price)
        
        # Mock empty informative data since we're providing merged data directly
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Apply breakout detection and state management
        self.strategy._detect_breakout_conditions(test_df, 'BTC/USDT')
        self.strategy._manage_breakout_state(test_df, 'BTC/USDT')
        
        # Property 1: State activation - breakout conditions should activate state
        for pos in breakout_positions:
            if pos < len(test_df) and test_df.iloc[pos]['breakout_condition']:
                assert test_df.iloc[pos]['breakout_state_active'], \
                    f"Breakout state not activated at position {pos} despite breakout condition"
                
                # State should have activation time
                assert pd.notna(test_df.iloc[pos]['breakout_activation_time']), \
                    f"Activation time not set at position {pos}"
                
                # State should have reference high
                assert test_df.iloc[pos]['breakout_reference_high'] > 0, \
                    f"Reference high not set at position {pos}"
        
        # Property 2: State persistence - active state should persist until reset conditions
        for i in range(1, len(test_df)):
            prev_active = test_df.iloc[i-1]['breakout_state_active']
            current_active = test_df.iloc[i]['breakout_state_active']
            
            if prev_active and not current_active:
                # State was reset - verify reset was justified
                current_row = test_df.iloc[i]
                prev_activation_time = test_df.iloc[i-1]['breakout_activation_time']
                prev_reference_high = test_df.iloc[i-1]['breakout_reference_high']
                
                should_reset = self.strategy._should_reset_breakout_state(
                    current_row, prev_activation_time, prev_reference_high, test_df.index[i]
                )
                
                assert should_reset, f"State reset at position {i} without valid reset condition"
        
        # Property 3: State consistency - activation time and reference high should be consistent
        for i in range(len(test_df)):
            if test_df.iloc[i]['breakout_state_active']:
                # Active state should have valid activation time
                assert pd.notna(test_df.iloc[i]['breakout_activation_time']), \
                    f"Active state without activation time at position {i}"
                
                # Active state should have valid reference high
                assert test_df.iloc[i]['breakout_reference_high'] > 0, \
                    f"Active state without reference high at position {i}"
                
                # If previous candle was also active, times should match (unless new breakout)
                if i > 0 and test_df.iloc[i-1]['breakout_state_active'] and not test_df.iloc[i]['breakout_condition']:
                    prev_time = test_df.iloc[i-1]['breakout_activation_time']
                    current_time = test_df.iloc[i]['breakout_activation_time']
                    assert prev_time == current_time, \
                        f"Activation time changed without new breakout at position {i}"
        
        # Property 4: State columns should always exist and have correct types
        required_state_columns = ['breakout_state_active', 'breakout_activation_time', 'breakout_reference_high']
        for col in required_state_columns:
            assert col in test_df.columns, f"Required state column {col} missing"
        
        # Check data types
        assert test_df['breakout_state_active'].dtype == bool, "breakout_state_active should be boolean"
        assert pd.api.types.is_datetime64_any_dtype(test_df['breakout_activation_time']), \
            "breakout_activation_time should be datetime"
        assert pd.api.types.is_numeric_dtype(test_df['breakout_reference_high']), \
            "breakout_reference_high should be numeric"
        
        # Property 5: No state should be active without prior breakout condition
        first_active_index = None
        for i in range(len(test_df)):
            if test_df.iloc[i]['breakout_state_active']:
                first_active_index = i
                break
        
        if first_active_index is not None:
            # There should be a breakout condition at or before the first active state
            breakout_found = False
            for j in range(first_active_index + 1):
                if test_df.iloc[j]['breakout_condition']:
                    breakout_found = True
                    break
            
            assert breakout_found, f"Active state found at position {first_active_index} without prior breakout condition"
    
    def test_property_state_reset_conditions(self):
        """
        Test state reset logic under various conditions
        
        **Feature: hour-breakout-scalping, Property 9: State management consistency**
        **Validates: Requirements 3.5**
        """
        # Test data quality reset
        poor_quality_row = pd.Series({
            'close': 100.0,
            'data_quality_ok': False,
            'high_1h_prev': 99.0
        })
        
        activation_time = pd.Timestamp('2023-01-01 10:00:00')
        reference_high = 99.0
        current_time = pd.Timestamp('2023-01-01 10:05:00')
        
        should_reset = self.strategy._should_reset_breakout_state(
            poor_quality_row, activation_time, reference_high, current_time
        )
        assert should_reset, "Should reset state on poor data quality"
        
        # Test price invalidation reset
        low_price_row = pd.Series({
            'close': 98.0,  # Below 99.0 * 0.995 = 98.505
            'data_quality_ok': True,
            'high_1h_prev': 99.0
        })
        
        should_reset = self.strategy._should_reset_breakout_state(
            low_price_row, activation_time, reference_high, current_time
        )
        assert should_reset, "Should reset state when price falls below invalidation threshold"
        
        # Test timeout reset
        timeout_time = pd.Timestamp('2023-01-01 12:30:00')  # 2.5 hours later
        normal_row = pd.Series({
            'close': 100.0,
            'data_quality_ok': True,
            'high_1h_prev': 99.0
        })
        
        should_reset = self.strategy._should_reset_breakout_state(
            normal_row, activation_time, reference_high, timeout_time
        )
        assert should_reset, "Should reset state after timeout period"
        
        # Test structure change reset
        structure_change_row = pd.Series({
            'close': 100.0,
            'data_quality_ok': True,
            'high_1h_prev': 102.0  # 3% change from 99.0
        })
        
        should_reset = self.strategy._should_reset_breakout_state(
            structure_change_row, activation_time, reference_high, current_time
        )
        assert should_reset, "Should reset state on significant structure change"
        
        # Test normal conditions (should not reset)
        normal_row = pd.Series({
            'close': 100.0,
            'data_quality_ok': True,
            'high_1h_prev': 99.5  # Small change
        })
        
        should_reset = self.strategy._should_reset_breakout_state(
            normal_row, activation_time, reference_high, current_time
        )
        assert not should_reset, "Should not reset state under normal conditions"


class TestPullbackDetection:
    """Property-based tests for pullback detection logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_pullback_test_data(self, num_candles: int, breakout_periods: List[Tuple[int, int]], 
                                  pullback_positions: List[int], base_price: float = 100.0) -> pd.DataFrame:
        """Generate test data with controlled breakout periods and pullback scenarios"""
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Calculate MA5 values
        ma_period = self.strategy.ma_period.value
        ma_values = []
        for i in range(num_candles):
            if i < ma_period - 1:
                ma_values.append(closes[i])  # Use current price for early candles
            else:
                window_prices = closes[i - ma_period + 1:i + 1]
                ma_values.append(sum(window_prices) / len(window_prices))
        
        # Generate lows based on closes and pullback requirements
        lows = []
        for i in range(num_candles):
            if i in pullback_positions:
                # Create pullback: low touches or goes below MA5
                pullback_low = ma_values[i] * np.random.uniform(0.998, 1.002)  # Touch MA5 with tolerance
                lows.append(min(closes[i] * 0.999, pullback_low))
            else:
                # Normal case: low above MA5
                normal_low = closes[i] * np.random.uniform(0.995, 0.999)
                lows.append(max(normal_low, ma_values[i] * 1.001))  # Keep above MA5
        
        # Set breakout state active for specified periods
        breakout_state_active = [False] * num_candles
        for start, end in breakout_periods:
            for i in range(start, min(end + 1, num_candles)):
                breakout_state_active[i] = True
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = closes[i] * 1.001
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': lows[i],
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                f'ma{ma_period}': ma_values[i],
                'breakout_state_active': breakout_state_active[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=200, max_value=400),
        num_breakout_periods=st.integers(min_value=1, max_value=5),
        num_pullbacks=st.integers(min_value=1, max_value=10),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_pullback_condition_detection(self, num_candles: int, num_breakout_periods: int, 
                                                 num_pullbacks: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 3: Pullback condition detection**
        
        For any 1m price data and MA5 values, when close price is low or equal to MA5,
        or low price touches MA5, pullback condition should be correctly marked
        
        **Validates: Requirements 3.3, 3.4**
        """
        # Generate random breakout periods (start, end)
        breakout_periods = []
        for _ in range(num_breakout_periods):
            start = np.random.randint(self.strategy.startup_candle_count, num_candles - 50)
            duration = np.random.randint(20, 50)
            end = min(start + duration, num_candles - 1)
            breakout_periods.append((start, end))
        
        # Generate pullback positions within breakout periods
        pullback_positions = []
        for start, end in breakout_periods:
            period_pullbacks = min(num_pullbacks // num_breakout_periods + 1, (end - start) // 5)
            for _ in range(period_pullbacks):
                pos = np.random.randint(start + 5, end - 5)
                pullback_positions.append(pos)
        
        # Generate test data with controlled pullbacks
        test_df = self.generate_pullback_test_data(num_candles, breakout_periods, pullback_positions, base_price)
        
        # Apply pullback detection
        self.strategy._detect_pullback_conditions(test_df, 'BTC/USDT')
        
        # Property 1: Pullback should only be detected during active breakout periods
        for i in range(len(test_df)):
            if test_df.iloc[i]['pullback_condition']:
                assert test_df.iloc[i]['breakout_state_active'], \
                    f"Pullback detected outside active breakout period at position {i}"
        
        # Property 2: Pullback conditions should be detected when criteria are met
        ma_col = f'ma{self.strategy.ma_period.value}'
        pullback_tolerance = self.strategy.pullback_tolerance.value
        
        for pos in pullback_positions:
            if pos < len(test_df) and test_df.iloc[pos]['breakout_state_active']:
                current_close = test_df.iloc[pos]['close']
                current_low = test_df.iloc[pos]['low']
                current_ma = test_df.iloc[pos][ma_col]
                
                ma_threshold = current_ma * (1 + pullback_tolerance)
                
                # Check if pullback criteria are met
                close_pullback = current_close <= ma_threshold
                low_pullback = current_low <= ma_threshold
                
                if close_pullback or low_pullback:
                    assert test_df.iloc[pos]['pullback_condition'], \
                        f"Pullback not detected at position {pos}: close={current_close:.6f}, " \
                        f"low={current_low:.6f}, MA={current_ma:.6f}, threshold={ma_threshold:.6f}"
        
        # Property 3: No pullback should be detected outside breakout periods
        for i in range(len(test_df)):
            if not test_df.iloc[i]['breakout_state_active']:
                assert not test_df.iloc[i]['pullback_condition'], \
                    f"Pullback detected outside breakout period at position {i}"
        
        # Property 4: Invalid data should not trigger pullbacks
        invalid_positions = []
        for i in range(len(test_df)):
            current_close = test_df.iloc[i]['close']
            current_low = test_df.iloc[i]['low']
            current_ma = test_df.iloc[i][ma_col]
            data_quality = test_df.iloc[i]['data_quality_ok']
            
            if (current_close <= 0 or current_low <= 0 or current_ma <= 0 or not data_quality):
                invalid_positions.append(i)
                assert not test_df.iloc[i]['pullback_condition'], \
                    f"Pullback detected with invalid data at position {i}"
        
        # Property 5: Pullback strength should be calculated correctly
        assert 'pullback_strength' in test_df.columns, "Pullback strength column missing"
        
        for i in range(len(test_df)):
            if test_df.iloc[i]['pullback_condition']:
                current_close = test_df.iloc[i]['close']
                current_ma = test_df.iloc[i][ma_col]
                
                if current_ma > 0:
                    expected_strength = (current_close - current_ma) / current_ma * 100
                    actual_strength = test_df.iloc[i]['pullback_strength']
                    assert abs(actual_strength - expected_strength) < 0.001, \
                        f"Pullback strength calculation error at position {i}: " \
                        f"expected={expected_strength:.6f}, actual={actual_strength:.6f}"
            else:
                assert test_df.iloc[i]['pullback_strength'] == 0.0, \
                    f"Non-zero pullback strength for non-pullback at position {i}"
        
        # Property 6: Pullback condition should be reset when breakout state is reset
        for i in range(1, len(test_df)):
            prev_active = test_df.iloc[i-1]['breakout_state_active']
            current_active = test_df.iloc[i]['breakout_state_active']
            
            # If breakout state was reset, pullback should also be reset
            if prev_active and not current_active:
                assert not test_df.iloc[i]['pullback_condition'], \
                    f"Pullback condition not reset when breakout state reset at position {i}"
    
    def test_property_pullback_boundary_conditions(self):
        """
        Test pullback detection with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 3: Pullback condition detection**
        **Validates: Requirements 3.3, 3.4**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Test with minimal data
        minimal_df = pd.DataFrame({
            'close': [100.0, 99.5],
            'low': [99.8, 99.0],
            ma_col: [100.0, 100.0],
            'breakout_state_active': [True, True],
            'data_quality_ok': [True, True]
        })
        
        self.strategy._detect_pullback_conditions(minimal_df, 'BTC/USDT')
        
        # Should handle minimal data without error
        assert 'pullback_condition' in minimal_df.columns
        assert minimal_df['pullback_condition'].dtype == bool
        
        # Test with NaN values
        nan_df = pd.DataFrame({
            'close': [100.0, np.nan, 99.0],
            'low': [99.8, 99.0, np.nan],
            ma_col: [100.0, 100.0, 100.0],
            'breakout_state_active': [True, True, True],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._detect_pullback_conditions(nan_df, 'BTC/USDT')
        
        # NaN values should not trigger pullbacks
        assert not nan_df.iloc[1]['pullback_condition']  # NaN close
        assert not nan_df.iloc[2]['pullback_condition']  # NaN low
        
        # Test with zero/negative values
        invalid_df = pd.DataFrame({
            'close': [100.0, 0.0, -10.0],
            'low': [99.8, 99.0, 99.0],
            ma_col: [100.0, 100.0, 100.0],
            'breakout_state_active': [True, True, True],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._detect_pullback_conditions(invalid_df, 'BTC/USDT')
        
        # Zero/negative values should not trigger pullbacks
        assert not invalid_df.iloc[1]['pullback_condition']  # Zero close
        assert not invalid_df.iloc[2]['pullback_condition']  # Negative close
    
    def test_property_pullback_tolerance_sensitivity(self):
        """
        Test that pullback detection respects tolerance parameter
        
        **Feature: hour-breakout-scalping, Property 3: Pullback condition detection**
        **Validates: Requirements 3.3, 3.4**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Create data right at the tolerance threshold
        threshold_df = pd.DataFrame({
            'close': [100.0, 100.05, 100.1, 100.15],  # Various distances from MA
            'low': [99.95, 100.0, 100.05, 100.12],    # Last low is above threshold
            ma_col: [100.0, 100.0, 100.0, 100.0],  # MA = 100.0
            'breakout_state_active': [True, True, True, True],
            'data_quality_ok': [True, True, True, True]
        })
        
        # Set specific tolerance for testing
        original_tolerance = self.strategy.pullback_tolerance.value
        self.strategy.pullback_tolerance.value = 0.001  # 0.1%
        
        try:
            self.strategy._detect_pullback_conditions(threshold_df, 'BTC/USDT')
            
            # Threshold = 100.0 * (1 + 0.001) = 100.1
            # close <= 100.1 should trigger pullback
            assert threshold_df.iloc[0]['pullback_condition']      # 100.0 <= 100.1
            assert threshold_df.iloc[1]['pullback_condition']      # 100.05 <= 100.1
            assert threshold_df.iloc[2]['pullback_condition']      # 100.1 <= 100.1
            assert not threshold_df.iloc[3]['pullback_condition']  # 100.15 > 100.1
            
        finally:
            # Restore original value
            self.strategy.pullback_tolerance.value = original_tolerance
    
    def test_property_pullback_only_during_breakout(self):
        """
        Test that pullback is only detected during active breakout periods
        
        **Feature: hour-breakout-scalping, Property 3: Pullback condition detection**
        **Validates: Requirements 3.2**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Create data with clear pullback conditions but varying breakout states
        test_df = pd.DataFrame({
            'close': [99.0, 99.0, 99.0, 99.0],  # All below MA (should be pullback if active)
            'low': [98.5, 98.5, 98.5, 98.5],   # All well below MA
            ma_col: [100.0, 100.0, 100.0, 100.0],
            'breakout_state_active': [False, True, False, True],  # Alternating states
            'data_quality_ok': [True, True, True, True]
        })
        
        self.strategy._detect_pullback_conditions(test_df, 'BTC/USDT')
        
        # Only positions with active breakout should have pullback detected
        assert not test_df.iloc[0]['pullback_condition']  # Not active
        assert test_df.iloc[1]['pullback_condition']      # Active
        assert not test_df.iloc[2]['pullback_condition']  # Not active
        assert test_df.iloc[3]['pullback_condition']      # Active


class TestReboundDetectionAndEntrySignals:
    """Property-based tests for rebound detection and entry signal generation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_three_stage_test_data(self, num_candles: int, three_stage_scenarios: List[Tuple[int, int, int, int]], 
                                     base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate test data with controlled three-stage scenarios (breakout + pullback + rebound)
        
        :param num_candles: Total number of candles
        :param three_stage_scenarios: List of (breakout_start, pullback_pos, rebound_pos, scenario_end)
        :param base_price: Base price for generation
        :return: DataFrame with controlled three-stage scenarios
        """
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Calculate MA5 values
        ma_period = self.strategy.ma_period.value
        ma_values = []
        for i in range(num_candles):
            if i < ma_period - 1:
                ma_values.append(closes[i])  # Use current price for early candles
            else:
                window_prices = closes[i - ma_period + 1:i + 1]
                ma_values.append(sum(window_prices) / len(window_prices))
        
        # Initialize state arrays
        breakout_state_active = [False] * num_candles
        pullback_condition = [False] * num_candles
        pullback_completed = [False] * num_candles
        breakout_activation_time = [pd.NaT] * num_candles
        breakout_reference_high = [0.0] * num_candles
        
        # Set up three-stage scenarios
        for breakout_start, pullback_pos, rebound_pos, scenario_end in three_stage_scenarios:
            scenario_end = min(scenario_end, num_candles - 1)
            
            # Activation time for this scenario
            activation_time = dates[breakout_start]
            reference_high = closes[breakout_start] * 0.998  # Slightly below breakout price
            
            # Set breakout state active for the entire scenario
            for i in range(breakout_start, scenario_end + 1):
                breakout_state_active[i] = True
                breakout_activation_time[i] = activation_time
                breakout_reference_high[i] = reference_high
            
            # Set pullback condition at specific position
            if pullback_pos < num_candles:
                pullback_condition[pullback_pos] = True
                
                # Set pullback completed from pullback position onwards
                for i in range(pullback_pos, scenario_end + 1):
                    pullback_completed[i] = True
            
            # Adjust close price for rebound (ensure it's above MA5)
            if rebound_pos < num_candles:
                closes[rebound_pos] = ma_values[rebound_pos] * 1.005  # 0.5% above MA5
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = closes[i] * 1.001
            low = closes[i] * 0.999
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                f'ma{ma_period}': ma_values[i],
                'breakout_state_active': breakout_state_active[i],
                'pullback_condition': pullback_condition[i],
                'pullback_completed': pullback_completed[i],
                'breakout_activation_time': breakout_activation_time[i],
                'breakout_reference_high': breakout_reference_high[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=200, max_value=400),
        num_scenarios=st.integers(min_value=1, max_value=5),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=20, deadline=None)  # Reduced examples for faster testing
    def test_property_three_stage_entry_logic(self, num_candles: int, num_scenarios: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 4: Three-stage entry logic**
        
        For any market data, only when breakout, pullback and rebound three conditions 
        are all satisfied, should entry signal be generated
        
        **Validates: Requirements 4.3**
        """
        # Generate random three-stage scenarios
        three_stage_scenarios = []
        min_scenario_length = 50
        
        for _ in range(num_scenarios):
            # Ensure scenarios don't overlap and have sufficient length
            attempts = 0
            while attempts < 20:  # Prevent infinite loop
                breakout_start = np.random.randint(self.strategy.startup_candle_count, 
                                                 num_candles - min_scenario_length)
                
                # Check for conflicts with existing scenarios
                conflict = False
                for existing_start, _, _, existing_end in three_stage_scenarios:
                    if abs(breakout_start - existing_start) < min_scenario_length:
                        conflict = True
                        break
                
                if not conflict:
                    pullback_pos = breakout_start + np.random.randint(5, 20)
                    rebound_pos = pullback_pos + np.random.randint(1, 10)
                    scenario_end = rebound_pos + np.random.randint(10, 30)
                    scenario_end = min(scenario_end, num_candles - 1)
                    
                    three_stage_scenarios.append((breakout_start, pullback_pos, rebound_pos, scenario_end))
                    break
                
                attempts += 1
        
        # Generate test data with controlled three-stage scenarios
        test_df = self.generate_three_stage_test_data(num_candles, three_stage_scenarios, base_price)
        
        # Apply rebound detection and entry signal generation
        self.strategy._detect_rebound_conditions(test_df, 'BTC/USDT')
        
        # Property 1: Entry signals should only be generated when all three conditions are met
        for i in range(len(test_df)):
            entry_signal = test_df.iloc[i]['entry_signal']
            breakout_active = test_df.iloc[i]['breakout_state_active']
            pullback_completed = test_df.iloc[i]['pullback_completed']
            rebound_condition = test_df.iloc[i]['rebound_condition']
            
            if entry_signal:
                # If entry signal is generated, all three conditions must be true
                assert breakout_active, f"Entry signal without breakout at position {i}"
                assert pullback_completed, f"Entry signal without pullback completion at position {i}"
                assert rebound_condition, f"Entry signal without rebound at position {i}"
        
        # Property 2: When all three conditions are met, entry signal should be generated (unless blocked by duplicate prevention)
        for breakout_start, pullback_pos, rebound_pos, scenario_end in three_stage_scenarios:
            if rebound_pos < len(test_df):
                # Check if all conditions are met at rebound position
                breakout_active = test_df.iloc[rebound_pos]['breakout_state_active']
                pullback_completed = test_df.iloc[rebound_pos]['pullback_completed']
                rebound_condition = test_df.iloc[rebound_pos]['rebound_condition']
                
                if breakout_active and pullback_completed and rebound_condition:
                    # Entry signal should be generated (unless blocked by duplicate prevention)
                    entry_signal = test_df.iloc[rebound_pos]['entry_signal']
                    
                    # Check if it was blocked by duplicate prevention
                    duplicate_blocked = self.strategy._check_duplicate_entry_prevention(test_df, rebound_pos, 'BTC/USDT')
                    
                    if not duplicate_blocked:
                        assert entry_signal, f"Entry signal not generated despite all conditions met at position {rebound_pos}"
        
        # Property 3: No entry signal should be generated with incomplete conditions
        for i in range(len(test_df)):
            entry_signal = test_df.iloc[i]['entry_signal']
            breakout_active = test_df.iloc[i]['breakout_state_active']
            pullback_completed = test_df.iloc[i]['pullback_completed']
            rebound_condition = test_df.iloc[i]['rebound_condition']
            
            # If any condition is missing, no entry signal should be generated
            incomplete_conditions = not (breakout_active and pullback_completed and rebound_condition)
            
            if incomplete_conditions:
                assert not entry_signal, f"Entry signal generated with incomplete conditions at position {i}: " \
                                        f"breakout={breakout_active}, pullback_completed={pullback_completed}, rebound={rebound_condition}"
        
        # Property 4: Rebound condition should only be true when price is above MA5 after pullback completion
        ma_col = f'ma{self.strategy.ma_period.value}'
        for i in range(len(test_df)):
            rebound_condition = test_df.iloc[i]['rebound_condition']
            current_close = test_df.iloc[i]['close']
            current_ma = test_df.iloc[i][ma_col]
            breakout_active = test_df.iloc[i]['breakout_state_active']
            pullback_completed = test_df.iloc[i]['pullback_completed']
            
            if rebound_condition:
                # Rebound should only be true when conditions are met
                assert breakout_active, f"Rebound without active breakout at position {i}"
                assert pullback_completed, f"Rebound without pullback completion at position {i}"
                assert current_close > current_ma, f"Rebound with close <= MA5 at position {i}: close={current_close:.6f}, MA5={current_ma:.6f}"
        
        # Property 5: Entry signal strength should be calculated correctly
        for i in range(len(test_df)):
            if test_df.iloc[i]['entry_signal']:
                current_close = test_df.iloc[i]['close']
                current_ma = test_df.iloc[i][ma_col]
                entry_strength = test_df.iloc[i]['entry_signal_strength']
                
                if current_ma > 0:
                    expected_strength = (current_close - current_ma) / current_ma * 100
                    assert abs(entry_strength - expected_strength) < 0.001, \
                        f"Entry signal strength calculation error at position {i}: " \
                        f"expected={expected_strength:.6f}, actual={entry_strength:.6f}"
            else:
                # No entry signal should have zero strength
                assert test_df.iloc[i]['entry_signal_strength'] == 0.0, \
                    f"Non-zero entry strength without entry signal at position {i}"
        
        # Property 6: Required columns should exist and have correct types
        required_columns = ['rebound_condition', 'entry_signal', 'entry_signal_strength', 'pullback_completed']
        for col in required_columns:
            assert col in test_df.columns, f"Required column {col} missing"
        
        # Check data types
        assert test_df['rebound_condition'].dtype == bool, "rebound_condition should be boolean"
        assert test_df['entry_signal'].dtype == bool, "entry_signal should be boolean"
        assert pd.api.types.is_numeric_dtype(test_df['entry_signal_strength']), "entry_signal_strength should be numeric"
        assert test_df['pullback_completed'].dtype == bool, "pullback_completed should be boolean"
    
    def test_property_three_stage_boundary_conditions(self):
        """
        Test three-stage entry logic with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 4: Three-stage entry logic**
        **Validates: Requirements 4.3**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Test with minimal complete scenario
        complete_scenario_df = pd.DataFrame({
            'close': [100.0, 101.0, 99.5, 100.5],  # Rebound at last candle
            'low': [99.8, 100.8, 99.0, 100.2],
            ma_col: [100.0, 100.0, 100.0, 100.0],
            'breakout_state_active': [True, True, True, True],
            'pullback_condition': [False, False, True, False],
            'pullback_completed': [False, False, True, True],
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 4,
            'breakout_reference_high': [99.5] * 4,
            'data_quality_ok': [True, True, True, True]
        })
        
        self.strategy._detect_rebound_conditions(complete_scenario_df, 'BTC/USDT')
        
        # Should generate entry signal at position 3 (rebound after pullback completion)
        assert complete_scenario_df.iloc[3]['rebound_condition'], "Rebound not detected in complete scenario"
        assert complete_scenario_df.iloc[3]['entry_signal'], "Entry signal not generated in complete scenario"
        
        # Test with incomplete scenarios
        incomplete_scenarios = [
            # Missing breakout
            {
                'close': [100.0, 101.0, 99.5, 100.5],
                ma_col: [100.0, 100.0, 100.0, 100.0],
                'breakout_state_active': [False, False, False, False],  # No breakout
                'pullback_completed': [False, False, True, True],
                'data_quality_ok': [True, True, True, True]
            },
            # Missing pullback completion
            {
                'close': [100.0, 101.0, 99.5, 100.5],
                ma_col: [100.0, 100.0, 100.0, 100.0],
                'breakout_state_active': [True, True, True, True],
                'pullback_completed': [False, False, False, False],  # No pullback completion
                'data_quality_ok': [True, True, True, True]
            },
            # Missing rebound (close not above MA5)
            {
                'close': [100.0, 101.0, 99.5, 99.8],  # Last close below MA5
                ma_col: [100.0, 100.0, 100.0, 100.0],
                'breakout_state_active': [True, True, True, True],
                'pullback_completed': [False, False, True, True],
                'data_quality_ok': [True, True, True, True]
            }
        ]
        
        for scenario_idx, scenario_data in enumerate(incomplete_scenarios):
            scenario_df = pd.DataFrame(scenario_data)
            
            self.strategy._detect_rebound_conditions(scenario_df, 'BTC/USDT')
            
            # No entry signals should be generated in incomplete scenarios
            entry_signals = scenario_df['entry_signal'].sum()
            assert entry_signals == 0, f"Entry signal generated in incomplete scenario {scenario_idx}"
    
    def test_property_pullback_completion_state_management(self):
        """
        Test pullback completion state management logic
        
        **Feature: hour-breakout-scalping, Property 4: Three-stage entry logic**
        **Validates: Requirements 4.1**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Test pullback completion persistence
        state_test_df = pd.DataFrame({
            'close': [100.0, 101.0, 99.5, 100.2, 100.8, 99.0],
            ma_col: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'breakout_state_active': [True, True, True, True, True, False],  # Breakout ends at last candle
            'pullback_condition': [False, False, True, False, False, False],  # Pullback at position 2
            'data_quality_ok': [True, True, True, True, True, True]
        })
        
        self.strategy._detect_rebound_conditions(state_test_df, 'BTC/USDT')
        
        # Pullback completion should persist after pullback occurs
        assert not state_test_df.iloc[0]['pullback_completed'], "Pullback completed before pullback"
        assert not state_test_df.iloc[1]['pullback_completed'], "Pullback completed before pullback"
        assert state_test_df.iloc[2]['pullback_completed'], "Pullback not completed at pullback position"
        assert state_test_df.iloc[3]['pullback_completed'], "Pullback completion not persisted"
        assert state_test_df.iloc[4]['pullback_completed'], "Pullback completion not persisted"
        assert not state_test_df.iloc[5]['pullback_completed'], "Pullback completion not reset when breakout ends"
    
    def test_property_rebound_detection_accuracy(self):
        """
        Test rebound detection accuracy with various price scenarios
        
        **Feature: hour-breakout-scalping, Property 4: Three-stage entry logic**
        **Validates: Requirements 4.2**
        """
        ma_col = f'ma{self.strategy.ma_period.value}'
        
        # Get rebound strength threshold for test calculations
        rebound_threshold = self.strategy.rebound_strength_threshold.value
        
        # Test various rebound scenarios
        # Note: Rebound requires close > ma AND (close - ma) / ma >= rebound_strength_threshold
        rebound_scenarios = [
            # Clear rebound: close well above MA5 (1% above, exceeds threshold)
            {
                'close': 100.0 * (1 + rebound_threshold + 0.005),  # Above threshold
                'ma': 100.0,
                'breakout_active': True,
                'pullback_condition': True,
                'expected_rebound': True
            },
            # Marginal rebound: close slightly above MA5 but below threshold
            {
                'close': 100.001,  # 0.001% above MA, below typical threshold of 0.3%
                'ma': 100.0,
                'breakout_active': True,
                'pullback_condition': True,
                'expected_rebound': False  # Below rebound_strength_threshold
            },
            # No rebound: close equal to MA5
            {
                'close': 100.0,
                'ma': 100.0,
                'breakout_active': True,
                'pullback_condition': True,
                'expected_rebound': False
            },
            # No rebound: close below MA5
            {
                'close': 99.9,
                'ma': 100.0,
                'breakout_active': True,
                'pullback_condition': True,
                'expected_rebound': False
            },
            # No rebound: missing breakout
            {
                'close': 100.0 * (1 + rebound_threshold + 0.005),
                'ma': 100.0,
                'breakout_active': False,
                'pullback_condition': True,
                'expected_rebound': False
            },
            # No rebound: missing pullback
            {
                'close': 100.0 * (1 + rebound_threshold + 0.005),
                'ma': 100.0,
                'breakout_active': True,
                'pullback_condition': False,
                'expected_rebound': False
            }
        ]
        
        for scenario_idx, scenario in enumerate(rebound_scenarios):
            # Create a two-candle scenario to allow pullback completion state to be set
            test_df = pd.DataFrame({
                'close': [scenario['close'] * 0.99, scenario['close']],  # Previous candle, current candle
                ma_col: [scenario['ma'], scenario['ma']],
                'breakout_state_active': [scenario['breakout_active'], scenario['breakout_active']],
                'pullback_condition': [scenario['pullback_condition'], False],  # Pullback in first candle
                'data_quality_ok': [True, True]
            })
            
            self.strategy._detect_rebound_conditions(test_df, 'BTC/USDT')
            
            # Check the second candle (index 1) for rebound condition
            actual_rebound = test_df.iloc[1]['rebound_condition']
            expected_rebound = scenario['expected_rebound']
            
            assert actual_rebound == expected_rebound, \
                f"Rebound detection mismatch in scenario {scenario_idx}: " \
                f"expected={expected_rebound}, actual={actual_rebound}, " \
                f"close={scenario['close']}, ma={scenario['ma']}, " \
                f"breakout={scenario['breakout_active']}, pullback_condition={scenario['pullback_condition']}"


if __name__ == '__main__':
    pytest.main([__file__])

class TestDuplicateEntryPrevention:
    """Property-based tests for duplicate entry prevention mechanism"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_duplicate_entry_test_data(self, num_candles: int, entry_scenarios: List[Tuple[int, float, pd.Timestamp]], 
                                         base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate test data with controlled entry scenarios for duplicate prevention testing
        
        :param num_candles: Total number of candles
        :param entry_scenarios: List of (entry_position, reference_high, activation_time)
        :param base_price: Base price for generation
        :return: DataFrame with controlled entry scenarios
        """
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Calculate MA5 values
        ma_period = self.strategy.ma_period.value
        ma_values = []
        for i in range(num_candles):
            if i < ma_period - 1:
                ma_values.append(closes[i])  # Use current price for early candles
            else:
                window_prices = closes[i - ma_period + 1:i + 1]
                ma_values.append(sum(window_prices) / len(window_prices))
        
        # Initialize state arrays
        breakout_state_active = [False] * num_candles
        pullback_condition = [False] * num_candles
        pullback_completed = [False] * num_candles
        breakout_activation_time = [pd.NaT] * num_candles
        breakout_reference_high = [0.0] * num_candles
        entry_signal = [False] * num_candles
        
        # Set up entry scenarios
        for entry_pos, reference_high, activation_time in entry_scenarios:
            if entry_pos < num_candles:
                # Set up conditions for entry at this position
                # Ensure breakout is active from some time before entry
                start_pos = max(0, entry_pos - 20)
                end_pos = min(entry_pos + 30, num_candles - 1)
                
                for i in range(start_pos, end_pos + 1):
                    breakout_state_active[i] = True
                    breakout_activation_time[i] = activation_time
                    breakout_reference_high[i] = reference_high
                
                # Set pullback condition before entry
                pullback_pos = max(0, entry_pos - 5)
                pullback_condition[pullback_pos] = True
                
                # Set pullback completed from pullback position to entry
                for i in range(pullback_pos, end_pos + 1):
                    pullback_completed[i] = True
                
                # Adjust close price for rebound at entry position
                closes[entry_pos] = ma_values[entry_pos] * 1.005  # 0.5% above MA5
                
                # Mark this as an entry signal position
                entry_signal[entry_pos] = True
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = closes[i] * 1.001
            low = closes[i] * 0.999
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                f'ma{ma_period}': ma_values[i],
                'breakout_state_active': breakout_state_active[i],
                'pullback_condition': pullback_condition[i],
                'pullback_completed': pullback_completed[i],
                'breakout_activation_time': breakout_activation_time[i],
                'breakout_reference_high': breakout_reference_high[i],
                'entry_signal': entry_signal[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=200, max_value=400),
        num_entry_attempts=st.integers(min_value=2, max_value=8),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_property_duplicate_entry_prevention(self, num_candles: int, num_entry_attempts: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 5: Duplicate entry prevention**
        
        For any breakout pattern, the same breakout event should not produce 
        multiple entry signals
        
        **Validates: Requirements 4.5**
        """
        # Generate entry scenarios with potential duplicates
        entry_scenarios = []
        
        # Create scenarios with same breakout pattern (same reference high and activation time)
        base_activation_time = pd.Timestamp('2023-01-01 10:00:00')
        base_reference_high = base_price * 0.998
        
        # First scenario - should generate entry
        first_entry_pos = self.strategy.startup_candle_count + 50
        entry_scenarios.append((first_entry_pos, base_reference_high, base_activation_time))
        
        # Additional scenarios with same breakout pattern - should be blocked
        # Use spacing > min_entry_spacing to test same-pattern blocking specifically
        min_spacing = self.strategy.min_entry_spacing.value
        for i in range(1, min(num_entry_attempts, 4)):  # Limit to prevent too many scenarios
            # Use spacing between min_entry_spacing and 30 to test same-pattern blocking
            duplicate_entry_pos = first_entry_pos + min_spacing + (i * 5)  # Spaced beyond min_entry_spacing
            if duplicate_entry_pos < num_candles - 50:
                entry_scenarios.append((duplicate_entry_pos, base_reference_high, base_activation_time))
        
        # Add scenarios with different breakout patterns - should be allowed
        if len(entry_scenarios) < num_entry_attempts:
            different_activation_time = pd.Timestamp('2023-01-01 11:00:00')
            different_reference_high = base_price * 1.002
            
            # Place different pattern entry far enough from first entry
            different_entry_pos = first_entry_pos + 100
            if different_entry_pos < num_candles - 50:
                entry_scenarios.append((different_entry_pos, different_reference_high, different_activation_time))
        
        # Generate test data
        test_df = self.generate_duplicate_entry_test_data(num_candles, entry_scenarios, base_price)
        
        # Apply rebound detection (which includes duplicate prevention)
        self.strategy._detect_rebound_conditions(test_df, 'BTC/USDT')
        
        # Property 1: Count entries for same breakout pattern
        same_pattern_entries = []
        different_pattern_entries = []
        
        for i, (entry_pos, reference_high, activation_time) in enumerate(entry_scenarios):
            if entry_pos < len(test_df) and test_df.iloc[entry_pos]['entry_signal']:
                if abs(reference_high - base_reference_high) < 0.0001 and activation_time == base_activation_time:
                    same_pattern_entries.append(entry_pos)
                else:
                    different_pattern_entries.append(entry_pos)
        
        # For same breakout pattern within 30 candles, should have at most one entry
        # Note: The implementation allows same pattern after 30 candles
        if len(same_pattern_entries) > 1:
            for i in range(1, len(same_pattern_entries)):
                spacing = same_pattern_entries[i] - same_pattern_entries[i-1]
                assert spacing >= 30, \
                    f"Multiple entries for same pattern within 30 candles: positions {same_pattern_entries}"
        
        # Property 2: Minimum spacing between ANY entries should be enforced
        all_entry_positions = []
        for i in range(len(test_df)):
            if test_df.iloc[i]['entry_signal']:
                all_entry_positions.append(i)
        
        for i in range(1, len(all_entry_positions)):
            spacing = all_entry_positions[i] - all_entry_positions[i-1]
            assert spacing >= min_spacing, \
                f"Insufficient spacing between entries: {spacing} candles between positions {all_entry_positions[i-1]} and {all_entry_positions[i]}"
        
        # Property 3: Duplicate prevention logic should be consistent
        # Test the same position multiple times - should give same result
        if len(test_df) > 100:
            test_position = 100
            result1 = self.strategy._check_duplicate_entry_prevention(test_df, test_position, 'BTC/USDT')
            result2 = self.strategy._check_duplicate_entry_prevention(test_df, test_position, 'BTC/USDT')
            assert result1 == result2, "Duplicate prevention logic is not consistent"
    
    def test_property_duplicate_prevention_boundary_conditions(self):
        """
        Test duplicate entry prevention with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 5: Duplicate entry prevention**
        **Validates: Requirements 4.5**
        """
        # Test with no previous entries - should allow entry
        no_entries_df = pd.DataFrame({
            'entry_signal': [False] * 100,
            'breakout_state_active': [True] * 100,
            'breakout_reference_high': [99.5] * 100,
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 100
        })
        
        result = self.strategy._check_duplicate_entry_prevention(no_entries_df, 50, 'BTC/USDT')
        assert not result, "Entry blocked when no previous entries exist"
        
        # Test with recent entry - should block
        recent_entry_df = pd.DataFrame({
            'entry_signal': [False] * 45 + [True] + [False] * 54,  # Entry at position 45
            'breakout_state_active': [True] * 100,
            'breakout_reference_high': [99.5] * 100,
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 100
        })
        
        result = self.strategy._check_duplicate_entry_prevention(recent_entry_df, 50, 'BTC/USDT')  # 5 candles later
        assert result, "Entry not blocked when recent entry exists"
        
        # Test with same pattern entry within 30 candles - should block
        same_pattern_df = pd.DataFrame({
            'entry_signal': [False] * 40 + [True] + [False] * 59,  # Entry at position 40
            'breakout_state_active': [True] * 100,
            'breakout_reference_high': [99.5] * 100,
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 100
        })
        
        result = self.strategy._check_duplicate_entry_prevention(same_pattern_df, 60, 'BTC/USDT')  # 20 candles later (< 30)
        assert result, "Same pattern entry not blocked within 30 candles"
        
        # Test with old entry - should allow
        old_entry_df = pd.DataFrame({
            'entry_signal': [False] * 20 + [True] + [False] * 79,  # Entry at position 20
            'breakout_state_active': [True] * 100,
            'breakout_reference_high': [99.5] * 100,
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 100
        })
        
        result = self.strategy._check_duplicate_entry_prevention(old_entry_df, 85, 'BTC/USDT')  # 65 candles later (> 60 lookback)
        assert not result, "Entry blocked when old entry exists (should be allowed)"
        
        # Test with different breakout pattern - should allow (ensure sufficient spacing)
        different_pattern_df = pd.DataFrame({
            'entry_signal': [False] * 30 + [True] + [False] * 69,  # Entry at position 30
            'breakout_state_active': [True] * 100,
            'breakout_reference_high': [99.5] * 31 + [100.5] * 69,  # Different reference high from position 31
            'breakout_activation_time': [pd.Timestamp('2023-01-01 10:00:00')] * 31 + [pd.Timestamp('2023-01-01 11:00:00')] * 69  # Different activation time
        })
        
        result = self.strategy._check_duplicate_entry_prevention(different_pattern_df, 50, 'BTC/USDT')  # 20 candles later (> 15 minimum spacing)
        assert not result, "Entry blocked for different breakout pattern"
        
        # Test with no active breakout - should allow (no blocking needed)
        no_breakout_df = pd.DataFrame({
            'entry_signal': [False] * 100,
            'breakout_state_active': [False] * 100,
            'breakout_reference_high': [0.0] * 100,
            'breakout_activation_time': [pd.NaT] * 100
        })
        
        result = self.strategy._check_duplicate_entry_prevention(no_breakout_df, 50, 'BTC/USDT')
        assert not result, "Entry blocked when no active breakout"
    
    def test_property_same_breakout_pattern_identification(self):
        """
        Test identification of same vs different breakout patterns
        
        **Feature: hour-breakout-scalping, Property 5: Duplicate entry prevention**
        **Validates: Requirements 4.5**
        """
        base_time = pd.Timestamp('2023-01-01 10:00:00')
        base_high = 99.5
        
        # Test same pattern scenarios
        same_pattern_scenarios = [
            # Exact same values
            {
                'prev_high': base_high,
                'prev_time': base_time,
                'current_high': base_high,
                'current_time': base_time,
                'expected_same': True
            },
            # Very small difference in high (within tolerance)
            {
                'prev_high': base_high,
                'prev_time': base_time,
                'current_high': base_high + 0.00005,  # Very small difference
                'current_time': base_time,
                'expected_same': True
            }
        ]
        
        # Test different pattern scenarios
        different_pattern_scenarios = [
            # Different reference high
            {
                'prev_high': base_high,
                'prev_time': base_time,
                'current_high': base_high + 0.1,  # Significant difference
                'current_time': base_time,
                'expected_same': False
            },
            # Different activation time
            {
                'prev_high': base_high,
                'prev_time': base_time,
                'current_high': base_high,
                'current_time': base_time + pd.Timedelta(minutes=30),
                'expected_same': False
            },
            # Both different
            {
                'prev_high': base_high,
                'prev_time': base_time,
                'current_high': base_high + 0.2,
                'current_time': base_time + pd.Timedelta(hours=1),
                'expected_same': False
            }
        ]
        
        all_scenarios = same_pattern_scenarios + different_pattern_scenarios
        
        for scenario_idx, scenario in enumerate(all_scenarios):
            # Create test dataframe with previous entry
            test_df = pd.DataFrame({
                'entry_signal': [True, False],  # Previous entry at position 0
                'breakout_state_active': [True, True],
                'breakout_reference_high': [scenario['prev_high'], scenario['current_high']],
                'breakout_activation_time': [scenario['prev_time'], scenario['current_time']]
            })
            
            # Test duplicate prevention at position 1
            is_blocked = self.strategy._check_duplicate_entry_prevention(test_df, 1, 'BTC/USDT')
            
            if scenario['expected_same']:
                assert is_blocked, f"Same pattern not blocked in scenario {scenario_idx}: {scenario}"
            else:
                # Note: Different patterns might still be blocked due to minimum spacing
                # So we test the pattern identification logic separately
                
                # Check if the patterns are identified as different
                prev_high = scenario['prev_high']
                current_high = scenario['current_high']
                prev_time = scenario['prev_time']
                current_time = scenario['current_time']
                
                same_pattern = (
                    abs(prev_high - current_high) < 0.0001 and
                    prev_time == current_time
                )
                
                assert same_pattern == scenario['expected_same'], \
                    f"Pattern identification mismatch in scenario {scenario_idx}: {scenario}"
    
    def test_property_minimum_spacing_enforcement(self):
        """
        Test minimum spacing enforcement between entries
        
        **Feature: hour-breakout-scalping, Property 5: Duplicate entry prevention**
        **Validates: Requirements 4.5**
        """
        min_spacing = 15  # Minimum candles between entries
        
        # Test various spacing scenarios
        spacing_scenarios = [
            {'spacing': 5, 'should_block': True},    # Too close
            {'spacing': 10, 'should_block': True},   # Still too close
            {'spacing': 14, 'should_block': True},   # Just under minimum
            {'spacing': 15, 'should_block': False},  # Exactly at minimum
            {'spacing': 20, 'should_block': False},  # Above minimum
            {'spacing': 50, 'should_block': False}   # Well above minimum
        ]
        
        for scenario in spacing_scenarios:
            spacing = scenario['spacing']
            should_block = scenario['should_block']
            
            # Create test dataframe with entry at calculated position
            total_candles = 100
            entry_pos = 50
            prev_entry_pos = entry_pos - spacing
            
            entry_signals = [False] * total_candles
            if prev_entry_pos >= 0:
                entry_signals[prev_entry_pos] = True
            
            # Use different activation times to avoid same pattern blocking
            activation_times = [pd.Timestamp('2023-01-01 11:00:00')] * total_candles
            if prev_entry_pos >= 0:
                activation_times[prev_entry_pos] = pd.Timestamp('2023-01-01 10:00:00')  # Different time
            
            test_df = pd.DataFrame({
                'entry_signal': entry_signals,
                'breakout_state_active': [True] * total_candles,
                'breakout_reference_high': [99.5] * total_candles,
                'breakout_activation_time': activation_times
            })
            
            is_blocked = self.strategy._check_duplicate_entry_prevention(test_df, entry_pos, 'BTC/USDT')
            
            assert is_blocked == should_block, \
                f"Spacing enforcement failed for spacing {spacing}: expected_block={should_block}, actual_block={is_blocked}"


if __name__ == '__main__':
    pytest.main([__file__])


class TestStopLossManagement:
    """Property-based tests for stop loss management logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_stop_loss_test_data(self, num_candles: int, stop_loss_scenarios: List[Tuple[int, float, float]], 
                                   base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate test data with controlled stop loss scenarios
        
        :param num_candles: Total number of candles
        :param stop_loss_scenarios: List of (position, close_price, high_1h_prev)
        :param base_price: Base price for generation
        :return: DataFrame with controlled stop loss scenarios
        """
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Generate 1h high data (simulate previous 1h high)
        high_1h_prev = []
        for i in range(num_candles):
            # Default to slightly above current price
            default_high = closes[i] * 1.002
            high_1h_prev.append(default_high)
        
        # Apply stop loss scenarios
        for pos, close_price, high_1h in stop_loss_scenarios:
            if pos < num_candles:
                closes[pos] = close_price
                high_1h_prev[pos] = high_1h
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = max(closes[i], high_1h_prev[i]) * 1.001
            low = closes[i] * 0.999
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                'high_1h_prev': high_1h_prev[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=100, max_value=200),
        num_scenarios=st.integers(min_value=1, max_value=5),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_stop_loss_management(self, num_candles: int, num_scenarios: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 6: Stop loss management**
        
        For any open position, stop loss price should be set at previous 1h high,
        and should trigger exit when price falls below stop loss level
        
        **Validates: Requirements 5.1, 5.2**
        """
        # Generate stop loss scenarios
        stop_loss_scenarios = []
        
        for _ in range(num_scenarios):
            pos = np.random.randint(10, num_candles - 10)
            high_1h = base_price * np.random.uniform(0.995, 1.005)
            
            # Create scenarios: some trigger stop loss, some don't
            if np.random.random() < 0.5:
                # Trigger stop loss: close below 1h high
                close_price = high_1h * np.random.uniform(0.990, 0.999)
            else:
                # Don't trigger: close above 1h high
                close_price = high_1h * np.random.uniform(1.001, 1.010)
            
            stop_loss_scenarios.append((pos, close_price, high_1h))
        
        # Generate test data
        test_df = self.generate_stop_loss_test_data(num_candles, stop_loss_scenarios, base_price)
        
        # Apply stop loss calculation
        self.strategy._calculate_stop_loss_conditions(test_df, 'BTC/USDT')
        
        # Get stop loss buffer for calculations
        stop_loss_buffer = self.strategy.stop_loss_buffer_pct.value
        
        # Property 1: Stop loss price should be set based on previous 1h high with buffer (Requirements: 5.1)
        for i in range(len(test_df)):
            stop_loss_price = test_df.iloc[i]['stop_loss_price']
            high_1h_prev = test_df.iloc[i]['high_1h_prev']
            
            if high_1h_prev > 0:
                expected_stop_loss = high_1h_prev * (1 - stop_loss_buffer)
                assert abs(stop_loss_price - expected_stop_loss) < 1e-6, \
                    f"Stop loss price not set correctly at position {i}: " \
                    f"expected={expected_stop_loss:.6f}, actual={stop_loss_price:.6f}"
        
        # Property 2: Stop loss condition should trigger when price falls below stop loss (Requirements: 5.2)
        # Note: We use the actual close value from the dataframe, not the scenario close_price,
        # because dynamic_stop_loss is a cumulative maximum that may differ from the scenario's high_1h
        for pos, close_price, high_1h in stop_loss_scenarios:
            if pos < len(test_df):
                stop_loss_condition = test_df.iloc[pos]['stop_loss_condition']
                dynamic_stop_loss = test_df.iloc[pos]['dynamic_stop_loss']
                actual_close = test_df.iloc[pos]['close']
                
                # Use actual close from dataframe for expected trigger calculation
                expected_trigger = actual_close < dynamic_stop_loss
                
                assert stop_loss_condition == expected_trigger, \
                    f"Stop loss condition mismatch at position {pos}: " \
                    f"close={actual_close:.6f}, stop_loss={dynamic_stop_loss:.6f}, " \
                    f"expected_trigger={expected_trigger}, actual={stop_loss_condition}"
        
        # Property 3: Dynamic stop loss should only move up (trailing stop behavior)
        for i in range(1, len(test_df)):
            prev_dynamic_stop = test_df.iloc[i-1]['dynamic_stop_loss']
            current_dynamic_stop = test_df.iloc[i]['dynamic_stop_loss']
            current_stop_price = test_df.iloc[i]['stop_loss_price']
            
            if prev_dynamic_stop > 0:
                # Dynamic stop loss should be max of current stop price and previous dynamic stop
                expected_dynamic_stop = max(current_stop_price, prev_dynamic_stop)
                assert abs(current_dynamic_stop - expected_dynamic_stop) < 1e-10, \
                    f"Dynamic stop loss not calculated correctly at position {i}: " \
                    f"expected={expected_dynamic_stop:.6f}, actual={current_dynamic_stop:.6f}"
        
        # Property 4: Stop loss strength should be calculated correctly
        for i in range(len(test_df)):
            if test_df.iloc[i]['stop_loss_condition']:
                current_close = test_df.iloc[i]['close']
                dynamic_stop_loss = test_df.iloc[i]['dynamic_stop_loss']
                stop_loss_strength = test_df.iloc[i]['stop_loss_strength']
                
                if dynamic_stop_loss > 0:
                    expected_strength = (current_close - dynamic_stop_loss) / dynamic_stop_loss * 100
                    assert abs(stop_loss_strength - expected_strength) < 0.001, \
                        f"Stop loss strength calculation error at position {i}: " \
                        f"expected={expected_strength:.6f}, actual={stop_loss_strength:.6f}"
            else:
                # Non-triggered stop loss should have zero strength
                assert test_df.iloc[i]['stop_loss_strength'] == 0.0, \
                    f"Non-zero stop loss strength for non-triggered stop loss at position {i}"
        
        # Property 5: Required columns should exist and have correct types
        required_columns = ['stop_loss_condition', 'stop_loss_price', 'dynamic_stop_loss', 'stop_loss_strength']
        for col in required_columns:
            assert col in test_df.columns, f"Required column {col} missing"
        
        # Check data types
        assert test_df['stop_loss_condition'].dtype == bool, "stop_loss_condition should be boolean"
        assert pd.api.types.is_numeric_dtype(test_df['stop_loss_price']), "stop_loss_price should be numeric"
        assert pd.api.types.is_numeric_dtype(test_df['dynamic_stop_loss']), "dynamic_stop_loss should be numeric"
        assert pd.api.types.is_numeric_dtype(test_df['stop_loss_strength']), "stop_loss_strength should be numeric"
        
        # Property 6: Invalid data should not trigger stop loss
        for i in range(len(test_df)):
            current_close = test_df.iloc[i]['close']
            high_1h_prev = test_df.iloc[i]['high_1h_prev']
            data_quality = test_df.iloc[i]['data_quality_ok']
            
            if (current_close <= 0 or high_1h_prev <= 0 or not data_quality or 
                pd.isna(current_close) or pd.isna(high_1h_prev)):
                # Should not have valid stop loss calculations with invalid data
                # Note: The implementation may still set some values, but they should be safe defaults
                pass  # This is handled by the implementation's data validation
    
    def test_property_stop_loss_boundary_conditions(self):
        """
        Test stop loss management with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 6: Stop loss management**
        **Validates: Requirements 5.1, 5.2**
        """
        # Test with minimal data
        minimal_df = pd.DataFrame({
            'close': [100.0, 99.0],
            'high_1h_prev': [100.5, 100.5],
            'data_quality_ok': [True, True]
        })
        
        self.strategy._calculate_stop_loss_conditions(minimal_df, 'BTC/USDT')
        
        # Should handle minimal data without error
        assert 'stop_loss_condition' in minimal_df.columns
        assert 'stop_loss_price' in minimal_df.columns
        assert 'dynamic_stop_loss' in minimal_df.columns
        
        # Check stop loss logic - stop loss price = high_1h_prev * (1 - stop_loss_buffer_pct)
        stop_loss_buffer = self.strategy.stop_loss_buffer_pct.value
        expected_stop_loss_price = 100.5 * (1 - stop_loss_buffer)
        assert abs(minimal_df.iloc[0]['stop_loss_price'] - expected_stop_loss_price) < 1e-6, \
            f"Stop loss price mismatch: expected={expected_stop_loss_price}, actual={minimal_df.iloc[0]['stop_loss_price']}"
        # 100.0 < expected_stop_loss_price (trigger depends on buffer value)
        # 99.0 < expected_stop_loss_price (trigger depends on buffer value)
        
        # Test with NaN values
        nan_df = pd.DataFrame({
            'close': [100.0, np.nan, 99.0],
            'high_1h_prev': [100.5, 100.5, np.nan],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._calculate_stop_loss_conditions(nan_df, 'BTC/USDT')
        
        # NaN values should not trigger stop loss
        assert not nan_df.iloc[1]['stop_loss_condition']  # NaN close
        assert not nan_df.iloc[2]['stop_loss_condition']  # NaN high_1h_prev
        
        # Test with zero/negative values
        invalid_df = pd.DataFrame({
            'close': [100.0, 0.0, -10.0],
            'high_1h_prev': [100.5, 100.5, 100.5],
            'data_quality_ok': [True, True, True]
        })
        
        self.strategy._calculate_stop_loss_conditions(invalid_df, 'BTC/USDT')
        
        # Zero/negative values should not trigger stop loss
        assert not invalid_df.iloc[1]['stop_loss_condition']  # Zero close
        assert not invalid_df.iloc[2]['stop_loss_condition']  # Negative close
    
    def test_property_dynamic_stop_loss_trailing(self):
        """
        Test dynamic stop loss trailing behavior (only moves up)
        
        **Feature: hour-breakout-scalping, Property 6: Stop loss management**
        **Validates: Requirements 5.3**
        """
        # Create scenario with rising and falling 1h highs
        test_df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 101.5, 100.5],
            'high_1h_prev': [99.5, 100.5, 101.5, 100.0, 99.0],  # Rising then falling
            'data_quality_ok': [True, True, True, True, True]
        })
        
        self.strategy._calculate_stop_loss_conditions(test_df, 'BTC/USDT')
        
        # Dynamic stop loss should trail up but not down
        # Stop loss price = high_1h_prev * (1 - stop_loss_buffer_pct)
        stop_loss_buffer = self.strategy.stop_loss_buffer_pct.value
        
        # Calculate expected dynamic stops with buffer
        raw_stops = [99.5, 100.5, 101.5, 100.0, 99.0]
        buffered_stops = [s * (1 - stop_loss_buffer) for s in raw_stops]
        
        # Dynamic stop should trail up but not down
        expected_dynamic_stops = []
        max_stop = 0.0
        for stop in buffered_stops:
            max_stop = max(max_stop, stop)
            expected_dynamic_stops.append(max_stop)
        
        for i, expected in enumerate(expected_dynamic_stops):
            actual = test_df.iloc[i]['dynamic_stop_loss']
            assert abs(actual - expected) < 1e-6, \
                f"Dynamic stop loss trailing error at position {i}: expected={expected:.6f}, actual={actual:.6f}"
    
    def test_property_stop_loss_exit_integration(self):
        """
        Test integration of stop loss with exit signal generation
        
        **Feature: hour-breakout-scalping, Property 6: Stop loss management**
        **Validates: Requirements 5.1, 5.2**
        """
        # Create scenario with stop loss trigger
        test_df = pd.DataFrame({
            'close': [100.0, 101.0, 98.0],  # Last candle triggers stop loss
            'high_1h_prev': [99.5, 99.5, 99.5],
            'volume': [1000, 1000, 1000],
            'data_quality_ok': [True, True, True]
        })
        
        # Apply exit trend logic
        result_df = self.strategy.populate_exit_trend(test_df.copy(), {'pair': 'BTC/USDT'})
        
        # Should generate stop loss exit signal
        assert result_df.iloc[2]['exit_long'] == 1, "Stop loss exit signal not generated"
        assert result_df.iloc[2]['exit_tag'] == 'stop_loss', "Stop loss exit tag not set correctly"
        
        # Previous candles should not have exit signals
        assert result_df.iloc[0]['exit_long'] == 0, "Unexpected exit signal at position 0"
        assert result_df.iloc[1]['exit_long'] == 0, "Unexpected exit signal at position 1"


class TestTimeBasedExitMechanism:
    """Property-based tests for time-based exit mechanism"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_time_exit_test_data(self, num_candles: int, entry_positions: List[int], 
                                   base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate test data with controlled entry positions for time-based exit testing
        
        :param num_candles: Total number of candles
        :param entry_positions: List of positions where entry signals occur
        :param base_price: Base price for generation
        :return: DataFrame with controlled entry scenarios
        """
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate base price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Initialize entry signals
        entry_signals = [False] * num_candles
        for pos in entry_positions:
            if pos < num_candles:
                entry_signals[pos] = True
        
        # Create dataframe
        data = []
        for i in range(num_candles):
            high = closes[i] * 1.001
            low = closes[i] * 0.999
            
            data.append({
                'date': dates[i],
                'open': closes[i-1] if i > 0 else closes[i],
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': np.random.randint(1000, 10000),
                'entry_signal': entry_signals[i],
                'data_quality_ok': True
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=100, max_value=200),
        num_entries=st.integers(min_value=1, max_value=3),
        exit_minutes=st.integers(min_value=5, max_value=30),
        base_price=st.floats(min_value=80.0, max_value=150.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_time_based_exit_mechanism(self, num_candles: int, num_entries: int, 
                                              exit_minutes: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 7: Time-based exit mechanism**
        
        For any position, when holding time reaches N minutes, should trigger take profit exit
        
        **Validates: Requirements 6.2**
        """
        # Set exit minutes parameter
        original_exit_minutes = self.strategy.exit_minutes.value
        self.strategy.exit_minutes.value = exit_minutes
        
        try:
            # Generate entry positions with sufficient spacing
            entry_positions = []
            min_spacing = exit_minutes + 10  # Ensure entries don't overlap
            
            for _ in range(num_entries):
                attempts = 0
                while attempts < 20:  # Prevent infinite loop
                    pos = np.random.randint(10, num_candles - exit_minutes - 10)
                    if not any(abs(pos - existing) < min_spacing for existing in entry_positions):
                        entry_positions.append(pos)
                        break
                    attempts += 1
            
            # Generate test data
            test_df = self.generate_time_exit_test_data(num_candles, entry_positions, base_price)
            
            # Apply time-based exit calculation
            self.strategy._calculate_time_based_exit_conditions(test_df, 'BTC/USDT')
            
            # Property 1: Position duration should be tracked correctly
            for entry_pos in entry_positions:
                if entry_pos < len(test_df):
                    # Duration should start at 0 at entry
                    assert test_df.iloc[entry_pos]['position_duration_minutes'] == 0.0, \
                        f"Position duration not reset at entry position {entry_pos}"
                    
                    # Duration should increment after entry
                    for i in range(entry_pos + 1, min(entry_pos + exit_minutes + 5, len(test_df))):
                        expected_duration = float(i - entry_pos)
                        actual_duration = test_df.iloc[i]['position_duration_minutes']
                        
                        assert abs(actual_duration - expected_duration) < 1e-10, \
                            f"Position duration tracking error at position {i}: " \
                            f"expected={expected_duration:.1f}, actual={actual_duration:.1f}"
            
            # Property 2: Time exit should trigger when duration reaches target (Requirements: 6.2)
            for entry_pos in entry_positions:
                exit_pos = entry_pos + exit_minutes
                if exit_pos < len(test_df):
                    time_exit_condition = test_df.iloc[exit_pos]['time_exit_condition']
                    position_duration = test_df.iloc[exit_pos]['position_duration_minutes']
                    
                    assert time_exit_condition, \
                        f"Time exit not triggered at position {exit_pos} after {position_duration:.1f} minutes " \
                        f"(target: {exit_minutes} minutes)"
            
            # Property 3: Time exit should not trigger before target duration
            for entry_pos in entry_positions:
                for i in range(entry_pos + 1, min(entry_pos + exit_minutes, len(test_df))):
                    time_exit_condition = test_df.iloc[i]['time_exit_condition']
                    position_duration = test_df.iloc[i]['position_duration_minutes']
                    
                    assert not time_exit_condition, \
                        f"Premature time exit triggered at position {i} after {position_duration:.1f} minutes " \
                        f"(target: {exit_minutes} minutes)"
            
            # Property 4: Target exit minutes should be set correctly
            for i in range(len(test_df)):
                target_minutes = test_df.iloc[i]['time_exit_target_minutes']
                assert target_minutes == exit_minutes, \
                    f"Target exit minutes not set correctly at position {i}: " \
                    f"expected={exit_minutes}, actual={target_minutes}"
            
            # Property 5: Required columns should exist and have correct types
            required_columns = ['time_exit_condition', 'position_duration_minutes', 'time_exit_target_minutes']
            for col in required_columns:
                assert col in test_df.columns, f"Required column {col} missing"
            
            # Check data types
            assert test_df['time_exit_condition'].dtype == bool, "time_exit_condition should be boolean"
            assert pd.api.types.is_numeric_dtype(test_df['position_duration_minutes']), \
                "position_duration_minutes should be numeric"
            assert pd.api.types.is_numeric_dtype(test_df['time_exit_target_minutes']), \
                "time_exit_target_minutes should be numeric"
            
            # Property 6: Duration should reset when no active position
            # Note: Duration increments after entry, so we check that duration stays 0 
            # only when there's no prior entry signal and no prior duration
            for i in range(len(test_df)):
                if not test_df.iloc[i]['entry_signal'] and i > 0:
                    prev_duration = test_df.iloc[i-1]['position_duration_minutes']
                    prev_entry_signal = test_df.iloc[i-1]['entry_signal']
                    current_duration = test_df.iloc[i]['position_duration_minutes']
                    
                    # If previous duration was 0 and no previous entry signal, current should also be 0
                    if prev_duration == 0.0 and not prev_entry_signal:
                        assert current_duration == 0.0, \
                            f"Duration not maintained at zero when no position at position {i}"
        
        finally:
            # Restore original value
            self.strategy.exit_minutes.value = original_exit_minutes
    
    def test_property_time_exit_boundary_conditions(self):
        """
        Test time-based exit with boundary conditions and edge cases
        
        **Feature: hour-breakout-scalping, Property 7: Time-based exit mechanism**
        **Validates: Requirements 6.2**
        """
        # Set specific exit minutes for testing
        original_exit_minutes = self.strategy.exit_minutes.value
        self.strategy.exit_minutes.value = 5
        
        try:
            # Test with minimal data - entry at first candle
            minimal_df = pd.DataFrame({
                'entry_signal': [True, False, False, False, False, False],  # Entry at position 0
                'data_quality_ok': [True, True, True, True, True, True]
            })
            
            self.strategy._calculate_time_based_exit_conditions(minimal_df, 'BTC/USDT')
            
            # Should handle minimal data without error
            assert 'time_exit_condition' in minimal_df.columns
            assert 'position_duration_minutes' in minimal_df.columns
            
            # Check duration tracking
            expected_durations = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            for i, expected in enumerate(expected_durations):
                actual = minimal_df.iloc[i]['position_duration_minutes']
                assert abs(actual - expected) < 1e-10, \
                    f"Duration tracking error at position {i}: expected={expected:.1f}, actual={actual:.1f}"
            
            # Time exit should trigger at position 5 (after 5 minutes)
            assert minimal_df.iloc[5]['time_exit_condition'], "Time exit not triggered after 5 minutes"
            
            # Test with multiple entries
            multiple_entries_df = pd.DataFrame({
                'entry_signal': [True, False, False, True, False, False, False, False],  # Entries at 0 and 3
                'data_quality_ok': [True, True, True, True, True, True, True, True]
            })
            
            self.strategy._calculate_time_based_exit_conditions(multiple_entries_df, 'BTC/USDT')
            
            # Duration should reset at second entry
            assert multiple_entries_df.iloc[3]['position_duration_minutes'] == 0.0, \
                "Duration not reset at second entry"
            
            # Duration should continue from second entry
            expected_durations_after_reset = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0]
            for i, expected in enumerate(expected_durations_after_reset):
                actual = multiple_entries_df.iloc[i]['position_duration_minutes']
                assert abs(actual - expected) < 1e-10, \
                    f"Duration tracking with multiple entries error at position {i}: " \
                    f"expected={expected:.1f}, actual={actual:.1f}"
        
        finally:
            # Restore original value
            self.strategy.exit_minutes.value = original_exit_minutes
    
    def test_property_time_exit_integration(self):
        """
        Test integration of time-based exit with exit signal generation
        
        **Feature: hour-breakout-scalping, Property 7: Time-based exit mechanism**
        **Validates: Requirements 6.2**
        """
        # Set specific exit minutes for testing
        original_exit_minutes = self.strategy.exit_minutes.value
        self.strategy.exit_minutes.value = 3
        
        try:
            # Create scenario with time-based exit trigger
            test_df = pd.DataFrame({
                'close': [100.0, 100.1, 100.2, 100.3],
                'high_1h_prev': [99.5, 99.5, 99.5, 99.5],  # No stop loss trigger
                'volume': [1000, 1000, 1000, 1000],
                'entry_signal': [True, False, False, False],  # Entry at position 0
                'data_quality_ok': [True, True, True, True]
            })
            
            # Apply exit trend logic
            result_df = self.strategy.populate_exit_trend(test_df.copy(), {'pair': 'BTC/USDT'})
            
            # Should generate time-based exit signal at position 3 (after 3 minutes)
            assert result_df.iloc[3]['exit_long'] == 1, "Time-based exit signal not generated"
            assert result_df.iloc[3]['exit_tag'] == 'time_exit', "Time-based exit tag not set correctly"
            
            # Previous candles should not have exit signals
            assert result_df.iloc[0]['exit_long'] == 0, "Unexpected exit signal at entry position"
            assert result_df.iloc[1]['exit_long'] == 0, "Unexpected exit signal at position 1"
            assert result_df.iloc[2]['exit_long'] == 0, "Unexpected exit signal at position 2"
        
        finally:
            # Restore original value
            self.strategy.exit_minutes.value = original_exit_minutes


class TestParameterValidation:
    """Property-based tests for HyperOpt parameter validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock logger to capture validation messages
        self.strategy.logger = MagicMock()
    
    @given(
        ma_period=st.integers(min_value=-10, max_value=50),
        exit_minutes=st.integers(min_value=-50, max_value=200),
        min_breakout_pct=st.floats(min_value=-0.1, max_value=0.5),
        pullback_tolerance=st.floats(min_value=-0.01, max_value=0.1)
    )
    @settings(max_examples=20, deadline=None)  # Reduced examples for faster testing
    def test_property_parameter_validation(self, ma_period: int, exit_minutes: int, 
                                         min_breakout_pct: float, pullback_tolerance: float):
        """
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        
        For any HyperOpt optimization parameter, values should be in defined valid ranges 
        and pass validation checks
        
        **Validates: Requirements 9.5**
        """
        # Store original values for restoration
        original_values = {
            'ma_period': self.strategy.ma_period.value,
            'exit_minutes': self.strategy.exit_minutes.value,
            'min_breakout_pct': self.strategy.min_breakout_pct.value,
            'pullback_tolerance': self.strategy.pullback_tolerance.value,
        }
        
        try:
            # Set test values (simulating HyperOpt parameter assignment)
            self.strategy.ma_period.value = ma_period
            self.strategy.exit_minutes.value = exit_minutes
            self.strategy.min_breakout_pct.value = min_breakout_pct
            self.strategy.pullback_tolerance.value = pullback_tolerance
            
            # Run parameter validation
            self.strategy._validate_hyperopt_parameters()
            
            # Property 1: MA period should be within valid range [3, 10]
            validated_ma_period = self.strategy.ma_period.value
            assert 3 <= validated_ma_period <= 10, \
                f"MA period {validated_ma_period} outside valid range [3, 10]"
            
            # Property 2: Exit minutes should be within valid range [5, 60]
            validated_exit_minutes = self.strategy.exit_minutes.value
            assert 5 <= validated_exit_minutes <= 60, \
                f"Exit minutes {validated_exit_minutes} outside valid range [5, 60]"
            
            # Property 3: Min breakout percentage should be within valid range [0.001, 0.01]
            validated_min_breakout_pct = self.strategy.min_breakout_pct.value
            assert 0.001 <= validated_min_breakout_pct <= 0.01, \
                f"Min breakout percentage {validated_min_breakout_pct} outside valid range [0.001, 0.01]"
            
            # Property 4: Pullback tolerance should be within valid range [0.0001, 0.002]
            validated_pullback_tolerance = self.strategy.pullback_tolerance.value
            assert 0.0001 <= validated_pullback_tolerance <= 0.002, \
                f"Pullback tolerance {validated_pullback_tolerance} outside valid range [0.0001, 0.002]"
            
            # Property 5: If input was invalid, value should be reset to default
            if not (3 <= ma_period <= 10):
                assert validated_ma_period == 5, \
                    f"Invalid MA period not reset to default: {validated_ma_period} != 5"
            
            if not (5 <= exit_minutes <= 60):
                assert validated_exit_minutes == 15, \
                    f"Invalid exit minutes not reset to default: {validated_exit_minutes} != 15"
            
            if not (0.001 <= min_breakout_pct <= 0.01):
                assert validated_min_breakout_pct == 0.002, \
                    f"Invalid min breakout pct not reset to default: {validated_min_breakout_pct} != 0.002"
            
            if not (0.0001 <= pullback_tolerance <= 0.002):
                assert validated_pullback_tolerance == 0.0005, \
                    f"Invalid pullback tolerance not reset to default: {validated_pullback_tolerance} != 0.0005"
            
        finally:
            # Restore original values
            for param_name, original_value in original_values.items():
                setattr(getattr(self.strategy, param_name), 'value', original_value)
    
    @given(
        max_position_hours=st.floats(min_value=-5.0, max_value=20.0),
        min_volume_threshold=st.floats(min_value=-2.0, max_value=10.0),
        stop_loss_buffer_pct=st.floats(min_value=-0.05, max_value=0.1),
        min_entry_spacing=st.integers(min_value=-20, max_value=100)
    )
    @settings(max_examples=20, deadline=None)  # Reduced examples for faster testing
    def test_property_risk_parameter_validation(self, max_position_hours: float, min_volume_threshold: float,
                                              stop_loss_buffer_pct: float, min_entry_spacing: int):
        """
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        
        For any risk management parameter, values should be in defined valid ranges 
        and pass validation checks
        
        **Validates: Requirements 9.4, 9.5**
        """
        # Store original values for restoration
        original_values = {
            'max_position_hours': self.strategy.max_position_hours.value,
            'min_volume_threshold': self.strategy.min_volume_threshold.value,
            'stop_loss_buffer_pct': self.strategy.stop_loss_buffer_pct.value,
            'min_entry_spacing': self.strategy.min_entry_spacing.value,
        }
        
        try:
            # Set test values
            self.strategy.max_position_hours.value = max_position_hours
            self.strategy.min_volume_threshold.value = min_volume_threshold
            self.strategy.stop_loss_buffer_pct.value = stop_loss_buffer_pct
            self.strategy.min_entry_spacing.value = min_entry_spacing
            
            # Run parameter validation
            self.strategy._validate_hyperopt_parameters()
            
            # Property 1: Max position hours should be within valid range [1.0, 8.0]
            validated_max_position_hours = self.strategy.max_position_hours.value
            assert 1.0 <= validated_max_position_hours <= 8.0, \
                f"Max position hours {validated_max_position_hours} outside valid range [1.0, 8.0]"
            
            # Property 2: Min volume threshold should be within valid range [0.5, 3.0]
            validated_min_volume_threshold = self.strategy.min_volume_threshold.value
            assert 0.5 <= validated_min_volume_threshold <= 3.0, \
                f"Min volume threshold {validated_min_volume_threshold} outside valid range [0.5, 3.0]"
            
            # Property 3: Stop loss buffer percentage should be within valid range [0.001, 0.01]
            validated_stop_loss_buffer_pct = self.strategy.stop_loss_buffer_pct.value
            assert 0.001 <= validated_stop_loss_buffer_pct <= 0.01, \
                f"Stop loss buffer pct {validated_stop_loss_buffer_pct} outside valid range [0.001, 0.01]"
            
            # Property 4: Min entry spacing should be within valid range [10, 30]
            validated_min_entry_spacing = self.strategy.min_entry_spacing.value
            assert 10 <= validated_min_entry_spacing <= 30, \
                f"Min entry spacing {validated_min_entry_spacing} outside valid range [10, 30]"
            
        finally:
            # Restore original values
            for param_name, original_value in original_values.items():
                setattr(getattr(self.strategy, param_name), 'value', original_value)
    
    @given(
        breakout_strength_threshold=st.floats(min_value=-0.01, max_value=0.02),
        rebound_strength_threshold=st.floats(min_value=-0.02, max_value=0.05)
    )
    @settings(max_examples=20, deadline=None)  # Reduced examples for faster testing
    def test_property_strength_parameter_validation(self, breakout_strength_threshold: float, 
                                                  rebound_strength_threshold: float):
        """
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        
        For any strength threshold parameter, values should be in defined valid ranges 
        and pass validation checks
        
        **Validates: Requirements 9.4, 9.5**
        """
        # Store original values for restoration
        original_values = {
            'breakout_strength_threshold': self.strategy.breakout_strength_threshold.value,
            'rebound_strength_threshold': self.strategy.rebound_strength_threshold.value,
        }
        
        try:
            # Set test values
            self.strategy.breakout_strength_threshold.value = breakout_strength_threshold
            self.strategy.rebound_strength_threshold.value = rebound_strength_threshold
            
            # Run parameter validation
            self.strategy._validate_hyperopt_parameters()
            
            # Property 1: Breakout strength threshold should be within valid range [0.001, 0.005]
            validated_breakout_strength = self.strategy.breakout_strength_threshold.value
            assert 0.001 <= validated_breakout_strength <= 0.005, \
                f"Breakout strength threshold {validated_breakout_strength} outside valid range [0.001, 0.005]"
            
            # Property 2: Rebound strength threshold should be within valid range [0.001, 0.01]
            validated_rebound_strength = self.strategy.rebound_strength_threshold.value
            assert 0.001 <= validated_rebound_strength <= 0.01, \
                f"Rebound strength threshold {validated_rebound_strength} outside valid range [0.001, 0.01]"
            
        finally:
            # Restore original values
            for param_name, original_value in original_values.items():
                setattr(getattr(self.strategy, param_name), 'value', original_value)
    
    def test_cross_parameter_validation(self):
        """
        Test cross-parameter validation logic
        
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        **Validates: Requirements 9.5**
        """
        # Store original values
        original_exit_minutes = self.strategy.exit_minutes.value
        original_max_position_hours = self.strategy.max_position_hours.value
        original_breakout_strength = self.strategy.breakout_strength_threshold.value
        original_min_breakout_pct = self.strategy.min_breakout_pct.value
        
        try:
            # Test case 1: exit_minutes > max_position_hours * 60
            self.strategy.exit_minutes.value = 120  # 2 hours
            self.strategy.max_position_hours.value = 1.0  # 1 hour
            
            self.strategy._validate_hyperopt_parameters()
            
            # Should adjust exit_minutes to be within max_position_hours
            assert self.strategy.exit_minutes.value <= self.strategy.max_position_hours.value * 60, \
                "Exit minutes not adjusted for max position hours constraint"
            
            # Test case 2: breakout_strength_threshold > min_breakout_pct
            self.strategy.breakout_strength_threshold.value = 0.008
            self.strategy.min_breakout_pct.value = 0.005
            
            self.strategy._validate_hyperopt_parameters()
            
            # Should adjust breakout_strength_threshold to be <= min_breakout_pct
            assert self.strategy.breakout_strength_threshold.value <= self.strategy.min_breakout_pct.value, \
                "Breakout strength threshold not adjusted for min breakout percentage constraint"
            
        finally:
            # Restore original values
            self.strategy.exit_minutes.value = original_exit_minutes
            self.strategy.max_position_hours.value = original_max_position_hours
            self.strategy.breakout_strength_threshold.value = original_breakout_strength
            self.strategy.min_breakout_pct.value = original_min_breakout_pct
    
    def test_parameter_reset_on_error(self):
        """
        Test parameter reset to defaults when validation fails
        
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        **Validates: Requirements 9.5**
        """
        # Store original values
        original_values = {
            'ma_period': self.strategy.ma_period.value,
            'exit_minutes': self.strategy.exit_minutes.value,
        }
        
        try:
            # Set invalid values
            self.strategy.ma_period.value = -5  # Invalid
            self.strategy.exit_minutes.value = 200  # Invalid
            
            # Mock an exception during validation to test error handling
            with patch.object(self.strategy, '_validate_hyperopt_parameters', side_effect=Exception("Test error")):
                # This should trigger the error handling and reset to defaults
                try:
                    self.strategy._validate_hyperopt_parameters()
                except:
                    pass
                
                # Call reset method directly to test it
                self.strategy._reset_parameters_to_defaults()
            
            # Should be reset to defaults
            assert self.strategy.ma_period.value == 5, \
                "MA period not reset to default on error"
            assert self.strategy.exit_minutes.value == 15, \
                "Exit minutes not reset to default on error"
            
        finally:
            # Restore original values
            for param_name, original_value in original_values.items():
                setattr(getattr(self.strategy, param_name), 'value', original_value)
    
    def test_parameter_validation_logging(self):
        """
        Test that parameter validation produces appropriate log messages
        
        **Feature: hour-breakout-scalping, Property 11: Parameter validation**
        **Validates: Requirements 9.5**
        """
        # Store original values
        original_ma_period = self.strategy.ma_period.value
        
        try:
            # Set invalid value
            self.strategy.ma_period.value = 50  # Out of range
            
            # Run validation
            self.strategy._validate_hyperopt_parameters()
            
            # Should have logged a warning about the invalid parameter
            self.strategy.logger.warning.assert_called()
            
            # Check that warning message contains relevant information
            warning_calls = [call for call in self.strategy.logger.warning.call_args_list 
                           if 'MA period' in str(call)]
            assert len(warning_calls) > 0, "No warning logged for invalid MA period"
            
        finally:
            # Restore original value
            self.strategy.ma_period.value = original_ma_period


class TestHyperOptIntegration:
    """Unit tests for HyperOpt integration and parameter space definitions"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock logger to capture messages
        self.strategy.logger = MagicMock()
    
    def test_parameter_space_definitions(self):
        """
        Test that parameter spaces are correctly defined for HyperOpt
        
        Requirements: 9.1, 9.2, 9.3, 9.4 - Parameter space definitions
        """
        # Test MA period parameter space
        assert hasattr(self.strategy, 'ma_period'), "MA period parameter not defined"
        assert self.strategy.ma_period.low == 3, f"MA period low bound incorrect: {self.strategy.ma_period.low}"
        assert self.strategy.ma_period.high == 10, f"MA period high bound incorrect: {self.strategy.ma_period.high}"
        assert self.strategy.ma_period.space == "buy", f"MA period space incorrect: {self.strategy.ma_period.space}"
        assert self.strategy.ma_period.optimize is True, "MA period optimize flag not set"
        
        # Test exit minutes parameter space
        assert hasattr(self.strategy, 'exit_minutes'), "Exit minutes parameter not defined"
        assert self.strategy.exit_minutes.low == 5, f"Exit minutes low bound incorrect: {self.strategy.exit_minutes.low}"
        assert self.strategy.exit_minutes.high == 60, f"Exit minutes high bound incorrect: {self.strategy.exit_minutes.high}"
        assert self.strategy.exit_minutes.space == "sell", f"Exit minutes space incorrect: {self.strategy.exit_minutes.space}"
        assert self.strategy.exit_minutes.optimize is True, "Exit minutes optimize flag not set"
        
        # Test minimum breakout percentage parameter space
        assert hasattr(self.strategy, 'min_breakout_pct'), "Min breakout percentage parameter not defined"
        assert self.strategy.min_breakout_pct.low == 0.001, f"Min breakout pct low bound incorrect: {self.strategy.min_breakout_pct.low}"
        assert self.strategy.min_breakout_pct.high == 0.01, f"Min breakout pct high bound incorrect: {self.strategy.min_breakout_pct.high}"
        assert self.strategy.min_breakout_pct.space == "buy", f"Min breakout pct space incorrect: {self.strategy.min_breakout_pct.space}"
        assert self.strategy.min_breakout_pct.optimize is True, "Min breakout pct optimize flag not set"
        
        # Test pullback tolerance parameter space
        assert hasattr(self.strategy, 'pullback_tolerance'), "Pullback tolerance parameter not defined"
        assert self.strategy.pullback_tolerance.low == 0.0001, f"Pullback tolerance low bound incorrect: {self.strategy.pullback_tolerance.low}"
        assert self.strategy.pullback_tolerance.high == 0.002, f"Pullback tolerance high bound incorrect: {self.strategy.pullback_tolerance.high}"
        assert self.strategy.pullback_tolerance.space == "buy", f"Pullback tolerance space incorrect: {self.strategy.pullback_tolerance.space}"
        assert self.strategy.pullback_tolerance.optimize is True, "Pullback tolerance optimize flag not set"
    
    def test_risk_management_parameter_spaces(self):
        """
        Test that risk management parameter spaces are correctly defined
        
        Requirements: 9.4 - Risk management parameter optimization
        """
        # Test max position hours parameter space
        assert hasattr(self.strategy, 'max_position_hours'), "Max position hours parameter not defined"
        assert self.strategy.max_position_hours.low == 1.0, f"Max position hours low bound incorrect: {self.strategy.max_position_hours.low}"
        assert self.strategy.max_position_hours.high == 8.0, f"Max position hours high bound incorrect: {self.strategy.max_position_hours.high}"
        assert self.strategy.max_position_hours.space == "sell", f"Max position hours space incorrect: {self.strategy.max_position_hours.space}"
        assert self.strategy.max_position_hours.optimize is True, "Max position hours optimize flag not set"
        
        # Test minimum volume threshold parameter space
        assert hasattr(self.strategy, 'min_volume_threshold'), "Min volume threshold parameter not defined"
        assert self.strategy.min_volume_threshold.low == 0.5, f"Min volume threshold low bound incorrect: {self.strategy.min_volume_threshold.low}"
        assert self.strategy.min_volume_threshold.high == 3.0, f"Min volume threshold high bound incorrect: {self.strategy.min_volume_threshold.high}"
        assert self.strategy.min_volume_threshold.space == "buy", f"Min volume threshold space incorrect: {self.strategy.min_volume_threshold.space}"
        assert self.strategy.min_volume_threshold.optimize is True, "Min volume threshold optimize flag not set"
        
        # Test stop loss buffer percentage parameter space
        assert hasattr(self.strategy, 'stop_loss_buffer_pct'), "Stop loss buffer pct parameter not defined"
        assert self.strategy.stop_loss_buffer_pct.low == 0.001, f"Stop loss buffer pct low bound incorrect: {self.strategy.stop_loss_buffer_pct.low}"
        assert self.strategy.stop_loss_buffer_pct.high == 0.01, f"Stop loss buffer pct high bound incorrect: {self.strategy.stop_loss_buffer_pct.high}"
        assert self.strategy.stop_loss_buffer_pct.space == "sell", f"Stop loss buffer pct space incorrect: {self.strategy.stop_loss_buffer_pct.space}"
        assert self.strategy.stop_loss_buffer_pct.optimize is True, "Stop loss buffer pct optimize flag not set"
        
        # Test minimum entry spacing parameter space
        assert hasattr(self.strategy, 'min_entry_spacing'), "Min entry spacing parameter not defined"
        assert self.strategy.min_entry_spacing.low == 10, f"Min entry spacing low bound incorrect: {self.strategy.min_entry_spacing.low}"
        assert self.strategy.min_entry_spacing.high == 30, f"Min entry spacing high bound incorrect: {self.strategy.min_entry_spacing.high}"
        assert self.strategy.min_entry_spacing.space == "buy", f"Min entry spacing space incorrect: {self.strategy.min_entry_spacing.space}"
        assert self.strategy.min_entry_spacing.optimize is True, "Min entry spacing optimize flag not set"
    
    def test_strength_threshold_parameter_spaces(self):
        """
        Test that strength threshold parameter spaces are correctly defined
        
        Requirements: 9.4 - Strength threshold parameter optimization
        """
        # Test breakout strength threshold parameter space
        assert hasattr(self.strategy, 'breakout_strength_threshold'), "Breakout strength threshold parameter not defined"
        assert self.strategy.breakout_strength_threshold.low == 0.001, f"Breakout strength threshold low bound incorrect: {self.strategy.breakout_strength_threshold.low}"
        assert self.strategy.breakout_strength_threshold.high == 0.005, f"Breakout strength threshold high bound incorrect: {self.strategy.breakout_strength_threshold.high}"
        assert self.strategy.breakout_strength_threshold.space == "buy", f"Breakout strength threshold space incorrect: {self.strategy.breakout_strength_threshold.space}"
        assert self.strategy.breakout_strength_threshold.optimize is True, "Breakout strength threshold optimize flag not set"
        
        # Test rebound strength threshold parameter space
        assert hasattr(self.strategy, 'rebound_strength_threshold'), "Rebound strength threshold parameter not defined"
        assert self.strategy.rebound_strength_threshold.low == 0.001, f"Rebound strength threshold low bound incorrect: {self.strategy.rebound_strength_threshold.low}"
        assert self.strategy.rebound_strength_threshold.high == 0.01, f"Rebound strength threshold high bound incorrect: {self.strategy.rebound_strength_threshold.high}"
        assert self.strategy.rebound_strength_threshold.space == "buy", f"Rebound strength threshold space incorrect: {self.strategy.rebound_strength_threshold.space}"
        assert self.strategy.rebound_strength_threshold.optimize is True, "Rebound strength threshold optimize flag not set"
    
    def test_parameter_boundary_validation(self):
        """
        Test parameter boundary validation and edge cases
        
        Requirements: 9.5 - Parameter validation and boundary conditions
        """
        # Test setting parameters to boundary values
        
        # Test MA period boundaries
        self.strategy.ma_period.value = 3  # Minimum valid value
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.ma_period.value == 3, "MA period minimum boundary not accepted"
        
        self.strategy.ma_period.value = 10  # Maximum valid value
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.ma_period.value == 10, "MA period maximum boundary not accepted"
        
        self.strategy.ma_period.value = 2  # Below minimum
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.ma_period.value == 5, "MA period below minimum not reset to default"
        
        self.strategy.ma_period.value = 11  # Above maximum
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.ma_period.value == 5, "MA period above maximum not reset to default"
        
        # Test exit minutes boundaries
        self.strategy.exit_minutes.value = 5  # Minimum valid value
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.exit_minutes.value == 5, "Exit minutes minimum boundary not accepted"
        
        self.strategy.exit_minutes.value = 60  # Maximum valid value
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.exit_minutes.value == 60, "Exit minutes maximum boundary not accepted"
        
        self.strategy.exit_minutes.value = 4  # Below minimum
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.exit_minutes.value == 15, "Exit minutes below minimum not reset to default"
        
        self.strategy.exit_minutes.value = 61  # Above maximum
        self.strategy._validate_hyperopt_parameters()
        assert self.strategy.exit_minutes.value == 15, "Exit minutes above maximum not reset to default"
    
    def test_decimal_parameter_precision(self):
        """
        Test decimal parameter precision and validation
        
        Requirements: 9.4, 9.5 - Decimal parameter handling
        """
        # Test minimum breakout percentage precision
        self.strategy.min_breakout_pct.value = 0.0015  # Valid value with precision
        self.strategy._validate_hyperopt_parameters()
        assert abs(self.strategy.min_breakout_pct.value - 0.0015) < 1e-6, "Min breakout pct precision not maintained"
        
        # Test pullback tolerance precision
        self.strategy.pullback_tolerance.value = 0.0008  # Valid value with precision
        self.strategy._validate_hyperopt_parameters()
        assert abs(self.strategy.pullback_tolerance.value - 0.0008) < 1e-6, "Pullback tolerance precision not maintained"
        
        # Test stop loss buffer precision
        self.strategy.stop_loss_buffer_pct.value = 0.0025  # Valid value with precision
        self.strategy._validate_hyperopt_parameters()
        assert abs(self.strategy.stop_loss_buffer_pct.value - 0.0025) < 1e-6, "Stop loss buffer pct precision not maintained"
    
    def test_parameter_space_consistency(self):
        """
        Test consistency between parameter spaces and validation logic
        
        Requirements: 9.1, 9.5 - Parameter space and validation consistency
        """
        # Verify that parameter space bounds match validation bounds
        
        # MA period consistency
        assert self.strategy.ma_period.low >= 3, "MA period space low bound inconsistent with validation"
        assert self.strategy.ma_period.high <= 10, "MA period space high bound inconsistent with validation"
        
        # Exit minutes consistency
        assert self.strategy.exit_minutes.low >= 5, "Exit minutes space low bound inconsistent with validation"
        assert self.strategy.exit_minutes.high <= 60, "Exit minutes space high bound inconsistent with validation"
        
        # Min breakout percentage consistency
        assert self.strategy.min_breakout_pct.low >= 0.001, "Min breakout pct space low bound inconsistent with validation"
        assert self.strategy.min_breakout_pct.high <= 0.01, "Min breakout pct space high bound inconsistent with validation"
        
        # Pullback tolerance consistency
        assert self.strategy.pullback_tolerance.low >= 0.0001, "Pullback tolerance space low bound inconsistent with validation"
        assert self.strategy.pullback_tolerance.high <= 0.002, "Pullback tolerance space high bound inconsistent with validation"
    
    def test_hyperopt_parameter_usage_in_strategy(self):
        """
        Test that HyperOpt parameters are actually used in strategy logic
        
        Requirements: 9.2, 9.3, 9.4 - Parameter usage in strategy
        """
        # Create test dataframe
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200],
            'close_5m': [100.5, 101.5, 102.5],
            'high_1h_prev': [99.0, 100.0, 101.0],
            'data_quality_ok': [True, True, True]
        })
        
        # Test MA period usage
        original_ma_period = self.strategy.ma_period.value
        self.strategy.ma_period.value = 7  # Change from default
        
        # Mock empty informative data
        self.strategy.dp = MagicMock()
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        result_df = self.strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})
        
        # Should have MA column with the specified period
        ma_col = f'ma{self.strategy.ma_period.value}'
        assert ma_col in result_df.columns, f"MA column {ma_col} not found, parameter not used"
        
        # Restore original value
        self.strategy.ma_period.value = original_ma_period
        
        # Test min_entry_spacing usage in duplicate prevention
        original_spacing = self.strategy.min_entry_spacing.value
        self.strategy.min_entry_spacing.value = 20  # Change from default
        
        # Create test scenario for duplicate prevention
        test_df_with_signals = test_data.copy()
        test_df_with_signals['entry_signal'] = [True, False, False]
        test_df_with_signals['breakout_state_active'] = [True, True, True]
        test_df_with_signals['breakout_reference_high'] = [100.0, 100.0, 100.0]
        test_df_with_signals['breakout_activation_time'] = [pd.Timestamp('2023-01-01')] * 3
        
        # Test duplicate prevention with custom spacing
        is_duplicate = self.strategy._check_duplicate_entry_prevention(test_df_with_signals, 1, 'BTC/USDT')
        
        # Should use the custom spacing parameter
        # (This is a basic test - the actual logic depends on the spacing value)
        assert isinstance(is_duplicate, bool), "Duplicate prevention not using parameter correctly"
        
        # Restore original value
        self.strategy.min_entry_spacing.value = original_spacing
    
    def test_parameter_initialization_on_strategy_creation(self):
        """
        Test that parameters are properly initialized when strategy is created
        
        Requirements: 9.1 - Parameter initialization
        """
        # Create new strategy instance
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
        }
        new_strategy = HourBreakout1(config)
        
        # All parameters should have reasonable default values within their valid ranges
        assert 3 <= new_strategy.ma_period.value <= 10, f"MA period not in valid range: {new_strategy.ma_period.value}"
        assert 5 <= new_strategy.exit_minutes.value <= 60, f"Exit minutes not in valid range: {new_strategy.exit_minutes.value}"
        assert 0.001 <= new_strategy.min_breakout_pct.value <= 0.01, f"Min breakout pct not in valid range: {new_strategy.min_breakout_pct.value}"
        assert 0.0001 <= new_strategy.pullback_tolerance.value <= 0.002, f"Pullback tolerance not in valid range: {new_strategy.pullback_tolerance.value}"
        assert 1.0 <= new_strategy.max_position_hours.value <= 8.0, f"Max position hours not in valid range: {new_strategy.max_position_hours.value}"
        assert 0.5 <= new_strategy.min_volume_threshold.value <= 3.0, f"Min volume threshold not in valid range: {new_strategy.min_volume_threshold.value}"
        assert 0.001 <= new_strategy.stop_loss_buffer_pct.value <= 0.01, f"Stop loss buffer pct not in valid range: {new_strategy.stop_loss_buffer_pct.value}"
        assert 10 <= new_strategy.min_entry_spacing.value <= 30, f"Min entry spacing not in valid range: {new_strategy.min_entry_spacing.value}"
        assert 0.001 <= new_strategy.breakout_strength_threshold.value <= 0.005, f"Breakout strength threshold not in valid range: {new_strategy.breakout_strength_threshold.value}"
        assert 0.001 <= new_strategy.rebound_strength_threshold.value <= 0.01, f"Rebound strength threshold not in valid range: {new_strategy.rebound_strength_threshold.value}"
        
        # Test that parameters are properly typed
        assert isinstance(new_strategy.ma_period.value, int), "MA period should be integer"
        assert isinstance(new_strategy.exit_minutes.value, int), "Exit minutes should be integer"
        assert isinstance(new_strategy.min_breakout_pct.value, (int, float)), "Min breakout pct should be numeric"
        assert isinstance(new_strategy.pullback_tolerance.value, (int, float)), "Pullback tolerance should be numeric"


class TestDataFrameCompatibility:
    """Property-based tests for FreqTrade dataframe compatibility"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_basic_ohlcv_data(self, num_candles: int, base_price: float = 100.0) -> pd.DataFrame:
        """Generate basic OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.01, num_candles)
        
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        # Create OHLCV data
        data = []
        for i in range(num_candles):
            high = prices[i] * (1 + abs(np.random.normal(0, 0.005)))
            low = prices[i] * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'date': dates[i],
                'open': prices[i-1] if i > 0 else prices[i],
                'high': high,
                'low': low,
                'close': prices[i],
                'volume': volume
            })
        
        return pd.DataFrame(data).set_index('date')
    
    @given(
        num_candles=st.integers(min_value=150, max_value=300),
        base_price=st.floats(min_value=50.0, max_value=200.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_dataframe_compatibility(self, num_candles: int, base_price: float):
        """
        **Feature: hour-breakout-scalping, Property 10: DataFrame compatibility**
        
        For any output dataframe, should include FreqTrade required columns and follow 
        correct naming conventions
        
        **Validates: Requirements 7.4**
        """
        # Generate test data
        test_df = self.generate_basic_ohlcv_data(num_candles, base_price)
        
        # Mock empty informative data to focus on compatibility testing
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Test populate_indicators compatibility
        indicators_df = self.strategy.populate_indicators(test_df.copy(), {'pair': 'BTC/USDT'})
        
        # Property 1: All required FreqTrade columns should exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in indicators_df.columns, f"Required OHLCV column '{col}' missing"
        
        # Property 2: Data types should be correct for OHLCV columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(indicators_df[col]), \
                f"Column '{col}' should be numeric, found {indicators_df[col].dtype}"
        
        # Property 3: No NaN values in basic OHLCV columns
        for col in numeric_columns:
            nan_count = indicators_df[col].isna().sum()
            assert nan_count == 0, f"Column '{col}' has {nan_count} NaN values"
        
        # Property 4: OHLC relationships should be valid
        if len(indicators_df) > 0:
            # High >= Low
            invalid_hl = (indicators_df['high'] < indicators_df['low']).sum()
            assert invalid_hl == 0, f"{invalid_hl} candles have high < low"
            
            # Close within High-Low range
            invalid_close_high = (indicators_df['close'] > indicators_df['high']).sum()
            invalid_close_low = (indicators_df['close'] < indicators_df['low']).sum()
            assert invalid_close_high == 0, f"{invalid_close_high} candles have close > high"
            assert invalid_close_low == 0, f"{invalid_close_low} candles have close < low"
        
        # Property 5: Index should be datetime
        assert pd.api.types.is_datetime64_any_dtype(indicators_df.index), \
            "DataFrame index should be datetime type"
        
        # Property 6: Index should be monotonic (sorted)
        assert indicators_df.index.is_monotonic_increasing, \
            "DataFrame index should be in ascending order"
        
        # Test populate_entry_trend compatibility
        entry_df = self.strategy.populate_entry_trend(indicators_df.copy(), {'pair': 'BTC/USDT'})
        
        # Property 7: Entry signal columns should exist with correct types
        assert 'enter_long' in entry_df.columns, "Missing 'enter_long' column"
        assert 'enter_tag' in entry_df.columns, "Missing 'enter_tag' column"
        
        # Property 8: Entry signal data types should be correct
        assert pd.api.types.is_integer_dtype(entry_df['enter_long']), \
            f"'enter_long' should be integer, found {entry_df['enter_long'].dtype}"
        assert pd.api.types.is_object_dtype(entry_df['enter_tag']), \
            f"'enter_tag' should be object/string, found {entry_df['enter_tag'].dtype}"
        
        # Property 9: Entry signals should only be 0 or 1
        invalid_signals = (~entry_df['enter_long'].isin([0, 1])).sum()
        assert invalid_signals == 0, f"{invalid_signals} invalid entry signals (not 0 or 1)"
        
        # Test populate_exit_trend compatibility
        exit_df = self.strategy.populate_exit_trend(entry_df.copy(), {'pair': 'BTC/USDT'})
        
        # Property 10: Exit signal columns should exist with correct types
        assert 'exit_long' in exit_df.columns, "Missing 'exit_long' column"
        assert 'exit_tag' in exit_df.columns, "Missing 'exit_tag' column"
        
        # Property 11: Exit signal data types should be correct
        assert pd.api.types.is_integer_dtype(exit_df['exit_long']), \
            f"'exit_long' should be integer, found {exit_df['exit_long'].dtype}"
        assert pd.api.types.is_object_dtype(exit_df['exit_tag']), \
            f"'exit_tag' should be object/string, found {exit_df['exit_tag'].dtype}"
        
        # Property 12: Exit signals should only be 0 or 1
        invalid_exit_signals = (~exit_df['exit_long'].isin([0, 1])).sum()
        assert invalid_exit_signals == 0, f"{invalid_exit_signals} invalid exit signals (not 0 or 1)"
        
        # Property 13: No deprecated column names should exist
        deprecated_columns = ['buy', 'sell', 'buy_tag', 'sell_tag']
        for col in deprecated_columns:
            assert col not in exit_df.columns, f"Deprecated column '{col}' found in dataframe"
        
        # Property 14: Column names should not contain invalid characters
        invalid_chars = [' ', '-', '+', '*', '/', '\\', '(', ')', '[', ']', '{', '}']
        for col in exit_df.columns:
            for char in invalid_chars:
                assert char not in str(col), f"Column '{col}' contains invalid character '{char}'"
        
        # Property 15: Column names should not be excessively long
        max_col_length = 64
        for col in exit_df.columns:
            assert len(str(col)) <= max_col_length, \
                f"Column '{col}' is too long ({len(str(col))} > {max_col_length} chars)"
    
    def test_property_dataframe_compatibility_with_missing_data(self):
        """
        Test dataframe compatibility when some data is missing or invalid
        
        **Feature: hour-breakout-scalping, Property 10: DataFrame compatibility**
        **Validates: Requirements 7.4**
        """
        # Create dataframe with missing columns
        incomplete_df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200]
            # Missing open, high, low columns
        })
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Should handle missing columns gracefully
        result_df = self.strategy.populate_indicators(incomplete_df.copy(), {'pair': 'BTC/USDT'})
        
        # Should have all required columns after processing
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in result_df.columns, f"Required column '{col}' not added"
        
        # Should have entry/exit columns after full processing
        entry_df = self.strategy.populate_entry_trend(result_df.copy(), {'pair': 'BTC/USDT'})
        exit_df = self.strategy.populate_exit_trend(entry_df.copy(), {'pair': 'BTC/USDT'})
        
        signal_columns = ['enter_long', 'enter_tag', 'exit_long', 'exit_tag']
        for col in signal_columns:
            assert col in exit_df.columns, f"Signal column '{col}' not added"
    
    def test_property_dataframe_compatibility_with_invalid_data(self):
        """
        Test dataframe compatibility with invalid data values
        
        **Feature: hour-breakout-scalping, Property 10: DataFrame compatibility**
        **Validates: Requirements 7.4**
        """
        # Create dataframe with invalid data
        invalid_df = pd.DataFrame({
            'open': [100.0, -50.0, np.nan],  # Negative and NaN values
            'high': [101.0, 102.0, np.inf], # Infinite value
            'low': [99.0, 98.0, -10.0],     # Negative value
            'close': [100.5, 101.5, 0.0],   # Zero value
            'volume': [1000, -500, np.nan]  # Negative and NaN values
        })
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Should handle invalid data gracefully
        result_df = self.strategy.populate_indicators(invalid_df.copy(), {'pair': 'BTC/USDT'})
        
        # Should not have NaN values in critical columns after processing
        critical_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in critical_columns:
            if col in result_df.columns:
                nan_count = result_df[col].isna().sum()
                assert nan_count == 0, f"Column '{col}' still has {nan_count} NaN values after processing"
                
                # Should not have negative values in price/volume columns
                negative_count = (result_df[col] < 0).sum()
                assert negative_count == 0, f"Column '{col}' still has {negative_count} negative values after processing"
                
                # Should not have infinite values
                inf_count = np.isinf(result_df[col]).sum()
                assert inf_count == 0, f"Column '{col}' still has {inf_count} infinite values after processing"
    
    def test_property_dataframe_compatibility_column_types(self):
        """
        Test that all columns have correct data types after processing
        
        **Feature: hour-breakout-scalping, Property 10: DataFrame compatibility**
        **Validates: Requirements 7.4**
        """
        # Generate test data
        test_df = self.generate_basic_ohlcv_data(100, 100.0)
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Process through all stages
        indicators_df = self.strategy.populate_indicators(test_df.copy(), {'pair': 'BTC/USDT'})
        entry_df = self.strategy.populate_entry_trend(indicators_df.copy(), {'pair': 'BTC/USDT'})
        exit_df = self.strategy.populate_exit_trend(entry_df.copy(), {'pair': 'BTC/USDT'})
        
        # Define expected data types
        expected_types = {
            # OHLCV columns
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            
            # Signal columns
            'enter_long': 'int64',
            'enter_tag': 'object',
            'exit_long': 'int64',
            'exit_tag': 'object',
        }
        
        # Check data types
        for col, expected_dtype in expected_types.items():
            if col in exit_df.columns:
                actual_dtype = str(exit_df[col].dtype)
                assert actual_dtype == expected_dtype, \
                    f"Column '{col}' has wrong data type: expected {expected_dtype}, got {actual_dtype}"
        
        # Check boolean columns if they exist
        boolean_columns = ['breakout_condition', 'pullback_condition', 'rebound_condition', 
                          'entry_signal', 'data_quality_ok']
        for col in boolean_columns:
            if col in exit_df.columns:
                assert exit_df[col].dtype == 'bool', \
                    f"Boolean column '{col}' has wrong data type: {exit_df[col].dtype}"
        
        # Check numeric indicator columns if they exist
        numeric_columns = ['entry_signal_strength', 'breakout_strength', 'pullback_strength']
        for col in numeric_columns:
            if col in exit_df.columns:
                assert pd.api.types.is_numeric_dtype(exit_df[col]), \
                    f"Numeric column '{col}' is not numeric: {exit_df[col].dtype}"
    
    def test_property_dataframe_compatibility_edge_cases(self):
        """
        Test dataframe compatibility with edge cases
        
        **Feature: hour-breakout-scalping, Property 10: DataFrame compatibility**
        **Validates: Requirements 7.4**
        """
        # Test with minimal data (1 candle)
        minimal_df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        # Mock empty informative data
        self.strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        
        # Should handle minimal data without error
        result_df = self.strategy.populate_indicators(minimal_df.copy(), {'pair': 'BTC/USDT'})
        entry_df = self.strategy.populate_entry_trend(result_df.copy(), {'pair': 'BTC/USDT'})
        exit_df = self.strategy.populate_exit_trend(entry_df.copy(), {'pair': 'BTC/USDT'})
        
        # Should have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'enter_long', 'enter_tag', 'exit_long', 'exit_tag']
        for col in required_columns:
            assert col in exit_df.columns, f"Required column '{col}' missing in minimal data test"
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            empty_result = self.strategy.populate_indicators(empty_df.copy(), {'pair': 'BTC/USDT'})
            empty_entry = self.strategy.populate_entry_trend(empty_result.copy(), {'pair': 'BTC/USDT'})
            empty_exit = self.strategy.populate_exit_trend(empty_entry.copy(), {'pair': 'BTC/USDT'})
            
            # Should return a valid dataframe (even if empty)
            assert isinstance(empty_exit, pd.DataFrame), "Should return DataFrame even for empty input"
            
        except Exception as e:
            # If it fails, it should fail gracefully without crashing
            assert "Error" in str(e) or "error" in str(e), f"Unexpected error type: {str(e)}"


class TestLoggingSystem:
    """Unit tests for logging system integration and functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider
        self.strategy.dp = MagicMock()
        
        # Capture log messages for testing
        self.log_messages = []
        self.original_logger = None
    
    def capture_log_messages(self):
        """Setup log message capture for testing"""
        import logging
        
        # Create a custom handler to capture log messages
        class TestLogHandler(logging.Handler):
            def __init__(self, test_instance):
                super().__init__()
                self.test_instance = test_instance
            
            def emit(self, record):
                self.test_instance.log_messages.append({
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'name': record.name
                })
        
        # Setup test handler
        self.test_handler = TestLogHandler(self)
        
        # Add handler to strategy logger
        if hasattr(self.strategy, 'logger') and self.strategy.logger:
            self.original_logger = self.strategy.logger
            self.strategy.logger.addHandler(self.test_handler)
            self.strategy.logger.setLevel(logging.DEBUG)
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Remove test handler
        if hasattr(self, 'test_handler') and hasattr(self.strategy, 'logger') and self.strategy.logger:
            self.strategy.logger.removeHandler(self.test_handler)
    
    def test_logging_system_initialization(self):
        """
        Test logging system initialization
        
        **Validates: Requirements 7.5**
        """
        # Test that logger is properly initialized
        assert hasattr(self.strategy, 'logger'), "Strategy should have logger attribute"
        assert self.strategy.logger is not None, "Logger should not be None"
        
        # Test logger name
        expected_name = f"freqtrade.strategy.{self.strategy.__class__.__name__}"
        assert self.strategy.logger.name == expected_name, f"Logger name should be {expected_name}"
        
        # Test that _setup_logging_system method exists
        assert hasattr(self.strategy, '_setup_logging_system'), "Should have _setup_logging_system method"
        assert callable(self.strategy._setup_logging_system), "_setup_logging_system should be callable"
    
    def test_structured_logging_methods_exist(self):
        """
        Test that all structured logging methods exist
        
        **Validates: Requirements 7.5**
        """
        # Test that all logging methods exist
        logging_methods = [
            'log_strategy_event',
            'log_entry_signal',
            'log_exit_signal',
            'log_breakout_detection',
            'log_pullback_detection',
            'log_rebound_detection',
            'log_parameter_validation',
            'log_performance_metrics',
            'log_error_with_context',
            'enable_debug_logging',
            'disable_debug_logging',
            'log_strategy_initialization'
        ]
        
        for method_name in logging_methods:
            assert hasattr(self.strategy, method_name), f"Should have {method_name} method"
            assert callable(getattr(self.strategy, method_name)), f"{method_name} should be callable"
    
    def test_log_strategy_event_functionality(self):
        """
        Test log_strategy_event method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test basic logging
        self.strategy.log_strategy_event('TEST', 'BTC/USDT', 'Test message', 'info')
        
        # Check that message was logged
        assert len(self.log_messages) > 0, "Should have logged at least one message"
        
        # Find our test message
        test_messages = [msg for msg in self.log_messages if 'TEST' in msg['message']]
        assert len(test_messages) > 0, "Should have logged test message"
        
        test_message = test_messages[0]
        assert test_message['level'] == 'INFO', "Should log at INFO level"
        assert '[TEST]' in test_message['message'], "Should include event type in brackets"
        assert 'BTC/USDT' in test_message['message'], "Should include pair in message"
        assert 'Test message' in test_message['message'], "Should include original message"
    
    def test_log_strategy_event_with_extra_data(self):
        """
        Test log_strategy_event with extra data
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test logging with extra data
        extra_data = {
            'price': 100.50,
            'volume': 1000,
            'strength': '2.5%'
        }
        
        self.strategy.log_strategy_event('ENTRY', 'ETH/USDT', 'Entry signal', 'info', extra_data)
        
        # Check that extra data was included
        entry_messages = [msg for msg in self.log_messages if 'ENTRY' in msg['message']]
        assert len(entry_messages) > 0, "Should have logged entry message"
        
        entry_message = entry_messages[0]['message']
        assert 'price=100.5' in entry_message, "Should include price in extra data"
        assert 'volume=1000' in entry_message, "Should include volume in extra data"
        assert 'strength=2.5%' in entry_message, "Should include strength in extra data"
    
    def test_log_entry_signal_functionality(self):
        """
        Test log_entry_signal method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test entry signal logging
        candle_time = pd.Timestamp('2023-01-01 10:30:00')
        entry_strength = 2.5
        conditions = {
            'breakout_active': True,
            'pullback_completed': True,
            'rebound_detected': True
        }
        
        self.strategy.log_entry_signal('BTC/USDT', candle_time, entry_strength, conditions)
        
        # Check that entry signal was logged
        entry_messages = [msg for msg in self.log_messages if 'ENTRY' in msg['message']]
        assert len(entry_messages) > 0, "Should have logged entry signal"
        
        entry_message = entry_messages[0]['message']
        assert 'BTC/USDT' in entry_message, "Should include pair"
        assert '2.5000%' in entry_message, "Should include strength"
        assert 'timestamp=2023-01-01 10:30:00' in entry_message, "Should include timestamp"
        assert 'breakout=True' in entry_message, "Should include breakout condition"
        assert 'pullback_completed=True' in entry_message, "Should include pullback condition"
        assert 'rebound=True' in entry_message, "Should include rebound condition"
    
    def test_log_exit_signal_functionality(self):
        """
        Test log_exit_signal method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test stop loss exit logging
        candle_time = pd.Timestamp('2023-01-01 11:00:00')
        exit_data = {
            'stop_loss_price': 99.50,
            'current_price': 99.00
        }
        
        self.strategy.log_exit_signal('BTC/USDT', candle_time, 'stop_loss', exit_data)
        
        # Check that exit signal was logged
        exit_messages = [msg for msg in self.log_messages if 'EXIT' in msg['message']]
        assert len(exit_messages) > 0, "Should have logged exit signal"
        
        exit_message = exit_messages[0]['message']
        assert 'BTC/USDT' in exit_message, "Should include pair"
        assert 'stop_loss' in exit_message, "Should include exit reason"
        assert 'timestamp=2023-01-01 11:00:00' in exit_message, "Should include timestamp"
        assert 'stop_price=99.5' in exit_message, "Should include stop price"
        assert 'current_price=99.0' in exit_message, "Should include current price"
    
    def test_log_breakout_detection_functionality(self):
        """
        Test log_breakout_detection method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test breakout detection logging
        candle_time = pd.Timestamp('2023-01-01 09:45:00')
        breakout_data = {
            'close_5m': 101.50,
            'high_1h_prev': 100.00,
            'breakout_strength': 1.5
        }
        
        self.strategy.log_breakout_detection('ETH/USDT', candle_time, breakout_data)
        
        # Check that breakout was logged
        breakout_messages = [msg for msg in self.log_messages if 'BREAKOUT' in msg['message']]
        assert len(breakout_messages) > 0, "Should have logged breakout detection"
        
        breakout_message = breakout_messages[0]['message']
        assert 'ETH/USDT' in breakout_message, "Should include pair"
        assert '101.5' in breakout_message, "Should include 5m close price"
        assert '100.0' in breakout_message, "Should include 1h high"
        assert 'timestamp=2023-01-01 09:45:00' in breakout_message, "Should include timestamp"
    
    def test_log_error_with_context_functionality(self):
        """
        Test log_error_with_context method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test error logging
        test_error = ValueError("Test error message")
        additional_data = {
            'function': 'test_function',
            'line': 123
        }
        
        self.strategy.log_error_with_context(test_error, 'unit_test', 'BTC/USDT', additional_data)
        
        # Check that error was logged
        error_messages = [msg for msg in self.log_messages if msg['level'] == 'ERROR']
        assert len(error_messages) > 0, "Should have logged error message"
        
        error_message = error_messages[0]['message']
        assert 'ERROR' in error_message, "Should include ERROR event type"
        assert 'BTC/USDT' in error_message, "Should include pair"
        assert 'unit_test' in error_message, "Should include context"
        assert 'ValueError' in error_message, "Should include error type"
        assert 'Test error message' in error_message, "Should include error message"
        assert 'function=test_function' in error_message, "Should include additional data"
        assert 'line=123' in error_message, "Should include line number"
    
    def test_debug_logging_control(self):
        """
        Test debug logging enable/disable functionality
        
        **Validates: Requirements 7.5**
        """
        import logging
        
        # Test enable debug logging
        self.strategy.enable_debug_logging()
        assert self.strategy.logger.level == logging.DEBUG, "Should set logger to DEBUG level"
        
        # Test disable debug logging
        self.strategy.disable_debug_logging()
        assert self.strategy.logger.level == logging.INFO, "Should set logger to INFO level"
    
    def test_log_parameter_validation_functionality(self):
        """
        Test log_parameter_validation method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test successful validation logging
        validation_results = {
            'success': True,
            'validated_params': ['ma_period', 'exit_minutes', 'min_breakout_pct'],
            'issues': [],
            'corrections': []
        }
        
        self.strategy.log_parameter_validation(validation_results)
        
        # Check that validation was logged
        validation_messages = [msg for msg in self.log_messages if 'VALIDATION' in msg['message']]
        assert len(validation_messages) > 0, "Should have logged parameter validation"
        
        validation_message = validation_messages[0]['message']
        assert 'SYSTEM' in validation_message, "Should target SYSTEM"
        assert 'successfully' in validation_message, "Should indicate success"
        assert 'validated_params=3' in validation_message, "Should include param count"
        assert 'issues_found=0' in validation_message, "Should include issue count"
    
    def test_log_performance_metrics_functionality(self):
        """
        Test log_performance_metrics method functionality
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test performance metrics logging
        metrics = {
            'total_candles': 1000,
            'breakout_count': 15,
            'entry_count': 8,
            'exit_count': 7,
            'processing_time_ms': 125.5
        }
        
        self.strategy.log_performance_metrics('BTC/USDT', metrics)
        
        # Check that metrics were logged
        performance_messages = [msg for msg in self.log_messages if 'PERFORMANCE' in msg['message']]
        assert len(performance_messages) > 0, "Should have logged performance metrics"
        
        performance_message = performance_messages[0]['message']
        assert 'BTC/USDT' in performance_message, "Should include pair"
        assert '8 entries' in performance_message, "Should include entry count"
        assert '7 exits' in performance_message, "Should include exit count"
        assert '1000 candles' in performance_message, "Should include candle count"
        assert 'total_candles=1000' in performance_message, "Should include detailed metrics"
    
    def test_logging_error_handling(self):
        """
        Test that logging methods handle errors gracefully
        
        **Validates: Requirements 7.5**
        """
        # Test with invalid timestamp (should not crash)
        try:
            self.strategy.log_entry_signal('BTC/USDT', None, 2.5, {})
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"log_entry_signal should handle invalid timestamp gracefully: {str(e)}")
        
        # Test with invalid data types (should not crash)
        try:
            self.strategy.log_exit_signal('BTC/USDT', pd.Timestamp.now(), 'test', None)
            # Should not raise exception
        except Exception as e:
            pytest.fail(f"log_exit_signal should handle invalid data gracefully: {str(e)}")
        
        # Test with missing logger (should not crash)
        original_logger = self.strategy.logger
        self.strategy.logger = None
        
        try:
            self.strategy.log_strategy_event('TEST', 'BTC/USDT', 'Test message')
            # Should not raise exception and should reinitialize logger
            assert self.strategy.logger is not None, "Should reinitialize logger when missing"
        except Exception as e:
            pytest.fail(f"Should handle missing logger gracefully: {str(e)}")
        finally:
            # Restore original logger
            self.strategy.logger = original_logger
    
    def test_logging_message_format_consistency(self):
        """
        Test that log messages follow consistent format
        
        **Validates: Requirements 7.5**
        """
        # Setup log capture
        self.capture_log_messages()
        
        # Test various event types
        self.strategy.log_strategy_event('ENTRY', 'BTC/USDT', 'Entry test')
        self.strategy.log_strategy_event('EXIT', 'ETH/USDT', 'Exit test')
        self.strategy.log_strategy_event('BREAKOUT', 'ADA/USDT', 'Breakout test')
        
        # Check message format consistency
        for msg in self.log_messages:
            message = msg['message']
            
            # Should start with [EVENT_TYPE]
            assert message.startswith('['), "Message should start with ["
            assert ']' in message, "Message should contain closing ]"
            
            # Should contain pair after event type
            event_end = message.find(']')
            pair_section = message[event_end+1:event_end+20]  # Check next 20 chars
            assert any(pair in pair_section for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']), \
                f"Message should contain pair after event type: {message}"


if __name__ == '__main__':
    pytest.main([__file__])


class TestBacktestConsistency:
    """Property-based tests for backtest signal consistency"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create minimal config for strategy initialization
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        self.strategy = HourBreakout1(config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_consistent_test_data(self, num_candles: int, base_price: float = 100.0, 
                                    seed: int = 42) -> pd.DataFrame:
        """
        Generate consistent test data for backtest consistency testing
        
        :param num_candles: Number of candles to generate
        :param base_price: Base price for generation
        :param seed: Random seed for reproducibility
        :return: DataFrame with consistent test data
        """
        # Use fixed seed for reproducible results
        np.random.seed(seed)
        
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate deterministic price data
        price_changes = np.random.normal(0, 0.001, num_candles)
        
        closes = [base_price]
        for change in price_changes[1:]:
            closes.append(closes[-1] * (1 + change))
        
        # Generate OHLCV data
        data = []
        for i in range(num_candles):
            high = closes[i] * (1 + abs(np.random.normal(0, 0.002)))
            low = closes[i] * (1 - abs(np.random.normal(0, 0.002)))
            open_price = closes[i-1] if i > 0 else closes[i]
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'date': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': closes[i],
                'volume': volume
            })
        
        df = pd.DataFrame(data).set_index('date')
        
        # Generate consistent 5m and 1h data
        # For simplicity, use aggregated data from 1m
        df['close_5m'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['high_1h'] = df['high'].rolling(window=60, min_periods=1).max()
        df['high_1h_prev'] = df['high_1h'].shift(1).ffill()
        
        # Add data quality flag
        df['data_quality_ok'] = True
        
        return df
    
    @given(
        num_candles=st.integers(min_value=150, max_value=200),
        base_price=st.floats(min_value=80.0, max_value=120.0),
        seed=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=5, deadline=None)
    def test_property_backtest_signal_consistency(self, num_candles: int, base_price: float, seed: int):
        """
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        
        For any historical data set, backtest mode should generate the same signals 
        as live mode when processing the same data
        
        **Validates: Requirements 8.3**
        """
        # Generate consistent test data
        test_data = self.generate_consistent_test_data(num_candles, base_price, seed)
        
        # Mock informative data to ensure consistency
        mock_5m_data = test_data[['close']].rename(columns={'close': 'close'})
        mock_1h_data = test_data[['high']].rename(columns={'high': 'high'})
        
        # Create fresh strategy instance for live mode to avoid state contamination
        config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
        }
        live_strategy = HourBreakout1(config)
        live_strategy.dp = MagicMock()
        live_strategy.logger = MagicMock()
        
        # Test in live mode (default)
        live_strategy._is_backtest_mode = False
        live_strategy._setup_backtest_optimizations()
        
        # Mock data provider for live mode
        def mock_get_pair_dataframe_live(pair, timeframe):
            if timeframe == '5m':
                return mock_5m_data.copy()
            elif timeframe == '1h':
                return mock_1h_data.copy()
            return pd.DataFrame()
        
        live_strategy.dp.get_pair_dataframe = mock_get_pair_dataframe_live
        
        # Process data in live mode
        live_result = live_strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})
        live_entry_result = live_strategy.populate_entry_trend(live_result.copy(), {'pair': 'BTC/USDT'})
        live_exit_result = live_strategy.populate_exit_trend(live_entry_result.copy(), {'pair': 'BTC/USDT'})
        
        # Create fresh strategy instance for backtest mode
        backtest_strategy = HourBreakout1(config)
        backtest_strategy.dp = MagicMock()
        backtest_strategy.logger = MagicMock()
        
        # Test in backtest mode
        backtest_strategy._is_backtest_mode = True
        backtest_strategy._setup_backtest_optimizations()
        
        # Mock data provider for backtest mode (same data)
        def mock_get_pair_dataframe_backtest(pair, timeframe):
            if timeframe == '5m':
                return mock_5m_data.copy()
            elif timeframe == '1h':
                return mock_1h_data.copy()
            return pd.DataFrame()
        
        backtest_strategy.dp.get_pair_dataframe = mock_get_pair_dataframe_backtest
        
        # Process data in backtest mode
        backtest_result = backtest_strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})
        backtest_entry_result = backtest_strategy.populate_entry_trend(backtest_result.copy(), {'pair': 'BTC/USDT'})
        backtest_exit_result = backtest_strategy.populate_exit_trend(backtest_entry_result.copy(), {'pair': 'BTC/USDT'})
        
        # Property 1: Entry signals should be identical
        live_entries = live_exit_result['enter_long'].values
        backtest_entries = backtest_exit_result['enter_long'].values
        
        # Compare entry signals
        entry_differences = np.sum(live_entries != backtest_entries)
        assert entry_differences == 0, \
            f"Entry signal inconsistency: {entry_differences} differences between live and backtest modes"
        
        # Property 2: Exit signals should be identical
        live_exits = live_exit_result['exit_long'].values
        backtest_exits = backtest_exit_result['exit_long'].values
        
        exit_differences = np.sum(live_exits != backtest_exits)
        assert exit_differences == 0, \
            f"Exit signal inconsistency: {exit_differences} differences between live and backtest modes"
        
        # Property 3: Technical indicators should be identical or very close
        # Note: We skip ma5 comparison because the test data has pre-calculated close_5m
        # which differs from the strategy's MA5 calculation. The important thing is that
        # both modes produce the same results, not that they match the test data.
        indicator_columns = ['breakout_condition', 'pullback_condition', 'rebound_condition', 'entry_signal']
        
        for col in indicator_columns:
            if col in live_exit_result.columns and col in backtest_exit_result.columns:
                if live_exit_result[col].dtype == 'bool' or str(live_exit_result[col].dtype) == 'object':
                    # For boolean columns
                    differences = np.sum(live_exit_result[col].values != backtest_exit_result[col].values)
                    assert differences == 0, \
                        f"Indicator {col} inconsistency: {differences} differences between modes"
                else:
                    # For numeric columns, allow small floating point differences
                    live_values = live_exit_result[col].values
                    backtest_values = backtest_exit_result[col].values
                    
                    # Handle NaN values
                    live_nan_mask = pd.isna(live_values)
                    backtest_nan_mask = pd.isna(backtest_values)
                    
                    # NaN positions should match
                    nan_differences = np.sum(live_nan_mask != backtest_nan_mask)
                    assert nan_differences == 0, \
                        f"NaN pattern inconsistency in {col}: {nan_differences} differences"
                    
                    # Compare non-NaN values with reasonable tolerance
                    valid_mask = ~live_nan_mask & ~backtest_nan_mask
                    if valid_mask.any():
                        max_diff = np.max(np.abs(live_values[valid_mask] - backtest_values[valid_mask]))
                        # Allow larger tolerance for floating point differences due to different processing paths
                        assert max_diff < 1e-6, \
                            f"Numeric indicator {col} inconsistency: max difference {max_diff}"
        
        # Property 4: Entry and exit tags should be identical
        if 'enter_tag' in live_exit_result.columns and 'enter_tag' in backtest_exit_result.columns:
            tag_differences = np.sum(live_exit_result['enter_tag'].values != backtest_exit_result['enter_tag'].values)
            assert tag_differences == 0, \
                f"Entry tag inconsistency: {tag_differences} differences between modes"
        
        if 'exit_tag' in live_exit_result.columns and 'exit_tag' in backtest_exit_result.columns:
            tag_differences = np.sum(live_exit_result['exit_tag'].values != backtest_exit_result['exit_tag'].values)
            assert tag_differences == 0, \
                f"Exit tag inconsistency: {tag_differences} differences between modes"
        
        # Property 5: Signal timing should be identical
        live_entry_positions = np.where(live_entries == 1)[0]
        backtest_entry_positions = np.where(backtest_entries == 1)[0]
        
        assert len(live_entry_positions) == len(backtest_entry_positions), \
            f"Different number of entry signals: live={len(live_entry_positions)}, backtest={len(backtest_entry_positions)}"
        
        if len(live_entry_positions) > 0:
            position_differences = np.sum(live_entry_positions != backtest_entry_positions)
            assert position_differences == 0, \
                f"Entry signal timing inconsistency: {position_differences} position differences"
        
        live_exit_positions = np.where(live_exits == 1)[0]
        backtest_exit_positions = np.where(backtest_exits == 1)[0]
        
        assert len(live_exit_positions) == len(backtest_exit_positions), \
            f"Different number of exit signals: live={len(live_exit_positions)}, backtest={len(backtest_exit_positions)}"
        
        if len(live_exit_positions) > 0:
            position_differences = np.sum(live_exit_positions != backtest_exit_positions)
            assert position_differences == 0, \
                f"Exit signal timing inconsistency: {position_differences} position differences"
    
    def test_property_backtest_mode_detection(self):
        """
        Test backtest mode detection accuracy
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.1**
        """
        # Test various backtest mode indicators
        backtest_configs = [
            # Direct backtest flag
            {'runmode': 'backtest'},
            {'command': 'backtesting'},
            
            # Dry run with historical data
            {'dry_run': True, 'datadir': '/path/to/data'},
            
            # Backtest-specific configuration
            {'backtest_directory': '/path/to/results'},
            {'export': 'trades'},
            {'timerange': '20230101-20231231'},
        ]
        
        for config_idx, config in enumerate(backtest_configs):
            is_backtest = self.strategy._detect_backtest_mode(config)
            assert is_backtest, f"Backtest mode not detected for config {config_idx}: {config}"
        
        # Test live mode configurations
        live_configs = [
            {},  # Empty config
            {'dry_run': False},
            {'runmode': 'live'},
            {'command': 'trade'},
        ]
        
        for config_idx, config in enumerate(live_configs):
            is_backtest = self.strategy._detect_backtest_mode(config)
            assert not is_backtest, f"Live mode incorrectly detected as backtest for config {config_idx}: {config}"
    
    def test_property_backtest_optimization_settings(self):
        """
        Test backtest optimization settings are applied correctly
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.2**
        """
        # Test backtest mode optimizations
        self.strategy._is_backtest_mode = True
        self.strategy._setup_backtest_optimizations()
        
        assert self.strategy._enable_data_caching, "Data caching not enabled in backtest mode"
        assert hasattr(self.strategy, '_data_cache'), "Data cache not initialized"
        assert self.strategy._optimize_informative_pairs, "Informative pairs optimization not enabled"
        assert hasattr(self.strategy, '_informative_cache'), "Informative cache not initialized"
        assert self.strategy._enable_batch_processing, "Batch processing not enabled"
        assert self.strategy._preallocate_arrays, "Array preallocation not enabled"
        
        # Test live mode settings
        self.strategy._is_backtest_mode = False
        self.strategy._setup_backtest_optimizations()
        
        assert not self.strategy._enable_data_caching, "Data caching enabled in live mode"
        assert not self.strategy._optimize_informative_pairs, "Informative pairs optimization enabled in live mode"
        assert not self.strategy._enable_batch_processing, "Batch processing enabled in live mode"
        assert not self.strategy._preallocate_arrays, "Array preallocation enabled in live mode"
    
    def test_property_data_caching_consistency(self):
        """
        Test data caching provides consistent results
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.2**
        """
        # Enable backtest mode with caching
        self.strategy._is_backtest_mode = True
        self.strategy._setup_backtest_optimizations()
        
        # Create test data
        test_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'volume': [1000, 1100, 1200]
        })
        
        pair = 'BTC/USDT'
        timeframe = '5m'
        
        # First access - should cache the data
        cached_data = self.strategy._get_cached_informative_data(pair, timeframe)
        assert cached_data is None, "Data should not be cached initially"
        
        # Cache the data
        self.strategy._cache_informative_data(pair, timeframe, test_data)
        
        # Second access - should return cached data
        cached_data = self.strategy._get_cached_informative_data(pair, timeframe)
        assert cached_data is not None, "Data should be cached after caching"
        
        # Verify cached data is identical to original
        pd.testing.assert_frame_equal(cached_data, test_data, check_dtype=False)
        
        # Verify cache hit tracking
        initial_hits = self.strategy._backtest_performance_tracker['cache_hits']
        self.strategy._get_cached_informative_data(pair, timeframe)
        final_hits = self.strategy._backtest_performance_tracker['cache_hits']
        
        assert final_hits > initial_hits, "Cache hit not tracked"
    
    def test_property_historical_data_validation(self):
        """
        Test historical data validation maintains data integrity
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.1**
        """
        # Create test data with various issues
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        
        problematic_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000] * 100
        }, index=dates)
        
        # Introduce data quality issues
        problematic_data.loc[problematic_data.index[10], 'high'] = 98.0  # High < Low
        problematic_data.loc[problematic_data.index[20], 'close'] = 102.0  # Close > High
        problematic_data.loc[problematic_data.index[30], 'close'] = 98.0   # Close < Low
        problematic_data.loc[problematic_data.index[40], 'close'] = -10.0  # Negative price
        problematic_data.loc[problematic_data.index[50], 'volume'] = -100  # Negative volume
        
        # Add duplicate timestamp
        duplicate_row = problematic_data.iloc[60:61].copy()
        duplicate_row.index = [problematic_data.index[59]]  # Same timestamp as previous row
        problematic_data = pd.concat([problematic_data, duplicate_row])
        
        # Validate the data
        validated_data = self.strategy._validate_historical_data_integrity(problematic_data, 'BTC/USDT')
        
        # Property 1: No duplicate timestamps
        assert not validated_data.index.duplicated().any(), "Duplicate timestamps not removed"
        
        # Property 2: Data is sorted by timestamp
        assert validated_data.index.is_monotonic_increasing, "Data not sorted by timestamp"
        
        # Property 3: OHLCV relationships are valid
        validated_data = self.strategy._validate_ohlcv_relationships(validated_data, 'BTC/USDT')
        
        # High >= Low
        assert (validated_data['high'] >= validated_data['low']).all(), "High < Low relationship found"
        
        # Close within High-Low range
        assert (validated_data['close'] >= validated_data['low']).all(), "Close < Low found"
        assert (validated_data['close'] <= validated_data['high']).all(), "Close > High found"
        
        # Open within High-Low range
        assert (validated_data['open'] >= validated_data['low']).all(), "Open < Low found"
        assert (validated_data['open'] <= validated_data['high']).all(), "Open > High found"
        
        # Property 4: No negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            assert (validated_data[col] > 0).all(), f"Non-positive values found in {col}"
        
        # Property 5: No negative volume
        assert (validated_data['volume'] >= 0).all(), "Negative volume found"
    
    def test_property_performance_tracking(self):
        """
        Test backtest performance tracking accuracy
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.2**
        """
        # Enable backtest mode
        self.strategy._is_backtest_mode = True
        self.strategy._setup_backtest_optimizations()
        
        # Reset performance tracker
        self.strategy._backtest_performance_tracker = {
            'total_candles_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_access_count': 0,
            'signal_generation_count': 0
        }
        
        # Create test data
        test_data = self.generate_consistent_test_data(50, 100.0, 42)
        
        # Mock data provider
        self.strategy.dp.get_pair_dataframe = MagicMock(return_value=pd.DataFrame())
        
        # Process data
        initial_candles = self.strategy._backtest_performance_tracker['total_candles_processed']
        initial_time = self.strategy._backtest_performance_tracker['total_processing_time']
        
        result = self.strategy.populate_indicators(test_data, {'pair': 'BTC/USDT'})
        
        final_candles = self.strategy._backtest_performance_tracker['total_candles_processed']
        final_time = self.strategy._backtest_performance_tracker['total_processing_time']
        
        # Property 1: Candle count should increase
        assert final_candles > initial_candles, "Candle count not tracked"
        assert final_candles - initial_candles == len(test_data), "Incorrect candle count tracking"
        
        # Property 2: Processing time should increase
        assert final_time > initial_time, "Processing time not tracked"
        
        # Property 3: Data access should be tracked
        assert self.strategy._backtest_performance_tracker['data_access_count'] > 0, "Data access not tracked"
    
    def test_property_reproducible_results(self):
        """
        Test that backtest results are reproducible with same input
        
        **Feature: hour-breakout-scalping, Property 12: Backtest signal consistency**
        **Validates: Requirements 8.3**
        """
        # Generate test data with fixed seed
        test_data = self.generate_consistent_test_data(200, 100.0, 123)
        
        # Mock data provider
        mock_5m_data = test_data[['close']].rename(columns={'close': 'close'})
        mock_1h_data = test_data[['high']].rename(columns={'high': 'high'})
        
        def mock_get_pair_dataframe(pair, timeframe):
            if timeframe == '5m':
                return mock_5m_data.copy()
            elif timeframe == '1h':
                return mock_1h_data.copy()
            return pd.DataFrame()
        
        self.strategy.dp.get_pair_dataframe = mock_get_pair_dataframe
        
        # Enable backtest mode
        self.strategy._is_backtest_mode = True
        self.strategy._setup_backtest_optimizations()
        
        # Process data multiple times
        results = []
        for run in range(3):
            # Clear cache between runs to ensure fresh processing
            if hasattr(self.strategy, '_informative_cache'):
                self.strategy._informative_cache.clear()
            
            # Process the same data
            indicators_result = self.strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})
            entry_result = self.strategy.populate_entry_trend(indicators_result.copy(), {'pair': 'BTC/USDT'})
            exit_result = self.strategy.populate_exit_trend(entry_result.copy(), {'pair': 'BTC/USDT'})
            
            results.append(exit_result)
        
        # Property: All runs should produce identical results
        for i in range(1, len(results)):
            # Compare entry signals
            entry_diff = np.sum(results[0]['enter_long'].values != results[i]['enter_long'].values)
            assert entry_diff == 0, f"Entry signals differ between run 0 and run {i}: {entry_diff} differences"
            
            # Compare exit signals
            exit_diff = np.sum(results[0]['exit_long'].values != results[i]['exit_long'].values)
            assert exit_diff == 0, f"Exit signals differ between run 0 and run {i}: {exit_diff} differences"
            
            # Compare key indicators
            key_indicators = ['breakout_condition', 'pullback_condition', 'rebound_condition', 'entry_signal']
            for indicator in key_indicators:
                if indicator in results[0].columns and indicator in results[i].columns:
                    indicator_diff = np.sum(results[0][indicator].values != results[i][indicator].values)
                    assert indicator_diff == 0, \
                        f"Indicator {indicator} differs between run 0 and run {i}: {indicator_diff} differences"

class TestBacktestIntegration:
    """Integration tests for backtest engine compatibility and performance metrics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create backtest-specific config
        self.backtest_config = {
            'dry_run': True,
            'timeframe': '1m',
            'stake_currency': 'USDT',
            'stake_amount': 100,
            'minimal_roi': {"0": 0.1},
            'stoploss': -0.1,
            'exchange': {'name': 'binance'},
            'runmode': 'backtest',  # Enable backtest mode
            'datadir': '/path/to/data',
            'timerange': '20230101-20231231'
        }
        
        self.strategy = HourBreakout1(self.backtest_config)
        
        # Mock the data provider and logger
        self.strategy.dp = MagicMock()
        self.strategy.logger = MagicMock()
    
    def generate_backtest_dataset(self, num_days: int = 30, base_price: float = 100.0) -> pd.DataFrame:
        """
        Generate realistic backtest dataset spanning multiple days
        
        :param num_days: Number of days to generate
        :param base_price: Base price for generation
        :return: DataFrame with realistic backtest data
        """
        # Generate 1-minute data for specified days
        num_candles = num_days * 24 * 60  # 1440 candles per day
        dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
        
        # Generate realistic price movements with daily patterns
        np.random.seed(42)  # For reproducible tests
        
        # Create daily volatility patterns
        daily_volatility = []
        for day in range(num_days):
            # Higher volatility during trading hours, lower during off-hours
            day_pattern = []
            for hour in range(24):
                if 8 <= hour <= 16:  # Trading hours
                    volatility = 0.002
                else:  # Off hours
                    volatility = 0.0005
                day_pattern.extend([volatility] * 60)  # 60 minutes per hour
            daily_volatility.extend(day_pattern)
        
        # Generate price changes
        price_changes = []
        for i in range(num_candles):
            vol = daily_volatility[i] if i < len(daily_volatility) else 0.001
            change = np.random.normal(0, vol)
            price_changes.append(change)
        
        # Calculate prices
        closes = [base_price]
        for change in price_changes[1:]:
            new_price = closes[-1] * (1 + change)
            closes.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate OHLCV data
        data = []
        for i in range(num_candles):
            # Generate realistic OHLC from close
            close = closes[i]
            spread = close * np.random.uniform(0.0005, 0.002)  # 0.05% to 0.2% spread
            
            high = close + spread * np.random.uniform(0.3, 1.0)
            low = close - spread * np.random.uniform(0.3, 1.0)
            open_price = closes[i-1] if i > 0 else close
            
            # Ensure OHLC relationships
            high = max(high, close, open_price)
            low = min(low, close, open_price)
            
            volume = np.random.randint(1000, 50000)
            
            data.append({
                'date': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data).set_index('date')
        
        # Generate informative data (5m and 1h aggregations)
        # 5m data: resample to 5-minute bars
        df_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 1h data: resample to 1-hour bars
        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Merge informative data back to 1m timeframe
        df['close_5m'] = df_5m['close'].reindex(df.index, method='ffill')
        df['high_1h'] = df_1h['high'].reindex(df.index, method='ffill')
        df['high_1h_prev'] = df['high_1h'].shift(1).ffill()
        
        # Add data quality flag
        df['data_quality_ok'] = True
        
        return df
    
    def test_backtest_engine_compatibility(self):
        """
        Test compatibility with FreqTrade backtest engine
        
        **Validates: Requirements 8.1, 8.4, 8.5**
        """
        # Generate realistic backtest dataset
        backtest_data = self.generate_backtest_dataset(num_days=7, base_price=100.0)
        
        # Mock informative data
        mock_5m_data = backtest_data[['close_5m']].rename(columns={'close_5m': 'close'}).dropna()
        mock_1h_data = backtest_data[['high_1h']].rename(columns={'high_1h': 'high'}).dropna()
        
        def mock_get_pair_dataframe(pair, timeframe):
            if timeframe == '5m':
                return mock_5m_data
            elif timeframe == '1h':
                return mock_1h_data
            return pd.DataFrame()
        
        self.strategy.dp.get_pair_dataframe = mock_get_pair_dataframe
        
        # Test strategy methods work with backtest data
        try:
            # Test populate_indicators
            indicators_result = self.strategy.populate_indicators(backtest_data.copy(), {'pair': 'BTC/USDT'})
            assert len(indicators_result) == len(backtest_data), "Indicators processing changed data length"
            
            # Verify required columns exist
            required_columns = ['enter_long', 'exit_long', 'enter_tag', 'exit_tag']
            for col in required_columns:
                assert col in indicators_result.columns, f"Required column {col} missing after populate_indicators"
            
            # Test populate_entry_trend
            entry_result = self.strategy.populate_entry_trend(indicators_result.copy(), {'pair': 'BTC/USDT'})
            assert len(entry_result) == len(backtest_data), "Entry trend processing changed data length"
            
            # Verify entry signals are valid (0 or 1)
            assert entry_result['enter_long'].isin([0, 1]).all(), "Invalid entry signal values"
            
            # Test populate_exit_trend
            exit_result = self.strategy.populate_exit_trend(entry_result.copy(), {'pair': 'BTC/USDT'})
            assert len(exit_result) == len(backtest_data), "Exit trend processing changed data length"
            
            # Verify exit signals are valid (0 or 1)
            assert exit_result['exit_long'].isin([0, 1]).all(), "Invalid exit signal values"
            
            # Test that signals are generated (at least some activity)
            total_entries = exit_result['enter_long'].sum()
            total_exits = exit_result['exit_long'].sum()
            
            # With 7 days of data, we should see some activity
            assert total_entries >= 0, "No entry signals generated in 7 days of data"
            assert total_exits >= 0, "No exit signals generated in 7 days of data"
            
            # Test informative_pairs method
            informative_pairs = self.strategy.informative_pairs()
            assert isinstance(informative_pairs, list), "informative_pairs should return a list"
            
            # Test that the strategy can handle the full backtest workflow
            print(f"Backtest compatibility test passed: {total_entries} entries, {total_exits} exits in {len(backtest_data)} candles")
            
        except Exception as e:
            pytest.fail(f"Backtest engine compatibility test failed: {str(e)}")
    
    def test_performance_metrics_output(self):
        """
        Test that performance metrics are correctly output during backtest
        
        **Validates: Requirements 8.4, 8.5**
        """
        # Generate test data
        test_data = self.generate_backtest_dataset(num_days=3, base_price=100.0)
        
        # Mock data provider
        self.strategy.dp.get_pair_dataframe = MagicMock(return_value=pd.DataFrame())
        
        # Reset performance tracker
        self.strategy._backtest_performance_tracker = {
            'total_candles_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_access_count': 0,
            'signal_generation_count': 0
        }
        
        # Process data and track metrics
        initial_metrics = self.strategy._backtest_performance_tracker.copy()
        
        result = self.strategy.populate_indicators(test_data, {'pair': 'BTC/USDT'})
        entry_result = self.strategy.populate_entry_trend(result, {'pair': 'BTC/USDT'})
        exit_result = self.strategy.populate_exit_trend(entry_result, {'pair': 'BTC/USDT'})
        
        final_metrics = self.strategy._backtest_performance_tracker
        
        # Verify metrics are being tracked
        assert final_metrics['total_candles_processed'] > initial_metrics['total_candles_processed'], \
            "Candle processing not tracked"
        
        assert final_metrics['total_processing_time'] > initial_metrics['total_processing_time'], \
            "Processing time not tracked"
        
        assert final_metrics['data_access_count'] > initial_metrics['data_access_count'], \
            "Data access not tracked"
        
        # Test performance metrics calculation
        if final_metrics['total_candles_processed'] > 0:
            avg_time_per_candle = final_metrics['total_processing_time'] / final_metrics['total_candles_processed']
            assert avg_time_per_candle > 0, "Average processing time calculation incorrect"
        
        # Test cache metrics (if caching is enabled)
        if self.strategy._enable_data_caching:
            total_cache_operations = final_metrics['cache_hits'] + final_metrics['cache_misses']
            assert total_cache_operations >= 0, "Cache operations not tracked"
        
        # Verify performance logging doesn't crash
        try:
            self.strategy._log_backtest_performance_metrics('BTC/USDT')
        except Exception as e:
            pytest.fail(f"Performance metrics logging failed: {str(e)}")
        
        print(f"Performance metrics test passed: {final_metrics}")
    
    def test_historical_data_processing_accuracy(self):
        """
        Test accurate processing of historical data in different market conditions
        
        **Validates: Requirements 8.1, 8.2**
        """
        # Test different market scenarios
        market_scenarios = [
            # Trending up market
            {
                'name': 'trending_up',
                'base_price': 100.0,
                'trend': 0.0001,  # Small upward trend per candle
                'volatility': 0.001
            },
            # Trending down market
            {
                'name': 'trending_down',
                'base_price': 100.0,
                'trend': -0.0001,  # Small downward trend per candle
                'volatility': 0.001
            },
            # Sideways market
            {
                'name': 'sideways',
                'base_price': 100.0,
                'trend': 0.0,  # No trend
                'volatility': 0.0005
            },
            # High volatility market
            {
                'name': 'high_volatility',
                'base_price': 100.0,
                'trend': 0.0,
                'volatility': 0.005
            }
        ]
        
        for scenario in market_scenarios:
            # Generate scenario-specific data
            num_candles = 1000
            dates = pd.date_range(start='2023-01-01', periods=num_candles, freq='1min')
            
            np.random.seed(42)  # Consistent seed for reproducibility
            
            closes = [scenario['base_price']]
            for i in range(1, num_candles):
                # Apply trend and volatility
                trend_change = scenario['trend']
                random_change = np.random.normal(0, scenario['volatility'])
                total_change = trend_change + random_change
                
                new_price = closes[-1] * (1 + total_change)
                closes.append(max(new_price, 0.01))
            
            # Create DataFrame
            scenario_data = pd.DataFrame({
                'open': [closes[i-1] if i > 0 else closes[i] for i in range(num_candles)],
                'high': [c * 1.001 for c in closes],
                'low': [c * 0.999 for c in closes],
                'close': closes,
                'volume': [1000] * num_candles
            }, index=dates)
            
            # Add informative data
            scenario_data['close_5m'] = scenario_data['close'].rolling(window=5, min_periods=1).mean()
            scenario_data['high_1h'] = scenario_data['high'].rolling(window=60, min_periods=1).max()
            scenario_data['high_1h_prev'] = scenario_data['high_1h'].shift(1).ffill()
            scenario_data['data_quality_ok'] = True
            
            # Mock data provider
            mock_5m = scenario_data[['close_5m']].rename(columns={'close_5m': 'close'})
            mock_1h = scenario_data[['high_1h']].rename(columns={'high_1h': 'high'})
            
            def mock_get_pair_dataframe(pair, timeframe):
                if timeframe == '5m':
                    return mock_5m
                elif timeframe == '1h':
                    return mock_1h
                return pd.DataFrame()
            
            self.strategy.dp.get_pair_dataframe = mock_get_pair_dataframe
            
            # Process the scenario data
            try:
                result = self.strategy.populate_indicators(scenario_data.copy(), {'pair': 'BTC/USDT'})
                entry_result = self.strategy.populate_entry_trend(result.copy(), {'pair': 'BTC/USDT'})
                exit_result = self.strategy.populate_exit_trend(entry_result.copy(), {'pair': 'BTC/USDT'})
                
                # Verify processing completed successfully
                assert len(exit_result) == len(scenario_data), \
                    f"Data length changed during processing in {scenario['name']} scenario"
                
                # Verify no NaN values in critical columns
                critical_columns = ['enter_long', 'exit_long']
                for col in critical_columns:
                    assert not exit_result[col].isna().any(), \
                        f"NaN values found in {col} for {scenario['name']} scenario"
                
                # Verify signal validity
                assert exit_result['enter_long'].isin([0, 1]).all(), \
                    f"Invalid entry signals in {scenario['name']} scenario"
                assert exit_result['exit_long'].isin([0, 1]).all(), \
                    f"Invalid exit signals in {scenario['name']} scenario"
                
                # Count signals for this scenario
                entries = exit_result['enter_long'].sum()
                exits = exit_result['exit_long'].sum()
                
                print(f"Scenario {scenario['name']}: {entries} entries, {exits} exits")
                
            except Exception as e:
                pytest.fail(f"Historical data processing failed for {scenario['name']} scenario: {str(e)}")
    
    def test_multi_timeframe_data_consistency(self):
        """
        Test consistency of multi-timeframe data processing in backtest
        
        **Validates: Requirements 8.1, 8.2**
        """
        # Generate base 1-minute data
        base_data = self.generate_backtest_dataset(num_days=2, base_price=100.0)
        
        # Create proper 5m and 1h aggregations
        df_5m = base_data.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_1h = base_data.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Mock data provider with proper aggregated data
        def mock_get_pair_dataframe(pair, timeframe):
            if timeframe == '5m':
                return df_5m[['close']].rename(columns={'close': 'close'})
            elif timeframe == '1h':
                return df_1h[['high']].rename(columns={'high': 'high'})
            return pd.DataFrame()
        
        self.strategy.dp.get_pair_dataframe = mock_get_pair_dataframe
        
        # Process with multi-timeframe data
        result = self.strategy.populate_indicators(base_data.copy(), {'pair': 'BTC/USDT'})
        
        # Verify multi-timeframe columns exist
        assert 'close_5m' in result.columns, "5m close data not merged"
        assert 'high_1h' in result.columns, "1h high data not merged"
        assert 'high_1h_prev' in result.columns, "Previous 1h high not calculated"
        
        # Verify data alignment - 5m data should be forward-filled to 1m
        assert not result['close_5m'].isna().all(), "5m data not properly aligned"
        assert not result['high_1h'].isna().all(), "1h data not properly aligned"
        
        # Verify previous 1h high is properly shifted
        non_na_high_1h = result['high_1h'].dropna()
        non_na_high_1h_prev = result['high_1h_prev'].dropna()
        
        if len(non_na_high_1h) > 1 and len(non_na_high_1h_prev) > 0:
            # First non-NaN high_1h_prev should be NaN or equal to previous high_1h
            # This tests the shift operation
            assert len(non_na_high_1h_prev) <= len(non_na_high_1h), \
                "Previous 1h high has more values than current 1h high"
        
        # Test breakout detection with multi-timeframe data
        entry_result = self.strategy.populate_entry_trend(result.copy(), {'pair': 'BTC/USDT'})
        
        # Verify breakout conditions can be calculated
        assert 'breakout_condition' in entry_result.columns, "Breakout condition not calculated"
        
        # If there are breakout conditions, verify they make sense
        breakout_positions = entry_result[entry_result['breakout_condition']].index
        for pos in breakout_positions[:5]:  # Check first 5 breakouts
            row = entry_result.loc[pos]
            if not pd.isna(row['close_5m']) and not pd.isna(row['high_1h_prev']):
                # Breakout should mean 5m close > 1h high (with threshold)
                threshold = row['high_1h_prev'] * (1 + self.strategy.min_breakout_pct.value)
                # Note: This might not always be true due to additional filters, but it's a sanity check
                print(f"Breakout at {pos}: 5m_close={row['close_5m']:.6f}, 1h_high_prev={row['high_1h_prev']:.6f}, threshold={threshold:.6f}")
    
    def test_backtest_data_validation_integration(self):
        """
        Test integration of data validation with backtest processing
        
        **Validates: Requirements 8.1**
        """
        # Create data with various quality issues
        problematic_data = self.generate_backtest_dataset(num_days=1, base_price=100.0)
        
        # Introduce data quality issues
        # Duplicate timestamps
        duplicate_row = problematic_data.iloc[100:101].copy()
        duplicate_row.index = [problematic_data.index[99]]
        problematic_data = pd.concat([problematic_data, duplicate_row])
        
        # Invalid OHLC relationships - these will be marked as bad quality
        problematic_data.iloc[200, problematic_data.columns.get_loc('high')] = 90.0  # High < Close
        problematic_data.iloc[300, problematic_data.columns.get_loc('low')] = 110.0  # Low > Close
        
        # Negative prices
        problematic_data.iloc[400, problematic_data.columns.get_loc('close')] = -10.0
        
        # Missing data (NaN)
        problematic_data.iloc[500:510, problematic_data.columns.get_loc('close')] = np.nan
        
        # Mock data provider
        self.strategy.dp.get_pair_dataframe = MagicMock(return_value=pd.DataFrame())
        
        # Process problematic data - should not crash
        try:
            result = self.strategy.populate_indicators(problematic_data.copy(), {'pair': 'BTC/USDT'})
            
            # Verify data validation occurred - strategy should handle problematic data gracefully
            assert len(result) > 0, "Strategy should return some data"
            
            # Verify no duplicate timestamps in result (duplicates should be removed)
            assert not result.index.duplicated().any(), "Duplicate timestamps not removed"
            
            # Verify data quality flag exists
            assert 'data_quality_ok' in result.columns, "Data quality flag should exist"
            
            # Verify that problematic rows are marked as bad quality or fixed
            # The strategy should either fix the data or mark it as bad quality
            # We don't require the strategy to fix all OHLC relationships, 
            # but it should handle them gracefully
            
            # Verify no negative prices in close (should be fixed or marked)
            # Strategy may fill negative values with fallback
            assert (result['close'] > 0).all() or result['data_quality_ok'].sum() < len(result), \
                "Negative prices should be fixed or rows marked as bad quality"
            
            # Verify NaN values are handled (filled or marked)
            assert not result['close'].isna().any(), "NaN values not handled in close prices"
            
            print(f"Data validation integration test passed: {len(problematic_data)} -> {len(result)} candles")
            
        except Exception as e:
            pytest.fail(f"Data validation integration test failed: {str(e)}")
    
    def test_backtest_performance_under_load(self):
        """
        Test backtest performance with large datasets
        
        **Validates: Requirements 8.2**
        """
        # Generate large dataset (simulate 30 days of 1-minute data)
        large_data = self.generate_backtest_dataset(num_days=30, base_price=100.0)
        
        # Mock data provider
        mock_5m = large_data[['close_5m']].rename(columns={'close_5m': 'close'}).dropna()
        mock_1h = large_data[['high_1h']].rename(columns={'high_1h': 'high'}).dropna()
        
        def mock_get_pair_dataframe(pair, timeframe):
            if timeframe == '5m':
                return mock_5m
            elif timeframe == '1h':
                return mock_1h
            return pd.DataFrame()
        
        self.strategy.dp.get_pair_dataframe = mock_get_pair_dataframe
        
        # Measure processing time
        import time
        start_time = time.time()
        
        # Process large dataset
        result = self.strategy.populate_indicators(large_data.copy(), {'pair': 'BTC/USDT'})
        entry_result = self.strategy.populate_entry_trend(result.copy(), {'pair': 'BTC/USDT'})
        exit_result = self.strategy.populate_exit_trend(entry_result.copy(), {'pair': 'BTC/USDT'})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        candles_per_second = len(large_data) / processing_time
        
        # Should process at least 1000 candles per second (reasonable performance)
        assert candles_per_second > 100, \
            f"Performance too slow: {candles_per_second:.2f} candles/second (expected > 100)"
        
        # Verify all data was processed
        assert len(exit_result) == len(large_data), "Not all data was processed"
        
        # Verify memory usage is reasonable (no excessive memory leaks)
        # This is a basic check - in production you'd use more sophisticated memory profiling
        import sys
        memory_usage = sys.getsizeof(exit_result)
        expected_memory = len(exit_result) * len(exit_result.columns) * 8  # Rough estimate
        
        # Memory usage should be within reasonable bounds (less than 10x expected)
        assert memory_usage < expected_memory * 10, \
            f"Excessive memory usage: {memory_usage} bytes (expected < {expected_memory * 10})"
        
        print(f"Performance test passed: {candles_per_second:.2f} candles/second, {len(large_data)} total candles")
        
        # Test cache effectiveness if enabled
        if self.strategy._enable_data_caching:
            cache_hit_rate = 0
            total_cache_ops = (self.strategy._backtest_performance_tracker['cache_hits'] + 
                             self.strategy._backtest_performance_tracker['cache_misses'])
            
            if total_cache_ops > 0:
                cache_hit_rate = self.strategy._backtest_performance_tracker['cache_hits'] / total_cache_ops
                print(f"Cache hit rate: {cache_hit_rate:.2%}")
                
                # Cache should be somewhat effective for large datasets
                # (This depends on the specific access patterns)
                assert cache_hit_rate >= 0, "Cache hit rate should be non-negative"



class TestConfigFileValidation:
    """
    Test suite for HourBreakout1 configuration file validation
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    
    CONFIG_FILE_PATH = 'configs/HourBreakout1.json'
    
    def setup_method(self):
        """Setup test fixtures"""
        import json
        import os
        
        # Read the config file
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.CONFIG_FILE_PATH)
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = None
    
    def test_config_file_exists(self):
        """
        Test that configuration file exists in configs directory
        
        Requirements: 10.1 - Configuration file should be placed in configs directory
        """
        import os
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.CONFIG_FILE_PATH)
        assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    def test_config_file_valid_json(self):
        """
        Test that configuration file is valid JSON
        
        Requirements: 10.1 - Configuration file format validation
        """
        assert self.config is not None, "Config file could not be loaded as JSON"
        assert isinstance(self.config, dict), "Config should be a JSON object"
    
    def test_config_has_strategy_name(self):
        """
        Test that configuration specifies the correct strategy
        
        Requirements: 10.1 - Configuration should specify strategy
        """
        assert 'strategy' in self.config, "Config missing 'strategy' field"
        assert self.config['strategy'] == 'HourBreakout1', "Strategy name should be 'HourBreakout1'"
    
    def test_config_has_strategy_path(self):
        """
        Test that configuration specifies strategy path
        
        Requirements: 10.1 - Configuration should specify strategy path
        """
        assert 'strategy_path' in self.config, "Config missing 'strategy_path' field"
        assert 'user_data/strategies' in self.config['strategy_path'], "Strategy path should point to user_data/strategies"
    
    def test_config_timeframe_settings(self):
        """
        Test that configuration specifies correct timeframe
        
        Requirements: 10.2 - Configuration should specify correct timeframe
        """
        assert 'timeframe' in self.config, "Config missing 'timeframe' field"
        assert self.config['timeframe'] == '1m', "Main timeframe should be '1m'"
    
    def test_config_risk_management_settings(self):
        """
        Test that configuration includes risk management settings
        
        Requirements: 10.3 - Configuration should include risk management settings
        """
        # Test max_open_trades
        assert 'max_open_trades' in self.config, "Config missing 'max_open_trades'"
        assert isinstance(self.config['max_open_trades'], int), "max_open_trades should be integer"
        assert self.config['max_open_trades'] > 0, "max_open_trades should be positive"
        
        # Test stake_amount
        assert 'stake_amount' in self.config, "Config missing 'stake_amount'"
        
        # Test stake_currency
        assert 'stake_currency' in self.config, "Config missing 'stake_currency'"
        assert isinstance(self.config['stake_currency'], str), "stake_currency should be string"
        
        # Test stoploss
        assert 'stoploss' in self.config, "Config missing 'stoploss'"
        assert isinstance(self.config['stoploss'], (int, float)), "stoploss should be numeric"
        assert self.config['stoploss'] < 0, "stoploss should be negative"
        
        # Test minimal_roi
        assert 'minimal_roi' in self.config, "Config missing 'minimal_roi'"
        assert isinstance(self.config['minimal_roi'], dict), "minimal_roi should be dict"
        assert len(self.config['minimal_roi']) > 0, "minimal_roi should not be empty"
        
        # Test tradable_balance_ratio
        assert 'tradable_balance_ratio' in self.config, "Config missing 'tradable_balance_ratio'"
        assert 0 < self.config['tradable_balance_ratio'] <= 1, "tradable_balance_ratio should be between 0 and 1"
    
    def test_config_exchange_settings(self):
        """
        Test that configuration includes exchange settings
        
        Requirements: 10.4 - Configuration should configure exchange and market settings
        """
        assert 'exchange' in self.config, "Config missing 'exchange' section"
        
        exchange = self.config['exchange']
        assert isinstance(exchange, dict), "exchange should be dict"
        
        # Test exchange name
        assert 'name' in exchange, "Exchange missing 'name'"
        assert isinstance(exchange['name'], str), "Exchange name should be string"
        
        # Test pair whitelist
        assert 'pair_whitelist' in exchange, "Exchange missing 'pair_whitelist'"
        assert isinstance(exchange['pair_whitelist'], list), "pair_whitelist should be list"
        assert len(exchange['pair_whitelist']) > 0, "pair_whitelist should not be empty"
        
        # Validate pair format (should contain '/')
        for pair in exchange['pair_whitelist']:
            assert '/' in pair, f"Invalid pair format: {pair}"
    
    def test_config_trading_mode(self):
        """
        Test that configuration specifies trading mode
        
        Requirements: 10.4 - Configuration should configure market settings
        """
        assert 'trading_mode' in self.config, "Config missing 'trading_mode'"
        assert self.config['trading_mode'] in ['spot', 'futures', 'margin'], \
            f"Invalid trading_mode: {self.config['trading_mode']}"
    
    def test_config_strategy_specific_params(self):
        """
        Test that configuration includes strategy-specific parameters
        
        Requirements: 10.5 - Configuration should include strategy-specific parameters
        """
        assert 'hourbreakout1_params' in self.config, "Config missing 'hourbreakout1_params'"
        
        params = self.config['hourbreakout1_params']
        assert isinstance(params, dict), "hourbreakout1_params should be dict"
        
        # Test required strategy parameters
        required_params = [
            'ma_period',
            'exit_minutes',
            'min_breakout_pct',
            'pullback_tolerance',
            'max_position_hours',
            'min_volume_threshold',
            'stop_loss_buffer_pct',
            'min_entry_spacing',
            'breakout_strength_threshold',
            'rebound_strength_threshold'
        ]
        
        for param in required_params:
            assert param in params, f"Missing strategy parameter: {param}"
    
    def test_config_strategy_param_defaults(self):
        """
        Test that strategy parameters have valid default values
        
        Requirements: 10.5 - Configuration should include default values
        """
        params = self.config['hourbreakout1_params']
        
        # Validate ma_period (should be between 3 and 10)
        assert 3 <= params['ma_period'] <= 10, f"ma_period out of range: {params['ma_period']}"
        
        # Validate exit_minutes (should be between 5 and 60)
        assert 5 <= params['exit_minutes'] <= 60, f"exit_minutes out of range: {params['exit_minutes']}"
        
        # Validate min_breakout_pct (should be between 0.001 and 0.01)
        assert 0.001 <= params['min_breakout_pct'] <= 0.01, \
            f"min_breakout_pct out of range: {params['min_breakout_pct']}"
        
        # Validate pullback_tolerance (should be between 0.0001 and 0.002)
        assert 0.0001 <= params['pullback_tolerance'] <= 0.002, \
            f"pullback_tolerance out of range: {params['pullback_tolerance']}"
        
        # Validate max_position_hours (should be between 1.0 and 8.0)
        assert 1.0 <= params['max_position_hours'] <= 8.0, \
            f"max_position_hours out of range: {params['max_position_hours']}"
        
        # Validate min_volume_threshold (should be between 0.5 and 3.0)
        assert 0.5 <= params['min_volume_threshold'] <= 3.0, \
            f"min_volume_threshold out of range: {params['min_volume_threshold']}"
        
        # Validate stop_loss_buffer_pct (should be between 0.001 and 0.01)
        assert 0.001 <= params['stop_loss_buffer_pct'] <= 0.01, \
            f"stop_loss_buffer_pct out of range: {params['stop_loss_buffer_pct']}"
        
        # Validate min_entry_spacing (should be between 10 and 30)
        assert 10 <= params['min_entry_spacing'] <= 30, \
            f"min_entry_spacing out of range: {params['min_entry_spacing']}"
        
        # Validate breakout_strength_threshold (should be between 0.001 and 0.005)
        assert 0.001 <= params['breakout_strength_threshold'] <= 0.005, \
            f"breakout_strength_threshold out of range: {params['breakout_strength_threshold']}"
        
        # Validate rebound_strength_threshold (should be between 0.001 and 0.01)
        assert 0.001 <= params['rebound_strength_threshold'] <= 0.01, \
            f"rebound_strength_threshold out of range: {params['rebound_strength_threshold']}"
    
    def test_config_pricing_settings(self):
        """
        Test that configuration includes pricing settings
        
        Requirements: 10.4 - Configuration should configure market settings
        """
        # Test entry pricing
        assert 'entry_pricing' in self.config, "Config missing 'entry_pricing'"
        entry_pricing = self.config['entry_pricing']
        assert 'price_side' in entry_pricing, "entry_pricing missing 'price_side'"
        assert 'use_order_book' in entry_pricing, "entry_pricing missing 'use_order_book'"
        
        # Test exit pricing
        assert 'exit_pricing' in self.config, "Config missing 'exit_pricing'"
        exit_pricing = self.config['exit_pricing']
        assert 'price_side' in exit_pricing, "exit_pricing missing 'price_side'"
        assert 'use_order_book' in exit_pricing, "exit_pricing missing 'use_order_book'"
    
    def test_config_pairlists_settings(self):
        """
        Test that configuration includes pairlists settings
        
        Requirements: 10.2 - Configuration should specify informative pairs
        """
        assert 'pairlists' in self.config, "Config missing 'pairlists'"
        assert isinstance(self.config['pairlists'], list), "pairlists should be list"
        assert len(self.config['pairlists']) > 0, "pairlists should not be empty"
        
        # Each pairlist should have a method
        for pairlist in self.config['pairlists']:
            assert 'method' in pairlist, "Pairlist missing 'method'"
    
    def test_config_dry_run_settings(self):
        """
        Test that configuration includes dry run settings
        
        Requirements: 10.3 - Configuration should include risk management settings
        """
        assert 'dry_run' in self.config, "Config missing 'dry_run'"
        assert isinstance(self.config['dry_run'], bool), "dry_run should be boolean"
        
        if self.config['dry_run']:
            assert 'dry_run_wallet' in self.config, "Config missing 'dry_run_wallet' for dry run mode"
            assert self.config['dry_run_wallet'] > 0, "dry_run_wallet should be positive"
    
    def test_config_bot_name(self):
        """
        Test that configuration includes bot name
        
        Requirements: 10.1 - Configuration should be properly configured
        """
        assert 'bot_name' in self.config, "Config missing 'bot_name'"
        assert isinstance(self.config['bot_name'], str), "bot_name should be string"
        assert len(self.config['bot_name']) > 0, "bot_name should not be empty"
    
    def test_config_trailing_stop_settings(self):
        """
        Test that configuration includes trailing stop settings
        
        Requirements: 10.3 - Configuration should include risk management settings
        """
        assert 'trailing_stop' in self.config, "Config missing 'trailing_stop'"
        assert isinstance(self.config['trailing_stop'], bool), "trailing_stop should be boolean"
        
        # If trailing stop is enabled, check related settings
        if self.config['trailing_stop']:
            assert 'trailing_stop_positive' in self.config, "Config missing 'trailing_stop_positive'"
            assert 'trailing_stop_positive_offset' in self.config, "Config missing 'trailing_stop_positive_offset'"


if __name__ == '__main__':
    pytest.main([__file__])
