# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class HourBreakout1(IStrategy):
    """
    HourBreakout1 - Multi-timeframe breakout scalping strategy
    
    Strategy Logic:
    1. Breakout Detection: 5m close price breaks above previous 1h high
    2. Pullback Confirmation: 1m price pulls back to MA5 support
    3. Rebound Entry: 1m price rebounds from MA5 for long entry
    4. Risk Management: Stop loss based on 1h high, time-based take profit
    
    Requirements: 7.1, 7.2, 7.3, 9.1
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy (main timeframe)
    timeframe = '1m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "15": 0.03,
        "0": 0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # HyperOpt parameters for strategy optimization
    # Requirements: 9.1, 9.2, 9.3, 9.4, 9.5 - HyperOpt parameter space definitions
    
    # MA period for pullback detection (Requirements: 9.2, 9.3)
    ma_period = IntParameter(3, 10, default=5, space="buy", optimize=True)
    
    # Time-based exit parameter in minutes (Requirements: 9.2, 9.4)
    exit_minutes = IntParameter(5, 60, default=15, space="sell", optimize=True)
    
    # Minimum breakout percentage to filter noise (Requirements: 9.4, 9.5)
    min_breakout_pct = DecimalParameter(0.001, 0.01, decimals=4, default=0.002, space="buy", optimize=True)
    
    # Pullback tolerance for MA5 touch detection (Requirements: 9.4, 9.5)
    pullback_tolerance = DecimalParameter(0.0001, 0.002, decimals=4, default=0.0005, space="buy", optimize=True)
    
    # Risk management parameters (Requirements: 9.4, 9.5)
    # Maximum position duration before forced exit (hours)
    max_position_hours = DecimalParameter(1.0, 8.0, decimals=1, default=4.0, space="sell", optimize=True)
    
    # Minimum volume threshold for entry (as percentage of average volume)
    min_volume_threshold = DecimalParameter(0.5, 3.0, decimals=1, default=1.0, space="buy", optimize=True)
    
    # Stop loss buffer percentage below 1h high
    stop_loss_buffer_pct = DecimalParameter(0.001, 0.01, decimals=4, default=0.005, space="sell", optimize=True)
    
    # Minimum candles between entries (duplicate prevention)
    min_entry_spacing = IntParameter(10, 30, default=15, space="buy", optimize=True)
    
    # Breakout strength threshold (minimum percentage above 1h high)
    breakout_strength_threshold = DecimalParameter(0.001, 0.005, decimals=4, default=0.002, space="buy", optimize=True)
    
    # Rebound strength threshold (minimum percentage above MA for entry)
    rebound_strength_threshold = DecimalParameter(0.001, 0.01, decimals=4, default=0.003, space="buy", optimize=True)
    
    def _validate_hyperopt_parameters(self) -> None:
        """
        Validate HyperOpt parameters and handle boundary conditions
        
        Requirements: 9.5 - Parameter validation logic
        """
        try:
            # Initialize logger if not available
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                self.logger = logging.getLogger(__name__)
            
            # Store default values (initial values are the defaults)
            default_values = {
                'ma_period': 5,
                'exit_minutes': 15,
                'min_breakout_pct': 0.002,
                'pullback_tolerance': 0.0005,
                'max_position_hours': 4.0,
                'min_volume_threshold': 1.0,
                'stop_loss_buffer_pct': 0.005,
                'min_entry_spacing': 15,
                'breakout_strength_threshold': 0.002,
                'rebound_strength_threshold': 0.003
            }
            
            # Validate MA period parameter
            ma_period_value = self.ma_period.value
            if not (3 <= ma_period_value <= 10):
                self.logger.warning(f"MA period {ma_period_value} out of valid range [3, 10], using default {default_values['ma_period']}")
                self.ma_period.value = default_values['ma_period']
            
            # Validate exit minutes parameter
            exit_minutes_value = self.exit_minutes.value
            if not (5 <= exit_minutes_value <= 60):
                self.logger.warning(f"Exit minutes {exit_minutes_value} out of valid range [5, 60], using default {default_values['exit_minutes']}")
                self.exit_minutes.value = default_values['exit_minutes']
            
            # Validate minimum breakout percentage
            min_breakout_pct_value = self.min_breakout_pct.value
            if not (0.001 <= min_breakout_pct_value <= 0.01):
                self.logger.warning(f"Min breakout percentage {min_breakout_pct_value} out of valid range [0.001, 0.01], using default {default_values['min_breakout_pct']}")
                self.min_breakout_pct.value = default_values['min_breakout_pct']
            
            # Validate pullback tolerance
            pullback_tolerance_value = self.pullback_tolerance.value
            if not (0.0001 <= pullback_tolerance_value <= 0.002):
                self.logger.warning(f"Pullback tolerance {pullback_tolerance_value} out of valid range [0.0001, 0.002], using default {default_values['pullback_tolerance']}")
                self.pullback_tolerance.value = default_values['pullback_tolerance']
            
            # Validate max position hours
            max_position_hours_value = self.max_position_hours.value
            if not (1.0 <= max_position_hours_value <= 8.0):
                self.logger.warning(f"Max position hours {max_position_hours_value} out of valid range [1.0, 8.0], using default {default_values['max_position_hours']}")
                self.max_position_hours.value = default_values['max_position_hours']
            
            # Validate minimum volume threshold
            min_volume_threshold_value = self.min_volume_threshold.value
            if not (0.5 <= min_volume_threshold_value <= 3.0):
                self.logger.warning(f"Min volume threshold {min_volume_threshold_value} out of valid range [0.5, 3.0], using default {default_values['min_volume_threshold']}")
                self.min_volume_threshold.value = default_values['min_volume_threshold']
            
            # Validate stop loss buffer percentage
            stop_loss_buffer_pct_value = self.stop_loss_buffer_pct.value
            if not (0.001 <= stop_loss_buffer_pct_value <= 0.01):
                self.logger.warning(f"Stop loss buffer percentage {stop_loss_buffer_pct_value} out of valid range [0.001, 0.01], using default {default_values['stop_loss_buffer_pct']}")
                self.stop_loss_buffer_pct.value = default_values['stop_loss_buffer_pct']
            
            # Validate minimum entry spacing
            min_entry_spacing_value = self.min_entry_spacing.value
            if not (10 <= min_entry_spacing_value <= 30):
                self.logger.warning(f"Min entry spacing {min_entry_spacing_value} out of valid range [10, 30], using default {default_values['min_entry_spacing']}")
                self.min_entry_spacing.value = default_values['min_entry_spacing']
            
            # Validate breakout strength threshold
            breakout_strength_threshold_value = self.breakout_strength_threshold.value
            if not (0.001 <= breakout_strength_threshold_value <= 0.005):
                self.logger.warning(f"Breakout strength threshold {breakout_strength_threshold_value} out of valid range [0.001, 0.005], using default {default_values['breakout_strength_threshold']}")
                self.breakout_strength_threshold.value = default_values['breakout_strength_threshold']
            
            # Validate rebound strength threshold
            rebound_strength_threshold_value = self.rebound_strength_threshold.value
            if not (0.001 <= rebound_strength_threshold_value <= 0.01):
                self.logger.warning(f"Rebound strength threshold {rebound_strength_threshold_value} out of valid range [0.001, 0.01], using default {default_values['rebound_strength_threshold']}")
                self.rebound_strength_threshold.value = default_values['rebound_strength_threshold']
            
            # Cross-parameter validation (Requirements: 9.5)
            # Ensure exit_minutes is reasonable compared to max_position_hours
            max_position_minutes = max_position_hours_value * 60
            if exit_minutes_value > max_position_minutes:
                self.logger.warning(f"Exit minutes {exit_minutes_value} exceeds max position duration {max_position_minutes} minutes")
                # Adjust exit_minutes to be within max_position_hours
                self.exit_minutes.value = min(exit_minutes_value, int(max_position_minutes))
            
            # Ensure breakout_strength_threshold is not larger than min_breakout_pct
            if breakout_strength_threshold_value > min_breakout_pct_value:
                self.logger.warning(f"Breakout strength threshold {breakout_strength_threshold_value} exceeds min breakout percentage {min_breakout_pct_value}")
                self.breakout_strength_threshold.value = min(breakout_strength_threshold_value, min_breakout_pct_value)
            
            # Ensure rebound_strength_threshold is reasonable compared to pullback_tolerance
            if rebound_strength_threshold_value < pullback_tolerance_value:
                self.logger.warning(f"Rebound strength threshold {rebound_strength_threshold_value} is less than pullback tolerance {pullback_tolerance_value}")
                # This is acceptable but log for awareness
            
            self.logger.info("HyperOpt parameter validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in parameter validation: {str(e)}")
            # Use default values on validation error
            self._reset_parameters_to_defaults()
    
    def _reset_parameters_to_defaults(self) -> None:
        """
        Reset all HyperOpt parameters to their default values
        
        Requirements: 9.5 - Handle parameter validation failures
        """
        try:
            # Default values
            self.ma_period.value = 5
            self.exit_minutes.value = 15
            self.min_breakout_pct.value = 0.002
            self.pullback_tolerance.value = 0.0005
            self.max_position_hours.value = 4.0
            self.min_volume_threshold.value = 1.0
            self.stop_loss_buffer_pct.value = 0.005
            self.min_entry_spacing.value = 15
            self.breakout_strength_threshold.value = 0.002
            self.rebound_strength_threshold.value = 0.003
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("All HyperOpt parameters reset to default values")
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error resetting parameters to defaults: {str(e)}")

    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        
        Requirements: 7.3 - Configure multi-timeframe informative pairs
        
        :return: List of tuples in the format (pair, interval)
        """
        pairs = self.dp.current_whitelist() if self.dp else []
        informative_pairs = []
        
        # Add 5m and 1h timeframes for all pairs in whitelist
        for pair in pairs:
            informative_pairs.extend([
                (pair, '5m'),   # For breakout detection
                (pair, '1h'),   # For high price reference
            ])
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Multi-timeframe data management and technical indicator calculation
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1 - Multi-timeframe data access and indicator calculation
        Requirements: 8.1, 8.2 - Backtest support and optimization
        
        :param dataframe: Dataframe with data from the exchange (1m timeframe)
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        try:
            # Get current pair from metadata
            pair = metadata['pair']
            
            # Initialize logger if not available (for testing)
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                self.logger = logging.getLogger(__name__)
            
            # Apply backtest optimizations if enabled (Requirements: 8.1, 8.2)
            if hasattr(self, '_is_backtest_mode') and self._is_backtest_mode:
                dataframe = self._optimize_historical_data_processing(dataframe, metadata)
            
            # Track data access for performance monitoring
            if hasattr(self, '_backtest_performance_tracker'):
                self._backtest_performance_tracker['data_access_count'] += 1
            
            # Get 5m informative data for breakout detection (Requirements: 1.2)
            # Use caching for backtest performance (Requirements: 8.2)
            informative_5m = self._get_cached_informative_data(pair, '5m')
            if informative_5m is None:
                informative_5m = self.dp.get_pair_dataframe(pair=pair, timeframe='5m')
                if hasattr(self, '_is_backtest_mode') and self._is_backtest_mode:
                    self._cache_informative_data(pair, '5m', informative_5m)
            
            if informative_5m.empty:
                self.logger.warning(f"No 5m data available for {pair}")
                # Initialize with safe defaults
                dataframe['close_5m'] = dataframe['close']
            else:
                # Merge 5m data with proper time alignment (Requirements: 1.4)
                dataframe = merge_informative_pair(dataframe, informative_5m, self.timeframe, '5m', ffill=True)
            
            # Get 1h informative data for high price reference (Requirements: 1.3)
            # Use caching for backtest performance (Requirements: 8.2)
            informative_1h = self._get_cached_informative_data(pair, '1h')
            if informative_1h is None:
                informative_1h = self.dp.get_pair_dataframe(pair=pair, timeframe='1h')
                if hasattr(self, '_is_backtest_mode') and self._is_backtest_mode:
                    self._cache_informative_data(pair, '1h', informative_1h)
            
            if informative_1h.empty:
                self.logger.warning(f"No 1h data available for {pair}")
                # Initialize with safe defaults
                dataframe['high_1h'] = dataframe['high']
            else:
                # Merge 1h data with proper time alignment (Requirements: 1.4)
                dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
            
            # Data validation and error handling (Requirements: 1.4, 1.5)
            if len(dataframe) < self.startup_candle_count:
                self.logger.warning(f"Insufficient data for {pair}: {len(dataframe)} candles, need {self.startup_candle_count}")
            
            # Validate critical columns exist
            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for {pair}: {missing_columns}")
            
            # Check for data quality issues (log only, don't modify original OHLCV data)
            if dataframe['close'].isna().any():
                self.logger.debug(f"NaN values detected in close prices for {pair}")
            
            if (dataframe['close'] <= 0).any():
                self.logger.debug(f"Zero or negative prices detected for {pair}")
            
            # Calculate MA5 on 1m timeframe for pullback detection (Requirements: 2.1, 3.1)
            ma_period = self.ma_period.value
            dataframe[f'ma{ma_period}'] = ta.SMA(dataframe['close'], timeperiod=ma_period)
            
            # Validate MA calculation
            if dataframe[f'ma{ma_period}'].isna().all():
                self.logger.error(f"MA{ma_period} calculation failed for {pair}")
                # Use close price as fallback
                dataframe[f'ma{ma_period}'] = dataframe['close']
            
            # Calculate previous 1h high for breakout detection (Requirements: 2.1)
            if 'high_1h' in dataframe.columns:
                # Shift 1h high by 1 to get previous period high
                dataframe['high_1h_prev'] = dataframe['high_1h'].shift(1)
                
                # Validate 1h high data
                if dataframe['high_1h_prev'].isna().all():
                    self.logger.warning(f"No previous 1h high data for {pair}")
                    # Use current high as fallback
                    dataframe['high_1h_prev'] = dataframe['high']
                else:
                    # Forward fill any NaN values in 1h high
                    dataframe['high_1h_prev'] = dataframe['high_1h_prev'].ffill()
            else:
                self.logger.warning(f"No 1h high column found for {pair}")
                dataframe['high_1h_prev'] = dataframe['high']
            
            # Initialize condition columns for strategy logic
            dataframe['breakout_condition'] = False
            dataframe['pullback_condition'] = False
            dataframe['rebound_condition'] = False
            
            # Add data quality flags for monitoring
            dataframe['data_quality_ok'] = True
            
            # Mark rows with insufficient lookback data
            if len(dataframe) > 0:
                lookback_required = max(ma_period, self.startup_candle_count)
                dataframe.loc[:lookback_required-1, 'data_quality_ok'] = False
            
            # Implement breakout condition detection (Requirements: 2.2, 2.3, 2.4, 2.5)
            self._detect_breakout_conditions(dataframe, pair)
            
            # Implement state management logic (Requirements: 2.4, 3.5)
            self._manage_breakout_state(dataframe, pair)
            
            # Implement pullback condition detection (Requirements: 3.2, 3.3, 3.4)
            self._detect_pullback_conditions(dataframe, pair)
            
            # Implement rebound detection and entry signal generation (Requirements: 4.1, 4.2, 4.3, 4.5)
            self._detect_rebound_conditions(dataframe, pair)
            
            # Log successful processing
            self.logger.debug(f"Successfully processed {len(dataframe)} candles for {pair}")
            
        except Exception as e:
            self.logger.error(f"Error in populate_indicators for {pair}: {str(e)}")
            # Ensure we return a valid dataframe even on error
            if f'ma{self.ma_period.value}' not in dataframe.columns:
                dataframe[f'ma{self.ma_period.value}'] = dataframe.get('close', 0)
            if 'high_1h_prev' not in dataframe.columns:
                dataframe['high_1h_prev'] = dataframe.get('high', 0)
            
            # Initialize condition columns
            dataframe['breakout_condition'] = False
            dataframe['pullback_condition'] = False
            dataframe['rebound_condition'] = False
            dataframe['entry_signal'] = False
            dataframe['entry_signal_strength'] = 0.0
            dataframe['pullback_completed'] = False
            dataframe['last_entry_candle'] = -1
            dataframe['data_quality_ok'] = False
        
        # Validate and ensure FreqTrade compatibility (Requirements: 7.4)
        dataframe = self._ensure_freqtrade_dataframe_compatibility(dataframe, metadata)
        
        return dataframe

    def _detect_breakout_conditions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Detect breakout conditions by comparing 5m close with previous 1h high
        
        Requirements: 2.2, 2.3, 2.4, 2.5 - Breakout condition identification
        
        :param dataframe: Main dataframe with merged multi-timeframe data
        :param pair: Trading pair name for logging
        """
        try:
            # Check if we have required columns for breakout detection
            required_columns = ['close_5m', 'high_1h_prev']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for breakout detection in {pair}: {missing_columns}")
                return
            
            # Get minimum breakout percentage parameter
            min_breakout_pct = self.min_breakout_pct.value
            
            # Get breakout strength threshold parameter (Requirements: 9.4, 9.5)
            breakout_strength_threshold = self.breakout_strength_threshold.value
            
            # Calculate breakout condition: 5m close > previous 1h high (Requirements: 2.2, 2.3)
            # Add minimum percentage threshold to filter noise
            breakout_threshold = dataframe['high_1h_prev'] * (1 + min_breakout_pct)
            
            # Calculate breakout strength for filtering
            breakout_strength_pct = np.where(
                dataframe['high_1h_prev'] > 0,
                (dataframe['close_5m'] - dataframe['high_1h_prev']) / dataframe['high_1h_prev'],
                0.0
            )
            
            # Detect breakout: 5m close price breaks above previous 1h high with sufficient strength
            dataframe['breakout_condition'] = (
                (dataframe['close_5m'] > breakout_threshold) &
                (breakout_strength_pct >= breakout_strength_threshold) &  # Strength filter
                (dataframe['high_1h_prev'] > 0) &  # Ensure valid 1h high data
                (dataframe['close_5m'] > 0) &      # Ensure valid 5m close data
                (dataframe['data_quality_ok'])     # Only use quality data
            )
            
            # Handle data insufficient boundary cases (Requirements: 2.5)
            # Mark early candles as invalid due to insufficient lookback
            if len(dataframe) > 0:
                # Need at least startup_candle_count for reliable signals
                insufficient_data_mask = dataframe.index < dataframe.index[min(self.startup_candle_count, len(dataframe)-1)]
                dataframe.loc[insufficient_data_mask, 'breakout_condition'] = False
            
            # Additional validation for breakout conditions
            # Ensure we have valid price data for comparison
            invalid_data_mask = (
                (dataframe['close_5m'].isna()) |
                (dataframe['high_1h_prev'].isna()) |
                (dataframe['close_5m'] <= 0) |
                (dataframe['high_1h_prev'] <= 0)
            )
            dataframe.loc[invalid_data_mask, 'breakout_condition'] = False
            
            # Log breakout detection statistics for monitoring
            breakout_count = dataframe['breakout_condition'].sum()
            total_valid_candles = (~invalid_data_mask).sum()
            
            if total_valid_candles > 0:
                breakout_rate = breakout_count / total_valid_candles * 100
                self.logger.debug(f"Breakout detection for {pair}: {breakout_count} breakouts "
                                f"in {total_valid_candles} valid candles ({breakout_rate:.2f}%)")
            
            # Log individual breakout detections (Requirements: 7.5)
            for i in range(len(dataframe)):
                if dataframe.iloc[i]['breakout_condition']:
                    breakout_data = {
                        'close_5m': dataframe.iloc[i]['close_5m'],
                        'high_1h_prev': dataframe.iloc[i]['high_1h_prev'],
                        'breakout_strength': breakout_strength_pct[i] * 100 if i < len(breakout_strength_pct) else 0.0
                    }
                    self.log_breakout_detection(pair, dataframe.index[i], breakout_data)
            
            # Add breakout strength indicator for analysis
            # Calculate percentage above 1h high
            dataframe['breakout_strength'] = np.where(
                dataframe['breakout_condition'] & (dataframe['high_1h_prev'] > 0),
                (dataframe['close_5m'] - dataframe['high_1h_prev']) / dataframe['high_1h_prev'] * 100,
                0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection for {pair}: {str(e)}")
            # Ensure breakout_condition column exists even on error
            if 'breakout_condition' not in dataframe.columns:
                dataframe['breakout_condition'] = False
            if 'breakout_strength' not in dataframe.columns:
                dataframe['breakout_strength'] = 0.0

    def _manage_breakout_state(self, dataframe: DataFrame, pair: str) -> None:
        """
        Manage breakout state persistence until conditions are satisfied or invalidated
        
        Requirements: 2.4, 3.5 - State management and reset mechanisms
        
        :param dataframe: Main dataframe with breakout conditions
        :param pair: Trading pair name for logging
        """
        try:
            # Initialize state management columns if not present
            if 'breakout_state_active' not in dataframe.columns:
                dataframe['breakout_state_active'] = False
            if 'breakout_activation_time' not in dataframe.columns:
                # Use timezone-aware dtype to match FreqTrade's date column
                dataframe['breakout_activation_time'] = pd.Series([pd.NaT] * len(dataframe), dtype='datetime64[ns, UTC]')
            if 'breakout_reference_high' not in dataframe.columns:
                dataframe['breakout_reference_high'] = 0.0
            
            # State management logic
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                # Use 'date' column for timestamp instead of index
                current_time = current_row.get('date', pd.NaT)
                
                # Check if we have a new breakout condition
                if current_row['breakout_condition'] and not current_row.get('breakout_state_active', False):
                    # Activate breakout state (Requirements: 2.4)
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_state_active')] = True
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_activation_time')] = current_time
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_reference_high')] = current_row.get('high_1h_prev', 0.0)
                    
                    self.logger.debug(f"Breakout state activated for {pair} at {current_time}")
                
                # Maintain breakout state from previous candle if still valid
                elif i > 0 and dataframe.iloc[i-1].get('breakout_state_active', False):
                    prev_activation_time = dataframe.iloc[i-1].get('breakout_activation_time')
                    prev_reference_high = dataframe.iloc[i-1].get('breakout_reference_high', 0.0)
                    
                    # Check state invalidation conditions (Requirements: 3.5)
                    state_should_reset = self._should_reset_breakout_state(
                        current_row, prev_activation_time, prev_reference_high, current_time
                    )
                    
                    if state_should_reset:
                        # Reset breakout state
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_state_active')] = False
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_activation_time')] = pd.NaT
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_reference_high')] = 0.0
                        
                        self.logger.debug(f"Breakout state reset for {pair} at {current_time}")
                    else:
                        # Maintain active state
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_state_active')] = True
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_activation_time')] = prev_activation_time
                        dataframe.iloc[i, dataframe.columns.get_loc('breakout_reference_high')] = prev_reference_high
                else:
                    # No active state
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_state_active')] = False
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_activation_time')] = pd.NaT
                    dataframe.iloc[i, dataframe.columns.get_loc('breakout_reference_high')] = 0.0
            
            # Log state management statistics
            active_states = dataframe['breakout_state_active'].sum()
            total_candles = len(dataframe)
            if total_candles > 0:
                active_rate = active_states / total_candles * 100
                self.logger.debug(f"State management for {pair}: {active_states} active states "
                                f"in {total_candles} candles ({active_rate:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Error in breakout state management for {pair}: {str(e)}")
            # Ensure state columns exist even on error
            if 'breakout_state_active' not in dataframe.columns:
                dataframe['breakout_state_active'] = False
            if 'breakout_activation_time' not in dataframe.columns:
                dataframe['breakout_activation_time'] = pd.Series([pd.NaT] * len(dataframe), dtype='datetime64[ns, UTC]')
            if 'breakout_reference_high' not in dataframe.columns:
                dataframe['breakout_reference_high'] = 0.0

    def _should_reset_breakout_state(self, current_row: pd.Series, activation_time: pd.Timestamp, 
                                   reference_high: float, current_time: pd.Timestamp) -> bool:
        """
        Determine if breakout state should be reset based on invalidation conditions
        
        Requirements: 3.5 - State reset mechanism
        
        :param current_row: Current candle data
        :param activation_time: When breakout state was activated
        :param reference_high: Reference 1h high price
        :param current_time: Current candle timestamp
        :return: True if state should be reset
        """
        try:
            # Reset if data quality is poor
            if not current_row.get('data_quality_ok', True):
                return True
            
            # Reset if price falls significantly below reference high (invalidation)
            # Use a tolerance to avoid premature resets due to minor fluctuations
            invalidation_threshold = reference_high * 0.995  # 0.5% below reference high
            current_close = current_row.get('close', 0.0)
            
            if current_close > 0 and current_close < invalidation_threshold:
                return True
            
            # Reset if too much time has passed without entry (timeout mechanism)
            # Maximum time to maintain breakout state: 2 hours (120 minutes)
            if pd.notna(activation_time) and pd.notna(current_time):
                time_elapsed = (current_time - activation_time).total_seconds() / 60  # minutes
                max_state_duration = 120  # 2 hours
                
                if time_elapsed > max_state_duration:
                    return True
            
            # Reset if new 1h high is significantly different (structure change)
            current_1h_high = current_row.get('high_1h_prev', 0.0)
            if current_1h_high > 0 and reference_high > 0:
                high_change_pct = abs(current_1h_high - reference_high) / reference_high
                max_structure_change = 0.02  # 2% change in 1h high structure
                
                if high_change_pct > max_structure_change:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in state reset logic: {str(e)}")
            # Default to reset on error for safety
            return True

    def _detect_pullback_conditions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Detect pullback conditions when 1m price pulls back to MA5 support
        
        Requirements: 3.2, 3.3, 3.4 - Pullback condition detection
        
        :param dataframe: Main dataframe with indicators and breakout state
        :param pair: Trading pair name for logging
        """
        try:
            # Check if we have required columns for pullback detection
            ma_col = f'ma{self.ma_period.value}'
            required_columns = [ma_col, 'close', 'low', 'breakout_state_active']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for pullback detection in {pair}: {missing_columns}")
                return
            
            # Get pullback tolerance parameter
            pullback_tolerance = self.pullback_tolerance.value
            
            # Initialize pullback condition if not present
            if 'pullback_condition' not in dataframe.columns:
                dataframe['pullback_condition'] = False
            
            # Detect pullback conditions only when breakout is active (Requirements: 3.2)
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                
                # Only check pullback when breakout state is active
                if not current_row.get('breakout_state_active', False):
                    dataframe.iloc[i, dataframe.columns.get_loc('pullback_condition')] = False
                    continue
                
                # Skip if data quality is poor
                if not current_row.get('data_quality_ok', True):
                    dataframe.iloc[i, dataframe.columns.get_loc('pullback_condition')] = False
                    continue
                
                # Get current values
                current_close = current_row.get('close', 0.0)
                current_low = current_row.get('low', 0.0)
                current_ma = current_row.get(ma_col, 0.0)
                
                # Validate data - check for NaN, zero, or negative values
                if (pd.isna(current_close) or pd.isna(current_low) or pd.isna(current_ma) or
                    current_close <= 0 or current_low <= 0 or current_ma <= 0):
                    dataframe.iloc[i, dataframe.columns.get_loc('pullback_condition')] = False
                    continue
                
                # Check pullback conditions (Requirements: 3.3, 3.4)
                pullback_detected = False
                
                # Condition 1: 1m close price is equal to or below MA5 (Requirements: 3.3)
                ma_threshold_high = current_ma * (1 + pullback_tolerance)
                
                if current_close <= ma_threshold_high:
                    pullback_detected = True
                    self.logger.debug(f"Pullback detected via close <= MA5 for {pair} at {dataframe.index[i]}: "
                                    f"close={current_close:.6f}, MA5={current_ma:.6f}")
                
                # Condition 2: 1m low price touches or goes below MA5 (Requirements: 3.4)
                # Only check low condition if close condition is not already met
                elif current_low <= ma_threshold_high:
                    pullback_detected = True
                    self.logger.debug(f"Pullback detected via low touching MA5 for {pair} at {dataframe.index[i]}: "
                                    f"low={current_low:.6f}, MA5={current_ma:.6f}")
                
                # Set pullback condition
                dataframe.iloc[i, dataframe.columns.get_loc('pullback_condition')] = pullback_detected
            
            # Add pullback strength indicator for analysis
            dataframe['pullback_strength'] = 0.0
            
            for i in range(len(dataframe)):
                if dataframe.iloc[i]['pullback_condition']:
                    current_close = dataframe.iloc[i].get('close', 0.0)
                    current_ma = dataframe.iloc[i].get(ma_col, 0.0)
                    
                    if current_ma > 0:
                        # Calculate how far below MA5 the price is (negative = below MA)
                        pullback_strength = (current_close - current_ma) / current_ma * 100
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_strength')] = pullback_strength
            
            # Reset pullback condition when breakout state is reset (Requirements: 3.5)
            for i in range(1, len(dataframe)):
                prev_active = dataframe.iloc[i-1].get('breakout_state_active', False)
                current_active = dataframe.iloc[i].get('breakout_state_active', False)
                
                # If breakout state was reset, also reset pullback condition
                if prev_active and not current_active:
                    dataframe.iloc[i, dataframe.columns.get_loc('pullback_condition')] = False
                    self.logger.debug(f"Pullback condition reset due to breakout state reset for {pair} at {dataframe.index[i]}")
            
            # Log pullback detection statistics
            pullback_count = dataframe['pullback_condition'].sum()
            active_breakout_count = dataframe['breakout_state_active'].sum()
            
            if active_breakout_count > 0:
                pullback_rate = pullback_count / active_breakout_count * 100
                self.logger.debug(f"Pullback detection for {pair}: {pullback_count} pullbacks "
                                f"during {active_breakout_count} active breakout periods ({pullback_rate:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error in pullback detection for {pair}: {str(e)}")
            # Ensure pullback_condition column exists even on error
            if 'pullback_condition' not in dataframe.columns:
                dataframe['pullback_condition'] = False
            if 'pullback_strength' not in dataframe.columns:
                dataframe['pullback_strength'] = 0.0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        Integrates all entry condition logic including three-stage entry and risk management
        
        Requirements: 4.4, 7.2 - Integrate entry conditions and implement required methods
        
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry column
        """
        try:
            # Get current pair from metadata
            pair = metadata.get('pair', 'UNKNOWN')
            
            # Initialize logger if not available (for testing)
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                self.logger = logging.getLogger(__name__)
            
            # Ensure all required columns exist (should be set by populate_indicators)
            required_columns = ['entry_signal', 'entry_signal_strength', 'data_quality_ok']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing entry signal columns for {pair}: {missing_columns}")
                # Initialize missing columns with safe defaults
                for col in missing_columns:
                    if col == 'entry_signal':
                        dataframe[col] = False
                    elif col == 'entry_signal_strength':
                        dataframe[col] = 0.0
                    elif col == 'data_quality_ok':
                        dataframe[col] = True
            
            # Initialize entry trend columns
            if 'enter_long' not in dataframe.columns:
                dataframe['enter_long'] = 0
            if 'enter_tag' not in dataframe.columns:
                dataframe['enter_tag'] = ''
            
            # Apply entry conditions with risk management validation (Requirements: 4.4)
            entry_conditions = (
                # Primary condition: entry signal from three-stage logic
                (dataframe['entry_signal'] == True) &
                
                # Data quality check
                (dataframe['data_quality_ok'] == True) &
                
                # Volume check (basic risk management)
                (dataframe['volume'] > 0) &
                
                # Additional risk management: ensure we have valid price data
                (dataframe['close'] > 0) &
                (dataframe['high'] > 0) &
                (dataframe['low'] > 0) &
                (dataframe['open'] > 0) &
                
                # Ensure high >= low (data sanity check)
                (dataframe['high'] >= dataframe['low']) &
                
                # Ensure close is within high-low range (data sanity check)
                (dataframe['close'] >= dataframe['low']) &
                (dataframe['close'] <= dataframe['high'])
            )
            
            # Set entry signals where conditions are met
            dataframe.loc[entry_conditions, 'enter_long'] = 1
            
            # Set entry tags with signal strength information
            dataframe.loc[entry_conditions, 'enter_tag'] = (
                'HourBreakout_' + 
                dataframe.loc[entry_conditions, 'entry_signal_strength'].round(2).astype(str) + '%'
            )
            
            # Ensure non-entry rows have correct values
            dataframe.loc[~entry_conditions, 'enter_long'] = 0
            dataframe.loc[~entry_conditions, 'enter_tag'] = ''
            
            # Log entry signal statistics
            entry_count = (dataframe['enter_long'] == 1).sum()
            total_candles = len(dataframe)
            
            if total_candles > 0:
                entry_rate = entry_count / total_candles * 100
                self.logger.debug(f"Entry trend for {pair}: {entry_count} entries "
                                f"in {total_candles} candles ({entry_rate:.2f}%)")
                
                # Log entry signal strength statistics
                if entry_count > 0:
                    entry_strengths = dataframe.loc[dataframe['enter_long'] == 1, 'entry_signal_strength']
                    avg_strength = entry_strengths.mean()
                    max_strength = entry_strengths.max()
                    min_strength = entry_strengths.min()
                    
                    self.logger.debug(f"Entry signal strengths for {pair}: "
                                    f"avg={avg_strength:.4f}%, max={max_strength:.4f}%, min={min_strength:.4f}%")
            
            # Validate output format (Requirements: 7.2)
            # Ensure enter_long is integer (0 or 1)
            dataframe['enter_long'] = dataframe['enter_long'].astype(int)
            
            # Ensure enter_tag is string
            dataframe['enter_tag'] = dataframe['enter_tag'].astype(str)
            
            # Final validation: check for any invalid entries
            invalid_entries = (
                (dataframe['enter_long'] == 1) & 
                (
                    (dataframe['entry_signal'] != True) |
                    (dataframe['data_quality_ok'] != True) |
                    (dataframe['volume'] <= 0)
                )
            )
            
            if invalid_entries.any():
                invalid_count = invalid_entries.sum()
                self.logger.warning(f"Found {invalid_count} invalid entries for {pair}, clearing them")
                dataframe.loc[invalid_entries, 'enter_long'] = 0
                dataframe.loc[invalid_entries, 'enter_tag'] = ''
            
        except Exception as e:
            self.logger.error(f"Error in populate_entry_trend for {pair}: {str(e)}")
            # Ensure we return a valid dataframe even on error
            if 'enter_long' not in dataframe.columns:
                dataframe['enter_long'] = 0
            if 'enter_tag' not in dataframe.columns:
                dataframe['enter_tag'] = ''
        
        # Validate and ensure FreqTrade compatibility (Requirements: 7.4)
        dataframe = self._ensure_freqtrade_dataframe_compatibility(dataframe, metadata)
        
        return dataframe

    def _detect_rebound_conditions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Detect rebound conditions when 1m price rebounds from MA5 after pullback
        Generate long entry signals when all three conditions are met
        
        Requirements: 4.1, 4.2, 4.3, 4.5 - Rebound detection and entry signal generation
        
        :param dataframe: Main dataframe with indicators and conditions
        :param pair: Trading pair name for logging
        """
        try:
            # Check if we have required columns for rebound detection
            ma_col = f'ma{self.ma_period.value}'
            required_columns = [ma_col, 'close']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for rebound detection in {pair}: {missing_columns}")
                return
            
            # Initialize rebound condition and entry signal columns if not present
            if 'rebound_condition' not in dataframe.columns:
                dataframe['rebound_condition'] = False
            if 'entry_signal' not in dataframe.columns:
                dataframe['entry_signal'] = False
            if 'entry_signal_strength' not in dataframe.columns:
                dataframe['entry_signal_strength'] = 0.0
            if 'last_entry_candle' not in dataframe.columns:
                dataframe['last_entry_candle'] = -1
            
            # Track pullback completion state for each candle
            if 'pullback_completed' not in dataframe.columns:
                dataframe['pullback_completed'] = False
            
            # Initialize other required columns with safe defaults if missing
            if 'breakout_state_active' not in dataframe.columns:
                dataframe['breakout_state_active'] = False
            if 'pullback_condition' not in dataframe.columns:
                dataframe['pullback_condition'] = False
            if 'data_quality_ok' not in dataframe.columns:
                dataframe['data_quality_ok'] = True
            if 'breakout_activation_time' not in dataframe.columns:
                dataframe['breakout_activation_time'] = pd.Series([pd.NaT] * len(dataframe), dtype='datetime64[ns, UTC]')
            if 'breakout_reference_high' not in dataframe.columns:
                dataframe['breakout_reference_high'] = 0.0
            
            # Detect rebound conditions and generate entry signals
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                
                # Skip if data quality is poor
                if not current_row.get('data_quality_ok', True):
                    dataframe.iloc[i, dataframe.columns.get_loc('rebound_condition')] = False
                    dataframe.iloc[i, dataframe.columns.get_loc('entry_signal')] = False
                    continue
                
                # Get current values
                current_close = current_row.get('close', 0.0)
                current_ma = current_row.get(ma_col, 0.0)
                breakout_active = current_row.get('breakout_state_active', False)
                pullback_active = current_row.get('pullback_condition', False)
                
                # Validate data - check for NaN, zero, or negative values
                if (pd.isna(current_close) or pd.isna(current_ma) or
                    current_close <= 0 or current_ma <= 0):
                    dataframe.iloc[i, dataframe.columns.get_loc('rebound_condition')] = False
                    dataframe.iloc[i, dataframe.columns.get_loc('entry_signal')] = False
                    continue
                
                # Update pullback completion state (Requirements: 4.1)
                # Once pullback occurs during active breakout, mark it as completed
                if i > 0:
                    prev_pullback_completed = dataframe.iloc[i-1].get('pullback_completed', False)
                    if breakout_active and (pullback_active or prev_pullback_completed):
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_completed')] = True
                    elif not breakout_active:
                        # Reset pullback completion when breakout state ends
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_completed')] = False
                    else:
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_completed')] = prev_pullback_completed
                else:
                    # For the first candle, set pullback_completed based on current pullback_condition
                    if breakout_active and pullback_active:
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_completed')] = True
                    else:
                        dataframe.iloc[i, dataframe.columns.get_loc('pullback_completed')] = False
                
                # Check rebound condition: price rebounds from MA5 (Requirements: 4.1, 4.2)
                # Rebound = close price is above MA5 after pullback has occurred
                pullback_completed = dataframe.iloc[i].get('pullback_completed', False)
                rebound_detected = False
                
                if breakout_active and pullback_completed and current_close > current_ma:
                    # Check rebound strength threshold (Requirements: 9.4, 9.5)
                    rebound_strength_pct = (current_close - current_ma) / current_ma
                    rebound_strength_threshold = self.rebound_strength_threshold.value
                    
                    if rebound_strength_pct >= rebound_strength_threshold:
                        rebound_detected = True
                        self.logger.debug(f"Rebound detected for {pair} at {dataframe.index[i]}: "
                                        f"close={current_close:.6f} > MA5={current_ma:.6f}, "
                                        f"strength={rebound_strength_pct:.6f} >= threshold={rebound_strength_threshold:.6f}")
                    else:
                        self.logger.debug(f"Rebound strength insufficient for {pair} at {dataframe.index[i]}: "
                                        f"strength={rebound_strength_pct:.6f} < threshold={rebound_strength_threshold:.6f}")
                
                dataframe.iloc[i, dataframe.columns.get_loc('rebound_condition')] = rebound_detected
                
                # Generate entry signal when all three conditions are met (Requirements: 4.3)
                # Three-stage logic: breakout + pullback completed + rebound
                three_stage_complete = (
                    breakout_active and 
                    pullback_completed and 
                    rebound_detected
                )
                
                # Implement duplicate entry prevention (Requirements: 4.5)
                entry_signal_generated = False
                if three_stage_complete:
                    # Check for duplicate entry prevention
                    duplicate_entry = self._check_duplicate_entry_prevention(dataframe, i, pair)
                    
                    if not duplicate_entry:
                        entry_signal_generated = True
                        # Mark this candle as the last entry candle for duplicate prevention
                        dataframe.iloc[i, dataframe.columns.get_loc('last_entry_candle')] = i
                        
                        # Calculate entry signal strength
                        rebound_strength = (current_close - current_ma) / current_ma * 100
                        dataframe.iloc[i, dataframe.columns.get_loc('entry_signal_strength')] = float(rebound_strength)
                        
                        # Log entry signal with structured logging (Requirements: 7.5)
                        entry_conditions = {
                            'breakout_active': breakout_active,
                            'pullback_completed': pullback_completed,
                            'rebound_detected': rebound_detected
                        }
                        self.log_entry_signal(pair, dataframe.index[i], rebound_strength, entry_conditions)
                        
                        self.logger.info(f"Entry signal generated for {pair} at {dataframe.index[i]}: "
                                       f"breakout={breakout_active}, pullback_completed={pullback_completed}, "
                                       f"rebound={rebound_detected}, strength={rebound_strength:.4f}%")
                    else:
                        self.logger.debug(f"Entry signal blocked due to duplicate prevention for {pair} at {dataframe.index[i]}")
                
                dataframe.iloc[i, dataframe.columns.get_loc('entry_signal')] = entry_signal_generated
                
                # Set entry signal strength to 0 if no signal
                if not entry_signal_generated:
                    dataframe.iloc[i, dataframe.columns.get_loc('entry_signal_strength')] = 0.0
            
            # Log entry signal statistics
            entry_count = dataframe['entry_signal'].sum()
            rebound_count = dataframe['rebound_condition'].sum()
            active_breakout_count = dataframe['breakout_state_active'].sum()
            
            if active_breakout_count > 0:
                entry_rate = entry_count / active_breakout_count * 100
                rebound_rate = rebound_count / active_breakout_count * 100
                self.logger.debug(f"Entry signal generation for {pair}: {entry_count} entries, "
                                f"{rebound_count} rebounds during {active_breakout_count} active breakout periods "
                                f"(entry rate: {entry_rate:.2f}%, rebound rate: {rebound_rate:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error in rebound detection for {pair}: {str(e)}")
            # Ensure required columns exist even on error
            if 'rebound_condition' not in dataframe.columns:
                dataframe['rebound_condition'] = False
            if 'entry_signal' not in dataframe.columns:
                dataframe['entry_signal'] = False
            if 'entry_signal_strength' not in dataframe.columns:
                dataframe['entry_signal_strength'] = 0.0
            if 'pullback_completed' not in dataframe.columns:
                dataframe['pullback_completed'] = False
            if 'last_entry_candle' not in dataframe.columns:
                dataframe['last_entry_candle'] = -1

    def _check_duplicate_entry_prevention(self, dataframe: DataFrame, current_index: int, pair: str) -> bool:
        """
        Check if entry signal should be blocked due to duplicate entry prevention
        
        Requirements: 4.5 - Prevent duplicate entries for same breakout pattern
        
        :param dataframe: Main dataframe with entry signals
        :param current_index: Current candle index
        :param pair: Trading pair name for logging
        :return: True if entry should be blocked (duplicate), False if entry is allowed
        """
        try:
            # Get current breakout reference information
            current_row = dataframe.iloc[current_index]
            current_breakout_active = current_row.get('breakout_state_active', False)
            current_reference_high = current_row.get('breakout_reference_high', 0.0)
            current_activation_time = current_row.get('breakout_activation_time')
            
            if not current_breakout_active or current_reference_high <= 0:
                return False  # No active breakout, allow entry
            
            # Get minimum spacing parameter from HyperOpt (Requirements: 9.4, 9.5)
            min_candles_between_entries = self.min_entry_spacing.value
            
            # Check if there's been a recent entry regardless of breakout pattern (minimum spacing)
            # Look back for recent entries within the minimum spacing window (exclusive of boundary)
            for i in range(max(0, current_index - min_candles_between_entries + 1), current_index):
                if dataframe.iloc[i].get('entry_signal', False):
                    spacing = current_index - i
                    self.logger.debug(f"Entry prevented due to recent entry for {pair}: "
                                    f"previous entry at index {i}, current at {current_index}, "
                                    f"spacing: {spacing} candles, minimum: {min_candles_between_entries}")
                    return True  # Block entry due to recent entry
            
            # Look back for entries with the same breakout pattern (longer lookback)
            lookback_period = 60  # Look back 60 candles (1 hour)
            start_index = max(0, current_index - lookback_period)
            
            for i in range(start_index, current_index):
                prev_row = dataframe.iloc[i]
                
                # Check if there was a previous entry signal
                if prev_row.get('entry_signal', False):
                    prev_reference_high = prev_row.get('breakout_reference_high', 0.0)
                    prev_activation_time = prev_row.get('breakout_activation_time')
                    
                    # Check if it's the same breakout pattern
                    same_breakout_pattern = (
                        abs(prev_reference_high - current_reference_high) < 0.0001 and  # Same reference high
                        prev_activation_time == current_activation_time  # Same activation time
                    )
                    
                    if same_breakout_pattern:
                        # Check if enough time has passed for the same pattern
                        # Allow same pattern after longer period (30 candles = 30 minutes)
                        time_since_entry = current_index - i
                        min_same_pattern_spacing = 30  # 30 candles for same pattern
                        
                        if time_since_entry < min_same_pattern_spacing:
                            self.logger.debug(f"Duplicate entry prevented for {pair}: "
                                            f"previous entry at index {i}, current at {current_index}, "
                                            f"same reference high {current_reference_high:.6f}, "
                                            f"time since entry: {time_since_entry} candles")
                            return True  # Block duplicate entry
            
            return False  # Allow entry
            
        except Exception as e:
            self.logger.error(f"Error in duplicate entry prevention for {pair}: {str(e)}")
            # Default to allowing entry on error (conservative approach)
            return False

    def _calculate_stop_loss_conditions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Calculate stop loss conditions based on previous 1h high
        
        Requirements: 5.1, 5.2, 5.3 - Stop loss mechanism implementation
        
        :param dataframe: Main dataframe with price data and indicators
        :param pair: Trading pair name for logging
        """
        try:
            # Initialize stop loss columns if not present
            if 'stop_loss_condition' not in dataframe.columns:
                dataframe['stop_loss_condition'] = False
            if 'stop_loss_price' not in dataframe.columns:
                dataframe['stop_loss_price'] = 0.0
            if 'dynamic_stop_loss' not in dataframe.columns:
                dataframe['dynamic_stop_loss'] = 0.0
            
            # Check if we have required columns for stop loss calculation
            required_columns = ['close', 'high_1h_prev']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for stop loss calculation in {pair}: {missing_columns}")
                return
            
            # Calculate stop loss conditions for each candle
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                
                # Skip if data quality is poor
                if not current_row.get('data_quality_ok', True):
                    continue
                
                # Get current values
                current_close = current_row.get('close', 0.0)
                current_1h_high_prev = current_row.get('high_1h_prev', 0.0)
                
                # Validate data - check for NaN, zero, or negative values
                if (pd.isna(current_close) or pd.isna(current_1h_high_prev) or
                    current_close <= 0 or current_1h_high_prev <= 0):
                    continue
                
                # Set stop loss price based on previous 1h high (Requirements: 5.1)
                # Stop loss is set at the previous 1h high level with buffer
                stop_loss_buffer = self.stop_loss_buffer_pct.value
                stop_loss_price = float(current_1h_high_prev * (1 - stop_loss_buffer))
                dataframe.iloc[i, dataframe.columns.get_loc('stop_loss_price')] = stop_loss_price
                
                # Implement dynamic stop loss update mechanism (Requirements: 5.3)
                # Update stop loss when new 1h high is formed and is higher than current stop loss
                if i > 0:
                    prev_stop_loss = float(dataframe.iloc[i-1].get('dynamic_stop_loss', 0.0))
                    # Use the higher of current 1h high or previous dynamic stop loss (trailing up only)
                    dynamic_stop_loss = float(max(stop_loss_price, prev_stop_loss) if prev_stop_loss > 0 else stop_loss_price)
                else:
                    dynamic_stop_loss = stop_loss_price
                
                dataframe.iloc[i, dataframe.columns.get_loc('dynamic_stop_loss')] = dynamic_stop_loss
                
                # Detect price breaking below stop loss level (Requirements: 5.2)
                # Use dynamic stop loss for the condition check
                stop_loss_triggered = current_close < dynamic_stop_loss
                dataframe.iloc[i, dataframe.columns.get_loc('stop_loss_condition')] = stop_loss_triggered
                
                # Log stop loss triggers for analysis
                if stop_loss_triggered:
                    self.logger.debug(f"Stop loss triggered for {pair} at {dataframe.index[i]}: "
                                    f"close={current_close:.6f} < stop_loss={dynamic_stop_loss:.6f}")
            
            # Add stop loss strength indicator for analysis
            dataframe['stop_loss_strength'] = 0.0
            
            for i in range(len(dataframe)):
                if dataframe.iloc[i]['stop_loss_condition']:
                    current_close = dataframe.iloc[i].get('close', 0.0)
                    dynamic_stop_loss = dataframe.iloc[i].get('dynamic_stop_loss', 0.0)
                    
                    if dynamic_stop_loss > 0:
                        # Calculate how far below stop loss the price is (negative = below stop loss)
                        stop_loss_strength = (current_close - dynamic_stop_loss) / dynamic_stop_loss * 100
                        dataframe.iloc[i, dataframe.columns.get_loc('stop_loss_strength')] = stop_loss_strength
            
            # Log stop loss statistics
            stop_loss_count = dataframe['stop_loss_condition'].sum()
            total_candles = len(dataframe)
            
            if total_candles > 0:
                stop_loss_rate = stop_loss_count / total_candles * 100
                self.logger.debug(f"Stop loss calculation for {pair}: {stop_loss_count} triggers "
                                f"in {total_candles} candles ({stop_loss_rate:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error in stop loss calculation for {pair}: {str(e)}")
            # Ensure stop loss columns exist even on error
            if 'stop_loss_condition' not in dataframe.columns:
                dataframe['stop_loss_condition'] = False
            if 'stop_loss_price' not in dataframe.columns:
                dataframe['stop_loss_price'] = 0.0
            if 'dynamic_stop_loss' not in dataframe.columns:
                dataframe['dynamic_stop_loss'] = 0.0
            if 'stop_loss_strength' not in dataframe.columns:
                dataframe['stop_loss_strength'] = 0.0

    def _calculate_time_based_exit_conditions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Calculate time-based exit conditions for take profit mechanism
        
        Requirements: 6.1, 6.2, 6.4 - Time-based take profit mechanism
        
        :param dataframe: Main dataframe with price data and indicators
        :param pair: Trading pair name for logging
        """
        try:
            # Initialize time-based exit columns if not present
            if 'time_exit_condition' not in dataframe.columns:
                dataframe['time_exit_condition'] = False
            if 'position_duration_minutes' not in dataframe.columns:
                dataframe['position_duration_minutes'] = 0.0
            if 'time_exit_target_minutes' not in dataframe.columns:
                dataframe['time_exit_target_minutes'] = 0.0
            if 'max_position_exit_condition' not in dataframe.columns:
                dataframe['max_position_exit_condition'] = False
            
            # Get time-based exit parameters (Requirements: 6.1, 9.4, 9.5)
            exit_minutes = self.exit_minutes.value
            max_position_hours = self.max_position_hours.value
            max_position_minutes = max_position_hours * 60
            
            # Track position duration and trigger time-based exits
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                
                # Skip if data quality is poor
                if not current_row.get('data_quality_ok', True):
                    continue
                
                # Set target exit time for this candle
                dataframe.iloc[i, dataframe.columns.get_loc('time_exit_target_minutes')] = exit_minutes
                
                # Check if we have an entry signal at this position
                entry_signal = current_row.get('entry_signal', False)
                
                if entry_signal:
                    # Reset position duration at entry
                    dataframe.iloc[i, dataframe.columns.get_loc('position_duration_minutes')] = 0.0
                    dataframe.iloc[i, dataframe.columns.get_loc('time_exit_condition')] = False
                else:
                    # Calculate position duration from previous candle
                    if i > 0:
                        prev_duration = dataframe.iloc[i-1].get('position_duration_minutes', 0.0)
                        prev_entry_signal = dataframe.iloc[i-1].get('entry_signal', False)
                        
                        if prev_entry_signal or prev_duration > 0:
                            # Increment duration (assuming 1-minute candles)
                            current_duration = prev_duration + 1.0
                            dataframe.iloc[i, dataframe.columns.get_loc('position_duration_minutes')] = current_duration
                            
                            # Check if time-based exit should be triggered (Requirements: 6.2)
                            time_exit_triggered = current_duration >= exit_minutes
                            dataframe.iloc[i, dataframe.columns.get_loc('time_exit_condition')] = time_exit_triggered
                            
                            # Check if maximum position duration is exceeded (Requirements: 9.4, 9.5)
                            max_position_exit_triggered = current_duration >= max_position_minutes
                            dataframe.iloc[i, dataframe.columns.get_loc('max_position_exit_condition')] = max_position_exit_triggered
                            
                            # Log time-based exits for analysis
                            if time_exit_triggered:
                                self.logger.debug(f"Time-based exit triggered for {pair} at {dataframe.index[i]}: "
                                                f"duration={current_duration:.1f} minutes >= target={exit_minutes} minutes")
                            
                            if max_position_exit_triggered:
                                self.logger.debug(f"Max position exit triggered for {pair} at {dataframe.index[i]}: "
                                                f"duration={current_duration:.1f} minutes >= max={max_position_minutes:.1f} minutes")
                        else:
                            # No active position
                            dataframe.iloc[i, dataframe.columns.get_loc('position_duration_minutes')] = 0.0
                            dataframe.iloc[i, dataframe.columns.get_loc('time_exit_condition')] = False
                            dataframe.iloc[i, dataframe.columns.get_loc('max_position_exit_condition')] = False
                    else:
                        # First candle
                        dataframe.iloc[i, dataframe.columns.get_loc('position_duration_minutes')] = 0.0
                        dataframe.iloc[i, dataframe.columns.get_loc('time_exit_condition')] = False
                        dataframe.iloc[i, dataframe.columns.get_loc('max_position_exit_condition')] = False
            
            # Handle timezone and market time considerations (Requirements: 6.4)
            # Note: FreqTrade handles timezone conversion automatically, so we work with the provided timestamps
            # The duration calculation above assumes 1-minute candles which matches our timeframe
            
            # Log time-based exit statistics
            time_exit_count = dataframe['time_exit_condition'].sum()
            total_candles = len(dataframe)
            
            if total_candles > 0:
                time_exit_rate = time_exit_count / total_candles * 100
                self.logger.debug(f"Time-based exit calculation for {pair}: {time_exit_count} triggers "
                                f"in {total_candles} candles ({time_exit_rate:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error in time-based exit calculation for {pair}: {str(e)}")
            # Ensure time exit columns exist even on error
            if 'time_exit_condition' not in dataframe.columns:
                dataframe['time_exit_condition'] = False
            if 'position_duration_minutes' not in dataframe.columns:
                dataframe['position_duration_minutes'] = 0.0
            if 'time_exit_target_minutes' not in dataframe.columns:
                dataframe['time_exit_target_minutes'] = 0.0
            if 'max_position_exit_condition' not in dataframe.columns:
                dataframe['max_position_exit_condition'] = False

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        Integrates stop loss and time-based take profit logic
        
        Requirements: 5.5, 7.2 - Integrate exit conditions and implement required methods
        
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit column
        """
        try:
            # Get current pair from metadata
            pair = metadata['pair']
            
            # Initialize logger if not available (for testing)
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                self.logger = logging.getLogger(__name__)
            
            # Initialize exit columns
            if 'exit_long' not in dataframe.columns:
                dataframe['exit_long'] = 0
            if 'exit_tag' not in dataframe.columns:
                dataframe['exit_tag'] = ''
            
            # Calculate stop loss conditions (Requirements: 5.1, 5.2, 5.3)
            self._calculate_stop_loss_conditions(dataframe, pair)
            
            # Calculate time-based exit conditions (Requirements: 6.1, 6.2, 6.4)
            self._calculate_time_based_exit_conditions(dataframe, pair)
            
            # Integrate stop loss and time-based exit logic (Requirements: 5.5)
            for i in range(len(dataframe)):
                current_row = dataframe.iloc[i]
                
                # Check stop loss condition
                stop_loss_triggered = current_row.get('stop_loss_condition', False)
                
                # Check time-based exit condition
                time_exit_triggered = current_row.get('time_exit_condition', False)
                
                # Check maximum position exit condition (Requirements: 9.4, 9.5)
                max_position_exit_triggered = current_row.get('max_position_exit_condition', False)
                
                # Determine exit signal and tag (priority order: stop loss > max position > time exit)
                exit_signal = 0
                exit_tag = ''
                
                if stop_loss_triggered:
                    exit_signal = 1
                    exit_tag = 'stop_loss'
                    
                    # Log stop loss exit with structured logging (Requirements: 7.5)
                    exit_data = {
                        'stop_loss_price': current_row.get('dynamic_stop_loss', 0.0),
                        'current_price': current_row.get('close', 0.0)
                    }
                    self.log_exit_signal(pair, dataframe.index[i], exit_tag, exit_data)
                    
                    # Log stop loss exit (Requirements: 7.2)
                    self.logger.debug(f"Stop loss exit signal generated for {pair} at {dataframe.index[i]}")
                elif max_position_exit_triggered:
                    exit_signal = 1
                    exit_tag = 'max_position'
                    
                    # Log max position exit with structured logging (Requirements: 7.5)
                    exit_data = {
                        'position_duration': current_row.get('position_duration_minutes', 0.0),
                        'max_duration': self.max_position_hours.value * 60
                    }
                    self.log_exit_signal(pair, dataframe.index[i], exit_tag, exit_data)
                    
                    # Log maximum position exit (Requirements: 7.2)
                    self.logger.debug(f"Maximum position exit signal generated for {pair} at {dataframe.index[i]}")
                elif time_exit_triggered:
                    exit_signal = 1
                    exit_tag = 'time_exit'
                    
                    # Log time exit with structured logging (Requirements: 7.5)
                    exit_data = {
                        'position_duration': current_row.get('position_duration_minutes', 0.0),
                        'target_minutes': current_row.get('time_exit_target_minutes', 0.0)
                    }
                    self.log_exit_signal(pair, dataframe.index[i], exit_tag, exit_data)
                    
                    # Log time-based exit (Requirements: 7.2)
                    self.logger.debug(f"Time-based exit signal generated for {pair} at {dataframe.index[i]}")
                
                # Set exit signal and tag
                dataframe.iloc[i, dataframe.columns.get_loc('exit_long')] = exit_signal
                dataframe.iloc[i, dataframe.columns.get_loc('exit_tag')] = exit_tag
            
            # Add exit logging for analysis (Requirements: 7.2)
            exit_count = (dataframe['exit_long'] == 1).sum()
            stop_loss_exits = (dataframe['exit_tag'] == 'stop_loss').sum()
            time_exits = (dataframe['exit_tag'] == 'time_exit').sum()
            max_position_exits = (dataframe['exit_tag'] == 'max_position').sum()
            
            if exit_count > 0:
                self.logger.debug(f"Exit signals generated for {pair}: {exit_count} total "
                                f"({stop_loss_exits} stop loss, {time_exits} time-based, {max_position_exits} max position)")
            
        except Exception as e:
            self.logger.error(f"Error in populate_exit_trend for {pair}: {str(e)}")
            # Ensure exit columns exist even on error
            if 'exit_long' not in dataframe.columns:
                dataframe['exit_long'] = 0
            if 'exit_tag' not in dataframe.columns:
                dataframe['exit_tag'] = ''
        
        # Validate and ensure FreqTrade compatibility (Requirements: 7.4)
        dataframe = self._ensure_freqtrade_dataframe_compatibility(dataframe, metadata)
        
        return dataframe

    def _ensure_freqtrade_dataframe_compatibility(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Ensure dataframe output format meets FreqTrade requirements
        Validates column naming conventions and data types
        
        Requirements: 7.4 - FreqTrade dataframe structure and column naming conventions
        
        :param dataframe: DataFrame to validate and fix
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with validated format
        """
        try:
            # Get current pair from metadata
            pair = metadata.get('pair', 'UNKNOWN')
            
            # Initialize logger if not available (for testing)
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                self.logger = logging.getLogger(__name__)
            
            # Define required FreqTrade columns with expected data types
            required_columns = {
                # Basic OHLCV columns (should already exist)
                'open': 'float64',
                'high': 'float64', 
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64',
                
                # Entry signal columns (FreqTrade requirements)
                'enter_long': 'int64',
                'enter_tag': 'object',  # string type
                
                # Exit signal columns (FreqTrade requirements)
                'exit_long': 'int64',
                'exit_tag': 'object',  # string type
            }
            
            # Optional columns that should exist if present (with expected types)
            optional_columns = {
                # Multi-timeframe data columns
                'close_5m': 'float64',
                'high_1h': 'float64',
                'high_1h_prev': 'float64',
                
                # Technical indicator columns
                f'ma{self.ma_period.value}': 'float64',
                
                # Strategy condition columns
                'breakout_condition': 'bool',
                'pullback_condition': 'bool',
                'rebound_condition': 'bool',
                'entry_signal': 'bool',
                'breakout_state_active': 'bool',
                'pullback_completed': 'bool',
                'data_quality_ok': 'bool',
                
                # Numeric indicator columns
                'entry_signal_strength': 'float64',
                'breakout_strength': 'float64',
                'pullback_strength': 'float64',
                'stop_loss_price': 'float64',
                'dynamic_stop_loss': 'float64',
                'stop_loss_strength': 'float64',
                'position_duration_minutes': 'float64',
                'time_exit_target_minutes': 'float64',
                'breakout_reference_high': 'float64',
                'last_entry_candle': 'int64',
                
                # Condition columns
                'stop_loss_condition': 'bool',
                'time_exit_condition': 'bool',
                'max_position_exit_condition': 'bool',
            }
            
            # Ensure all required columns exist with correct data types
            missing_required = []
            for col_name, expected_dtype in required_columns.items():
                if col_name not in dataframe.columns:
                    missing_required.append(col_name)
                    # Add missing column with appropriate default value
                    if expected_dtype == 'int64':
                        dataframe[col_name] = 0
                    elif expected_dtype == 'float64':
                        dataframe[col_name] = 0.0
                    elif expected_dtype == 'bool':
                        dataframe[col_name] = False
                    else:  # object/string
                        dataframe[col_name] = ''
                    
                    self.logger.warning(f"Added missing required column '{col_name}' for {pair}")
                
                # Validate and fix data type
                try:
                    if dataframe[col_name].dtype != expected_dtype:
                        if expected_dtype == 'int64':
                            # Convert to int, handling NaN values
                            dataframe[col_name] = pd.to_numeric(dataframe[col_name], errors='coerce').fillna(0).astype('int64')
                        elif expected_dtype == 'float64':
                            # Convert to float, ensuring proper handling of all numeric types
                            dataframe[col_name] = pd.to_numeric(dataframe[col_name], errors='coerce')
                            # Fill NaN values appropriately
                            if col_name == 'volume':
                                dataframe[col_name] = dataframe[col_name].fillna(0.0)
                            else:
                                # For price columns, use forward/backward fill
                                dataframe[col_name] = dataframe[col_name].ffill().bfill().fillna(0.0)
                            # Ensure float64 type
                            dataframe[col_name] = dataframe[col_name].astype('float64')
                        elif expected_dtype == 'bool':
                            # Convert to boolean
                            dataframe[col_name] = dataframe[col_name].fillna(False).astype('bool')
                        else:  # object/string
                            # Convert to string
                            dataframe[col_name] = dataframe[col_name].astype('str').fillna('')
                        
                        self.logger.debug(f"Fixed data type for column '{col_name}' to {expected_dtype} for {pair}")
                        
                except Exception as dtype_error:
                    self.logger.error(f"Failed to fix data type for column '{col_name}' in {pair}: {str(dtype_error)}")
                    # Use safe default values on conversion error
                    if expected_dtype == 'int64':
                        dataframe[col_name] = 0
                    elif expected_dtype == 'float64':
                        dataframe[col_name] = 0.0
                    elif expected_dtype == 'bool':
                        dataframe[col_name] = False
                    else:
                        dataframe[col_name] = ''
            
            # Validate and fix optional columns if they exist
            for col_name, expected_dtype in optional_columns.items():
                if col_name in dataframe.columns:
                    try:
                        if dataframe[col_name].dtype != expected_dtype:
                            if expected_dtype == 'int64':
                                dataframe[col_name] = dataframe[col_name].fillna(0).astype('int64')
                            elif expected_dtype == 'float64':
                                dataframe[col_name] = pd.to_numeric(dataframe[col_name], errors='coerce').fillna(0.0)
                            elif expected_dtype == 'bool':
                                dataframe[col_name] = dataframe[col_name].astype('bool')
                            else:  # object/string
                                dataframe[col_name] = dataframe[col_name].astype('str').fillna('')
                            
                            self.logger.debug(f"Fixed optional column '{col_name}' data type to {expected_dtype} for {pair}")
                            
                    except Exception as dtype_error:
                        self.logger.warning(f"Failed to fix optional column '{col_name}' data type in {pair}: {str(dtype_error)}")
            
            # Validate FreqTrade column naming conventions
            self._validate_column_naming_conventions(dataframe, pair)
            
            # Validate data integrity and ranges
            self._validate_dataframe_data_integrity(dataframe, pair)
            
            # Log validation summary
            if missing_required:
                self.logger.info(f"DataFrame compatibility validation for {pair}: added {len(missing_required)} missing columns")
            else:
                self.logger.debug(f"DataFrame compatibility validation passed for {pair}")
            
        except Exception as e:
            self.logger.error(f"Error in dataframe compatibility validation for {pair}: {str(e)}")
            # Ensure basic required columns exist even on error
            for col_name, expected_dtype in required_columns.items():
                if col_name not in dataframe.columns:
                    if expected_dtype == 'int64':
                        dataframe[col_name] = 0
                    elif expected_dtype == 'float64':
                        dataframe[col_name] = 0.0
                    elif expected_dtype == 'bool':
                        dataframe[col_name] = False
                    else:
                        dataframe[col_name] = ''
        
        return dataframe

    def _validate_column_naming_conventions(self, dataframe: DataFrame, pair: str) -> None:
        """
        Validate FreqTrade column naming conventions
        
        Requirements: 7.4 - Column naming conventions validation
        
        :param dataframe: DataFrame to validate
        :param pair: Trading pair name for logging
        """
        try:
            # FreqTrade column naming rules and conventions
            naming_issues = []
            
            # Check for required FreqTrade signal columns
            required_signal_columns = ['enter_long', 'exit_long', 'enter_tag', 'exit_tag']
            for col in required_signal_columns:
                if col not in dataframe.columns:
                    naming_issues.append(f"Missing required signal column: {col}")
            
            # Check for deprecated column names (FreqTrade v3+ compatibility)
            deprecated_mappings = {
                'buy': 'enter_long',
                'sell': 'exit_long', 
                'buy_tag': 'enter_tag',
                'sell_tag': 'exit_tag'
            }
            
            for old_name, new_name in deprecated_mappings.items():
                if old_name in dataframe.columns:
                    naming_issues.append(f"Deprecated column name '{old_name}' found, should be '{new_name}'")
            
            # Check for invalid column names (containing spaces, special characters)
            invalid_chars = [' ', '-', '+', '*', '/', '\\', '(', ')', '[', ']', '{', '}']
            for col in dataframe.columns:
                if any(char in str(col) for char in invalid_chars):
                    naming_issues.append(f"Invalid column name '{col}' contains special characters")
            
            # Check for excessively long column names (FreqTrade limitation)
            max_col_length = 64  # Reasonable limit for database compatibility
            for col in dataframe.columns:
                if len(str(col)) > max_col_length:
                    naming_issues.append(f"Column name '{col}' is too long ({len(str(col))} > {max_col_length} chars)")
            
            # Log naming issues
            if naming_issues:
                for issue in naming_issues:
                    self.logger.warning(f"Column naming issue for {pair}: {issue}")
            else:
                self.logger.debug(f"Column naming conventions validated for {pair}")
                
        except Exception as e:
            self.logger.error(f"Error in column naming validation for {pair}: {str(e)}")

    def _validate_dataframe_data_integrity(self, dataframe: DataFrame, pair: str) -> None:
        """
        Validate dataframe data integrity and value ranges
        
        Requirements: 7.4 - Data type validation and integrity checks
        
        :param dataframe: DataFrame to validate
        :param pair: Trading pair name for logging
        """
        try:
            integrity_issues = []
            
            # Validate OHLCV data integrity
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if col in dataframe.columns:
                    # Check for negative values (invalid for prices/volume)
                    negative_count = (dataframe[col] < 0).sum()
                    if negative_count > 0:
                        integrity_issues.append(f"Column '{col}' has {negative_count} negative values")
                    
                    # Check for NaN values
                    nan_count = dataframe[col].isna().sum()
                    if nan_count > 0:
                        integrity_issues.append(f"Column '{col}' has {nan_count} NaN values")
                    
                    # Check for infinite values
                    inf_count = np.isinf(dataframe[col]).sum()
                    if inf_count > 0:
                        integrity_issues.append(f"Column '{col}' has {inf_count} infinite values")
            
            # Validate OHLC relationships (High >= Low, Close within High-Low range)
            if all(col in dataframe.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Low
                invalid_hl = (dataframe['high'] < dataframe['low']).sum()
                if invalid_hl > 0:
                    integrity_issues.append(f"{invalid_hl} candles have high < low")
                
                # Close should be within high-low range
                invalid_close_high = (dataframe['close'] > dataframe['high']).sum()
                invalid_close_low = (dataframe['close'] < dataframe['low']).sum()
                if invalid_close_high > 0:
                    integrity_issues.append(f"{invalid_close_high} candles have close > high")
                if invalid_close_low > 0:
                    integrity_issues.append(f"{invalid_close_low} candles have close < low")
            
            # Validate signal columns (should be 0 or 1 for entry/exit signals)
            signal_columns = ['enter_long', 'exit_long']
            for col in signal_columns:
                if col in dataframe.columns:
                    invalid_signals = (~dataframe[col].isin([0, 1])).sum()
                    if invalid_signals > 0:
                        integrity_issues.append(f"Column '{col}' has {invalid_signals} invalid signal values (not 0 or 1)")
            
            # Validate boolean columns
            boolean_columns = ['breakout_condition', 'pullback_condition', 'rebound_condition', 
                             'entry_signal', 'breakout_state_active', 'data_quality_ok']
            for col in boolean_columns:
                if col in dataframe.columns:
                    if dataframe[col].dtype != 'bool':
                        integrity_issues.append(f"Column '{col}' should be boolean type, found {dataframe[col].dtype}")
            
            # Validate percentage/ratio columns (should be reasonable ranges)
            percentage_columns = ['entry_signal_strength', 'breakout_strength', 'pullback_strength', 'stop_loss_strength']
            for col in percentage_columns:
                if col in dataframe.columns:
                    # Check for extremely large values (> 1000% change is suspicious)
                    extreme_values = (abs(dataframe[col]) > 1000).sum()
                    if extreme_values > 0:
                        integrity_issues.append(f"Column '{col}' has {extreme_values} extreme values (>1000%)")
            
            # Validate index integrity
            # Note: FreqTrade uses integer index, datetime is in 'date' column
            if 'date' in dataframe.columns:
                # Check for duplicate timestamps in date column
                duplicate_timestamps = dataframe['date'].duplicated().sum()
                if duplicate_timestamps > 0:
                    integrity_issues.append(f"DataFrame has {duplicate_timestamps} duplicate timestamps")
                
                # Check for non-monotonic timestamps (should be sorted)
                if not dataframe['date'].is_monotonic_increasing:
                    integrity_issues.append("DataFrame timestamps are not in ascending order")
            
            # Log integrity issues
            if integrity_issues:
                for issue in integrity_issues:
                    self.logger.warning(f"Data integrity issue for {pair}: {issue}")
            else:
                self.logger.debug(f"Data integrity validation passed for {pair}")
                
        except Exception as e:
            self.logger.error(f"Error in data integrity validation for {pair}: {str(e)}")

    def _setup_logging_system(self) -> None:
        """
        Setup FreqTrade logging system integration
        
        Requirements: 7.5 - Use FreqTrade logging system
        """
        try:
            # Import FreqTrade logging if available
            try:
                from freqtrade.configuration import setup_utils_configuration
                from freqtrade.loggers import setup_logging
                # FreqTrade logging is available
                self._freqtrade_logging_available = True
            except ImportError:
                # Fallback to standard Python logging
                self._freqtrade_logging_available = False
            
            # Initialize logger if not already set
            if not hasattr(self, 'logger') or self.logger is None:
                import logging
                
                # Create strategy-specific logger
                logger_name = f"freqtrade.strategy.{self.__class__.__name__}"
                self.logger = logging.getLogger(logger_name)
                
                # Set appropriate log level
                if not self.logger.handlers:
                    # Only add handler if none exists (avoid duplicate handlers)
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
                    self.logger.setLevel(logging.INFO)
            
            # Log system initialization
            self.logger.info(f"Logging system initialized for {self.__class__.__name__}")
            self.logger.debug(f"FreqTrade logging available: {self._freqtrade_logging_available}")
            
        except Exception as e:
            # Fallback to print if logging setup fails
            print(f"Error setting up logging system: {str(e)}")
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)

    def log_strategy_event(self, event_type: str, pair: str, message: str, 
                          level: str = 'info', extra_data: dict = None) -> None:
        """
        Log strategy events with structured format
        
        Requirements: 7.5 - Add key event logging
        
        :param event_type: Type of event (entry, exit, breakout, etc.)
        :param pair: Trading pair
        :param message: Log message
        :param level: Log level (debug, info, warning, error)
        :param extra_data: Additional data to include in log
        """
        try:
            # Ensure logger is available
            if not hasattr(self, 'logger') or self.logger is None:
                self._setup_logging_system()
            
            # Format structured log message
            structured_message = f"[{event_type.upper()}] {pair}: {message}"
            
            # Add extra data if provided
            if extra_data:
                extra_str = ", ".join([f"{k}={v}" for k, v in extra_data.items()])
                structured_message += f" | {extra_str}"
            
            # Log at appropriate level
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(structured_message)
            
        except Exception as e:
            # Fallback logging
            print(f"Error in structured logging: {str(e)} | Original message: {message}")

    def log_entry_signal(self, pair: str, candle_time: pd.Timestamp, entry_strength: float, 
                        conditions: dict) -> None:
        """
        Log entry signal generation with detailed information
        
        Requirements: 7.5 - Key event logging for entry signals
        
        :param pair: Trading pair
        :param candle_time: Timestamp of the candle
        :param entry_strength: Signal strength percentage
        :param conditions: Dictionary of conditions that triggered entry
        """
        try:
            # Handle different timestamp types
            if hasattr(candle_time, 'strftime'):
                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(candle_time, (int, float)):
                timestamp_str = str(candle_time)
            else:
                timestamp_str = str(candle_time)
            
            extra_data = {
                'timestamp': timestamp_str,
                'strength': f"{entry_strength:.4f}%",
                'breakout': conditions.get('breakout_active', False),
                'pullback_completed': conditions.get('pullback_completed', False),
                'rebound': conditions.get('rebound_detected', False)
            }
            
            message = f"Entry signal generated with strength {entry_strength:.4f}%"
            self.log_strategy_event('ENTRY', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging entry signal for {pair}: {str(e)}")

    def log_exit_signal(self, pair: str, candle_time, exit_reason: str, 
                       exit_data: dict) -> None:
        """
        Log exit signal generation with detailed information
        
        Requirements: 7.5 - Key event logging for exit signals
        
        :param pair: Trading pair
        :param candle_time: Timestamp of the candle
        :param exit_reason: Reason for exit (stop_loss, time_exit, etc.)
        :param exit_data: Dictionary of exit-related data
        """
        try:
            # Handle different timestamp types
            if hasattr(candle_time, 'strftime'):
                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(candle_time, (int, float)):
                timestamp_str = str(candle_time)
            else:
                timestamp_str = str(candle_time)
            
            extra_data = {
                'timestamp': timestamp_str,
                'reason': exit_reason,
            }
            
            # Add reason-specific data
            if exit_reason == 'stop_loss':
                extra_data.update({
                    'stop_price': exit_data.get('stop_loss_price', 0.0),
                    'current_price': exit_data.get('current_price', 0.0)
                })
            elif exit_reason == 'time_exit':
                extra_data.update({
                    'duration_minutes': exit_data.get('position_duration', 0.0),
                    'target_minutes': exit_data.get('target_minutes', 0.0)
                })
            
            message = f"Exit signal generated due to {exit_reason}"
            self.log_strategy_event('EXIT', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging exit signal for {pair}: {str(e)}")

    def log_breakout_detection(self, pair: str, candle_time, 
                              breakout_data: dict) -> None:
        """
        Log breakout detection events
        
        Requirements: 7.5 - Key event logging for breakout detection
        
        :param pair: Trading pair
        :param candle_time: Timestamp of the candle (can be pd.Timestamp, datetime, or other)
        :param breakout_data: Dictionary of breakout-related data
        """
        try:
            # Handle different timestamp types
            if hasattr(candle_time, 'strftime'):
                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(candle_time, (int, float)):
                timestamp_str = str(candle_time)
            else:
                timestamp_str = str(candle_time)
            
            extra_data = {
                'timestamp': timestamp_str,
                '5m_close': breakout_data.get('close_5m', 0.0),
                '1h_high_prev': breakout_data.get('high_1h_prev', 0.0),
                'breakout_strength': f"{breakout_data.get('breakout_strength', 0.0):.4f}%"
            }
            
            message = f"Breakout detected: 5m close {breakout_data.get('close_5m', 0.0):.6f} > 1h high {breakout_data.get('high_1h_prev', 0.0):.6f}"
            self.log_strategy_event('BREAKOUT', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging breakout detection for {pair}: {str(e)}")

    def log_pullback_detection(self, pair: str, candle_time: pd.Timestamp, 
                              pullback_data: dict) -> None:
        """
        Log pullback detection events
        
        Requirements: 7.5 - Key event logging for pullback detection
        
        :param pair: Trading pair
        :param candle_time: Timestamp of the candle
        :param pullback_data: Dictionary of pullback-related data
        """
        try:
            # Handle different timestamp types
            if hasattr(candle_time, 'strftime'):
                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(candle_time, (int, float)):
                timestamp_str = str(candle_time)
            else:
                timestamp_str = str(candle_time)
            
            extra_data = {
                'timestamp': timestamp_str,
                'close': pullback_data.get('close', 0.0),
                'low': pullback_data.get('low', 0.0),
                'ma5': pullback_data.get('ma5', 0.0),
                'pullback_strength': f"{pullback_data.get('pullback_strength', 0.0):.4f}%"
            }
            
            message = f"Pullback detected: price touched MA5 support"
            self.log_strategy_event('PULLBACK', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging pullback detection for {pair}: {str(e)}")

    def log_rebound_detection(self, pair: str, candle_time, 
                             rebound_data: dict) -> None:
        """
        Log rebound detection events
        
        Requirements: 7.5 - Key event logging for rebound detection
        
        :param pair: Trading pair
        :param candle_time: Timestamp of the candle
        :param rebound_data: Dictionary of rebound-related data
        """
        try:
            # Handle different timestamp types
            if hasattr(candle_time, 'strftime'):
                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(candle_time, (int, float)):
                timestamp_str = str(candle_time)
            else:
                timestamp_str = str(candle_time)
            
            extra_data = {
                'timestamp': timestamp_str,
                'close': rebound_data.get('close', 0.0),
                'ma5': rebound_data.get('ma5', 0.0),
                'rebound_strength': f"{rebound_data.get('rebound_strength', 0.0):.4f}%"
            }
            
            message = f"Rebound detected: price bounced from MA5"
            self.log_strategy_event('REBOUND', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging rebound detection for {pair}: {str(e)}")

    def log_parameter_validation(self, validation_results: dict) -> None:
        """
        Log parameter validation results
        
        Requirements: 7.5 - Key event logging for parameter validation
        
        :param validation_results: Dictionary of validation results
        """
        try:
            if validation_results.get('success', True):
                message = "Parameter validation completed successfully"
                level = 'info'
            else:
                message = f"Parameter validation issues found: {validation_results.get('issues', [])}"
                level = 'warning'
            
            extra_data = {
                'validated_params': len(validation_results.get('validated_params', [])),
                'issues_found': len(validation_results.get('issues', [])),
                'corrections_made': len(validation_results.get('corrections', []))
            }
            
            self.log_strategy_event('VALIDATION', 'SYSTEM', message, level, extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging parameter validation: {str(e)}")

    def log_performance_metrics(self, pair: str, metrics: dict) -> None:
        """
        Log performance and statistics
        
        Requirements: 7.5 - Debug information output
        
        :param pair: Trading pair
        :param metrics: Dictionary of performance metrics
        """
        try:
            extra_data = {
                'total_candles': metrics.get('total_candles', 0),
                'breakout_count': metrics.get('breakout_count', 0),
                'entry_count': metrics.get('entry_count', 0),
                'exit_count': metrics.get('exit_count', 0),
                'processing_time_ms': metrics.get('processing_time_ms', 0.0)
            }
            
            message = f"Performance metrics: {metrics.get('entry_count', 0)} entries, {metrics.get('exit_count', 0)} exits in {metrics.get('total_candles', 0)} candles"
            self.log_strategy_event('PERFORMANCE', pair, message, 'debug', extra_data)
            
        except Exception as e:
            self.logger.error(f"Error logging performance metrics for {pair}: {str(e)}")

    def log_error_with_context(self, error: Exception, context: str, pair: str = None, 
                              additional_data: dict = None) -> None:
        """
        Log errors with contextual information
        
        Requirements: 7.5 - Debug information output for error handling
        
        :param error: Exception that occurred
        :param context: Context where error occurred
        :param pair: Trading pair (if applicable)
        :param additional_data: Additional contextual data
        """
        try:
            extra_data = {
                'error_type': type(error).__name__,
                'context': context,
                'error_message': str(error)
            }
            
            if additional_data:
                extra_data.update(additional_data)
            
            message = f"Error in {context}: {str(error)}"
            target_pair = pair or 'SYSTEM'
            
            self.log_strategy_event('ERROR', target_pair, message, 'error', extra_data)
            
        except Exception as logging_error:
            # Fallback to basic logging if structured logging fails
            print(f"Critical logging error: {str(logging_error)} | Original error: {str(error)} in {context}")

    def enable_debug_logging(self) -> None:
        """
        Enable debug-level logging for detailed troubleshooting
        
        Requirements: 7.5 - Debug information output
        """
        try:
            import logging
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.setLevel(logging.DEBUG)
                self.logger.info("Debug logging enabled for detailed troubleshooting")
            else:
                self._setup_logging_system()
                self.enable_debug_logging()
                
        except Exception as e:
            print(f"Error enabling debug logging: {str(e)}")

    def disable_debug_logging(self) -> None:
        """
        Disable debug-level logging to reduce log volume
        
        Requirements: 7.5 - Debug information output control
        """
        try:
            import logging
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.setLevel(logging.INFO)
                self.logger.info("Debug logging disabled")
                
        except Exception as e:
            print(f"Error disabling debug logging: {str(e)}")

    def log_strategy_initialization(self) -> None:
        """
        Log strategy initialization information
        
        Requirements: 7.5 - Key event logging for strategy startup
        """
        try:
            # Setup logging system first
            self._setup_logging_system()
            
            # Log initialization details
            init_data = {
                'strategy_name': self.__class__.__name__,
                'timeframe': self.timeframe,
                'interface_version': self.INTERFACE_VERSION,
                'startup_candle_count': self.startup_candle_count,
                'ma_period': self.ma_period.value,
                'exit_minutes': self.exit_minutes.value,
                'min_breakout_pct': self.min_breakout_pct.value,
                'pullback_tolerance': self.pullback_tolerance.value
            }
            
            message = f"Strategy initialized with timeframe {self.timeframe}"
            self.log_strategy_event('INIT', 'SYSTEM', message, 'debug', init_data)
            
        except Exception as e:
            print(f"Error logging strategy initialization: {str(e)}")

    # ================================
    # BACKTEST SUPPORT AND OPTIMIZATION
    # Requirements: 8.1, 8.2, 8.3 - Backtest support and validation
    # ================================

    def __init__(self, config: dict) -> None:
        """
        Initialize strategy with parameter validation and backtest mode detection
        
        Requirements: 9.5 - Parameter validation and boundary conditions
        Requirements: 8.1 - Backtest engine compatibility
        
        :param config: Strategy configuration dictionary
        """
        super().__init__(config)
        
        # Initialize logging system (Requirements: 7.5)
        self.log_strategy_initialization()
        
        # Detect backtest mode (Requirements: 8.1)
        self._is_backtest_mode = self._detect_backtest_mode(config)
        
        # Initialize backtest optimization settings (Requirements: 8.2)
        self._setup_backtest_optimizations()
        
        # Validate HyperOpt parameters on initialization
        self._validate_hyperopt_parameters()
        
        # Initialize performance tracking for backtest
        self._backtest_performance_tracker = {
            'total_candles_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_access_count': 0,
            'signal_generation_count': 0
        }

    def _detect_backtest_mode(self, config: dict) -> bool:
        """
        Detect if strategy is running in backtest mode
        
        Requirements: 8.1 - Backtest mode detection
        
        :param config: Strategy configuration dictionary
        :return: True if running in backtest mode
        """
        try:
            # Check various indicators of backtest mode
            backtest_indicators = [
                # Direct backtest flag
                config.get('runmode') == 'backtest',
                config.get('command') == 'backtesting',
                
                # Dry run with historical data (common in backtest)
                config.get('dry_run', False) and config.get('datadir') is not None,
                
                # Presence of backtest-specific configuration
                'backtest_directory' in config,
                'export' in config,
                'timerange' in config,
                
                # Check if we're in a backtest environment
                hasattr(self, 'dp') and hasattr(self.dp, 'runmode') and self.dp.runmode == 'backtest'
            ]
            
            is_backtest = any(backtest_indicators)
            
            # Log backtest mode detection
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Backtest mode detected: {is_backtest}")
                if is_backtest:
                    self.logger.info("Enabling backtest optimizations")
            
            return is_backtest
            
        except Exception as e:
            # Default to False if detection fails
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error detecting backtest mode: {str(e)}, defaulting to live mode")
            return False

    def _setup_backtest_optimizations(self) -> None:
        """
        Setup optimizations specific to backtest mode
        
        Requirements: 8.2 - Optimize historical data processing and multi-timeframe access
        """
        try:
            if self._is_backtest_mode:
                # Enable data caching for backtest performance
                self._enable_data_caching = True
                self._data_cache = {}
                self._cache_max_size = 1000  # Maximum cached entries
                
                # Optimize multi-timeframe data access
                self._optimize_informative_pairs = True
                self._informative_cache = {}
                
                # Reduce logging verbosity in backtest for performance
                self._backtest_log_level = 'WARNING'
                
                # Enable batch processing for indicators
                self._enable_batch_processing = True
                self._batch_size = 100
                
                # Pre-allocate arrays for better performance
                self._preallocate_arrays = True
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("Backtest optimizations enabled: caching, batch processing, reduced logging")
            else:
                # Live mode settings
                self._enable_data_caching = False
                self._data_cache = None
                self._optimize_informative_pairs = False
                self._informative_cache = None
                self._backtest_log_level = 'INFO'
                self._enable_batch_processing = False
                self._preallocate_arrays = False
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("Live mode optimizations enabled: real-time processing, full logging")
                    
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error setting up backtest optimizations: {str(e)}")

    def _get_cached_informative_data(self, pair: str, timeframe: str) -> Optional[DataFrame]:
        """
        Get cached informative data for backtest performance optimization
        
        Requirements: 8.2 - Optimize multi-timeframe data access
        
        :param pair: Trading pair
        :param timeframe: Timeframe (5m, 1h)
        :return: Cached DataFrame or None if not cached
        """
        try:
            if not self._enable_data_caching or not hasattr(self, '_informative_cache'):
                return None
            
            cache_key = f"{pair}_{timeframe}"
            
            if cache_key in self._informative_cache:
                self._backtest_performance_tracker['cache_hits'] += 1
                return self._informative_cache[cache_key].copy()
            else:
                self._backtest_performance_tracker['cache_misses'] += 1
                return None
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error accessing informative data cache: {str(e)}")
            return None

    def _cache_informative_data(self, pair: str, timeframe: str, dataframe: DataFrame) -> None:
        """
        Cache informative data for backtest performance optimization
        
        Requirements: 8.2 - Optimize multi-timeframe data access
        
        :param pair: Trading pair
        :param timeframe: Timeframe (5m, 1h)
        :param dataframe: DataFrame to cache
        """
        try:
            if not self._enable_data_caching or not hasattr(self, '_informative_cache'):
                return
            
            cache_key = f"{pair}_{timeframe}"
            
            # Implement cache size limit
            if len(self._informative_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._informative_cache))
                del self._informative_cache[oldest_key]
            
            # Cache the dataframe
            self._informative_cache[cache_key] = dataframe.copy()
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error caching informative data: {str(e)}")

    def _optimize_historical_data_processing(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Optimize historical data processing for backtest performance
        
        Requirements: 8.1, 8.2 - Ensure historical data correct handling and optimize processing
        
        :param dataframe: Input dataframe
        :param metadata: Metadata including pair information
        :return: Optimized dataframe
        """
        try:
            pair = metadata.get('pair', 'UNKNOWN')
            
            # Track processing start time
            import time
            start_time = time.time()
            
            # Validate historical data integrity (Requirements: 8.1)
            dataframe = self._validate_historical_data_integrity(dataframe, pair)
            
            # Optimize data types for backtest performance (Requirements: 8.2)
            if self._is_backtest_mode:
                dataframe = self._optimize_dataframe_dtypes(dataframe)
            
            # Pre-allocate result arrays if enabled (Requirements: 8.2)
            if self._preallocate_arrays and self._is_backtest_mode:
                dataframe = self._preallocate_indicator_columns(dataframe)
            
            # Track processing time
            processing_time = time.time() - start_time
            self._backtest_performance_tracker['total_processing_time'] += processing_time
            self._backtest_performance_tracker['total_candles_processed'] += len(dataframe)
            
            # Log performance metrics periodically
            if self._backtest_performance_tracker['total_candles_processed'] % 10000 == 0:
                self._log_backtest_performance_metrics(pair)
            
            return dataframe
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error optimizing historical data processing for {pair}: {str(e)}")
            return dataframe

    def _validate_historical_data_integrity(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Validate historical data integrity for backtest accuracy
        
        Requirements: 8.1 - Ensure historical data correct handling
        
        :param dataframe: Input dataframe
        :param pair: Trading pair name
        :return: Validated dataframe
        """
        try:
            original_length = len(dataframe)
            
            # Check for missing timestamps (gaps in historical data)
            if len(dataframe) > 1 and 'date' in dataframe.columns:
                time_diffs = dataframe['date'].diff()
                expected_diff = pd.Timedelta(minutes=1)  # 1-minute timeframe
                
                # Allow some tolerance for weekend gaps or exchange downtime
                tolerance = pd.Timedelta(minutes=5)
                large_gaps = time_diffs > (expected_diff + tolerance)
                
                if large_gaps.any():
                    gap_count = large_gaps.sum()
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"Found {gap_count} large time gaps in historical data for {pair}")
            
            # Check for duplicate timestamps (log only, don't modify)
            if 'date' in dataframe.columns:
                duplicate_timestamps = dataframe['date'].duplicated()
                if duplicate_timestamps.any():
                    duplicate_count = duplicate_timestamps.sum()
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"Found {duplicate_count} duplicate timestamps for {pair}")
                
                # Check if data is sorted (log only, don't modify)
                if not dataframe['date'].is_monotonic_increasing:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"Historical data not sorted by timestamp for {pair}")
            
            # Validate OHLCV data ranges and relationships (read-only validation)
            self._validate_ohlcv_relationships(dataframe, pair)
            
            # Check for sufficient data length
            if len(dataframe) < self.startup_candle_count:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug(f"Insufficient historical data for {pair}: {len(dataframe)} < {self.startup_candle_count}")
            
            return dataframe
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error validating historical data integrity for {pair}: {str(e)}")
            return dataframe

    def _validate_ohlcv_relationships(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Validate OHLCV data relationships for historical accuracy
        
        Requirements: 8.1 - Ensure historical data correct handling
        
        :param dataframe: Input dataframe
        :param pair: Trading pair name
        :return: Validated dataframe
        """
        try:
            issues_found = 0
            
            # Check High >= Low relationship
            invalid_hl = dataframe['high'] < dataframe['low']
            if invalid_hl.any():
                issues_found += invalid_hl.sum()
                # Fix by swapping high and low
                dataframe.loc[invalid_hl, ['high', 'low']] = dataframe.loc[invalid_hl, ['low', 'high']].values
            
            # Check Close within High-Low range
            close_too_high = dataframe['close'] > dataframe['high']
            if close_too_high.any():
                issues_found += close_too_high.sum()
                # Fix by setting close to high
                dataframe.loc[close_too_high, 'close'] = dataframe.loc[close_too_high, 'high']
            
            close_too_low = dataframe['close'] < dataframe['low']
            if close_too_low.any():
                issues_found += close_too_low.sum()
                # Fix by setting close to low
                dataframe.loc[close_too_low, 'close'] = dataframe.loc[close_too_low, 'low']
            
            # Check Open within High-Low range
            open_too_high = dataframe['open'] > dataframe['high']
            if open_too_high.any():
                issues_found += open_too_high.sum()
                dataframe.loc[open_too_high, 'open'] = dataframe.loc[open_too_high, 'high']
            
            open_too_low = dataframe['open'] < dataframe['low']
            if open_too_low.any():
                issues_found += open_too_low.sum()
                dataframe.loc[open_too_low, 'open'] = dataframe.loc[open_too_low, 'low']
            
            # Check for zero or negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                invalid_prices = dataframe[col] <= 0
                if invalid_prices.any():
                    issues_found += invalid_prices.sum()
                    # Use forward fill to fix invalid prices
                    dataframe.loc[invalid_prices, col] = np.nan
                    dataframe[col] = dataframe[col].ffill().bfill()
            
            # Check for negative volume
            negative_volume = dataframe['volume'] < 0
            if negative_volume.any():
                issues_found += negative_volume.sum()
                dataframe.loc[negative_volume, 'volume'] = 0
            
            if issues_found > 0 and hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Fixed {issues_found} OHLCV relationship issues in historical data for {pair}")
            
            return dataframe
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error validating OHLCV relationships for {pair}: {str(e)}")
            return dataframe

    def _optimize_dataframe_dtypes(self, dataframe: DataFrame) -> DataFrame:
        """
        Optimize dataframe data types for backtest performance
        
        Requirements: 8.2 - Optimize multi-timeframe data access
        
        :param dataframe: Input dataframe
        :return: Optimized dataframe
        """
        try:
            # Optimize numeric columns to use appropriate precision
            float_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in float_columns:
                if col in dataframe.columns:
                    # Use float32 for better memory efficiency in backtest
                    if dataframe[col].dtype == 'float64':
                        # Check if values fit in float32 range
                        max_val = dataframe[col].max()
                        min_val = dataframe[col].min()
                        
                        if (max_val < 3.4e38 and min_val > -3.4e38 and 
                            not pd.isna(max_val) and not pd.isna(min_val)):
                            dataframe[col] = dataframe[col].astype('float32')
            
            # Optimize boolean columns
            bool_columns = ['breakout_condition', 'pullback_condition', 'rebound_condition', 
                          'entry_signal', 'data_quality_ok']
            for col in bool_columns:
                if col in dataframe.columns and dataframe[col].dtype != 'bool':
                    dataframe[col] = dataframe[col].astype('bool')
            
            # Optimize integer columns
            int_columns = ['enter_long', 'exit_long']
            for col in int_columns:
                if col in dataframe.columns:
                    # Use int8 for binary signals (0/1)
                    if dataframe[col].dtype in ['int64', 'int32']:
                        max_val = dataframe[col].max()
                        min_val = dataframe[col].min()
                        
                        if max_val <= 127 and min_val >= -128:
                            dataframe[col] = dataframe[col].astype('int8')
            
            return dataframe
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error optimizing dataframe dtypes: {str(e)}")
            return dataframe

    def _preallocate_indicator_columns(self, dataframe: DataFrame) -> DataFrame:
        """
        Pre-allocate indicator columns for better backtest performance
        
        Requirements: 8.2 - Optimize multi-timeframe data access
        
        :param dataframe: Input dataframe
        :return: Dataframe with pre-allocated columns
        """
        try:
            # Pre-allocate common indicator columns with appropriate dtypes
            indicator_columns = {
                f'ma{self.ma_period.value}': 'float32',
                'breakout_condition': 'bool',
                'pullback_condition': 'bool',
                'rebound_condition': 'bool',
                'entry_signal': 'bool',
                'breakout_state_active': 'bool',
                'pullback_completed': 'bool',
                'data_quality_ok': 'bool',
                'entry_signal_strength': 'float32',
                'breakout_strength': 'float32',
                'pullback_strength': 'float32',
                'stop_loss_price': 'float32',
                'dynamic_stop_loss': 'float32',
                'position_duration_minutes': 'float32',
                'enter_long': 'int8',
                'exit_long': 'int8',
                'enter_tag': 'object',
                'exit_tag': 'object'
            }
            
            for col_name, dtype in indicator_columns.items():
                if col_name not in dataframe.columns:
                    if dtype == 'bool':
                        dataframe[col_name] = False
                    elif dtype in ['float32', 'float64']:
                        dataframe[col_name] = 0.0
                    elif dtype in ['int8', 'int32', 'int64']:
                        dataframe[col_name] = 0
                    else:  # object/string
                        dataframe[col_name] = ''
                    
                    # Set the appropriate dtype
                    if dtype != 'object':
                        dataframe[col_name] = dataframe[col_name].astype(dtype)
            
            return dataframe
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error pre-allocating indicator columns: {str(e)}")
            return dataframe

    def _log_backtest_performance_metrics(self, pair: str) -> None:
        """
        Log backtest performance metrics for monitoring
        
        Requirements: 8.2 - Optimize multi-timeframe data access
        
        :param pair: Trading pair name
        """
        try:
            metrics = self._backtest_performance_tracker
            
            if metrics['total_candles_processed'] > 0:
                avg_processing_time = metrics['total_processing_time'] / metrics['total_candles_processed'] * 1000  # ms per candle
                
                cache_hit_rate = 0.0
                if (metrics['cache_hits'] + metrics['cache_misses']) > 0:
                    cache_hit_rate = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100
                
                performance_data = {
                    'total_candles': metrics['total_candles_processed'],
                    'avg_processing_time_ms': round(avg_processing_time, 4),
                    'cache_hit_rate_pct': round(cache_hit_rate, 2),
                    'cache_hits': metrics['cache_hits'],
                    'cache_misses': metrics['cache_misses'],
                    'data_access_count': metrics['data_access_count']
                }
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Backtest performance metrics for {pair}: {performance_data}")
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Error logging backtest performance metrics: {str(e)}")