"""
Property-based tests for CLI parser.

Tests the command line argument parsing functionality using Hypothesis
for property-based testing.

**Property 7: CLI 参数解析正确性**
**Validates: Requirements 5.1-5.8**
"""

import pytest
import os
import json
import tempfile
from typing import List, Optional
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from parallel_backtest.cli import (
    CLIParser,
    parse_args,
    parse_pairs,
    get_default_workers,
    get_cpu_count,
    load_pairs_from_config,
    validate_config_file,
    validate_workers,
    validate_timeout,
    create_parser
)
from parallel_backtest.models import BacktestConfig


# Custom strategies for generating test data
@st.composite
def valid_pair_strategy(draw):
    """Generate valid trading pair strings like BTC/USDT"""
    base_currencies = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'DOT', 'LINK', 'AVAX']
    quote_currencies = ['USDT', 'BUSD', 'USD', 'EUR', 'BTC']
    
    base = draw(st.sampled_from(base_currencies))
    quote = draw(st.sampled_from(quote_currencies))
    
    # Ensure base and quote are different
    assume(base != quote)
    
    return f"{base}/{quote}"


@st.composite
def valid_pairs_list_strategy(draw):
    """Generate a list of valid trading pairs"""
    num_pairs = draw(st.integers(min_value=1, max_value=10))
    pairs = []
    
    for _ in range(num_pairs):
        pair = draw(valid_pair_strategy())
        if pair not in pairs:  # Avoid duplicates
            pairs.append(pair)
    
    return pairs


@st.composite
def valid_timerange_strategy(draw):
    """Generate valid timerange strings like 20240101-20241231"""
    start_year = draw(st.integers(min_value=2020, max_value=2024))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    
    end_year = draw(st.integers(min_value=start_year, max_value=2025))
    end_month = draw(st.integers(min_value=1, max_value=12))
    end_day = draw(st.integers(min_value=1, max_value=28))
    
    # Ensure end date is after start date
    if end_year == start_year and end_month == start_month:
        assume(end_day > start_day)
    elif end_year == start_year:
        assume(end_month > start_month)
    
    start_str = f"{start_year}{start_month:02d}{start_day:02d}"
    end_str = f"{end_year}{end_month:02d}{end_day:02d}"
    
    return f"{start_str}-{end_str}"


def create_temp_config(config_data: dict) -> str:
    """Create a temporary config file and return its path."""
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(config_data, f)
    except:
        os.close(fd)
        raise
    return path


def cleanup_temp_file(path: str) -> None:
    """Safely cleanup a temporary file."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except PermissionError:
        pass  # Ignore permission errors on Windows


class TestParsePairs:
    """Tests for parse_pairs function"""
    
    def test_parse_empty_pairs(self):
        """Test parsing empty pairs list"""
        assert parse_pairs(None) == []
        assert parse_pairs([]) == []
    
    def test_parse_space_separated_pairs(self):
        """Test parsing space-separated pairs"""
        result = parse_pairs(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        assert result == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    def test_parse_comma_separated_pairs(self):
        """Test parsing comma-separated pairs"""
        result = parse_pairs(['BTC/USDT,ETH/USDT,SOL/USDT'])
        assert result == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    def test_parse_mixed_format_pairs(self):
        """Test parsing mixed format pairs"""
        result = parse_pairs(['BTC/USDT', 'ETH/USDT,SOL/USDT'])
        assert result == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    def test_parse_pairs_removes_duplicates(self):
        """Test that duplicate pairs are removed"""
        result = parse_pairs(['BTC/USDT', 'BTC/USDT', 'ETH/USDT'])
        assert result == ['BTC/USDT', 'ETH/USDT']
    
    def test_parse_pairs_preserves_order(self):
        """Test that pair order is preserved"""
        result = parse_pairs(['SOL/USDT', 'BTC/USDT', 'ETH/USDT'])
        assert result == ['SOL/USDT', 'BTC/USDT', 'ETH/USDT']
    
    def test_parse_pairs_strips_whitespace(self):
        """Test that whitespace is stripped"""
        result = parse_pairs(['  BTC/USDT  ', '  ETH/USDT  '])
        assert result == ['BTC/USDT', 'ETH/USDT']
    
    @given(pairs=valid_pairs_list_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_parse_pairs_preserves_unique_pairs(self, pairs: List[str]):
        """
        **Feature: parallel-backtest-tool, Property 7: CLI 参数解析正确性**
        
        For any list of unique trading pairs, parsing should preserve all pairs
        and maintain their order.
        
        **Validates: Requirements 5.6**
        """
        result = parse_pairs(pairs)
        
        # All unique pairs should be preserved
        assert len(result) == len(pairs)
        
        # Order should be preserved
        for i, pair in enumerate(pairs):
            assert result[i] == pair


class TestDefaultWorkers:
    """Tests for default worker calculation"""
    
    def test_get_cpu_count_returns_positive(self):
        """Test that CPU count is always positive"""
        cpu_count = get_cpu_count()
        assert cpu_count >= 1
    
    def test_get_default_workers_returns_positive(self):
        """Test that default workers is always positive"""
        workers = get_default_workers()
        assert workers >= 1
    
    def test_get_default_workers_less_than_cpu_count(self):
        """Test that default workers is CPU count - 1 (minimum 1)"""
        cpu_count = get_cpu_count()
        workers = get_default_workers()
        
        expected = max(1, cpu_count - 1)
        assert workers == expected


class TestValidation:
    """Tests for validation functions"""
    
    def test_validate_workers_accepts_positive(self):
        """Test that positive worker counts are accepted"""
        validate_workers(1)
        validate_workers(4)
        validate_workers(100)
    
    def test_validate_workers_rejects_zero(self):
        """Test that zero workers is rejected"""
        with pytest.raises(ValueError):
            validate_workers(0)
    
    def test_validate_workers_rejects_negative(self):
        """Test that negative workers is rejected"""
        with pytest.raises(ValueError):
            validate_workers(-1)
    
    def test_validate_timeout_accepts_positive(self):
        """Test that positive timeout is accepted"""
        validate_timeout(1)
        validate_timeout(3600)
        validate_timeout(86400)
    
    def test_validate_timeout_rejects_zero(self):
        """Test that zero timeout is rejected"""
        with pytest.raises(ValueError):
            validate_timeout(0)
    
    def test_validate_timeout_rejects_negative(self):
        """Test that negative timeout is rejected"""
        with pytest.raises(ValueError):
            validate_timeout(-1)
    
    def test_validate_config_file_rejects_nonexistent(self):
        """Test that nonexistent config file is rejected"""
        with pytest.raises(FileNotFoundError):
            validate_config_file('/nonexistent/path/config.json')


class TestLoadPairsFromConfig:
    """Tests for loading pairs from config file"""
    
    def test_load_pairs_from_exchange_section(self):
        """Test loading pairs from exchange.pair_whitelist"""
        config = {
            'exchange': {
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT']
            }
        }
        path = create_temp_config(config)
        try:
            pairs = load_pairs_from_config(path)
            assert pairs == ['BTC/USDT', 'ETH/USDT']
        finally:
            cleanup_temp_file(path)
    
    def test_load_pairs_from_top_level(self):
        """Test loading pairs from top-level pair_whitelist"""
        config = {
            'pair_whitelist': ['SOL/USDT', 'XRP/USDT']
        }
        path = create_temp_config(config)
        try:
            pairs = load_pairs_from_config(path)
            assert pairs == ['SOL/USDT', 'XRP/USDT']
        finally:
            cleanup_temp_file(path)
    
    def test_load_pairs_raises_on_missing_whitelist(self):
        """Test that missing pair_whitelist raises KeyError"""
        config = {'exchange': {'name': 'binance'}}
        path = create_temp_config(config)
        try:
            with pytest.raises(KeyError):
                load_pairs_from_config(path)
        finally:
            cleanup_temp_file(path)


class TestParseArgs:
    """Tests for full argument parsing"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing"""
        config = {
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            }
        }
        path = create_temp_config(config)
        yield path
        cleanup_temp_file(path)
    
    def test_parse_required_args_only(self, temp_config_file):
        """Test parsing with only required arguments"""
        config, extra = parse_args([
            '--config', temp_config_file,
            '--strategy', 'TestStrategy'
        ])
        
        assert config.config_path == temp_config_file
        assert config.strategy == 'TestStrategy'
        assert config.pairs == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # From config
        assert config.timerange is None
        assert config.max_workers == get_default_workers()
        assert config.output_dir == 'user_data/backtest_results'
        assert config.timeout == 3600
        assert config.debug is False
        assert extra == []
    
    def test_parse_all_args(self, temp_config_file):
        """Test parsing with all arguments"""
        config, extra = parse_args([
            '--config', temp_config_file,
            '--strategy', 'TestStrategy',
            '--timerange', '20240101-20241231',
            '--workers', '4',
            '--output', '/custom/output',
            '--pairs', 'DOGE/USDT', 'SHIB/USDT',
            '--timeout', '1800',
            '--debug'
        ])
        
        assert config.config_path == temp_config_file
        assert config.strategy == 'TestStrategy'
        assert config.pairs == ['DOGE/USDT', 'SHIB/USDT']  # CLI overrides config
        assert config.timerange == '20240101-20241231'
        assert config.max_workers == 4
        assert config.output_dir == '/custom/output'
        assert config.timeout == 1800
        assert config.debug is True
    
    def test_parse_extra_args(self, temp_config_file):
        """Test parsing with extra Freqtrade arguments"""
        config, extra = parse_args([
            '--config', temp_config_file,
            '--strategy', 'TestStrategy',
            '--', '--cache', 'none', '--enable-protections'
        ])
        
        assert config.extra_args == ['--cache', 'none', '--enable-protections']
        assert extra == ['--cache', 'none', '--enable-protections']
    
    def test_parse_pairs_override_config(self, temp_config_file):
        """Test that --pairs overrides config file pairs"""
        config, _ = parse_args([
            '--config', temp_config_file,
            '--strategy', 'TestStrategy',
            '--pairs', 'LINK/USDT'
        ])
        
        # CLI pairs should override config pairs
        assert config.pairs == ['LINK/USDT']
        assert 'BTC/USDT' not in config.pairs
    
    def test_parse_short_options(self, temp_config_file):
        """Test parsing with short option names"""
        config, _ = parse_args([
            '-c', temp_config_file,
            '-s', 'TestStrategy',
            '-w', '2',
            '-o', '/output',
            '-p', 'BTC/USDT'
        ])
        
        assert config.config_path == temp_config_file
        assert config.strategy == 'TestStrategy'
        assert config.max_workers == 2
        assert config.output_dir == '/output'
        assert config.pairs == ['BTC/USDT']
    
    @given(
        workers=st.integers(min_value=1, max_value=32),
        timeout=st.integers(min_value=1, max_value=86400),
        debug=st.booleans(),
        timerange=st.one_of(st.none(), valid_timerange_strategy())
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_parsed_config_reflects_args(
        self, 
        temp_config_file, 
        workers: int, 
        timeout: int, 
        debug: bool,
        timerange: Optional[str]
    ):
        """
        **Feature: parallel-backtest-tool, Property 7: CLI 参数解析正确性**
        
        For any valid command line argument combination, the parsed BacktestConfig
        should correctly reflect all argument values, and unspecified arguments
        should use default values.
        
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**
        """
        args = [
            '--config', temp_config_file,
            '--strategy', 'TestStrategy',
            '--workers', str(workers),
            '--timeout', str(timeout)
        ]
        
        if debug:
            args.append('--debug')
        
        if timerange:
            args.extend(['--timerange', timerange])
        
        config, _ = parse_args(args)
        
        # Property 1: All specified values should be reflected
        assert config.max_workers == workers
        assert config.timeout == timeout
        assert config.debug == debug
        assert config.timerange == timerange
        
        # Property 2: Required args should be set
        assert config.config_path == temp_config_file
        assert config.strategy == 'TestStrategy'
        
        # Property 3: Default values for unspecified args
        assert config.output_dir == 'user_data/backtest_results'
        
        # Property 4: Config should be a valid BacktestConfig
        assert isinstance(config, BacktestConfig)


class TestCLIParserClass:
    """Tests for CLIParser class"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing"""
        config = {
            'exchange': {
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT']
            }
        }
        path = create_temp_config(config)
        yield path
        cleanup_temp_file(path)
    
    def test_cli_parser_class_instantiation(self):
        """Test CLIParser class can be instantiated"""
        parser = CLIParser()
        assert parser is not None
    
    def test_cli_parser_parse_args(self, temp_config_file):
        """Test CLIParser.parse_args method"""
        parser = CLIParser()
        config = parser.parse_args([
            '--config', temp_config_file,
            '--strategy', 'TestStrategy'
        ])
        
        assert isinstance(config, BacktestConfig)
        assert config.strategy == 'TestStrategy'


class TestErrorHandling:
    """Tests for error handling in CLI parsing"""
    
    def test_missing_required_config(self):
        """Test error when --config is missing"""
        with pytest.raises(SystemExit):
            parse_args(['--strategy', 'TestStrategy'])
    
    def test_missing_required_strategy(self):
        """Test error when --strategy is missing"""
        config = {'exchange': {'pair_whitelist': ['BTC/USDT']}}
        path = create_temp_config(config)
        try:
            with pytest.raises(SystemExit):
                parse_args(['--config', path])
        finally:
            cleanup_temp_file(path)
    
    def test_invalid_config_file(self):
        """Test error when config file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            parse_args([
                '--config', '/nonexistent/config.json',
                '--strategy', 'TestStrategy'
            ])
    
    def test_invalid_json_config(self):
        """Test error when config file is not valid JSON"""
        fd, path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write('not valid json')
            
            with pytest.raises(json.JSONDecodeError):
                parse_args([
                    '--config', path,
                    '--strategy', 'TestStrategy'
                ])
        finally:
            cleanup_temp_file(path)
    
    def test_empty_pairs_error(self):
        """Test error when no pairs are specified"""
        # Config without pair_whitelist
        config = {'exchange': {'name': 'binance'}}
        path = create_temp_config(config)
        try:
            with pytest.raises((ValueError, KeyError)):
                parse_args([
                    '--config', path,
                    '--strategy', 'TestStrategy'
                ])
        finally:
            cleanup_temp_file(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
