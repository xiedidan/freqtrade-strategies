"""
Property-based tests for ConfigGenerator.

Tests the configuration generation functionality using Hypothesis
for property-based testing.

**Property 1: 交易对解析一致性**
**Property 2: 资源隔离唯一性**
**Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3**
"""

import pytest
import os
import json
import tempfile
import shutil
from typing import List
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from parallel_backtest.config import ConfigGenerator, create_config_generator
from parallel_backtest.models import BacktestConfig, WorkerConfig


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
def unique_pairs_list_strategy(draw):
    """Generate a list of unique trading pairs (1-10 pairs)"""
    num_pairs = draw(st.integers(min_value=1, max_value=10))
    pairs = set()
    
    while len(pairs) < num_pairs:
        pair = draw(valid_pair_strategy())
        pairs.add(pair)
    
    return list(pairs)


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


def cleanup_temp_dir(path: str) -> None:
    """Safely cleanup a temporary directory."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except PermissionError:
        pass  # Ignore permission errors on Windows


class TestConfigGeneratorBasic:
    """Basic unit tests for ConfigGenerator"""
    
    @pytest.fixture
    def base_config(self):
        """Create a base config dictionary"""
        return {
            'strategy': 'TestStrategy',
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                'pair_blacklist': ['BNB/.*']
            },
            'stake_currency': 'USDT',
            'dry_run': True
        }
    
    @pytest.fixture
    def temp_config_file(self, base_config):
        """Create a temporary config file"""
        path = create_temp_config(base_config)
        yield path
        cleanup_temp_file(path)
    
    def test_init_creates_temp_dir(self, temp_config_file):
        """Test that initialization creates a temp directory"""
        generator = ConfigGenerator(temp_config_file)
        try:
            assert os.path.exists(generator.temp_dir)
            assert os.path.isdir(generator.temp_dir)
        finally:
            generator.cleanup()
    
    def test_init_with_custom_temp_dir(self, temp_config_file):
        """Test initialization with custom temp directory"""
        custom_dir = tempfile.mkdtemp(prefix='test_custom_')
        try:
            generator = ConfigGenerator(temp_config_file, temp_dir=custom_dir)
            assert generator.temp_dir == custom_dir
            generator.cleanup()
        finally:
            cleanup_temp_dir(custom_dir)
    
    def test_init_raises_on_missing_config(self):
        """Test that initialization raises on missing config file"""
        with pytest.raises(FileNotFoundError):
            ConfigGenerator('/nonexistent/config.json')
    
    def test_get_pairs_from_config(self, temp_config_file):
        """Test extracting pairs from config"""
        generator = ConfigGenerator(temp_config_file)
        try:
            pairs = generator.get_pairs_from_config()
            assert pairs == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        finally:
            generator.cleanup()
    
    def test_get_pairs_from_top_level(self):
        """Test extracting pairs from top-level pair_whitelist"""
        config = {'pair_whitelist': ['DOGE/USDT', 'SHIB/USDT']}
        path = create_temp_config(config)
        try:
            generator = ConfigGenerator(path)
            pairs = generator.get_pairs_from_config()
            assert pairs == ['DOGE/USDT', 'SHIB/USDT']
            generator.cleanup()
        finally:
            cleanup_temp_file(path)
    
    def test_get_pairs_raises_on_missing(self):
        """Test that missing pair_whitelist raises KeyError"""
        config = {'exchange': {'name': 'binance'}}
        path = create_temp_config(config)
        try:
            generator = ConfigGenerator(path)
            with pytest.raises(KeyError):
                generator.get_pairs_from_config()
            generator.cleanup()
        finally:
            cleanup_temp_file(path)


class TestWorkerConfigGeneration:
    """Tests for worker config generation"""
    
    @pytest.fixture
    def base_config(self):
        """Create a base config dictionary"""
        return {
            'strategy': 'TestStrategy',
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                'pair_blacklist': ['BNB/.*']
            },
            'stake_currency': 'USDT',
            'dry_run': True
        }
    
    @pytest.fixture
    def temp_config_file(self, base_config):
        """Create a temporary config file"""
        path = create_temp_config(base_config)
        yield path
        cleanup_temp_file(path)
    
    def test_generate_worker_config_creates_files(self, temp_config_file):
        """Test that worker config generation creates necessary files"""
        generator = ConfigGenerator(temp_config_file)
        try:
            worker_config = generator.generate_worker_config('BTC/USDT', worker_id=0)
            
            # Check WorkerConfig fields
            assert worker_config.pair == 'BTC/USDT'
            assert worker_config.worker_id == 0
            
            # Check files exist
            assert os.path.exists(worker_config.config_path)
            assert os.path.exists(worker_config.result_dir)
            assert os.path.isdir(worker_config.result_dir)
        finally:
            generator.cleanup()
    
    def test_worker_config_contains_single_pair(self, temp_config_file):
        """Test that worker config only contains the assigned pair"""
        generator = ConfigGenerator(temp_config_file)
        try:
            worker_config = generator.generate_worker_config('ETH/USDT', worker_id=1)
            
            # Read the generated config
            with open(worker_config.config_path, 'r') as f:
                config = json.load(f)
            
            # Check pair_whitelist only contains the assigned pair
            assert config['exchange']['pair_whitelist'] == ['ETH/USDT']
            # Check blacklist is cleared
            assert config['exchange']['pair_blacklist'] == []
        finally:
            generator.cleanup()
    
    def test_generate_all_worker_configs(self, temp_config_file):
        """Test generating configs for all pairs"""
        generator = ConfigGenerator(temp_config_file)
        try:
            pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            configs = generator.generate_all_worker_configs(pairs)
            
            assert len(configs) == 3
            
            for i, config in enumerate(configs):
                assert config.pair == pairs[i]
                assert config.worker_id == i
                assert os.path.exists(config.config_path)
        finally:
            generator.cleanup()
    
    def test_context_manager_cleanup(self, temp_config_file):
        """Test that context manager cleans up properly"""
        with ConfigGenerator(temp_config_file) as generator:
            temp_dir = generator.temp_dir
            generator.generate_worker_config('BTC/USDT', worker_id=0)
            assert os.path.exists(temp_dir)
        
        # After context exit, temp dir should be cleaned up
        assert not os.path.exists(temp_dir)


class TestPropertyPairParsingConsistency:
    """
    Property 1: 交易对解析一致性
    
    For any list of trading pairs provided via --pairs, the number of backtest
    tasks created should equal the number of pairs, and each task should
    correspond to a unique pair.
    
    **Validates: Requirements 1.1, 1.2**
    """
    
    @pytest.fixture
    def base_config(self):
        """Create a base config dictionary"""
        return {
            'strategy': 'TestStrategy',
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['DEFAULT/USDT'],
                'pair_blacklist': []
            }
        }
    
    @given(pairs=unique_pairs_list_strategy())
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_pair_parsing_consistency(self, pairs: List[str], base_config):
        """
        **Feature: parallel-backtest-tool, Property 1: 交易对解析一致性**
        
        For any list of trading pairs, the number of generated worker configs
        should equal the number of pairs, and each config should correspond
        to a unique pair.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Create temp config file
        config_path = create_temp_config(base_config)
        
        try:
            generator = ConfigGenerator(config_path)
            
            try:
                # Generate worker configs for all pairs
                worker_configs = generator.generate_all_worker_configs(pairs)
                
                # Property 1: Number of configs equals number of pairs
                assert len(worker_configs) == len(pairs)
                
                # Property 2: Each config corresponds to a unique pair
                config_pairs = [wc.pair for wc in worker_configs]
                assert len(set(config_pairs)) == len(pairs)
                
                # Property 3: All original pairs are represented
                for pair in pairs:
                    assert pair in config_pairs
                
                # Property 4: Each worker has a unique ID
                worker_ids = [wc.worker_id for wc in worker_configs]
                assert len(set(worker_ids)) == len(pairs)
                
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)


class TestPropertyResourceIsolationUniqueness:
    """
    Property 2: 资源隔离唯一性
    
    For any N generated WorkerConfigs, all config file paths, output directory
    paths, and log file paths should be pairwise distinct (3N paths total,
    no duplicates).
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    
    @pytest.fixture
    def base_config(self):
        """Create a base config dictionary"""
        return {
            'strategy': 'TestStrategy',
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['DEFAULT/USDT'],
                'pair_blacklist': []
            }
        }
    
    @given(pairs=unique_pairs_list_strategy())
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_resource_isolation_uniqueness(self, pairs: List[str], base_config):
        """
        **Feature: parallel-backtest-tool, Property 2: 资源隔离唯一性**
        
        For any N generated WorkerConfigs, all config file paths, output
        directory paths, and log file paths should be pairwise distinct.
        
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Create temp config file
        config_path = create_temp_config(base_config)
        
        try:
            generator = ConfigGenerator(config_path)
            
            try:
                # Generate worker configs for all pairs
                worker_configs = generator.generate_all_worker_configs(pairs)
                
                n = len(worker_configs)
                
                # Collect all paths
                config_paths = [wc.config_path for wc in worker_configs]
                result_dirs = [wc.result_dir for wc in worker_configs]
                log_files = [wc.log_file for wc in worker_configs]
                
                # Property 1: All config paths are unique
                assert len(set(config_paths)) == n, \
                    f"Config paths not unique: {config_paths}"
                
                # Property 2: All result directories are unique
                assert len(set(result_dirs)) == n, \
                    f"Result dirs not unique: {result_dirs}"
                
                # Property 3: All log files are unique
                assert len(set(log_files)) == n, \
                    f"Log files not unique: {log_files}"
                
                # Property 4: All 3N paths are pairwise distinct
                all_paths = config_paths + result_dirs + log_files
                assert len(set(all_paths)) == 3 * n, \
                    f"Not all paths are distinct: found {len(set(all_paths))} unique out of {3*n}"
                
                # Property 5: All config files actually exist
                for config_path in config_paths:
                    assert os.path.exists(config_path), \
                        f"Config file does not exist: {config_path}"
                
                # Property 6: All result directories actually exist
                for result_dir in result_dirs:
                    assert os.path.exists(result_dir), \
                        f"Result dir does not exist: {result_dir}"
                    assert os.path.isdir(result_dir), \
                        f"Result dir is not a directory: {result_dir}"
                
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)
    
    def test_resource_isolation_with_same_pair_multiple_times(self, base_config):
        """
        Test that generating configs for the same pair multiple times
        still produces unique paths.
        """
        config_path = create_temp_config(base_config)
        
        try:
            generator = ConfigGenerator(config_path)
            
            try:
                # Generate multiple configs for the same pair
                config1 = generator.generate_worker_config('BTC/USDT', worker_id=0)
                config2 = generator.generate_worker_config('BTC/USDT', worker_id=1)
                config3 = generator.generate_worker_config('BTC/USDT', worker_id=2)
                
                # All paths should be unique even for the same pair
                all_paths = [
                    config1.config_path, config1.result_dir, config1.log_file,
                    config2.config_path, config2.result_dir, config2.log_file,
                    config3.config_path, config3.result_dir, config3.log_file,
                ]
                
                assert len(set(all_paths)) == 9, \
                    "Paths should be unique even for same pair"
                
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)


class TestCreateConfigGeneratorFactory:
    """Tests for the factory function"""
    
    def test_create_config_generator_from_backtest_config(self):
        """Test creating ConfigGenerator from BacktestConfig"""
        base_config = {
            'exchange': {
                'pair_whitelist': ['BTC/USDT']
            }
        }
        config_path = create_temp_config(base_config)
        
        try:
            backtest_config = BacktestConfig(
                config_path=config_path,
                strategy='TestStrategy',
                pairs=['BTC/USDT']
            )
            
            generator = create_config_generator(backtest_config)
            
            try:
                assert generator.base_config_path == config_path
                assert os.path.exists(generator.temp_dir)
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)


class TestWorkerConfigContent:
    """Tests for the content of generated worker configs"""
    
    @pytest.fixture
    def complex_base_config(self):
        """Create a complex base config with many settings"""
        return {
            'strategy': 'TestStrategy',
            'strategy_path': 'user_data/strategies',
            'max_open_trades': 3,
            'stake_currency': 'USDT',
            'stake_amount': 'unlimited',
            'exchange': {
                'name': 'binance',
                'pair_whitelist': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                'pair_blacklist': ['BNB/.*', 'DOGE/.*']
            },
            'dry_run': True,
            'dry_run_wallet': 1000,
            'timeframe': '1m',
            'minimal_roi': {
                '60': 0.01,
                '30': 0.02,
                '0': 0.05
            },
            'stoploss': -0.05
        }
    
    def test_worker_config_preserves_base_settings(self, complex_base_config):
        """Test that worker config preserves non-pair settings"""
        config_path = create_temp_config(complex_base_config)
        
        try:
            generator = ConfigGenerator(config_path)
            
            try:
                worker_config = generator.generate_worker_config('BTC/USDT', worker_id=0)
                
                # Read the generated config
                with open(worker_config.config_path, 'r') as f:
                    config = json.load(f)
                
                # Check that base settings are preserved
                assert config['strategy'] == 'TestStrategy'
                assert config['stake_currency'] == 'USDT'
                assert config['dry_run'] is True
                assert config['dry_run_wallet'] == 1000
                assert config['timeframe'] == '1m'
                assert config['stoploss'] == -0.05
                assert config['minimal_roi'] == complex_base_config['minimal_roi']
                
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)
    
    def test_worker_config_clears_blacklist(self, complex_base_config):
        """Test that worker config clears the pair blacklist"""
        config_path = create_temp_config(complex_base_config)
        
        try:
            generator = ConfigGenerator(config_path)
            
            try:
                # Generate config for a pair that would normally be blacklisted
                worker_config = generator.generate_worker_config('BNB/USDT', worker_id=0)
                
                # Read the generated config
                with open(worker_config.config_path, 'r') as f:
                    config = json.load(f)
                
                # Blacklist should be cleared
                assert config['exchange']['pair_blacklist'] == []
                # Whitelist should only contain the assigned pair
                assert config['exchange']['pair_whitelist'] == ['BNB/USDT']
                
            finally:
                generator.cleanup()
        finally:
            cleanup_temp_file(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
