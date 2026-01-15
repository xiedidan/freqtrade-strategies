"""
Configuration generator for parallel backtest workers.

This module handles generation of isolated configuration files for each
backtest worker, ensuring resource isolation between parallel processes.
"""

import json
import os
import shutil
import tempfile
import uuid
from typing import List, Optional

from .models import BacktestConfig, WorkerConfig


class ConfigGenerator:
    """
    Generate isolated configuration files for each backtest worker.
    
    This class is responsible for:
    - Reading the base Freqtrade configuration
    - Extracting trading pairs from the config
    - Generating unique temporary config files for each worker
    - Managing temporary directories and cleanup
    
    Attributes:
        base_config_path: Path to the base Freqtrade configuration file
        temp_dir: Directory for storing temporary files
        _base_config: Cached base configuration dictionary
        _worker_configs: List of generated worker configurations
    """
    
    def __init__(self, base_config_path: str, temp_dir: Optional[str] = None):
        """
        Initialize the configuration generator.
        
        Args:
            base_config_path: Path to the base Freqtrade configuration file
            temp_dir: Directory for temporary files (auto-generated if None)
            
        Raises:
            FileNotFoundError: If base config file doesn't exist
            json.JSONDecodeError: If base config is not valid JSON
        """
        self.base_config_path = base_config_path
        self._base_config: Optional[dict] = None
        self._worker_configs: List[WorkerConfig] = []
        
        # Create or use provided temp directory
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='parallel_backtest_')
        else:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
        
        # Load and validate base config
        self._load_base_config()
    
    def _load_base_config(self) -> None:
        """
        Load the base configuration file.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is not valid JSON
        """
        if not os.path.exists(self.base_config_path):
            raise FileNotFoundError(f"Config file not found: {self.base_config_path}")
        
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            self._base_config = json.load(f)
    
    @property
    def base_config(self) -> dict:
        """
        Get the base configuration dictionary.
        
        Returns:
            Base configuration as dictionary
        """
        if self._base_config is None:
            self._load_base_config()
        return self._base_config
    
    def get_pairs_from_config(self) -> List[str]:
        """
        Extract trading pairs from the base configuration.
        
        Looks for pairs in exchange.pair_whitelist or top-level pair_whitelist.
        
        Returns:
            List of trading pairs from config
            
        Raises:
            KeyError: If no pair_whitelist found in config
        """
        config = self.base_config
        
        # Try exchange.pair_whitelist first
        if 'exchange' in config and 'pair_whitelist' in config['exchange']:
            return list(config['exchange']['pair_whitelist'])
        
        # Fallback to top-level pair_whitelist
        if 'pair_whitelist' in config:
            return list(config['pair_whitelist'])
        
        raise KeyError("No pair_whitelist found in config file")

    def generate_worker_config(self, pair: str, worker_id: int, time_chunk: str = None) -> WorkerConfig:
        """
        Generate an isolated configuration for a single trading pair.
        
        Creates a unique temporary directory for the worker with:
        - A modified config file with only the specified pair
        - An isolated result directory
        - An isolated log file path
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            worker_id: Unique worker identifier
            time_chunk: Optional time chunk label for split tasks
            
        Returns:
            WorkerConfig with paths to isolated resources
        """
        # Create unique worker directory using worker_id and UUID for extra uniqueness
        unique_suffix = uuid.uuid4().hex[:8]
        worker_dir = os.path.join(self.temp_dir, f'worker_{worker_id}_{unique_suffix}')
        os.makedirs(worker_dir, exist_ok=True)
        
        # Create result directory
        result_dir = os.path.join(worker_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # Generate config file path
        config_path = os.path.join(worker_dir, 'config.json')
        
        # Generate log file path
        log_file = os.path.join(worker_dir, 'freqtrade.log')
        
        # Create modified config for this worker
        worker_config_dict = self._create_worker_config_dict(pair)
        
        # Write config file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(worker_config_dict, f, indent=2)
        
        # Create WorkerConfig object
        worker_config = WorkerConfig(
            pair=pair,
            config_path=config_path,
            result_dir=result_dir,
            log_file=log_file,
            worker_id=worker_id,
            time_chunk=time_chunk
        )
        
        # Track generated configs for cleanup
        self._worker_configs.append(worker_config)
        
        return worker_config
    
    def _create_worker_config_dict(self, pair: str) -> dict:
        """
        Create a modified configuration dictionary for a single pair.
        
        Modifies the base config to:
        - Set pair_whitelist to only the specified pair
        - Disable pair blacklist for this specific pair
        
        Args:
            pair: Trading pair to configure
            
        Returns:
            Modified configuration dictionary
        """
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Update pair_whitelist to only include this pair
        if 'exchange' in config:
            config['exchange']['pair_whitelist'] = [pair]
            # Clear blacklist to ensure the pair isn't filtered out
            config['exchange']['pair_blacklist'] = []
        else:
            config['pair_whitelist'] = [pair]
            if 'pair_blacklist' in config:
                config['pair_blacklist'] = []
        
        return config
    
    def generate_all_worker_configs(self, pairs: List[str]) -> List[WorkerConfig]:
        """
        Generate isolated configurations for all trading pairs.
        
        Args:
            pairs: List of trading pairs to generate configs for
            
        Returns:
            List of WorkerConfig objects, one per pair
        """
        configs = []
        for idx, pair in enumerate(pairs):
            config = self.generate_worker_config(pair, worker_id=idx)
            configs.append(config)
        return configs
    
    def cleanup(self, force: bool = False) -> None:
        """
        Clean up temporary files and directories.
        
        Args:
            force: If True, remove even if debug mode might want to preserve
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._worker_configs.clear()
    
    @property
    def worker_configs(self) -> List[WorkerConfig]:
        """
        Get list of generated worker configurations.
        
        Returns:
            List of WorkerConfig objects
        """
        return list(self._worker_configs)
    
    def __enter__(self) -> 'ConfigGenerator':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup temporary files."""
        self.cleanup()


def create_config_generator(backtest_config: BacktestConfig) -> ConfigGenerator:
    """
    Factory function to create a ConfigGenerator from BacktestConfig.
    
    Args:
        backtest_config: Backtest configuration with base config path
        
    Returns:
        Configured ConfigGenerator instance
    """
    return ConfigGenerator(
        base_config_path=backtest_config.config_path,
        temp_dir=None  # Auto-generate temp directory
    )
