"""
DataFrame Backend Abstraction Layer

This module provides a unified interface for pandas and cuDF DataFrames,
allowing seamless switching between CPU (pandas) and GPU (cuDF) backends
for performance optimization.

Requirements: TASK-001 - CuDF acceleration experiment
"""

import sys
from typing import Any, Optional, Union
import numpy as np

# Try to import cuDF, fall back to pandas if not available
try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

import pandas as pd


class DataFrameBackend:
    """
    Abstract DataFrame backend for pandas/cuDF compatibility
    
    This class provides a unified interface for DataFrame operations
    that can work with both pandas (CPU) and cuDF (GPU) backends.
    """
    
    # Backend selection: 'auto', 'pandas', 'cudf'
    _backend = 'auto'
    _use_cudf = False
    
    @classmethod
    def initialize(cls, backend: str = 'auto') -> None:
        """
        Initialize the DataFrame backend
        
        Args:
            backend: Backend to use ('auto', 'pandas', 'cudf')
                    'auto' will use cuDF if available, otherwise pandas
        """
        cls._backend = backend
        
        if backend == 'cudf':
            if not CUDF_AVAILABLE:
                raise ImportError(
                    "cuDF is not available. Please install RAPIDS cuDF or use 'pandas' backend."
                )
            cls._use_cudf = True
        elif backend == 'pandas':
            cls._use_cudf = False
        else:  # auto
            cls._use_cudf = CUDF_AVAILABLE
    
    @classmethod
    def get_backend_name(cls) -> str:
        """Get the name of the currently active backend"""
        return 'cudf' if cls._use_cudf else 'pandas'
    
    @classmethod
    def is_cudf_available(cls) -> bool:
        """Check if cuDF is available"""
        return CUDF_AVAILABLE
    
    @classmethod
    def is_using_cudf(cls) -> bool:
        """Check if currently using cuDF backend"""
        return cls._use_cudf
    
    @classmethod
    def create_dataframe(cls, data: Any = None, **kwargs) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Create a DataFrame using the active backend
        
        Args:
            data: Data to create DataFrame from
            **kwargs: Additional arguments passed to DataFrame constructor
            
        Returns:
            DataFrame (pandas or cuDF depending on backend)
        """
        if cls._use_cudf:
            return cudf.DataFrame(data, **kwargs)
        else:
            return pd.DataFrame(data, **kwargs)
    
    @classmethod
    def to_pandas(cls, df: Union[pd.DataFrame, 'cudf.DataFrame']) -> pd.DataFrame:
        """
        Convert DataFrame to pandas DataFrame
        
        Args:
            df: DataFrame to convert
            
        Returns:
            pandas DataFrame
        """
        if cls._use_cudf and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df
    
    @classmethod
    def to_cudf(cls, df: pd.DataFrame) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Convert pandas DataFrame to cuDF DataFrame if cuDF is available
        
        Args:
            df: pandas DataFrame to convert
            
        Returns:
            cuDF DataFrame if available, otherwise original pandas DataFrame
        """
        if cls._use_cudf and CUDF_AVAILABLE:
            return cudf.from_pandas(df)
        return df
    
    @classmethod
    def merge(cls, 
              left: Union[pd.DataFrame, 'cudf.DataFrame'],
              right: Union[pd.DataFrame, 'cudf.DataFrame'],
              **kwargs) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Merge two DataFrames
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            **kwargs: Additional arguments passed to merge function
            
        Returns:
            Merged DataFrame
        """
        if cls._use_cudf:
            return cudf.merge(left, right, **kwargs)
        else:
            return pd.merge(left, right, **kwargs)
    
    @classmethod
    def concat(cls,
               dfs: list,
               **kwargs) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Concatenate DataFrames
        
        Args:
            dfs: List of DataFrames to concatenate
            **kwargs: Additional arguments passed to concat function
            
        Returns:
            Concatenated DataFrame
        """
        if cls._use_cudf:
            return cudf.concat(dfs, **kwargs)
        else:
            return pd.concat(dfs, **kwargs)
    
    @classmethod
    def read_csv(cls, filepath: str, **kwargs) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Read CSV file into DataFrame
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments passed to read_csv function
            
        Returns:
            DataFrame
        """
        if cls._use_cudf:
            return cudf.read_csv(filepath, **kwargs)
        else:
            return pd.read_csv(filepath, **kwargs)
    
    @classmethod
    def read_feather(cls, filepath: str, **kwargs) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Read Feather file into DataFrame
        
        Args:
            filepath: Path to Feather file
            **kwargs: Additional arguments passed to read_feather function
            
        Returns:
            DataFrame
        """
        if cls._use_cudf:
            # cuDF doesn't have direct feather support, read with pandas then convert
            df = pd.read_feather(filepath, **kwargs)
            return cudf.from_pandas(df)
        else:
            return pd.read_feather(filepath, **kwargs)


# Initialize with auto-detection by default
DataFrameBackend.initialize('auto')


def get_backend_info() -> dict:
    """
    Get information about the current DataFrame backend
    
    Returns:
        Dictionary with backend information
    """
    return {
        'backend': DataFrameBackend.get_backend_name(),
        'cudf_available': DataFrameBackend.is_cudf_available(),
        'using_cudf': DataFrameBackend.is_using_cudf(),
        'pandas_version': pd.__version__,
        'cudf_version': cudf.__version__ if CUDF_AVAILABLE else None,
    }


def print_backend_info() -> None:
    """Print DataFrame backend information"""
    info = get_backend_info()
    print("=" * 50)
    print("DataFrame Backend Information")
    print("=" * 50)
    print(f"Active Backend: {info['backend']}")
    print(f"cuDF Available: {info['cudf_available']}")
    print(f"Using cuDF: {info['using_cudf']}")
    print(f"Pandas Version: {info['pandas_version']}")
    if info['cudf_version']:
        print(f"cuDF Version: {info['cudf_version']}")
    print("=" * 50)


if __name__ == '__main__':
    # Test the backend
    print_backend_info()
    
    # Test DataFrame creation
    print("\nTesting DataFrame creation...")
    df = DataFrameBackend.create_dataframe({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"Created DataFrame type: {type(df)}")
    print(df)
    
    # Test conversion
    print("\nTesting conversion to pandas...")
    pdf = DataFrameBackend.to_pandas(df)
    print(f"Converted DataFrame type: {type(pdf)}")
    print(pdf)
