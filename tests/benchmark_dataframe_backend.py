"""
DataFrame Backend Performance Benchmark

This script benchmarks the performance of pandas vs cuDF backends
for common operations used in the HourBreakout1 strategy.

Requirements: TASK-001 - CuDF acceleration experiment
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parallel_backtest.dataframe_backend import DataFrameBackend, get_backend_info, print_backend_info


class DataFrameBenchmark:
    """Benchmark DataFrame operations"""
    
    def __init__(self, n_rows: int = 100000, n_iterations: int = 10):
        """
        Initialize benchmark
        
        Args:
            n_rows: Number of rows in test DataFrame
            n_iterations: Number of iterations for each test
        """
        self.n_rows = n_rows
        self.n_iterations = n_iterations
        self.results = {}
    
    def generate_ohlcv_data(self) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 50000.0
        price_changes = np.random.randn(self.n_rows) * 100
        close_prices = base_price + np.cumsum(price_changes)
        
        data = {
            'date': pd.date_range('2024-01-01', periods=self.n_rows, freq='1min'),
            'open': close_prices + np.random.randn(self.n_rows) * 10,
            'high': close_prices + np.abs(np.random.randn(self.n_rows) * 20),
            'low': close_prices - np.abs(np.random.randn(self.n_rows) * 20),
            'close': close_prices,
            'volume': np.random.randint(100, 10000, self.n_rows).astype(float),
        }
        
        return pd.DataFrame(data)
    
    def time_operation(self, operation: Callable, name: str) -> float:
        """
        Time a DataFrame operation
        
        Args:
            operation: Function to time
            name: Name of the operation
            
        Returns:
            Average execution time in seconds
        """
        times = []
        
        for i in range(self.n_iterations):
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  {name}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        
        return avg_time
    
    def benchmark_dataframe_creation(self, backend: str) -> float:
        """Benchmark DataFrame creation"""
        DataFrameBackend.initialize(backend)
        pdf = self.generate_ohlcv_data()
        
        def operation():
            df = DataFrameBackend.to_cudf(pdf) if backend == 'cudf' else pdf.copy()
            return df
        
        return self.time_operation(operation, "DataFrame Creation")
    
    def benchmark_rolling_mean(self, backend: str) -> float:
        """Benchmark rolling mean calculation (MA indicator)"""
        DataFrameBackend.initialize(backend)
        pdf = self.generate_ohlcv_data()
        df = DataFrameBackend.to_cudf(pdf) if backend == 'cudf' else pdf
        
        def operation():
            result = df['close'].rolling(window=5).mean()
            return result
        
        return self.time_operation(operation, "Rolling Mean (MA5)")
    
    def benchmark_shift_operation(self, backend: str) -> float:
        """Benchmark shift operation"""
        DataFrameBackend.initialize(backend)
        pdf = self.generate_ohlcv_data()
        df = DataFrameBackend.to_cudf(pdf) if backend == 'cudf' else pdf
        
        def operation():
            result = df['high'].shift(1)
            return result
        
        return self.time_operation(operation, "Shift Operation")
    
    def benchmark_conditional_filtering(self, backend: str) -> float:
        """Benchmark conditional filtering"""
        DataFrameBackend.initialize(backend)
        pdf = self.generate_ohlcv_data()
        df = DataFrameBackend.to_cudf(pdf) if backend == 'cudf' else pdf
        df['ma5'] = df['close'].rolling(window=5).mean()
        
        def operation():
            result = df[df['close'] > df['ma5']]
            return result
        
        return self.time_operation(operation, "Conditional Filtering")
    
    def benchmark_column_operations(self, backend: str) -> float:
        """Benchmark column arithmetic operations"""
        DataFrameBackend.initialize(backend)
        pdf = self.generate_ohlcv_data()
        df = DataFrameBackend.to_cudf(pdf) if backend == 'cudf' else pdf
        
        def operation():
            df['range'] = df['high'] - df['low']
            df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100
            return df
        
        return self.time_operation(operation, "Column Operations")
    
    def benchmark_merge_operation(self, backend: str) -> float:
        """Benchmark DataFrame merge"""
        DataFrameBackend.initialize(backend)
        pdf1 = self.generate_ohlcv_data()
        pdf2 = pdf1[['date', 'high']].copy()
        pdf2.columns = ['date', 'high_1h']
        
        df1 = DataFrameBackend.to_cudf(pdf1) if backend == 'cudf' else pdf1
        df2 = DataFrameBackend.to_cudf(pdf2) if backend == 'cudf' else pdf2
        
        def operation():
            result = DataFrameBackend.merge(df1, df2, on='date', how='left')
            return result
        
        return self.time_operation(operation, "DataFrame Merge")
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """
        Run all benchmarks for both backends
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'pandas': {},
            'cudf': {} if DataFrameBackend.is_cudf_available() else None
        }
        
        benchmarks = [
            ('DataFrame Creation', self.benchmark_dataframe_creation),
            ('Rolling Mean', self.benchmark_rolling_mean),
            ('Shift Operation', self.benchmark_shift_operation),
            ('Conditional Filtering', self.benchmark_conditional_filtering),
            ('Column Operations', self.benchmark_column_operations),
            ('Merge Operation', self.benchmark_merge_operation),
        ]
        
        # Benchmark pandas
        print("\n" + "=" * 60)
        print("Benchmarking pandas backend")
        print("=" * 60)
        for name, benchmark_func in benchmarks:
            results['pandas'][name] = benchmark_func('pandas')
        
        # Benchmark cuDF if available
        if DataFrameBackend.is_cudf_available():
            print("\n" + "=" * 60)
            print("Benchmarking cuDF backend")
            print("=" * 60)
            for name, benchmark_func in benchmarks:
                results['cudf'][name] = benchmark_func('cudf')
        else:
            print("\n" + "=" * 60)
            print("cuDF not available - skipping cuDF benchmarks")
            print("=" * 60)
        
        return results
    
    def print_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print comparison of benchmark results"""
        print("\n" + "=" * 60)
        print("Performance Comparison")
        print("=" * 60)
        print(f"{'Operation':<30} {'pandas (ms)':<15} {'cuDF (ms)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        if results['cudf'] is None:
            print("cuDF not available - no comparison possible")
            return
        
        for operation in results['pandas'].keys():
            pandas_time = results['pandas'][operation] * 1000
            cudf_time = results['cudf'][operation] * 1000
            speedup = pandas_time / cudf_time if cudf_time > 0 else 0
            
            print(f"{operation:<30} {pandas_time:>10.2f}     {cudf_time:>10.2f}     {speedup:>6.2f}x")
        
        # Calculate average speedup
        if results['cudf']:
            speedups = [
                results['pandas'][op] / results['cudf'][op]
                for op in results['pandas'].keys()
                if results['cudf'][op] > 0
            ]
            avg_speedup = np.mean(speedups)
            print("-" * 60)
            print(f"{'Average Speedup':<30} {'':<15} {'':<15} {avg_speedup:>6.2f}x")


def main():
    """Main benchmark function"""
    print_backend_info()
    
    # Configuration
    n_rows = 100000  # Simulate 100k candles (~69 days of 1m data)
    n_iterations = 10
    
    print(f"\nBenchmark Configuration:")
    print(f"  Rows: {n_rows:,}")
    print(f"  Iterations: {n_iterations}")
    
    # Run benchmarks
    benchmark = DataFrameBenchmark(n_rows=n_rows, n_iterations=n_iterations)
    results = benchmark.run_all_benchmarks()
    
    # Print comparison
    benchmark.print_comparison(results)
    
    # Save results
    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
