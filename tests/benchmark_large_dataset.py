"""
Large Dataset Benchmark for cuDF vs pandas

This script tests performance with larger datasets to demonstrate
GPU acceleration benefits.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.benchmark_dataframe_backend import DataFrameBenchmark

def main():
    print("=" * 60)
    print("Large Dataset Performance Benchmark")
    print("=" * 60)
    print()
    
    # Test with 1 million rows
    print("Testing with 1,000,000 rows (5 iterations)...")
    print("-" * 60)
    benchmark = DataFrameBenchmark(n_rows=1000000, n_iterations=5)
    results = benchmark.run_all_benchmarks()
    benchmark.print_comparison(results)
    
    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
