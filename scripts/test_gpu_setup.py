#!/usr/bin/env python3
"""
GPU Setup Verification Script

This script verifies that the GPU environment is correctly set up
for RAPIDS cuDF acceleration.

Usage:
    python scripts/test_gpu_setup.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_gpu_availability():
    """Check if GPU is available"""
    print_section("GPU Availability Check")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ GPU detected")
            print("\nGPU Information:")
            # Get GPU name and memory
            gpu_info = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            print(gpu_info.stdout)
            return True
        else:
            print("✗ GPU not detected")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        return False

def check_cuda_availability():
    """Check CUDA availability"""
    print_section("CUDA Availability Check")
    
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA Toolkit detected")
            # Extract version
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ CUDA Toolkit not found")
            return False
    except FileNotFoundError:
        print("✗ nvcc not found (CUDA Toolkit may not be installed)")
        return False

def check_python_packages():
    """Check required Python packages"""
    print_section("Python Packages Check")
    
    packages = {
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'cudf': 'cuDF (RAPIDS)',
        'freqtrade': 'Freqtrade',
    }
    
    results = {}
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
            results[package] = True
        except ImportError:
            print(f"✗ {name}: Not installed")
            results[package] = False
    
    return results

def check_dataframe_backend():
    """Check DataFrame backend"""
    print_section("DataFrame Backend Check")
    
    try:
        from parallel_backtest.dataframe_backend import get_backend_info, print_backend_info
        
        print_backend_info()
        
        info = get_backend_info()
        return info['using_cudf']
    except Exception as e:
        print(f"✗ Error checking backend: {e}")
        return False

def run_simple_cudf_test():
    """Run a simple cuDF test"""
    print_section("cuDF Functionality Test")
    
    try:
        import cudf
        import pandas as pd
        import time
        
        # Create test data
        n_rows = 100000
        print(f"Creating test DataFrame with {n_rows:,} rows...")
        
        # Test with pandas
        start = time.perf_counter()
        pdf = pd.DataFrame({
            'a': range(n_rows),
            'b': range(n_rows, n_rows * 2)
        })
        pdf['c'] = pdf['a'] + pdf['b']
        pandas_time = time.perf_counter() - start
        
        # Test with cuDF
        start = time.perf_counter()
        gdf = cudf.DataFrame({
            'a': range(n_rows),
            'b': range(n_rows, n_rows * 2)
        })
        gdf['c'] = gdf['a'] + gdf['b']
        cudf_time = time.perf_counter() - start
        
        print(f"✓ cuDF test successful")
        print(f"  Pandas time: {pandas_time*1000:.2f}ms")
        print(f"  cuDF time: {cudf_time*1000:.2f}ms")
        print(f"  Speedup: {pandas_time/cudf_time:.2f}x")
        
        return True
    except Exception as e:
        print(f"✗ cuDF test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("GPU Setup Verification")
    print("=" * 60)
    
    results = {
        'gpu': check_gpu_availability(),
        'cuda': check_cuda_availability(),
    }
    
    package_results = check_python_packages()
    results.update(package_results)
    
    results['backend'] = check_dataframe_backend()
    
    if results.get('cudf', False):
        results['cudf_test'] = run_simple_cudf_test()
    
    # Summary
    print_section("Verification Summary")
    
    all_passed = True
    critical_checks = ['gpu', 'cudf', 'backend']
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
        if check in critical_checks and not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All critical checks passed!")
        print("Your environment is ready for GPU-accelerated backtesting.")
        print("\nNext steps:")
        print("  1. Run benchmark: python tests/benchmark_dataframe_backend.py")
        print("  2. Test with strategy: python -m parallel_backtest --config configs/HourBreakout1.json --strategy HourBreakout1")
    else:
        print("✗ Some critical checks failed.")
        print("Please review the errors above and fix the issues.")
        print("\nCommon solutions:")
        print("  - Ensure NVIDIA drivers are installed on Windows host")
        print("  - Install CUDA Toolkit in WSL2")
        print("  - Install RAPIDS cuDF: conda install -c rapidsai cudf")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
