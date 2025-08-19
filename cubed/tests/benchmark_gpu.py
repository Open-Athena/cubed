#!/usr/bin/env python
"""Performance benchmarks comparing CPU and GPU backends for cubed."""

import os
import sys
import time
import importlib
import argparse

import cubed
import cubed.array_api as xp
import cubed.backend_array_api


def benchmark_backend(name, backend_module, instance_type="unknown"):
    """Run benchmarks for a specific backend."""
    print(f'\n{"="*60}')
    print(f'Testing {name}')
    print("="*60)

    os.environ['CUBED_BACKEND_ARRAY_API_MODULE'] = backend_module

    # Reload to pick up new backend
    importlib.reload(cubed.backend_array_api)

    spec = cubed.Spec('/tmp/bench', allowed_mem='4GB')

    results = []

    # Test 1: Array sum
    start = time.time()
    a = xp.arange(10_000_000, chunks=(1_000_000,), spec=spec)
    result = xp.sum(a).compute()
    elapsed = time.time() - start
    print(f'  Sum of 10M elements: {elapsed:.3f}s (result: {result})')
    results.append(('Sum 10M', elapsed))

    # Test 2: Array operations
    start = time.time()
    b = xp.ones((5000, 5000), chunks=(500, 500), spec=spec)
    c = b * 2 + 1
    result = c[0, 0].compute()
    elapsed = time.time() - start
    print(f'  Array operations (5000x5000): {elapsed:.3f}s (sample value: {result})')
    results.append(('Array ops 5000x5000', elapsed))

    # Test 3: Reductions
    start = time.time()
    d = xp.arange(1_000_000, chunks=(100_000,), spec=spec)
    mean_val = xp.mean(d).compute()
    elapsed = time.time() - start
    print(f'  Mean of 1M elements: {elapsed:.3f}s (result: {mean_val})')
    results.append(('Mean 1M', elapsed))

    # Test 4: Matrix multiplication (smaller size for reasonable runtime)
    start = time.time()
    e = xp.ones((2000, 2000), chunks=(500, 500), spec=spec)
    f = xp.ones((2000, 2000), chunks=(500, 500), spec=spec)
    g = xp.matmul(e, f)
    result = g[0, 0].compute()
    elapsed = time.time() - start
    print(f'  Matrix multiplication (2000x2000): {elapsed:.3f}s (sample value: {result})')
    results.append(('Matmul 2000x2000', elapsed))

    return results


def compare_backends(instance_type="unknown"):
    """Compare CPU and GPU backend performance."""
    print('Cubed GPU Backend Performance Comparison')
    print(f'Instance type: {instance_type}')

    all_results = {}

    # Test CPU (NumPy)
    try:
        cpu_results = benchmark_backend('NumPy (CPU)', 'numpy', instance_type)
        all_results['CPU'] = cpu_results
    except Exception as e:
        print(f'\nCPU test failed: {e}')

    # Test GPU (CuPy)
    try:
        import cupy
        print(f"\nCuPy version: {cupy.__version__}")
        gpu_results = benchmark_backend('CuPy (GPU)', 'array_api_compat.cupy', instance_type)
        all_results['GPU'] = gpu_results
    except ImportError:
        print('\nCuPy not available, skipping GPU tests')
    except Exception as e:
        print(f'\nGPU test failed: {e}')

    # Print comparison if both backends were tested
    if 'CPU' in all_results and 'GPU' in all_results:
        print(f'\n{"="*60}')
        print('Performance Comparison Summary')
        print("="*60)
        print(f'{"Test":<20} {"CPU (s)":<12} {"GPU (s)":<12} {"Speedup":<10}')
        print("-" * 54)

        for (cpu_test, cpu_time), (gpu_test, gpu_time) in zip(all_results['CPU'], all_results['GPU']):
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f'{cpu_test:<20} {cpu_time:<12.3f} {gpu_time:<12.3f} {speedup:<10.2f}x')

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark cubed GPU backends')
    parser.add_argument('--instance-type', default='unknown',
                        help='EC2 instance type for reporting')
    args = parser.parse_args()

    try:
        results = compare_backends(args.instance_type)
        # Return success if at least one backend was tested
        sys.exit(0 if results else 1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()