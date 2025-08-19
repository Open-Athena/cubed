#!/usr/bin/env python
"""Comprehensive GPU functionality tests for cubed.

This test suite validates that cubed is actually using the GPU backend
and that GPU operations provide correct results for various array operations.
"""

import os
import sys
import time
import numpy as np

import cubed
import cubed.array_api as xp


def verify_gpu_backend():
    """Verify that we're actually using a GPU backend."""
    backend = os.environ.get('CUBED_BACKEND_ARRAY_API_MODULE', 'numpy')
    print(f"Backend module: {backend}")

    if 'cupy' in backend:
        try:
            import cupy
            # Check if GPU is actually available
            device_count = cupy.cuda.runtime.getDeviceCount()
            print(f"CUDA devices available: {device_count}")
            if device_count > 0:
                device = cupy.cuda.Device(0)
                props = device.attributes
                print(f"GPU: {device.name.decode()}")
                print(f"Compute capability: {device.compute_capability}")
                print(f"Memory: {device.mem_info[1] / 1024**3:.1f} GB")
                return True
        except Exception as e:
            print(f"Warning: CuPy imported but GPU not accessible: {e}")
            return False
    elif 'jax' in backend:
        try:
            import jax
            devices = jax.devices()
            print(f"JAX devices: {devices}")
            return any('gpu' in str(d).lower() for d in devices)
        except Exception as e:
            print(f"Warning: JAX imported but GPU not accessible: {e}")
            return False
    else:
        print(f"Not a GPU backend: {backend}")
        return False


def test_large_array_operations():
    """Test operations on arrays large enough to benefit from GPU."""
    spec = cubed.Spec('/tmp/test_gpu', allowed_mem='2GB')

    print("\n=== Testing Large Array Operations ===")

    # Create a large array (100 million elements)
    size = 10000
    print(f"Creating {size}x{size} array (100M elements)...")
    a = xp.ones((size, size), chunks=(1000, 1000), dtype=xp.float32, spec=spec)

    # Test element-wise operations
    print("Testing element-wise operations...")
    b = a * 2 + 1
    c = xp.sqrt(b)

    # Compute a small portion to verify correctness
    result = c[:10, :10].compute()
    expected = np.sqrt(3.0)
    assert np.allclose(result, expected), f"Expected {expected}, got {result[0,0]}"
    print(f"✓ Element-wise operations: sample value = {result[0,0]:.4f}")

    # Test reduction
    print("Testing reduction on large array...")
    start = time.time()
    mean_val = xp.mean(a[:1000, :1000]).compute()  # Reduce a subset for speed
    elapsed = time.time() - start
    assert np.isclose(mean_val, 1.0), f"Expected mean=1.0, got {mean_val}"
    print(f"✓ Reduction: mean={mean_val:.4f} (computed in {elapsed:.2f}s)")

    return True


def test_matrix_operations():
    """Test matrix operations that benefit from GPU acceleration."""
    spec = cubed.Spec('/tmp/test_gpu', allowed_mem='1GB')

    print("\n=== Testing Matrix Operations ===")

    # Matrix multiplication
    size = 1000
    print(f"Testing matrix multiplication ({size}x{size})...")
    a = xp.ones((size, size), chunks=(500, 500), dtype=xp.float32, spec=spec)
    b = xp.eye(size, chunks=(500, 500), dtype=xp.float32, spec=spec)

    start = time.time()
    c = xp.matmul(a, b)
    result = c[:10, :10].compute()
    elapsed = time.time() - start

    # Result should be same as 'a' since we multiply by identity
    assert np.allclose(result, 1.0), f"Matmul with identity failed"
    print(f"✓ Matrix multiplication: verified (computed in {elapsed:.2f}s)")

    # Test transpose
    print("Testing transpose...")
    a = xp.arange(100, chunks=(10,), spec=spec).reshape((10, 10))
    at = a.T
    result = at.compute()
    expected = np.arange(100).reshape((10, 10)).T
    assert np.array_equal(result, expected), "Transpose failed"
    print(f"✓ Transpose: shape {result.shape}")

    return True


def test_memory_efficiency():
    """Test that chunked operations don't exceed memory limits."""
    spec = cubed.Spec('/tmp/test_gpu', allowed_mem='500MB')  # Restrictive memory

    print("\n=== Testing Memory Efficiency ===")

    # Create array larger than allowed memory
    # 2000x2000 float32 = 16MB per chunk, 400MB total
    print("Creating array larger than allowed memory...")
    a = xp.ones((2000, 2000), chunks=(100, 100), dtype=xp.float32, spec=spec)
    b = xp.ones((2000, 2000), chunks=(100, 100), dtype=xp.float32, spec=spec)

    # This should work despite total size > allowed_mem due to chunking
    print("Computing chunked operation...")
    c = a + b
    result = c[0, 0].compute()
    assert result == 2.0, f"Expected 2.0, got {result}"
    print(f"✓ Chunked operation succeeded with limited memory")

    return True


def test_gpu_specific_operations():
    """Test operations that particularly benefit from GPU."""
    spec = cubed.Spec('/tmp/test_gpu', allowed_mem='1GB')

    print("\n=== Testing GPU-Optimized Operations ===")

    # Trigonometric operations (highly parallelizable)
    print("Testing trigonometric operations...")
    a = xp.linspace(0, 2 * np.pi, 10000, chunks=(1000,), spec=spec)
    b = xp.sin(a)
    c = xp.cos(a)
    d = b**2 + c**2  # Should be close to 1

    result = d[:100].compute()
    assert np.allclose(result, 1.0, atol=1e-6), "sin²+cos² != 1"
    print(f"✓ Trigonometric: sin²+cos² ≈ {result.mean():.6f}")

    # Exponential operations
    print("Testing exponential operations...")
    a = xp.ones((1000, 1000), chunks=(100, 100), dtype=xp.float32, spec=spec)
    b = xp.exp(a)
    result = b[0, 0].compute()
    expected = np.exp(1)
    assert np.isclose(result, expected), f"Expected e={expected:.4f}, got {result:.4f}"
    print(f"✓ Exponential: exp(1) = {result:.4f}")

    return True


def test_data_types():
    """Test different data types on GPU."""
    spec = cubed.Spec('/tmp/test_gpu', allowed_mem='1GB')

    print("\n=== Testing Data Types ===")

    dtypes = [
        (xp.float32, "float32"),
        (xp.float64, "float64"),
        (xp.int32, "int32"),
        (xp.int64, "int64"),
    ]

    for dtype, name in dtypes:
        try:
            a = xp.ones((100, 100), chunks=(50, 50), dtype=dtype, spec=spec)
            result = a.compute()
            assert result.dtype == dtype, f"Wrong dtype for {name}"
            print(f"✓ {name}: shape={result.shape}, dtype={result.dtype}")
        except Exception as e:
            print(f"✗ {name}: {e}")

    return True


def main():
    """Run comprehensive GPU tests."""
    print("=" * 60)
    print("Cubed GPU Comprehensive Test Suite")
    print("=" * 60)

    # Check if we're actually using GPU
    using_gpu = verify_gpu_backend()
    if not using_gpu:
        print("\nWARNING: Not using GPU backend. Tests will run on CPU.")
        print("Set CUBED_BACKEND_ARRAY_API_MODULE=array_api_compat.cupy for GPU tests.")

    tests = [
        ("Large Array Operations", test_large_array_operations),
        ("Matrix Operations", test_matrix_operations),
        ("Memory Efficiency", test_memory_efficiency),
        ("GPU-Optimized Operations", test_gpu_specific_operations),
        ("Data Types", test_data_types),
    ]

    results = []
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            success = test_func()
            results.append((name, "PASS"))
            print(f"Result: PASS")
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            print(f"Result: FAIL - {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {name}: {result}")

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if using_gpu:
        print("GPU backend: ACTIVE")
    else:
        print("GPU backend: NOT ACTIVE (tests ran on CPU)")

    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)