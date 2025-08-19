#!/usr/bin/env python
"""Basic GPU functionality tests for cubed with CuPy backend."""

import sys
import cubed
import cubed.array_api as xp


def test_basic_functionality():
    """Test basic array creation and computation with GPU backend."""
    spec = cubed.Spec('/tmp/test', allowed_mem='1GB')

    print("Testing cubed with GPU backend...")

    # Test basic array creation and computation
    a = xp.ones((100, 100), chunks=(50, 50), spec=spec)
    result = a.compute()
    assert result.shape == (100, 100)
    assert result.sum() == 10000
    print(f'✓ Basic array creation: shape={result.shape}, sum={result.sum()}')

    # Test array operations
    b = xp.arange(1000, chunks=(100,), spec=spec)
    result = xp.sum(b).compute()
    assert result == 499500
    print(f'✓ Array operations: sum of arange(1000)={result}')

    # Test chunked operations
    c = xp.ones((1000, 1000), chunks=(100, 100), spec=spec)
    d = c + 1
    result = d.compute()
    assert result.shape == (1000, 1000)
    assert result.mean() == 2.0
    print(f'✓ Chunked operations: shape={result.shape}, mean={result.mean()}')

    # Test reductions
    e = xp.arange(100, chunks=(10,), spec=spec)
    mean_val = xp.mean(e).compute()
    max_val = xp.max(e).compute()
    assert mean_val == 49.5
    assert max_val == 99
    print(f'✓ Reductions: mean={mean_val}, max={max_val}')

    print("\nAll basic GPU tests passed!")
    return True


if __name__ == "__main__":
    try:
        import cupy
        print(f"CuPy version: {cupy.__version__}")

        # Verify GPU is available
        device = cupy.cuda.Device(0)
        print(f"GPU device: {device}")
        print(f"Compute capability: {device.compute_capability}")

        success = test_basic_functionality()
        sys.exit(0 if success else 1)

    except ImportError as e:
        print(f"Error: CuPy not available - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)