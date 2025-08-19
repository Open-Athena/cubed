#!/bin/bash
# Document known GPU/Zarr integration issues

cat > cupy_test_results.md << 'EOF'
## Known CuPy/Zarr Integration Issues

Current testing shows that CuPy arrays work with cubed's core functionality,
but there are compatibility issues with zarr's buffer handling when using
GPU-backed arrays. The error 'Implicit conversion to a NumPy array is not allowed'
occurs when zarr tries to slice CuPy arrays.

### Working features:
- Basic array creation and computation
- Array operations (when not involving zarr storage)
- Tests using single-threaded or threads executors
- Performance improvements for compute-bound operations

### Areas needing work:
- Integration with zarr GPU buffers (zarr.buffer.gpu.Buffer)
- Tests involving from_array with CuPy arrays
- Full compatibility with processes executor

### Test Results:
EOF

# Append actual test results if available
if [ -f pytest_results.txt ]; then
    echo "" >> cupy_test_results.md
    echo "#### Pytest Output:" >> cupy_test_results.md
    echo '```' >> cupy_test_results.md
    cat pytest_results.txt >> cupy_test_results.md
    echo '```' >> cupy_test_results.md
fi

echo "Documentation created in cupy_test_results.md"