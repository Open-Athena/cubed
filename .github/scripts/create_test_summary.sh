#!/bin/bash
# Create comprehensive test summary for GPU tests

INSTANCE_TYPE="${1:-unknown}"

cat > test_summary.md << EOF
## GPU Test Summary

**Date:** $(date)
**Instance:** ${INSTANCE_TYPE}

### GPU Information
EOF

# Add GPU info if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo '```' >> test_summary.md
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv >> test_summary.md
    echo '```' >> test_summary.md
else
    echo "GPU information not available (nvidia-smi not found)" >> test_summary.md
fi

echo "" >> test_summary.md

# Add performance results if available
if [ -f performance_results.txt ]; then
    echo "### Performance Results" >> test_summary.md
    echo '```' >> test_summary.md
    cat performance_results.txt >> test_summary.md
    echo '```' >> test_summary.md
    echo "" >> test_summary.md
fi

# Add CuPy test results if available
if [ -f cupy_test_results.md ]; then
    cat cupy_test_results.md >> test_summary.md
    echo "" >> test_summary.md
fi

# Add basic test results if available
if [ -f basic_test_results.txt ]; then
    echo "### Basic Functionality Tests" >> test_summary.md
    echo '```' >> test_summary.md
    cat basic_test_results.txt >> test_summary.md
    echo '```' >> test_summary.md
fi

echo "Test summary created in test_summary.md"