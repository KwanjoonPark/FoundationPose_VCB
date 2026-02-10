#!/bin/bash
# Build C++ and CUDA extensions inside the Jetson container
# Run this on first container startup (or after code changes)
# Skips kaolin build (not needed for model-based pipeline)
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building mycpp (C++ extensions) ==="
cd "$DIR/mycpp"
mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE="$(which python3)"
make -j$(nproc)

echo "=== Building mycuda (CUDA grid encoder) ==="
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.7}"
cd "$DIR/bundlesdf/mycuda"
rm -rf build *.egg-info
pip3 install -e .

echo "=== Extensions built successfully ==="
cd "$DIR"
