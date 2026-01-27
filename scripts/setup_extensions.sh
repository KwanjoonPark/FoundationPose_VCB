#!/bin/bash
# FoundationPose 익스텐션 빌드 스크립트
# 컨테이너 시작 후 최초 1회 실행 필요

set -e

FOUNDATIONPOSE_DIR="${1:-/home/ebduser/FoundationPose}"

echo "=========================================="
echo "FoundationPose Extensions Setup"
echo "=========================================="

# Conda 환경 활성화
source /opt/conda/etc/profile.d/conda.sh
conda activate my

# 1. mycpp 빌드 (C++ 익스텐션)
echo ""
echo "[1/2] Building mycpp extension..."
cd "$FOUNDATIONPOSE_DIR/mycpp"
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build
cmake ..
make -j$(nproc)
echo "mycpp build complete!"

# 2. mycuda 빌드 (CUDA 익스텐션 - gridencoder 포함)
echo ""
echo "[2/2] Building mycuda extension (gridencoder)..."
cd "$FOUNDATIONPOSE_DIR/bundlesdf/mycuda"
pip install -e .
echo "mycuda build complete!"

echo ""
echo "=========================================="
echo "All extensions built successfully!"
echo "=========================================="
echo ""
echo "Optional: Run 'prepare_linemod.sh' to prepare LINEMOD dataset"
