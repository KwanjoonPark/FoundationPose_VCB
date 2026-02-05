# 1. 기존 베이스 이미지 사용
FROM wenbowen123/foundationpose

# 2. 패키지 설치 시 사용자 확인창 방지
ENV DEBIAN_FRONTEND=noninteractive

# 3. 필수 도구 및 Xvfb 설치 (렌더링 에러 해결용)
USER root
RUN apt-get update && apt-get install -y \
    xvfb \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# 4. Node.js 20 및 Gemini CLI 설치 (기존 동일)
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y nodejs \
    && npm install -g @google/gemini-cli

RUN npm install -g @anthropic-ai/claude-code

# 5. EGL 백엔드 설정 (headless offscreen 렌더링용)
ENV PYOPENGL_PLATFORM=egl

# 5-1. YOLOv8 (ultralytics) 설치
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate my && \
    pip install ultralytics>=8.4.0

# 5-2. Detectron2 설치 (Mask R-CNN용)
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate my && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Kaolin 재빌드 (sm_70 GPU 아키텍처 지원 - Quadro GV100 등)
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate my && \
    pip uninstall kaolin -y && \
    cd /tmp && rm -rf kaolin && \
    git clone --branch v0.15.0 --depth 1 https://github.com/NVIDIAGameWorks/kaolin.git && \
    cd kaolin && \
    TORCH_CUDA_ARCH_LIST="7.0" FORCE_CUDA=1 pip install . && \
    rm -rf /tmp/kaolin

WORKDIR /home/ebduser/FoundationPose

# 7. 유틸리티 스크립트 생성

# 7-1. LINEMOD 데이터셋 준비 스크립트 (mask_refined 심볼릭 링크 생성)
RUN printf '#!/bin/bash\n\
REF_VIEW_DIR="${1:-/home/ebduser/FoundationPose/linemod/ref_views}"\n\
if [ ! -d "$REF_VIEW_DIR" ]; then\n\
    echo "Error: Directory not found: $REF_VIEW_DIR"\n\
    echo "Usage: $0 [ref_view_dir]"\n\
    exit 1\n\
fi\n\
echo "Creating mask_refined symlinks in $REF_VIEW_DIR..."\n\
for dir in "$REF_VIEW_DIR"/ob_*/; do\n\
    if [ -d "${dir}mask" ] && [ ! -e "${dir}mask_refined" ]; then\n\
        ln -s "${dir}mask" "${dir}mask_refined"\n\
        echo "Created: ${dir}mask_refined"\n\
    fi\n\
done\n\
echo "Done!"' > /usr/local/bin/prepare_linemod.sh && chmod +x /usr/local/bin/prepare_linemod.sh

# 7-2. 익스텐션 빌드 스크립트 (컨테이너 시작 후 최초 1회 실행)
RUN printf '#!/bin/bash\n\
set -e\n\
FOUNDATIONPOSE_DIR="${1:-/home/ebduser/FoundationPose}"\n\
echo "=========================================="\n\
echo "FoundationPose Extensions Setup"\n\
echo "=========================================="\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate my\n\
# mycpp 빌드\n\
echo "[1/2] Building mycpp extension..."\n\
cd "$FOUNDATIONPOSE_DIR/mycpp"\n\
rm -rf build && mkdir build && cd build\n\
cmake .. && make -j$(nproc)\n\
# mycuda 빌드 (gridencoder 포함)\n\
echo "[2/2] Building mycuda extension..."\n\
cd "$FOUNDATIONPOSE_DIR/bundlesdf/mycuda"\n\
pip install -e .\n\
echo "=========================================="\n\
echo "All extensions built successfully!"\n\
echo "=========================================="' > /usr/local/bin/setup_extensions.sh && chmod +x /usr/local/bin/setup_extensions.sh