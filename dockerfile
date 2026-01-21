# 1. 원본 이미지를 기반으로 시작
FROM wenbowen123/foundationpose

# 2. 패키지 설치 시 사용자 확인창 방지 및 환경 설정
ENV DEBIAN_FRONTEND=noninteractive

# 3. 최신 Node.js 20 및 Gemini CLI 설치
USER root

# 기존 노드 관련 패키지 정리 및 필수 도구(curl) 설치
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

# NodeSource에서 Node.js 20 버전 레포지토리 등록 및 설치
RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y nodejs

# Gemini CLI 설치 (이제 최신 노드 버전이므로 에러 없이 설치됩니다)
RUN npm install -g @google/gemini-cli

# 4. 작업 디렉토리 설정
WORKDIR /home/ebduser/FoundationPose