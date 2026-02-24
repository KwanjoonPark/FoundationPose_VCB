# FoundationPose VCB

NVIDIA [FoundationPose](https://nvlabs.github.io/FoundationPose/) (CVPR 2024 Highlight)의 포크로, Texture-less한 고반사 단색 금속이라는 성질이 있는 VCB(산업용 핸들) pose 추정에 특화된 확장 버전입니다. x86 GPU 서버에서 ZeroMQ 기반 클라이언트-서버 구조로 실시간 6DoF pose 추정 및 트래킹을 지원합니다.

**원본 논문:** [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/abs/2312.08344)

**주요 확장 기능:**
- Mask R-CNN / YOLO 기반 세그멘테이션
- Mask IoU 스코어링 (산업 현장 클러터 환경 대응)
- ZeroMQ 기반 실시간 클라이언트-서버 아키텍처
- ROS 통합 (`PoseStamped` 발행, TF 브로드캐스트)
- RGB 전용 모드 (금속/반사 표면 대응)
- Z축 보정 (벽면 장착 물체 대응)

> Jetson AGX Orin 환경은 [`jetson`](https://github.com/KwanjoonPark/FoundationPose_VCB/tree/jetson) 브랜치를 참조하세요.

---

## 사전 준비

### 1. 저장소 클론

```bash
git clone https://github.com/KwanjoonPark/FoundationPose_VCB.git
cd FoundationPose_VCB
```

### 2. 모델 가중치 다운로드

Google Drive ([Refiner/Scorer](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing), [R-CNN Mask](https://drive.google.com/drive/folders/1FEajd4v0Y7THdY5KOvsY_cdBPCKg70Ng?usp=drive_link)) 에서 네트워크 가중치를 다운로드하여 `weights/` 폴더에 배치합니다.

- Refiner: `weights/2023-10-28-18-33-37/`
- Scorer: `weights/2024-01-11-20-02-45/`
- R-CNN Mask: `weights/2026-02-12-13-41-52/`

### 3. Test Dataset 다운로드 (선택)

Google Drive ([test_scene](https://drive.google.com/drive/folders/1FEajd4v0Y7THdY5KOvsY_cdBPCKg70Ng?usp=drive_link)) 에서 테스트 데이터셋을 다운로드 하여 `vcb/ref_views/` 폴더에 배치합니다.

```
test_scene/
├── depth/
│   ├── 000000.png
│   └── 000001.png
├── rgb/
│   ├── 000000.png
│   └── 000001.png
└── cam_K.txt
```

### 4. Docker 컨테이너 실행

**Prerequisites:**
- CUDA 지원 x86 GPU (NVIDIA GPU)
- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)
- 베이스 이미지: `nvidia/cudagl:11.3.0-devel-ubuntu20.04`

```bash
# 이미지 빌드 (최초 1회)
cd docker && docker build --network host -t foundationpose .

# 컨테이너 실행
bash docker/run_container.sh

# C++/CUDA 익스텐션 빌드 (컨테이너 안에서 최초 1회)
bash build_all.sh
```

> 4090 등 최신 GPU는 커스텀 이미지를 사용합니다:
> ```bash
> docker pull shingarey/foundationpose_custom_cuda121:latest
> ```
> 이후 `docker/run_container.sh`에서 이미지 이름을 변경하세요. ([참고](https://github.com/NVlabs/FoundationPose/issues/27))

재접속 시:
```bash
docker exec -it foundationpose bash
```

### 5. 환경 설정 (Conda, 선택)

Docker 대신 Conda 환경을 사용할 수도 있습니다 (실험적):

```bash
conda create -n foundationpose python=3.9
conda activate foundationpose
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

---

## Test Dataset 으로 실행

### run_est.py — 오프라인 Pose 추정

저장된 이미지 시퀀스에 대해 pose 추정 수행:
```bash
python run_est.py --mask_model <mask_weights_path> --debug
```

**주요 옵션 (default):**
```bash
python run_est.py \
    --mesh_file vcb/ref_views/ob_000001/model/model_vc.ply \
    --mask_model weights/2026-02-12-13-41-52/model_best.pth \
    --mask_type maskrcnn \
    --mask_conf 0.5 \
    --input_mode rgb \
    --symmetry z180
```

---

## 실시간 실행 (ZeroMQ 클라이언트-서버)

`realtime/` 디렉토리는 ZeroMQ 기반 클라이언트-서버 구조로, GPU 서버와 로컬 PC를 분리하여 실시간 pose 추정을 수행합니다.

```
┌─────────────────────────┐       ZeroMQ (TCP)       ┌─────────────────────────┐
│      Local PC           │                           │      GPU Server         │
│                         │      Image (JPEG)         │                         │
│  ┌─────────────────┐    │  ──────────────────────►  │  ┌─────────────────┐    │
│  │ RealSense D435  │    │                           │  │  FoundationPose │    │
│  └─────────────────┘    │      Pose (4x4 matrix)    │  │  + Mask R-CNN   │    │
│                         │  ◄──────────────────────  │  └─────────────────┘    │
│  ┌─────────────────┐    │                           │                         │
│  │  Visualization  │    │                           │  NVIDIA GPU (CUDA)      │
│  └─────────────────┘    │                           │                         │
└─────────────────────────┘                           └─────────────────────────┘
       client.py                                            server.py
```

### server.py — GPU 서버

GPU 서버에서 FoundationPose + 마스크 모델을 로드하고 클라이언트 요청을 처리합니다.

```bash
python realtime/server.py --port 5555 \
    --mesh_file vcb/ref_views/ob_000001/model/model_vc.ply \
    --mask_model vcb/rcnn360.pth
```

**서버 옵션:**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--port` | 5555 | ZeroMQ 포트 |
| `--mesh_file` | model_vc.ply | 3D 메시 파일 경로 |
| `--mesh_scale` | 0.01 | 메시 스케일 (cm→m) |
| `--mask_model` | rcnn360.pth | 마스크 모델 경로 |
| `--mask_type` | maskrcnn | 마스크 모델 타입 (maskrcnn/yolo) |
| `--input_mode` | rgb | 입력 모드 (rgb/rgbd) |
| `--symmetry` | z180 | 대칭 타입 |
| `--debug` | False | 디버그 이미지 저장 |

### client.py — 로컬 PC 클라이언트

로컬 PC에서 카메라 이미지를 캡처하여 GPU 서버로 전송합니다.

```bash
# RealSense 카메라 사용
python realtime/client.py --server_ip <GPU_SERVER_IP> --port 5555

# 웹캠 사용 (depth 없음)
python realtime/client.py --server_ip <GPU_SERVER_IP> --webcam

# 단일 이미지 테스트
python realtime/client.py --server_ip <GPU_SERVER_IP> --test_image ./test.png

# 비디오 파일 처리
python realtime/client.py --server_ip <GPU_SERVER_IP> --video ./test.mp4
```

**클라이언트 옵션:**

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--server_ip` | localhost | 서버 IP 주소 |
| `--port` | 5555 | ZeroMQ 포트 |
| `--width` | 640 | 이미지 너비 |
| `--height` | 480 | 이미지 높이 |
| `--fps` | 30 | 카메라 FPS |
| `--jpeg_quality` | 80 | JPEG 압축 품질 |
| `--realsense` | True | RealSense 사용 |
| `--webcam` | False | 웹캠 사용 |
| `--test_image` | None | 테스트 이미지 경로 |
| `--video` | None | 비디오 파일 경로 |
| `--no_vis` | False | 시각화 비활성화 |
| `--save` | False | 결과 저장 |

**클라이언트 의존성 (로컬 PC):**
```bash
pip install -r realtime/requirements_client.txt
# RealSense 사용 시: pip install pyrealsense2
```

### test_local.py — 로컬 테스트

카메라 없이 저장된 테스트 데이터로 서버를 테스트합니다.

```bash
# 서버를 먼저 실행한 후
python realtime/test_local.py --server_ip localhost --test_dir vcb/ref_views/test_scene
```

### 예상 지연시간

| 구간 | 시간 |
|---|---|
| 이미지 압축 (클라이언트) | ~5ms |
| 네트워크 전송 | ~10-30ms |
| 마스크 생성 | ~20-30ms |
| Pose 추정 | ~100-150ms |
| **합계** | **~150-200ms** |

### 성능 최적화 팁

- JPEG 품질 낮추기: `--jpeg_quality 60`
- 해상도 낮추기: `--width 480 --height 360`
- RGB 모드 사용: `--input_mode rgb` (depth 전송 생략)
- 유선 네트워크 사용

---

## ROS 통합

### ros_client.py — ROS 노드

ROS 토픽에서 이미지를 수신하여 GPU 서버로 전송하고 결과를 발행합니다.

```bash
# rosrun
rosrun foundation_pose ros_client.py _server_ip:=<GPU_SERVER_IP> _port:=5555

# roslaunch (RealSense 함께 실행)
roslaunch foundation_pose foundation_pose.launch \
    server_ip:=<GPU_SERVER_IP> \
    launch_realsense:=true

# roslaunch (RealSense 별도 실행)
roslaunch realsense2_camera rs_camera.launch align_depth:=true
roslaunch foundation_pose foundation_pose.launch server_ip:=<GPU_SERVER_IP>
```

### ROS 토픽

**Subscribe (입력):**

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/camera/color/image_raw` | `Image` | RGB 이미지 |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | Depth (선택) |
| `/camera/color/camera_info` | `CameraInfo` | 카메라 정보 |

**Publish (출력):**

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/foundation_pose/pose` | `PoseStamped` | 6DoF pose (position + quaternion) |
| `/foundation_pose/pose_array` | `PoseArray` | RViz 시각화용 |
| `/foundation_pose/visualization` | `Image` | 결과 오버레이 이미지 |
| `/foundation_pose/status` | `String` | 상태 메시지 |

**TF Broadcast:** `camera_color_optical_frame` → `object`

### ROS Parameters

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `~server_ip` | localhost | GPU 서버 IP |
| `~port` | 5555 | ZeroMQ 포트 |
| `~rate_limit` | 10.0 | 최대 처리 속도 (Hz) |
| `~jpeg_quality` | 80 | JPEG 압축 품질 |
| `~publish_tf` | true | TF 발행 여부 |
| `~frame_id` | camera_color_optical_frame | 카메라 프레임 |
| `~object_frame_id` | object | 객체 프레임 |

### 네트워크 설정

방화벽 설정 (GPU Server):
```bash
# Ubuntu/Debian
sudo ufw allow 5555/tcp
```

SSH 터널링 (방화벽 우회):
```bash
ssh -L 5555:localhost:5555 user@gpu-server
python realtime/client.py --server_ip localhost --port 5555
```

---

## 옵션 설명

### 메시 파일 (`--mesh_file`)

| 파일 | 설명 |
|---|---|
| `model.obj` | 원본 메시 (텍스처 매핑) |
| `model_vc.ply` | Vertex color 메시 |
| `model_vc_final.ply` | 최종 vertex color 메시 |

경로: `vcb/ref_views/ob_000001/model/`

### 입력 모드 (`--input_mode`)

| 모드 | 설명 |
|---|---|
| `rgb` | RGB만 사용 (기본값). 금속/반사 표면에 적합 |
| `rgbd` | RGB + Depth. 텍스처가 있는 물체에 유리 |

### 대칭 (`--symmetry`)

VCB 핸들은 원통형이므로 `z180` 사용 (Z축 기준 180도 대칭).

### Z축 보정 (`--fix_z_axis`)

벽면 장착 물체용. 물체 Z축이 항상 카메라를 향하도록 보정합니다.
- 가설 단계: 뒷면 후보 ~50% 제거 (`front_hemisphere_only`)
- 추정 후: Z축 방향 재확인 + 필요 시 FLIP_X 적용

---

## Troubleshooting

- GPU 4090 등 최신 GPU 설정: [Issue #27](https://github.com/NVlabs/FoundationPose/issues/27)
- Windows 환경 설정: [Issue #148](https://github.com/NVlabs/FoundationPose/issues/148)
- 비정상적인 결과가 나올 경우: [Issue #44](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043), [설정 가이드](https://github.com/030422Lee/FoundationPose_manual)

---

## 인용 (Citation)

```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

---

## 라이선스

코드와 데이터는 [NVIDIA Source Code License](LICENSE) 하에 배포됩니다. Copyright © 2024, NVIDIA Corporation. All rights reserved.

## 연락처

원본 FoundationPose 관련 문의: [Bowen Wen](https://wenbowen123.github.io/)
