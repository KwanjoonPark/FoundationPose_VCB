# FoundationPose VCB — 사용 설명서

## 사전 준비

### 1. Docker 컨테이너 실행

```bash
# 이미지 빌드 (최초 1회, ~2-3.5시간)
docker build -f docker/dockerfile.jetson -t foundationpose-jetson docker/

# 컨테이너 실행
bash docker/run_container_jetson.sh

# C++/CUDA 익스텐션 빌드 (컨테이너 안에서 최초 1회)
bash docker/build_extensions_jetson.sh
```

재접속 시:
```bash
docker exec -it foundationpose-jetson bash
```

### 2. RealSense 카메라 실행 (호스트에서)

카메라는 컨테이너가 아닌 **호스트**에서 실행합니다. 컨테이너는 `--network=host`로 ROS 토픽에 접근합니다.

```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

### 3. 모델 가중치

`weights/` 폴더에 다음 디렉토리가 있어야 합니다:
- Refiner: `weights/2023-10-28-18-33-37/`
- Scorer: `weights/2024-01-11-20-02-45/`

---

## camera/ 도구

모든 도구는 컨테이너 안에서 실행합니다.

### pose_estimator.py — 인터랙티브 Pose 추정

카메라 피드를 보면서 수동으로 pose 추정을 트리거합니다.

```bash
python camera/pose_estimator.py
```

**키 조작:**
| 키 | 기능 |
|---|---|
| `p` / Space | Pose 추정 (단일) |
| `t` | 트래킹 모드 ON/OFF (연속 추정) |
| `r` | 리셋 (트래킹 초기화) |
| `s` | 현재 프레임 + 결과 저장 |
| `q` | 종료 |

**주요 옵션:**
```bash
python camera/pose_estimator.py \
    --mesh_file vcb/ref_views/ob_000001/model/model.obj \
    --mask_model vcb/rcnn500.pth \
    --mask_type maskrcnn \
    --mask_conf 0.9 \
    --input_mode rgb \
    --symmetry z180 \
    --use_light True
```

### pose_streamer.py — 연속 자동 스트리머

키 입력 없이 매 프레임 자동으로 pose를 추정하여 ROS 토픽에 발행합니다.

```bash
# GUI 모드
python camera/pose_streamer.py

# Headless 모드 (GUI 없이 ROS만 발행, 배포용)
python camera/pose_streamer.py --headless
```

**키 조작 (GUI 모드):**
| 키 | 기능 |
|---|---|
| `q` | 종료 |

### pose_debugger.py — 디버그 시각화

FoundationPose 내부 시각화(scorer/refiner 후보 렌더링)를 별도 창으로 보여주고 파일로 저장합니다.

```bash
python camera/pose_debugger.py
```

ROS publisher 없이 디버깅 전용. `camera/debug/{session}/` 에 프레임별 결과 저장.

키 조작은 pose_estimator.py와 동일합니다.

### raw_images_collector.py — 프레임 캡처

RGB + Depth 이미지를 파일로 저장합니다 (학습 데이터 수집용).

```bash
python camera/raw_images_collector.py --output captured_frames --frames 50
```

**키 조작:**
| 키 | 기능 |
|---|---|
| `s` / Space | 한 장 캡처 |
| `b` | 연속 녹화 시작 |
| `e` | 연속 녹화 종료 |
| `q` | 종료 |

---

## ROS 토픽

pose_estimator, pose_streamer가 발행하는 토픽:

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/foundation_pose/pose` | `PoseStamped` | 6DoF pose (position + quaternion) |
| `/foundation_pose/result` | `String` (JSON) | pose, euler 각도, confidence 포함 |

JSON 예시:
```json
{
  "object_found": true,
  "pose_6d": {
    "translation": {"x": 0.05, "y": -0.02, "z": 0.35},
    "rotation_euler_deg": {"roll": 2.1, "pitch": 15.3, "yaw": -3.7},
    "rotation_quaternion": {"x": 0.1, "y": 0.13, "z": -0.03, "w": 0.99}
  },
  "confidence": 0.87
}
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

### Shading (`--use_light`)

| 값 | 설명 |
|---|---|
| `True` | Phong shading (광원 효과 적용) |
| `False` | Constant shading (vertex color 그대로) |

### Z축 보정 (`--fix_z_axis`)

벽면 장착 물체용. 물체 Z축이 항상 카메라를 향하도록 보정합니다.
- 가설 단계: 뒷면 후보 ~50% 제거 (`front_hemisphere_only`)
- 추정 후: Z축 방향 재확인 + 필요 시 FLIP_X 적용

---

## 기타 도구

### run_est.py — 오프라인 Pose 추정

저장된 이미지 시퀀스에 대해 pose 추정 수행:
```bash
python run_est.py \
    --mesh_file vcb/ref_views/ob_000001/model/model_vc_final.ply \
    --test_scene_dir vcb/ref_views/test_scene \
    --mask_model vcb/rcnn500.pth \
    --mask_type maskrcnn \
    --symmetry z180 \
    --input_mode rgb \
    --debug 2
```

### realtime/server.py — ZeroMQ GPU 서버

원격 클라이언트로부터 이미지를 받아 pose 추정 수행:
```bash
python realtime/server.py --port 5555 \
    --mesh_file vcb/ref_views/ob_000001/model/model_vc_final.ply \
    --mask_model vcb/rcnn500.pth
```
