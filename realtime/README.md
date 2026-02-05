# FoundationPose Real-time System

ZeroMQ 기반 실시간 6DoF Pose Estimation 시스템.

## Architecture

```
┌─────────────────────────┐         ZeroMQ (TCP)         ┌─────────────────────────┐
│      Local PC           │                              │      GPU Server         │
│                         │                              │                         │
│  ┌─────────────────┐    │      Image (JPEG)           │  ┌─────────────────┐    │
│  │ RealSense D435  │────┼──────────────────────────►  │  │  FoundationPose │    │
│  └─────────────────┘    │                              │  │  + Mask R-CNN   │    │
│                         │      Pose (4x4 matrix)       │  └─────────────────┘    │
│  ┌─────────────────┐    │  ◄──────────────────────────┼                         │
│  │  Visualization  │    │                              │  NVIDIA GPU (CUDA)      │
│  │  / Robot Ctrl   │    │                              │                         │
│  └─────────────────┘    │                              │                         │
└─────────────────────────┘                              └─────────────────────────┘
       client.py                                               server.py
```

## Requirements

### GPU Server
- CUDA 지원 GPU
- FoundationPose 환경 (이미 설정됨)
- pyzmq: `pip install pyzmq`

### Local PC
- Python 3.8+
- pyzmq: `pip install pyzmq`
- OpenCV: `pip install opencv-python`
- RealSense SDK (카메라 사용 시): `pip install pyrealsense2`

## Quick Start

### 1. GPU Server 시작

```bash
# SSH로 GPU 서버 접속
ssh user@gpu-server

# FoundationPose 디렉토리로 이동
cd /path/to/FoundationPose

# 서버 실행
python realtime/server.py --port 5555 \
    --mesh_file vcb/ref_views/ob_000001/model/model_vc.ply \
    --mask_model vcb/rcnn360.pth
```

### 2. Local PC에서 Client 실행

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

## Command Line Options

### Server (server.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 5555 | ZeroMQ 포트 |
| `--mesh_file` | model_vc.ply | 3D 메시 파일 경로 |
| `--mesh_scale` | 0.01 | 메시 스케일 (cm→m) |
| `--mask_model` | rcnn360.pth | 마스크 모델 경로 |
| `--mask_type` | maskrcnn | 마스크 모델 타입 |
| `--input_mode` | rgb | 입력 모드 (rgb/rgbd) |
| `--symmetry` | z180 | 대칭 타입 |
| `--debug` | False | 디버그 이미지 저장 |

### Client (client.py)

| Option | Default | Description |
|--------|---------|-------------|
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
| `--shutdown` | False | 서버 종료 명령 |

## Network Setup

### 방화벽 설정 (GPU Server)

```bash
# Ubuntu/Debian
sudo ufw allow 5555/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=5555/tcp --permanent
sudo firewall-cmd --reload
```

### SSH 터널링 (방화벽 우회)

```bash
# Local PC에서 실행
ssh -L 5555:localhost:5555 user@gpu-server

# 이후 client에서 localhost로 연결
python realtime/client.py --server_ip localhost --port 5555
```

## Performance

### Expected Latency

| Component | Time |
|-----------|------|
| Image compression (client) | ~5ms |
| Network transfer | ~10-30ms |
| Mask generation | ~20-30ms |
| Pose estimation | ~100-150ms |
| **Total** | **~150-200ms** |

### Tips for Better Performance

1. **JPEG 품질 낮추기**: `--jpeg_quality 60` (품질↓, 속도↑)
2. **해상도 낮추기**: `--width 480 --height 360`
3. **RGB 모드 사용**: `--input_mode rgb` (depth 전송 생략)
4. **유선 네트워크 사용**: WiFi보다 안정적

## Troubleshooting

### "Server not responding"
- 서버가 실행 중인지 확인
- 방화벽 설정 확인
- IP 주소/포트 확인

### "pyrealsense2 not installed"
```bash
pip install pyrealsense2
```

### "CUDA out of memory"
- 다른 GPU 프로세스 종료
- batch size 줄이기

### 네트워크 지연이 너무 큼
- 유선 네트워크 사용
- JPEG 품질/해상도 낮추기
- 서버와 같은 로컬 네트워크 사용

## Integration with ROS

ROS에서 사용하려면 `client.py`의 결과를 ROS 토픽으로 발행:

```python
import rospy
from geometry_msgs.msg import PoseStamped

# client.send_frame() 호출 후
if result.get('success'):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = 'camera_link'

    trans = result['translation']
    pose_msg.pose.position.x = trans[0]
    pose_msg.pose.position.y = trans[1]
    pose_msg.pose.position.z = trans[2]

    # rotation은 quaternion으로 변환 필요
    pose_pub.publish(pose_msg)
```

## API Reference

### Server Response Format

```python
{
    'success': True,
    'pose': [[4x4 matrix]],           # SE(3) transformation
    'translation': [x, y, z],          # meters
    'rotation_matrix': [[3x3 matrix]],
    'euler_angles': {
        'roll': float,   # degrees
        'pitch': float,
        'yaw': float
    },
    'mask_confidence': float,          # 0-1
    'latency_ms': float,               # server processing time
    'latency_breakdown': {
        'mask_ms': float,
        'pose_ms': float
    },
    'fps_avg': float                   # average server FPS
}
```

### Error Response

```python
{
    'success': False,
    'error': 'Error message',
    'latency_ms': float
}
```
