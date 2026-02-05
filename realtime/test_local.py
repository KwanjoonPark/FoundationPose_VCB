#!/usr/bin/env python3
"""
서버와 클라이언트를 로컬에서 테스트하는 스크립트.
카메라 없이 기존 테스트 이미지로 테스트.

Usage:
    # 서버를 먼저 실행한 후
    python test_local.py --server_ip localhost --test_dir ../vcb/ref_views/test_scene
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


def load_intrinsics(cam_file: str) -> np.ndarray:
    """카메라 intrinsics 로드."""
    with open(cam_file, 'r') as f:
        lines = f.readlines()
    K = []
    for line in lines[:3]:
        K.append([float(x) for x in line.strip().split()])
    return np.array(K, dtype=np.float32)


def check_display_available() -> bool:
    """디스플레이 사용 가능 여부 확인."""
    import os
    return os.environ.get('DISPLAY') is not None


def test_with_dataset(server_ip: str, port: int, test_dir: str, num_frames: int = 10, show_vis: bool = True):
    """테스트 데이터셋으로 서버 테스트."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Headless 환경 체크
    if show_vis and not check_display_available():
        logger.warning("No display available. Disabling visualization.")
        show_vis = False

    test_dir = Path(test_dir)
    rgb_dir = test_dir / 'rgb'
    depth_dir = test_dir / 'depth'

    # 이미지 파일 목록
    rgb_files = sorted(rgb_dir.glob('*.png'))[:num_frames]
    if not rgb_files:
        logger.error(f"No images found in {rgb_dir}")
        return

    # Camera intrinsics
    cam_file = test_dir / 'cam_K.txt'
    if cam_file.exists():
        K = load_intrinsics(str(cam_file))
        logger.info(f"Loaded camera K from {cam_file}")
    else:
        K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=np.float32)
        logger.warning("Using default camera K")

    # ZeroMQ 연결
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 30000)
    socket.connect(f"tcp://{server_ip}:{port}")

    # Ping
    socket.send_pyobj({'command': 'ping'})
    response = socket.recv_pyobj()
    logger.info(f"Server status: {response}")

    # 테스트 실행
    total_latency = 0
    success_count = 0

    for i, rgb_file in enumerate(rgb_files):
        logger.info(f"\n=== Frame {i+1}/{len(rgb_files)}: {rgb_file.name} ===")

        # 이미지 로드
        color = cv2.imread(str(rgb_file))
        if color is None:
            logger.error(f"Failed to load {rgb_file}")
            continue

        # Depth 로드 (있으면)
        depth_file = depth_dir / rgb_file.name
        depth = None
        if depth_file.exists():
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

        # JPEG 압축
        _, color_jpg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # 데이터 준비
        data = {
            'color': color_jpg.tobytes(),
            'K': K.flatten().tolist(),
        }
        if depth is not None:
            data['depth'] = depth.astype(np.uint16).tobytes()
            data['depth_shape'] = depth.shape

        # 전송
        t_start = time.time()
        socket.send_pyobj(data)
        result = socket.recv_pyobj()
        t_total = time.time() - t_start

        if result.get('success'):
            success_count += 1
            total_latency += result.get('latency_ms', 0)

            trans = result.get('translation', [0, 0, 0])
            euler = result.get('euler_angles', {})

            logger.info(f"Success!")
            logger.info(f"  Translation: X={trans[0]*100:.2f}cm, Y={trans[1]*100:.2f}cm, Z={trans[2]*100:.2f}cm")
            logger.info(f"  Rotation: Roll={euler.get('roll',0):.2f}°, Pitch={euler.get('pitch',0):.2f}°, Yaw={euler.get('yaw',0):.2f}°")
            logger.info(f"  Latency: {result.get('latency_ms', 0):.1f}ms (total: {t_total*1000:.1f}ms)")

            # 시각화 (디스플레이 있을 때만)
            if show_vis:
                vis = visualize_result(color.copy(), result)
                cv2.imshow('Result', vis)
                key = cv2.waitKey(500)
                if key == ord('q'):
                    break
        else:
            logger.warning(f"Failed: {result.get('error', 'Unknown')}")

    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info(f"Test Complete: {success_count}/{len(rgb_files)} frames successful")
    if success_count > 0:
        logger.info(f"Average latency: {total_latency/success_count:.1f}ms")

    if show_vis:
        cv2.destroyAllWindows()
    socket.close()
    context.term()


def visualize_result(color: np.ndarray, result: dict) -> np.ndarray:
    """결과 시각화."""
    vis = color.copy()

    if not result.get('success'):
        cv2.putText(vis, f"Error: {result.get('error', 'Unknown')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    trans = result.get('translation', [0, 0, 0])
    euler = result.get('euler_angles', {})
    latency = result.get('latency_ms', 0)

    texts = [
        f"X: {trans[0]*100:+6.2f} cm",
        f"Y: {trans[1]*100:+6.2f} cm",
        f"Z: {trans[2]*100:+6.2f} cm",
        f"Roll:  {euler.get('roll', 0):+7.2f} deg",
        f"Pitch: {euler.get('pitch', 0):+7.2f} deg",
        f"Yaw:   {euler.get('yaw', 0):+7.2f} deg",
        f"Latency: {latency:.1f}ms",
    ]

    y = 25
    for text in texts:
        cv2.putText(vis, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25

    return vis


def main():
    parser = argparse.ArgumentParser(description='Test FoundationPose server locally')
    parser.add_argument('--server_ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--test_dir', type=str,
                        default='vcb/ref_views/test_scene')
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable visualization')
    args = parser.parse_args()

    test_with_dataset(args.server_ip, args.port, args.test_dir, args.num_frames,
                      show_vis=not args.no_vis)


if __name__ == '__main__':
    main()
