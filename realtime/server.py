#!/usr/bin/env python3
"""
FoundationPose ZeroMQ Server
GPU 서버에서 실행 - 이미지를 받아 pose estimation 수행

Usage:
    python server.py --port 5555 --mesh_file ../vcb/ref_views/ob_000001/model/model_vc.ply
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import zmq

# FoundationPose imports
import trimesh
import nvdiffrast.torch as dr
from estimater import (
    FoundationPose, ScorePredictor, PoseRefinePredictor,
    set_logging_format, set_seed
)


@dataclass
class ServerConfig:
    """서버 설정."""
    port: int = 5555
    mesh_file: str = 'vcb/ref_views/ob_000001/model/model_vc.ply'
    mesh_scale: float = 0.01
    mask_model: str = 'vcb/rcnn360.pth'
    mask_type: str = 'maskrcnn'
    input_mode: str = 'rgb'  # 'rgb' or 'rgbd'
    use_mask_iou: bool = True
    fix_z_axis: bool = True
    symmetry: str = 'z180'
    debug: bool = False


class PoseEstimationServer:
    """ZeroMQ 기반 Pose Estimation 서버."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        set_logging_format()
        set_seed(0)

        self._init_foundation_pose()
        self._init_mask_generator()
        self._init_zmq()

        self.frame_count = 0
        self.total_time = 0

    def _init_foundation_pose(self):
        """FoundationPose 초기화."""
        self.logger.info(f"Loading mesh: {self.config.mesh_file}")

        mesh = trimesh.load(self.config.mesh_file)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if self.config.mesh_scale != 1.0:
            mesh.apply_scale(self.config.mesh_scale)

        self.mesh = mesh

        # Symmetry transforms
        symmetry_tfs = self._create_symmetry_transforms()

        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            symmetry_tfs=symmetry_tfs,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=1 if self.config.debug else 0,
            debug_dir='./realtime/debug',
            front_hemisphere_only=self.config.fix_z_axis,
            use_mask_iou=self.config.use_mask_iou,
        )

        self.logger.info("FoundationPose initialized")

    def _create_symmetry_transforms(self) -> np.ndarray:
        """대칭 변환 행렬 생성."""
        if self.config.symmetry == 'none':
            return np.array([np.eye(4)])
        elif self.config.symmetry == 'z180':
            rz180 = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            return np.array([np.eye(4), rz180])
        else:
            return np.array([np.eye(4)])

    def _init_mask_generator(self):
        """마스크 생성기 초기화."""
        from mask_generator import create_mask_generator

        self.mask_generator = create_mask_generator(
            model_path=self.config.mask_model,
            model_type=self.config.mask_type,
            conf_threshold=0.5
        )
        self.logger.info(f"Mask generator initialized: {self.config.mask_type}")

    def _init_zmq(self):
        """ZeroMQ 소켓 초기화."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.config.port}")
        self.logger.info(f"ZeroMQ server listening on port {self.config.port}")

    def process_frame(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """단일 프레임 처리."""
        t_start = time.time()

        # 이미지 복원
        color = cv2.imdecode(
            np.frombuffer(data['color'], np.uint8),
            cv2.IMREAD_COLOR
        )
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # Depth 복원 (있는 경우)
        depth = None
        if 'depth' in data and data['depth'] is not None:
            depth = np.frombuffer(data['depth'], dtype=np.uint16).reshape(data['depth_shape'])
            depth = depth.astype(np.float32) / 1000.0  # mm to m

        # Camera intrinsics
        K = np.array(data['K']).reshape(3, 3) if 'K' in data else self._default_K(color.shape)

        # 마스크 생성
        t_mask = time.time()
        mask, mask_info = self.mask_generator.get_mask_with_depth(color, depth)

        if mask is None:
            return {
                'success': False,
                'error': 'No mask detected',
                'latency_ms': (time.time() - t_start) * 1000
            }

        mask = mask.astype(bool)
        t_mask_done = time.time()

        # Pose 추정
        t_pose = time.time()
        depth_input = None if self.config.input_mode == 'rgb' else depth

        pose = self.estimator.register(
            K=K,
            rgb=color,
            depth=depth_input,
            ob_mask=mask,
            iteration=5
        )
        t_pose_done = time.time()

        # Z축 방향 보정
        if self.config.fix_z_axis:
            pose = self._fix_z_axis(pose)

        # 결과 생성
        t_total = time.time() - t_start
        self.frame_count += 1
        self.total_time += t_total

        result = {
            'success': True,
            'pose': pose.tolist(),
            'translation': pose[:3, 3].tolist(),
            'rotation_matrix': pose[:3, :3].tolist(),
            'euler_angles': self._to_euler(pose[:3, :3]),
            'mask_confidence': mask_info.get('confidence', 0),
            'latency_ms': t_total * 1000,
            'latency_breakdown': {
                'mask_ms': (t_mask_done - t_mask) * 1000,
                'pose_ms': (t_pose_done - t_pose) * 1000,
            },
            'fps_avg': self.frame_count / self.total_time if self.total_time > 0 else 0
        }

        return result

    def _fix_z_axis(self, pose: np.ndarray) -> np.ndarray:
        """Z축이 카메라를 향하도록 보정."""
        z_axis = pose[:3, 2]
        if z_axis[2] > 0:
            flip_x = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            pose = pose @ flip_x
        return pose

    def _to_euler(self, R: np.ndarray) -> Dict[str, float]:
        """회전 행렬을 오일러 각도로 변환."""
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        euler = r.as_euler('xyz', degrees=True)
        return {
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2])
        }

    def _default_K(self, shape: Tuple[int, ...]) -> np.ndarray:
        """기본 카메라 intrinsics (RealSense D435 기준)."""
        h, w = shape[:2]
        fx = fy = 615.0 * (w / 640)
        cx, cy = w / 2, h / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def run(self):
        """서버 메인 루프."""
        self.logger.info("Server ready. Waiting for requests...")

        while True:
            try:
                # 요청 수신
                data = self.socket.recv_pyobj()

                # 종료 명령
                if data.get('command') == 'shutdown':
                    self.logger.info("Shutdown command received")
                    self.socket.send_pyobj({'status': 'shutdown'})
                    break

                # 상태 확인
                if data.get('command') == 'ping':
                    self.socket.send_pyobj({
                        'status': 'ok',
                        'frame_count': self.frame_count,
                        'fps_avg': self.frame_count / self.total_time if self.total_time > 0 else 0
                    })
                    continue

                # 프레임 처리
                result = self.process_frame(data)
                self.socket.send_pyobj(result)

                if self.frame_count % 10 == 0:
                    self.logger.info(
                        f"Frame {self.frame_count}: "
                        f"latency={result.get('latency_ms', 0):.1f}ms, "
                        f"avg_fps={result.get('fps_avg', 0):.1f}"
                    )

            except KeyboardInterrupt:
                self.logger.info("Interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                self.socket.send_pyobj({'success': False, 'error': str(e)})

        self.cleanup()

    def cleanup(self):
        """리소스 정리."""
        self.socket.close()
        self.context.term()
        self.logger.info("Server shutdown complete")


def parse_args():
    parser = argparse.ArgumentParser(description='FoundationPose ZeroMQ Server')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--mesh_file', type=str,
                        default='vcb/ref_views/ob_000001/model/model_vc.ply')
    parser.add_argument('--mesh_scale', type=float, default=0.01)
    parser.add_argument('--mask_model', type=str, default='vcb/rcnn360.pth')
    parser.add_argument('--mask_type', type=str, default='maskrcnn',
                        choices=['yolo', 'maskrcnn'])
    parser.add_argument('--input_mode', type=str, default='rgb',
                        choices=['rgb', 'rgbd'])
    parser.add_argument('--symmetry', type=str, default='z180')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    config = ServerConfig(
        port=args.port,
        mesh_file=args.mesh_file,
        mesh_scale=args.mesh_scale,
        mask_model=args.mask_model,
        mask_type=args.mask_type,
        input_mode=args.input_mode,
        symmetry=args.symmetry,
        debug=args.debug,
    )

    server = PoseEstimationServer(config)
    server.run()


if __name__ == '__main__':
    main()
