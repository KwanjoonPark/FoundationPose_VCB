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
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Rotation Utilities (from run_est.py)
# =============================================================================

class RotationUtils:
    """회전 관련 유틸리티 함수 모음."""

    @staticmethod
    def to_euler(R_mat: np.ndarray) -> Tuple[float, float, float]:
        """3x3 회전 행렬을 오일러 각도(pitch, yaw, roll)로 변환."""
        sy = np.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(R_mat[2, 1], R_mat[2, 2]))
            yaw = np.degrees(np.arctan2(-R_mat[2, 0], sy))
            roll = np.degrees(np.arctan2(R_mat[1, 0], R_mat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R_mat[1, 2], R_mat[1, 1]))
            yaw = np.degrees(np.arctan2(-R_mat[2, 0], sy))
            roll = 0.0

        return pitch, yaw, roll

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """각도를 [-90, 90] 범위로 정규화."""
        if angle > 90:
            return angle - 180
        elif angle < -90:
            return angle + 180
        return angle

    @staticmethod
    def rot_x(angle_deg: float) -> np.ndarray:
        """X축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(a), -np.sin(a), 0],
            [0, np.sin(a), np.cos(a), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def rot_y(angle_deg: float) -> np.ndarray:
        """Y축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [np.cos(a), 0, np.sin(a), 0],
            [0, 1, 0, 0],
            [-np.sin(a), 0, np.cos(a), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def rot_z(angle_deg: float) -> np.ndarray:
        """Z축 회전 행렬 생성."""
        a = np.radians(angle_deg)
        return np.array([
            [np.cos(a), -np.sin(a), 0, 0],
            [np.sin(a), np.cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)


# =============================================================================
# Symmetry Generator (from run_est.py)
# =============================================================================

class SymmetryGenerator:
    """대칭 변환 행렬 생성기."""

    AXIS_ROTATIONS = {
        'x': RotationUtils.rot_x,
        'y': RotationUtils.rot_y,
        'z': RotationUtils.rot_z,
    }

    @classmethod
    def create(cls, symmetry: str, angle_step: float = 5.0) -> np.ndarray:
        """대칭 변환 행렬 배열 생성."""
        if symmetry is None or symmetry == 'none':
            return np.array([np.eye(4)])

        symmetry = symmetry.lower()
        tfs = [np.eye(4)]

        # 단일 축 연속 대칭
        if symmetry in cls.AXIS_ROTATIONS:
            rot_fn = cls.AXIS_ROTATIONS[symmetry]
            for angle in np.arange(angle_step, 360, angle_step):
                tfs.append(rot_fn(angle))

        # 단일 축 180도 대칭
        elif symmetry.endswith('180') and symmetry[:-3] in cls.AXIS_ROTATIONS:
            axis = symmetry[:-3]
            tfs.append(cls.AXIS_ROTATIONS[axis](180))

        # 복합 대칭
        elif symmetry in ('xy', 'xz', 'yz', 'xyz'):
            tfs.extend(cls._create_combined_symmetry(symmetry))

        else:
            logging.warning(f"알 수 없는 대칭 타입: {symmetry}, 'none' 사용")

        return np.array(tfs, dtype=np.float32)

    @classmethod
    def _create_combined_symmetry(cls, symmetry: str):
        """복합 대칭 변환 생성."""
        rx = RotationUtils.rot_x(180)
        ry = RotationUtils.rot_y(180)
        rz = RotationUtils.rot_z(180)

        transforms = {
            'xy': [rx, ry, rx @ ry],
            'xz': [rx, rz, rx @ rz],
            'yz': [ry, rz, ry @ rz],
            'xyz': [rx, ry, rz, rx @ ry, rx @ rz, ry @ rz, rx @ ry @ rz],
        }
        return transforms.get(symmetry, [])


# =============================================================================
# Pose Corrector (from run_est.py)
# =============================================================================

class PoseCorrector:
    """Pose 보정 유틸리티."""

    # 180도 X축 플립 행렬
    FLIP_X = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # 180도 Z축 플립 행렬
    FLIP_Z = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    @classmethod
    def fix_z_axis_direction(cls, pose: np.ndarray) -> np.ndarray:
        """Z축이 카메라 방향을 향하도록 보정."""
        z_axis = pose[:3, 2]
        if z_axis[2] > 0:
            pose = pose @ cls.FLIP_X
        return pose

    @classmethod
    def fix_pitch_yaw_sign(cls, pose: np.ndarray) -> np.ndarray:
        """pitch/yaw 부호 모호성 보정. pitch를 양수(+)로 통일."""
        pitch, _, _ = RotationUtils.to_euler(pose[:3, :3])
        if pitch < 0:
            pose = pose @ cls.FLIP_Z
        return pose

    @classmethod
    def convert_for_output(cls, pose: np.ndarray) -> Dict[str, Any]:
        """출력용으로 정규화된 euler 각도 반환."""
        pitch, yaw, roll = RotationUtils.to_euler(pose[:3, :3])

        # GT 좌표계 변환 및 정규화
        pitch_out = RotationUtils.normalize_angle(-pitch)
        yaw_out = RotationUtils.normalize_angle(-yaw)
        roll_out = RotationUtils.normalize_angle(roll)

        return {
            'roll': float(roll_out),
            'pitch': float(pitch_out),
            'yaw': float(yaw_out)
        }


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
    symmetry_step: float = 5.0
    min_n_views: int = 40
    inplane_step: int = 60
    mask_dilate: int = 0  # 0=disabled, 3~7 recommended
    mask_dilate_iter: int = 2
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

        # Symmetry transforms (using SymmetryGenerator from run_est.py)
        symmetry_tfs = SymmetryGenerator.create(
            self.config.symmetry,
            self.config.symmetry_step
        )
        self.logger.info(f"Symmetry: {self.config.symmetry} ({len(symmetry_tfs)} transforms)")

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
            min_n_views=self.config.min_n_views,
            inplane_step=self.config.inplane_step,
        )

        self.logger.info(
            f"FoundationPose initialized: "
            f"front_hemisphere={self.config.fix_z_axis}, "
            f"mask_iou={self.config.use_mask_iou}, "
            f"min_n_views={self.config.min_n_views}, "
            f"inplane_step={self.config.inplane_step}"
        )

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

        # 마스크 팽창 (dilation) - tight mask 문제 해결
        if self.config.mask_dilate > 0:
            kernel = np.ones(
                (self.config.mask_dilate, self.config.mask_dilate),
                np.uint8
            )
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(
                mask_uint8, kernel,
                iterations=self.config.mask_dilate_iter
            )
            mask = (mask_dilated > 127)
        else:
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

        # Pose 보정 (run_est.py와 동일한 로직)
        pose = self._correct_pose(pose)

        # 결과 생성
        t_total = time.time() - t_start
        self.frame_count += 1
        self.total_time += t_total

        # Euler 각도 (run_est.py와 동일한 방식)
        euler_angles = PoseCorrector.convert_for_output(pose)

        result = {
            'success': True,
            'pose': pose.tolist(),
            'translation': pose[:3, 3].tolist(),
            'rotation_matrix': pose[:3, :3].tolist(),
            'euler_angles': euler_angles,
            'mask_confidence': mask_info.get('confidence', 0),
            'latency_ms': t_total * 1000,
            'latency_breakdown': {
                'mask_ms': (t_mask_done - t_mask) * 1000,
                'pose_ms': (t_pose_done - t_pose) * 1000,
            },
            'fps_avg': self.frame_count / self.total_time if self.total_time > 0 else 0
        }

        return result

    def _correct_pose(self, pose: np.ndarray) -> np.ndarray:
        """Pose 보정 적용 (run_est.py와 동일)."""
        # Z축 방향 보정
        if self.config.fix_z_axis:
            pose = PoseCorrector.fix_z_axis_direction(pose)

        # 180도 대칭 사용 시 pitch/yaw 부호 보정
        if self.config.symmetry and '180' in self.config.symmetry:
            pose = PoseCorrector.fix_pitch_yaw_sign(pose)

        return pose

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
    parser = argparse.ArgumentParser(
        description='FoundationPose ZeroMQ Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Network
    parser.add_argument('--port', type=int, default=5555,
                        help='ZeroMQ port')

    # Mesh
    parser.add_argument('--mesh_file', type=str,
                        default='vcb/ref_views/ob_000001/model/model_vc.ply',
                        help='Path to mesh file (PLY or OBJ)')
    parser.add_argument('--mesh_scale', type=float, default=0.01,
                        help='Mesh scale factor (e.g., 0.01 for cm to m)')

    # Mask
    parser.add_argument('--mask_model', type=str, default='vcb/rcnn360.pth',
                        help='Path to mask model')
    parser.add_argument('--mask_type', type=str, default='maskrcnn',
                        choices=['yolo', 'maskrcnn'],
                        help='Mask model type')
    parser.add_argument('--mask_dilate', type=int, default=0,
                        help='Mask dilation kernel size (0=disabled, 3~7 recommended)')
    parser.add_argument('--mask_dilate_iter', type=int, default=2,
                        help='Mask dilation iterations')

    # Pose estimation
    parser.add_argument('--input_mode', type=str, default='rgb',
                        choices=['rgb', 'rgbd'],
                        help='Input mode: rgb (RGB only) or rgbd (RGB + Depth)')
    parser.add_argument('--symmetry', type=str, default='z180',
                        choices=['none', 'z', 'z180', 'x', 'x180', 'y', 'y180', 'xy', 'xz', 'yz', 'xyz'],
                        help='Symmetry type')
    parser.add_argument('--symmetry_step', type=float, default=5.0,
                        help='Angle step for continuous symmetry (degrees)')
    parser.add_argument('--fix_z_axis', type=lambda x: x.lower() == 'true', default=True,
                        help='Filter back-facing poses (front hemisphere only)')
    parser.add_argument('--use_mask_iou', type=lambda x: x.lower() == 'true', default=True,
                        help='Use mask IoU bonus in scoring')
    parser.add_argument('--min_n_views', type=int, default=40,
                        help='Number of viewpoints for pose hypothesis')
    parser.add_argument('--inplane_step', type=int, default=60,
                        help='In-plane rotation step in degrees')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    return parser.parse_args()


def main():
    args = parse_args()

    config = ServerConfig(
        port=args.port,
        mesh_file=args.mesh_file,
        mesh_scale=args.mesh_scale,
        mask_model=args.mask_model,
        mask_type=args.mask_type,
        mask_dilate=args.mask_dilate,
        mask_dilate_iter=args.mask_dilate_iter,
        input_mode=args.input_mode,
        symmetry=args.symmetry,
        symmetry_step=args.symmetry_step,
        fix_z_axis=args.fix_z_axis,
        use_mask_iou=args.use_mask_iou,
        min_n_views=args.min_n_views,
        inplane_step=args.inplane_step,
        debug=args.debug,
    )

    server = PoseEstimationServer(config)
    server.run()


if __name__ == '__main__':
    main()
