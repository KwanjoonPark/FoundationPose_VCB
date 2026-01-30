# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
6DoF pose estimation using FoundationPose with YOLO or Mask R-CNN segmentation.

This script runs FoundationPose on a sequence of RGB-D frames, using a learned
segmentation model (YOLO or Mask R-CNN) for object detection.

Usage:
    # YOLO segmentation
    python run_est.py --mesh_file model.obj --test_scene_dir ./scene \\
        --mask_model yolo_seg.pt --mask_type yolo

    # Mask R-CNN segmentation
    python run_est.py --mesh_file model.obj --test_scene_dir ./scene \\
        --mask_model model.pth --mask_type maskrcnn

    # Process every 10th frame
    python run_est.py --mask_model model.pt --frame_step 10

Output:
    - debug/ob_in_cam/*.txt: Object-to-camera transformation matrices
    - debug/cam_in_ob/*.txt: Camera-to-object transformation matrices
    - debug/track_vis/*.png: Visualization images (if debug >= 2)
"""

import argparse
import logging
import os
from typing import Tuple

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from estimater import (
    FoundationPose, ScorePredictor, PoseRefinePredictor,
    draw_posed_3d_box, draw_xyz_axis,
    set_logging_format, set_seed
)
from datareader import YcbineoatReader
from mask_generator import create_mask_generator

# Optional imports (used in debug mode)
try:
    import trimesh
    import open3d as o3d
    import nvdiffrast.torch as dr
    from estimater import depth2xyzmap, toOpen3dCloud
except ImportError:
    pass


# =============================================================================
# Pose Utilities
# =============================================================================

def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert 3x3 rotation matrix to Euler angles.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (pitch, yaw, roll) in degrees
    """
    sy = np.sqrt(R[0, 0]** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0

    return pitch, yaw, roll


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-90, 90] range."""
    if angle > 90:
        return angle - 180
    elif angle < -90:
        return angle + 180
    return angle


def convert_pose_for_saving(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix with normalized Euler angles for saving.

    Args:
        pose: 4x4 transformation matrix

    Returns:
        Modified 4x4 transformation matrix with corrected rotation
    """
    pitch, yaw, roll = rotation_matrix_to_euler(pose[:3, :3])
    pitch_save = normalize_angle(-pitch)
    yaw_save = normalize_angle(-yaw)
    roll_save = normalize_angle(roll)

    rot_corrected = Rot.from_euler('xyz', [pitch_save, yaw_save, roll_save], degrees=True)
    pose_save = pose.copy()
    pose_save[:3, :3] = rot_corrected.as_matrix()

    return pose_save


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(
    color: np.ndarray,
    pose: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    bbox: np.ndarray,
    extents: np.ndarray
) -> np.ndarray:
    """
    Create visualization image with pose, mask, and text overlay.

    Args:
        color: RGB image (H, W, 3)
        pose: 4x4 transformation matrix
        mask: Binary mask (H, W)
        K: 3x3 camera intrinsic matrix
        bbox: Object bounding box (2, 3)
        extents: Object extents (3,)

    Returns:
        Visualization image (H, W, 3)
    """
    # Draw 3D bounding box
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=pose, bbox=bbox)

    # Draw coordinate axes (with 180 deg rotation for visualization)
    axis_scale = max(extents)
    Rx_180 = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    pose_vis = pose @ Rx_180
    vis = draw_xyz_axis(
        vis, ob_in_cam=pose_vis, scale=axis_scale, K=K,
        thickness=3, transparency=0, is_input_rgb=True
    )

    # Overlay mask (red with yellow contour)
    mask_overlay = np.zeros_like(vis)
    mask_overlay[mask] = [255, 0, 0]
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

    # Add text overlay
    vis = add_pose_text(vis, pose)

    return vis


def add_pose_text(vis: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Add pose information as text overlay on image."""
    trans = pose[:3, 3]
    pitch, yaw, roll = rotation_matrix_to_euler(pose[:3, :3])
    pitch = normalize_angle(-pitch)
    yaw = normalize_angle(-yaw)
    roll = normalize_angle(roll)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)

    texts = [
        (f'X: {trans[0] * 100:+6.2f} cm', 25),
        (f'Y: {trans[1] * 100:+6.2f} cm', 50),
        (f'Z: {trans[2] * 100:+6.2f} cm', 75),
        (f'Roll:  {roll:+7.2f} deg', 105),
        (f'Pitch: {pitch:+7.2f} deg', 130),
        (f'Yaw:   {yaw:+7.2f} deg', 155),
    ]

    for text, y in texts:
        cv2.putText(vis_bgr, text, (10, y), font, 0.6, color, 2)

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description='6DoF pose estimation with FoundationPose',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mesh arguments
    mesh_group = parser.add_argument_group('Mesh')
    mesh_group.add_argument(
        '--mesh_file', type=str,
        default=f'{code_dir}/vcb/ref_views/ob_000001/model/model.obj',
        help='Path to object mesh file (.obj)'
    )
    mesh_group.add_argument(
        '--mesh_scale', type=float, default=0.01,
        help='Scale factor for mesh (e.g., 0.01 for cm to m)'
    )

    # Scene arguments
    scene_group = parser.add_argument_group('Scene')
    scene_group.add_argument(
        '--test_scene_dir', type=str,
        default=f'{code_dir}/vcb/ref_views/test_scene',
        help='Directory containing test scene (rgb/, depth/)'
    )
    scene_group.add_argument(
        '--frame_step', type=int, default=1,
        help='Process every N-th frame'
    )
    scene_group.add_argument(
        '--frame_start', type=int, default=0,
        help='Start frame index'
    )
    scene_group.add_argument(
        '--frame_end', type=int, default=-1,
        help='End frame index (-1 for all)'
    )

    # Pose estimation arguments
    pose_group = parser.add_argument_group('Pose Estimation')
    pose_group.add_argument(
        '--est_refine_iter', type=int, default=5,
        help='Refinement iterations for registration'
    )
    pose_group.add_argument(
        '--track_refine_iter', type=int, default=2,
        help='Refinement iterations for tracking'
    )
    pose_group.add_argument(
        '--use_tracking', action='store_true',
        help='Use tracking after first frame (faster but less robust)'
    )

    # Mask generation arguments
    mask_group = parser.add_argument_group('Mask Generation')
    mask_group.add_argument(
        '--mask_model', type=str, required=True,
        help='Path to segmentation model (.pt for YOLO, .pth for Mask R-CNN)'
    )
    mask_group.add_argument(
        '--mask_type', type=str, default='yolo',
        choices=['yolo', 'maskrcnn'],
        help='Segmentation model type'
    )
    mask_group.add_argument(
        '--mask_conf', type=float, default=0.5,
        help='Confidence threshold for mask generation'
    )
    mask_group.add_argument(
        '--mask_config', type=str, default=None,
        help='Detectron2 config file (Mask R-CNN only)'
    )

    # Debug arguments
    debug_group = parser.add_argument_group('Debug')
    debug_group.add_argument(
        '--debug', type=int, default=2,
        help='Debug level (0=none, 1=vis, 2=save, 3=full)'
    )
    debug_group.add_argument(
        '--debug_dir', type=str,
        default=f'{code_dir}/vcb/debug/rcnn',
        help='Directory for debug outputs'
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for pose estimation."""
    args = parse_args()

    set_logging_format()
    set_seed(0)

    # -------------------------------------------------------------------------
    # Load mesh
    # -------------------------------------------------------------------------
    mesh = trimesh.load(args.mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if args.mesh_scale != 1.0:
        mesh.apply_scale(args.mesh_scale)
        logging.info(f"Mesh scaled by {args.mesh_scale} (extents: {mesh.extents})")

    extents = mesh.extents
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # -------------------------------------------------------------------------
    # Setup directories
    # -------------------------------------------------------------------------
    debug_dir = args.debug_dir
    subdirs = ['track_vis', 'ob_in_cam', 'cam_in_ob']
    os.makedirs(debug_dir, exist_ok=True)
    for subdir in subdirs:
        subdir_path = os.path.join(debug_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        # Clean existing files
        for f in os.listdir(subdir_path):
            os.remove(os.path.join(subdir_path, f))

    # -------------------------------------------------------------------------
    # Initialize FoundationPose
    # -------------------------------------------------------------------------
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=args.debug,
        glctx=glctx
    )
    logging.info("FoundationPose estimator initialized")

    # -------------------------------------------------------------------------
    # Initialize data reader
    # -------------------------------------------------------------------------
    reader = YcbineoatReader(
        video_dir=args.test_scene_dir,
        shorter_side=None,
        zfar=np.inf
    )
    logging.info(f"Loaded {len(reader.color_files)} frames from {args.test_scene_dir}")

    # -------------------------------------------------------------------------
    # Initialize mask generator
    # -------------------------------------------------------------------------
    mask_kwargs = {}
    if args.mask_type == 'maskrcnn' and args.mask_config:
        mask_kwargs['config_file'] = args.mask_config

    mask_generator = create_mask_generator(
        model_path=args.mask_model,
        model_type=args.mask_type,
        conf_threshold=args.mask_conf,
        **mask_kwargs
    )
    logging.info(f"Mask generator: {args.mask_type} (conf={args.mask_conf})")

    # -------------------------------------------------------------------------
    # Compute frame indices
    # -------------------------------------------------------------------------
    total_frames = len(reader.color_files)
    frame_end = args.frame_end if args.frame_end >= 0 else total_frames
    frame_indices = list(range(
        args.frame_start,
        min(frame_end, total_frames),
        args.frame_step
    ))
    logging.info(
        f"Processing {len(frame_indices)} frames "
        f"(step={args.frame_step}, range={args.frame_start}-{frame_end})"
    )

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    pose = None

    for i in frame_indices:
        logging.info(f"Processing frame {i}/{total_frames - 1}")

        color = reader.get_color(i)
        depth = reader.get_depth(i)

        # Generate mask
        mask, mask_info = mask_generator.get_mask_with_depth(color, depth)

        if mask is None:
            logging.warning(f"Frame {i}: No mask - {mask_info.get('error', 'unknown')}")
            continue

        logging.info(f"Frame {i}: Mask OK (conf={mask_info.get('confidence', 0):.2f})")
        mask = mask.astype(bool)

        # Pose estimation
        if pose is None or not args.use_tracking:
            pose = est.register(
                K=reader.K,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                iteration=args.est_refine_iter
            )

            # Debug: export transformed mesh and point cloud
            if args.debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')

                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(
                rgb=color,
                depth=depth,
                K=reader.K,
                iteration=args.track_refine_iter
            )

        # Save pose matrices
        pose_save = convert_pose_for_saving(pose)
        frame_id = reader.id_strs[i]

        np.savetxt(f'{debug_dir}/ob_in_cam/{frame_id}.txt', pose_save.reshape(4, 4))
        np.savetxt(f'{debug_dir}/cam_in_ob/{frame_id}.txt', np.linalg.inv(pose_save).reshape(4, 4))

        # Visualization
        if args.debug >= 1:
            vis = create_visualization(color, pose, mask, reader.K, bbox, extents)

            if args.debug >= 2:
                imageio.imwrite(f'{debug_dir}/track_vis/{frame_id}.png', vis)

    logging.info(f"Done! Results saved to {debug_dir}/")


if __name__ == '__main__':
    main()
