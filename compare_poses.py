#!/usr/bin/env python3
"""Compare estimated poses with ground truth (both OpenCV ob_in_cam, meters)."""

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def load_pose(filepath):
    """Load 4x4 pose matrix from text file."""
    return np.loadtxt(filepath).reshape(4, 4)

def pose_to_trans_rot(pose):
    """Extract translation and rotation (euler angles) from pose."""
    trans = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True)
    return trans, rot

def compute_error(pose1, pose2):
    """Compute translation and rotation error between two poses."""
    trans1, _ = pose_to_trans_rot(pose1)
    trans2, _ = pose_to_trans_rot(pose2)

    trans_error = np.linalg.norm(trans1 - trans2)

    # Rotation error using trace formula
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    R_diff = R1 @ R2.T
    trace = np.trace(R_diff)
    rot_error = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi

    return trans_error, rot_error

# Paths
est_dir = '/home/ebduser/FoundationPose/debug/ob_in_cam'
gt_dir = '/home/ebduser/FoundationPose/vcb/ref_views/test_scene/ob_in_cam'

print("=" * 70)
print("Comparing Estimated vs Ground Truth Poses")
print("Coordinate: OpenCV, Format: ob_in_cam, Unit: meters")
print("=" * 70)

# Compare all frames
est_files = sorted([f for f in os.listdir(est_dir) if f.endswith('.txt')])
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
common_files = set(est_files) & set(gt_files)

if not common_files:
    print(f"\nNo common files found!")
    print(f"  Est dir: {est_dir}")
    print(f"  GT dir:  {gt_dir}")
    exit(1)

trans_errors = []
rot_errors = []

print(f"\n{'Frame':<12} {'Trans (cm)':>10} {'Rot (deg)':>10}")
print("-" * 34)

for fname in sorted(common_files):
    est_pose = load_pose(f'{est_dir}/{fname}')
    gt_pose = load_pose(f'{gt_dir}/{fname}')

    trans_err, rot_err = compute_error(est_pose, gt_pose)
    trans_errors.append(trans_err)
    rot_errors.append(rot_err)
    print(f"{fname:<12} {trans_err*100:>10.2f} {rot_err:>10.2f}")

print("-" * 34)
print(f"{'Mean':<12} {np.mean(trans_errors)*100:>10.2f} {np.mean(rot_errors):>10.2f}")
print(f"{'Std':<12} {np.std(trans_errors)*100:>10.2f} {np.std(rot_errors):>10.2f}")
print(f"{'Max':<12} {np.max(trans_errors)*100:>10.2f} {np.max(rot_errors):>10.2f}")
print(f"\nTotal frames: {len(common_files)}")
