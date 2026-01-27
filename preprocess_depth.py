#!/usr/bin/env python3
"""Preprocess depth images by inpainting invalid (zero/near-zero) pixels.

This script fills holes in depth images using OpenCV's inpainting algorithm,
which significantly improves pose estimation accuracy.

Usage:
    python preprocess_depth.py --depth_dir vcb/ref_views/test_scene/depth
"""

import cv2
import numpy as np
import os
import argparse
from glob import glob


def inpaint_depth(depth_path, threshold_mm=10, inpaint_radius=5):
    """Apply inpainting to fill invalid depth pixels.

    Args:
        depth_path: Path to depth image (uint16 PNG, values in mm)
        threshold_mm: Pixels below this value are considered invalid
        inpaint_radius: Radius for inpainting algorithm

    Returns:
        Inpainted depth image (uint16)
    """
    depth_raw = cv2.imread(depth_path, -1)
    if depth_raw is None:
        raise ValueError(f"Failed to read: {depth_path}")

    # Find invalid pixels
    invalid_mask = (depth_raw < threshold_mm).astype(np.uint8)
    invalid_count = np.sum(invalid_mask)

    if invalid_count == 0:
        return depth_raw, 0

    # Apply inpainting
    depth_float = depth_raw.astype(np.float32)
    depth_filled = cv2.inpaint(depth_float, invalid_mask, inpaint_radius, cv2.INPAINT_NS)
    depth_filled = depth_filled.astype(np.uint16)

    return depth_filled, invalid_count


def main():
    parser = argparse.ArgumentParser(description='Preprocess depth images with inpainting')
    parser.add_argument('--depth_dir', type=str, required=True, help='Directory containing depth images')
    parser.add_argument('--threshold_mm', type=int, default=10, help='Invalid depth threshold in mm')
    parser.add_argument('--backup', action='store_true', help='Create backup of original files')
    args = parser.parse_args()

    depth_files = sorted(glob(os.path.join(args.depth_dir, '*.png')))

    if not depth_files:
        print(f"No PNG files found in {args.depth_dir}")
        return

    print(f"Processing {len(depth_files)} depth images...")

    total_invalid = 0
    total_pixels = 0

    for depth_path in depth_files:
        if '_original' in depth_path:
            continue

        # Backup if requested
        if args.backup:
            backup_path = depth_path.replace('.png', '_original.png')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy(depth_path, backup_path)

        # Inpaint
        depth_filled, invalid_count = inpaint_depth(depth_path, args.threshold_mm)

        # Save
        cv2.imwrite(depth_path, depth_filled)

        total_invalid += invalid_count
        total_pixels += depth_filled.size

        if invalid_count > 0:
            print(f"  {os.path.basename(depth_path)}: filled {invalid_count} invalid pixels")

    print(f"\nDone! Total: {total_invalid}/{total_pixels} pixels filled ({total_invalid/total_pixels*100:.2f}%)")


if __name__ == '__main__':
    main()
