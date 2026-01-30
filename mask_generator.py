# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Mask generation utilities for 6DoF pose estimation.

This module provides mask generators using different segmentation backends:
- YoloMaskGenerator: YOLO-based segmentation (ultralytics)
- MaskRCNNMaskGenerator: Mask R-CNN segmentation (Detectron2)

Both generators support optional depth-based mask refinement for improved
accuracy in cluttered scenes.

Example:
    >>> from mask_generator import YoloMaskGenerator
    >>> mask_gen = YoloMaskGenerator('model.pt', conf_threshold=0.5)
    >>> mask, info = mask_gen.get_mask(rgb_image)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import cv2
import torch


def _refine_mask_with_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_threshold: float = 0.15,
    min_ratio: float = 0.3
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Refine a binary mask using depth information.

    Args:
        mask: Binary mask (H, W), uint8
        depth: Depth image (H, W), in meters
        depth_threshold: Maximum depth difference from median object depth
        min_ratio: Minimum ratio of refined mask size to original

    Returns:
        refined_mask: Refined binary mask (H, W)
        object_depth: Median depth of the object, or None if refinement failed
    """
    masked_depths = depth[mask > 0]
    valid_depths = masked_depths[(masked_depths > 0.001) & (masked_depths < 10)]

    if len(valid_depths) == 0:
        return mask, None

    obj_depth = float(np.median(valid_depths))
    depth_mask = (np.abs(depth - obj_depth) < depth_threshold) & (depth > 0.001)
    refined_mask = mask & depth_mask.astype(np.uint8)

    # Only use refined mask if it retains enough of the original
    if refined_mask.sum() > mask.sum() * min_ratio:
        return refined_mask, obj_depth

    return mask, None


class MaskRCNNMaskGenerator:
    """
    Mask R-CNN based mask generator using Detectron2.

    Supports custom trained models with instance segmentation.
    Requires: pip install 'git+https://github.com/facebookresearch/detectron2.git'
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
        config_file: Optional[str] = None,
        num_classes: int = 1
    ):
        """
        Initialize Mask R-CNN mask generator.

        Args:
            model_path: Path to .pth weights file
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            config_file: Path to Detectron2 config yaml. If None, uses default
                         Mask R-CNN R50-FPN config from COCO.
            num_classes: Number of object classes (excluding background)
        """
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo

        self.conf_threshold = conf_threshold

        cfg = get_cfg()

        if config_file:
            cfg.merge_from_file(config_file)
        else:
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.WEIGHTS = model_path

        if device:
            cfg.MODEL.DEVICE = device
        elif not torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cpu'

        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

        print(f"[MaskRCNNMaskGenerator] Loaded model: {model_path}")

    def get_mask(
        self,
        rgb: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate mask using Mask R-CNN segmentation.

        Args:
            rgb: Input RGB image (H, W, 3), uint8
            conf_threshold: Override default confidence threshold

        Returns:
            mask: Binary mask (H, W) as uint8, or None if no detection
            info: Dict containing detection metadata:
                - method: 'maskrcnn'
                - confidence: Detection confidence score
                - class_id: Predicted class ID
                - num_detections: Total number of detections
                - error: Error message if detection failed
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Detectron2 expects BGR format
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        outputs = self.predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return None, {'error': 'No detection'}

        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()
        labels = instances.pred_classes.numpy()

        # Filter by confidence threshold
        if conf_threshold is not None:
            valid_idx = scores >= conf
            if not valid_idx.any():
                return None, {'error': 'No detection above threshold'}
            scores = scores[valid_idx]
            masks = masks[valid_idx]
            labels = labels[valid_idx]

        # Select highest confidence detection
        best_idx = scores.argmax()
        mask = masks[best_idx].astype(np.uint8)
        conf_score = scores[best_idx]
        class_id = int(labels[best_idx])

        info = {
            'method': 'maskrcnn',
            'confidence': float(conf_score),
            'class_id': class_id,
            'num_detections': len(scores)
        }

        return mask, info

    def get_mask_with_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        conf_threshold: Optional[float] = None,
        depth_refine: bool = True,
        depth_threshold: float = 0.15
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate mask using Mask R-CNN, optionally refined with depth.

        Args:
            rgb: Input RGB image (H, W, 3)
            depth: Depth image (H, W) in meters
            conf_threshold: Override default confidence threshold
            depth_refine: Whether to refine mask using depth
            depth_threshold: Maximum depth difference for refinement (meters)

        Returns:
            mask: Binary mask (H, W) or None if no detection
            info: Dict with detection info (includes 'depth_refined' and
                  'object_depth' if depth refinement was applied)
        """
        mask, info = self.get_mask(rgb, conf_threshold)

        if mask is None:
            return None, info

        if depth_refine and depth is not None:
            refined_mask, obj_depth = _refine_mask_with_depth(
                mask, depth, depth_threshold)
            if obj_depth is not None:
                mask = refined_mask
                info['depth_refined'] = True
                info['object_depth'] = obj_depth

        return mask, info


class YoloMaskGenerator:
    """
    YOLO-based mask generator using ultralytics.

    Recommended for textureless objects when a trained YOLO segmentation
    model is available. Supports YOLOv8 and newer versions.

    Requires: pip install ultralytics
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO mask generator.

        Args:
            model_path: Path to YOLO .pt weights file
            conf_threshold: Confidence threshold for detection (0.0-1.0)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names

        print(f"[YoloMaskGenerator] Loaded model with classes: {self.class_names}")

    def get_mask(
        self,
        rgb: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate mask using YOLO segmentation.

        Args:
            rgb: Input RGB image (H, W, 3), uint8
            conf_threshold: Override default confidence threshold

        Returns:
            mask: Binary mask (H, W) as uint8, or None if no detection
            info: Dict containing detection metadata:
                - method: 'yolo'
                - confidence: Detection confidence score
                - class_id: Predicted class ID
                - class_name: Predicted class name
                - num_detections: Total number of detections
                - error: Error message if detection failed
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model(rgb, verbose=False, conf=conf, device=self.device)

        if len(results) == 0 or results[0].masks is None:
            return None, {'error': 'No detection'}

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes

        if len(masks) == 0:
            return None, {'error': 'No masks'}

        # Select highest confidence detection
        confidences = boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()

        mask = masks[best_idx]
        conf_score = confidences[best_idx]
        class_id = int(boxes.cls[best_idx].cpu().numpy())

        # Resize mask to original image size if needed
        h, w = rgb.shape[:2]
        mh, mw = mask.shape[:2]
        if mh != h or mw != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binarize mask
        mask = (mask > 0.5).astype(np.uint8)

        info = {
            'method': 'yolo',
            'confidence': float(conf_score),
            'class_id': class_id,
            'class_name': self.class_names.get(class_id, 'unknown'),
            'num_detections': len(masks)
        }

        return mask, info

    def get_mask_with_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        conf_threshold: Optional[float] = None,
        depth_refine: bool = True,
        depth_threshold: float = 0.15
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Generate mask using YOLO, optionally refined with depth.

        Args:
            rgb: Input RGB image (H, W, 3)
            depth: Depth image (H, W) in meters
            conf_threshold: Override default confidence threshold
            depth_refine: Whether to refine mask using depth
            depth_threshold: Maximum depth difference for refinement (meters)

        Returns:
            mask: Binary mask (H, W) or None if no detection
            info: Dict with detection info (includes 'depth_refined' and
                  'object_depth' if depth refinement was applied)
        """
        mask, info = self.get_mask(rgb, conf_threshold)

        if mask is None:
            return None, info

        if depth_refine and depth is not None:
            refined_mask, obj_depth = _refine_mask_with_depth(
                mask, depth, depth_threshold)
            if obj_depth is not None:
                mask = refined_mask
                info['depth_refined'] = True
                info['object_depth'] = obj_depth

        return mask, info


# =============================================================================
# Factory Functions
# =============================================================================

def create_mask_generator(
    model_path: str,
    model_type: str = 'yolo',
    conf_threshold: float = 0.5,
    **kwargs
) -> Union[YoloMaskGenerator, MaskRCNNMaskGenerator]:
    """
    Factory function to create a mask generator.

    Args:
        model_path: Path to model weights
        model_type: 'yolo' or 'maskrcnn'
        conf_threshold: Confidence threshold for detection
        **kwargs: Additional arguments passed to the generator

    Returns:
        Mask generator instance
    """
    if model_type == 'yolo':
        return YoloMaskGenerator(model_path, conf_threshold, **kwargs)
    elif model_type == 'maskrcnn':
        return MaskRCNNMaskGenerator(model_path, conf_threshold, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_yolo_mask_generator(
    model_path: str,
    conf_threshold: float = 0.5
) -> YoloMaskGenerator:
    """Factory function to create YOLO mask generator."""
    return YoloMaskGenerator(model_path, conf_threshold)


def create_maskrcnn_mask_generator(
    model_path: str,
    conf_threshold: float = 0.5,
    config_file: Optional[str] = None
) -> MaskRCNNMaskGenerator:
    """Factory function to create Mask R-CNN mask generator."""
    return MaskRCNNMaskGenerator(model_path, conf_threshold, config_file=config_file)


# =============================================================================
# CLI Interface
# =============================================================================

def _visualize_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on RGB image with red color and yellow contour."""
    vis = rgb.copy()
    mask_overlay = np.zeros_like(vis)
    mask_overlay[mask > 0] = [255, 0, 0]  # Red
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    contours, _ = cv2.findContours(
        mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)  # Yellow

    return vis


def main():
    """CLI entry point for mask generation visualization."""
    import argparse
    import os
    import glob
    import imageio

    parser = argparse.ArgumentParser(
        description='Generate and visualize segmentation masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model weights (.pt for YOLO, .pth for Mask R-CNN)')
    parser.add_argument('--model_type', type=str, default='yolo',
                        choices=['yolo', 'maskrcnn'],
                        help='Model type')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Detectron2 config yaml (Mask R-CNN only)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='debug/masks',
                        help='Output directory for visualizations')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    args = parser.parse_args()

    # Validate input
    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image_dir")

    # Initialize mask generator
    if args.model_type == 'yolo':
        mask_gen = YoloMaskGenerator(args.model, conf_threshold=args.conf)
    else:
        mask_gen = MaskRCNNMaskGenerator(
            args.model,
            conf_threshold=args.conf,
            config_file=args.config_file
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    if args.image:
        images = [args.image]
    else:
        images = sorted(
            glob.glob(f'{args.image_dir}/*.png') +
            glob.glob(f'{args.image_dir}/*.jpg') +
            glob.glob(f'{args.image_dir}/*.jpeg')
        )

    print(f"Processing {len(images)} images...")

    for img_path in images:
        rgb = imageio.imread(img_path)
        if len(rgb.shape) == 2:  # Grayscale
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] == 4:  # RGBA
            rgb = rgb[:, :, :3]

        mask, info = mask_gen.get_mask(rgb)

        if mask is not None:
            vis = _visualize_mask(rgb, mask)
            status = f"conf={info['confidence']:.2f}"
        else:
            vis = rgb
            status = info.get('error', 'No detection')

        basename = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, basename)
        imageio.imwrite(out_path, vis)
        print(f"  {basename}: {status}")

    print(f"Done! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
