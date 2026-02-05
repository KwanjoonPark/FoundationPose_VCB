#!/bin/bash
# Ablation Study for FoundationPose VCB Handle

MASK_MODEL="vcb/rcnn360.pth"
MESH_OBJ="vcb/ref_views/ob_000001/model/model.obj"
MESH_VC="vcb/ref_views/ob_000001/model/model_vc.ply"

echo "=========================================="
echo "Ablation Study Start"
echo "=========================================="

# A1: Baseline (no fix_z, no z180, no IoU, model.obj, RGBD)
echo ""
echo "[A1] Baseline"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_OBJ \
    --fix_z_axis False --symmetry none --use_mask_iou False \
    --input_mode rgbd \
    --debug_dir vcb/ablation/A1_baseline --debug 1

# A2: +fix_z
echo ""
echo "[A2] +fix_z"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_OBJ \
    --fix_z_axis True --symmetry none --use_mask_iou False \
    --input_mode rgbd \
    --debug_dir vcb/ablation/A2_fix_z --debug 1

# A3: +z180
echo ""
echo "[A3] +fix_z +z180"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_OBJ \
    --fix_z_axis True --symmetry z180 --use_mask_iou False \
    --input_mode rgbd \
    --debug_dir vcb/ablation/A3_z180 --debug 1

# A4: +IoU
echo ""
echo "[A4] +fix_z +z180 +IoU"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_OBJ \
    --fix_z_axis True --symmetry z180 --use_mask_iou True \
    --input_mode rgbd \
    --debug_dir vcb/ablation/A4_iou --debug 1

# A5: +Vertex Color
echo ""
echo "[A5] +fix_z +z180 +IoU +VC (RGBD)"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_VC \
    --fix_z_axis True --symmetry z180 --use_mask_iou True \
    --input_mode rgbd \
    --debug_dir vcb/ablation/A5_vc_rgbd --debug 1

# A6: RGB-only (Final)
echo ""
echo "[A6] +fix_z +z180 +IoU +VC (RGB-only)"
python run_est.py --mask_model $MASK_MODEL \
    --mesh_file $MESH_VC \
    --fix_z_axis True --symmetry z180 --use_mask_iou True \
    --input_mode rgb \
    --debug_dir vcb/ablation/A6_rgb --debug 1

echo ""
echo "=========================================="
echo "Ablation Study Complete"
echo "=========================================="

# Compare all results
echo ""
echo "Results Summary:"
for dir in vcb/ablation/A*; do
    name=$(basename $dir)
    echo ""
    echo "=== $name ==="
    python compare_poses.py --est_dir $dir/ob_in_cam --gt_dir vcb/ref_views/test_scene/ob_in_cam 2>&1 | grep -A3 "All frames"
done
