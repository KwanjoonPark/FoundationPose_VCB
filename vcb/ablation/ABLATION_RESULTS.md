# Ablation Study Results

## VCB Handle 6DoF Pose Estimation (n=200 frames)

### Main Results

| Experiment                | Trans MAE | Rot MAE  | Δ Rot    |
|--------------------------:|----------:|---------:|---------:|
| A1: Baseline              |   2.69 cm |  46.80°  |    -     |
| A2: +fix_z                |   2.65 cm |  47.21°  |  +0.41°  |
| A3: +fix_z +z180          |   2.70 cm |  21.14°  | -25.66°  |
| A4: +fix_z +z180 +IoU     |   2.47 cm |  15.56°  |  -5.58°  |
| A5: +fix_z +z180 +IoU +VC |   2.43 cm |  13.96°  |  -1.60°  |
| A6: RGB-only (Final)      |   2.04 cm |   6.22°  |  -7.74°  |
| **Total Improvement**     | **-0.65 cm** | **-40.58°** | **86.7%** |

### Component Contribution

| Component          | Rot Improvement | Contribution |
|-------------------:|----------------:|-------------:|
| z180 symmetry      |        -25.66°  |        54.9% |
| RGB-only mode      |         -7.74°  |        16.5% |
| Mask IoU bonus     |         -5.58°  |        11.9% |
| Vertex Color       |         -1.60°  |         3.4% |
| fix_z alone        |         +0.41°  |   (no effect)|

### Key Findings

1. **z180 Symmetry (54.9%)**: Most significant contribution. Reduces rotation error by 25.66°.
   - VCB handle has 180° rotational ambiguity around Z-axis
   - Without symmetry constraint, half of pose hypotheses are 180° flipped

2. **RGB-only Mode (16.5%)**: Surprising improvement of 7.74° by removing depth.
   - Depth sensor noise may confuse the scorer
   - RGB features are more discriminative for this object
   - xyz_map comparison in scorer may not be well-calibrated

3. **Mask IoU Bonus (11.9%)**: Novel scoring mechanism improves by 5.58°.
   - Computes IoU between rendered mask and detected mask
   - Helps select poses where object shape matches detection
   - Original FoundationPose does not use mask in scoring

4. **Vertex Color (3.4%)**: Modest improvement of 1.60°.
   - Simple vertex colors help distinguish handle from body
   - More effective than complex UV texture mapping for industrial objects

5. **Front Hemisphere Filter**: No improvement alone (+0.41°).
   - Only effective when combined with z180 symmetry
   - Filters back-facing poses for wall-mounted objects

### Experiment Details

- **A1 (Baseline)**: Original FoundationPose settings
  - `fix_z_axis=False, symmetry=none, use_mask_iou=False, mesh=model.obj, input=RGBD`

- **A2 (+fix_z)**: Add front hemisphere filter
  - `fix_z_axis=True, symmetry=none, use_mask_iou=False, mesh=model.obj, input=RGBD`

- **A3 (+z180)**: Add Z-axis 180° symmetry
  - `fix_z_axis=True, symmetry=z180, use_mask_iou=False, mesh=model.obj, input=RGBD`

- **A4 (+IoU)**: Add mask IoU scoring bonus
  - `fix_z_axis=True, symmetry=z180, use_mask_iou=True, mesh=model.obj, input=RGBD`

- **A5 (+VC)**: Use vertex color model
  - `fix_z_axis=True, symmetry=z180, use_mask_iou=True, mesh=model_vc.ply, input=RGBD`

- **A6 (RGB-only)**: Remove depth input
  - `fix_z_axis=True, symmetry=z180, use_mask_iou=True, mesh=model_vc.ply, input=RGB`

### Test Configuration

- Dataset: VCB Handle test scene (200 frames)
- Mask model: Mask R-CNN (rcnn360.pth)
- GPU: Quadro GV100
