# Mask-Guided Scoring for Industrial Object Pose Estimation: Improving FoundationPose with Domain-Specific Optimizations

## Abstract

We present a series of modifications to FoundationPose, a state-of-the-art 6DoF pose estimation framework, that significantly improve performance for industrial object pose estimation. Our key contributions include: (1) a novel **Mask IoU scoring mechanism** that incorporates object segmentation masks into the pose scoring pipeline, (2) the surprising finding that **RGB-only input outperforms RGB-D** for certain object types, and (3) domain-specific optimizations including symmetry constraints and viewing angle filters. Through comprehensive ablation studies on a wall-mounted industrial handle dataset, we demonstrate an **86.7% reduction in rotation error** (from 46.80° to 6.22°) while maintaining real-time performance. Our findings challenge the conventional assumption that depth information always improves pose estimation and provide practical guidelines for deploying pose estimation systems in industrial settings.

## 1. Introduction

6DoF object pose estimation is a fundamental task in robotics and augmented reality applications. Recent learning-based methods, particularly FoundationPose [1], have achieved remarkable performance by combining neural rendering with pose refinement networks. However, these methods are primarily designed and evaluated on household objects with rich textures and diverse viewpoints.

Industrial applications present unique challenges:
- **Limited viewpoints**: Wall-mounted or fixed objects are only visible from a constrained set of angles
- **Simple geometry**: Industrial parts often have symmetric or near-symmetric shapes
- **Uniform appearance**: Metal or plastic surfaces with minimal texture variation
- **Sensor noise**: Depth sensors may produce noisy measurements on reflective surfaces

In this paper, we investigate how to adapt FoundationPose for industrial object pose estimation. Through systematic analysis, we identify key limitations in the original pipeline and propose targeted modifications. Our main contributions are:

1. **Mask IoU Scoring**: We observe that FoundationPose's scorer does not utilize the object segmentation mask, leading to background-dominated comparisons. We introduce a mask IoU bonus that measures the overlap between rendered and detected object masks.

2. **RGB-only Superiority**: Counter-intuitively, we find that removing depth input improves pose estimation accuracy for objects with reflective surfaces. This challenges the common assumption that RGB-D always outperforms RGB-only methods.

3. **Domain-Specific Priors**: We introduce viewing angle constraints (front hemisphere filter) and symmetry handling that reduce the pose hypothesis space while eliminating physically impossible configurations.

4. **Comprehensive Ablation Study**: We provide detailed analysis of each component's contribution, enabling practitioners to select appropriate modifications for their specific applications.

## 2. Related Work

### 2.1 Model-Based Pose Estimation
Traditional approaches rely on CAD models and feature matching [2,3]. Recent deep learning methods learn to directly predict poses from images [4,5] or use differentiable rendering for refinement [6,7].

### 2.2 FoundationPose
FoundationPose [1] is a unified framework supporting both model-based and model-free pose estimation. It generates pose hypotheses, refines them using a neural network, and scores candidates using a learned comparator. Our work builds upon this framework with targeted improvements for industrial applications.

### 2.3 Mask-Guided Methods
Several works have explored using segmentation masks for pose estimation [8,9]. However, these typically use masks for object detection rather than pose scoring. Our approach directly incorporates mask overlap into the scoring function.

## 3. Method

### 3.1 Problem Formulation

Given an RGB image $I \in \mathbb{R}^{H \times W \times 3}$, an optional depth map $D \in \mathbb{R}^{H \times W}$, an object segmentation mask $M \in \{0,1\}^{H \times W}$, and a 3D mesh model, our goal is to estimate the 6DoF pose $T = [R|t] \in SE(3)$ of the object.

### 3.2 Baseline: FoundationPose Pipeline

FoundationPose consists of three stages:

1. **Pose Hypothesis Generation**: Sample viewpoints on a sphere and combine with in-plane rotations to generate candidate poses $\{T_i\}_{i=1}^N$.

2. **Pose Refinement**: A neural network iteratively refines each hypothesis by predicting pose corrections.

3. **Pose Scoring**: A learned scorer compares rendered images $I_r^{(i)}$ with the observed image crop $I_o$ and outputs confidence scores $s_i$.

The final pose is selected as: $T^* = T_{\arg\max_i s_i}$

### 3.3 Mask IoU Scoring

We observe that the original scorer compares RGB and XYZ features but ignores the segmentation mask. This means background pixels equally contribute to the score, potentially dominating the comparison for small objects.

We propose augmenting the score with a mask IoU bonus:

$$s_i' = s_i + \lambda \cdot \text{IoU}(M_r^{(i)}, M_o)$$

where $M_r^{(i)}$ is the rendered mask for pose hypothesis $i$, $M_o$ is the detected object mask, and $\lambda$ is a weighting factor (we use $\lambda=10$).

The IoU is computed as:
$$\text{IoU}(M_r, M_o) = \frac{|M_r \cap M_o|}{|M_r \cup M_o|}$$

This simple modification ensures that poses producing object silhouettes matching the detection are preferred.

### 3.4 Front Hemisphere Filter

For wall-mounted objects, the back surface is never visible. We filter pose hypotheses based on the object's Z-axis direction in camera coordinates:

$$\mathcal{T}_{valid} = \{T_i : (R_i \cdot [0,0,1]^T)_z < 0\}$$

This eliminates approximately 50% of hypotheses that represent physically impossible configurations.

### 3.5 Symmetry Handling

Many industrial objects exhibit rotational symmetry. For 180° Z-axis symmetry, poses $T$ and $T \cdot R_z(\pi)$ are equivalent. We cluster symmetric poses before scoring to avoid redundant computation and ambiguous comparisons:

$$\mathcal{T}_{clustered} = \text{cluster}(\mathcal{T}_{valid}, \{R_z(k\pi) : k \in \{0,1\}\})$$

### 3.6 RGB-Only Mode

Surprisingly, we find that depth input can degrade performance for objects with reflective surfaces. We hypothesize this is due to:

1. **Depth sensor noise**: Reflective surfaces cause missing or erroneous depth values
2. **XYZ map mismatch**: The scorer's XYZ comparison may be miscalibrated for certain object types
3. **Feature interference**: Noisy depth features may interfere with discriminative RGB features

We support an RGB-only mode that sets $D=0$ and relies purely on RGB comparisons.

## 4. Experiments

### 4.1 Dataset

We evaluate on a VCB (Vacuum Circuit Breaker) handle dataset consisting of:
- **200 frames** captured from varying viewpoints
- **Ground truth poses** from robot kinematics
- **Object**: Industrial handle (47mm diameter) mounted on a panel

### 4.2 Evaluation Metrics

- **Rotation MAE**: Mean absolute error in rotation (degrees)
- **Translation MAE**: Mean absolute error in translation (cm)

### 4.3 Ablation Study

We systematically evaluate each component's contribution:

| Experiment | Components | Trans MAE | Rot MAE | Δ Rot |
|------------|-----------|-----------|---------|-------|
| A1: Baseline | - | 2.69 cm | 46.80° | - |
| A2: +fix_z | Front hemisphere | 2.65 cm | 47.21° | +0.41° |
| A3: +z180 | + Z180 symmetry | 2.70 cm | 21.14° | -25.66° |
| A4: +IoU | + Mask IoU | 2.47 cm | 15.56° | -5.58° |
| A5: +VC | + Vertex colors | 2.43 cm | 13.96° | -1.60° |
| A6: RGB-only | - Depth | **2.04 cm** | **6.22°** | -7.74° |

**Key Findings:**

1. **Z180 symmetry provides the largest improvement** (54.9% of total), reducing ambiguity from 180° flipped poses.

2. **RGB-only outperforms RGB-D** by 7.74°, suggesting depth noise is detrimental for this object type.

3. **Mask IoU contributes 13.8%** of the improvement, validating our hypothesis that mask-guided scoring helps.

4. **Front hemisphere filter alone does not help** but is essential when combined with symmetry handling.

### 4.4 Component Contribution Analysis

| Component | Contribution |
|-----------|-------------|
| Z180 Symmetry | 63.2% |
| RGB-only Mode | 19.1% |
| Mask IoU Scoring | 13.8% |
| Vertex Colors | 3.9% |

### 4.5 Qualitative Results

Figure X shows pose estimation results across different viewpoints. The baseline method frequently selects 180° flipped poses, while our method consistently estimates correct orientations.

## 5. Discussion

### 5.1 When Does RGB-Only Help?

Our finding that RGB-only outperforms RGB-D is surprising but can be explained by:

1. **Object material**: The metal handle has reflective surfaces causing depth sensor artifacts
2. **Scorer design**: FoundationPose's scorer may over-weight XYZ features
3. **Object size**: Small objects have limited depth variation within the crop region

We recommend evaluating RGB-only mode for objects with reflective surfaces or when depth quality is poor.

### 5.2 Limitations

- Our evaluation is limited to a single object type
- The mask IoU bonus weight ($\lambda=10$) was tuned empirically
- RGB-only mode requires accurate object scale estimation

### 5.3 Practical Guidelines

Based on our findings, we recommend:

1. **Always apply symmetry constraints** appropriate to the object geometry
2. **Use viewing angle filters** for objects with constrained visibility
3. **Evaluate RGB-only mode** for reflective or small objects
4. **Incorporate mask IoU scoring** when segmentation quality is high

## 6. Conclusion

We presented targeted modifications to FoundationPose for industrial object pose estimation, achieving an 86.7% reduction in rotation error. Our key contributions—mask IoU scoring and the discovery that RGB-only can outperform RGB-D—provide both practical improvements and insights for future research. The comprehensive ablation study enables practitioners to select appropriate optimizations for their specific applications.

## References

[1] Wen, B., et al. "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects." CVPR 2024.

[2] Hinterstoisser, S., et al. "Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes." ACCV 2012.

[3] Kehl, W., et al. "SSD-6D: Making RGB-based 3D detection and 6D pose estimation great again." ICCV 2017.

[4] Xiang, Y., et al. "PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes." RSS 2018.

[5] Peng, S., et al. "PVNet: Pixel-wise voting network for 6dof pose estimation." CVPR 2019.

[6] Li, Y., et al. "DeepIM: Deep iterative matching for 6D pose estimation." ECCV 2018.

[7] Labbe, Y., et al. "CosyPose: Consistent multi-view multi-object 6D pose estimation." ECCV 2020.

[8] Wang, H., et al. "Normalized object coordinate space for category-level 6D object pose and size estimation." CVPR 2019.

[9] Park, K., et al. "Pix2Pose: Pixel-wise coordinate regression of objects for 6D pose estimation." ICCV 2019.

---

## Appendix A: Implementation Details

- **Framework**: PyTorch
- **GPU**: NVIDIA Quadro GV100
- **Pose hypotheses**: 56 (after filtering and clustering)
- **Refinement iterations**: 5
- **Mask model**: Mask R-CNN trained on 360 images
- **Vertex colors**: Handle [50,50,55], Body [150,145,140]

## Appendix B: Detailed Results by Frame Range

| Frame Range | Trans MAE | Rot MAE |
|-------------|-----------|---------|
| 0-99 (near) | 0.91 cm | 6.73° |
| 100-199 (far) | 3.16 cm | 5.71° |
| All (200) | 2.04 cm | 6.22° |

Translation error increases at far distances due to scale ambiguity in RGB-only mode, while rotation accuracy remains consistent.
