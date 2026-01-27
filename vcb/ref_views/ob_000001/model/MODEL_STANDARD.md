# VCB Mesh Standard Coordinate System

## Size (unit: cm, use --mesh_scale 0.01 for FoundationPose)
| Axis | Size | Description |
|------|------|-------------|
| X | 4.01 | Handle direction |
| Y | 1.80 | Thickness |
| Z | 4.40 | Body length |

## Origin
- Location: Mesh center (0, 0, 0)

## Bounding Box
| Axis | Min | Max |
|------|-----|-----|
| X | -2.005 | +2.005 |
| Y | -0.900 | +0.900 |
| Z | -2.645 | +1.755 |

## Axis Directions
```
        +Y (Up)
         |
         |
         +------ +X (Handle Right)
        /
       /
     +Z (Body Front)
```

| Direction | Description |
|-----------|-------------|
| +X | Handle right side |
| -X | Handle left side |
| +Y | Up |
| -Y | Down |
| +Z | Body front |
| -Z | Body back |

## Usage with FoundationPose
```bash
python run_demo.py \
  --mesh_file vcb/ref_views/ob_000001/model/model.obj \
  --mesh_scale 0.01 \
  --test_scene_dir <your_data_dir>
```

## Mesh Info
- Vertices: 1001
- Faces: 1420
- Components: 11
