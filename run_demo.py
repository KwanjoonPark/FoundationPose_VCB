# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--mesh_scale', type=float, default=1.0, help='Scale factor for mesh (e.g., 0.01 for cm to m)')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
  if args.mesh_scale != 1.0:
    mesh.apply_scale(args.mesh_scale)
    logging.info(f"Mesh scaled by {args.mesh_scale} (new extents: {mesh.extents})")

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  # Use axis-aligned bounding box instead of oriented_bounds
  # oriented_bounds rotates the box which doesn't match our defined coordinate system
  extents = mesh.extents  # axis-aligned extents
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    # No coordinate transformation - use raw pose
    # (Different GT may use different conventions)
    pose_gt = pose.copy()

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose_gt.reshape(4,4))

    # Save as cam_in_ob (inverse of ob_in_cam) for comparison with ground truth
    os.makedirs(f'{debug_dir}/cam_in_ob', exist_ok=True)
    cam_in_ob = np.linalg.inv(pose_gt)
    np.savetxt(f'{debug_dir}/cam_in_ob/{reader.id_strs[i]}.txt', cam_in_ob.reshape(4,4))

    if debug>=1:
      # Use pose directly since we're using axis-aligned bbox (no to_origin transform)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=pose, bbox=bbox)
      # Use axis scale proportional to object size (reduced for better visibility)
      axis_scale = max(extents) * 1  # 1x the largest dimension
      # Flip Y and Z axes for visualization (Rx_180)
      Rx_180_vis = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float64)
      pose_vis = pose @ Rx_180_vis
      vis = draw_xyz_axis(vis, ob_in_cam=pose_vis, scale=axis_scale, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)

      # Display 6DoF on top-left corner
      from scipy.spatial.transform import Rotation as R
      trans = pose_gt[:3, 3]
      rot_euler = R.from_matrix(pose_gt[:3, :3]).as_euler('xyz', degrees=True)

      # Convert to BGR for cv2.putText
      vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.6
      color_text = (0, 255, 0)  # Green
      thickness_text = 2

      cv2.putText(vis_bgr, f'X: {trans[0]*100:+6.2f} cm', (10, 25), font, font_scale, color_text, thickness_text)
      cv2.putText(vis_bgr, f'Y: {trans[1]*100:+6.2f} cm', (10, 50), font, font_scale, color_text, thickness_text)
      cv2.putText(vis_bgr, f'Z: {trans[2]*100:+6.2f} cm', (10, 75), font, font_scale, color_text, thickness_text)
      cv2.putText(vis_bgr, f'Roll:  {rot_euler[0]:+7.2f} deg', (10, 105), font, font_scale, color_text, thickness_text)
      cv2.putText(vis_bgr, f'Pitch: {rot_euler[1]:+7.2f} deg', (10, 130), font, font_scale, color_text, thickness_text)
      cv2.putText(vis_bgr, f'Yaw:   {rot_euler[2]:+7.2f} deg', (10, 155), font, font_scale, color_text, thickness_text)

      vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
      #cv2.imshow('1', vis[...,::-1])
      #cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

