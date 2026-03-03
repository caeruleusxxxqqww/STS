#!/usr/bin/env python3
"""
📊 Qualitative Comparison Visualization Script

- Row 4: Ground Truth


   python scripts/visualize_qualitative_comparison.py \
       --input data/s3dis/raw/Area_5_office_1.npy \
       --ckpt-baseline results/baseline/pointnext-b_ckpt_best.pth \
       --ckpt-ours results/ours/pointnext-b_ckpt_best.pth \
       --config-baseline cfgs/s3dis/pointnext-b-baseline.yaml \
       --config-ours cfgs/s3dis/pointnext-b-mixratio.yaml \
       --output qualitative_comparison.png \
       --dataset s3dis \
       --show-zoom

   python scripts/visualize_qualitative_comparison.py \
       --input data/s3dis/raw/Area_5_office_1.npy \
       --pred-baseline results/baseline/Area_5_office_1_pred.npy \
       --pred-ours results/ours/Area_5_office_1_pred.npy \
       --output qualitative_comparison.png \
       --dataset s3dis \
       --show-zoom

   from scripts.visualize_qualitative_comparison import visualize_from_predictions_dir

   visualize_from_predictions_dir(
       data_path='data/s3dis/raw/Area_5_office_1.npy',
       predictions_dir='results/predictions',
       output_dir='visualizations',
       dataset_name='s3dis',
       scene_name='Area_5_office_1',
       show_zoom_in=True
   )

Author: Generated for STS Paper
"""

import argparse
import os
import sys
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# isort: split
if True:
    from openpoints.utils import EasyConfig, load_checkpoint
    from openpoints.transforms import build_transforms_from_cfg
    from openpoints.models import build_model_from_cfg
    from openpoints.dataset import get_features_by_keys



try:
    from scipy.spatial.distance import pdist
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] sklearn not found. Error region clustering will be disabled.")


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# ============================================================================

# ============================================================================
# ============================================================================


def get_colormap_from_matplotlib(num_classes, colormap_name='tab20'):
    """


    Args:

    Returns:
    """
    import matplotlib.pyplot as plt

    if num_classes <= 4:
        cmap = plt.cm.Set1
    else:
        cmap = plt.cm.tab20

    colors = cmap(np.linspace(0, 1, num_classes))

    colormap = colors[:, :3]

    return colormap


S3DIS_COLORMAP = get_colormap_from_matplotlib(13, 'tab20')

SCANNET_COLORMAP = get_colormap_from_matplotlib(20, 'tab20')

VIS_CONFIG = {
    'point_size': 0.03,
    'bg_point_size': 0.02,
    'error_alpha': 0.3,
    'correct_alpha': 0.3,
    'error_color': [0.85, 0.15, 0.15],
    'correct_color': [0.0, 0.60, 0.20],
    'zoom_scale': 2.0,
    'render_width': 1200,
    'render_height': 900,
}


# ============================================================================
# ============================================================================

def render_pointcloud_open3d(points, colors, output_path=None,
                             width=VIS_CONFIG['render_width'],
                             height=VIS_CONFIG['render_height'],
                             point_size=VIS_CONFIG['point_size'],
                             camera_pose=None,
                             background_color=[1.0, 1.0, 1.0],
                             return_camera_params=False,
                             return_depth=False,
                             norm_center=None,
                             norm_scale=None,
                             norm_factor=1.0):
    """

    Args:

    Returns:
    """
    if len(points) == 0:
        raise ValueError("ERROR: Empty point cloud! Cannot render.")
    print(
        f"[Debug] Rendering {len(points)} points, point cloud range: min={points.min(axis=0)}, max={points.max(axis=0)}")

    if norm_center is not None and norm_scale is not None:
        center = norm_center
        scale = norm_scale
    else:
        center = points.mean(axis=0)
        scale = np.abs(points - center).max() + 1e-6

    effective_norm_factor = norm_factor if norm_factor is not None else 1.0
    points_norm = (points - center) / scale * effective_norm_factor
    print(
        f"[Debug] Normalized point cloud range: min={points_norm.min(axis=0)}, max={points_norm.max(axis=0)}, center={center}, scale={scale}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_norm)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.]))

    depth_map = None

    use_open3d_success = False

    if hasattr(o3d.visualization.rendering, 'OffscreenRenderer'):
        try:
            print("[Info] Using Open3D OffscreenRenderer for rendering...")
            renderer = o3d.visualization.rendering.OffscreenRenderer(
                width, height)

            material = o3d.visualization.rendering.MaterialRecord()
            material.point_size = max(point_size * 200, 10.0)
            print(f"[Debug] Point size set to: {material.point_size} pixels")

            if len(pcd.points) == 0:
                raise ValueError("Point cloud is empty after normalization!")
            print(f"[Debug] Adding {len(pcd.points)} points to scene")

            renderer.scene.add_geometry("pointcloud", pcd, material)

            if len(background_color) == 3:
                bg_color_rgba = np.array([background_color[0], background_color[1],
                                         background_color[2], 1.0], dtype=np.float32).reshape(4, 1)
            else:
                bg_color_rgba = np.array(
                    background_color, dtype=np.float32).reshape(4, 1)
            renderer.scene.set_background(bg_color_rgba)

            camera = renderer.scene.camera
            camera_setup_success = False

            if camera_pose is None:
                bounds = pcd.get_axis_aligned_bounding_box()
                bounds_center = bounds.get_center()
                extent = bounds.get_extent()
                max_extent = max(extent) if max(extent) > 0 else 1.0
                print(
                    f"[Debug] Point cloud bounds: center={bounds_center}, extent={extent}, max_extent={max_extent}")

                near_plane = 0.01
                far_plane = 10.0
                try:
                    camera.set_projection(45.0, width / height, near_plane, far_plane,
                                          o3d.visualization.rendering.Camera.FovType.Vertical)
                    print(
                        f"[Debug] Camera projection set: fov=45, near={near_plane}, far={far_plane}")
                except Exception as e:
                    print(
                        f"[Warning] Failed to set projection: {e}. Using default.")
                    try:
                        camera.set_projection(45.0, width / height, near_plane, far_plane,
                                              o3d.visualization.rendering.Camera.FovType.Horizontal)
                    except:
                        pass

                camera_distance = 2.5
                iso_angle_xy = np.pi / 4 + np.pi
                iso_angle_z = np.arctan(1 / np.sqrt(2))

                xy_distance = camera_distance * np.cos(iso_angle_z)
                eye_offset_x = xy_distance * np.cos(iso_angle_xy)
                eye_offset_y = xy_distance * np.sin(iso_angle_xy)
                eye_offset_z = camera_distance * np.sin(iso_angle_z)

                eye = bounds_center + \
                    np.array([eye_offset_x, eye_offset_y, eye_offset_z])
                up_vector = np.array([0, 0, 1])
                print(
                    f"[Debug] Camera setup (ISO view): eye={eye}, center={bounds_center}, up={up_vector}")
                try:
                    camera.look_at(bounds_center, eye, up_vector)
                    camera_setup_success = True
                    print(f"[Debug] Camera look_at set successfully")
                except Exception as e:
                    print(
                        f"[Warning] Failed to set camera look_at: {e}. Will use default view.")

                camera_params = o3d.camera.PinholeCameraParameters()
                camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width, height, width * 0.7, height * 0.7, width / 2, height / 2)
                camera_params.extrinsic = np.eye(4)
            else:
                print(
                    f"[Debug] Camera pose provided, but recalculating based on normalized point cloud")

                bounds = pcd.get_axis_aligned_bounding_box()
                bounds_center = bounds.get_center()
                extent = bounds.get_extent()
                max_extent = max(extent) if max(extent) > 0 else 1.0

                near_plane = 0.01
                far_plane = 10.0
                try:
                    camera.set_projection(45.0, width / height, near_plane, far_plane,
                                          o3d.visualization.rendering.Camera.FovType.Vertical)
                    print(
                        f"[Debug] Camera projection set: fov=45, near={near_plane}, far={far_plane}")
                except Exception as e:
                    print(
                        f"[Warning] Failed to set projection: {e}. Using default.")
                    try:
                        camera.set_projection(45.0, width / height, near_plane, far_plane,
                                              o3d.visualization.rendering.Camera.FovType.Horizontal)
                    except:
                        pass

                try:
                    extrinsic = camera_pose.extrinsic
                    R = extrinsic[:3, :3]

                    extent_max = max(extent) if max(extent) > 0 else 1.0
                    if extent_max < 0.5:
                        camera_distance = 1.5
                    elif extent_max < 1.0:
                        camera_distance = 1.8
                    else:
                        camera_distance = 2.0

                    t = extrinsic[:3, 3]
                    camera_position_world = -R.T @ t

                    camera_forward = R.T @ np.array([0, 0, 1])
                    camera_forward = camera_forward / \
                        np.linalg.norm(camera_forward)

                    camera_up_world = R.T @ np.array([0, 1, 0])
                    camera_up_world = camera_up_world / \
                        np.linalg.norm(camera_up_world)

                    if np.linalg.norm(camera_position_world) < 0.1:
                        eye = bounds_center - camera_forward * camera_distance
                    else:
                        eye = camera_position_world + center
                        if np.linalg.norm(eye) > 10:
                            eye = camera_position_world

                    camera_up = camera_up_world

                    print(
                        f"[Debug] Camera from pose: eye={eye}, center={bounds_center}, up={camera_up}")
                    camera.look_at(bounds_center, eye, camera_up)
                    camera_setup_success = True
                    print(f"[Debug] Camera look_at from pose set successfully")
                except Exception as e:
                    print(
                        f"[Warning] Failed to set camera from pose: {e}. Using default view.")
                    camera_distance = 2.5
                    iso_angle_xy = np.pi / 4 + np.pi
                    iso_angle_z = np.arctan(1 / np.sqrt(2))
                    xy_distance = camera_distance * np.cos(iso_angle_z)
                    eye_offset_x = xy_distance * np.cos(iso_angle_xy)
                    eye_offset_y = xy_distance * np.sin(iso_angle_xy)
                    eye_offset_z = camera_distance * np.sin(iso_angle_z)
                    eye = bounds_center + \
                        np.array([eye_offset_x, eye_offset_y, eye_offset_z])
                    up_vector = np.array([0, 0, 1])
                    try:
                        camera.look_at(bounds_center, eye, up_vector)
                        camera_setup_success = True
                    except:
                        pass

                camera_params = o3d.camera.PinholeCameraParameters()
                camera_params.intrinsic = camera_pose.intrinsic
                camera_params.extrinsic = np.eye(4)


            try:
                img_o3d = renderer.render_to_image()
                if isinstance(img_o3d, np.ndarray):
                    img = img_o3d
                else:
                    img = np.asarray(img_o3d)

                print(
                    f"[Debug] Rendered image shape: {img.shape}, dtype: {img.dtype}, min={img.min()}, max={img.max()}")

                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                unique_values = np.unique(img)
                print(
                    f"[Debug] Converted image: shape={img.shape}, unique values count={len(unique_values)}, min={img.min()}, max={img.max()}")
                if len(unique_values) <= 1:
                    print(
                        f"[Warning] Rendered image is uniform (all pixels = {unique_values[0] if len(unique_values) > 0 else 'N/A'})")
                elif len(unique_values) <= 10:
                    print(
                        f"[Warning] Rendered image has very few unique values: {unique_values}")

                use_open3d_success = True
                print("[Info] Open3D OffscreenRenderer rendering successful!")

                del renderer

            except Exception as render_error:
                print(
                    f"[Warning] OffscreenRenderer rendering failed: {render_error}")
                del renderer
                raise render_error

        except Exception as e:
            print(
                f"[Warning] OffscreenRenderer failed: {e}")
            use_open3d_success = False

    if not use_open3d_success:
        is_headless = (
            os.environ.get('DISPLAY') is None or
            os.environ.get('DISPLAY') == '' or
            os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or
            os.environ.get('OPEN3D_HEADLESS') == '1'
        )

        if is_headless:
            print(
                "[Info] Detected headless environment. Skipping Visualizer, using matplotlib fallback...")
            img, camera_params = render_pointcloud_matplotlib_fallback(
                points_norm, colors, width, height, camera_pose)
            if return_depth:
                depth_map = None
            use_open3d_success = False
        else:
            try:
                print("[Info] Trying Open3D Visualizer as fallback...")
                vis = o3d.visualization.Visualizer()
                success = vis.create_window(
                    width=width, height=height, visible=False)

                if not success:
                    raise RuntimeError("Failed to create window")

                vis.add_geometry(pcd)

                render_option = vis.get_render_option()
                if render_option is None:
                    raise RuntimeError("Failed to get render option")

                render_option.background_color = np.array(background_color)
                render_option.point_size = point_size

                if camera_pose is None:
                    ctr = vis.get_view_control()
                    if ctr is None:
                        raise RuntimeError("Failed to get view control")
                    ctr.set_zoom(0.7)
                    iso_angle_xy = np.pi / 4 + np.pi
                    iso_angle_z = np.arctan(1 / np.sqrt(2))
                    xy_dist = np.cos(iso_angle_z)
                    front_x = -xy_dist * np.cos(iso_angle_xy)
                    front_y = -xy_dist * np.sin(iso_angle_xy)
                    front_z = -np.sin(iso_angle_z)
                    ctr.set_front([front_x, front_y, front_z])
                    ctr.set_lookat(points_norm.mean(axis=0))
                    ctr.set_up([0, 0, 1])
                else:
                    ctr = vis.get_view_control()
                    if ctr is None:
                        raise RuntimeError("Failed to get view control")
                    extrinsic = camera_pose.extrinsic
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    camera_center = -R.T @ t
                    camera_forward = -R.T @ np.array([0, 0, 1])
                    camera_up = np.array([0, 0, 1])
                    ctr.set_front(camera_forward)
                    ctr.set_lookat(camera_center +
                                   camera_forward * 2.0)
                    ctr.set_up(camera_up)

                img = np.asarray(
                    vis.capture_screen_float_buffer(do_render=True))
                img = (img * 255).astype(np.uint8)

                depth_map = None
                if return_depth:
                    try:
                        depth_buffer = vis.capture_depth_float_buffer(
                            do_render=True)
                        depth_map = np.asarray(depth_buffer)
                    except Exception as e:
                        print(f"[Warning] Failed to get depth map: {e}")

                vis.destroy_window()

                use_open3d_success = True
                print("[Info] Open3D Visualizer rendering successful!")

            except Exception as e2:
                print(
                    f"[Warning] All Open3D rendering methods failed: {e2}. Using matplotlib fallback...")
                img, camera_params = render_pointcloud_matplotlib_fallback(
                    points_norm, colors, width, height, camera_pose)
                if return_depth:
                    depth_map = None
                use_open3d_success = False

    if output_path:
        plt.imsave(output_path, img)

    if return_camera_params and return_depth:
        return img, camera_params, center, scale, depth_map
    elif return_camera_params:
        return img, camera_params, center, scale
    elif return_depth:
        return img, depth_map
    return img


def render_pointcloud_matplotlib_fallback(points_norm, colors, width, height, camera_pose=None):
    """

    Args:

    Returns:
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_norm[:, 0], points_norm[:, 1], points_norm[:, 2],
               c=colors, s=1, alpha=0.6)

    max_range = np.array([points_norm[:, 0].max() - points_norm[:, 0].min(),
                          points_norm[:, 1].max() - points_norm[:, 1].min(),
                          points_norm[:, 2].max() - points_norm[:, 2].min()]).max() / 2.0
    mid_x = (points_norm[:, 0].max() + points_norm[:, 0].min()) * 0.5
    mid_y = (points_norm[:, 1].max() + points_norm[:, 1].min()) * 0.5
    mid_z = (points_norm[:, 2].max() + points_norm[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    try:
        from PIL import Image
        img_pil = Image.fromarray(img)
        if hasattr(Image, 'Resampling'):
            img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
        else:
            img_pil = img_pil.resize((width, height), Image.LANCZOS)
        img = np.array(img_pil)
    except ImportError:
        from scipy.ndimage import zoom
        scale_x = width / img.shape[1]
        scale_y = height / img.shape[0]
        img = zoom(img, (scale_y, scale_x, 1), order=1)
        img = (img.clip(0, 255)).astype(np.uint8)

    plt.close(fig)

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, width * 0.7, height * 0.7, width / 2, height / 2)
    camera_params.extrinsic = np.eye(4)

    return img, camera_params


def get_camera_pose_for_view(points, view='default'):
    """

    Args:
        view: 'default', 'top', 'side', 'front'

    Returns:
        o3d.camera.PinholeCameraParameters
    """
    center = points.mean(axis=0)
    scale = np.abs(points - center).max() + 1e-6

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        VIS_CONFIG['render_width'], VIS_CONFIG['render_height'],
        VIS_CONFIG['render_width'] * 0.7, VIS_CONFIG['render_height'] * 0.7,
        VIS_CONFIG['render_width'] / 2, VIS_CONFIG['render_height'] / 2)

    if view == 'top':
        camera_params.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, scale * 2],
            [0, 0, 0, 1]
        ])
    elif view == 'side':
        camera_params.extrinsic = np.array([
            [0, 0, -1, scale * 2],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
    elif view == 'front':
        camera_params.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, scale * 2],
            [0, 0, 0, 1]
        ])
    else:
        iso_distance = scale * 2.5
        iso_angle_xy = np.pi / 4 + np.pi
        iso_angle_z = np.arctan(1 / np.sqrt(2))

        xy_distance = iso_distance * np.cos(iso_angle_z)
        eye_x = xy_distance * np.cos(iso_angle_xy)
        eye_y = xy_distance * np.sin(iso_angle_xy)
        eye_z = iso_distance * np.sin(iso_angle_z)

        eye = np.array([eye_x, eye_y, eye_z])
        target = np.array([0, 0, 0])

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1])

        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R_world_to_cam = np.column_stack([right, up, -forward])

        t = -R_world_to_cam @ eye

        camera_params.extrinsic = np.eye(4)
        camera_params.extrinsic[:3, :3] = R_world_to_cam
        camera_params.extrinsic[:3, 3] = t

    return camera_params


def render_pointcloud_matplotlib_fallback(points_norm, colors, width, height, camera_pose=None):
    """

    Args:

    Returns:
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_norm[:, 0], points_norm[:, 1], points_norm[:, 2],
               c=colors, s=1, alpha=0.6)

    if camera_pose is None:
        ax.view_init(elev=30, azim=45 + 180)
    else:
        ax.view_init(elev=30, azim=45 + 180)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.set_axis_off()

    max_range = np.array([points_norm[:, 0].max() - points_norm[:, 0].min(),
                          points_norm[:, 1].max() - points_norm[:, 1].min(),
                          points_norm[:, 2].max() - points_norm[:, 2].min()]).max() / 2.0
    mid_x = (points_norm[:, 0].max() + points_norm[:, 0].min()) * 0.5
    mid_y = (points_norm[:, 1].max() + points_norm[:, 1].min()) * 0.5
    mid_z = (points_norm[:, 2].max() + points_norm[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    try:
        from PIL import Image
        img_pil = Image.fromarray(img)
        if hasattr(Image, 'Resampling'):
            img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
        else:
            img_pil = img_pil.resize((width, height), Image.LANCZOS)
        img = np.array(img_pil)
    except ImportError:
        from scipy.ndimage import zoom
        scale_x = width / img.shape[1]
        scale_y = height / img.shape[0]
        img = zoom(img, (scale_y, scale_x, 1), order=1)
        img = (img.clip(0, 255)).astype(np.uint8)

    plt.close(fig)

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, width * 0.7, height * 0.7, width / 2, height / 2)
    camera_params.extrinsic = np.eye(4)

    return img, camera_params


def get_camera_pose_for_view(points, view='default'):
    """

    Args:
        view: 'default', 'top', 'side', 'front'

    Returns:
        o3d.camera.PinholeCameraParameters
    """
    center = points.mean(axis=0)
    scale = np.abs(points - center).max() + 1e-6

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        VIS_CONFIG['render_width'], VIS_CONFIG['render_height'],
        VIS_CONFIG['render_width'] * 0.7, VIS_CONFIG['render_height'] * 0.7,
        VIS_CONFIG['render_width'] / 2, VIS_CONFIG['render_height'] / 2)

    if view == 'top':
        camera_params.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, scale * 2],
            [0, 0, 0, 1]
        ])
    elif view == 'side':
        camera_params.extrinsic = np.array([
            [0, 0, 1, scale * 2],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    elif view == 'front':
        camera_params.extrinsic = np.array([
            [1, 0, 0, scale * 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    else:  # default
        camera_params.extrinsic = np.array([
            [0.7, -0.3, 0.6, scale * 1.5],
            [0.3, 0.9, 0.3, -scale * 0.5],
            [-0.6, 0.3, 0.7, scale * 1.2],
            [0, 0, 0, 1]
        ])

    return camera_params


# ============================================================================
# ============================================================================

def find_error_regions(points, pred, gt, min_points=50, use_clustering=True):
    """

    Args:

    Returns:
    """
    error_mask = (pred != gt)
    error_indices = np.where(error_mask)[0]

    if len(error_indices) == 0:
        return []

    error_regions = []

    if use_clustering and len(error_indices) > min_points and HAS_SKLEARN:
        error_points = points[error_indices]

        MAX_SAMPLES_FOR_EPS = 5000

        if len(error_points) > 1:
            if len(error_points) > MAX_SAMPLES_FOR_EPS:
                sample_indices = np.random.choice(
                    len(error_points),
                    size=min(MAX_SAMPLES_FOR_EPS, len(error_points)),
                    replace=False
                )
                sample_points = error_points[sample_indices]
                distances = pdist(sample_points)
                eps = np.percentile(distances, 10)
            else:
                distances = pdist(error_points)
                eps = np.percentile(distances, 10)
        else:
            bbox_size = points.max(axis=0) - points.min(axis=0)
            eps = np.mean(bbox_size) * 0.01
            if eps < 0.01:
                eps = 0.1

        MAX_POINTS_FOR_CLUSTERING = 50000
        if len(error_points) > MAX_POINTS_FOR_CLUSTERING:
            print(f"[Warning] Too many error points ({len(error_points)}). "
                  f"Using simplified region detection.")
            error_regions.append({
                'indices': error_indices,
                'center': error_points.mean(axis=0),
                'bbox': (error_points.min(axis=0), error_points.max(axis=0)),
                'count': len(error_indices)
            })
        else:
            clustering = DBSCAN(
                eps=eps, min_samples=min_points // 2).fit(error_points)
            labels = clustering.labels_

            unique_labels = set(labels) - {-1}
            for label in unique_labels:
                cluster_mask = (labels == label)
                cluster_indices = error_indices[cluster_mask]
                if len(cluster_indices) >= min_points:
                    cluster_points = error_points[cluster_mask]
                    error_regions.append({
                        'indices': cluster_indices,
                        'center': cluster_points.mean(axis=0),
                        'bbox': (cluster_points.min(axis=0), cluster_points.max(axis=0)),
                        'count': len(cluster_indices)
                    })
    else:
        if len(error_indices) >= min_points:
            error_points = points[error_indices]
            error_regions.append({
                'indices': error_indices,
                'center': error_points.mean(axis=0),
                'bbox': (error_points.min(axis=0), error_points.max(axis=0)),
                'count': len(error_indices)
            })

    return error_regions


def find_correct_regions(pred, gt, error_indices, min_points=50):
    """

    Args:

    Returns:
    """
    correct_mask = (pred == gt)
    correct_indices = np.where(correct_mask)[0]

    correct_regions = []
    if len(correct_indices) >= min_points:
        correct_regions.append({
            'indices': correct_indices,
            'count': len(correct_indices)
        })

    return correct_regions


def project_3d_to_2d(points_3d, camera_params, center, scale):
    """

    Args:
        camera_params: o3d.camera.PinholeCameraParameters

    Returns:
    """
    points_norm = (points_3d - center) / scale * 0.6

    intrinsic = camera_params.intrinsic.intrinsic_matrix
    extrinsic = camera_params.extrinsic

    points_homo = np.hstack([points_norm, np.ones((len(points_norm), 1))])

    points_cam = (extrinsic @ points_homo.T).T[:, :3]

    points_2d_homo = (intrinsic @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

    depth = points_cam[:, 2]

    return points_2d, depth


def draw_error_annotations(img, points_3d, error_regions, camera_params, center, scale,
                           color=[255, 0, 0], linewidth=3, linestyle='--',
                           depth_map=None, depth_tolerance=0.05):
    """

    Args:

    Returns:
    """
    if len(error_regions) == 0:
        return img

    annotated_img = img.copy()
    H, W = img.shape[:2]

    for region in error_regions:
        region_points_3d = points_3d[region['indices']]

        points_2d, depth = project_3d_to_2d(
            region_points_3d, camera_params, center, scale)

        valid_mask = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        )

        if valid_mask.sum() == 0:
            continue

        valid_points_2d = points_2d[valid_mask]
        valid_depth = depth[valid_mask]

        if depth_map is not None:
            visible_mask = np.ones(len(valid_points_2d), dtype=bool)

            valid_depth_pixels = depth_map[depth_map > 0]
            if len(valid_depth_pixels) > 0:
                depth_median = np.median(valid_depth_pixels)
                depth_std = np.std(valid_depth_pixels)
            else:
                depth_median = 1.0
                depth_std = 0.1

            for i, (px, py) in enumerate(valid_points_2d):
                px_int = int(np.round(px))
                py_int = int(np.round(py))

                if 0 <= px_int < W and 0 <= py_int < H:
                    depth_at_pixel = depth_map[py_int, px_int]

                    if depth_at_pixel > 0:

                        point_depth = valid_depth[i]

                        relative_diff = abs(
                            point_depth - depth_at_pixel) / (abs(depth_at_pixel) + 1e-6)

                        abs_diff = point_depth - depth_at_pixel

                        if abs_diff > 0:
                            if relative_diff > depth_tolerance or abs_diff > depth_std * depth_tolerance:
                                visible_mask[i] = False

            min_visible_ratio = 0.3
            min_visible_count = max(
                5, int(len(valid_points_2d) * min_visible_ratio))

            if visible_mask.sum() < min_visible_count:
                continue

            valid_points_2d = valid_points_2d[visible_mask]

        if len(valid_points_2d) == 0:
            continue

        x_min, y_min = valid_points_2d.min(axis=0).astype(int)
        x_max, y_max = valid_points_2d.max(axis=0).astype(int)

        region_width = x_max - x_min
        region_height = y_max - y_min
        margin_x = max(15, int(region_width * 0.05))
        margin_y = max(15, int(region_height * 0.05))

        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(W - 1, x_max + margin_x)
        y_max = min(H - 1, y_max + margin_y)

        min_box_size = 20
        if x_max - x_min < min_box_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_box_size // 2)
            x_max = min(W - 1, center_x + min_box_size // 2)
        if y_max - y_min < min_box_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_box_size // 2)
            y_max = min(H - 1, center_y + min_box_size // 2)

        if linestyle == '--':
            dash_length = 10
            gap_length = 5

            x = x_min
            while x < x_max:
                x_end = min(x + dash_length, x_max)
                annotated_img[y_min:y_min+linewidth, x:x_end] = color
                x = x_end + gap_length

            x = x_min
            while x < x_max:
                x_end = min(x + dash_length, x_max)
                annotated_img[y_max-linewidth:y_max, x:x_end] = color
                x = x_end + gap_length

            y = y_min
            while y < y_max:
                y_end = min(y + dash_length, y_max)
                annotated_img[y:y_end, x_min:x_min+linewidth] = color
                y = y_end + gap_length

            y = y_min
            while y < y_max:
                y_end = min(y + dash_length, y_max)
                annotated_img[y:y_end, x_max-linewidth:x_max] = color
                y = y_end + gap_length
        else:
            annotated_img[y_min:y_min+linewidth, x_min:x_max] = color
            annotated_img[y_max-linewidth:y_max, x_min:x_max] = color
            annotated_img[y_min:y_max, x_min:x_min+linewidth] = color
            annotated_img[y_min:y_max, x_max-linewidth:x_max] = color

    return annotated_img


# ============================================================================
# ============================================================================

def normalize_coordinate_system(points):
    """

    Args:

    Returns:
    """
    if len(points) == 0:
        return points, np.eye(4), 2

    ranges = points.max(axis=0) - points.min(axis=0)

    height_axis = np.argmax(ranges)

    print(
        f"[Info] Detected height axis: {['X', 'Y', 'Z'][height_axis]} (range: {ranges[height_axis]:.2f})")

    if height_axis == 2:
        return points, np.eye(4), 2

    transform_matrix = np.eye(4)

    if height_axis == 0:
        transform_matrix[:3, :3] = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    elif height_axis == 1:
        transform_matrix[:3, :3] = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])

    points_homo = np.hstack([points, np.ones((len(points), 1))])
    points_normalized = (transform_matrix @ points_homo.T).T[:, :3]

    print(
        f"[Info] Coordinate system normalized: {['X', 'Y', 'Z'][height_axis]}-up -> Z-up")

    return points_normalized, transform_matrix, height_axis


# ============================================================================
# ============================================================================

def visualize_qualitative_comparison(
    points,
    colors_input,
    pred_baseline,
    pred_ours,
    gt_labels,
    colormap,
    output_path,
    dataset_name='s3dis',
    show_zoom_in=True,
    zoom_regions=None,
    camera_view='default'
):
    """

    Args:
    """
    print(
        f"[Debug] Original point cloud range: min={points.min(axis=0)}, max={points.max(axis=0)}")
    points, coord_transform, height_axis = normalize_coordinate_system(points)
    print(
        f"[Debug] Normalized point cloud range: min={points.min(axis=0)}, max={points.max(axis=0)}")

    print(f"[Debug] Before ceiling transparency: {len(points)} points")
    if gt_labels is not None:
        if dataset_name == 's3dis':
            ceiling_class_id = 1
            ceiling_count = (gt_labels == ceiling_class_id).sum()
            if ceiling_count > 0:
                print(
                    f"[Info] Making ceiling points transparent: {ceiling_count} ceiling points found, setting to background color...")
                ceiling_mask = (gt_labels == ceiling_class_id)
                background_color = np.array([1.0, 1.0, 1.0])
                colors_input[ceiling_mask] = background_color
                print(
                    f"[Info] Ceiling points set to transparent (background color)")

    if len(points) == 0:
        raise ValueError(
            "ERROR: All points were filtered out! Please check the ceiling filter logic.")
    if len(points) < 100:
        print(
            f"[Warning] Very few points remaining after filtering: {len(points)}. The visualization may be empty.")

    camera_pose = get_camera_pose_for_view(points, camera_view)

    colors_baseline = colormap[pred_baseline]
    colors_ours = colormap[pred_ours]
    colors_gt = colormap[gt_labels]

    if gt_labels is not None and dataset_name == 's3dis':
        ceiling_class_id = 1
        ceiling_mask = (gt_labels == ceiling_class_id)
        if ceiling_mask.sum() > 0:
            background_color = np.array([1.0, 1.0, 1.0])
            colors_input[ceiling_mask] = background_color
            colors_baseline[ceiling_mask] = background_color
            colors_ours[ceiling_mask] = background_color
            colors_gt[ceiling_mask] = background_color
            print(
                f"[Info] Set {ceiling_mask.sum()} ceiling points to transparent (background color)")

    error_regions_baseline = find_error_regions(
        points, pred_baseline, gt_labels)
    error_regions_ours = find_error_regions(points, pred_ours, gt_labels)

    if show_zoom_in and zoom_regions:
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, len(zoom_regions) + 1,
                               figure=fig,
                               width_ratios=[2] + [1] * len(zoom_regions),
                               height_ratios=[1, 1, 1, 1],
                               hspace=0.15, wspace=0.1)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.15, wspace=0.1)

    render_result = render_pointcloud_open3d(
        points, colors_input,
        point_size=VIS_CONFIG['point_size'],
        camera_pose=camera_pose,
        return_camera_params=True,
        return_depth=True,
        norm_factor=1.2
    )
    img_input, camera_params, center, scale, depth_map = render_result

    # ========== Row 1: Input ==========
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_title('Input (with Corruption)',
                       fontsize=14, fontweight='bold', pad=10)
    ax_input.imshow(img_input)
    ax_input.axis('off')

    # ========== Row 2: Baseline Prediction ==========
    ax_baseline = fig.add_subplot(gs[1, 0])
    ax_baseline.set_title('Baseline Prediction',
                          fontsize=14, fontweight='bold', pad=10)
    img_baseline = render_pointcloud_open3d(
        points, colors_baseline,
        point_size=VIS_CONFIG['point_size'],
        camera_pose=camera_pose,
        norm_center=center,
        norm_scale=scale,
        norm_factor=1.2
    )
    if error_regions_baseline:
        error_color = [int(c * 255) for c in VIS_CONFIG['error_color']]
        img_baseline = draw_error_annotations(
            img_baseline, points, error_regions_baseline,
            camera_params, center, scale,
            color=error_color, linewidth=3, linestyle='--',
            depth_map=depth_map,
            depth_tolerance=0.05
        )
    ax_baseline.imshow(img_baseline)
    ax_baseline.axis('off')

    # ========== Row 3: Ours Prediction ==========
    ax_ours = fig.add_subplot(gs[2, 0])
    ax_ours.set_title('Ours (STS) Prediction', fontsize=14,
                      fontweight='bold', pad=10)
    img_ours = render_pointcloud_open3d(
        points, colors_ours,
        point_size=VIS_CONFIG['point_size'],
        camera_pose=camera_pose,
        norm_center=center,
        norm_scale=scale,
        norm_factor=1.2
    )
    if error_regions_baseline:
        correct_regions = []
        for err_region in error_regions_baseline:
            err_indices = err_region['indices']
            correct_mask = (pred_ours[err_indices] == gt_labels[err_indices])
            if correct_mask.sum() > 50:
                correct_indices = err_indices[correct_mask]
                correct_points = points[correct_indices]
                correct_regions.append({
                    'indices': correct_indices,
                    'center': correct_points.mean(axis=0),
                    'bbox': (correct_points.min(axis=0), correct_points.max(axis=0)),
                    'count': len(correct_indices)
                })

        if correct_regions:
            correct_color = [int(c * 255) for c in VIS_CONFIG['correct_color']]
            img_ours = draw_error_annotations(
                img_ours, points, correct_regions,
                camera_params, center, scale,
                color=correct_color, linewidth=3, linestyle='--',
                depth_map=depth_map,
                depth_tolerance=0.05
            )
    ax_ours.imshow(img_ours)
    ax_ours.axis('off')

    # ========== Row 4: Ground Truth ==========
    ax_gt = fig.add_subplot(gs[3, 0])
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    img_gt = render_pointcloud_open3d(
        points, colors_gt,
        point_size=VIS_CONFIG['point_size'],
        camera_pose=camera_pose,
        norm_center=center,
        norm_scale=scale,
        norm_factor=1.2
    )
    ax_gt.imshow(img_gt)
    ax_gt.axis('off')

    # ========== Zoom-in Views ==========
    if show_zoom_in and zoom_regions:
        for zoom_idx, (zoom_center, zoom_size) in enumerate(zoom_regions):
            zoom_mask = (
                (points[:, 0] >= zoom_center[0] - zoom_size[0] / 2) &
                (points[:, 0] <= zoom_center[0] + zoom_size[0] / 2) &
                (points[:, 1] >= zoom_center[1] - zoom_size[1] / 2) &
                (points[:, 1] <= zoom_center[1] + zoom_size[1] / 2) &
                (points[:, 2] >= zoom_center[2] - zoom_size[2] / 2) &
                (points[:, 2] <= zoom_center[2] + zoom_size[2] / 2)
            )
            zoom_points = points[zoom_mask]
            zoom_colors_input = colors_input[zoom_mask]
            zoom_colors_baseline = colors_baseline[zoom_mask]
            zoom_colors_ours = colors_ours[zoom_mask]
            zoom_colors_gt = colors_gt[zoom_mask]

            if len(zoom_points) == 0:
                continue

            for row_idx, (zoom_colors, title) in enumerate([
                (zoom_colors_input, 'Input'),
                (zoom_colors_baseline, 'Baseline'),
                (zoom_colors_ours, 'Ours'),
                (zoom_colors_gt, 'GT')
            ]):
                ax_zoom = fig.add_subplot(gs[row_idx, zoom_idx + 1])
                if row_idx == 0:
                    ax_zoom.set_title(
                        f'Zoom-in {zoom_idx + 1}', fontsize=12, fontweight='bold', pad=5)
                zoom_center_local = zoom_points.mean(axis=0)
                zoom_scale_local = np.abs(
                    zoom_points - zoom_center_local).max() + 1e-6

                img_zoom = render_pointcloud_open3d(
                    zoom_points, zoom_colors,
                    point_size=VIS_CONFIG['point_size'] *
                    VIS_CONFIG['zoom_scale'],
                    camera_pose=camera_pose,
                    norm_center=zoom_center_local,
                    norm_scale=zoom_scale_local,
                    norm_factor=1.2
                )
                ax_zoom.imshow(img_zoom)
                ax_zoom.axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved qualitative comparison to: {output_path}")


# ============================================================================
# ============================================================================

def load_pointcloud_data(data_path, dataset_name='s3dis', filter_ceiling=True):
    """

    Args:

    Returns:
    """
    if data_path.endswith('.npy'):
        data = np.load(data_path)
        points = data[:, :3]
        colors = data[:, 3:6] / 255.0
        if data.shape[1] > 6:
            labels = data[:, 6].astype(np.int64)
        else:
            labels = None
    elif data_path.endswith('.pth'):
        data = torch.load(data_path)
        points = data[0].numpy() if torch.is_tensor(data[0]) else data[0]
        colors = data[1].numpy() if torch.is_tensor(data[1]) else data[1]
        colors = np.clip((colors + 1) / 2.0, 0, 1)
        if len(data) > 2:
            labels = data[2].numpy() if torch.is_tensor(data[2]) else data[2]
        else:
            labels = None
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if filter_ceiling and labels is not None:
        if dataset_name == 's3dis':
            ceiling_class_id = 1
            non_ceiling_mask = (labels != ceiling_class_id)
            if non_ceiling_mask.sum() < len(labels):
                print(
                    f"[Info] Filtering ceiling points: {len(labels) - non_ceiling_mask.sum()} points removed")
                points = points[non_ceiling_mask]
                colors = colors[non_ceiling_mask]
                labels = labels[non_ceiling_mask]

    return points, colors, labels


def load_predictions(pred_path):
    """

    Args:

    Returns:
    """
    if pred_path.endswith('.npy'):
        predictions = np.load(pred_path).astype(np.int64)
    else:
        raise ValueError(f"Unsupported prediction file format: {pred_path}")

    return predictions


# ============================================================================
# ============================================================================

def inference_with_checkpoint(
    data_path,
    checkpoint_path,
    config_path,
    dataset_name='s3dis',
    device='cuda',
    apply_corruption=None,
    corruption_severity=1
):
    """

    Args:

    Returns:
    """
    print(f"📂 Loading config from: {config_path}")
    cfg = EasyConfig()
    cfg.load(config_path, recursive=True)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    print(f"🏗️  Building model...")
    model = build_model_from_cfg(cfg.model).to(device)

    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    model_module = model.module if hasattr(model, 'module') else model
    load_checkpoint(model, pretrained_path=checkpoint_path)
    model.eval()

    print(f"📂 Loading point cloud from: {data_path}")
    points, colors, gt_labels = load_pointcloud_data(data_path, dataset_name)

    trans_split = 'val' if cfg.datatransforms.get(
        'test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim

    coord = points.copy()
    coord -= coord.min(0)
    feat = colors.copy()

    data = {'pos': coord}
    if feat is not None:
        data['x'] = feat

    if pipe_transform is not None:
        data = pipe_transform(data)

    if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
        data['heights'] = torch.from_numpy(
            coord[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)

    if not cfg.dataset.common.get('variable', False):
        if 'x' in data.keys():
            data['x'] = data['x'].unsqueeze(0)
        data['pos'] = data['pos'].unsqueeze(0)
    else:
        data['o'] = torch.IntTensor([len(coord)])
        data['batch'] = torch.LongTensor([0] * len(coord))

    for key in data.keys():
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device)

    data['x'] = get_features_by_keys(data, cfg.feature_keys)

    if apply_corruption and apply_corruption != 'clean':
        from openpoints.loss.custom_innovations import \
            apply_scannet_c_corruption
        data = apply_scannet_c_corruption(
            data, apply_corruption, corruption_severity, cfg.feature_keys)

    print(f"🔮 Running inference...")
    with torch.no_grad():
        output_dict = model(data)
        logits = output_dict['logits']

    if not cfg.dataset.common.get('variable', False):
        if logits.dim() == 3:  # (B, C, N)
            logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
        predictions = logits.argmax(dim=1).cpu().numpy()
    else:
        predictions = logits.argmax(dim=1).cpu().numpy()

    print(f"✅ Inference complete. Predictions shape: {predictions.shape}")

    return predictions, points, colors, gt_labels


# ============================================================================
# ============================================================================

def visualize_from_predictions_dir(
    data_path,
    predictions_dir,
    output_dir,
    dataset_name='s3dis',
    scene_name=None,
    show_zoom_in=True,
    camera_view='default'
):
    """


    Args:
    """
    import glob

    if scene_name is None:
        scene_name = os.path.splitext(os.path.basename(data_path))[0]

    baseline_patterns = [
        os.path.join(predictions_dir, f'{scene_name}_baseline.npy'),
        os.path.join(predictions_dir, f'{scene_name}_pred_baseline.npy'),
        os.path.join(predictions_dir, f'*{scene_name}*baseline*.npy'),
    ]
    ours_patterns = [
        os.path.join(predictions_dir, f'{scene_name}_ours.npy'),
        os.path.join(predictions_dir, f'{scene_name}_pred_ours.npy'),
        os.path.join(predictions_dir, f'*{scene_name}*ours*.npy'),
    ]

    pred_baseline_path = None
    pred_ours_path = None

    for pattern in baseline_patterns:
        matches = glob.glob(pattern)
        if matches:
            pred_baseline_path = matches[0]
            break

    for pattern in ours_patterns:
        matches = glob.glob(pattern)
        if matches:
            pred_ours_path = matches[0]
            break

    if pred_baseline_path is None:
        raise FileNotFoundError(
            f"Baseline prediction file not found for scene: {scene_name}")
    if pred_ours_path is None:
        raise FileNotFoundError(
            f"Ours prediction file not found for scene: {scene_name}")

    print(f"📂 Found predictions:")
    print(f"   Baseline: {pred_baseline_path}")
    print(f"   Ours: {pred_ours_path}")

    points, colors_input, gt_labels = load_pointcloud_data(
        data_path, dataset_name)
    pred_baseline = load_predictions(pred_baseline_path)
    pred_ours = load_predictions(pred_ours_path)

    assert len(points) == len(pred_baseline), \
        f"Mismatch: points ({len(points)}) vs baseline pred ({len(pred_baseline)})"
    assert len(points) == len(pred_ours), \
        f"Mismatch: points ({len(points)}) vs ours pred ({len(pred_ours)})"

    if dataset_name == 's3dis':
        colormap = S3DIS_COLORMAP
    else:
        colormap = SCANNET_COLORMAP

    zoom_regions = None
    if show_zoom_in:
        error_mask = (pred_baseline !=
                      gt_labels) if gt_labels is not None else None
        if error_mask is not None and error_mask.sum() > 0:
            error_regions = find_error_regions(
                points, pred_baseline, gt_labels)
            if error_regions:
                largest_region = max(error_regions, key=lambda x: x['count'])
                bbox_min, bbox_max = largest_region['bbox']
                zoom_center = largest_region['center']
                zoom_size = (bbox_max - bbox_min) * 1.5
                zoom_regions = [(zoom_center, zoom_size)]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'{scene_name}_qualitative_comparison.png')

    visualize_qualitative_comparison(
        points=points,
        colors_input=colors_input,
        pred_baseline=pred_baseline,
        pred_ours=pred_ours,
        gt_labels=gt_labels if gt_labels is not None else pred_baseline,
        colormap=colormap,
        output_path=output_path,
        dataset_name=dataset_name,
        show_zoom_in=show_zoom_in,
        zoom_regions=zoom_regions,
        camera_view=camera_view
    )

    return output_path


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate qualitative comparison visualization')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                             help='Path to input point cloud data (.npy or .pth)')
    input_group.add_argument('--input-list', type=str,
                             help='Path to file containing list of input data paths (one per line)')

    pred_group = parser.add_argument_group('Prediction input (choose one)')
    pred_group.add_argument('--pred-baseline', type=str,
                            help='Path to baseline predictions (.npy)')
    pred_group.add_argument('--pred-ours', type=str,
                            help='Path to ours predictions (.npy)')

    ckpt_group = parser.add_argument_group(
        'Checkpoint input (alternative to --pred-*)')
    ckpt_group.add_argument('--ckpt-baseline', type=str,
                            help='Path to baseline checkpoint (.pth)')
    ckpt_group.add_argument('--ckpt-ours', type=str,
                            help='Path to ours checkpoint (.pth)')
    ckpt_group.add_argument('--config-baseline', type=str,
                            help='Path to baseline config file (.yaml)')
    ckpt_group.add_argument('--config-ours', type=str,
                            help='Path to ours config file (.yaml)')

    parser.add_argument('--output', type=str, required=True,
                        help='Output image path or directory')
    parser.add_argument('--dataset', type=str, default='s3dis',
                        choices=['s3dis', 'scannet'],
                        help='Dataset name')
    parser.add_argument('--show-zoom', action='store_true',
                        help='Show zoom-in views')
    parser.add_argument('--zoom-regions', type=str, default=None,
                        help='Zoom regions in format: "x1,y1,z1,dx1,dy1,dz1;x2,y2,z2,dx2,dy2,dz2"')
    parser.add_argument('--camera-view', type=str, default='default',
                        choices=['default', 'top', 'side', 'front'],
                        help='Camera view angle')
    parser.add_argument('--corruption', type=str, default='clean',
                        choices=['clean', 'noise', 'jitter', 'dropout'],
                        help='Apply corruption to input (for robustness visualization)')
    parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Corruption severity level')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')

    args = parser.parse_args()

    if args.input:
        input_list = [args.input]
    else:
        with open(args.input_list, 'r') as f:
            input_list = [line.strip() for line in f if line.strip()]

    if len(input_list) == 1:
        output_path = args.output
    else:
        os.makedirs(args.output, exist_ok=True)
        output_path = None

    for input_idx, input_path in enumerate(input_list):
        print(f"\n{'='*70}")
        print(f"Processing [{input_idx+1}/{len(input_list)}]: {input_path}")
        print(f"{'='*70}")

        if args.pred_baseline and args.pred_ours:
            print(f"📂 Loading point cloud from: {input_path}")
            points_raw, colors_raw, gt_labels_raw = load_pointcloud_data(
                input_path, args.dataset, filter_ceiling=False)

            print(f"📂 Loading baseline predictions from: {args.pred_baseline}")
            pred_baseline_raw = load_predictions(args.pred_baseline)

            print(f"📂 Loading ours predictions from: {args.pred_ours}")
            pred_ours_raw = load_predictions(args.pred_ours)

            assert len(points_raw) == len(pred_baseline_raw), \
                f"Mismatch: points ({len(points_raw)}) vs baseline pred ({len(pred_baseline_raw)})"
            assert len(points_raw) == len(pred_ours_raw), \
                f"Mismatch: points ({len(points_raw)}) vs ours pred ({len(pred_ours_raw)})"

            if gt_labels_raw is not None:
                if args.dataset == 's3dis':
                    ceiling_class_id = 1
                    non_ceiling_mask = (gt_labels_raw != ceiling_class_id)
                elif args.dataset == 'scannet':
                    non_ceiling_mask = np.ones(len(points_raw), dtype=bool)
                else:
                    non_ceiling_mask = np.ones(len(points_raw), dtype=bool)

                if non_ceiling_mask.sum() < len(points_raw):
                    print(
                        f"[Info] Filtering ceiling points: {len(points_raw) - non_ceiling_mask.sum()} points removed")
                    points = points_raw[non_ceiling_mask]
                    colors_input = colors_raw[non_ceiling_mask]
                    pred_baseline = pred_baseline_raw[non_ceiling_mask]
                    pred_ours = pred_ours_raw[non_ceiling_mask]
                    gt_labels = gt_labels_raw[non_ceiling_mask]
                else:
                    points = points_raw
                    colors_input = colors_raw
                    pred_baseline = pred_baseline_raw
                    pred_ours = pred_ours_raw
                    gt_labels = gt_labels_raw
            else:
                points = points_raw
                colors_input = colors_raw
                pred_baseline = pred_baseline_raw
                pred_ours = pred_ours_raw
                gt_labels = None
        elif args.ckpt_baseline and args.ckpt_ours:
            print(f"📂 Loading point cloud from: {input_path}")
            points, colors_input, gt_labels = load_pointcloud_data(
                input_path, args.dataset, filter_ceiling=True)

            if not args.config_baseline or not args.config_ours:
                raise ValueError(
                    "--config-baseline and --config-ours are required when using --ckpt-*")

            print(f"🔮 Running baseline inference...")
            pred_baseline, _, _, _ = inference_with_checkpoint(
                input_path, args.ckpt_baseline, args.config_baseline,
                args.dataset, args.device, args.corruption, args.severity
            )

            print(f"🔮 Running ours inference...")
            pred_ours, _, _, _ = inference_with_checkpoint(
                input_path, args.ckpt_ours, args.config_ours,
                args.dataset, args.device, args.corruption, args.severity
            )
        else:
            raise ValueError(
                "Must provide either (--pred-baseline and --pred-ours) or (--ckpt-baseline and --ckpt-ours)")

        assert len(points) == len(pred_baseline), \
            f"Mismatch: points ({len(points)}) vs baseline pred ({len(pred_baseline)})"
        assert len(points) == len(pred_ours), \
            f"Mismatch: points ({len(points)}) vs ours pred ({len(pred_ours)})"
        if gt_labels is not None:
            assert len(points) == len(gt_labels), \
                f"Mismatch: points ({len(points)}) vs gt ({len(gt_labels)})"

        if args.dataset == 's3dis':
            colormap = S3DIS_COLORMAP
        else:
            colormap = SCANNET_COLORMAP

        zoom_regions = None
        if args.show_zoom and args.zoom_regions:
            zoom_regions = []
            for region_str in args.zoom_regions.split(';'):
                coords = [float(x) for x in region_str.split(',')]
                if len(coords) == 6:
                    center = np.array(coords[:3])
                    size = np.array(coords[3:])
                    zoom_regions.append((center, size))

        if args.show_zoom and zoom_regions is None:
            error_mask = (pred_baseline !=
                          gt_labels) if gt_labels is not None else None
            if error_mask is not None and error_mask.sum() > 0:
                error_points = points[error_mask]
                if len(error_points) > 0:
                    error_center = error_points.mean(axis=0)
                    error_size = (error_points.max(axis=0) -
                                  error_points.min(axis=0)) * 1.5
                    zoom_regions = [(error_center, error_size)]

        if len(input_list) == 1:
            current_output = args.output
        else:
            scene_name = os.path.splitext(os.path.basename(input_path))[0]
            current_output = os.path.join(
                args.output, f'{scene_name}_qualitative_comparison.png')

        print(f"🎨 Generating qualitative comparison visualization...")
        visualize_qualitative_comparison(
            points=points,
            colors_input=colors_input,
            pred_baseline=pred_baseline,
            pred_ours=pred_ours,
            gt_labels=gt_labels if gt_labels is not None else pred_baseline,
            colormap=colormap,
            output_path=current_output,
            dataset_name=args.dataset,
            show_zoom_in=args.show_zoom,
            zoom_regions=zoom_regions,
            camera_view=args.camera_view
        )

        print(f"✅ Done! Visualization saved to: {current_output}")

    if len(input_list) > 1:
        print(f"\n✅ All visualizations saved to: {args.output}")


if __name__ == '__main__':
    main()
