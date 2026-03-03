#!/usr/bin/env python3
# isort: skip_file
"""
📊 Teaser Figure Generator with Real Model Inference (V2)
"""

from openpoints.loss.custom_innovations import SemanticConflictGenerator
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
import warnings
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from scipy.linalg import svd
import open3d as o3d
import numpy as np
import torch
import sys
import os
import argparse


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================================

warnings.filterwarnings('ignore')


SCANNET_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
]

TARGET_CLASS_OPTIONS = {
    'sofa': 5,      # ScanNet class id for sofa
    'table': 6,     # ScanNet class id for table
    'toilet': 16,   # ScanNet class id for toilet
}

COLORS = {
    'bg': '#ffffff',
    'panel_bg': '#ffffff',
    'text': '#000000',
    'text_secondary': '#444444',
    'accent': '#0366d6',
    'success': '#28a745',
    'error': '#cb2431',
    'warning': '#f9a825',
    'border': '#e1e4e8',

    'chair': np.array([0.35, 0.15, 0.55]),
    'sofa': np.array([0.35, 0.15, 0.55]),

    'table': np.array([0.85, 0.45, 0.10]),

    'toilet': np.array([0.15, 0.55, 0.65]),

    'wall': np.array([0.60, 0.60, 0.60]),
    'floor': np.array([0.50, 0.50, 0.50]),

    'correct': np.array([0.0, 0.60, 0.20]),

    'wrong': np.array([0.85, 0.15, 0.15]),

    'geometry': np.array([0.10, 0.40, 0.80]),
    'texture': np.array([0.90, 0.60, 0.10]),
    'causal_arrow': np.array([0.50, 0.50, 0.55]),
    'cut': '#cb2431',
}


def load_model(cfg, ckpt_path, device):
    """"""
    model = build_model_from_cfg(cfg.model).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  Loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("    ✅ Loaded")

    model.eval()
    return model


def find_best_sample(dataloader, generator, baseline, ours, device, cfg, max_search=50, target_name='sofa'):
    """

    """
    print(
        f"\n[Search] Finding good samples for '{target_name}' to demonstrate texture bias...")

    dataset = dataloader.dataset
    room_names = getattr(dataset, 'data_list', None)
    if room_names:
        print(f"  Dataset has {len(room_names)} rooms in validation set")

    good_samples = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= max_search:
                break

            room_name = room_names[batch_idx] if room_names and batch_idx < len(
                room_names) else f"sample_{batch_idx}"

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            data_attacked, attack_info = generator.generate_conflict_sample(
                data, return_attack_info=True
            )

            if not attack_info['success']:
                continue

            target_class = attack_info['target_class'][1]
            if target_class != target_name:
                continue

            data_attacked['x'] = get_features_by_keys(
                data_attacked, cfg.feature_keys)

            out_base = baseline(data_attacked)
            pred_base = out_base['logits'].argmax(
                dim=1).reshape(-1).cpu().numpy()

            out_ours = ours(data_attacked)
            pred_ours = out_ours['logits'].argmax(
                dim=1).reshape(-1).cpu().numpy()

            attack_mask = attack_info['attack_mask']
            target_id = attack_info['target_class'][0]

            base_tr = (pred_base[attack_mask] ==
                       target_id).sum() / attack_mask.sum() * 100
            ours_tr = (pred_ours[attack_mask] ==
                       target_id).sum() / attack_mask.sum() * 100

            diff = ours_tr - base_tr

            n_target_points = attack_mask.sum()
            print(f"  Sample {batch_idx} [{room_name}]: {target_class} <- {attack_info['source_class'][1]}, "
                  f"pts={n_target_points}, Baseline TR={base_tr:.1f}%, Ours TR={ours_tr:.1f}%, Δ={diff:+.1f}%")

            is_good_demo = (ours_tr >= 70 and diff >
                            70 and n_target_points >= 5000)

            if is_good_demo:
                sample = {
                    'data': {k: v.clone() if torch.is_tensor(v) else v
                             for k, v in data_attacked.items()},
                    'attack_info': attack_info,
                    'pred_base': pred_base,
                    'pred_ours': pred_ours,
                    'base_tr': base_tr,
                    'ours_tr': ours_tr,
                    'diff': diff,
                    'n_points': n_target_points,
                    'batch_idx': batch_idx,
                    'room_name': room_name,
                    'target_name': target_name
                }
                good_samples.append(sample)
                print(
                    f"    → Added to candidate list (total: {len(good_samples)})")

    if good_samples:
        import random
        random.seed()
        selected = random.choice(good_samples)
        print(
            f"\n  ✅ Randomly selected sample from {len(good_samples)} candidates!")
        print(f"     Sample index = {selected['batch_idx']}")
        print(f"     Room name    = {selected['room_name']}")
        print(f"     Points = {selected['n_points']} ()")
        print(f"     Baseline TR = {selected['base_tr']:.1f}% ()")
        print(f"     Ours TR     = {selected['ours_tr']:.1f}% ()")
        print(f"      Δ      = +{selected['diff']:.1f}%")
        return selected
    else:
        print("\n  ⚠️ No good sample found with enough points for clear visualization.")
        print("     Try increasing --max-search or relaxing criteria.")
        return None


def compute_pca_view_angle(points):
    """ PCA """
    centered = points - points.mean(axis=0)
    _, _, Vt = svd(centered, full_matrices=False)

    main_axis = Vt[0]

    azim = np.degrees(np.arctan2(main_axis[1], main_axis[0]))

    azim_view = azim + 90

    return azim_view


def render_pointcloud_open3d(fg_points, fg_colors, bg_points=None, bg_colors=None,
                             output_path='temp_render.png', width=800, height=600,
                             point_size_fg=0.05, point_size_bg=0.03,
                             add_shadow=True, auto_view=True, fixed_azim=None):
    """

    Args:

    Returns:
    """
    if bg_points is not None and len(bg_points) > 0:
        all_points = np.vstack([bg_points, fg_points])
    else:
        all_points = fg_points

    center = all_points.mean(axis=0)
    scale = np.abs(all_points - center).max() + 1e-6

    if bg_points is not None and len(bg_points) > 0:
        norm_scale = 0.65
    else:
        norm_scale = 0.35
    fg_points_norm = (fg_points - center) / scale * norm_scale

    pcd_fg = o3d.geometry.PointCloud()
    pcd_fg.points = o3d.utility.Vector3dVector(fg_points_norm)
    pcd_fg.colors = o3d.utility.Vector3dVector(np.clip(fg_colors, 0, 1))

    pcd_fg.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_fg.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.]))

    geometries = []

    if bg_points is not None and len(bg_points) > 0:
        bg_points_norm = (bg_points - center) / scale * norm_scale
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(bg_points_norm)
        if bg_colors is not None:
            if bg_colors.ndim == 1:
                bg_colors_arr = np.ones((len(bg_points), 3)) * bg_colors
            else:
                bg_colors_arr = bg_colors
        else:
            bg_colors_arr = np.ones((len(bg_points), 3)) * 0.7
        pcd_bg.colors = o3d.utility.Vector3dVector(
            np.clip(bg_colors_arr, 0, 1))
        pcd_bg.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_bg.orient_normals_towards_camera_location(
            camera_location=np.array([0., 0., 0.]))
        geometries.append(pcd_bg)

    if add_shadow:
        z_min = fg_points_norm[:, 2].min()
        shadow_z = z_min - 0.03

        n_orig = len(fg_points_norm)
        n_extra = n_orig * 2

        shadow_base = fg_points_norm[:, :2].copy()

        np.random.seed(123)
        extra_offsets = np.random.normal(0, 0.015, (n_extra, 2))
        extra_points_xy = np.tile(shadow_base, (2, 1)) + extra_offsets

        all_shadow_xy = np.vstack([shadow_base, extra_points_xy])
        all_shadow_z = np.full((len(all_shadow_xy), 1), shadow_z)
        shadow_points = np.hstack([all_shadow_xy, all_shadow_z])

        core_colors = np.ones((n_orig, 3)) * 0.88
        edge_colors = np.ones((n_extra, 3)) * 0.92
        shadow_colors = np.vstack([core_colors, edge_colors])

        pcd_shadow = o3d.geometry.PointCloud()
        pcd_shadow.points = o3d.utility.Vector3dVector(shadow_points)
        pcd_shadow.colors = o3d.utility.Vector3dVector(shadow_colors)
        geometries.append(pcd_shadow)

    geometries.append(pcd_fg)

    render = o3d.visualization.rendering.OffscreenRenderer(width, height)

    render.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat_fg = o3d.visualization.rendering.MaterialRecord()
    mat_fg.shader = "defaultLit"
    mat_fg.point_size = point_size_fg * 150
    mat_fg.base_roughness = 0.8
    mat_fg.base_metallic = 0.0
    mat_fg.base_reflectance = 0.1

    mat_bg = o3d.visualization.rendering.MaterialRecord()
    mat_bg.shader = "defaultLit"
    mat_bg.point_size = point_size_bg * 80
    mat_bg.base_roughness = 0.9
    mat_bg.base_metallic = 0.0
    mat_bg.base_reflectance = 0.1

    mat_shadow = o3d.visualization.rendering.MaterialRecord()
    mat_shadow.shader = "defaultUnlit"
    mat_shadow.point_size = point_size_fg * 60

    render.scene.scene.set_sun_light(
        direction=[-0.5, -0.5, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=100000
    )
    render.scene.scene.enable_sun_light(True)

    if add_shadow:
        render.scene.add_geometry("shadow", pcd_shadow, mat_shadow)
    if bg_points is not None and len(bg_points) > 0:
        render.scene.add_geometry("bg_points", pcd_bg, mat_bg)
    render.scene.add_geometry("fg_points", pcd_fg, mat_fg)

    if fixed_azim is not None:
        azim = fixed_azim
    elif auto_view:
        azim = compute_pca_view_angle(fg_points_norm)
    else:
        azim = -60

    elev = 25

    dist = 1.5
    azim_rad = np.radians(azim)
    elev_rad = np.radians(elev)

    eye = np.array([
        dist * np.cos(elev_rad) * np.cos(azim_rad),
        dist * np.cos(elev_rad) * np.sin(azim_rad),
        dist * np.sin(elev_rad)
    ])
    center_cam = np.array([0, 0, 0])
    up = np.array([0, 0, 1])

    render.setup_camera(45.0, center_cam, eye, up)

    render.scene.scene.enable_indirect_light(True)
    render.scene.scene.set_indirect_light_intensity(20000)

    img = render.render_to_image()

    o3d.io.write_image(output_path, img)

    img_np = np.asarray(img)

    return img_np


def render_pointcloud_open3d_with_angle(fg_points, fg_colors, bg_points=None, bg_colors=None,
                                        output_path='temp_render.png', width=800, height=600,
                                        eye=None, add_shadow=True):
    """

    Args:
    """
    if bg_points is not None and len(bg_points) > 0:
        all_points = np.vstack([bg_points, fg_points])
    else:
        all_points = fg_points

    center = all_points.mean(axis=0)
    scale = np.abs(all_points - center).max() + 1e-6

    if bg_points is not None and len(bg_points) > 0:
        norm_scale = 0.65
    else:
        norm_scale = 0.35
    fg_points_norm = (fg_points - center) / scale * norm_scale

    pcd_fg = o3d.geometry.PointCloud()
    pcd_fg.points = o3d.utility.Vector3dVector(fg_points_norm)
    pcd_fg.colors = o3d.utility.Vector3dVector(np.clip(fg_colors, 0, 1))

    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat_fg = o3d.visualization.rendering.MaterialRecord()
    mat_fg.shader = "defaultLit"
    mat_fg.point_size = 12.0
    mat_fg.base_roughness = 0.8
    mat_fg.base_metallic = 0.0
    mat_fg.base_reflectance = 0.1

    mat_bg = o3d.visualization.rendering.MaterialRecord()
    mat_bg.shader = "defaultLit"
    mat_bg.point_size = 10.0
    mat_bg.base_roughness = 0.9
    mat_bg.base_metallic = 0.0
    mat_bg.base_reflectance = 0.1

    render.scene.scene.set_sun_light(
        [-0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 100000)
    render.scene.scene.enable_sun_light(True)

    if add_shadow:
        z_min = fg_points_norm[:, 2].min()
        shadow_z = z_min - 0.03

        n_orig = len(fg_points_norm)
        n_extra = n_orig * 2

        shadow_base = fg_points_norm[:, :2].copy()
        np.random.seed(123)
        extra_offsets = np.random.normal(0, 0.015, (n_extra, 2))
        extra_points_xy = np.tile(shadow_base, (2, 1)) + extra_offsets

        all_shadow_xy = np.vstack([shadow_base, extra_points_xy])
        all_shadow_z = np.full((len(all_shadow_xy), 1), shadow_z)
        shadow_points = np.hstack([all_shadow_xy, all_shadow_z])

        core_colors = np.ones((n_orig, 3)) * 0.88
        edge_colors = np.ones((n_extra, 3)) * 0.92
        shadow_colors = np.vstack([core_colors, edge_colors])

        pcd_shadow = o3d.geometry.PointCloud()
        pcd_shadow.points = o3d.utility.Vector3dVector(shadow_points)
        pcd_shadow.colors = o3d.utility.Vector3dVector(shadow_colors)

        mat_shadow = o3d.visualization.rendering.MaterialRecord()
        mat_shadow.shader = "defaultUnlit"
        mat_shadow.point_size = 15.0
        render.scene.add_geometry("shadow", pcd_shadow, mat_shadow)

    if bg_points is not None and len(bg_points) > 0:
        bg_points_norm = (bg_points - center) / scale * norm_scale
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(bg_points_norm)
        if bg_colors is not None:
            if bg_colors.ndim == 1:
                bg_colors_arr = np.ones((len(bg_points), 3)) * bg_colors
            else:
                bg_colors_arr = bg_colors
        else:
            bg_colors_arr = np.ones((len(bg_points), 3)) * 0.7
        pcd_bg.colors = o3d.utility.Vector3dVector(
            np.clip(bg_colors_arr, 0, 1))
        render.scene.add_geometry("bg_points", pcd_bg, mat_bg)

    render.scene.add_geometry("fg_points", pcd_fg, mat_fg)

    if eye is None:
        eye = np.array([1.5, -1.5, 1.0])
    center_cam = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    render.setup_camera(45.0, center_cam, eye, up)

    img = render.render_to_image()
    o3d.io.write_image(output_path, img)

    return np.asarray(img)


def plot_pointcloud_3d(ax, points, colors, title='', sample_idx=None, point_size=4):
    """3D (matplotlib )"""
    center = points.mean(axis=0)
    points = points - center
    scale = np.abs(points).max() + 1e-6
    points = points / scale * 0.8

    if sample_idx is not None:
        points = points[sample_idx]
        colors = colors[sample_idx]
    elif len(points) > 3000:
        idx = np.random.choice(len(points), 3000, replace=False)
        points = points[idx]
        colors = colors[idx]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colors, s=point_size, alpha=0.9, edgecolors='none')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title, fontsize=11, color=COLORS['text'], pad=10)
    ax.set_axis_off()
    ax.set_facecolor(COLORS['panel_bg'])

    ax.view_init(elev=20, azim=-60)


def plot_pointcloud_with_context(ax, fg_points, fg_colors, bg_points, bg_color, title=''):
    """3D (matplotlib )"""
    all_points = np.vstack([bg_points, fg_points])
    center = all_points.mean(axis=0)
    scale = np.abs(all_points - center).max() + 1e-6

    bg_points_norm = (bg_points - center) / scale * 0.8
    fg_points_norm = (fg_points - center) / scale * 0.8

    bg_colors = np.ones((len(bg_points), 3)) * bg_color
    ax.scatter(bg_points_norm[:, 0], bg_points_norm[:, 1], bg_points_norm[:, 2],
               c=bg_colors, s=4, alpha=0.25, edgecolors='none', zorder=1)

    ax.scatter(fg_points_norm[:, 0], fg_points_norm[:, 1], fg_points_norm[:, 2],
               c=fg_colors, s=12, alpha=0.9, edgecolors='none', zorder=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(title, fontsize=11, color=COLORS['text'], pad=10)
    ax.set_axis_off()
    ax.set_facecolor(COLORS['panel_bg'])

    ax.view_init(elev=20, azim=-60)


def draw_causal_graph(ax, title='', cut_spurious=False):
    """"""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['panel_bg'])

    nodes = {
        'G': (1.5, 1.5),
        'T': (3.5, 1.5),
        'Y': (2.5, 0.5),
        'C': (2.5, 2.5)
    }

    for name, (x, y) in nodes.items():
        circle = Circle((x, y), 0.35, facecolor=COLORS['accent'],
                        edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

    arrow_style = dict(arrowstyle='->', mutation_scale=15, linewidth=2)

    ax.annotate('', xy=(2.2, 0.7), xytext=(1.8, 1.2),
                arrowprops={**arrow_style, 'color': COLORS['success']})

    if cut_spurious:
        ax.annotate('', xy=(2.8, 0.7), xytext=(3.2, 1.2),
                    arrowprops={**arrow_style, 'color': COLORS['error'],
                                'linestyle': '--', 'alpha': 0.3})
        ax.plot([2.9, 3.1], [0.8, 1.1], color=COLORS['error'], linewidth=3)
        ax.plot([2.9, 3.1], [1.1, 0.8], color=COLORS['error'], linewidth=3)
    else:
        ax.annotate('', xy=(2.8, 0.7), xytext=(3.2, 1.2),
                    arrowprops={**arrow_style, 'color': COLORS['warning']})

    ax.annotate('', xy=(1.7, 1.8), xytext=(2.2, 2.2),
                arrowprops={**arrow_style, 'color': COLORS['text_secondary']})
    ax.annotate('', xy=(3.3, 1.8), xytext=(2.8, 2.2),
                arrowprops={**arrow_style, 'color': COLORS['text_secondary']})

    ax.set_title(title, fontsize=11, color=COLORS['text'], pad=5)


def create_teaser_figure(sample, output_path, dpi=300):
    """ Teaser Figure"""
    print("\n[Visualize] Creating teaser figure...")

    attack_info = sample['attack_info']
    attack_mask = attack_info['attack_mask']
    target_id = attack_info['target_class'][0]
    target_name = sample.get(
        'target_name', attack_info['target_class'][1])
    source_id = attack_info['source_class'][0]
    source_name = attack_info['source_class'][1]
    room_name = sample.get('room_name', 'unknown')

    print(f"  📍 Sample source: {room_name}")
    print(f"     (ScanNet validation set)")

    data = sample['data']
    pos = data['pos'].cpu().numpy()
    if pos.ndim == 3:
        pos = pos[0]

    N = len(pos)
    print(
        f"  Points: {N}, Attack mask sum: {attack_mask.sum()}, Attack mask len: {len(attack_mask)}")

    x = data['x'].cpu().numpy()
    if x.ndim == 3:
        x = x[0]

    if x.shape[0] < x.shape[1]:
        colors = x[:3, :].T  # -> (N, 3)
    else:
        colors = x[:, :3]

    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0, 1)

    pred_base = sample['pred_base']
    pred_ours = sample['pred_ours']

    print(f"  Colors shape: {colors.shape}, Preds shape: {pred_base.shape}")

    np.random.seed(42)

    labels = data['y'].cpu().numpy()
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    all_target_indices = np.where(labels == target_id)[0]
    print(f"  Total {target_name} points in scene: {len(all_target_indices)}")

    from sklearn.cluster import DBSCAN

    target_points_all = pos[all_target_indices]

    clustering = DBSCAN(eps=0.3, min_samples=50).fit(target_points_all)
    cluster_labels = clustering.labels_

    unique_labels, counts = np.unique(
        cluster_labels[cluster_labels >= 0], return_counts=True)
    if len(unique_labels) > 0:
        largest_cluster_id = unique_labels[np.argmax(counts)]
        largest_cluster_mask = cluster_labels == largest_cluster_id

        target_indices = all_target_indices[largest_cluster_mask]
        print(
            f"  Found {len(unique_labels)} separate {target_name} regions, keeping largest with {len(target_indices)} points")
    else:
        target_indices = all_target_indices
        print(
            f"  No valid clusters found, using all {len(target_indices)} points")

    bg_indices = np.where(labels != target_id)[0]
    n_target_total = len(target_indices)
    n_bg_total = len(bg_indices)

    print(
        f"  Using largest connected {target_name} region: {n_target_total} points")
    print(f"  (attack_mask had {attack_mask.sum()} points)")

    largest_target_mask = np.zeros(N, dtype=bool)
    largest_target_mask[target_indices] = True

    max_target_vis = min(n_target_total, 8000)
    if n_target_total > max_target_vis:
        target_sample_idx = np.random.choice(
            target_indices, max_target_vis, replace=False)
    else:
        target_sample_idx = target_indices

    target_points = pos[target_indices]
    target_min = target_points.min(axis=0)
    target_max = target_points.max(axis=0)

    expand_dist = 0.5
    bbox_min = target_min - expand_dist
    bbox_max = target_max + expand_dist

    bg_points_all = pos[bg_indices]
    in_bbox = np.all((bg_points_all >= bbox_min) &
                     (bg_points_all <= bbox_max), axis=1)
    bg_indices_in_bbox = bg_indices[in_bbox]

    print(
        f"  Bounding box cropping: {len(bg_indices)} -> {len(bg_indices_in_bbox)} background points")

    max_bg_vis = min(len(bg_indices_in_bbox), 5000)
    if len(bg_indices_in_bbox) > max_bg_vis:
        bg_sample_idx = np.random.choice(
            bg_indices_in_bbox, max_bg_vis, replace=False)
    else:
        bg_sample_idx = bg_indices_in_bbox

    print(
        f"  {target_name.capitalize()} points: {n_target_total} total, {len(target_sample_idx)} sampled")
    print(
        f"  Background points (in bbox): {len(bg_indices_in_bbox)} total, {len(bg_sample_idx)} sampled")

    combined_idx = np.concatenate([bg_sample_idx, target_sample_idx])
    n_bg_sampled = len(bg_sample_idx)
    n_target_sampled = len(target_sample_idx)

    pos_vis = pos[combined_idx]
    colors_vis = colors[combined_idx]
    pred_base_vis = pred_base[combined_idx]
    pred_ours_vis = pred_ours[combined_idx]

    target_mask_vis = largest_target_mask[combined_idx]

    print(
        f"  Total vis points: {len(pos_vis)} (bg: {n_bg_sampled}, {target_name}: {n_target_sampled})")

    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['bg'])
    gs = gridspec.GridSpec(3, 12, height_ratios=[0.08, 0.62, 0.30],
                           wspace=0.08, hspace=0.15)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5,
                  f'Texture Bias in 3D Semantic Segmentation: {target_name.capitalize()} ← {source_name.capitalize()} Texture',
                  ha='center', va='center', fontsize=16, fontweight='bold',
                  color=COLORS['text'])

    TEXTURE_COLORS = {
        'wall': np.array([0.78, 0.80, 0.82]),
        'floor': np.array([0.65, 0.55, 0.45]),
        'cabinet': np.array([0.55, 0.48, 0.40]),
        'bed': np.array([0.72, 0.62, 0.72]),
        'curtain': np.array([0.82, 0.75, 0.68]),
    }
    texture_color = TEXTURE_COLORS.get(
        source_name, np.array([0.70, 0.65, 0.55]))

    fg_mask = target_mask_vis.copy()
    bg_mask = ~fg_mask

    fg_idx = np.where(fg_mask)[0]
    bg_idx = np.where(bg_mask)[0]

    pos_fg = pos_vis[fg_idx]
    pos_bg = pos_vis[bg_idx]
    pred_base_fg = pred_base_vis[fg_idx]
    pred_ours_fg = pred_ours_vis[fg_idx]
    n_fg = len(fg_idx)
    n_bg = len(bg_idx)

    SEMANTIC_COLORS = {
        0: np.array([0.78, 0.80, 0.82]),
        1: np.array([0.65, 0.55, 0.45]),
        2: np.array([0.55, 0.48, 0.40]),
        3: np.array([0.72, 0.62, 0.72]),
        4: np.array([0.72, 0.52, 0.52]),
        5: np.array([0.35, 0.15, 0.55]),
        6: np.array([0.68, 0.58, 0.48]),
        7: np.array([0.58, 0.45, 0.35]),
        8: np.array([0.72, 0.82, 0.92]),
        9: np.array([0.52, 0.45, 0.38]),
        10: np.array([0.85, 0.85, 0.78]),
        11: np.array([0.62, 0.58, 0.55]),
        12: np.array([0.65, 0.55, 0.45]),
        13: np.array([0.82, 0.75, 0.68]),
        14: np.array([0.75, 0.78, 0.80]),
        15: np.array([0.88, 0.88, 0.85]),
        16: np.array([0.85, 0.85, 0.88]),
        17: np.array([0.80, 0.82, 0.85]),
        18: np.array([0.88, 0.88, 0.90]),
        19: np.array([0.62, 0.62, 0.62]),
    }

    labels_vis = labels[combined_idx]
    labels_bg = labels_vis[bg_idx]
    bg_colors_semantic = np.zeros((n_bg, 3))
    for i, label in enumerate(labels_bg):
        bg_colors_semantic[i] = SEMANTIC_COLORS.get(
            int(label), np.array([0.62, 0.62, 0.62]))

    print(
        f"  Visualization: {n_fg} {target_name} points + {n_bg} background points (with room context)")

    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_renders')
    os.makedirs(temp_dir, exist_ok=True)

    print("  Rendering with Open3D (SSAO enabled)...")

    # figsize=(18, 10) @ 300 dpi -> 5400 x 3000 pixels
    render_width = 1400
    render_height = 1200

    if 'scene0672_00' in room_name:
        fixed_azim = 135.0
        print(f"  Using fixed azimuth for scene0672_00: {fixed_azim:.1f}°")
    elif 'scene0203_00' in room_name:
        fixed_azim = 225.0
        print(f"  Using fixed azimuth for scene0203_00: {fixed_azim:.1f}°")
    elif 'scene0590_01' in room_name:
        fixed_azim = 135.0
        print(f"  Using fixed azimuth for scene0590_01: {fixed_azim:.1f}°")
    elif 'scene0642_00' in room_name:
        fixed_azim = 90.0
        print(f"  Using fixed azimuth for scene0642_00: {fixed_azim:.1f}°")
    else:
        fixed_azim = compute_pca_view_angle(pos_fg)
        print(f"  Fixed camera azimuth: {fixed_azim:.1f}°")

    target_color = COLORS.get(target_name, np.array([0.35, 0.15, 0.55]))

    input_colors_fg = np.zeros((n_fg, 3))
    for i in range(n_fg):
        if i % 2 == 0:
            input_colors_fg[i] = target_color
        else:
            input_colors_fg[i] = texture_color

    img_a = render_pointcloud_open3d(
        pos_fg, input_colors_fg, pos_bg, bg_colors_semantic,
        os.path.join(temp_dir, 'panel_a.png'),
        width=render_width, height=render_height,
        auto_view=False, fixed_azim=fixed_azim)

    base_colors_fg = np.zeros((n_fg, 3))
    base_correct = pred_base_fg == target_id
    base_wrong = pred_base_fg != target_id
    base_colors_fg[base_correct] = COLORS['correct']
    base_colors_fg[base_wrong] = COLORS['wrong']

    img_b = render_pointcloud_open3d(
        pos_fg, base_colors_fg, pos_bg, bg_colors_semantic,
        os.path.join(temp_dir, 'panel_b.png'),
        width=render_width, height=render_height,
        auto_view=False, fixed_azim=fixed_azim)

    ours_colors_fg = np.zeros((n_fg, 3))
    ours_correct = pred_ours_fg == target_id
    ours_wrong = pred_ours_fg != target_id
    ours_colors_fg[ours_correct] = COLORS['correct']
    ours_colors_fg[ours_wrong] = COLORS['wrong']

    img_c = render_pointcloud_open3d(
        pos_fg, ours_colors_fg, pos_bg, bg_colors_semantic,
        os.path.join(temp_dir, 'panel_c.png'),
        width=render_width, height=render_height,
        auto_view=False, fixed_azim=fixed_azim)

    gt_colors_fg = np.ones((n_fg, 3)) * target_color

    img_d = render_pointcloud_open3d(
        pos_fg, gt_colors_fg, pos_bg, bg_colors_semantic,
        os.path.join(temp_dir, 'panel_d.png'),
        width=render_width, height=render_height,
        auto_view=False, fixed_azim=fixed_azim)

    print("  Open3D rendering complete!")

    ax1 = fig.add_subplot(gs[1, 0:3])
    ax1.imshow(img_a)
    ax1.axis('off')
    ax1.set_title(f'(a) Input: {target_name.capitalize()} + {source_name.capitalize()} Texture',
                  fontsize=11, color=COLORS['text'], pad=10)
    ax1.text(0.5, -0.02, f'Colored = {target_name.capitalize()} with conflicting texture\nBackground = Room context (semantic colors)',
             transform=ax1.transAxes, ha='center', fontsize=9,
             color=COLORS['text_secondary'])

    ax2 = fig.add_subplot(gs[1, 3:6])
    ax2.imshow(img_b)
    ax2.axis('off')
    ax2.set_title('(b) Baseline (PointNeXt-XL)',
                  fontsize=11, color=COLORS['text'], pad=10)
    ax2.text(0.5, -0.02,
             f'× TR = {sample["base_tr"]:.0f}%\n(Deceived by {source_name} texture)',
             transform=ax2.transAxes, ha='center', fontsize=9,
             color=COLORS['error'], fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 6:9])
    ax3.imshow(img_c)
    ax3.axis('off')
    ax3.set_title('(c) Ours (STS)', fontsize=11, color=COLORS['text'], pad=10)
    ax3.text(0.5, -0.02,
             f'√ TR = {sample["ours_tr"]:.0f}%\n(Robust to texture deception)',
             transform=ax3.transAxes, ha='center', fontsize=9,
             color=COLORS['success'], fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 9:12])
    ax4.imshow(img_d)
    ax4.axis('off')
    ax4.set_title('(d) Ground Truth', fontsize=11,
                  color=COLORS['text'], pad=10)
    ax4.text(0.5, -0.02,
             f'Colored = {target_name.capitalize()} (target)\nGray = Background context',
             transform=ax4.transAxes, ha='center', fontsize=9,
             color=COLORS['text_secondary'])

    ax5_left = fig.add_subplot(gs[2, 1:5])
    ax5_left.set_facecolor(COLORS['panel_bg'])
    ax5_left.set_xlim(0, 5)
    ax5_left.set_ylim(0, 3)
    ax5_left.axis('off')

    draw_causal_subgraph_horizontal(ax5_left, cut=False)
    ax5_left.set_title('(e1) Baseline: Spurious Correlation', fontsize=11,
                       color=COLORS['error'], pad=5)
    ax5_left.text(0.5, -0.18, 'Model relies on texture shortcut\n$Y \\leftarrow C \\rightarrow T$ path active',
                  ha='center', va='top', fontsize=9, color=COLORS['text_secondary'],
                  transform=ax5_left.transAxes)

    ax5_right = fig.add_subplot(gs[2, 7:11])
    ax5_right.set_facecolor(COLORS['panel_bg'])
    ax5_right.set_xlim(0, 5)
    ax5_right.set_ylim(0, 3)
    ax5_right.axis('off')

    draw_causal_subgraph_horizontal(ax5_right, cut=True)
    ax5_right.set_title('(e2) Ours (STS): Causal Intervention', fontsize=11,
                        color=COLORS['success'], pad=5)
    ax5_right.text(0.5, -0.18, 'STS breaks spurious path\n$do(T)$ forces geometry reliance',
                   ha='center', va='top', fontsize=9, color=COLORS['text_secondary'],
                   transform=ax5_right.transAxes)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print(f"  ✅ Saved: {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print(f"  ✅ Saved: {pdf_path}")

    info_path = output_path.replace('.png', '_info.txt')
    with open(info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Teaser Figure Sample Information\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scene Name: {room_name}\n")
        f.write(f"Dataset: ScanNet (validation set)\n")
        f.write(
            f"Target Object: {target_name.capitalize()} (class_id={target_id})\n")
        f.write(f"Texture Source: {source_name} (class_id={source_id})\n")
        f.write(f"{target_name.capitalize()} Points: {n_fg}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"  Baseline TR: {sample['base_tr']:.1f}%\n")
        f.write(f"  Ours TR: {sample['ours_tr']:.1f}%\n")
        f.write(f"  Improvement: +{sample['diff']:.1f}%\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"  ✅ Saved: {info_path}")

    plt.close()

    print("  Saving multi-view renders...")
    views_dir = os.path.join(os.path.dirname(output_path), 'multi_views')
    os.makedirs(views_dir, exist_ok=True)

    view_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for angle in view_angles:
        dist = 1.5
        elev = 25
        azim_rad = np.radians(angle)
        elev_rad = np.radians(elev)

        eye = np.array([
            dist * np.cos(elev_rad) * np.cos(azim_rad),
            dist * np.cos(elev_rad) * np.sin(azim_rad),
            dist * np.sin(elev_rad)
        ])

        render_pointcloud_open3d_with_angle(
            pos_fg, gt_colors_fg, pos_bg, bg_colors_semantic,
            os.path.join(views_dir, f'{target_name}_view_{angle:03d}.png'),
            eye=eye
        )

    print(f"  ✅ Multi-view renders saved to: {views_dir}")


def draw_causal_subgraph(ax, y_offset, cut=False):
    """ ()"""
    nodes = {
        'G': (1.0, 1.2 + y_offset),
        'T': (4.0, 1.2 + y_offset),
        'Y': (2.5, 0.3 + y_offset),
        'C': (2.5, 2.1 + y_offset)
    }

    for name, (x, y) in nodes.items():
        color = COLORS['accent'] if name in ['G', 'Y'] else COLORS['warning']
        circle = Circle((x, y), 0.25, facecolor=color,
                        edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    arrow_style = dict(arrowstyle='->', mutation_scale=10, linewidth=1.5)

    ax.annotate('', xy=(2.0, 0.4 + y_offset), xytext=(1.3, 1.0 + y_offset),
                arrowprops={**arrow_style, 'color': COLORS['success']})

    if cut:
        ax.annotate('', xy=(3.0, 0.4 + y_offset), xytext=(3.7, 1.0 + y_offset),
                    arrowprops={**arrow_style, 'color': COLORS['error'],
                                'linestyle': '--', 'alpha': 0.3})
        ax.plot([3.2, 3.5], [0.6 + y_offset, 0.9 + y_offset],
                color=COLORS['error'], linewidth=2)
        ax.plot([3.2, 3.5], [0.9 + y_offset, 0.6 + y_offset],
                color=COLORS['error'], linewidth=2)
    else:
        ax.annotate('', xy=(3.0, 0.4 + y_offset), xytext=(3.7, 1.0 + y_offset),
                    arrowprops={**arrow_style, 'color': COLORS['warning']})

    ax.annotate('', xy=(1.2, 1.4 + y_offset), xytext=(2.2, 1.9 + y_offset),
                arrowprops={**arrow_style, 'color': COLORS['text_secondary'], 'alpha': 0.5})
    ax.annotate('', xy=(3.8, 1.4 + y_offset), xytext=(2.8, 1.9 + y_offset),
                arrowprops={**arrow_style, 'color': COLORS['text_secondary'], 'alpha': 0.5})


def draw_causal_subgraph_horizontal(ax, cut=False):
    """ -  ( + LaTeX)"""
    from matplotlib.patches import FancyBboxPatch

    nodes = {
        'C': (2.5, 2.3),
        'G': (1.0, 1.3),
        'T': (4.0, 1.3),
        'Y': (2.5, 0.4),
    }

    labels = {
        'C': r'$\mathcal{C}$',
        'G': r'$\mathcal{G}$',
        'T': r'$\mathcal{T}$',
        'Y': r'$\mathcal{Y}$',
    }

    subtitles = {
        'C': 'Context',
        'G': 'Geometry',
        'T': 'Texture',
        'Y': 'Prediction',
    }

    node_width = 0.7
    node_height = 0.45

    for name, (x, y) in nodes.items():
        if name == 'G':
            color = COLORS['geometry']
        elif name == 'T':
            color = COLORS['texture']
        elif name == 'Y':
            color = COLORS['accent']
        else:
            color = COLORS['warning']

        bbox = FancyBboxPatch(
            (x - node_width/2, y - node_height/2),
            node_width, node_height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.92,
            zorder=5
        )
        ax.add_patch(bbox)

        ax.text(x, y + 0.02, labels[name], ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=6)

        ax.text(x, y - 0.55, subtitles[name], ha='center', va='top',
                fontsize=8, color='#586069', style='italic')

    arrow_color_normal = '#4a5568'
    arrow_style = dict(arrowstyle='-|>', mutation_scale=18, linewidth=2)

    ax.annotate('', xy=(1.35, 1.55), xytext=(2.15, 2.05),
                arrowprops={**arrow_style, 'color': arrow_color_normal, 'alpha': 0.5})

    ax.annotate('', xy=(3.65, 1.55), xytext=(2.85, 2.05),
                arrowprops={**arrow_style, 'color': arrow_color_normal, 'alpha': 0.5})

    ax.annotate('', xy=(2.15, 0.62), xytext=(1.35, 1.08),
                arrowprops={**arrow_style, 'color': COLORS['success'], 'linewidth': 2.5})

    if cut:
        ax.annotate('', xy=(2.85, 0.62), xytext=(3.65, 1.08),
                    arrowprops={**arrow_style, 'color': COLORS['error'],
                                'linestyle': (0, (5, 3)), 'alpha': 0.4, 'linewidth': 2})
        ax.text(1.55, 0.95, r'$\checkmark$', fontsize=12, color=COLORS['success'],
                fontweight='bold', ha='center', va='center')
        mid_x, mid_y = 3.25, 0.85
        ax.plot([mid_x - 0.12, mid_x + 0.12], [mid_y - 0.12, mid_y + 0.12],
                color=COLORS['error'], linewidth=3, zorder=10)
        ax.plot([mid_x - 0.12, mid_x + 0.12], [mid_y + 0.12, mid_y - 0.12],
                color=COLORS['error'], linewidth=3, zorder=10)
        ax.text(mid_x + 0.35, mid_y, r'$do(\mathcal{T})$', fontsize=10, color=COLORS['error'],
                fontweight='bold', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=COLORS['error'], alpha=0.8))
    else:
        ax.annotate('', xy=(2.85, 0.62), xytext=(3.65, 1.08),
                    arrowprops={**arrow_style, 'color': COLORS['warning'], 'linewidth': 2})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='figures/teaser_v2.png')
    parser.add_argument('--cfg', type=str,
                        default='cfgs/scannet/pointnext-b-mixratio.yaml')
    parser.add_argument('--baseline-ckpt', type=str,
                        default='log/scannet/scannet-b-base-32000/checkpoint/scannet-train-pointnext-b-baseline-ngpus1-20260106-092535-iZpHs7whrVDbtDYKaixqsk_ckpt_best.pth')
    parser.add_argument('--ours-ckpt', type=str,
                        default='log/scannet/scannet-b-ours/checkpoint/scannet-train-pointnext-b-mixratio-ngpus1-20260107-224716-RjVXUG5LmehovZfMDLEZyb_ckpt_best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-search', type=int, default=68)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--target', type=str, default='sofa',
                        choices=['sofa', 'table', 'toilet'],
                        help='Target class to attack (sofa, table, or toilet)')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📊 TEASER FIGURE GENERATOR (V2 - Same as Paper)")
    print("=" * 70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Device: {device} (GPU {args.gpu})")

    print(f"\n[2] Loading config: {args.cfg}")
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    print("\n[3] Loading models...")
    baseline = load_model(cfg, args.baseline_ckpt, device)
    ours = load_model(cfg, args.ours_ckpt, device)

    print("\n[4] Building dataloader (same as paper)...")
    val_loader = build_dataloader_from_cfg(
        1,  # batch_size=1
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split='val',
        distributed=False
    )
    print(f"  Validation samples: {len(val_loader.dataset)}")

    target_name = args.target
    target_id = TARGET_CLASS_OPTIONS[target_name]
    print(
        f"\n[5] Initializing conflict generator for '{target_name}' (class_id={target_id})...")
    generator = SemanticConflictGenerator(
        dataset='scannet',
        target_classes={target_id: target_name},
        attack_ratio=1.0,
        seed=42
    )

    best_sample = find_best_sample(
        val_loader, generator, baseline, ours, device, cfg, args.max_search,
        target_name=target_name
    )

    if best_sample is None:
        print("\n× No suitable sample found. Try increasing --max-search")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_teaser_figure(best_sample, args.output, args.dpi)

    print("\n" + "=" * 70)
    print("✨ Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
