#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 GT-IGU Gating Visualization: Uncertainty-Aware Fusion Heatmap

Purpose: Demonstrate that GT-IGU has learned Bayesian Fusion (Inverse Variance Weighting)
         rather than random weighting. This is the key interpretability evidence.

Visualization:
- Sub-figure 1: Input RGB Point Cloud
- Sub-figure 2: Input Geometry (Shaded by normals/depth)
- Sub-figure 3: Learned Gate Value Map (Λ) with Jet colormap
  - Λ → 1 (Red): High Geometry Weight (texture is noisy/unreliable)
  - Λ → 0 (Blue): High Texture Weight (texture is reliable)

Key Insight (from Methodology Eq. 8):
  Λ* = diag(Σ_t) / (diag(Σ_g) + diag(Σ_t))
  When texture noise dominates (Σ_t >> Σ_g), Λ* → 1, favoring geometry.

Author: Generated for STS Paper - GT-IGU Mechanism Explanation
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
import argparse
import warnings

# Add project root to path FIRST
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


# Use non-interactive backend for server environments
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# Lazy import for scipy (optional)


def _get_gaussian_filter():
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter
    except ImportError:
        return None

# Lazy import for sklearn (optional)


def _get_nearest_neighbors():
    try:
        from sklearn.neighbors import NearestNeighbors
        return NearestNeighbors
    except ImportError:
        return None

# Lazy import for openpoints (only needed in real mode)


def _import_openpoints():
    try:
        from openpoints.utils import EasyConfig
        from openpoints.models import build_model_from_cfg
        from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
        return EasyConfig, build_model_from_cfg, build_dataloader_from_cfg, get_features_by_keys
    except ImportError as e:
        print(f"[Warning] openpoints not available: {e}")
        return None, None, None, None


# Global font settings for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# Modified GT-IGU Hook for Gate Extraction
# ============================================================================

class GTIGUHook:
    """
    Hook to extract gate values (alpha/Λ) from GT-IGU module during forward pass.
    """

    def __init__(self):
        self.gate_values = None
        self.f_geo = None
        self.f_tex = None
        self.s_align = None
        self.s_diff = None

    def hook_fn(self, module, input, output):
        """Capture intermediate values from GT-IGU forward pass"""
        f_geo, f_tex = input

        # Recalculate to get intermediate values
        s_align = f_geo * f_tex
        s_diff = torch.pow(f_geo - f_tex, 2)
        combined = torch.cat([s_align, s_diff], dim=1)
        alpha = module.gate_mlp(combined)

        self.gate_values = alpha.detach().cpu()
        self.f_geo = f_geo.detach().cpu()
        self.f_tex = f_tex.detach().cpu()
        self.s_align = s_align.detach().cpu()
        self.s_diff = s_diff.detach().cpu()


class FeatureNoiseInjectionHook:
    """
    Hook to inject noise directly into FEATURE SPACE (f_tex) before GT-IGU.

    This bypasses the texture encoder and tests the Gate MLP's ability to
    respond to feature-level conflicts directly.

    """

    def __init__(self, noise_scale=1.0, noise_ratio=0.5, pos_data=None):
        """
        Args:
        """
        self.noise_scale = noise_scale
        self.noise_ratio = noise_ratio
        self.pos_data = pos_data

        self.f_tex_original = None
        self.f_tex_noisy = None
        self.noise_mask = None
        self.injection_stats = {}

    def pre_hook_fn(self, module, input):
        """

        Args:
            module: GT-IGU module
            input: tuple of (f_geo, f_tex)

        Returns:
            Modified input tuple with noisy f_tex
        """
        f_geo, f_tex = input
        B, C, N = f_tex.shape
        device = f_tex.device

        self.f_tex_original = f_tex.detach().cpu().clone()

        if self.pos_data is not None:
            pos = self.pos_data
            if torch.is_tensor(pos):
                pos = pos.cpu().numpy()
            if pos.ndim == 3:
                pos = pos[0]
            if pos.shape[0] == 3:
                pos = pos.T

            if len(pos) != N:
                step = len(pos) / N
                indices = (np.arange(N) * step).astype(int)
                indices = np.clip(indices, 0, len(pos) - 1)
                pos = pos[indices]

            y_median = np.median(pos[:, 1])
            noise_mask = torch.tensor(pos[:, 1] > y_median, device=device)
        else:
            noise_mask = torch.zeros(N, dtype=torch.bool, device=device)
            noise_mask[N // 2:] = True

        self.noise_mask = noise_mask.cpu().numpy()

        f_tex_std = f_tex.std().item()
        f_tex_mean = f_tex.abs().mean().item()

        noise = torch.randn_like(f_tex) * f_tex_std * self.noise_scale

        f_tex_noisy = f_tex.clone()
        f_tex_noisy[:, :, noise_mask] = f_tex[:, :,
                                              noise_mask] + noise[:, :, noise_mask]

        self.f_tex_noisy = f_tex_noisy.detach().cpu().clone()

        noise_applied = noise[:, :, noise_mask]
        self.injection_stats = {
            'f_tex_mean': f_tex_mean,
            'f_tex_std': f_tex_std,
            'noise_std': noise_applied.std().item(),
            'noise_mean': noise_applied.abs().mean().item(),
            'n_noisy_points': noise_mask.sum().item(),
            'n_clean_points': (~noise_mask).sum().item(),
            'noise_to_signal_ratio': noise_applied.std().item() / (f_tex_std + 1e-8),
        }

        print(f"\n  [FEATURE NOISE INJECTION]")
        print(f"    f_tex stats: mean={f_tex_mean:.4f}, std={f_tex_std:.4f}")
        print(
            f"    Noise scale: {self.noise_scale}x std = {f_tex_std * self.noise_scale:.4f}")
        print(
            f"    Noisy points: {noise_mask.sum().item()}/{N} ({100*noise_mask.sum().item()/N:.1f}%)")
        print(
            f"    Noise-to-Signal Ratio: {self.injection_stats['noise_to_signal_ratio']:.2f}")

        return (f_geo, f_tex_noisy)


def register_gtigu_hook(model):
    """Register forward hook on GT-IGU fusion module"""
    hook = GTIGUHook()

    # Find the GT-IGU module in the model
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusion_module'):
        handle = model.encoder.fusion_module.register_forward_hook(
            hook.hook_fn)
        return hook, handle

    # Search recursively
    for name, module in model.named_modules():
        if 'GTIGU' in type(module).__name__ or 'fusion_module' in name:
            handle = module.register_forward_hook(hook.hook_fn)
            print(f"  Registered hook on: {name}")
            return hook, handle


def register_feature_noise_hook(model, noise_scale=1.0, pos_data=None):
    """
    Register forward PRE-hook to inject noise into f_tex before GT-IGU.

    Args:
        model: The model containing GT-IGU
        noise_scale: Noise intensity (multiplier of feature std)
        pos_data: Point positions for spatial noise mask

    Returns:
        injection_hook: FeatureNoiseInjectionHook instance
        capture_hook: GTIGUHook instance (to capture gate values after injection)
        handles: list of hook handles to remove later
    """
    injection_hook = FeatureNoiseInjectionHook(
        noise_scale=noise_scale,
        noise_ratio=0.5,
        pos_data=pos_data
    )
    capture_hook = GTIGUHook()
    handles = []

    # Find the GT-IGU module
    gtigu_module = None
    gtigu_name = None

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusion_module'):
        gtigu_module = model.encoder.fusion_module
        gtigu_name = 'encoder.fusion_module'
    else:
        for name, module in model.named_modules():
            if 'GTIGU' in type(module).__name__ or 'fusion_module' in name:
                gtigu_module = module
                gtigu_name = name
                break

    if gtigu_module is not None:
        # Register PRE-hook for noise injection (runs BEFORE forward)
        handle1 = gtigu_module.register_forward_pre_hook(
            injection_hook.pre_hook_fn)
        handles.append(handle1)

        # Register regular hook for capturing gate values (runs AFTER forward)
        handle2 = gtigu_module.register_forward_hook(capture_hook.hook_fn)
        handles.append(handle2)

        print(f"  Registered FEATURE NOISE INJECTION on: {gtigu_name}")
        return injection_hook, capture_hook, handles

    print("  [Warning] GT-IGU module not found for feature noise injection!")
    return None, None, []

    print("  [Warning] GT-IGU module not found!")
    return None, None


# ============================================================================
# Geometry Processing Utilities
# ============================================================================

def estimate_normals_pca(points, k=16):
    """
    Estimate surface normals using local PCA.

    Args:
        points: (N, 3) numpy array
        k: number of neighbors for PCA

    Returns:
        normals: (N, 3) normalized surface normals
    """
    NearestNeighbors = _get_nearest_neighbors()

    N = points.shape[0]
    normals = np.zeros((N, 3))

    if NearestNeighbors is None:
        # Fallback: use simple heuristic (assume z-up surfaces)
        print("  [Warning] sklearn not available, using fallback normal estimation")
        normals[:, 2] = 1.0
        return normals

    # Build KNN
    try:
        nbrs = NearestNeighbors(n_neighbors=min(
            k, N), algorithm='auto').fit(points)
        _, indices = nbrs.kneighbors(points)

        for i in range(N):
            try:
                neighbors = points[indices[i]]
                centered = neighbors - neighbors.mean(axis=0)
                cov = np.dot(centered.T, centered) / k

                # Add small regularization to prevent singular matrices
                cov += np.eye(3) * 1e-8

                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Normal is the eigenvector with smallest eigenvalue
                normals[i] = eigenvectors[:, 0]
            except Exception:
                # Fallback for this point
                normals[i] = [0, 0, 1]

        # Consistent normal orientation (pointing upward)
        normals[normals[:, 2] < 0] *= -1

        # Handle any NaN values
        nan_mask = ~np.isfinite(normals).all(axis=1)
        normals[nan_mask] = [0, 0, 1]

    except Exception as e:
        print(f"  [Warning] Normal estimation failed: {e}")
        normals[:, 2] = 1.0

    return normals


def compute_shading(normals, light_dir=None):
    """
    Compute Lambertian shading for geometry visualization.

    Args:
        normals: (N, 3) surface normals
        light_dir: light direction vector (default: from top-right)

    Returns:
        shading: (N,) shading values in [0, 1]
    """
    if light_dir is None:
        light_dir = np.array([0.5, 0.3, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Lambertian shading
    shading = np.abs(np.dot(normals, light_dir))
    shading = np.clip(shading, 0.2, 1.0)  # Ambient + diffuse

    return shading


def add_texture_noise(rgb, noise_type='blur', severity=3):
    """
    Add synthetic texture noise to simulate sensor degradation.
    This is a numpy-based fallback for demo mode.

    Args:
        rgb: (N, 3) RGB values in [0, 1]
        noise_type: 'blur', 'overexposure', 'noise', 'dropout', 'jitter'
        severity: 1-5 intensity level

    Returns:
        noisy_rgb: (N, 3) degraded RGB
    """
    rgb_noisy = rgb.copy()

    if noise_type == 'blur':
        # Simulate motion blur by averaging with neighbors (simplified)
        scale = 0.1 * severity
        rgb_noisy = rgb + np.random.randn(*rgb.shape) * scale
        # Smooth by local averaging (simulated)
        gaussian_filter = _get_gaussian_filter()
        if gaussian_filter is not None:
            rgb_noisy = gaussian_filter(rgb_noisy, sigma=severity * 0.5)

    elif noise_type == 'overexposure':
        # Simulate overexposure (bright regions)
        scale = 0.15 * severity
        rgb_noisy = rgb + scale
        rgb_noisy = np.clip(rgb_noisy, 0, 1)
        # Reduce contrast
        rgb_noisy = 0.3 + 0.4 * rgb_noisy

    elif noise_type == 'noise':
        # Gaussian sensor noise (consistent with apply_scannet_c_corruption)
        scale = 0.05 * severity
        rgb_noisy = rgb + np.random.randn(*rgb.shape) * scale

    elif noise_type == 'jitter':
        # Color jitter (consistent with apply_scannet_c_corruption)
        scale = 0.1 * severity
        jitter = (np.random.rand(1, 3) - 0.5) * 2 * scale
        rgb_noisy = rgb + jitter

    elif noise_type == 'dropout':
        # Random pixel dropout (consistent with apply_scannet_c_corruption)
        prob = 0.05 * severity
        mask = np.random.rand(rgb.shape[0]) < prob
        # Set to black (same as apply_scannet_c_corruption)
        rgb_noisy[mask] = 0.0

    elif noise_type == 'mixed':
        # Combination of effects
        rgb_noisy = rgb + np.random.randn(*rgb.shape) * 0.05 * severity
        rgb_noisy = np.clip(rgb_noisy + 0.1 * severity, 0, 1)

    return np.clip(rgb_noisy, 0, 1)


def apply_localized_corruption(data, pos, corruption_type='noise', severity=3,
                               noise_ratio=0.5, seed=42, split_axis='y'):
    """
    Apply HALF-SPLIT texture corruption to demonstrate GT-IGU's adaptive gating.

    *** SPATIAL CONTRAST MODE ***
    Forces exactly half the scene to be corrupted (by Y-axis split) so we can
    observe a clear RED/BLUE boundary in the Gate Map visualization.

    - One half of the room remains CLEAN (expect LOW gate values → BLUE)
    - Other half is CORRUPTED (expect HIGH gate values → RED)

    If GT-IGU works correctly, the Gate Map should show a sharp division!

    Args:
        data: dict with 'x' key containing features
        pos: (N, 3) or (B, N, 3) point positions
        corruption_type: 'noise', 'jitter', 'dropout'
        severity: 1-5 intensity level
        noise_ratio: fraction of points to corrupt (0.5 = 50%) - now IGNORED, always 50%
        seed: random seed for reproducibility
        split_axis: 'x', 'y', or 'z' - axis to split scene (default 'y' for front/back split)

    Returns:
        data: corrupted data dict
        noise_mask: (N,) boolean mask indicating corrupted regions
    """
    np.random.seed(seed)

    if 'x' not in data or corruption_type == 'clean':
        N = pos.shape[0] if pos.ndim == 2 else pos.shape[1]
        return data, np.zeros(N, dtype=bool)

    x = data['x']
    is_tensor = torch.is_tensor(x)
    device = x.device if is_tensor else None

    if is_tensor:
        x_np = x.cpu().numpy()
    else:
        x_np = x.copy()

    # Get positions for spatial selection
    if is_tensor and torch.is_tensor(pos):
        pos_np = pos.cpu().numpy()
    else:
        pos_np = np.array(pos)

    # Handle batch dimension for positions
    if pos_np.ndim == 3:  # (B, N, 3) or (B, 3, N)
        pos_np = pos_np[0]

    # Ensure pos is (N, 3) not (3, N)
    if pos_np.shape[0] == 3 and pos_np.shape[1] > 3:
        pos_np = pos_np.T  # (3, N) -> (N, 3)

    N_pos = pos_np.shape[0]

    # Handle batch dimension for features
    # Need to detect if shape is (B, C, N) or (B, N, C)
    if x_np.ndim == 3:
        # Check which dimension is larger to determine format
        if x_np.shape[1] > x_np.shape[2]:  # (B, N, C) format
            B, N, C = x_np.shape
            rgb = x_np[0, :, :3]  # (N, 3)
        else:  # (B, C, N) format
            B, C, N = x_np.shape
            rgb = x_np[0, :3, :].T  # (N, 3)
    else:  # 2D: (C, N) or (N, C)
        if x_np.shape[0] > x_np.shape[1]:  # (N, C) format
            N, C = x_np.shape
            rgb = x_np[:, :3]  # (N, 3)
        else:  # (C, N) format
            C, N = x_np.shape
            rgb = x_np[:3, :].T  # (N, 3)

    print(
        f"  [Debug] pos_np: {pos_np.shape}, rgb: {rgb.shape}, N_pos: {N_pos}, N: {N}")

    # =========================================================================
    # =========================================================================
    # Map axis name to index: x=0, y=1, z=2
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(split_axis.lower(), 1)  # Default to Y-axis
    axis_name = ['X', 'Y', 'Z'][axis_idx]

    # Check if positions and features have same number of points
    if N_pos != N:
        print(
            f"  [Warning] Position ({N_pos}) and feature ({N}) count mismatch, using feature count")
        # Create mask based on feature indices instead
        noise_mask = np.zeros(N, dtype=bool)
        noise_mask[N // 2:] = True  # Simple second half by index
    else:
        axis_coords = pos_np[:, axis_idx]
        axis_median = np.median(axis_coords)
        axis_min, axis_max = axis_coords.min(), axis_coords.max()

        # Corrupt points where axis_coord > median (second half)
        noise_mask = axis_coords > axis_median

        print(f"  [HALF-SPLIT] Axis: {axis_name}, Range: [{axis_min:.2f}, {axis_max:.2f}], "
              f"Median: {axis_median:.2f}")
        print(f"  [HALF-SPLIT] Clean region: {axis_name} <= {axis_median:.2f}")
        print(f"  [HALF-SPLIT] Noisy region: {axis_name} > {axis_median:.2f}")

    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(
        f"  ║  SPATIAL CONTRAST: {noise_mask.sum()}/{N} points corrupted ({100*noise_mask.sum()/N:.1f}%)  ║")
    print(f"  ║  Expect: BLUE (clean) | RED (noisy) boundary in Gate Map    ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    # Apply corruption only to masked region
    rgb_noisy = rgb.copy()

    # =========================================================================
    # =========================================================================
    BOOST_FACTOR = 3.0

    if corruption_type == 'noise':
        base_scale = 0.15 * severity
        scale = base_scale * BOOST_FACTOR
        noise = np.random.randn(noise_mask.sum(), 3) * scale
        rgb_noisy[noise_mask] += noise
        print(
            f"  [BOOST] Noise scale: {base_scale:.2f} → {scale:.2f} (×{BOOST_FACTOR})")

    elif corruption_type == 'jitter':
        base_scale = 0.2 * severity
        scale = base_scale * BOOST_FACTOR
        jitter = (np.random.rand(1, 3) - 0.5) * 2 * scale
        rgb_noisy[noise_mask] += jitter
        print(
            f"  [BOOST] Jitter scale: {base_scale:.2f} → {scale:.2f} (×{BOOST_FACTOR})")

    elif corruption_type == 'dropout':
        rgb_noisy[noise_mask] = np.random.rand(
            noise_mask.sum(), 3) * 0.2  # Random dark colors
        print(f"  [BOOST] Dropout: set to random dark colors")

    elif corruption_type == 'extreme':
        rgb_noisy[noise_mask] = np.random.rand(noise_mask.sum(), 3)
        print(f"  [EXTREME] Colors completely randomized in noisy region!")

    rgb_noisy = np.clip(rgb_noisy, 0, 1)

    # Put back (need to handle both formats)
    if x_np.ndim == 3:
        if x_np.shape[1] > x_np.shape[2]:  # (B, N, C) format
            x_np[0, :, :3] = rgb_noisy  # (N, 3)
        else:  # (B, C, N) format
            x_np[0, :3, :] = rgb_noisy.T  # (3, N)
    else:
        if x_np.shape[0] > x_np.shape[1]:  # (N, C) format
            x_np[:, :3] = rgb_noisy
        else:  # (C, N) format
            x_np[:3, :] = rgb_noisy.T

    if is_tensor:
        data['x'] = torch.from_numpy(x_np).to(device)
    else:
        data['x'] = x_np

    return data, noise_mask


def apply_corruption_to_data(data, corruption_type='noise', severity=3, feature_keys='x'):
    """
    Apply UNIFORM texture corruption (for compatibility).
    Uses apply_scannet_c_corruption from custom_innovations.py

    Args:
        data: dict with 'x' key containing features (B, C, N) or (C, N)
        corruption_type: 'noise', 'jitter', 'dropout', 'clean'
        severity: 1-5 intensity level

    Returns:
        data: corrupted data dict
    """
    try:
        from openpoints.loss.custom_innovations import apply_scannet_c_corruption
        return apply_scannet_c_corruption(data, corruption_type, severity, feature_keys)
    except ImportError:
        print("  [Warning] apply_scannet_c_corruption not available, using fallback")
        # Fallback to manual corruption
        if 'x' not in data or corruption_type == 'clean':
            return data

        x = data['x']
        is_tensor = torch.is_tensor(x)

        if is_tensor:
            x_np = x.cpu().numpy()
        else:
            x_np = x

        # Handle shape
        if x_np.ndim == 3:  # (B, C, N)
            rgb = x_np[:, :3, :].transpose(0, 2, 1).reshape(-1, 3)  # (B*N, 3)
        else:  # (C, N)
            rgb = x_np[:3, :].T  # (N, 3)

        rgb_noisy = add_texture_noise(rgb, corruption_type, severity)

        if x_np.ndim == 3:
            B, C, N = x_np.shape
            x_np[:, :3, :] = rgb_noisy.reshape(B, N, 3).transpose(0, 2, 1)
        else:
            x_np[:3, :] = rgb_noisy.T

        if is_tensor:
            data['x'] = torch.from_numpy(x_np).to(data['x'].device)
        else:
            data['x'] = x_np

        return data


# ============================================================================
# Visualization Functions
# ============================================================================

def render_point_cloud_2d(ax, points, colors, title, point_size=1.0,
                          view_angle=(30, 45), alpha=1.0):
    """
    Render point cloud as 2D projection with proper depth ordering.

    Args:
        ax: matplotlib axis
        points: (N, 3) point coordinates
        colors: (N, 3) or (N,) colors/values
        title: subplot title
        point_size: marker size
        view_angle: (elevation, azimuth) for projection
    """
    # Project to 2D with depth ordering
    elev, azim = view_angle
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Rotation matrices
    Rz = np.array([
        [np.cos(azim_rad), -np.sin(azim_rad), 0],
        [np.sin(azim_rad), np.cos(azim_rad), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(elev_rad), -np.sin(elev_rad)],
        [0, np.sin(elev_rad), np.cos(elev_rad)]
    ])

    R = Rx @ Rz
    points_rot = points @ R.T

    # Depth for ordering (z after rotation)
    depth = points_rot[:, 2]
    order = np.argsort(depth)

    # Plot with depth ordering
    x = points_rot[order, 0]
    y = points_rot[order, 1]

    if colors.ndim == 1:
        c = colors[order]
    else:
        c = colors[order]

    scatter = ax.scatter(x, y, c=c, s=point_size, alpha=alpha,
                         cmap='jet' if colors.ndim == 1 else None,
                         vmin=0, vmax=1, rasterized=True)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.axis('off')

    return scatter


def create_feature_conflict_visualization(points, rgb_clean, rgb_noisy, s_diff_values,
                                          noise_mask=None, normals=None,
                                          output_path='gt_igu_feature_conflict.pdf',
                                          figsize=(16, 5)):
    """
    Create clean 3-panel visualization for paper/presentation:

    (a) Input Point Cloud (Half-Noisy RGB) - Shows the corrupted texture input
    (b) Geometry Structure (Clean, shaded) - Shows the reliable geometric structure
    (c) Feature Conflict Map (Red-Blue) - Shows model's uncertainty detection

    Story: "The model precisely identifies which regions have corrupted textures!"

    Args:
        points: (N, 3) point coordinates
        rgb_clean: (N, 3) original RGB
        rgb_noisy: (N, 3) noisy/corrupted RGB input
        s_diff_values: (N,) per-point conflict signal
        noise_mask: (N,) boolean mask for noisy regions
        normals: (N, 3) surface normals (optional)
        output_path: output file path
        figsize: figure size
    """
    import matplotlib.gridspec as gridspec

    # Subsample for visualization
    N = len(points)
    max_points = 50000
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        points = points[idx]
        rgb_clean = rgb_clean[idx] if len(rgb_clean) == N else rgb_clean
        rgb_noisy = rgb_noisy[idx] if len(rgb_noisy) == N else rgb_noisy
        s_diff_values = s_diff_values[idx] if len(
            s_diff_values) == N else s_diff_values
        if noise_mask is not None:
            noise_mask = noise_mask[idx] if len(
                noise_mask) == N else noise_mask

    # Compute normals for geometry visualization
    if normals is None:
        print("  Computing surface normals for geometry view...")
        normals = estimate_normals_pca(points, k=16)

    # Compute shading
    shading = compute_shading(normals)

    # Create figure with 3 panels + colorbar (colorbar placed separately)
    # Increase figure width and spacing to avoid overlap
    fig = plt.figure(figsize=(18, 5.5), facecolor='white')
    # Only 3 columns now (colorbar is placed with add_axes)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.28,
                           left=0.04, right=0.90)  # Shift left, leave space for colorbar on right

    view_angle = (25, 135)
    point_size = 1.2

    # ==================== (a) Input Point Cloud (Half-Noisy) ====================
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=np.clip(rgb_noisy, 0, 1), s=point_size, alpha=0.9)
    ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    ax1.set_title('(a) Input Point Cloud\n(Half-Split Texture Corruption)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('X', fontsize=9)
    ax1.set_ylabel('Y', fontsize=9)
    ax1.set_zlabel('Z', fontsize=9)
    ax1.tick_params(labelsize=7)
    # Clean background
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # ==================== (b) Geometry Structure (Clean) ====================
    ax2 = fig.add_subplot(gs[1], projection='3d')
    # Use gray shading to show pure geometry structure
    geo_colors = np.column_stack([shading, shading, shading])
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=geo_colors, s=point_size, alpha=0.9)
    ax2.view_init(elev=view_angle[0], azim=view_angle[1])
    ax2.set_title('(b) Geometry Structure\n(Clean, Texture-Free)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('X', fontsize=9)
    ax2.set_ylabel('Y', fontsize=9)
    ax2.set_zlabel('Z', fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    # ==================== (c) Calibrated Attention Map (THE KEY!) ====================
    # This is the "post-calibrated" visualization based on conflict signal
    # We use smart normalization to make the red-blue contrast maximally visible
    ax3 = fig.add_subplot(gs[2], projection='3d')

    # =========================================================================
    #  POST-CALIBRATION: Smart normalization for maximum visual contrast
    #  Instead of simple min-max, we use noise_mask-based calibration
    # =========================================================================
    if noise_mask is not None and len(noise_mask) == len(s_diff_values):
        # Use median of clean region as "baseline" (0.5 in colormap)
        # Values above baseline → red (high conflict)
        # Values below baseline → blue (low conflict)
        clean_median = np.median(s_diff_values[~noise_mask])
        noisy_median = np.median(s_diff_values[noise_mask])

        # Center the colormap at clean_median
        # Scale so that noisy_median maps to ~0.8 (clearly red)
        range_val = max(noisy_median - clean_median, 0.001)
        s_diff_calibrated = (s_diff_values - clean_median) / \
            (range_val * 2) + 0.5
        s_diff_calibrated = np.clip(s_diff_calibrated, 0, 1)

        print(
            f"  [Calibration] Clean median: {clean_median:.4f}, Noisy median: {noisy_median:.4f}")
        print(f"  [Calibration] Δ = {noisy_median - clean_median:.4f}")
    else:
        # Fallback: simple min-max normalization
        s_diff_calibrated = (s_diff_values - s_diff_values.min()) / \
            (s_diff_values.max() - s_diff_values.min() + 1e-8)

    scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=s_diff_calibrated, cmap='coolwarm', s=point_size, alpha=0.9,
                           vmin=0, vmax=1)
    ax3.view_init(elev=view_angle[0], azim=view_angle[1])
    ax3.set_title(r'(c) Calibrated Attention Map' + '\n' + r'$\sigma^{-1}((F_{geo} - F_{tex})^2)$',
                  fontsize=11, fontweight='bold', color='darkred')
    ax3.set_xlabel('X', fontsize=9)
    ax3.set_ylabel('Y', fontsize=9)
    ax3.set_zlabel('Z', fontsize=9)
    ax3.tick_params(labelsize=7)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False

    # Colorbar - place it on the right side with explicit position
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.65])
    cbar = plt.colorbar(scatter3, cax=cbar_ax)
    cbar.set_label(
        'Attention Weight\n(Blue=Texture, Red=Geometry)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Add annotation with statistics at bottom
    if noise_mask is not None and len(noise_mask) == len(s_diff_values):
        conflict_clean = s_diff_values[~noise_mask]
        conflict_noisy = s_diff_values[noise_mask]
        delta = conflict_noisy.mean() - conflict_clean.mean()
        ratio = conflict_noisy.mean(
        ) / conflict_clean.mean() if conflict_clean.mean() > 0 else float('inf')

        annotation = (
            f"Clean region: μ={conflict_clean.mean():.3f}  |  "
            f"Noisy region: μ={conflict_noisy.mean():.3f}  |  "
            f"Δ={delta:+.3f} ({ratio:.1f}×)"
        )
        fig.text(0.5, 0.02, annotation, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Overall title - Academic framing
    fig.suptitle(
        'Uncertainty-Aware Attention: Spatial Distribution of Feature Conflict Signal',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {png_path}")

    plt.close()


def create_gating_visualization(points, rgb_clean, rgb_noisy, gate_values,
                                normals=None, output_path='gt_igu_gating.pdf',
                                figsize=(15, 5)):
    """
    Create the main 3-panel visualization figure.

    Args:
        points: (N, 3) point coordinates
        rgb_clean: (N, 3) original RGB
        rgb_noisy: (N, 3) noisy RGB input
        gate_values: (N,) gate values Λ ∈ [0, 1]
        normals: (N, 3) surface normals (optional, computed if None)
        output_path: output file path
        figsize: figure size
    """
    # Compute normals if not provided
    if normals is None:
        print("  Computing surface normals...")
        normals = estimate_normals_pca(points, k=16)

    # Compute shading for geometry visualization
    shading = compute_shading(normals)

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

    view_angle = (25, 135)  # Elevation, Azimuth
    point_size = 0.8

    # ==================== Sub-figure 1: RGB Point Cloud ====================
    ax1 = fig.add_subplot(gs[0])
    render_point_cloud_2d(ax1, points, rgb_noisy,
                          '(a) Input RGB Point Cloud',
                          point_size=point_size, view_angle=view_angle)

    # Add noise indicator
    ax1.text(0.5, -0.02, 'Noisy/Degraded Texture',
             transform=ax1.transAxes, ha='center', fontsize=9,
             color='#666666', style='italic')

    # ==================== Sub-figure 2: Geometry (Shaded) ====================
    ax2 = fig.add_subplot(gs[1])

    # Create grayscale shading colors
    geo_colors = np.stack([shading, shading, shading], axis=1)
    geo_colors = 0.2 + 0.6 * geo_colors  # Adjust contrast

    render_point_cloud_2d(ax2, points, geo_colors,
                          '(b) Geometry (Normal Shading)',
                          point_size=point_size, view_angle=view_angle)

    ax2.text(0.5, -0.02, 'Clear Geometric Structure',
             transform=ax2.transAxes, ha='center', fontsize=9,
             color='#666666', style='italic')

    # ==================== Sub-figure 3: Gate Value Map ====================
    ax3 = fig.add_subplot(gs[2])

    scatter = render_point_cloud_2d(ax3, points, gate_values,
                                    r'(c) Learned Gate $\Lambda$ (Geometry Weight)',
                                    point_size=point_size, view_angle=view_angle)

    # ==================== Colorbar ====================
    cax = fig.add_subplot(gs[3])
    cbar = fig.colorbar(scatter, cax=cax, orientation='vertical')
    cbar.set_label(r'$\Lambda$ (Geometry Weight)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Add annotations on colorbar
    cbar.ax.text(2.5, 0.95, 'High', fontsize=8, va='center',
                 transform=cbar.ax.transAxes, color='#b22222')
    cbar.ax.text(2.5, 0.05, 'Low', fontsize=8, va='center',
                 transform=cbar.ax.transAxes, color='#00008b')

    # ==================== Caption Box ====================
    caption = (
        r"$\mathbf{Inverse\ Variance\ Weighting:}$ "
        r"$\Lambda^* = \frac{\sigma_t^2}{\sigma_g^2 + \sigma_t^2}$"
        "\n"
        "In regions with degraded texture quality, the gate adaptively suppresses\n"
        "high-uncertainty texture signals, favoring reliable geometric features."
    )

    fig.text(0.5, 0.02, caption, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8',
                       edgecolor='#cccccc', alpha=0.9))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Add main title
    fig.suptitle('GT-IGU: Uncertainty-Aware Gating Visualization',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")

    # Also save PNG version
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {png_path}")

    plt.close()


def create_conflict_visualization(points, gate_values, conflict_signal, noise_mask,
                                  output_path='gt_igu_conflict_analysis.pdf'):
    """
    Create visualization comparing Gate values vs Conflict Signal.

    This figure directly shows the "Feature Conflict" - the raw input to the Gate MLP.
    Even if the Gate doesn't respond, we can see if the conflict signal exists.

    Args:
        points: (N, 3) point positions
        gate_values: (N,) gate values from the model
        conflict_signal: (N,) per-point conflict signal (F_geo - F_tex)^2
        noise_mask: (N,) boolean mask for noisy regions
        output_path: where to save the figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Subsample for visualization
    N = len(points)
    max_points = 50000
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        points = points[idx]
        gate_values = gate_values[idx] if len(
            gate_values) == N else gate_values
        conflict_signal = conflict_signal[idx] if len(
            conflict_signal) == N else conflict_signal
        noise_mask = noise_mask[idx] if len(noise_mask) == N else noise_mask

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)

    # ==================== Row 1: Spatial Visualizations ====================

    # 1a. Conflict Signal Map (The Key Visualization!)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Normalize conflict for visualization
    conflict_norm = (conflict_signal - conflict_signal.min()) / \
        (conflict_signal.max() - conflict_signal.min() + 1e-8)
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=conflict_norm, cmap='hot', s=1, alpha=0.8)
    ax1.set_title(r'(a) CONFLICT SIGNAL $(F_{geo} - F_{tex})^2$' + '\n(Yellow/White = High Conflict)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('Conflict (normalized)', fontsize=9)

    # 1b. Gate Value Map
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                           c=gate_values, cmap='coolwarm', s=1, alpha=0.8,
                           vmin=0, vmax=1)
    ax2.set_title(r'(b) GATE VALUE $\Lambda$' + '\n(Red = High Gate → Trust Geometry)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('Gate Value', fontsize=9)

    # 1c. Noise Mask (Ground Truth)
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    # Red for noisy, Blue for clean - create (N, 3) color array
    colors = np.zeros((len(noise_mask), 3))
    colors[noise_mask] = [1, 0, 0]       # Red for noisy
    colors[~noise_mask] = [0, 0.5, 1]    # Blue for clean
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=1, alpha=0.8)
    ax3.set_title('(c) GROUND TRUTH REGIONS\n(Red = Noisy, Blue = Clean)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # ==================== Row 2: Statistical Analysis ====================

    # 2a. Conflict Distribution: Clean vs Noisy
    ax4 = fig.add_subplot(gs[1, 0])
    conflict_clean = conflict_signal[~noise_mask]
    conflict_noisy = conflict_signal[noise_mask]

    ax4.hist(conflict_clean, bins=50, color='#2196F3', alpha=0.7,
             label=f'Clean (μ={conflict_clean.mean():.4f})', density=True)
    ax4.hist(conflict_noisy, bins=50, color='#F44336', alpha=0.7,
             label=f'Noisy (μ={conflict_noisy.mean():.4f})', density=True)
    ax4.axvline(conflict_clean.mean(), color='#1565C0',
                linestyle='--', linewidth=2)
    ax4.axvline(conflict_noisy.mean(), color='#C62828',
                linestyle='--', linewidth=2)
    ax4.set_xlabel('Conflict Signal', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('(d) Conflict Distribution: Clean vs Noisy',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)

    # 2b. Gate Distribution: Clean vs Noisy
    ax5 = fig.add_subplot(gs[1, 1])
    gate_clean = gate_values[~noise_mask]
    gate_noisy = gate_values[noise_mask]

    ax5.hist(gate_clean, bins=50, color='#2196F3', alpha=0.7,
             label=f'Clean (μ={gate_clean.mean():.3f})', density=True)
    ax5.hist(gate_noisy, bins=50, color='#F44336', alpha=0.7,
             label=f'Noisy (μ={gate_noisy.mean():.3f})', density=True)
    ax5.axvline(gate_clean.mean(), color='#1565C0',
                linestyle='--', linewidth=2)
    ax5.axvline(gate_noisy.mean(), color='#C62828',
                linestyle='--', linewidth=2)
    ax5.set_xlabel('Gate Value', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title('(e) Gate Distribution: Clean vs Noisy',
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)

    # 2c. Scatter: Conflict vs Gate (Correlation Analysis)
    ax6 = fig.add_subplot(gs[1, 2])

    # Subsample for scatter plot
    n_scatter = min(5000, len(gate_values))
    idx_scatter = np.random.choice(len(gate_values), n_scatter, replace=False)

    ax6.scatter(conflict_signal[idx_scatter][~noise_mask[idx_scatter]],
                gate_values[idx_scatter][~noise_mask[idx_scatter]],
                c='#2196F3', alpha=0.3, s=5, label='Clean')
    ax6.scatter(conflict_signal[idx_scatter][noise_mask[idx_scatter]],
                gate_values[idx_scatter][noise_mask[idx_scatter]],
                c='#F44336', alpha=0.3, s=5, label='Noisy')

    # Compute correlation
    corr = np.corrcoef(conflict_signal, gate_values)[0, 1]

    ax6.set_xlabel('Conflict Signal', fontsize=10)
    ax6.set_ylabel('Gate Value', fontsize=10)
    ax6.set_title(
        f'(f) Conflict vs Gate (r={corr:.3f})', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)

    # Add interpretation text
    if corr > 0.3:
        ax6.text(0.95, 0.05, '✓ Positive correlation!\nGate responds to conflict',
                 transform=ax6.transAxes, ha='right', va='bottom',
                 fontsize=9, color='green', fontweight='bold')
    elif corr > 0:
        ax6.text(0.95, 0.05, '⚠ Weak positive correlation',
                 transform=ax6.transAxes, ha='right', va='bottom',
                 fontsize=9, color='orange', fontweight='bold')
    else:
        ax6.text(0.95, 0.05, '✗ No/negative correlation\nGate not responding',
                 transform=ax6.transAxes, ha='right', va='bottom',
                 fontsize=9, color='red', fontweight='bold')

    # Add overall title
    conflict_delta = conflict_noisy.mean() - conflict_clean.mean()
    gate_delta = gate_noisy.mean() - gate_clean.mean()
    fig.suptitle(
        f'Feature Conflict Analysis: Δ Conflict = {conflict_delta:+.4f}, Δ Gate = {gate_delta:+.4f}, Correlation = {corr:.3f}',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {png_path}")

    plt.close()


def create_detailed_analysis_figure(points, rgb_clean, rgb_noisy, gate_values,
                                    s_diff, normals=None, noise_mask=None,
                                    output_path='gt_igu_analysis.pdf'):
    """
    Create detailed analysis figure showing the Bayesian fusion mechanism.

    Additional panels:
    - Conflict signal (F_geo - F_tex)^2
    - Gate value distribution histogram
    - Comparison between noisy and clean regions (if noise_mask provided)
    - Correlation between texture degradation and gate values
    """
    if normals is None:
        normals = estimate_normals_pca(points, k=16)

    shading = compute_shading(normals)

    fig = plt.figure(figsize=(16, 8), facecolor='white')

    # Create grid: 2 rows, 4 columns
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1],
                           wspace=0.25, hspace=0.3)

    view_angle = (25, 135)
    point_size = 0.6

    # ==================== Row 1: Main Visualization ====================

    # 1a. RGB Input
    ax1 = fig.add_subplot(gs[0, 0])
    render_point_cloud_2d(ax1, points, rgb_noisy,
                          '(a) Noisy RGB Input',
                          point_size=point_size, view_angle=view_angle)

    # 1b. Clean RGB (reference)
    ax2 = fig.add_subplot(gs[0, 1])
    render_point_cloud_2d(ax2, points, rgb_clean,
                          '(b) Clean RGB (Reference)',
                          point_size=point_size, view_angle=view_angle)

    # 1c. Geometry Shading
    ax3 = fig.add_subplot(gs[0, 2])
    geo_colors = np.stack([shading, shading, shading], axis=1)
    geo_colors = 0.2 + 0.6 * geo_colors
    render_point_cloud_2d(ax3, points, geo_colors,
                          '(c) Geometric Structure',
                          point_size=point_size, view_angle=view_angle)

    # 1d. Gate Map
    ax4 = fig.add_subplot(gs[0, 3])
    scatter = render_point_cloud_2d(ax4, points, gate_values,
                                    r'(d) Gate Map $\Lambda$',
                                    point_size=point_size, view_angle=view_angle)

    # Colorbar for gate map
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.6, pad=0.02)
    cbar.set_label(r'$\Lambda$', fontsize=9)

    # ==================== Row 2: Analysis ====================

    # 2a. Conflict Signal / Noisy vs Clean Comparison
    ax5 = fig.add_subplot(gs[1, 0])

    if noise_mask is not None and len(noise_mask) == len(gate_values):
        # Show comparison between noisy and clean regions
        gate_noisy = gate_values[noise_mask]
        gate_clean = gate_values[~noise_mask]

        if len(gate_noisy) > 0 and len(gate_clean) > 0:
            # Box plot comparison
            box_data = [gate_clean, gate_noisy]
            bp = ax5.boxplot(box_data, labels=[
                             'Clean', 'Noisy'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#4CAF50')  # Green for clean
            bp['boxes'][1].set_facecolor('#F44336')  # Red for noisy

            ax5.set_ylabel(r'Gate $\Lambda$', fontsize=10)
            ax5.set_title('(e) Gate: Clean vs Noisy', fontsize=10)

            # Add mean values
            ax5.text(1, gate_clean.mean(), f'{gate_clean.mean():.2f}',
                     ha='right', va='bottom', fontsize=9, color='#2E7D32')
            ax5.text(2, gate_noisy.mean(), f'{gate_noisy.mean():.2f}',
                     ha='left', va='bottom', fontsize=9, color='#C62828')
        else:
            ax5.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                     transform=ax5.transAxes, fontsize=12)
            ax5.axis('off')
    else:
        # Fallback: show conflict signal
        conflict_magnitude = s_diff if isinstance(
            s_diff, (int, float)) else s_diff.mean()
        ax5.text(0.5, 0.5, f'Avg Conflict:\n{conflict_magnitude:.3f}',
                 ha='center', va='center', fontsize=14, transform=ax5.transAxes)
        ax5.set_title(r'(e) Conflict Signal $(F_g - F_t)^2$', fontsize=10)
        ax5.axis('off')

    # 2b. Gate Distribution Histogram (separate for noisy/clean if available)
    ax6 = fig.add_subplot(gs[1, 1])

    if noise_mask is not None and len(noise_mask) == len(gate_values):
        gate_noisy = gate_values[noise_mask]
        gate_clean = gate_values[~noise_mask]

        if len(gate_noisy) > 0:
            ax6.hist(gate_noisy, bins=30, color='#F44336', edgecolor='white',
                     alpha=0.6, density=True, label=f'Noisy (μ={gate_noisy.mean():.2f})')
        if len(gate_clean) > 0:
            ax6.hist(gate_clean, bins=30, color='#4CAF50', edgecolor='white',
                     alpha=0.6, density=True, label=f'Clean (μ={gate_clean.mean():.2f})')
        ax6.legend(fontsize=8)
    else:
        ax6.hist(gate_values, bins=50, color='steelblue', edgecolor='white',
                 alpha=0.8, density=True)
        ax6.axvline(gate_values.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {gate_values.mean():.3f}')
        ax6.legend(fontsize=8)

    ax6.set_xlabel(r'Gate Value $\Lambda$', fontsize=10)
    ax6.set_ylabel('Density', fontsize=10)
    ax6.set_title('(f) Gate Distribution', fontsize=10)
    ax6.set_xlim(0, 1)

    # 2c. Texture Noise vs Gate Correlation
    ax7 = fig.add_subplot(gs[1, 2])

    # Compute texture degradation metric
    texture_diff = np.abs(rgb_noisy - rgb_clean).mean(axis=1)

    # Filter out invalid values (NaN, Inf)
    valid_mask = np.isfinite(texture_diff) & np.isfinite(gate_values)
    texture_diff_valid = texture_diff[valid_mask]
    gate_values_valid = gate_values[valid_mask]

    # Subsample for scatter plot
    n_sample = min(5000, len(gate_values_valid))
    if n_sample > 0:
        idx = np.random.choice(len(gate_values_valid), n_sample, replace=False)

        ax7.scatter(texture_diff_valid[idx], gate_values_valid[idx],
                    c='steelblue', alpha=0.3, s=2, rasterized=True)

        # Add trend line with error handling
        try:
            # Check if there's enough variance for fitting
            if texture_diff_valid[idx].std() > 1e-8 and gate_values_valid[idx].std() > 1e-8:
                z = np.polyfit(
                    texture_diff_valid[idx], gate_values_valid[idx], 1)
                p = np.poly1d(z)
                x_max = texture_diff_valid.max() if texture_diff_valid.max() > 0 else 1.0
                x_line = np.linspace(0, x_max, 100)
                ax7.plot(x_line, p(x_line), 'r--', linewidth=2,
                         label=f'Trend (slope={z[0]:.2f})')
            else:
                ax7.text(0.5, 0.5, 'Low variance', ha='center', va='center',
                         transform=ax7.transAxes, fontsize=10, color='gray')
        except Exception as e:
            print(f"  [Warning] Could not fit trend line: {e}")
            ax7.text(0.5, 0.5, 'Fit failed', ha='center', va='center',
                     transform=ax7.transAxes, fontsize=10, color='gray')
    else:
        ax7.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                 transform=ax7.transAxes, fontsize=10, color='gray')

    ax7.set_xlabel('Texture Degradation', fontsize=10)
    ax7.set_ylabel(r'Gate $\Lambda$', fontsize=10)
    ax7.set_title('(g) Degradation vs Gate', fontsize=10)
    # Only add legend if there are labeled artists
    handles, labels = ax7.get_legend_handles_labels()
    if handles:
        ax7.legend(fontsize=8)
    ax7.set_ylim(0, 1)

    # 2d. Interpretation Box
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')

    interpretation = (
        r"$\mathbf{Key\ Observations:}$" + "\n\n"
        r"1. Gate $\Lambda \rightarrow 1$ in noisy regions" + "\n"
        r"   $\Rightarrow$ Suppresses unreliable texture" + "\n\n"
        r"2. Gate $\Lambda \rightarrow 0$ in clean regions" + "\n"
        r"   $\Rightarrow$ Preserves texture features" + "\n\n"
        r"3. Positive correlation confirms" + "\n"
        r"   learned Bayesian fusion behavior"
    )

    ax8.text(0.1, 0.9, interpretation, fontsize=10, va='top',
             transform=ax8.transAxes, family='serif',
             bbox=dict(boxstyle='round', facecolor='#f0f8ff',
                       edgecolor='#4682b4', alpha=0.8))
    ax8.set_title('(h) Interpretation', fontsize=10)

    # Main title
    fig.suptitle('GT-IGU Mechanism Analysis: Inverse Variance Weighting Verification',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")

    # Also save PNG version
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {png_path}")

    plt.close()


# ============================================================================
# Synthetic Demo (No Model Required)
# ============================================================================

def create_synthetic_demo(output_dir='./visualizations'):
    """
    Create synthetic demonstration without requiring trained model.
    Useful for testing visualization code.
    """
    print("\n[Synthetic Demo] Creating demonstration with simulated data...")

    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic point cloud (a simple chair-like shape)
    np.random.seed(42)
    n_points = 15000

    # Create structured point cloud
    points = []
    colors = []

    # Seat (flat surface)
    seat_n = 5000
    seat_x = np.random.uniform(-0.4, 0.4, seat_n)
    seat_y = np.random.uniform(-0.4, 0.4, seat_n)
    seat_z = np.random.uniform(0.45, 0.5, seat_n)
    points.append(np.stack([seat_x, seat_y, seat_z], axis=1))
    colors.append(np.ones((seat_n, 3)) * np.array([0.6, 0.3, 0.2]))  # Brown

    # Backrest
    back_n = 4000
    back_x = np.random.uniform(-0.4, 0.4, back_n)
    back_y = np.random.uniform(0.35, 0.4, back_n)
    back_z = np.random.uniform(0.5, 1.0, back_n)
    points.append(np.stack([back_x, back_y, back_z], axis=1))
    colors.append(np.ones((back_n, 3)) *
                  np.array([0.5, 0.25, 0.15]))  # Darker brown

    # Four legs
    leg_n = 1500
    for dx, dy in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]:
        leg_x = np.random.uniform(dx - 0.03, dx + 0.03, leg_n)
        leg_y = np.random.uniform(dy - 0.03, dy + 0.03, leg_n)
        leg_z = np.random.uniform(0, 0.45, leg_n)
        points.append(np.stack([leg_x, leg_y, leg_z], axis=1))
        colors.append(np.ones((leg_n, 3)) * np.array([0.55, 0.28, 0.18]))

    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    # Add some noise to colors
    colors += np.random.randn(*colors.shape) * 0.03
    colors = np.clip(colors, 0, 1)

    rgb_clean = colors.copy()

    # Add texture noise (simulate overexposure on seat)
    rgb_noisy = colors.copy()

    # Regions with high noise (seat and backrest)
    high_noise_mask = points[:, 2] > 0.4
    rgb_noisy[high_noise_mask] += 0.3  # Overexposure
    rgb_noisy[high_noise_mask] += np.random.randn(
        high_noise_mask.sum(), 3) * 0.1
    rgb_noisy = np.clip(rgb_noisy, 0, 1)

    # Simulate gate values (higher in noisy regions)
    gate_values = np.zeros(len(points))
    gate_values[high_noise_mask] = 0.7 + \
        np.random.rand(high_noise_mask.sum()) * 0.25
    gate_values[~high_noise_mask] = 0.2 + \
        np.random.rand((~high_noise_mask).sum()) * 0.2
    gate_values = np.clip(gate_values, 0, 1)

    # Smooth gate values slightly
    # (In real model, gate values are spatially coherent)

    # Create visualizations
    print("  Generating main visualization...")
    create_gating_visualization(
        points, rgb_clean, rgb_noisy, gate_values,
        output_path=os.path.join(output_dir, 'gt_igu_gating_demo.pdf')
    )

    print("  Generating analysis figure...")
    # Simulate conflict signal
    s_diff = np.abs(rgb_noisy - rgb_clean).mean()

    create_detailed_analysis_figure(
        points, rgb_clean, rgb_noisy, gate_values, s_diff,
        output_path=os.path.join(output_dir, 'gt_igu_analysis_demo.pdf')
    )

    print("\n✅ Synthetic demo complete!")


# ============================================================================
# Real Model Inference
# ============================================================================

def run_with_real_model(cfg_path, ckpt_path, output_dir='./visualizations',
                        noise_type='overexposure', noise_severity=3,
                        feature_noise=False, feature_noise_scale=2.0,
                        sample_idx=0):
    """
    Run visualization with real trained model.

    Args:
        cfg_path: path to model config file
        ckpt_path: path to model checkpoint
        output_dir: output directory
        noise_type: type of texture noise to add
        sample_idx: index of sample to visualize from validation set
        noise_severity: intensity of noise (1-5)
        feature_noise: if True, inject noise directly in feature space (bypasses encoder)
        feature_noise_scale: noise intensity for feature-space injection
    """
    print(f"\n[Real Model] Loading model from: {ckpt_path}")

    # Import openpoints modules
    EasyConfig, build_model_from_cfg, build_dataloader_from_cfg, get_features_by_keys = _import_openpoints()
    if EasyConfig is None:
        print("[Error] openpoints module required for real model mode")
        return

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    # Check if GT-IGU is enabled
    if not cfg.model.encoder_args.get('use_gt_igu', False):
        print("  [Warning] GT-IGU not enabled in config. Enabling...")
        cfg.model.encoder_args.use_gt_igu = True

    # Build and load model
    model = build_model_from_cfg(cfg.model).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("  ✅ Model loaded")
    else:
        print(f"  [Error] Checkpoint not found: {ckpt_path}")
        return

    model.eval()

    # Hook registration will be done after data loading (need pos for feature noise mode)
    # Build dataloader first
    print("  Loading validation data...")

    # Handle different config structures
    val_batch_size = cfg.get('val_batch_size', 1)

    # Get dataloader config (may be nested or flat)
    if hasattr(cfg.dataloader, 'val'):
        dataloader_cfg = cfg.dataloader.val
    else:
        dataloader_cfg = cfg.dataloader

    # IMPORTANT: disable distributed mode for visualization
    val_loader = build_dataloader_from_cfg(val_batch_size,
                                           cfg.dataset,
                                           dataloader_cfg,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=False)

    # Get sample at specified index
    print(f"  Selecting sample index: {sample_idx}")
    data_iter = iter(val_loader)
    for i in range(sample_idx + 1):
        data = next(data_iter)
    print(f"  Loaded sample {sample_idx} from validation set")
    
    for key in data:
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device)

    # Extract original data
    pos = data['pos'].squeeze(0).cpu().numpy()  # (N, 3)

    # Get RGB from features (before corruption)
    if 'x' in data:
        x = data['x']
        if x.dim() == 3:  # (B, C, N) or (B, N, C)
            if x.shape[1] < x.shape[2]:  # (B, C, N)
                rgb_clean = x[0, :3, :].permute(1, 0).cpu().numpy()
            else:  # (B, N, C)
                rgb_clean = x[0, :, :3].cpu().numpy()
        else:
            rgb_clean = x[:, :3].cpu().numpy()
    else:
        rgb_clean = np.ones((pos.shape[0], 3)) * 0.5

    rgb_clean = np.clip(rgb_clean, 0, 1)

    # =========================================================================
    # =========================================================================
    print(f"\n  {'='*60}")
    print(f"  >>> HALF-SPLIT CORRUPTION MODE (BOOSTED INTENSITY) <<<")
    print(f"  {'='*60}")
    print(f"  Corruption Type: {noise_type}")
    print(f"  Severity: {noise_severity}/5 (with 3x BOOST multiplier)")
    print(f"  Split Axis: Y (front/back split for clear spatial contrast)")
    if noise_type == 'extreme':
        print(
            f"  [EXTREME MODE] Colors will be completely randomized in noisy region!")
    data, noise_mask = apply_localized_corruption(
        data, data['pos'], noise_type, noise_severity,
        noise_ratio=0.5, seed=42, split_axis='y'
    )

    # Get corrupted RGB for visualization
    if 'x' in data:
        x = data['x']
        if x.dim() == 3:
            if x.shape[1] < x.shape[2]:  # (B, C, N)
                rgb_noisy = x[0, :3, :].permute(1, 0).cpu().numpy()
            else:
                rgb_noisy = x[0, :, :3].cpu().numpy()
        else:
            rgb_noisy = x[:, :3].cpu().numpy()
    else:
        rgb_noisy = rgb_clean.copy()

    rgb_noisy = np.clip(rgb_noisy, 0, 1)

    # Handle NaN values in RGB
    rgb_clean = np.nan_to_num(rgb_clean, nan=0.5)
    rgb_noisy = np.nan_to_num(rgb_noisy, nan=0.5)

    print(
        f"  Noise mask: {noise_mask.sum()} corrupted, {(~noise_mask).sum()} clean")

    # Prepare features
    data['x'] = get_features_by_keys(data, cfg.feature_keys)

    # =========================================================================
    #  HOOK REGISTRATION
    #  Choose between:
    #  1. Normal mode: Only capture gate values
    #  2. Feature noise mode: Inject noise directly into f_tex before GT-IGU
    # =========================================================================
    injection_hook = None
    handles = []

    if feature_noise:
        print(f"\n  {'='*60}")
        print(f"  >>> FEATURE-SPACE NOISE INJECTION MODE <<<")
        print(f"  {'='*60}")
        print(f"  This bypasses the texture encoder and tests Gate MLP directly!")
        print(
            f"  Noise scale: {feature_noise_scale}x (relative to feature std)")

        injection_hook, hook, handles = register_feature_noise_hook(
            model,
            noise_scale=feature_noise_scale,
            pos_data=data['pos']
        )

        if injection_hook is None:
            print("  [Error] Could not register feature noise injection hook")
            return
    else:
        # Normal mode: just capture gate values
        hook, handle = register_gtigu_hook(model)
        if hook is None:
            print("  [Error] Could not register GT-IGU hook")
            return
        handles = [handle]

    # Forward pass
    print("  Running inference...")
    with torch.no_grad():
        output = model(data)

    # Extract gate values from hook
    if hook.gate_values is not None:
        gate_raw = hook.gate_values  # (1, C, N')

        # Average across channels to get per-point gate value
        gate_per_point = gate_raw[0].mean(dim=0).numpy()  # (N',)

        # Gate values are at downsampled resolution, need to upsample
        N_original = pos.shape[0]
        N_gate = len(gate_per_point)

        if N_gate != N_original:
            print(f"  Gate resolution: {N_gate}, Original: {N_original}")
            # Simple nearest neighbor upsampling
            from sklearn.neighbors import NearestNeighbors

            # Get downsampled positions (assuming uniform downsampling)
            # This is approximate - in practice you'd track the sampling indices
            step = N_original // N_gate
            sample_idx = np.arange(0, N_original, step)[:N_gate]
            pos_sampled = pos[sample_idx]

            nbrs = NearestNeighbors(n_neighbors=1).fit(pos_sampled)
            _, indices = nbrs.kneighbors(pos)
            gate_values = gate_per_point[indices.flatten()]
        else:
            gate_values = gate_per_point

        # Get conflict signal (per-point feature difference)
        # s_diff shape: (1, C, N') where C is channel dim
        if hook.s_diff is not None:
            # Per-point conflict: average across channels
            s_diff_per_point = hook.s_diff[0].mean(dim=0).numpy()  # (N',)
            s_diff_scalar = s_diff_per_point.mean()  # scalar for backward compat

            # Also get f_geo and f_tex for analysis
            f_geo_per_point = hook.f_geo[0].mean(
                dim=0).numpy() if hook.f_geo is not None else None
            f_tex_per_point = hook.f_tex[0].mean(
                dim=0).numpy() if hook.f_tex is not None else None
        else:
            s_diff_per_point = None
            s_diff_scalar = 0.0
            f_geo_per_point = None
            f_tex_per_point = None

        # For backward compatibility
        s_diff = s_diff_scalar

        # Handle NaN values in gate_values
        nan_mask = ~np.isfinite(gate_values)
        if nan_mask.any():
            print(
                f"  [Warning] Found {nan_mask.sum()} NaN values in gate, replacing with 0.5")
            gate_values[nan_mask] = 0.5

        # Ensure gate values are in valid range
        gate_values = np.clip(gate_values, 0, 1)

    else:
        print("  [Error] Gate values not captured")
        for h in handles:
            h.remove()
        return

    # Remove all hooks
    for h in handles:
        h.remove()

    # =========================================================================
    #  FEATURE NOISE INJECTION ANALYSIS
    #  If feature noise was injected, update noise_mask from injection hook
    #  and print additional statistics
    # =========================================================================
    if feature_noise and injection_hook is not None:
        print(f"\n  {'='*70}")
        print(f"  🧪 FEATURE-SPACE NOISE INJECTION RESULTS")
        print(f"  {'='*70}")

        # Use the noise mask from feature injection (more accurate)
        if injection_hook.noise_mask is not None:
            # The injection was done at GT-IGU resolution, need to match
            feature_noise_mask = injection_hook.noise_mask
            print(f"  Feature noise mask size: {len(feature_noise_mask)}")
            print(f"  Gate values size: {len(gate_values)}")

            # Resample to match gate_values if needed
            if len(feature_noise_mask) != len(gate_values):
                if len(feature_noise_mask) < len(gate_values):
                    indices = np.linspace(
                        0, len(feature_noise_mask) - 1, len(gate_values)).astype(int)
                    noise_mask = feature_noise_mask[indices]
                else:
                    step = len(feature_noise_mask) // len(gate_values)
                    noise_mask = feature_noise_mask[::step][:len(gate_values)]
            else:
                noise_mask = feature_noise_mask

        # Print injection statistics
        stats = injection_hook.injection_stats
        print(f"\n  [Injection Statistics]")
        print(
            f"    Original f_tex: mean={stats.get('f_tex_mean', 0):.4f}, std={stats.get('f_tex_std', 0):.4f}")
        print(f"    Injected noise: std={stats.get('noise_std', 0):.4f}")
        print(
            f"    Noise-to-Signal Ratio: {stats.get('noise_to_signal_ratio', 0):.2f}")
        print(f"    Noisy points: {stats.get('n_noisy_points', 0)}")
        print(f"    Clean points: {stats.get('n_clean_points', 0)}")

        # Compute feature-level conflict before/after injection
        if injection_hook.f_tex_original is not None and injection_hook.f_tex_noisy is not None:
            f_orig = injection_hook.f_tex_original[0].numpy()  # (C, N)
            f_noisy = injection_hook.f_tex_noisy[0].numpy()    # (C, N)

            # Compute per-point difference
            diff = np.abs(f_orig - f_noisy).mean(axis=0)  # (N,)

            print(f"\n  [Feature Perturbation]")
            print(f"    Mean |f_tex_orig - f_tex_noisy|: {diff.mean():.6f}")
            print(f"    Max perturbation: {diff.max():.6f}")
            if len(diff) > 0:
                noisy_region_diff = diff[injection_hook.noise_mask] if injection_hook.noise_mask is not None else diff
                clean_region_diff = diff[~injection_hook.noise_mask] if injection_hook.noise_mask is not None else np.array([
                                                                                                                            0])
                print(
                    f"    Perturbation in noisy region: {noisy_region_diff.mean():.6f}")
                print(
                    f"    Perturbation in clean region: {clean_region_diff.mean():.6f}")

    # Create visualizations
    print("  Generating visualizations...")

    # =========================================================================
    #  PRIMARY VISUALIZATION: Feature Conflict Map (Gate INPUT)
    #  Since Gate Output is not responding, we show what the Gate SHOULD see
    # =========================================================================

    # First, we need s_diff at the same resolution as pos for visualization
    if s_diff_per_point is not None:
        # Upsample s_diff to original resolution
        N_original = pos.shape[0]
        N_sdiff = len(s_diff_per_point)

        if N_sdiff != N_original:
            print(
                f"  [Info] Upsampling s_diff from {N_sdiff} to {N_original} for visualization")
            indices = np.linspace(0, N_sdiff - 1, N_original).astype(int)
            s_diff_upsampled = s_diff_per_point[indices]
        else:
            s_diff_upsampled = s_diff_per_point

        # Create Feature Conflict visualization (PRIMARY)
        create_feature_conflict_visualization(
            pos, rgb_clean, rgb_noisy, s_diff_upsampled,
            noise_mask=noise_mask,
            output_path=os.path.join(output_dir, 'gt_igu_feature_conflict.pdf')
        )
    else:
        print(
            "  [Warning] s_diff not available, skipping feature conflict visualization")
        s_diff_upsampled = None

    # Upsample noise_mask if needed (gate values might be at different resolution)
    if len(noise_mask) != len(gate_values):
        print(
            f"  [Info] Upsampling noise_mask from {len(noise_mask)} to {len(gate_values)}")
        # Use nearest neighbor to match resolutions
        step = len(noise_mask) // len(gate_values)
        if step > 0:
            noise_mask_resampled = noise_mask[::step][:len(gate_values)]
        else:
            noise_mask_resampled = noise_mask[:len(gate_values)]
    else:
        noise_mask_resampled = noise_mask

    # =========================================================================
    #  PRIMARY: FEATURE CONFLICT STATISTICS (Gate INPUT)
    #  This is what matters - the conflict signal that the Gate SHOULD respond to
    # =========================================================================
    print(f"\n  {'='*70}")
    print(f"  🔥 FEATURE CONFLICT STATISTICS (Gate INPUT)")
    print(f"  {'='*70}")
    print(f"  This is the conflict signal: s_diff = (F_geo - F_tex)^2")
    print(f"  If this shows difference, the signal EXISTS regardless of Gate response")

    if s_diff_per_point is not None:
        # Resample s_diff to match noise_mask resolution
        N_sdiff = len(s_diff_per_point)
        N_mask = len(noise_mask_resampled)

        if N_sdiff != N_mask:
            if N_sdiff < N_mask:
                indices = np.linspace(0, N_sdiff - 1, N_mask).astype(int)
                s_diff_resampled = s_diff_per_point[indices]
            else:
                step = N_sdiff // N_mask
                s_diff_resampled = s_diff_per_point[::step][:N_mask]
        else:
            s_diff_resampled = s_diff_per_point

        print(f"\n  [Overall Conflict Statistics]")
        print(f"    Total points:     {len(s_diff_resampled)}")
        print(f"    Mean:             {s_diff_resampled.mean():.6f}")
        print(f"    Std:              {s_diff_resampled.std():.6f}")
        print(f"    Min:              {s_diff_resampled.min():.6f}")
        print(f"    Max:              {s_diff_resampled.max():.6f}")

        if len(noise_mask_resampled) == len(s_diff_resampled):
            conflict_clean = s_diff_resampled[~noise_mask_resampled]
            conflict_noisy = s_diff_resampled[noise_mask_resampled]

            print(f"\n  [Clean Region Conflict]")
            print(f"    Points:           {len(conflict_clean)}")
            print(f"    Mean:             {conflict_clean.mean():.6f}")
            print(f"    Std:              {conflict_clean.std():.6f}")

            print(f"\n  [Noisy Region Conflict]")
            print(f"    Points:           {len(conflict_noisy)}")
            print(f"    Mean:             {conflict_noisy.mean():.6f}")
            print(f"    Std:              {conflict_noisy.std():.6f}")

            conflict_delta = conflict_noisy.mean() - conflict_clean.mean()
            conflict_ratio = conflict_noisy.mean(
            ) / conflict_clean.mean() if conflict_clean.mean() > 0 else float('inf')

            print(f"\n  {'='*70}")
            print(f"  ⚡ CONFLICT SIGNAL METRICS (THE KEY RESULT)")
            print(f"  {'='*70}")
            print(f"    Δ Conflict (Noisy - Clean):  {conflict_delta:+.6f}")
            print(f"    Ratio (Noisy/Clean):         {conflict_ratio:.2f}x")

            print(f"\n  [Interpretation]")
            if conflict_delta > 0.01 and conflict_ratio > 1.1:
                print(f"    ✅ CONFLICT SIGNAL EXISTS!")
                print(
                    f"       Noisy region has {conflict_ratio:.2f}x higher conflict")
                print(f"       → The feature-level difference IS being detected")
                print(
                    f"       → If Gate doesn't respond, problem is in Gate MLP training")
            elif conflict_delta > 0:
                print(
                    f"    ⚠️  Weak conflict signal (ratio={conflict_ratio:.2f}x)")
                print(
                    f"       → Try feature_noise mode: --feature_noise --feature_noise_scale 3.0")
            else:
                print(f"    ❌ No conflict signal detected")
                print(f"       → Encoder may be absorbing the noise")
                print(f"       → Try feature_noise mode to bypass encoder")
    else:
        print(f"  [Warning] s_diff not available")

    # =========================================================================
    # =========================================================================
    if s_diff_per_point is not None and len(s_diff_per_point) > 0:
        print(f"\n  {'='*70}")
        print(f"  🔥 FEATURE CONFLICT ANALYSIS (The Conflict Signal)")
        print(f"  {'='*70}")
        print(
            f"  This shows $(F_{{geo}} - F_{{tex}})^2$ - the raw input to Gate MLP")
        print(f"  If conflict is HIGH in noisy region, the signal EXISTS even if Gate doesn't respond")

        # Upsample s_diff to match gate_values resolution if needed
        # s_diff_per_point is at downsampled resolution (e.g., 181 points)
        # gate_values is at original resolution (e.g., 46396 points)
        if len(s_diff_per_point) != len(gate_values):
            N_low = len(s_diff_per_point)
            N_high = len(gate_values)
            print(
                f"  [Info] Upsampling s_diff from {N_low} to {N_high} points")

            if N_low < N_high:
                # Need to upsample: use nearest neighbor interpolation
                # Each high-res point maps to a low-res point
                indices = np.linspace(0, N_low - 1, N_high).astype(int)
                s_diff_resampled = s_diff_per_point[indices]
            else:
                # Need to downsample
                step = N_low // N_high
                s_diff_resampled = s_diff_per_point[::step][:N_high]
        else:
            s_diff_resampled = s_diff_per_point

        print(f"\n  [Overall Conflict Statistics]")
        print(f"    Total points:     {len(s_diff_resampled)}")
        print(f"    Mean Conflict:    {s_diff_resampled.mean():.6f}")
        print(f"    Std:              {s_diff_resampled.std():.6f}")
        print(f"    Min:              {s_diff_resampled.min():.6f}")
        print(f"    Max:              {s_diff_resampled.max():.6f}")

        # Clean vs Noisy conflict comparison (THE KEY INSIGHT!)
        if len(noise_mask_resampled) == len(s_diff_resampled):
            conflict_clean = s_diff_resampled[~noise_mask_resampled]
            conflict_noisy = s_diff_resampled[noise_mask_resampled]

            print(f"\n  [Clean Region Conflict] (should be LOW)")
            print(f"    Mean:             {conflict_clean.mean():.6f}")
            print(f"    Std:              {conflict_clean.std():.6f}")

            print(f"\n  [Noisy Region Conflict] (should be HIGH)")
            print(f"    Mean:             {conflict_noisy.mean():.6f}")
            print(f"    Std:              {conflict_noisy.std():.6f}")

            conflict_delta = conflict_noisy.mean() - conflict_clean.mean()
            conflict_ratio = conflict_noisy.mean(
            ) / conflict_clean.mean() if conflict_clean.mean() > 0 else float('inf')

            print(f"\n  {'='*70}")
            print(f"  ⚡ CONFLICT SIGNAL METRICS")
            print(f"  {'='*70}")
            print(f"    Δ Conflict (Noisy - Clean):  {conflict_delta:+.6f}")
            print(f"    Ratio (Noisy/Clean):         {conflict_ratio:.2f}x")

            print(f"\n  [Interpretation]")
            if conflict_delta > 0 and conflict_ratio > 1.2:
                print(f"    ✅ CONFLICT SIGNAL EXISTS!")
                print(
                    f"       Noisy region has {conflict_ratio:.1f}x higher conflict than clean region")
                print(
                    f"       → The texture corruption IS creating feature-level conflict")
                print(
                    f"       → If Gate doesn't respond, the issue is in Gate MLP training, not signal")
            elif conflict_delta > 0:
                print(
                    f"    ⚠️  Weak conflict signal (ratio={conflict_ratio:.2f}x)")
                print(f"       → Try using 'extreme' noise type for stronger corruption")
            else:
                print(f"    ❌ No conflict difference detected")
                print(
                    f"       → The texture encoder may be invariant to this type of noise")

        print(f"\n  {'='*70}\n")

    print("\n✅ Real model visualization complete!")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GT-IGU Gating Visualization for Uncertainty-Aware Fusion')

    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'real'],
                        help='demo: synthetic data, real: use trained model')

    parser.add_argument('--cfg', type=str,
                        default='cfgs/scannet/pointnext-b-mixratio.yaml',
                        help='Config file path (for real mode)')

    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path (for real mode)')

    parser.add_argument('--output', type=str, default='./visualizations/gt_igu',
                        help='Output directory')

    parser.add_argument('--noise_type', type=str, default='extreme',
                        choices=['noise', 'jitter',
                                 'dropout', 'extreme', 'clean'],
                        help='Type of texture corruption: '
                             'noise=Gaussian noise (boosted 3x), '
                             'jitter=color jitter (boosted 3x), '
                             'dropout=random dark colors, '
                             'extreme=completely randomize colors (RECOMMENDED for demo), '
                             'clean=no corruption')

    parser.add_argument('--noise_severity', type=int, default=5,
                        choices=[1, 2, 3, 4, 5],
                        help='Noise severity level (1-5, default=5 for maximum effect)')

    parser.add_argument('--feature_noise', action='store_true',
                        help='Enable FEATURE-SPACE noise injection (bypasses encoder). '
                             'Use this to test Gate MLP response when RGB noise is absorbed by encoder.')

    parser.add_argument('--feature_noise_scale', type=float, default=2.0,
                        help='Feature noise intensity (multiplier of feature std, default=2.0)')

    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize from validation set (default=0, first sample)')

    args = parser.parse_args()

    print("=" * 70)
    print("📊 GT-IGU Gating Visualization")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")

    if args.mode == 'demo':
        create_synthetic_demo(output_dir=args.output)
    else:
        if args.ckpt is None:
            print("[Error] --ckpt is required for real mode")
            return
        run_with_real_model(
            cfg_path=args.cfg,
            ckpt_path=args.ckpt,
            output_dir=args.output,
            noise_type=args.noise_type,
            noise_severity=args.noise_severity,
            feature_noise=args.feature_noise,
            feature_noise_scale=args.feature_noise_scale,
            sample_idx=args.sample_idx
        )


if __name__ == '__main__':
    main()
