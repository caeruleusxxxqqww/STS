#!/usr/bin/env python3
"""



Usage:
    python scripts/visualize_feature_manifold.py \
        --cfg cfgs/s3dis/pointnext-s.yaml \
        --pretrained_path_baseline log/s3dis/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/s3dis/ours_sts/checkpoint/best.pth \
        --dataset s3dis \
        --gpu 3 \
        --max_samples 50 \
        --output_dir figures/feature_manifold

    python scripts/visualize_feature_manifold.py \
        --cfg cfgs/scannet/pointnext-s.yaml \
        --pretrained_path_baseline log/scannet/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/scannet/ours_sts/checkpoint/best.pth \
        --dataset scannet \
        --gpu 3 \
        --max_samples 50 \
        --output_dir figures/feature_manifold

Author: Auto-generated for STS visualization
"""
# fmt: off
# isort: skip_file
# ============================================================================
# ============================================================================
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ============================================================================

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[Warning] seaborn not found, using default matplotlib style")

from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.loss.custom_innovations import SemanticConflictGenerator

# fmt: on

if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def extract_features(model, dataloader, device, model_name, max_samples=None, generator=None, feature_keys=None, class_names=None):
    """

    Args:

    Returns:
    """
    was_training = model.training
    model.train()
    model.to(device)

    features_list = []
    labels_list = []
    texture_info_list = []

    print(f"\n[Extract Features] Extracting from {model_name}...")
    print(
        f"  [Note] Temporarily using training mode to extract features (not logits)")

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Extracting {model_name}")):
            if max_samples and batch_idx >= max_samples:
                break

            if generator is not None:
                data, attack_info = generator.generate_conflict_sample(
                    data, return_attack_info=True)
                if not attack_info['success']:
                    continue
            else:
                attack_info = None

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            if feature_keys is not None:
                data['x'] = get_features_by_keys(data, feature_keys)

            try:
                output = model(data)

                if isinstance(output, dict):
                    features = output.get('features', output.get('feat', None))

                    if features is None:
                        for key in ['encoder_features', 'backbone_features', 'feat_out', 'x']:
                            if key in output:
                                features = output[key]
                                if batch_idx == 0:
                                    print(
                                        f"  [Info] Using '{key}' as features (shape: {features.shape})")
                                break

                    if features is None:
                        logits = output.get('logits', output.get('seg_logits'))
                        if logits is not None:
                            if batch_idx == 0:
                                num_classes = len(
                                    class_names) if class_names else logits.shape[-1]
                                print(f"  [Info] Using logits as features (shape: {logits.shape}). "
                                      f"Feature dimension = num_classes ({num_classes}), which is acceptable for visualization.")
                            features = F.normalize(logits, p=2, dim=-1)
                else:
                    features = output

                if features is None:
                    print(
                        f"  [Warning] No features found in output for batch {batch_idx}")
                    continue

                feat_dim = features.shape[-1]
                if batch_idx == 0:
                    num_classes = len(
                        class_names) if class_names else 'unknown'
                    print(
                        f"  [Info] Feature dimension: {feat_dim} (dataset has {num_classes} classes)")
                    if class_names and feat_dim == len(class_names):
                        print(f"  [Note] Feature dimension ({feat_dim}) equals num_classes ({len(class_names)}). "
                              f"Using logits as features (normalized). This is acceptable for visualization.")
                    elif class_names and feat_dim < len(class_names):
                        print(f"  [Warning] Feature dimension ({feat_dim}) < num_classes ({len(class_names)}). "
                              f"This is unusual. Please check feature extraction.")

                if features.dim() == 3:
                    if features.shape[1] < features.shape[2]:  # (B, C, N)
                        features = features.transpose(
                            1, 2).contiguous()  # (B, N, C)
                    # (N, C)
                    features = features.reshape(-1, features.shape[-1])

                labels = data.get('y', data.get('label'))
                if labels is not None:
                    if labels.dim() > 1:
                        labels = labels.reshape(-1)
                    labels = labels.cpu().numpy()
                else:
                    continue

                features = features.cpu().numpy()

                texture_info = {
                    'attack_applied': attack_info is not None and attack_info.get('success', False),
                    'target_class': attack_info['target_class'][0] if attack_info and attack_info.get('success') else None,
                    'source_class': attack_info['source_class'][0] if attack_info and attack_info.get('success') else None,
                    'attack_mask': attack_info['attack_mask'] if attack_info and attack_info.get('success') else None,
                }

                features_list.append(features)
                labels_list.append(labels)
                texture_info_list.append(texture_info)

            except Exception as e:
                print(f"  [Warning] Forward failed for batch {batch_idx}: {e}")
                continue

    print(f"  Extracted {len(features_list)} batches, "
          f"total {sum(f.shape[0] for f in features_list)} points")

    return features_list, labels_list, texture_info_list


def filter_classes(features, labels, texture_labels, class_names, selected_classes=None):
    """

    Args:

    Returns:
    """
    if selected_classes is None or len(selected_classes) == 0:
        return features, labels, texture_labels, np.unique(labels)

    selected_class_ids = []
    for cls in selected_classes:
        if isinstance(cls, str):
            try:
                cls_id = class_names.index(cls)
                selected_class_ids.append(cls_id)
            except ValueError:
                print(
                    f"  [Warning] Class '{cls}' not found in class_names, skipping")
        elif isinstance(cls, int):
            selected_class_ids.append(cls)
        else:
            print(f"  [Warning] Invalid class specifier: {cls}, skipping")

    if len(selected_class_ids) == 0:
        print("  [Warning] No valid classes selected, using all classes")
        return features, labels, texture_labels, np.unique(labels)

    selected_class_ids = np.array(selected_class_ids)

    mask = np.isin(labels, selected_class_ids)

    if mask.sum() == 0:
        print("  [Warning] No points found for selected classes, using all classes")
        return features, labels, texture_labels, np.unique(labels)

    print(
        f"  [Filter] Selected {len(selected_class_ids)} classes: {[class_names[i] for i in selected_class_ids]}")
    print(f"  [Filter] Points before: {len(labels)}, after: {mask.sum()}")

    return features[mask], labels[mask], texture_labels[mask], selected_class_ids


def prepare_data_for_visualization(features_list, labels_list, texture_info_list,
                                   max_points_per_class=5000, subsample=True):
    """

    Args:

    Returns:
    """
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    texture_labels = np.zeros(len(labels), dtype=int)

    attack_mask_global = np.zeros(len(labels), dtype=bool)
    offset = 0
    for idx, texture_info in enumerate(texture_info_list):
        n_points = len(features_list[idx])
        if texture_info.get('attack_applied', False) and texture_info.get('attack_mask') is not None:
            mask = texture_info['attack_mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)

            if mask.dtype != bool:
                mask = mask.astype(bool)

            if len(mask) == n_points:
                end_idx = offset + n_points
                attack_mask_global[offset:end_idx] = mask
                texture_labels[offset:end_idx][mask] = 1
            else:
                min_len = min(len(mask), n_points)
                if min_len > 0:
                    end_idx = offset + min_len
                    attack_mask_global[offset:end_idx] = mask[:min_len]
                    texture_labels[offset:end_idx][mask[:min_len]] = 1
        offset += n_points

    if subsample and len(features) > max_points_per_class * len(np.unique(labels)):
        print(f"\n[Subsample] Original: {len(features)} points")
        indices = []
        for cls_id in np.unique(labels):
            cls_mask = labels == cls_id
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) > max_points_per_class:
                selected = np.random.choice(
                    cls_indices, max_points_per_class, replace=False)
            else:
                selected = cls_indices
            indices.append(selected)
        indices = np.concatenate(indices)
        features = features[indices]
        labels = labels[indices]
        texture_labels = texture_labels[indices]
        print(f"  After subsample: {len(features)} points")

    return features, labels, texture_labels


def visualize_tsne(features, labels, texture_labels, class_names, model_name,
                   output_path, perplexity=30, n_iter=1000, random_state=42):
    """

    Args:
    """
    print(f"\n[t-SNE] Computing t-SNE embedding for {model_name}...")
    print(f"  Input shape: {features.shape}")

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    print(f"  [Normalization] Features normalized (mean=0, std=1)")

    if features_normalized.shape[1] > 50:
        print(
            f"  [PCA] Reducing from {features_normalized.shape[1]} to 50 dimensions...")
        pca = PCA(n_components=50, random_state=random_state)
        features_normalized = pca.fit_transform(features_normalized)
        print(
            f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    adjusted_perplexity = min(perplexity, len(
        features_normalized) - 1)
    adjusted_perplexity = max(
        30, min(adjusted_perplexity, 50))

    print(f"  [t-SNE] Using perplexity={adjusted_perplexity}, n_iter={n_iter}, "
          f"early_exaggeration=20, learning_rate=100")

    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=n_iter,
                early_exaggeration=20, learning_rate=100,
                random_state=random_state, verbose=1, min_grad_norm=1e-7)
    features_2d = tsne.fit_transform(features_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, cls_id in enumerate(unique_labels):
        mask = labels == cls_id
        if mask.sum() > 0:
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[colors[i]], label=class_names[int(cls_id)] if cls_id < len(
                            class_names) else f'Class {int(cls_id)}',
                        alpha=0.6, s=10)

    ax1.set_title(f'{model_name}: Feature Distribution by Semantic Class',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    # ax1.grid(True, alpha=0.3)

    ax2 = axes[1]

    mask_original = texture_labels == 0
    if mask_original.sum() > 0:
        ax2.scatter(features_2d[mask_original, 0], features_2d[mask_original, 1],
                    c='blue', label='Original Texture (Blue)',
                    alpha=0.15, s=12, zorder=1)

    mask_attacked = texture_labels == 1
    if mask_attacked.sum() > 0:
        ax2.scatter(features_2d[mask_attacked, 0], features_2d[mask_attacked, 1],
                    c='red', label='Attacked Texture (Red)',
                    alpha=0.6, s=18, zorder=2, edgecolors='darkred', linewidths=0.5)
    else:
        ax2.text(0.5, 0.95, 'No attacked samples in this view',
                 transform=ax2.transAxes, ha='center', va='top',
                 fontsize=9, style='italic', bbox=dict(boxstyle='round',
                                                       facecolor='wheat', alpha=0.5))

    ax2.set_title(f'{model_name}: Feature Distribution by Texture Type',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    legend = ax2.legend(fontsize=10, framealpha=0.9, loc='best')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"  Saved to {output_path}")
    plt.close()


def visualize_comparison(features_baseline, labels_baseline, texture_labels_baseline,
                         features_ours, labels_ours, texture_labels_ours,
                         class_names, output_path, perplexity=30, n_iter=1000, random_state=42):
    """

    Args:
    """
    print(f"\n[t-SNE Comparison] Computing joint t-SNE embedding...")

    features_combined = np.vstack([features_baseline, features_ours])

    scaler = StandardScaler()
    features_combined_normalized = scaler.fit_transform(features_combined)
    print(f"  [Normalization] Features normalized (mean=0, std=1)")

    if features_combined_normalized.shape[1] > 50:
        print(
            f"  [PCA] Reducing from {features_combined_normalized.shape[1]} to 50 dimensions...")
        pca = PCA(n_components=50, random_state=random_state)
        features_combined_normalized = pca.fit_transform(
            features_combined_normalized)

    adjusted_perplexity = min(perplexity, len(
        features_combined_normalized) - 1)
    adjusted_perplexity = max(30, min(adjusted_perplexity, 50))

    print(f"  [t-SNE] Using perplexity={adjusted_perplexity}, n_iter={n_iter}, "
          f"early_exaggeration=20, learning_rate=100")

    tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, n_iter=n_iter,
                early_exaggeration=20, learning_rate=100,
                random_state=random_state, verbose=1, min_grad_norm=1e-7)
    features_2d_combined = tsne.fit_transform(features_combined_normalized)

    n_baseline = len(features_baseline)
    features_2d_baseline = features_2d_combined[:n_baseline]
    features_2d_ours = features_2d_combined[n_baseline:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax1 = axes[0, 0]
    unique_labels = np.unique(labels_baseline)
    if len(unique_labels) <= 4:
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, cls_id in enumerate(unique_labels):
        mask = labels_baseline == cls_id
        if mask.sum() > 0:
            ax1.scatter(features_2d_baseline[mask, 0], features_2d_baseline[mask, 1],
                        c=[colors[i]], label=class_names[int(cls_id)] if cls_id < len(
                            class_names) else f'Class {int(cls_id)}',
                        alpha=0.7, s=15)

    ax1.set_title('Baseline: Feature Distribution by Semantic Class',
                  fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    # ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    mask_original = texture_labels_baseline == 0
    mask_attacked = texture_labels_baseline == 1
    if mask_original.sum() > 0:
        ax2.scatter(features_2d_baseline[mask_original, 0], features_2d_baseline[mask_original, 1],
                    c='blue', label='Original Texture (Blue)',
                    alpha=0.15, s=12, zorder=1)
    if mask_attacked.sum() > 0:
        ax2.scatter(features_2d_baseline[mask_attacked, 0], features_2d_baseline[mask_attacked, 1],
                    c='red', label='Attacked Texture (Red)',
                    alpha=0.6, s=18, zorder=2,
                    edgecolors='darkred', linewidths=0.5)
    ax2.set_title('Baseline: Feature Distribution by Texture Type',
                  fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    legend2 = ax2.legend(fontsize=9, framealpha=0.9, loc='best')
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor('gray')
    # ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    unique_labels = np.unique(labels_ours)
    if len(unique_labels) <= 4:
        colors_ours = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    else:
        colors_ours = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, cls_id in enumerate(unique_labels):
        mask = labels_ours == cls_id
        if mask.sum() > 0:
            ax3.scatter(features_2d_ours[mask, 0], features_2d_ours[mask, 1],
                        c=[colors_ours[i]], label=class_names[int(cls_id)] if cls_id < len(
                            class_names) else f'Class {int(cls_id)}',
                        alpha=0.7, s=15)

    ax3.set_title('Ours (STS): Feature Distribution by Semantic Class',
                  fontsize=11, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    # ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    mask_original = texture_labels_ours == 0
    mask_attacked = texture_labels_ours == 1
    if mask_original.sum() > 0:
        ax4.scatter(features_2d_ours[mask_original, 0], features_2d_ours[mask_original, 1],
                    c='blue', label='Original Texture (Blue)',
                    alpha=0.15, s=12, zorder=1)
    if mask_attacked.sum() > 0:
        ax4.scatter(features_2d_ours[mask_attacked, 0], features_2d_ours[mask_attacked, 1],
                    c='red', label='Attacked Texture (Red)',
                    alpha=0.6, s=18, zorder=2,
                    edgecolors='darkred', linewidths=0.5)
    ax4.set_title('Ours (STS): Feature Distribution by Texture Type',
                  fontsize=11, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.axis('off')
    legend4 = ax4.legend(fontsize=9, framealpha=0.9, loc='best')
    legend4.get_frame().set_facecolor('white')
    legend4.get_frame().set_edgecolor('gray')
    # ax4.grid(True, alpha=0.3)

    plt.suptitle('Feature Manifold Comparison: Baseline vs Ours (STS)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"  Saved to {output_path}")
    plt.close()


def compute_intra_class_compactness(features, labels, class_names):
    """

    Args:

    Returns:
        dict: {class_name: compactness_score}
    """
    compactness_scores = {}

    for cls_id in np.unique(labels):
        mask = labels == cls_id
        if mask.sum() < 2:
            continue

        cls_features = features[mask]

        centroid = cls_features.mean(axis=0)
        distances = np.linalg.norm(cls_features - centroid, axis=1)
        avg_distance = distances.mean()

        cls_name = class_names[int(cls_id)] if cls_id < len(
            class_names) else f'Class {int(cls_id)}'
        compactness_scores[cls_name] = avg_distance

    return compactness_scores


def compute_texture_robustness_metrics(features, labels, texture_labels, class_names):
    """


    Args:

    Returns:
    """
    metrics = {}

    texture_invariance = {}
    for cls_id in np.unique(labels):
        cls_mask = labels == cls_id
        if cls_mask.sum() < 2:
            continue

        original_mask = cls_mask & (texture_labels == 0)
        attacked_mask = cls_mask & (texture_labels == 1)

        if original_mask.sum() == 0 or attacked_mask.sum() == 0:
            continue

        original_features = features[original_mask]
        attacked_features = features[attacked_mask]

        original_centroid = original_features.mean(axis=0)
        distances = np.linalg.norm(
            attacked_features - original_centroid, axis=1)
        avg_distance = distances.mean()

        cls_name = class_names[int(cls_id)] if cls_id < len(
            class_names) else f'Class {int(cls_id)}'
        texture_invariance[cls_name] = avg_distance

    metrics['texture_invariance'] = texture_invariance

    unique_classes = np.unique(labels)
    if len(unique_classes) > 1:
        class_centroids = {}
        for cls_id in unique_classes:
            cls_mask = labels == cls_id
            if cls_mask.sum() > 0:
                class_centroids[cls_id] = features[cls_mask].mean(axis=0)

        distances = []
        class_ids = list(class_centroids.keys())
        for i in range(len(class_ids)):
            for j in range(i + 1, len(class_ids)):
                dist = np.linalg.norm(
                    class_centroids[class_ids[i]] - class_centroids[class_ids[j]])
                distances.append(dist)

        metrics['inter_class_separation'] = np.mean(
            distances) if distances else 0.0
    else:
        metrics['inter_class_separation'] = 0.0

    texture_agnostic_compactness = {}
    for cls_id in np.unique(labels):
        cls_mask = labels == cls_id
        if cls_mask.sum() < 2:
            continue

        cls_features = features[cls_mask]
        centroid = cls_features.mean(axis=0)
        distances = np.linalg.norm(cls_features - centroid, axis=1)
        avg_distance = distances.mean()

        cls_name = class_names[int(cls_id)] if cls_id < len(
            class_names) else f'Class {int(cls_id)}'
        texture_agnostic_compactness[cls_name] = avg_distance

    metrics['texture_agnostic_compactness'] = texture_agnostic_compactness

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Feature Manifold & Orthogonality Visualization')

    # Config
    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file path')

    # Model paths
    parser.add_argument('--pretrained_path_baseline', type=str, default=None,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--pretrained_path_ours', type=str, default=None,
                        help='Path to STS-trained model checkpoint')

    # GPU settings
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU ID to use')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='s3dis',
                        choices=['s3dis', 'scannet'],
                        help='Dataset name')

    # Visualization settings
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to process')
    parser.add_argument('--max_points_per_class', type=int, default=5000,
                        help='Maximum points per class for subsampling')
    parser.add_argument('--use_conflict', action='store_true', default=True,
                        help='Use Semantic Conflict augmentation (default: True, required for meaningful comparison with semantic conflict test results)')
    parser.add_argument('--perplexity', type=int, default=40,
                        help='t-SNE perplexity parameter (default: 40, higher values promote tighter clusters)')
    parser.add_argument('--n_iter', type=int, default=2000,
                        help='t-SNE number of iterations (default: 2000, more iterations for better convergence)')
    parser.add_argument('--focus_classes', type=str, nargs='+', default=None,
                        help='Focus on specific classes for visualization (e.g., "table floor chair wall"). '
                             'If not specified and --use_focus is set, uses recommended classes for the dataset.')
    parser.add_argument('--use_focus', action='store_true', default=True,
                        help='Use focused visualization (recommended classes) instead of all classes. '
                             'Default: True. Recommended classes: 7-8 most texture-confusable classes '
                             '(S3DIS: table, floor, chair, wall, ceiling, door, bookcase, sofa; '
                             'ScanNet: table, floor, chair, wall, sofa, desk, cabinet, counter).')
    parser.add_argument('--all_classes', action='store_true',
                        help='Visualize all classes (overrides --use_focus)')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='figures/feature_manifold',
                        help='Output directory for figures')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] Using {device}")

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.seed = 0

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'scannet':
        from openpoints.loss.custom_innovations import SemanticConflictGenerator
        class_names = SemanticConflictGenerator.SCANNET_CLASSES
        recommended_focus_classes = [
            'bathtub', 'bed', 'chair', 'curtain', 'sofa']
    else:  # s3dis
        from openpoints.loss.custom_innovations import SemanticConflictGenerator
        class_names = SemanticConflictGenerator.S3DIS_CLASSES

        recommended_focus_classes = [
            'window', 'floor', 'door', 'column', 'sofa']

    invalid_classes = [
        cls for cls in recommended_focus_classes if cls not in class_names]
    if invalid_classes:
        print(
            f"  [Warning] Some recommended focus classes are not in class_names: {invalid_classes}")
        recommended_focus_classes = [
            cls for cls in recommended_focus_classes if cls in class_names]
        print(f"  [Info] Validated focus classes: {recommended_focus_classes}")

    if args.all_classes:
        focus_classes = None
        print(f"\n[Visualization Mode] Showing ALL classes")
    elif args.focus_classes is not None:
        focus_classes = args.focus_classes
        print(f"\n[Focus Mode] Using user-specified classes: {focus_classes}")
    elif args.use_focus:
        focus_classes = recommended_focus_classes
        print(f"\n{'='*70}")
        print(
            f"[Focus Mode] Using recommended focus classes ({len(recommended_focus_classes)} classes): {recommended_focus_classes}")
        print(f"{'='*70}")
        print(
            f"  Rationale: These {len(recommended_focus_classes)} classes are most texture-confusable:")
        if args.dataset == 'scannet':
            print(
                f"    - Selected classes: {', '.join(recommended_focus_classes)}")
        else:  # s3dis
            print(f"    - Furniture (wood/fabric texture): chair, sofa")
            print(
                f"    - Structure (brick/stone/wood texture): wall, floor, ceiling, door, window, column")
        print(f"  Expected Results:")
        print(f"    - Baseline: These classes will MIX together (texture bias)")
        print(f"    - Ours (STS): These classes will SEPARATE clearly (geometry-focused)")
        print(f"{'='*70}")
    else:
        focus_classes = None
        print(f"\n[Visualization Mode] Showing ALL classes")

    print(f"\n[DataLoader] Building dataloader for {args.dataset}...")

    if hasattr(cfg.dataloader, 'val'):
        dataloader_cfg = cfg.dataloader.val
    else:
        dataloader_cfg = cfg.dataloader

    val_batch_size = getattr(cfg, 'val_batch_size', None)
    if val_batch_size is None:
        val_batch_size = getattr(dataloader_cfg, 'batch_size', 1)
    if val_batch_size is None:
        val_batch_size = 1

    val_loader = build_dataloader_from_cfg(
        val_batch_size,
        cfg.dataset,
        dataloader_cfg,
        datatransforms_cfg=cfg.datatransforms,
        split='val',
        distributed=False
    )

    generator = None
    if args.use_conflict:
        generator = SemanticConflictGenerator(dataset=args.dataset)
        print(f"  [Semantic Conflict] Enabled")
        print(
            f"  [Note] Attack texture sources: {list(generator.source_classes.values())}")
        print(
            f"  [Important] Using semantic conflict attack data for visualization.")
        print(f"              This aligns with the Semantic Conflict Test results.")
    else:
        print(f"  [Warning] Semantic Conflict is DISABLED!")
        print(f"            Visualization uses original data (no texture attack).")
        print(f"            This may NOT align with Semantic Conflict Test results.")
        print(f"            Consider using --use_conflict to enable texture attack.")

    features_baseline_list = None
    labels_baseline_list = None
    texture_info_baseline_list = None

    if args.pretrained_path_baseline:
        print(f"\n{'='*70}")
        print(f"Extracting Baseline Features")
        print(f"{'='*70}")

        model_baseline = build_model_from_cfg(cfg.model).to(device)
        load_checkpoint(
            model_baseline, pretrained_path=args.pretrained_path_baseline)

        features_baseline_list, labels_baseline_list, texture_info_baseline_list = extract_features(
            model_baseline, val_loader, device, 'Baseline',
            max_samples=args.max_samples, generator=generator, feature_keys=cfg.feature_keys, class_names=class_names)

    features_ours_list = None
    labels_ours_list = None
    texture_info_ours_list = None

    if args.pretrained_path_ours:
        print(f"\n{'='*70}")
        print(f"Extracting Ours (STS) Features")
        print(f"{'='*70}")

        model_ours = build_model_from_cfg(cfg.model).to(device)
        load_checkpoint(model_ours, pretrained_path=args.pretrained_path_ours)

        features_ours_list, labels_ours_list, texture_info_ours_list = extract_features(
            model_ours, val_loader, device, 'Ours (STS)',
            max_samples=args.max_samples, generator=generator, feature_keys=cfg.feature_keys, class_names=class_names)

    if features_baseline_list:
        features_baseline, labels_baseline, texture_labels_baseline = prepare_data_for_visualization(
            features_baseline_list, labels_baseline_list, texture_info_baseline_list,
            max_points_per_class=args.max_points_per_class)

    if features_ours_list:
        features_ours, labels_ours, texture_labels_ours = prepare_data_for_visualization(
            features_ours_list, labels_ours_list, texture_info_ours_list,
            max_points_per_class=args.max_points_per_class)

    if focus_classes is not None:
        print(f"\n{'='*70}")
        print(f"Filtering Classes for Focused Visualization")
        print(f"{'='*70}")

        if features_baseline_list:
            features_baseline, labels_baseline, texture_labels_baseline, _ = filter_classes(
                features_baseline, labels_baseline, texture_labels_baseline, class_names, focus_classes)

        if features_ours_list:
            features_ours, labels_ours, texture_labels_ours, _ = filter_classes(
                features_ours, labels_ours, texture_labels_ours, class_names, focus_classes)

    print(f"\n{'='*70}")
    print(f"Visualization")
    print(f"{'='*70}")

    if features_baseline_list:
        output_path = os.path.join(
            args.output_dir, f'baseline_tsne_{args.dataset}.png')
        visualize_tsne(features_baseline, labels_baseline, texture_labels_baseline,
                       class_names, 'Baseline', output_path,
                       perplexity=args.perplexity, n_iter=args.n_iter)

        compactness_baseline = compute_intra_class_compactness(
            features_baseline, labels_baseline, class_names)
        print(f"\n[Baseline] Intra-class Compactness:")
        for cls_name, score in sorted(compactness_baseline.items(), key=lambda x: x[1]):
            print(f"  {cls_name:<20}: {score:.4f}")

        if generator is not None:
            robustness_baseline = compute_texture_robustness_metrics(
                features_baseline, labels_baseline, texture_labels_baseline, class_names)
            print(f"\n[Baseline] Texture Robustness Metrics:")
            print(
                f"  Inter-class Separation: {robustness_baseline['inter_class_separation']:.4f} (larger is better)")
            if robustness_baseline['texture_invariance']:
                print(f"  Texture Invariance (attacked points distance to original):")
                for cls_name, score in sorted(robustness_baseline['texture_invariance'].items(), key=lambda x: x[1]):
                    print(
                        f"    {cls_name:<20}: {score:.4f} (smaller is better)")

    if features_ours_list:
        output_path = os.path.join(
            args.output_dir, f'ours_tsne_{args.dataset}.png')
        visualize_tsne(features_ours, labels_ours, texture_labels_ours,
                       class_names, 'Ours (STS)', output_path,
                       perplexity=args.perplexity, n_iter=args.n_iter)

        compactness_ours = compute_intra_class_compactness(
            features_ours, labels_ours, class_names)
        print(f"\n[Ours (STS)] Intra-class Compactness:")
        for cls_name, score in sorted(compactness_ours.items(), key=lambda x: x[1]):
            print(f"  {cls_name:<20}: {score:.4f}")

        if generator is not None:
            robustness_ours = compute_texture_robustness_metrics(
                features_ours, labels_ours, texture_labels_ours, class_names)
            print(f"\n[Ours (STS)] Texture Robustness Metrics:")
            print(
                f"  Inter-class Separation: {robustness_ours['inter_class_separation']:.4f} (larger is better)")
            if robustness_ours['texture_invariance']:
                print(f"  Texture Invariance (attacked points distance to original):")
                for cls_name, score in sorted(robustness_ours['texture_invariance'].items(), key=lambda x: x[1]):
                    print(
                        f"    {cls_name:<20}: {score:.4f} (smaller is better)")

    if features_baseline_list and features_ours_list:
        output_path = os.path.join(
            args.output_dir, f'comparison_tsne_{args.dataset}.png')
        visualize_comparison(features_baseline, labels_baseline, texture_labels_baseline,
                             features_ours, labels_ours, texture_labels_ours,
                             class_names, output_path,
                             perplexity=args.perplexity, n_iter=args.n_iter)

        print(f"\n[Comparison] Compactness Improvement:")
        common_classes = set(compactness_baseline.keys()
                             ) & set(compactness_ours.keys())
        for cls_name in sorted(common_classes):
            baseline_score = compactness_baseline[cls_name]
            ours_score = compactness_ours[cls_name]
            improvement = (baseline_score - ours_score) / baseline_score * 100
            print(f"  {cls_name:<20}: {baseline_score:.4f} -> {ours_score:.4f} "
                  f"({improvement:+.1f}%)")

        if generator is not None:
            robustness_baseline = compute_texture_robustness_metrics(
                features_baseline, labels_baseline, texture_labels_baseline, class_names)
            robustness_ours = compute_texture_robustness_metrics(
                features_ours, labels_ours, texture_labels_ours, class_names)

            print(f"\n[Comparison] Texture Robustness Improvement:")
            print(f"  Inter-class Separation: {robustness_baseline['inter_class_separation']:.4f} -> "
                  f"{robustness_ours['inter_class_separation']:.4f} "
                  f"({((robustness_ours['inter_class_separation'] - robustness_baseline['inter_class_separation']) / robustness_baseline['inter_class_separation'] * 100):+.1f}%)")

            common_ti_classes = set(robustness_baseline['texture_invariance'].keys()) & \
                set(robustness_ours['texture_invariance'].keys())
            if common_ti_classes:
                print(f"  Texture Invariance (attacked points stay close to original):")
                for cls_name in sorted(common_ti_classes):
                    baseline_ti = robustness_baseline['texture_invariance'][cls_name]
                    ours_ti = robustness_ours['texture_invariance'][cls_name]
                    improvement = (baseline_ti - ours_ti) / baseline_ti * 100
                    print(f"    {cls_name:<20}: {baseline_ti:.4f} -> {ours_ti:.4f} "
                          f"({improvement:+.1f}%, smaller is better)")
                print(
                    f"\n  [Note] Texture Invariance measures how well attacked points stay close to original points.")
                print(
                    f"         Lower values indicate better texture robustness (aligned with Semantic Conflict Test).")

    print(f"\n{'='*70}")
    print(f"✅ Visualization Complete!")
    print(f"   Output directory: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
