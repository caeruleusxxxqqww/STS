#!/usr/bin/env python
"""
    python scripts/cross_dataset_eval.py \
        --pretrained_path log/scannet/.../checkpoint/xxx_ckpt_best.pth \
        --cfg cfgs/scannet/pointnext-b.yaml
"""
import sys
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa

from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.dataset.data_util import voxelize

try:
    from torch_scatter import scatter
except ImportError:
    print("[WARNING] torch_scatter not found, using fallback implementation")
    scatter = None


# ============================================================================
# ============================================================================
#                  'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
#                  'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
#                  'bathtub', 'otherfurniture']
#
#                  'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']

# ScanNet class index -> S3DIS class index (-1 means no mapping)
SCANNET_TO_S3DIS = {
    0: 2,    # wall -> wall
    1: 1,    # floor -> floor
    2: -1,   # cabinet -> ignore
    3: -1,   # bed -> ignore
    4: 7,    # chair -> chair
    5: 10,   # sofa -> sofa
    6: 8,    # table -> table
    7: 6,    # door -> door
    8: 5,    # window -> window
    9: 9,    # bookshelf -> bookcase
    10: -1,  # picture -> ignore
    11: -1,  # counter -> ignore
    12: 8,
    13: -1,  # curtain -> ignore
    14: -1,  # refrigerator -> ignore
    15: -1,  # shower curtain -> ignore
    16: -1,  # toilet -> ignore
    17: -1,  # sink -> ignore
    18: -1,  # bathtub -> ignore
    19: 12,  # otherfurniture -> clutter
}

# S3DIS class names for display
S3DIS_CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                 'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter']


def scatter_mean_fallback(src, index, dim=0, dim_size=None):
    """Fallback implementation of scatter mean without torch_scatter"""
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Create output tensor
    output_shape = list(src.shape)
    output_shape[dim] = dim_size
    output = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)

    # Expand index to match src shape
    index_expanded = index
    if src.dim() > 1 and dim == 0:
        index_expanded = index.unsqueeze(1).expand_as(src)

    output.scatter_add_(dim, index_expanded, src)
    count.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))

    # Avoid division by zero
    count = count.clamp(min=1)

    if dim == 0:
        output = output / count.unsqueeze(1)
    else:
        output = output / count

    return output


def generate_data_list(data_root, test_area=5):
    """ Area """
    raw_root = os.path.join(data_root, 'raw')
    data_list = sorted(os.listdir(raw_root))
    data_list = [os.path.join(raw_root, item) for item in data_list if
                 f'Area_{test_area}' in item]
    return data_list


def load_data(data_path, voxel_size=0.04, test_mode='multi_voxel'):
    """

    Args:

    Returns:
    """
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    coord = coord - coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx = None, None, None

    if voxel_size is not None and voxel_size > 0:
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)

        if test_mode == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + \
                np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)
            idx_points.append(idx_part)
            reverse_idx = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(
                    np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(coord.shape[0]))

    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx


def get_features_by_keys(data, feature_keys):
    """ feature_keys 

    """
    features = []
    if 'pos' in feature_keys and 'pos' in data:
        features.append(data['pos'])
    if 'x' in feature_keys and 'x' in data:
        features.append(data['x'])
    if 'heights' in feature_keys and 'heights' in data:
        features.append(data['heights'])

    if len(features) == 0:
        return data.get('x', None)

    return torch.cat(features, dim=-1)


class ConfusionMatrix:
    """ IoU"""

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.value = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred, label):
        if self.ignore_index is not None:
            mask = label != self.ignore_index
            pred = pred[mask]
            label = label[mask]

        valid_mask = (label >= 0) & (label < self.num_classes)
        pred = pred[valid_mask]
        label = label[valid_mask]

        if len(pred) == 0:
            return

        pred = pred.clamp(0, self.num_classes - 1)

        k = (label * self.num_classes + pred).to(torch.long)
        self.value += torch.bincount(k, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes).to(self.value.device)

    @property
    def tp(self):
        return self.value.diag()

    @property
    def union(self):
        return self.value.sum(0) + self.value.sum(1) - self.value.diag()

    @property
    def count(self):
        return self.value.sum(1)

    def all_metrics(self):
        tp = self.tp.float()
        union = self.union.float()
        count = self.count.float()

        # IoU per class
        ious = tp / (union + 1e-10) * 100
        ious[union == 0] = float('nan')

        # Accuracy per class
        accs = tp / (count + 1e-10) * 100
        accs[count == 0] = float('nan')

        # Mean metrics (ignore nan)
        miou = torch.nanmean(ious).item()
        macc = torch.nanmean(accs).item()

        # Overall accuracy
        oa = tp.sum() / (count.sum() + 1e-10) * 100
        oa = oa.item()

        return miou, macc, oa, ious.cpu().numpy(), accs.cpu().numpy()


@torch.no_grad()
def test_room(model, coord, feat, label, idx_points,
              voxel_idx, reverse_idx_part, reverse_idx,
              pipe_transform, feature_keys, num_classes,
              scannet_num_classes=20, device='cuda', gravity_dim=2):
    """

    """
    model.eval()
    all_logits = []

    len_part = len(idx_points)
    nearest_neighbor = len_part == 1 and voxel_idx is not None

    for idx_subcloud in range(len(idx_points)):
        if not (nearest_neighbor and idx_subcloud > 0):
            idx_part = idx_points[idx_subcloud]
            coord_part = coord[idx_part].copy()
            coord_part -= coord_part.min(0)
            feat_part = feat[idx_part] if feat is not None else None

            data = {'pos': coord_part}
            if feat_part is not None:
                data['x'] = feat_part

            if pipe_transform is not None:
                data = pipe_transform(data)

            if 'heights' not in data.keys():
                data['heights'] = torch.from_numpy(
                    coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32))

            if 'x' in data.keys():
                if isinstance(data['x'], np.ndarray):
                    data['x'] = torch.from_numpy(data['x'])
                data['x'] = data['x'].unsqueeze(0)
            if isinstance(data['pos'], np.ndarray):
                data['pos'] = torch.from_numpy(data['pos'])
            data['pos'] = data['pos'].unsqueeze(0)
            if isinstance(data['heights'], np.ndarray):
                data['heights'] = torch.from_numpy(data['heights'])
            data['heights'] = data['heights'].unsqueeze(0)

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device, non_blocking=True)

            data['x'] = get_features_by_keys(data, feature_keys)

            if data['x'] is not None:
                data['x'] = data['x'].transpose(1, 2)  # (B, N, C) -> (B, C, N)

            output_dict = model(data)
            logits = output_dict['logits']

        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    if all_logits.dim() == 3:
        if all_logits.shape[1] == scannet_num_classes:  # (B, C, N)
            all_logits = all_logits.transpose(
                1, 2).reshape(-1, scannet_num_classes)
        else:  # (B, N, C)
            all_logits = all_logits.reshape(-1, scannet_num_classes)

    if not nearest_neighbor:
        idx_points_tensor = torch.from_numpy(
            np.hstack(idx_points)).to(device, non_blocking=True).long()

        if scatter is not None:
            all_logits = scatter(
                all_logits, idx_points_tensor, dim=0, reduce='mean')
        else:
            all_logits = scatter_mean_fallback(all_logits, idx_points_tensor, dim=0,
                                               dim_size=coord.shape[0])
    else:
        all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]

    return all_logits


def map_logits_scannet_to_s3dis(scannet_logits, num_s3dis_classes=13):
    """

    """
    device = scannet_logits.device
    batch_size = scannet_logits.shape[0]
    s3dis_logits = torch.full(
        (batch_size, num_s3dis_classes), -float('inf'), device=device)

    for scannet_cls, s3dis_cls in SCANNET_TO_S3DIS.items():
        if s3dis_cls >= 0:
            s3dis_logits[:, s3dis_cls] = torch.max(
                s3dis_logits[:, s3dis_cls],
                scannet_logits[:, scannet_cls]
            )

    return s3dis_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='ScanNet config (e.g., cfgs/scannet/pointnext-b.yaml)')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to ScanNet pretrained checkpoint')
    parser.add_argument('--s3dis_data_root', type=str,
                        default='/home/LIANGYudong_2023/PointNeXt/data/s3disfull')
    parser.add_argument('--test_area', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.04,
                        help='Voxel size for evaluation (default: 0.04)')
    parser.add_argument('--test_mode', type=str, default='multi_voxel',
                        choices=['multi_voxel', 'nearest_neighbor'],
                        help='Test mode: multi_voxel (standard) or nearest_neighbor (fast)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable CUDA debug mode')
    args = parser.parse_args()

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("[DEBUG] CUDA_LAUNCH_BLOCKING enabled")

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    print(f"\n{'='*60}")
    print("Cross-Dataset Zero-Shot Evaluation (Standard Protocol)")
    print(f"ScanNet Model -> S3DIS Test (Area {args.test_area})")
    print(f"{'='*60}")
    print(f"Test Mode: {args.test_mode}")
    print(f"Voxel Size: {args.voxel_size}")

    print("\n[Step 1] Building ScanNet model...")
    model = build_model_from_cfg(cfg.model).to(device)

    print(f"\n[Step 2] Loading checkpoint: {args.pretrained_path}")
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('fusion_gate_mlp', 'fusion_module.gate_mlp')
        new_state_dict[new_key] = v
    state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("  Checkpoint loaded successfully!")

    print(f"\n[Step 3] Loading S3DIS test data (Area {args.test_area})...")
    data_list = generate_data_list(args.s3dis_data_root, args.test_area)
    print(f"  Found {len(data_list)} rooms to evaluate")

    trans_split = 'val' if cfg.datatransforms.get(
        'test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    feature_keys = cfg.get('feature_keys', 'x,heights')
    gravity_dim = cfg.datatransforms.kwargs.get('gravity_dim', 2)
    scannet_num_classes = cfg.get('num_classes', 20)

    print(f"\n[Step 4] Evaluating with {args.test_mode} mode...")

    all_cm = ConfusionMatrix(num_classes=13, ignore_index=None)

    for room_idx, data_path in enumerate(tqdm(data_list, desc="Rooms")):
        room_name = os.path.basename(data_path)

        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = \
            load_data(data_path, args.voxel_size, args.test_mode)

        label_tensor = torch.from_numpy(label.astype(np.int64)).to(device)

        scannet_logits = test_room(
            model, coord, feat, label, idx_points,
            voxel_idx, reverse_idx_part, reverse_idx,
            pipe_transform, feature_keys, 13,
            scannet_num_classes, device, gravity_dim
        )

        s3dis_logits = map_logits_scannet_to_s3dis(
            scannet_logits, num_s3dis_classes=13)

        pred = s3dis_logits.argmax(dim=1)

        all_cm.update(pred, label_tensor)

    miou, macc, oa, ious, accs = all_cm.all_metrics()

    print(f"\n{'='*60}")
    print("Cross-Dataset Zero-Shot Results (ScanNet -> S3DIS)")
    print(f"Evaluation Protocol: {args.test_mode.upper()}")
    print(f"{'='*60}")
    print(f"Overall Accuracy (OA): {oa:.2f}%")
    print(f"Mean Accuracy (mAcc):  {macc:.2f}%")
    print(f"Mean IoU (mIoU):       {miou:.2f}%")
    print(f"\nPer-class IoU:")
    for i, cls_name in enumerate(S3DIS_CLASSES):
        iou = ious[i]
        if np.isnan(iou):
            print(f"  {cls_name:15s}: N/A (no samples)")
        else:
            print(f"  {cls_name:15s}: {iou:.2f}%")

    # floor, wall, window, door, chair, table, bookcase, sofa, clutter
    mappable_classes = [1, 2, 5, 6, 7, 8, 9, 10, 12]
    mappable_ious = [ious[i]
                     for i in mappable_classes if not np.isnan(ious[i])]
    mappable_miou = np.mean(mappable_ious) if mappable_ious else 0.0
    if mappable_ious:
        print(f"\n[Mappable Classes Only] mIoU: {mappable_miou:.2f}%")
        print(
            f"  Mapped classes: {[S3DIS_CLASSES[i] for i in mappable_classes]}")

    checkpoint_dir = os.path.dirname(args.pretrained_path)
    save_dir = os.path.dirname(checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"cross_dataset_eval_{args.test_mode}_{timestamp}.txt"
    result_path = os.path.join(save_dir, result_filename)

    result_lines = []
    result_lines.append("=" * 60)
    result_lines.append("Cross-Dataset Zero-Shot Results (ScanNet -> S3DIS)")
    result_lines.append(f"Evaluation Protocol: {args.test_mode.upper()}")
    result_lines.append("=" * 60)
    result_lines.append(f"Checkpoint: {args.pretrained_path}")
    result_lines.append(
        f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result_lines.append(f"Test Area: {args.test_area}")
    result_lines.append(f"Voxel Size: {args.voxel_size}")
    result_lines.append(f"Test Mode: {args.test_mode}")
    result_lines.append("")
    result_lines.append(f"Overall Accuracy (OA): {oa:.2f}%")
    result_lines.append(f"Mean Accuracy (mAcc):  {macc:.2f}%")
    result_lines.append(f"Mean IoU (mIoU):       {miou:.2f}%")
    result_lines.append("")
    result_lines.append("Per-class IoU:")
    for i, cls_name in enumerate(S3DIS_CLASSES):
        iou = ious[i]
        if np.isnan(iou):
            result_lines.append(f"  {cls_name:15s}: N/A (no samples)")
        else:
            result_lines.append(f"  {cls_name:15s}: {iou:.2f}%")
    result_lines.append("")
    result_lines.append(f"[Mappable Classes Only] mIoU: {mappable_miou:.2f}%")

    with open(result_path, 'w') as f:
        f.write('\n'.join(result_lines))

    print(f"\n[Results saved to] {result_path}")


if __name__ == '__main__':
    main()
