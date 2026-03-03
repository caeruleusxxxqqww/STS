#!/usr/bin/env python
"""

- block_size: 1.5m
- stride: 0.75m (50% overlap)

    python scripts/cross_dataset_eval_s3dis_to_scannet.py \
        --pretrained_path log/s3dis/.../checkpoint/xxx_ckpt_best.pth
"""
import sys
import os
import argparse
import torch
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.transforms import build_transforms_from_cfg


# ============================================================================
# ============================================================================
S3DIS_TO_SCANNET = {
    0: -1,
    1: 1,    # floor -> floor
    2: 0,    # wall -> wall
    3: -1,   # beam -> ignore
    4: -1,   # column -> ignore
    5: 8,    # window -> window
    6: 7,    # door -> door
    7: 4,    # chair -> chair
    8: 6,    # table -> table
    9: 9,    # bookcase -> bookshelf
    10: 5,   # sofa -> sofa
    11: -1,  # board -> ignore
    12: 19,  # clutter -> otherfurniture
}

SCANNET_CLASSES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                   'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                   'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                   'bathtub', 'otherfurniture']

S3DIS_NUM_CLASSES = 13


def map_s3dis_to_scannet(predictions):
    """ S3DIS  ScanNet """
    mapped = torch.full_like(predictions, -1)
    for s3dis_cls, scannet_cls in S3DIS_TO_SCANNET.items():
        if scannet_cls >= 0:
            mapped[predictions == s3dis_cls] = scannet_cls
    return mapped


def compute_metrics(predictions, labels, num_classes=20, ignore_index=-1):
    """ mIoU, OA, mAcc"""
    valid_mask = labels != ignore_index
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]

    if len(predictions) == 0:
        return {'oa': 0, 'macc': 0, 'miou': 0, 'iou_per_class': np.zeros(num_classes)}

    correct = (predictions == labels).sum().item()
    total = len(labels)
    oa = correct / total * 100

    iou_per_class = []
    acc_per_class = []

    for cls in range(num_classes):
        pred_cls = predictions == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        if union > 0:
            iou_per_class.append(intersection / union * 100)
        else:
            iou_per_class.append(np.nan)

        if label_cls.sum() > 0:
            acc_per_class.append(intersection / label_cls.sum().item() * 100)
        else:
            acc_per_class.append(np.nan)

    iou_per_class = np.array(iou_per_class)
    acc_per_class = np.array(acc_per_class)

    return {
        'oa': oa,
        'macc': np.nanmean(acc_per_class),
        'miou': np.nanmean(iou_per_class),
        'iou_per_class': iou_per_class
    }


def sliding_window_split(pos, block_size=1.5, stride=0.75, min_points=100):
    """

    Args:

    Returns:
        chunks: list of (chunk_indices, chunk_center)
    """
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos

    min_xyz = pos_np.min(axis=0)
    max_xyz = pos_np.max(axis=0)

    chunks = []

    x_start = min_xyz[0]
    while x_start < max_xyz[0]:
        y_start = min_xyz[1]
        while y_start < max_xyz[1]:
            x_end = x_start + block_size
            y_end = y_start + block_size

            mask = (
                (pos_np[:, 0] >= x_start) & (pos_np[:, 0] < x_end) &
                (pos_np[:, 1] >= y_start) & (pos_np[:, 1] < y_end)
            )

            indices = np.where(mask)[0]

            if len(indices) >= min_points:
                center = np.array([
                    (x_start + x_end) / 2,
                    (y_start + y_end) / 2,
                    0
                ])
                chunks.append((indices, center))

            y_start += stride
        x_start += stride

    return chunks


def process_chunk(model, chunk_pos, chunk_rgb, device, use_pos=False):
    """


    Args:

    Returns:
    """
    chunk_center = chunk_pos.mean(dim=0, keepdim=True)
    normalized_pos = chunk_pos - chunk_center  # (N, 3)

    z_min = normalized_pos[:, 2].min()
    normalized_pos[:, 2] = normalized_pos[:, 2] - z_min

    heights = normalized_pos[:, 2:3]  # (N, 1)

    if use_pos:
        # feature_keys: pos,x,heights -> in_channels: 7
        features = torch.cat(
            [normalized_pos, chunk_rgb, heights], dim=-1)  # (N, 7)
    else:
        # feature_keys: x,heights -> in_channels: 4
        features = torch.cat([chunk_rgb, heights], dim=-1)  # (N, 4)

    pos_input = normalized_pos.unsqueeze(0)  # (1, N, 3)
    features = features.unsqueeze(0).transpose(1, 2).contiguous()  # (1, C, N)
    heights = heights.unsqueeze(0)  # (1, N, 1)

    data_input = {
        'pos': pos_input,
        'x': features,
        'heights': heights
    }

    output = model(data_input)

    if isinstance(output, dict):
        logits = output.get('logits', output.get('seg_logits'))
    else:
        logits = output

    # logits: (1, num_classes, N) -> (N, num_classes)
    if logits.dim() == 3:
        logits = logits.squeeze(0).transpose(0, 1)  # (N, num_classes)

    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='S3DIS ')
    parser.add_argument('--scannet_data_root', type=str,
                        default='/home/LIANGYudong_2023/PointNeXt/data/scannet_data',
                        help='ScanNet ')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'], help='')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ')
    parser.add_argument('--block_size', type=float, default=1.5,
                        help='')
    parser.add_argument('--stride', type=float, default=0.75,
                        help='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("[DEBUG] CUDA_LAUNCH_BLOCKING enabled")

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = os.path.dirname(args.pretrained_path)
    run_dir = os.path.dirname(checkpoint_dir)
    saved_cfg_path = os.path.join(run_dir, 'cfg.yaml')

    if not os.path.exists(saved_cfg_path):
        raise FileNotFoundError(f": {saved_cfg_path}")

    print(f"[] : {saved_cfg_path}")
    with open(saved_cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    if not isinstance(cfg, EasyConfig):
        cfg_dict = cfg
        cfg = EasyConfig()
        cfg.update(cfg_dict)

    print(f"\n{'='*60}")
    print("Cross-Dataset Zero-Shot Evaluation (Sliding Window)")
    print(f"S3DIS Model -> ScanNet Test")
    print(f"Block Size: {args.block_size}m, Stride: {args.stride}m")
    print(f"{'='*60}")

    print("\n[Step 1] Building model...")
    model_num_classes = cfg.model.cls_args.get('num_classes', 13)
    encoder_in_channels = cfg.model.encoder_args.get('in_channels', 4)
    encoder_width = cfg.model.encoder_args.get('width', 64)
    print(
        f"  num_classes: {model_num_classes}, in_channels: {encoder_in_channels}, width: {encoder_width}")

    model = build_model_from_cfg(cfg.model).to(device)

    print(f"\n[Step 2] Loading checkpoint...")
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('fusion_gate_mlp', 'fusion_module.gate_mlp')
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=False)
    model.eval()
    if missing_keys:
        print(
            f"  [] : {len(missing_keys)} ")
        for k in missing_keys[:5]:
            print(f"         - {k}")
        if len(missing_keys) > 5:
            print(f"         ...  {len(missing_keys) - 5} ")
    if unexpected_keys:
        print(
            f"  [] : {len(unexpected_keys)} ")
    print("  ！")

    print(f"\n[Step 3] Loading ScanNet {args.split} data...")

    from openpoints.dataset.scannetv2.scannet import ScanNet

    test_dataset = ScanNet(
        data_root=args.scannet_data_root,
        split=args.split,
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        presample=False,
        variable=True,
        loop=1
    )

    print(f"  Loaded {len(test_dataset)} scenes")

    feature_keys = cfg.get('feature_keys', 'x,heights')
    if isinstance(feature_keys, str):
        feature_keys = feature_keys.split(',')
    use_pos = 'pos' in feature_keys
    print(f"  Feature keys: {feature_keys}, use_pos: {use_pos}")

    color_mean = np.array([0.5136457, 0.49523646, 0.44921124])
    color_std = np.array([0.18308958, 0.18415008, 0.19252081])
    print(
        f"  Using S3DIS color normalization: mean={color_mean}, std={color_std}")

    print(f"\n[Step 4] Evaluating with sliding window...")

    num_scenes = len(test_dataset)
    print(f"  Total scenes to process: {num_scenes}")

    if num_scenes == 0:
        raise RuntimeError(
            "No scenes found in the dataset! Check data_root path.")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for scene_idx in tqdm(range(num_scenes), desc="Scenes"):
            try:
                data = test_dataset[scene_idx]

                if isinstance(data['pos'], torch.Tensor):
                    scene_pos = data['pos'].float()
                    scene_rgb = data['x'].float()
                    scene_labels = data['y']
                else:
                    scene_pos = torch.from_numpy(data['pos']).float()
                    scene_rgb = torch.from_numpy(data['x']).float()
                    scene_labels = torch.from_numpy(data['y'])

                N = scene_pos.shape[0]

                scene_rgb_np = scene_rgb.numpy()
                if scene_rgb_np.max() > 1:
                    scene_rgb_np = scene_rgb_np / 255.0
                scene_rgb_normalized = (scene_rgb_np - color_mean) / color_std
                scene_rgb_normalized = torch.from_numpy(
                    scene_rgb_normalized).float()

                vote_logits = torch.zeros(N, S3DIS_NUM_CLASSES)
                vote_counts = torch.zeros(N)

                chunks = sliding_window_split(
                    scene_pos,
                    block_size=args.block_size,
                    stride=args.stride,
                    min_points=512
                )

                if scene_idx == 0:
                    print(
                        f"\n  [DEBUG] Scene 0: {N} points, {len(chunks)} chunks")

                for chunk_indices, chunk_center in chunks:
                    if len(chunk_indices) < 256:
                        continue

                    chunk_pos = scene_pos[chunk_indices].to(device)
                    chunk_rgb = scene_rgb_normalized[chunk_indices].to(device)

                    try:
                        logits = process_chunk(
                            model, chunk_pos, chunk_rgb, device, use_pos)
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "invalid" in str(e):
                            print(
                                f"\n  [WARN] Chunk with {len(chunk_indices)} points failed: {e}")
                            continue
                        raise

                    vote_logits[chunk_indices] += logits.cpu()
                    vote_counts[chunk_indices] += 1

                uncovered = vote_counts == 0
                if uncovered.any():
                    vote_counts[uncovered] = 1
                    vote_logits[uncovered, 12] = 1.0

                vote_logits = vote_logits / vote_counts.unsqueeze(1)

                predictions = vote_logits.argmax(dim=1)

                mapped_predictions = map_s3dis_to_scannet(predictions)

                all_predictions.append(mapped_predictions)
                all_labels.append(scene_labels)

            except Exception as e:
                print(f"\n[ERROR] Scene {scene_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    if len(all_predictions) == 0:
        raise RuntimeError("No predictions generated! All scenes failed.")

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(
        all_predictions, all_labels, num_classes=20, ignore_index=-1)

    print(f"\n{'='*60}")
    print("Cross-Dataset Zero-Shot Results (S3DIS -> ScanNet)")
    print(f"{'='*60}")
    print(f"Overall Accuracy (OA): {metrics['oa']:.2f}%")
    print(f"Mean Accuracy (mAcc):  {metrics['macc']:.2f}%")
    print(f"Mean IoU (mIoU):       {metrics['miou']:.2f}%")
    print(f"\nPer-class IoU:")
    for i, cls_name in enumerate(SCANNET_CLASSES):
        iou = metrics['iou_per_class'][i]
        if np.isnan(iou):
            print(f"  {cls_name:18s}: N/A (no samples)")
        else:
            print(f"  {cls_name:18s}: {iou:.2f}%")

    mappable_classes = [0, 1, 4, 5, 6, 7, 8, 9, 19]
    mappable_ious = [metrics['iou_per_class'][i]
                     for i in mappable_classes
                     if not np.isnan(metrics['iou_per_class'][i])]
    mappable_miou = np.mean(mappable_ious) if mappable_ious else 0.0
    print(f"\n[Mappable Classes Only] mIoU: {mappable_miou:.2f}%")

    save_dir = run_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        save_dir, f"cross_eval_s3dis_to_scannet_{timestamp}.txt")

    result_lines = [
        "=" * 60,
        "Cross-Dataset Zero-Shot Results (S3DIS -> ScanNet)",
        "=" * 60,
        f"Checkpoint: {args.pretrained_path}",
        f"Block Size: {args.block_size}m, Stride: {args.stride}m",
        f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Overall Accuracy (OA): {metrics['oa']:.2f}%",
        f"Mean Accuracy (mAcc):  {metrics['macc']:.2f}%",
        f"Mean IoU (mIoU):       {metrics['miou']:.2f}%",
        "",
        "Per-class IoU:"
    ]
    for i, cls_name in enumerate(SCANNET_CLASSES):
        iou = metrics['iou_per_class'][i]
        if np.isnan(iou):
            result_lines.append(f"  {cls_name:18s}: N/A")
        else:
            result_lines.append(f"  {cls_name:18s}: {iou:.2f}%")
    result_lines.append("")
    result_lines.append(f"[Mappable Classes Only] mIoU: {mappable_miou:.2f}%")

    with open(result_path, 'w') as f:
        f.write('\n'.join(result_lines))

    print(f"\n[Results saved to] {result_path}")


if __name__ == '__main__':
    main()
