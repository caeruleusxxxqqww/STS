#!/usr/bin/env python3
"""


Usage:
    python scripts/run_semantic_conflict_visualization.py \
        --cfg cfgs/scannet/pointnext-b-mixratio.yaml \
        --pretrained_path_baseline log/scannet/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/scannet/ours/checkpoint/best.pth \
        --dataset scannet \
        --gpu 1 \
        --max_samples 100 \
        --output_dir visualizations/conflict_test \
        --max_viz_samples 10
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
from tqdm import tqdm

from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.loss.custom_innovations import (
    SemanticConflictGenerator,
    SemanticConflictEvaluator,
    visualize_semantic_conflict_test
)
# fmt: on


def parse_args():
    parser = argparse.ArgumentParser(
        description='Semantic Conflict Test with Visualization for STS Validation')

    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file path (e.g., cfgs/scannet/pointnext-s.yaml)')

    # Model paths
    parser.add_argument('--pretrained_path_baseline', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--pretrained_path_ours', type=str, required=True,
                        help='Path to STS-trained model checkpoint')

    # GPU settings
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU ID to use (default: 3)')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='s3dis',
                        choices=['s3dis', 'scannet'],
                        help='Dataset name: s3dis (13 classes) or scannet (20 classes)')

    # Test settings
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (None = all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # Attack settings
    parser.add_argument('--attack_ratio', type=float, default=1.0,
                        help='Ratio of target points to attack (0.0-1.0)')
    parser.add_argument('--preserve_boundary', action='store_true',
                        help='Preserve boundary points of target objects')

    # Visualization settings
    parser.add_argument('--output_dir', type=str, default='visualizations/conflict_test',
                        help='Directory to save visualization images')
    parser.add_argument('--max_viz_samples', type=int, default=10,
                        help='Maximum number of samples to visualize')

    return parser.parse_args()


def load_model(cfg, pretrained_path, device):
    """Load model from config and checkpoint ( run_semantic_conflict_test.py )"""
    model = build_model_from_cfg(cfg.model).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading checkpoint: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle DataParallel prefix
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Loaded successfully")
    else:
        print(f"  ⚠️ No checkpoint loaded, using random weights")

    model.eval()
    return model


def evaluate_on_conflict_samples_with_viz(
    model_baseline, model_ours, dataloader, generator,
    evaluator_baseline, evaluator_ours, device, cfg,
    max_samples=None, max_viz_samples=10, output_dir='visualizations/conflict_test',
    dataset='s3dis'
):
    """

    Args:
        generator: SemanticConflictGenerator

    Returns:
    """
    model_baseline.eval()
    model_ours.eval()
    evaluator_baseline.reset()
    evaluator_ours.reset()

    attack_success_count = 0
    total_count = 0
    visualization_paths = []
    viz_count = 0

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing & Visualizing")

        for batch_idx, data in enumerate(pbar):
            if max_samples and batch_idx >= max_samples:
                break

            total_count += 1

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            data_original = {k: v.clone() if torch.is_tensor(
                v) else v for k, v in data.items()}

            data_attacked, attack_info = generator.generate_conflict_sample(
                data, return_attack_info=True
            )

            if not attack_info['success']:
                continue

            attack_success_count += 1


            data_attacked_baseline = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in data_attacked.items()}
            data_attacked_baseline['x'] = get_features_by_keys(
                data_attacked_baseline, cfg.feature_keys)

            data_attacked_ours = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in data_attacked.items()}
            data_attacked_ours['x'] = get_features_by_keys(
                data_attacked_ours, cfg.feature_keys)


            try:
                output_baseline = model_baseline(data_attacked_baseline)
                if isinstance(output_baseline, dict):
                    logits_baseline = output_baseline.get(
                        'logits', output_baseline.get('seg_logits'))
                else:
                    logits_baseline = output_baseline

                if logits_baseline.dim() == 3:  # (B, C, N)
                    pred_baseline = logits_baseline.argmax(dim=1)  # (B, N)
                    pred_baseline = pred_baseline.reshape(-1)
                else:
                    pred_baseline = logits_baseline.argmax(dim=-1)
                    if pred_baseline.dim() > 1:
                        pred_baseline = pred_baseline.reshape(-1)
            except Exception as e:
                print(f"\n  [Warning] Baseline forward error: {e}")
                continue

            try:
                output_ours = model_ours(data_attacked_ours)
                if isinstance(output_ours, dict):
                    logits_ours = output_ours.get(
                        'logits', output_ours.get('seg_logits'))
                else:
                    logits_ours = output_ours

                if logits_ours.dim() == 3:  # (B, C, N)
                    pred_ours = logits_ours.argmax(dim=1)  # (B, N)
                    pred_ours = pred_ours.reshape(-1)
                else:
                    pred_ours = logits_ours.argmax(dim=-1)
                    if pred_ours.dim() > 1:
                        pred_ours = pred_ours.reshape(-1)
            except Exception as e:
                print(f"\n  [Warning] Ours forward error: {e}")
                continue

            labels = data_attacked.get('y', data_attacked.get('label'))
            if labels is not None:
                if labels.dim() > 1:
                    labels = labels.reshape(-1)

                evaluator_baseline.update(
                    pred_baseline.cpu(), labels.cpu(), attack_info)
                evaluator_ours.update(
                    pred_ours.cpu(), labels.cpu(), attack_info)

            if viz_count < max_viz_samples:
                output_path = os.path.join(
                    output_dir,
                    f"conflict_test_sample_{batch_idx:04d}.png"
                )

                try:
                    result = visualize_semantic_conflict_test(
                        data_original=data_original,
                        data_conflict=data_attacked,
                        pred_baseline=pred_baseline.cpu(),
                        pred_ours=pred_ours.cpu(),
                        attack_info=attack_info,
                        output_path=output_path,
                        dataset=dataset,
                        model_name_baseline="Baseline",
                        model_name_ours="Ours (STS)"
                    )
                    if result is not None:
                        visualization_paths.append(output_path)
                        viz_count += 1
                except Exception as e:
                    print(
                        f"  [Warning] Visualization failed for batch {batch_idx}: {e}")

            if attack_success_count > 0:
                tr_baseline = evaluator_baseline.correct_geo_predictions / \
                    max(evaluator_baseline.total_attacked_points, 1)
                tr_ours = evaluator_ours.correct_geo_predictions / \
                    max(evaluator_ours.total_attacked_points, 1)
                pbar.set_postfix({
                    'attacks': attack_success_count,
                    'TR_b': f'{tr_baseline:.1%}',
                    'TR_o': f'{tr_ours:.1%}',
                    'viz': f'{viz_count}/{max_viz_samples}'
                })

    print(
        f"\nAttack success rate: {attack_success_count}/{total_count} ({attack_success_count/max(total_count,1):.1%})")

    return evaluator_baseline.compute_metrics(), evaluator_ours.compute_metrics(), visualization_paths


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print("\n" + "=" * 70)
    print("🔬 SEMANTIC CONFLICT TEST WITH VISUALIZATION (GOLD STANDARD)")
    print("=" * 70)

    print(f"\n[Step 1] Loading config: {args.cfg}")
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(
            f"[Step 2] Using GPU {args.gpu} (cuda:0 after CUDA_VISIBLE_DEVICES)")
    else:
        device = torch.device('cpu')
        print(f"[Step 2] Using device: CPU")

    print(f"[Step 3] Building validation dataloader...")
    val_loader = build_dataloader_from_cfg(
        args.batch_size,  # batch_size
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split='val',
        distributed=False
    )
    print(f"  Validation samples: {len(val_loader.dataset)}")

    print(
        f"[Step 4] Initializing conflict generator for {args.dataset.upper()}...")
    generator = SemanticConflictGenerator(
        dataset=args.dataset,
        attack_ratio=args.attack_ratio,
        preserve_boundary=args.preserve_boundary,
        seed=42
    )

    print(f"\n[Step 5] Loading models...")
    print(f"Loading Baseline model...")
    model_baseline = load_model(cfg, args.pretrained_path_baseline, device)
    print(f"Loading Ours (STS) model...")
    model_ours = load_model(cfg, args.pretrained_path_ours, device)

    print(f"\n[Step 6] Running semantic conflict test with visualization...")
    evaluator_baseline = SemanticConflictEvaluator(dataset=args.dataset)
    evaluator_ours = SemanticConflictEvaluator(dataset=args.dataset)

    metrics_baseline, metrics_ours, visualization_paths = evaluate_on_conflict_samples_with_viz(
        model_baseline=model_baseline,
        model_ours=model_ours,
        dataloader=val_loader,
        generator=generator,
        evaluator_baseline=evaluator_baseline,
        evaluator_ours=evaluator_ours,
        device=device,
        cfg=cfg,
        max_samples=args.max_samples,
        max_viz_samples=args.max_viz_samples,
        output_dir=args.output_dir,
        dataset=args.dataset
    )

    print("\n" + "=" * 70)
    print(f"📊 Semantic Conflict Test Results")
    print("=" * 70)

    evaluator_baseline.print_report("Baseline")
    evaluator_ours.print_report("Ours (STS)")

    print("\n" + "🎯" * 35)
    print("FINAL COMPARISON")
    print("🎯" * 35)

    if 'error' not in metrics_baseline and 'error' not in metrics_ours:
        tr_baseline = metrics_baseline['texture_robustness']
        tr_ours = metrics_ours['texture_robustness']
        tdr_baseline = metrics_baseline['texture_deception_rate']
        tdr_ours = metrics_ours['texture_deception_rate']

        print(f"\n{'Metric':<35} {'Baseline':<15} {'Ours (STS)':<15} {'Δ':<15}")
        print("-" * 80)
        print(
            f"{'Texture Robustness (TR) ↑':<35} {tr_baseline:<15.2%} {tr_ours:<15.2%} {tr_ours - tr_baseline:+.2%}")
        print(f"{'Texture Deception Rate (TDR) ↓':<35} {tdr_baseline:<15.2%} {tdr_ours:<15.2%} {tdr_ours - tdr_baseline:+.2%}")

        miou_b = metrics_baseline.get('miou_attacked_classes', 0)
        miou_o = metrics_ours.get('miou_attacked_classes', 0)
        print(
            f"{'mIoU on Attacked Classes ↑':<35} {miou_b:<15.2%} {miou_o:<15.2%} {miou_o - miou_b:+.2%}")

        print("\n" + "=" * 80)

        if tr_ours > tr_baseline + 0.05:
            print("✅ SUCCESS: STS significantly improves texture robustness!")
        elif tr_ours > tr_baseline:
            print("⚠️  PARTIAL: STS shows improvement, consider longer training.")
        else:
            print("❌ UNEXPECTED: STS did not improve robustness. Check training.")

    print(f"\n🖼️  Visualizations ({len(visualization_paths)} images):")
    for path in visualization_paths:
        print(f"  - {path}")

    print("\n✅ Semantic Conflict Test with Visualization Complete!")


if __name__ == '__main__':
    main()
