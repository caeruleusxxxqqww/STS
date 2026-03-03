#!/usr/bin/env python3
"""




Usage:
    python scripts/run_semantic_conflict_test.py \
        --cfg cfgs/s3dis/pointnext-s.yaml \
        --pretrained_path log/s3dis/your_model/checkpoint/best.pth \
        --dataset s3dis \
        --gpu 3

    python scripts/run_semantic_conflict_test.py \
        --cfg cfgs/s3dis/pointnext-s.yaml \
        --pretrained_path_baseline log/s3dis/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/s3dis/ours_sts/checkpoint/best.pth \
        --dataset s3dis \
        --gpu 3
        
    python scripts/run_semantic_conflict_test.py \
        --cfg cfgs/scannet/pointnext-s.yaml \
        --pretrained_path_baseline log/scannet/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/scannet/ours_sts/checkpoint/best.pth \
        --dataset scannet \
        --gpu 3

    python scripts/run_semantic_conflict_test.py \
        --cfg cfgs/scannet/pointnext-b-mixratio.yaml \
        --data_cfg cfgs/s3dis/default.yaml \
        --pretrained_path_baseline log/scannet/baseline/checkpoint/best.pth \
        --pretrained_path_ours log/scannet/ours_sts/checkpoint/best.pth \
        --gpu 3

Author: Auto-generated for STS validation
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
    run_semantic_conflict_test,
    compare_models_on_conflict_test
)
# fmt: on


def parse_args():
    parser = argparse.ArgumentParser(
        description='Semantic Conflict Test for STS Validation')

    # Config
    parser.add_argument('--cfg', type=str, required=True,
                        help='Config file path for model (e.g., cfgs/scannet/pointnext-b-mixratio.yaml)')
    parser.add_argument('--data_cfg', type=str, default=None,
                        help='Separate config file for data loading (cross-domain test). '
                             'E.g., use S3DIS config while model is trained on ScanNet. '
                             'Labels will be auto-mapped to the model\'s class space. '
                             'If not provided, data is loaded from --cfg.')

    # Model paths
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained model checkpoint (for single model test)')
    parser.add_argument('--pretrained_path_baseline', type=str, default=None,
                        help='Path to baseline model checkpoint (for comparison)')
    parser.add_argument('--pretrained_path_ours', type=str, default=None,
                        help='Path to STS-trained model checkpoint (for comparison)')

    # GPU settings
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU ID to use (default: 3)')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='s3dis',
                        choices=['s3dis', 'scannet'],
                        help='Dataset name: s3dis (13 classes) or scannet (20 classes)')

    # Test settings
    parser.add_argument('--max_samples', type=int, default=68,
                        help='Maximum number of samples to test (None = all, default=30)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # Attack settings
    parser.add_argument('--attack_ratio', type=float, default=1.0,
                        help='Ratio of target points to attack (0.0-1.0)')
    parser.add_argument('--preserve_boundary', action='store_true',
                        help='Preserve boundary points of target objects')
    parser.add_argument('--all_classes', action='store_true',
                        help='Test ALL classes as both targets and sources (not just default subset)')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/semantic_conflict',
                        help='Directory to save results')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save attacked samples for visualization')

    return parser.parse_args()


def load_model(cfg, pretrained_path, device):
    """Load model from config and checkpoint"""
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


def evaluate_on_conflict_samples(model, dataloader, generator, evaluator, device, cfg,
                                 max_samples=None):
    """

    Args:

    Returns:
    """
    model.eval()
    evaluator.reset()

    attack_success_count = 0
    total_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch_idx, data in enumerate(pbar):
            if max_samples and batch_idx >= max_samples:
                break

            total_count += 1

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            data_attacked, attack_infos = generator.generate_all_conflict_samples(
                data
            )

            if len(attack_infos) == 0:
                continue

            attack_success_count += 1

            data_attacked['x'] = get_features_by_keys(
                data_attacked, cfg.feature_keys)

            try:
                output = model(data_attacked)

                if isinstance(output, dict):
                    logits = output.get('logits', output.get('seg_logits'))
                else:
                    logits = output

                if logits.dim() == 3:  # (B, C, N)
                    predictions = logits.argmax(dim=1)  # (B, N)
                    predictions = predictions.reshape(-1)
                else:
                    predictions = logits.argmax(dim=-1)
                    if predictions.dim() > 1:
                        predictions = predictions.reshape(-1)

            except Exception as e:
                print(f"\n  [Warning] Forward error: {e}")
                continue

            labels = data_attacked.get('y', data_attacked.get('label'))
            if labels is not None:
                if labels.dim() > 1:
                    labels = labels.reshape(-1)

                for attack_info in attack_infos:
                    evaluator.update(predictions.cpu(),
                                     labels.cpu(), attack_info)

            if attack_success_count > 0:
                current_tr = evaluator.correct_geo_predictions / \
                    max(evaluator.total_attacked_points, 1)
                current_tdr = evaluator.deceptive_tex_predictions / \
                    max(evaluator.total_attacked_points, 1)
                pbar.set_postfix({
                    'attacks': attack_success_count,
                    'TR': f'{current_tr:.1%}',
                    'TDR': f'{current_tdr:.1%}'
                })

    print(
        f"\nAttack success rate: {attack_success_count}/{total_count} ({attack_success_count/max(total_count,1):.1%})")

    return evaluator.compute_metrics()


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print("\n" + "=" * 70)
    print("🔬 SEMANTIC CONFLICT TEST (GOLD STANDARD)")
    print("=" * 70)

    print(f"\n[Step 1] Loading config: {args.cfg}")
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    # Ensure feature_keys is defined (some model configs don't include it)
    if not cfg.get('feature_keys', None):
        model_dataset = cfg.dataset.common.get('NAME', '').upper()
        if model_dataset == 'SCANNET':
            cfg.feature_keys = 'pos,x,heights'
        else:  # S3DIS or default
            cfg.feature_keys = 'x,heights'
        print(
            f"  [Info] feature_keys not in config, set to: {cfg.feature_keys}")

    is_cross_domain = False
    prediction_remap = None
    cross_domain_target = None
    cross_domain_source = None
    data_cfg = cfg

    if args.data_cfg:
        print(f"  [Cross-Domain] Loading data config: {args.data_cfg}")
        data_cfg = EasyConfig()
        data_cfg.load(args.data_cfg, recursive=True)

        model_num_classes = cfg.get(
            'num_classes', cfg.model.cls_args.get('num_classes', None))
        data_num_classes = data_cfg.get('num_classes', 13)
        model_dataset_name = cfg.dataset.common.get('NAME', '').upper()
        data_dataset_name = data_cfg.dataset.common.get('NAME', '').upper()

        if model_dataset_name != data_dataset_name:
            is_cross_domain = True
            print(f"  [Cross-Domain] Detected: model={model_dataset_name}({model_num_classes} classes) "
                  f"→ data={data_dataset_name}({data_num_classes} classes)")

            if data_dataset_name == 'S3DIS' and model_dataset_name == 'SCANNET':
                args.dataset = 's3dis'
                prediction_remap = SemanticConflictGenerator.SCANNET_TO_S3DIS_MAP
                cross_domain_target = SemanticConflictGenerator.CROSS_DOMAIN_SCANNET_TO_S3DIS_TARGET
                cross_domain_source = SemanticConflictGenerator.CROSS_DOMAIN_SCANNET_TO_S3DIS_SOURCE

                print(
                    f"  [Cross-Domain] Strategy: Evaluate in S3DIS label space, remap model predictions")
                print(
                    f"  [Cross-Domain] Target classes (geometry): {list(cross_domain_target.values())}")
                print(
                    f"  [Cross-Domain] Source classes (texture):  {list(cross_domain_source.values())}")
                print(f"  [Cross-Domain] Prediction remapping (ScanNet→S3DIS):")
                s3dis_classes = SemanticConflictGenerator.S3DIS_CLASSES
                scannet_classes = SemanticConflictGenerator.SCANNET_CLASSES
                for sn_id, s3_id in sorted(prediction_remap.items()):
                    print(f"    ScanNet[{sn_id}]={scannet_classes[sn_id]} → "
                          f"S3DIS[{s3_id}]={s3dis_classes[s3_id]}")

            elif data_dataset_name == 'SCANNET' and model_dataset_name == 'S3DIS':
                args.dataset = 'scannet'
                prediction_remap = SemanticConflictGenerator.S3DIS_TO_SCANNET_MAP
                cross_domain_target = SemanticConflictGenerator.CROSS_DOMAIN_S3DIS_TO_SCANNET_TARGET
                cross_domain_source = SemanticConflictGenerator.CROSS_DOMAIN_S3DIS_TO_SCANNET_SOURCE

                print(
                    f"  [Cross-Domain] Strategy: Evaluate in ScanNet label space, remap model predictions")
                print(
                    f"  [Cross-Domain] Target classes (geometry): {list(cross_domain_target.values())}")
                print(
                    f"  [Cross-Domain] Source classes (texture):  {list(cross_domain_source.values())}")
                print(f"  [Cross-Domain] Prediction remapping (S3DIS→ScanNet):")
                s3dis_classes = SemanticConflictGenerator.S3DIS_CLASSES
                scannet_classes = SemanticConflictGenerator.SCANNET_CLASSES
                for s3_id, sn_id in sorted(prediction_remap.items()):
                    print(f"    S3DIS[{s3_id}]={s3dis_classes[s3_id]} → "
                          f"ScanNet[{sn_id}]={scannet_classes[sn_id]}")

            else:
                print(f"  [Cross-Domain] WARNING: No mapping defined for "
                      f"{data_dataset_name}→{model_dataset_name}, using data's dataset")
                args.dataset = data_dataset_name.lower()
        else:
            print(f"  [Cross-Domain] Same dataset detected ({model_dataset_name}), "
                  f"no label mapping needed")

    # ============================================================================
    # ============================================================================
    if is_cross_domain:
        model_dataset_name = cfg.dataset.common.get('NAME', '').upper()
        data_dataset_name = data_cfg.dataset.common.get('NAME', '').upper()

        if model_dataset_name == 'SCANNET' and data_dataset_name == 'S3DIS':
            print(
                "\n  ✨ [Trick 1] Color Normalization Alignment (ScanNet model on S3DIS data)")
            print("     S3DIS std~0.18 → ScanNet std~0.69 (4x difference fixed!)")

            data_cfg.datatransforms.val = [
                'NumpyChromaticNormalize', 'PointsToTensor']

            model_dt = cfg.get('datatransforms', None)
            model_kwargs = model_dt.get('kwargs', {}) if model_dt else {}
            if 'color_mean' in model_kwargs:
                data_cfg.datatransforms.kwargs.color_mean = model_kwargs.color_mean
                data_cfg.datatransforms.kwargs.color_std = model_kwargs.color_std
                print(
                    f"     Using model's color_mean={list(model_kwargs.color_mean)}")
                print(
                    f"     Using model's color_std={list(model_kwargs.color_std)}")
            else:
                data_cfg.datatransforms.kwargs.color_mean = [
                    0.46259782, 0.46253258, 0.46253258]
                data_cfg.datatransforms.kwargs.color_std = [
                    0.693565, 0.6852543, 0.68061745]
                print("     Using ScanNet default color stats (fallback)")

            print(
                "  ✨ [Trick 2] Skip PointCloudXYZAlign (ScanNet model doesn't expect it)")

        elif model_dataset_name == 'S3DIS' and data_dataset_name == 'SCANNET':
            print(
                "\n  ✨ [Trick 1] Color Normalization Alignment (S3DIS model on ScanNet data)")
            print("     ScanNet std~0.69 → S3DIS std~0.18 (aligning to S3DIS model)")

            data_cfg.datatransforms.val = ['NumpyChromaticNormalize']

            model_dt = cfg.get('datatransforms', None)
            model_kwargs = model_dt.get('kwargs', {}) if model_dt else {}
            if 'color_mean' in model_kwargs:
                data_cfg.datatransforms.kwargs.color_mean = model_kwargs.color_mean
                data_cfg.datatransforms.kwargs.color_std = model_kwargs.color_std
                print(
                    f"     Using model's color_mean={list(model_kwargs.color_mean)}")
                print(
                    f"     Using model's color_std={list(model_kwargs.color_std)}")
            else:
                data_cfg.datatransforms.kwargs.color_mean = [
                    0.5136457, 0.49523646, 0.44921124]
                data_cfg.datatransforms.kwargs.color_std = [
                    0.18308958, 0.18415008, 0.19252081]
                print(
                    "     Using S3DIS default color stats (ChromaticNormalize defaults)")

            print(
                "  ✨ [Trick 2] ScanNet data doesn't use PointCloudXYZAlign (no change needed)")

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
        data_cfg.dataset,
        data_cfg.dataloader,
        datatransforms_cfg=data_cfg.datatransforms,
        split='val',
        distributed=False
    )
    print(f"  Validation samples: {len(val_loader.dataset)}")
    if is_cross_domain:
        print(
            f"  [Cross-Domain] Data source: {data_cfg.dataset.common.get('NAME', 'unknown')}")

    print(
        f"[Step 4] Initializing conflict generator for {args.dataset.upper()}...")
    generator_kwargs = dict(
        dataset=args.dataset,
        attack_ratio=args.attack_ratio,
        preserve_boundary=args.preserve_boundary,
        seed=42,
    )

    if args.all_classes:
        if args.dataset == 's3dis':
            exclude_from_target = {0, 1, 2}
            if is_cross_domain:
                mapped_s3dis_ids = set(
                    SemanticConflictGenerator.SCANNET_TO_S3DIS_MAP.values())
                target_classes = {i: name for i, name in enumerate(
                    SemanticConflictGenerator.S3DIS_CLASSES)
                    if i not in exclude_from_target and i in mapped_s3dis_ids}
                print(
                    f"  [Cross-Domain] Only targeting mapped S3DIS classes (shared with ScanNet)")
            else:
                target_classes = {i: name for i, name in enumerate(
                    SemanticConflictGenerator.S3DIS_CLASSES) if i not in exclude_from_target}
            source_classes = {0: 'ceiling', 1: 'floor', 2: 'wall'}
        else:  # scannet
            exclude_from_target = {0, 1}  # wall, floor
            if is_cross_domain:
                mapped_scannet_ids = set(
                    SemanticConflictGenerator.S3DIS_TO_SCANNET_MAP.values())
                target_classes = {i: name for i, name in enumerate(
                    SemanticConflictGenerator.SCANNET_CLASSES)
                    if i not in exclude_from_target and i in mapped_scannet_ids}
                print(
                    f"  [Cross-Domain] Only targeting mapped ScanNet classes (shared with S3DIS)")
            else:
                target_classes = {i: name for i, name in enumerate(
                    SemanticConflictGenerator.SCANNET_CLASSES) if i not in exclude_from_target}
            source_classes = {0: 'wall', 1: 'floor'}
        generator_kwargs['target_classes'] = target_classes
        generator_kwargs['source_classes'] = source_classes
        print(
            f"  [All Classes] Target classes ({len(target_classes)}): {list(target_classes.values())}")
        print(
            f"  [All Classes] Source classes ({len(source_classes)}): {list(source_classes.values())}")
    elif cross_domain_target is not None:
        generator_kwargs['target_classes'] = cross_domain_target
        if cross_domain_source is not None:
            generator_kwargs['source_classes'] = cross_domain_source

    generator = SemanticConflictGenerator(**generator_kwargs)

    if args.pretrained_path_baseline and args.pretrained_path_ours:
        if is_cross_domain:
            print(
                f"\n[Step 5] CROSS-DOMAIN COMPARISON MODE: Baseline vs Ours (STS)")
            print(f"  Model trained on: {cfg.dataset.common.get('NAME', 'unknown')} "
                  f"({cfg.get('num_classes', '?')} classes)")
            print(f"  Test data from:   {data_cfg.dataset.common.get('NAME', 'unknown')} "
                  f"({data_cfg.get('num_classes', '?')} classes)")
        else:
            print(
                f"\n[Step 5] COMPARISON MODE: Baseline vs Ours (STS) on {args.dataset.upper()}")

        print(f"\nLoading Baseline model...")
        model_baseline = load_model(cfg, args.pretrained_path_baseline, device)

        print(f"\nLoading Ours (STS) model...")
        model_ours = load_model(cfg, args.pretrained_path_ours, device)

        print(f"\n{'='*70}")
        print("Testing BASELINE...")
        print("="*70)
        evaluator_baseline = SemanticConflictEvaluator(
            dataset=args.dataset, prediction_remap=prediction_remap)
        metrics_baseline = evaluate_on_conflict_samples(
            model_baseline, val_loader, generator, evaluator_baseline,
            device, cfg, args.max_samples
        )
        evaluator_baseline.print_report("Baseline")

        val_loader = build_dataloader_from_cfg(
            args.batch_size,
            data_cfg.dataset,
            data_cfg.dataloader,
            datatransforms_cfg=data_cfg.datatransforms,
            split='val',
            distributed=False
        )

        generator_kwargs['seed'] = 42
        generator = SemanticConflictGenerator(**generator_kwargs)

        print(f"\n{'='*70}")
        print("Testing OURS (STS)...")
        print("="*70)
        evaluator_ours = SemanticConflictEvaluator(
            dataset=args.dataset, prediction_remap=prediction_remap)
        metrics_ours = evaluate_on_conflict_samples(
            model_ours, val_loader, generator, evaluator_ours,
            device, cfg, args.max_samples
        )
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

    else:
        print(f"\n[Step 5] SINGLE MODEL MODE on {args.dataset.upper()}")

        model = load_model(cfg, args.pretrained_path, device)

        evaluator = SemanticConflictEvaluator(
            dataset=args.dataset, prediction_remap=prediction_remap)
        metrics = evaluate_on_conflict_samples(
            model, val_loader, generator, evaluator,
            device, cfg, args.max_samples
        )
        evaluator.print_report("Model")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result_file = os.path.join(
            args.output_dir, 'semantic_conflict_results.txt')

        with open(result_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("🔬 SEMANTIC CONFLICT TEST (GOLD STANDARD)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Dataset: {args.dataset.upper()}\n")
            f.write(f"Model Config: {args.cfg}\n")
            if is_cross_domain:
                f.write(f"Data Config: {args.data_cfg}\n")
                f.write(f"Cross-Domain: Yes (model={cfg.dataset.common.get('NAME', '?')}"
                        f"({cfg.get('num_classes', '?')} classes) → "
                        f"data={data_cfg.dataset.common.get('NAME', '?')}"
                        f"({data_cfg.get('num_classes', '?')} classes))\n")
            else:
                f.write(f"Cross-Domain: No\n")
            f.write(f"GPU: {args.gpu}\n")
            f.write(f"Attack Ratio: {args.attack_ratio:.0%}\n\n")

            if args.pretrained_path_baseline and args.pretrained_path_ours:
                f.write(f"Baseline Model: {args.pretrained_path_baseline}\n")
                f.write(f"Ours (STS) Model: {args.pretrained_path_ours}\n\n")

                f.write("=" * 70 + "\n")
                f.write("🔬 Semantic Conflict Test Report: Baseline\n")
                f.write("=" * 70 + "\n\n")
                f.write(
                    f"📊 Global Metrics (on {metrics_baseline.get('total_attacked_points', 0):,} attacked points):\n")
                f.write(
                    f"   ✅ Texture Robustness (TR):     {metrics_baseline.get('texture_robustness', 0):.2%}\n")
                f.write(
                    f"   ❌ Texture Deception Rate (TDR): {metrics_baseline.get('texture_deception_rate', 0):.2%}\n")
                f.write(
                    f"   ❓ Other Predictions:            {metrics_baseline.get('other_prediction_rate', 0):.2%}\n")
                f.write(
                    f"   📈 mIoU on Attacked Classes:     {metrics_baseline.get('miou_attacked_classes', 0):.2%}\n\n")

                f.write("📋 Per-Class Breakdown:\n")
                f.write(
                    f"   {'Class':<20} {'Attacked':<12} {'Robustness':<12} {'Deception':<12}\n")
                f.write(f"   {'-'*56}\n")
                for cls_name, stats in metrics_baseline.get('per_class_robustness', {}).items():
                    f.write(f"   {cls_name:<20} {stats['attacked_points']:<12} "
                            f"{stats['robustness']:.2%}       {stats['deception_rate']:.2%}\n")
                f.write("=" * 70 + "\n\n")

                f.write("=" * 70 + "\n")
                f.write("🔬 Semantic Conflict Test Report: Ours (STS)\n")
                f.write("=" * 70 + "\n\n")
                f.write(
                    f"📊 Global Metrics (on {metrics_ours.get('total_attacked_points', 0):,} attacked points):\n")
                f.write(
                    f"   ✅ Texture Robustness (TR):     {metrics_ours.get('texture_robustness', 0):.2%}\n")
                f.write(
                    f"   ❌ Texture Deception Rate (TDR): {metrics_ours.get('texture_deception_rate', 0):.2%}\n")
                f.write(
                    f"   ❓ Other Predictions:            {metrics_ours.get('other_prediction_rate', 0):.2%}\n")
                f.write(
                    f"   📈 mIoU on Attacked Classes:     {metrics_ours.get('miou_attacked_classes', 0):.2%}\n\n")

                f.write("📋 Per-Class Breakdown:\n")
                f.write(
                    f"   {'Class':<20} {'Attacked':<12} {'Robustness':<12} {'Deception':<12}\n")
                f.write(f"   {'-'*56}\n")
                for cls_name, stats in metrics_ours.get('per_class_robustness', {}).items():
                    f.write(f"   {cls_name:<20} {stats['attacked_points']:<12} "
                            f"{stats['robustness']:.2%}       {stats['deception_rate']:.2%}\n")
                f.write("=" * 70 + "\n\n")

                f.write("🎯" * 35 + "\n")
                f.write("FINAL COMPARISON\n")
                f.write("🎯" * 35 + "\n\n")

                tr_baseline = metrics_baseline.get('texture_robustness', 0)
                tr_ours = metrics_ours.get('texture_robustness', 0)
                tdr_baseline = metrics_baseline.get(
                    'texture_deception_rate', 0)
                tdr_ours = metrics_ours.get('texture_deception_rate', 0)
                miou_baseline = metrics_baseline.get(
                    'miou_attacked_classes', 0)
                miou_ours = metrics_ours.get('miou_attacked_classes', 0)

                f.write(
                    f"{'Metric':<35} {'Baseline':<15} {'Ours (STS)':<15} {'Δ':<15}\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'Texture Robustness (TR) ↑':<35} {tr_baseline:<15.2%} {tr_ours:<15.2%} {tr_ours - tr_baseline:+.2%}\n")
                f.write(
                    f"{'Texture Deception Rate (TDR) ↓':<35} {tdr_baseline:<15.2%} {tdr_ours:<15.2%} {tdr_ours - tdr_baseline:+.2%}\n")
                f.write(
                    f"{'mIoU on Attacked Classes ↑':<35} {miou_baseline:<15.2%} {miou_ours:<15.2%} {miou_ours - miou_baseline:+.2%}\n\n")

                f.write("=" * 80 + "\n")
                if tr_ours > tr_baseline + 0.05:
                    f.write(
                        "✅ SUCCESS: STS significantly improves texture robustness!\n")
                elif tr_ours > tr_baseline:
                    f.write(
                        "⚠️  PARTIAL: STS shows improvement, consider longer training.\n")
                else:
                    f.write(
                        "❌ UNEXPECTED: STS did not improve robustness. Check training.\n")

            else:
                f.write(f"Model: {args.pretrained_path}\n\n")

                f.write("=" * 70 + "\n")
                f.write("🔬 Semantic Conflict Test Report: Model\n")
                f.write("=" * 70 + "\n\n")
                f.write(
                    f"📊 Global Metrics (on {metrics.get('total_attacked_points', 0):,} attacked points):\n")
                f.write(
                    f"   ✅ Texture Robustness (TR):     {metrics.get('texture_robustness', 0):.2%}\n")
                f.write(
                    f"   ❌ Texture Deception Rate (TDR): {metrics.get('texture_deception_rate', 0):.2%}\n")
                f.write(
                    f"   ❓ Other Predictions:            {metrics.get('other_prediction_rate', 0):.2%}\n")
                f.write(
                    f"   📈 mIoU on Attacked Classes:     {metrics.get('miou_attacked_classes', 0):.2%}\n\n")

                f.write("📋 Per-Class Breakdown:\n")
                f.write(
                    f"   {'Class':<20} {'Attacked':<12} {'Robustness':<12} {'Deception':<12}\n")
                f.write(f"   {'-'*56}\n")
                for cls_name, stats in metrics.get('per_class_robustness', {}).items():
                    f.write(f"   {cls_name:<20} {stats['attacked_points']:<12} "
                            f"{stats['robustness']:.2%}       {stats['deception_rate']:.2%}\n")
                f.write("=" * 70 + "\n")

        print(f"\n📁 Results saved to: {result_file}")

    print("\n✅ Semantic Conflict Test Complete!")


if __name__ == "__main__":
    main()
