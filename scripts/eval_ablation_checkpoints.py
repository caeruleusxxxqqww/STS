#!/usr/bin/env python3
"""
    python scripts/eval_ablation_checkpoints.py \
        --cfg cfgs/ablation/exp0-baseline.yaml \
        --checkpoint path/to/checkpoint.pth

    python scripts/eval_ablation_checkpoints.py --batch \
        --checkpoint_dir /path/to/checkpoints \
        --output results/ablation_eval.csv
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_checkpoints(checkpoint_dir, pattern="**/ckpt_best.pth"):
    """

    Args:

    Returns:
        dict: {exp_name: checkpoint_path}
    """
    checkpoints = {}

    search_path = os.path.join(checkpoint_dir, pattern)
    for ckpt_path in glob.glob(search_path, recursive=True):
        parent_dir = os.path.dirname(ckpt_path)
        exp_name = os.path.basename(parent_dir)

        if exp_name.startswith('exp') or any(key in exp_name.lower() for key in ['baseline', 'gtigu', 'tsi', 'sts', 'full']):
            checkpoints[exp_name] = ckpt_path

    return checkpoints


def get_config_for_checkpoint(checkpoint_path, cfg_dir="cfgs/ablation"):
    """

    Args:

    Returns:
    """
    parent_dir = os.path.dirname(checkpoint_path)
    exp_name = os.path.basename(parent_dir)

    possible_cfgs = [
        os.path.join(cfg_dir, f"{exp_name}.yaml"),
        os.path.join(cfg_dir, f"{exp_name.lower()}.yaml"),
    ]

    name_mapping = {
        'baseline': 'exp0-baseline',
        'gtigu': 'exp1-gtigu-only',
        'tsi': 'exp2-tsi-only',
        'sts': 'exp3-sts-only',
        'gtigu_tsi': 'exp4-gtigu-tsi',
        'gtigu_sts': 'exp5-gtigu-sts',
        'tsi_sts': 'exp6-tsi-sts',
        'full': 'exp7-full',
    }

    for key, cfg_name in name_mapping.items():
        if key in exp_name.lower():
            possible_cfgs.append(os.path.join(cfg_dir, f"{cfg_name}.yaml"))

    for cfg_path in possible_cfgs:
        if os.path.exists(cfg_path):
            return cfg_path

    return None


def run_evaluation(cfg_path, checkpoint_path, output_dir=None):
    """

    Args:

    Returns:
    """
    import subprocess
    import re

    cmd = [
        "python", "examples/segmentation/main.py",
        "--cfg", cfg_path,
        "--mode", "test",
        "--pretrained_path", checkpoint_path,
    ]

    print(f"\n{'='*60}")
    print(f"📊 : {os.path.basename(cfg_path)}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        output = result.stdout + result.stderr

        metrics = {
            'miou': None,
            'macc': None,
            'oa': None,
        }

        patterns = [
            r'test_miou:\s*([\d.]+).*test_macc:\s*([\d.]+).*test_oa:\s*([\d.]+)',
            r'val_miou:\s*([\d.]+).*val_macc:\s*([\d.]+).*val_oa:\s*([\d.]+)',
            r'mIoU:\s*([\d.]+).*mAcc:\s*([\d.]+).*OA:\s*([\d.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics['miou'] = float(match.group(1))
                metrics['macc'] = float(match.group(2))
                metrics['oa'] = float(match.group(3))
                break

        if metrics['miou'] is not None:
            print(f"   ✅ mIoU: {metrics['miou']:.2f}%")
            print(f"   ✅ mAcc: {metrics['macc']:.2f}%")
            print(f"   ✅ OA: {metrics['oa']:.2f}%")
        else:
            print(f"   ⚠️ ")
            print(f"   : {output[:500]}...")

        return metrics

    except Exception as e:
        print(f"   ❌ : {e}")
        return {'miou': None, 'macc': None, 'oa': None, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description=' Checkpoint ')
    parser.add_argument('--cfg', type=str, default=None,
                        help=' ()')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint  ()')
    parser.add_argument('--batch', action='store_true',
                        help='')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint  ()')
    parser.add_argument('--output', type=str, default=None,
                        help='')
    args = parser.parse_args()

    print("=" * 70)
    print("📊  Checkpoint ")
    print("=" * 70)
    print(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    if args.batch and args.checkpoint_dir:
        print(f"\n🔍  checkpoint: {args.checkpoint_dir}")
        checkpoints = find_checkpoints(args.checkpoint_dir)

        if not checkpoints:
            print("❌  checkpoint")
            return 1

        print(f"    {len(checkpoints)}  checkpoint:")
        for name, path in checkpoints.items():
            print(f"   - {name}: {path}")

        for exp_name, ckpt_path in checkpoints.items():
            cfg_path = get_config_for_checkpoint(ckpt_path)
            if cfg_path:
                metrics = run_evaluation(cfg_path, ckpt_path)
                results[exp_name] = metrics
            else:
                print(f"   ⚠️  {exp_name} ")

    elif args.cfg and args.checkpoint:
        exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
        metrics = run_evaluation(args.cfg, args.checkpoint)
        results[exp_name] = metrics

    else:
        parser.print_help()
        return 1

    print("\n" + "=" * 70)
    print("📋 ")
    print("=" * 70)
    print(f"{'':<20} {'mIoU (%)':<12} {'mAcc (%)':<12} {'OA (%)':<12}")
    print("-" * 60)

    for exp_name, metrics in results.items():
        miou = f"{metrics['miou']:.2f}" if metrics.get('miou') else '-'
        macc = f"{metrics['macc']:.2f}" if metrics.get('macc') else '-'
        oa = f"{metrics['oa']:.2f}" if metrics.get('oa') else '-'
        print(f"{exp_name:<20} {miou:<12} {macc:<12} {oa:<12}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 : {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
