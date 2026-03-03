#!/usr/bin/env python3
"""
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_YYYYMMDD_HHMMSS
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_YYYYMMDD_HHMMSS --output results.csv
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_YYYYMMDD_HHMMSS --latex
"""

import os
import re
import glob
import argparse
import json
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️ pandas ")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("⚠️ tabulate ")


# =========================================================================
# =========================================================================
EXPERIMENTS = {
    'exp0-baseline': {
        'name': 'Baseline',
        'gt_igu': False,
        'tsi': False,
        'sts': False,
        'description': '',
    },
    'exp1-gtigu-only': {
        'name': 'GT-IGU',
        'gt_igu': True,
        'tsi': False,
        'sts': False,
        'description': '-',
    },
    'exp2-tsi-only': {
        'name': 'TSI',
        'gt_igu': False,
        'tsi': True,
        'sts': False,
        'description': '',
    },
    'exp3-sts-only': {
        'name': 'STS',
        'gt_igu': False,
        'tsi': False,
        'sts': True,
        'description': '',
    },
    'exp4-gtigu-tsi': {
        'name': 'GT-IGU+TSI',
        'gt_igu': True,
        'tsi': True,
        'sts': False,
        'description': '+',
    },
    'exp5-gtigu-sts': {
        'name': 'GT-IGU+STS',
        'gt_igu': True,
        'tsi': False,
        'sts': True,
        'description': '+',
    },
    'exp6-tsi-sts': {
        'name': 'TSI+STS',
        'gt_igu': False,
        'tsi': True,
        'sts': True,
        'description': '',
    },
    'exp7-full': {
        'name': 'Full (Ours)',
        'gt_igu': True,
        'tsi': True,
        'sts': True,
        'description': '',
    },
}


def parse_log_file(log_path):
    """

    Args:

    Returns:
    """
    results = {
        'best_epoch': None,
        'best_miou': None,
        'best_macc': None,
        'best_oa': None,
        'final_miou': None,
        'final_macc': None,
        'final_oa': None,
        'per_class_iou': None,
        'training_time': None,
        'log_exists': False,
    }

    if not os.path.exists(log_path):
        return results

    results['log_exists'] = True

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # "Best ckpt @E50, val_oa, val_macc, val_miou: 87.23 72.45 65.12"
    best_pattern = r'Best.*@E(\d+).*val_oa.*val_macc.*val_miou:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    best_match = re.search(best_pattern, content, re.IGNORECASE)
    if best_match:
        results['best_epoch'] = int(best_match.group(1))
        results['best_oa'] = float(best_match.group(2))
        results['best_macc'] = float(best_match.group(3))
        results['best_miou'] = float(best_match.group(4))

    # "mIoU: 65.12, mAcc: 72.45, OA: 87.23"
    alt_pattern = r'mIoU:\s*([\d.]+).*mAcc:\s*([\d.]+).*OA:\s*([\d.]+)'
    for match in re.finditer(alt_pattern, content, re.IGNORECASE):
        results['final_miou'] = float(match.group(1))
        results['final_macc'] = float(match.group(2))
        results['final_oa'] = float(match.group(3))

    # "val_miou: 65.12"
    miou_pattern = r'val_miou:\s*([\d.]+)'
    for match in re.finditer(miou_pattern, content, re.IGNORECASE):
        results['final_miou'] = float(match.group(1))

    macc_pattern = r'val_macc:\s*([\d.]+)'
    for match in re.finditer(macc_pattern, content, re.IGNORECASE):
        results['final_macc'] = float(match.group(1))

    oa_pattern = r'val_oa:\s*([\d.]+)'
    for match in re.finditer(oa_pattern, content, re.IGNORECASE):
        results['final_oa'] = float(match.group(1))

    # "iou per cls is: [65.12 72.45 ...]"
    iou_pattern = r'iou per cls.*:\s*\[([\d.\s]+)\]'
    iou_match = re.search(iou_pattern, content, re.IGNORECASE)
    if iou_match:
        iou_str = iou_match.group(1)
        results['per_class_iou'] = [float(x) for x in iou_str.split()]

    time_pattern = r'Total training time:\s*([\d.]+)\s*hours?'
    time_match = re.search(time_pattern, content, re.IGNORECASE)
    if time_match:
        results['training_time'] = float(time_match.group(1))

    return results


def collect_ablation_results(log_dir, checkpoint_dir=None):
    """

    Args:

    Returns:
    """
    results = []

    for exp_id, exp_info in EXPERIMENTS.items():
        log_path = os.path.join(log_dir, f'{exp_id}.log')
        metrics = parse_log_file(log_path)

        miou = metrics.get('best_miou') or metrics.get('final_miou')
        macc = metrics.get('best_macc') or metrics.get('final_macc')
        oa = metrics.get('best_oa') or metrics.get('final_oa')

        results.append({
            'exp_id': exp_id,
            'name': exp_info['name'],
            'gt_igu': exp_info['gt_igu'],
            'tsi': exp_info['tsi'],
            'sts': exp_info['sts'],
            'miou': miou,
            'macc': macc,
            'oa': oa,
            'best_epoch': metrics.get('best_epoch'),
            'per_class_iou': metrics.get('per_class_iou'),
            'training_time': metrics.get('training_time'),
            'log_exists': metrics.get('log_exists', False),
        })

    return results


def calculate_improvements(results):
    """

    Args:

    Returns:
    """
    baseline = None
    for r in results:
        if r['exp_id'] == 'exp0-baseline':
            baseline = r
            break

    if baseline is None or baseline['miou'] is None:
        print("⚠️ Baseline ")
        for r in results:
            r['delta_miou'] = None
            r['delta_macc'] = None
            r['delta_oa'] = None
        return results

    for r in results:
        if r['miou'] is not None and baseline['miou'] is not None:
            r['delta_miou'] = r['miou'] - baseline['miou']
        else:
            r['delta_miou'] = None

        if r['macc'] is not None and baseline['macc'] is not None:
            r['delta_macc'] = r['macc'] - baseline['macc']
        else:
            r['delta_macc'] = None

        if r['oa'] is not None and baseline['oa'] is not None:
            r['delta_oa'] = r['oa'] - baseline['oa']
        else:
            r['delta_oa'] = None

    return results


def format_value(value, fmt='.2f', show_plus=False):
    """"""
    if value is None:
        return '-'
    if show_plus and value > 0:
        return f'+{value:{fmt}}'
    return f'{value:{fmt}}'


def print_results_table(results):
    """"""

    if HAS_PANDAS and HAS_TABULATE:
        data = []
        for r in results:
            data.append({
                'Experiment': r['name'],
                'GT-IGU': '✓' if r['gt_igu'] else '',
                'TSI': '✓' if r['tsi'] else '',
                'STS': '✓' if r['sts'] else '',
                'mIoU (%)': format_value(r['miou']),
                'mAcc (%)': format_value(r['macc']),
                'OA (%)': format_value(r['oa']),
                'Δ mIoU': format_value(r.get('delta_miou'), show_plus=True),
            })

        df = pd.DataFrame(data)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        return df
    else:
        print("\n{:<15} {:^8} {:^8} {:^8} {:>10} {:>10} {:>10} {:>10}".format(
            'Experiment', 'GT-IGU', 'TSI', 'STS', 'mIoU(%)', 'mAcc(%)', 'OA(%)', 'Δ mIoU'))
        print("-" * 90)

        for r in results:
            print("{:<15} {:^8} {:^8} {:^8} {:>10} {:>10} {:>10} {:>10}".format(
                r['name'],
                '✓' if r['gt_igu'] else '',
                '✓' if r['tsi'] else '',
                '✓' if r['sts'] else '',
                format_value(r['miou']),
                format_value(r['macc']),
                format_value(r['oa']),
                format_value(r.get('delta_miou'), show_plus=True),
            ))
        return None


def generate_latex_table(results):
    """ LaTeX  ()"""

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation study on S3DIS Area-5. }")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\begin{tabular}{l|ccc|ccc}")
    latex.append(r"\toprule")
    latex.append(
        r"Method & GT-IGU & TSI & STS & mIoU (\%) & mAcc (\%) & OA (\%) \\")
    latex.append(r"\midrule")

    for r in results:
        gt_igu = r"\checkmark" if r['gt_igu'] else ""
        tsi = r"\checkmark" if r['tsi'] else ""
        sts = r"\checkmark" if r['sts'] else ""

        miou = format_value(r['miou'])
        macc = format_value(r['macc'])
        oa = format_value(r['oa'])

        if r['exp_id'] == 'exp7-full':
            miou = r"\textbf{" + miou + "}"
            macc = r"\textbf{" + macc + "}"
            oa = r"\textbf{" + oa + "}"

        latex.append(
            f"{r['name']} & {gt_igu} & {tsi} & {sts} & {miou} & {macc} & {oa} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def analyze_module_contribution(results):
    """"""

    print("\n" + "=" * 70)
    print("📈 ")
    print("=" * 70)

    baseline = None
    for r in results:
        if r['exp_id'] == 'exp0-baseline':
            baseline = r
            break

    if baseline is None or baseline['miou'] is None:
        print("⚠️ Baseline ")
        return

    print(f"\n📌 Baseline mIoU: {format_value(baseline['miou'])}%")

    print("\n🔹  ( Baseline):")
    single_modules = ['exp1-gtigu-only', 'exp2-tsi-only', 'exp3-sts-only']
    for exp_id in single_modules:
        r = next((x for x in results if x['exp_id'] == exp_id), None)
        if r and r['delta_miou'] is not None:
            print(
                f"   {r['name']:<15}: Δ mIoU = {format_value(r['delta_miou'], show_plus=True)}%")

    print("\n🔹 :")
    dual_modules = ['exp4-gtigu-tsi', 'exp5-gtigu-sts', 'exp6-tsi-sts']
    for exp_id in dual_modules:
        r = next((x for x in results if x['exp_id'] == exp_id), None)
        if r and r['delta_miou'] is not None:
            print(
                f"   {r['name']:<15}: Δ mIoU = {format_value(r['delta_miou'], show_plus=True)}%")

    print("\n🔹 :")
    full = next((x for x in results if x['exp_id'] == 'exp7-full'), None)
    if full and full['delta_miou'] is not None:
        print(
            f"   {full['name']:<15}: Δ mIoU = {format_value(full['delta_miou'], show_plus=True)}%")

    print("\n🔹 :")
    gt_igu = next(
        (x for x in results if x['exp_id'] == 'exp1-gtigu-only'), None)
    tsi = next((x for x in results if x['exp_id'] == 'exp2-tsi-only'), None)
    sts = next((x for x in results if x['exp_id'] == 'exp3-sts-only'), None)

    if all(x and x['delta_miou'] is not None for x in [gt_igu, tsi, sts, full]):
        sum_individual = gt_igu['delta_miou'] + \
            tsi['delta_miou'] + sts['delta_miou']
        synergy = full['delta_miou'] - sum_individual
        print(f"   : {format_value(sum_individual, show_plus=True)}%")
        print(
            f"   :   {format_value(full['delta_miou'], show_plus=True)}%")
        print(f"   :       {format_value(synergy, show_plus=True)}%")

        if synergy > 0:
            print("   ✅ ")
        elif synergy < 0:
            print("   ⚠️ ")
        else:
            print("   ➡️ ")


def find_best_configuration(results):
    """"""

    print("\n" + "=" * 70)
    print("🏆 ")
    print("=" * 70)

    valid_results = [r for r in results if r['miou'] is not None]

    if not valid_results:
        print("⚠️ ")
        return

    best = max(valid_results, key=lambda x: x['miou'])
    print(f"\n✅ : {best['name']}")
    print(f"   mIoU:  {format_value(best['miou'])}%")
    print(f"   mAcc:  {format_value(best['macc'])}%")
    print(f"   OA:    {format_value(best['oa'])}%")

    if best.get('delta_miou') is not None:
        print(
            f"    Baseline : {format_value(best['delta_miou'], show_plus=True)}%")


def save_results(results, output_path):
    """"""

    if output_path.endswith('.csv'):
        if HAS_PANDAS:
            data = []
            for r in results:
                data.append({
                    'exp_id': r['exp_id'],
                    'name': r['name'],
                    'gt_igu': r['gt_igu'],
                    'tsi': r['tsi'],
                    'sts': r['sts'],
                    'miou': r['miou'],
                    'macc': r['macc'],
                    'oa': r['oa'],
                    'delta_miou': r.get('delta_miou'),
                    'delta_macc': r.get('delta_macc'),
                    'delta_oa': r.get('delta_oa'),
                    'best_epoch': r.get('best_epoch'),
                })
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"\n📁 : {output_path}")
        else:
            print("⚠️  pandas  CSV ")

    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_20240101_120000
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_20240101_120000 --output results.csv
    python scripts/analyze_ablation_results.py --log_dir logs/ablation_20240101_120000 --latex
        """
    )
    parser.add_argument('--log_dir', type=str, required=True,
                        help='')
    parser.add_argument('--output', type=str, default=None,
                        help=' (.csv  .json)')
    parser.add_argument('--latex', action='store_true',
                        help=' LaTeX ')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='')
    args = parser.parse_args()

    print("=" * 70)
    print("📊 ")
    print("=" * 70)
    print(f": {args.log_dir}")
    print(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not os.path.exists(args.log_dir):
        print(f"❌ : {args.log_dir}")
        return 1

    results = collect_ablation_results(args.log_dir)

    valid_count = sum(1 for r in results if r['log_exists'])
    print(f" {valid_count}/{len(results)} ")

    if valid_count == 0:
        print("⚠️ ")
        print(f"   :")
        for exp_id in EXPERIMENTS.keys():
            print(f"   - {exp_id}.log")
        return 1

    results = calculate_improvements(results)

    print("\n" + "=" * 70)
    print("📋 ")
    print("=" * 70)
    print_results_table(results)

    analyze_module_contribution(results)

    find_best_configuration(results)

    if args.latex:
        print("\n" + "=" * 70)
        print("📄 LaTeX ")
        print("=" * 70)
        latex_code = generate_latex_table(results)
        print(latex_code)

    if args.output:
        save_results(results, args.output)

    print("\n" + "=" * 70)
    print("✅ ")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
