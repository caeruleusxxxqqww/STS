#!/usr/bin/env python3
"""
📷 Visualization Column Cropper & Stitcher


    python scripts/crop_visualization_columns.py \
        --input visualizations/conflict_test/conflict_test_sample_0001.png \
        --output-dir visualizations/cropped

    python scripts/crop_visualization_columns.py \
        --input visualizations/conflict_test/conflict_test_sample_0001.png \
        --output-dir visualizations/cropped \
        --stitch --gap 5

    python scripts/crop_visualization_columns.py \
        --input-dir visualizations/conflict_test \
        --output-dir visualizations/cropped \
        --stitch --gap 5

Author: Generated for STS Paper
"""

import argparse
import os
import glob
from PIL import Image
import numpy as np


DEFAULT_COLUMN_NAMES = [
    'clean_input',
    'corrupted_input',
    'baseline_prediction',
    'ours_prediction',
    'ground_truth'
]

FOUR_COLUMN_NAMES = [
    'input',
    'baseline_prediction',
    'ours_prediction',
    'ground_truth'
]


def detect_column_boundaries(img_array, num_columns=5, margin_threshold=250):
    """


    Args:

    Returns:
    """
    H, W = img_array.shape[:2]

    column_brightness = np.mean(img_array, axis=(0, 2))  # (W,)

    content_mask = column_brightness < margin_threshold

    if not np.any(content_mask):
        col_width = W // num_columns
        return [(i * col_width, (i + 1) * col_width) for i in range(num_columns)]

    transitions = np.diff(content_mask.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if content_mask[0]:
        starts = np.concatenate([[0], starts])
    if content_mask[-1]:
        ends = np.concatenate([ends, [W]])

    min_len = min(len(starts), len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]

    min_column_width = W // (num_columns * 2)
    boundaries = []
    i = 0
    while i < len(starts):
        start = starts[i]
        end = ends[i]

        while i + 1 < len(starts) and starts[i + 1] - ends[i] < min_column_width:
            i += 1
            end = ends[i]

        if end - start >= min_column_width:
            boundaries.append((start, end))
        i += 1

    if len(boundaries) != num_columns:
        print(
            f"[Warning] Detected {len(boundaries)} columns, expected {num_columns}. Using uniform split.")
        col_width = W // num_columns
        boundaries = [(i * col_width, (i + 1) * col_width)
                      for i in range(num_columns)]

    return boundaries


def crop_columns(image_path, output_dir, num_columns=5, column_names=None,
                 padding=0, auto_detect=True):
    """

    Args:

    Returns:
    """
    if column_names is None:
        if num_columns == 5:
            column_names = DEFAULT_COLUMN_NAMES
        elif num_columns == 4:
            column_names = FOUR_COLUMN_NAMES
        else:
            column_names = [f'column_{i+1}' for i in range(num_columns)]

    img = Image.open(image_path)
    img_array = np.array(img)

    H, W = img_array.shape[:2]
    print(f"[Info] Image size: {W} x {H}")

    if auto_detect:
        boundaries = detect_column_boundaries(img_array, num_columns)
        print(f"[Info] Detected column boundaries: {boundaries}")
    else:
        col_width = W // num_columns
        boundaries = [(i * col_width, (i + 1) * col_width)
                      for i in range(num_columns)]

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    output_paths = []
    for i, (start, end) in enumerate(boundaries):
        start = max(0, start - padding)
        end = min(W, end + padding)

        cropped = img.crop((start, 0, end, H))

        col_name = column_names[i] if i < len(
            column_names) else f'column_{i+1}'
        output_path = os.path.join(output_dir, f'{base_name}_{col_name}.png')
        cropped.save(output_path, 'PNG')
        output_paths.append(output_path)
        print(f"  ✅ Saved: {output_path} ({end - start} x {H})")

    return output_paths


def crop_with_fixed_ratios(image_path, output_dir, column_ratios=None, column_names=None):
    """

    Args:
    """
    if column_ratios is None:
        column_ratios = [0.2, 0.2, 0.2, 0.2, 0.2]

    if column_names is None:
        if len(column_ratios) == 5:
            column_names = DEFAULT_COLUMN_NAMES
        elif len(column_ratios) == 4:
            column_names = FOUR_COLUMN_NAMES
        else:
            column_names = [f'column_{i+1}' for i in range(len(column_ratios))]

    img = Image.open(image_path)
    W, H = img.size

    boundaries = []
    x = 0
    for ratio in column_ratios:
        col_width = int(W * ratio)
        boundaries.append((x, x + col_width))
        x += col_width

    if boundaries:
        boundaries[-1] = (boundaries[-1][0], W)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    output_paths = []
    for i, (start, end) in enumerate(boundaries):
        cropped = img.crop((start, 0, end, H))
        col_name = column_names[i] if i < len(
            column_names) else f'column_{i+1}'
        output_path = os.path.join(output_dir, f'{base_name}_{col_name}.png')
        cropped.save(output_path, 'PNG')
        output_paths.append(output_path)
        print(f"  ✅ Saved: {output_path} ({end - start} x {H})")

    return output_paths


def remove_title_area(image_path, output_path=None, title_height_ratio=0.05):
    """

    Args:

    Returns:
    """
    img = Image.open(image_path)
    W, H = img.size

    title_height = int(H * title_height_ratio)

    cropped = img.crop((0, title_height, W, H))

    if output_path is None:
        output_path = image_path

    cropped.save(output_path, 'PNG')
    print(f"  ✅ Removed title ({title_height}px), saved: {output_path}")

    return output_path


def trim_whitespace(img, threshold=250, margin=2):
    """

    Args:
        img: PIL Image

    Returns:
    """
    img_array = np.array(img)

    if len(img_array.shape) == 3:
        is_white = np.all(img_array > threshold, axis=2)
    else:
        is_white = img_array > threshold

    non_white_rows = np.where(~np.all(is_white, axis=1))[0]
    non_white_cols = np.where(~np.all(is_white, axis=0))[0]

    if len(non_white_rows) == 0 or len(non_white_cols) == 0:
        return img

    top = max(0, non_white_rows[0] - margin)
    bottom = min(img_array.shape[0], non_white_rows[-1] + 1 + margin)
    left = max(0, non_white_cols[0] - margin)
    right = min(img_array.shape[1], non_white_cols[-1] + 1 + margin)

    return img.crop((left, top, right, bottom))


def stitch_images_horizontal(image_paths, output_path, gap=5, background_color=(255, 255, 255),
                             trim_borders=True, trim_threshold=250):
    """

    Args:

    Returns:
    """
    if not image_paths:
        print("[Warning] No images to stitch")
        return None

    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path)
            if trim_borders:
                original_size = img.size
                img = trim_whitespace(img, trim_threshold)
                if img.size != original_size:
                    print(f"    Trimmed: {original_size} -> {img.size}")
            images.append(img)
        else:
            print(f"[Warning] Image not found: {path}")

    if not images:
        print("[Error] No valid images found")
        return None

    max_height = max(img.size[1] for img in images)
    total_width = sum(img.size[0] for img in images) + gap * (len(images) - 1)

    print(f"  [Info] Final stitched size: {total_width} x {max_height}")

    stitched = Image.new('RGB', (total_width, max_height), background_color)

    x_offset = 0
    for img in images:
        y_offset = (max_height - img.size[1]) // 2
        stitched.paste(img, (x_offset, y_offset))
        x_offset += img.size[0] + gap

    stitched.save(output_path, 'PNG')
    print(
        f"  ✅ Stitched image saved: {output_path} ({total_width} x {max_height})")

    return output_path


def crop_and_stitch(image_path, output_dir, num_columns=5, column_names=None,
                    padding=0, auto_detect=True, gap=5, keep_individual=True,
                    trim_borders=True):
    """

    Args:

    Returns:
    """
    cropped_paths = crop_columns(
        image_path, output_dir, num_columns, column_names, padding, auto_detect
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    stitched_path = os.path.join(output_dir, f'{base_name}_stitched.png')
    stitch_images_horizontal(cropped_paths, stitched_path,
                             gap, trim_borders=trim_borders)

    if not keep_individual:
        for path in cropped_paths:
            if os.path.exists(path):
                os.remove(path)
        print(f"  🗑️  Removed individual column images")

    return stitched_path


def batch_crop_columns(input_dir, output_dir, pattern='*.png', num_columns=5,
                       column_names=None, auto_detect=True, remove_title=False,
                       stitch=False, gap=5, keep_individual=True, trim_borders=True):
    """

    Args:
    """
    search_pattern = os.path.join(input_dir, pattern)
    image_files = glob.glob(search_pattern)

    if not image_files:
        print(f"[Warning] No files found matching: {search_pattern}")
        return

    print(f"[Info] Found {len(image_files)} images to process")

    for i, image_path in enumerate(sorted(image_files)):
        print(
            f"\n[{i+1}/{len(image_files)}] Processing: {os.path.basename(image_path)}")

        try:
            if remove_title:
                temp_path = image_path.replace('.png', '_notitle.png')
                remove_title_area(image_path, temp_path)
                process_path = temp_path
            else:
                process_path = image_path

            if stitch:
                crop_and_stitch(
                    process_path, output_dir, num_columns, column_names,
                    auto_detect=auto_detect, gap=gap, keep_individual=keep_individual,
                    trim_borders=trim_borders
                )
            else:
                crop_columns(process_path, output_dir, num_columns, column_names,
                             auto_detect=auto_detect)

            if remove_title and os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"[Error] Failed to process {image_path}: {e}")
            continue

    print(f"\n✅ Batch processing complete. Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Crop visualization columns from images and optionally stitch them')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                             help='Path to input image')
    input_group.add_argument('--input-dir', type=str,
                             help='Directory containing input images')

    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for cropped images')

    parser.add_argument('--num-columns', type=int, default=5,
                        help='Number of columns (default: 5)')
    parser.add_argument('--column-names', type=str, nargs='+', default=None,
                        help='Custom column names')
    parser.add_argument('--no-auto-detect', action='store_true',
                        help='Disable auto-detection, use uniform split')
    parser.add_argument('--padding', type=int, default=0,
                        help='Extra padding around each column (pixels)')
    parser.add_argument('--remove-title', action='store_true',
                        help='Remove title area from top of image')
    parser.add_argument('--title-height', type=float, default=0.05,
                        help='Title height ratio (default: 0.05)')

    parser.add_argument('--stitch', action='store_true',
                        help='Stitch cropped columns into a single row image')
    parser.add_argument('--gap', type=int, default=5,
                        help='Gap between columns when stitching (pixels, default: 5)')
    parser.add_argument('--no-keep-individual', action='store_true',
                        help='Do not keep individual column images after stitching')
    parser.add_argument('--no-trim', action='store_true',
                        help='Do not trim white borders from each column before stitching')

    parser.add_argument('--pattern', type=str, default='*.png',
                        help='File pattern for batch processing (default: *.png)')

    args = parser.parse_args()

    if args.input:
        print(f"📷 Processing: {args.input}")

        if args.remove_title:
            temp_path = args.input.replace('.png', '_notitle_temp.png')
            remove_title_area(args.input, temp_path, args.title_height)
            process_path = temp_path
        else:
            process_path = args.input

        if args.stitch:
            crop_and_stitch(
                process_path,
                args.output_dir,
                args.num_columns,
                args.column_names,
                args.padding,
                auto_detect=not args.no_auto_detect,
                gap=args.gap,
                keep_individual=not args.no_keep_individual,
                trim_borders=not args.no_trim
            )
        else:
            crop_columns(
                process_path,
                args.output_dir,
                args.num_columns,
                args.column_names,
                args.padding,
                auto_detect=not args.no_auto_detect
            )

        if args.remove_title and 'temp_path' in dir() and os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"\n✅ Done! Output saved to: {args.output_dir}")

    else:
        print(f"📷 Batch processing: {args.input_dir}")
        batch_crop_columns(
            args.input_dir,
            args.output_dir,
            args.pattern,
            args.num_columns,
            args.column_names,
            auto_detect=not args.no_auto_detect,
            remove_title=args.remove_title,
            stitch=args.stitch,
            gap=args.gap,
            keep_individual=not args.no_keep_individual,
            trim_borders=not args.no_trim
        )


if __name__ == '__main__':
    main()
