#!/usr/bin/env python3
"""

    python scripts/preprocess_scannet.py --data_root /path/to/scannet_data
"""

import os
import json
import glob
import torch
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# ============================================================
# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util_3d.py

SCANNET_VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

LABEL_NAME_TO_NYU40 = {
    'wall': 1, 'floor': 2, 'cabinet': 3, 'bed': 4, 'chair': 5,
    'sofa': 6, 'table': 7, 'door': 8, 'window': 9, 'bookshelf': 10,
    'picture': 11, 'counter': 12, 'blinds': 13, 'desk': 14, 'shelves': 15,
    'curtain': 16, 'dresser': 17, 'pillow': 18, 'mirror': 19, 'floor mat': 20,
    'clothes': 21, 'ceiling': 22, 'books': 23, 'refrigerator': 24, 'television': 25,
    'paper': 26, 'towel': 27, 'shower curtain': 28, 'box': 29, 'whiteboard': 30,
    'person': 31, 'nightstand': 32, 'toilet': 33, 'sink': 34, 'lamp': 35,
    'bathtub': 36, 'bag': 37, 'otherstructure': 38, 'otherfurniture': 39, 'otherprop': 40,
    'kitchen counter': 12, 'kitchen cabinets': 3, 'kitchen cabinet': 3,
    'shower': 28, 'shower walls': 1, 'doorframe': 8, 'door frame': 8,
    'object': 39, 'objects': 39, 'other': 39, 'misc': 39,
    'armchair': 5, 'office chair': 5, 'swivel chair': 5,
    'couch': 6, 'loveseat': 6,
    'coffee table': 7, 'dining table': 7, 'end table': 7,
    'bookcase': 10, 'book shelf': 10,
    'tv': 25, 'monitor': 25,
    'trash can': 39, 'recycling bin': 39, 'garbage bin': 39,
    'plant': 39, 'plants': 39,
    'rug': 20, 'carpet': 20,
    'stairs': 38, 'staircase': 38,
    'railing': 38, 'banister': 38,
}

NYU40_TO_SCANNET20 = {nyu_id: i for i,
                      nyu_id in enumerate(SCANNET_VALID_CLASS_IDS)}

TRAIN_SCENES = None
VAL_SCENES = None


def load_scene_splits(data_root):
    """"""
    global TRAIN_SCENES, VAL_SCENES

    scans_dir = os.path.join(data_root, 'scans')
    all_scenes = sorted([d for d in os.listdir(
        scans_dir) if d.startswith('scene')])

    # scene0000_00 -> scene0000
    scene_ids = sorted(set([s[:12] for s in all_scenes]))
    np.random.seed(42)
    np.random.shuffle(scene_ids)

    n_train = int(len(scene_ids) * 0.8)
    train_scene_ids = set(scene_ids[:n_train])
    val_scene_ids = set(scene_ids[n_train:])

    TRAIN_SCENES = [s for s in all_scenes if s[:12] in train_scene_ids]
    VAL_SCENES = [s for s in all_scenes if s[:12] in val_scene_ids]

    print(f": {len(TRAIN_SCENES)}")
    print(f": {len(VAL_SCENES)}")

    return TRAIN_SCENES, VAL_SCENES


def read_ply(ply_path):
    """ PLY """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    coords = np.vstack(
        [vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    colors = np.vstack([vertex['red'], vertex['green'],
                       vertex['blue']]).T.astype(np.float32)

    return coords, colors


def read_aggregation(agg_path):
    """ aggregation.json segment ID """
    with open(agg_path, 'r') as f:
        agg_data = json.load(f)

    seg_to_label = {}
    for seg_group in agg_data.get('segGroups', []):
        label = seg_group['label'].lower()
        for seg_id in seg_group['segments']:
            seg_to_label[seg_id] = label

    return seg_to_label


def read_segmentation(seg_path):
    """ segment ID"""
    with open(seg_path, 'r') as f:
        seg_data = json.load(f)

    seg_indices = np.array(seg_data['segIndices'], dtype=np.int32)
    return seg_indices


def get_labels(seg_indices, seg_to_label):
    """ segment ID """
    n_points = len(seg_indices)
    labels = np.full(n_points, -1, dtype=np.int32)

    for i, seg_id in enumerate(seg_indices):
        if seg_id in seg_to_label:
            label_name = seg_to_label[seg_id]

            nyu40_id = LABEL_NAME_TO_NYU40.get(
                label_name, 39)

            if nyu40_id in NYU40_TO_SCANNET20:
                labels[i] = NYU40_TO_SCANNET20[nyu40_id]
            else:
                labels[i] = -1

    return labels


def process_scene(scene_name, scans_dir, output_dir):
    """"""
    scene_dir = os.path.join(scans_dir, scene_name)

    ply_path = os.path.join(scene_dir, f'{scene_name}_vh_clean_2.ply')
    agg_path = os.path.join(scene_dir, f'{scene_name}.aggregation.json')
    seg_path = os.path.join(
        scene_dir, f'{scene_name}_vh_clean_2.0.010000.segs.json')

    if not all(os.path.exists(p) for p in [ply_path, agg_path, seg_path]):
        return None, f": {scene_name}"

    try:
        coords, colors = read_ply(ply_path)
        seg_to_label = read_aggregation(agg_path)
        seg_indices = read_segmentation(seg_path)

        labels = get_labels(seg_indices, seg_to_label)

        colors = colors / 255.0

        coords = coords - coords.mean(axis=0)

        output_path = os.path.join(output_dir, f'{scene_name}.pth')
        torch.save((coords, colors, labels), output_path)

        n_labeled = np.sum(labels >= 0)
        n_total = len(labels)

        return output_path, f": {scene_name} ({n_labeled}/{n_total} )"

    except Exception as e:
        return None, f": {scene_name} - {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='ScanNet ')
    parser.add_argument('--data_root', type=str,
                        default='/data/LIANGYudong_2023/PointNeXt/data/scannet_data',
                        help='ScanNet ')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='')
    args = parser.parse_args()

    scans_dir = os.path.join(args.data_root, 'scans')

    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_scenes, val_scenes = load_scene_splits(args.data_root)

    print("\n" + "="*60)
    print("...")
    print("="*60)

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_scene, scene, scans_dir, train_dir): scene
            for scene in train_scenes
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            result, msg = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1
                print(f"  {msg}")

    print(f":  {success_count},  {fail_count}")

    print("\n" + "="*60)
    print("...")
    print("="*60)

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_scene, scene, scans_dir, val_dir): scene
            for scene in val_scenes
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=""):
            result, msg = future.result()
            if result:
                success_count += 1
            else:
                fail_count += 1
                print(f"  {msg}")

    print(f":  {success_count},  {fail_count}")

    train_files = glob.glob(os.path.join(train_dir, '*.pth'))
    val_files = glob.glob(os.path.join(val_dir, '*.pth'))

    print("\n" + "="*60)
    print("！")
    print("="*60)
    print(f": {len(train_files)} ")
    print(f": {len(val_files)} ")
    print(f"\n:")
    print(f"  : {train_dir}")
    print(f"  : {val_dir}")
    print("\n data_root :")
    print(f"  data_root: {args.data_root}")


if __name__ == '__main__':
    main()
