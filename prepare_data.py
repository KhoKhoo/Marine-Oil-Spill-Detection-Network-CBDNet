"""
Prepare the refined SOS dataset for training.

Expected source layout (already extracted):
    SRC/images/images/train
    SRC/images/images/val
    SRC/masks/masks/train
    SRC/masks/masks/val

Usage:
    python prepare_data.py --src SRC --out OUT [--sensor palsar|sentinel|all]
                            [--holdout 0.1] [--seed 42] [--dry_run]
"""
import argparse
import json
import os
import random
import re
import shutil
import sys

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = ['_sat', '_img', '_image']
MASK_SUFFIXES = ['_mask']
SENSOR_PATTERN = re.compile(r'(palsar|sentinel)', re.IGNORECASE)
SAMPLE_SIZE = 10
EXPECTED_SIZE = (256, 256)


def list_files(folder):
    if not os.path.isdir(folder):
        print(f'ERROR: folder does not exist: {folder}')
        sys.exit(1)
    names = sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and not f.startswith('.')
    )
    return names


def report_folder(label, folder, names):
    exts = sorted({os.path.splitext(n)[1] for n in names})
    print(f'[{label}] {folder}')
    print(f'  files: {len(names)}')
    print(f'  extensions: {exts}')
    print(f'  examples: {names[:5]}')


def canonical_id(stem, suffixes):
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def index_by_id(names, suffixes):
    """Map canonical id -> filename. Reports and stops on duplicate ids."""
    mapping = {}
    for name in names:
        stem, _ = os.path.splitext(name)
        cid = canonical_id(stem, suffixes)
        if cid in mapping:
            print(f'ERROR: duplicate id "{cid}" from files "{mapping[cid]}" and "{name}"')
            sys.exit(1)
        mapping[cid] = name
    return mapping


def pair_split(split_name, image_names, mask_names):
    img_map = index_by_id(image_names, IMAGE_SUFFIXES)
    mask_map = index_by_id(mask_names, MASK_SUFFIXES)

    img_ids = set(img_map)
    mask_ids = set(mask_map)

    unmatched_images = sorted(img_ids - mask_ids)
    unmatched_masks = sorted(mask_ids - img_ids)

    if unmatched_images or unmatched_masks:
        print(f'ERROR: unmatched image/mask pairs in split "{split_name}":')
        if unmatched_images:
            print(f'  images with no mask ({len(unmatched_images)}): '
                  f'{[img_map[i] for i in unmatched_images]}')
        if unmatched_masks:
            print(f'  masks with no image ({len(unmatched_masks)}): '
                  f'{[mask_map[i] for i in unmatched_masks]}')
        sys.exit(1)

    pairs = []
    for cid in sorted(img_ids):
        pairs.append((cid, img_map[cid], mask_map[cid]))
    return pairs


def detect_sensor(name):
    m = SENSOR_PATTERN.search(name)
    return m.group(1).lower() if m else None


def sensor_counts(pairs):
    counts = {'palsar': 0, 'sentinel': 0, 'unknown': 0}
    for cid, img_name, _ in pairs:
        sensor = detect_sensor(img_name)
        counts[sensor or 'unknown'] += 1
    return counts


def filter_by_sensor(pairs, sensor):
    if sensor == 'all':
        return pairs
    return [p for p in pairs if detect_sensor(p[1]) == sensor]


def check_samples(split_name, img_dir, mask_dir, pairs, seed):
    if not pairs:
        return
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(SAMPLE_SIZE, len(pairs)))
    problems = []
    for cid, img_name, mask_name in sample:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        with Image.open(img_path) as im:
            if im.size != EXPECTED_SIZE:
                problems.append(f'{img_name}: image size {im.size} != {EXPECTED_SIZE}')

        with Image.open(mask_path) as mk:
            if mk.size != EXPECTED_SIZE:
                problems.append(f'{mask_name}: mask size {mk.size} != {EXPECTED_SIZE}')
            values = set(np.unique(np.array(mk)).tolist())

        if not (values <= {0, 255} or values <= {0, 1}):
            problems.append(f'{mask_name}: unexpected mask values {sorted(values)}')

    print(f'[{split_name}] checked {len(sample)} sample pairs')
    if problems:
        print(f'  anomalies found ({len(problems)}):')
        for p in problems:
            print(f'    {p}')
    else:
        print('  no anomalies found')


def write_split(split_name, pairs, img_dir, mask_dir, out_root):
    split_dir = os.path.join(out_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for cid, img_name, mask_name in pairs:
        img_ext = os.path.splitext(img_name)[1]
        mask_ext = os.path.splitext(mask_name)[1]

        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(split_dir, f'{cid}_sat{img_ext}')
        shutil.copyfile(src_img, dst_img)

        src_mask = os.path.join(mask_dir, mask_name)
        dst_mask = os.path.join(split_dir, f'{cid}_mask.png')
        if mask_ext.lower() == '.png':
            shutil.copyfile(src_mask, dst_mask)
        else:
            with Image.open(src_mask) as mk:
                mk.save(dst_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Root of extracted SOS dataset')
    parser.add_argument('--out', required=True, help='Output root for prepared dataset')
    parser.add_argument('--sensor', choices=['palsar', 'sentinel', 'all'], default='all')
    parser.add_argument('--holdout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    img_train_dir = os.path.join(args.src, 'images', 'images', 'train')
    img_val_dir = os.path.join(args.src, 'images', 'images', 'val')
    mask_train_dir = os.path.join(args.src, 'masks', 'masks', 'train')
    mask_val_dir = os.path.join(args.src, 'masks', 'masks', 'val')

    # Step 1: scan
    img_train_names = list_files(img_train_dir)
    img_val_names = list_files(img_val_dir)
    mask_train_names = list_files(mask_train_dir)
    mask_val_names = list_files(mask_val_dir)

    print('--- Step 1: scan ---')
    report_folder('images/train', img_train_dir, img_train_names)
    report_folder('images/val', img_val_dir, img_val_names)
    report_folder('masks/train', mask_train_dir, mask_train_names)
    report_folder('masks/val', mask_val_dir, mask_val_names)

    # Step 2: pair
    print('--- Step 2: pairing ---')
    train_pairs = pair_split('train', img_train_names, mask_train_names)
    val_pairs = pair_split('val', img_val_names, mask_val_names)
    print(f'train: {len(train_pairs)} matched pairs')
    print(f'val: {len(val_pairs)} matched pairs')

    # Step 3: sensor detection
    print('--- Step 3: sensor detection ---')
    train_sensor_counts = sensor_counts(train_pairs)
    val_sensor_counts = sensor_counts(val_pairs)
    print(f'train sensor counts: {train_sensor_counts}')
    print(f'val sensor counts: {val_sensor_counts}')

    total_known = (train_sensor_counts['palsar'] + train_sensor_counts['sentinel']
                   + val_sensor_counts['palsar'] + val_sensor_counts['sentinel'])
    if total_known == 0:
        print('NOTE: no filenames contain a recognizable sensor label (palsar/sentinel).')

    train_pairs = filter_by_sensor(train_pairs, args.sensor)
    val_pairs = filter_by_sensor(val_pairs, args.sensor)
    print(f'after --sensor={args.sensor} filter: train={len(train_pairs)}, val={len(val_pairs)}')

    # Step 4: sample checks
    print('--- Step 4: sample checks ---')
    check_samples('train', img_train_dir, mask_train_dir, train_pairs, args.seed)
    check_samples('val', img_val_dir, mask_val_dir, val_pairs, args.seed)

    # Step 5: holdout split
    print('--- Step 5: holdout split ---')
    rng = random.Random(args.seed)
    shuffled = train_pairs[:]
    rng.shuffle(shuffled)
    holdout_count = round(len(shuffled) * args.holdout)
    holdout_pairs = shuffled[:holdout_count]
    final_train_pairs = shuffled[holdout_count:]
    print(f'train (final): {len(final_train_pairs)}, holdout: {len(holdout_pairs)} '
          f'(fraction={args.holdout}, seed={args.seed})')
    print(f'--sensor is applied to train, holdout and val (already filtered above).')

    if args.dry_run:
        print('--dry_run set: stopping before writing any files.')
        return

    # Step 6: write flat folders
    print('--- Step 6: writing output ---')
    os.makedirs(args.out, exist_ok=True)
    write_split('train', final_train_pairs, img_train_dir, mask_train_dir, args.out)
    if holdout_pairs:
        write_split('holdout', holdout_pairs, img_train_dir, mask_train_dir, args.out)
    write_split('val', val_pairs, img_val_dir, mask_val_dir, args.out)

    # Step 7: summary
    print('--- Step 7: summary ---')
    summary = {
        'sensor_filter': args.sensor,
        'holdout_fraction': args.holdout,
        'seed': args.seed,
        'splits': {
            'train': {'total': len(final_train_pairs), 'by_sensor': sensor_counts(final_train_pairs)},
            'holdout': {'total': len(holdout_pairs), 'by_sensor': sensor_counts(holdout_pairs)},
            'val': {'total': len(val_pairs), 'by_sensor': sensor_counts(val_pairs)},
        },
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary written to {os.path.join(args.out, "summary.json")}')


if __name__ == '__main__':
    main()
