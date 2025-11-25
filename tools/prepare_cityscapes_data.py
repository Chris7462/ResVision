"""
Cityscapes Dataset Preparation Script
- Scans leftImg8bit and gtFine folders
- Uses official 34 -> 19 class mapping
- Splits train into 90% train / 10% test
- Uses original val as val
- Computes class distribution and class weights
- Note: Images kept at original size (2048×1024), resizing handled in dataloader
"""

import os
import sys
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import json
import argparse


# Official Cityscapes 34 -> 19 class mapping
CITYSCAPES_CLASSES = {
    'unlabeled'            : (-1, (  0,  0,  0)),
    'ego vehicle'          : (-1, (  0,  0,  0)),
    'rectification border' : (-1, (  0,  0,  0)),
    'out of roi'           : (-1, (  0,  0,  0)),
    'static'               : (-1, (  0,  0,  0)),
    'dynamic'              : (-1, (111, 74,  0)),
    'ground'               : (-1, ( 81,  0, 81)),
    'road'                 : ( 0, (128, 64,128)),
    'sidewalk'             : ( 1, (244, 35,232)),
    'parking'              : (-1, (250,170,160)),
    'rail track'           : (-1, (230,150,140)),
    'building'             : ( 2, ( 70, 70, 70)),
    'wall'                 : ( 3, (102,102,156)),
    'fence'                : ( 4, (190,153,153)),
    'guard rail'           : (-1, (180,165,180)),
    'bridge'               : (-1, (150,100,100)),
    'tunnel'               : (-1, (150,120, 90)),
    'pole'                 : ( 5, (153,153,153)),
    'polegroup'            : (-1, (153,153,153)),
    'traffic light'        : ( 6, (250,170, 30)),
    'traffic sign'         : ( 7, (220,220,  0)),
    'vegetation'           : ( 8, (107,142, 35)),
    'terrain'              : ( 9, (152,251,152)),
    'sky'                  : (10, ( 70,130,180)),
    'person'               : (11, (220, 20, 60)),
    'rider'                : (12, (255,  0,  0)),
    'car'                  : (13, (  0,  0,142)),
    'truck'                : (14, (  0,  0, 70)),
    'bus'                  : (15, (  0, 60,100)),
    'caravan'              : (-1, (  0,  0, 90)),
    'trailer'              : (-1, (  0,  0,110)),
    'train'                : (16, (  0, 80,100)),
    'motorcycle'           : (17, (  0,  0,230)),
    'bicycle'              : (18, (119, 11, 32)),
    'license plate'        : (-1, (  0,  0,142)),
}

TRAINING_CLASSES = [
    'road',           # 0
    'sidewalk',       # 1
    'building',       # 2
    'wall',           # 3
    'fence',          # 4
    'pole',           # 5
    'traffic light',  # 6
    'traffic sign',   # 7
    'vegetation',     # 8
    'terrain',        # 9
    'sky',            # 10
    'person',         # 11
    'rider',          # 12
    'car',            # 13
    'truck',          # 14
    'bus',            # 15
    'train',          # 16
    'motorcycle',     # 17
    'bicycle',        # 18
]

# Mapping from labelId (0-33) to trainId (0-18 or 255 for ignore)
LABEL_ID_TO_TRAIN_ID = {
    0: 255,   # unlabeled -> ignore
    1: 255,   # ego vehicle -> ignore
    2: 255,   # rectification border -> ignore
    3: 255,   # out of roi -> ignore
    4: 255,   # static -> ignore
    5: 255,   # dynamic -> ignore
    6: 255,   # ground -> ignore
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking -> ignore
    10: 255,  # rail track -> ignore
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail -> ignore
    15: 255,  # bridge -> ignore
    16: 255,  # tunnel -> ignore
    17: 5,    # pole
    18: 255,  # polegroup -> ignore
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan -> ignore
    30: 255,  # trailer -> ignore
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,  # license plate -> ignore
}


def get_color_mapping():
    """Get trainId to RGB color mapping"""
    color_mapping = {}
    for name, (train_id, color) in CITYSCAPES_CLASSES.items():
        if train_id >= 0:  # Only valid training classes
            color_mapping[train_id] = color
    return color_mapping


def find_image_pairs(leftimg_dir, gtfine_dir, split):
    """
    Find matching pairs of images and labels for a given split

    Args:
        leftimg_dir: Base directory for leftImg8bit
        gtfine_dir: Base directory for gtFine
        split: 'train' or 'val'

    Returns:
        List of (city, basename) tuples
    """
    pairs = []
    split_img_dir = os.path.join(leftimg_dir, split)
    split_label_dir = os.path.join(gtfine_dir, split)

    if not os.path.exists(split_img_dir):
        print(f"Warning: {split_img_dir} does not exist")
        return pairs

    # Iterate through cities
    for city in sorted(os.listdir(split_img_dir)):
        city_img_dir = os.path.join(split_img_dir, city)
        city_label_dir = os.path.join(split_label_dir, city)

        if not os.path.isdir(city_img_dir):
            continue

        # Get all images in this city
        for img_file in sorted(os.listdir(city_img_dir)):
            if img_file.endswith('_leftImg8bit.png'):
                # Extract basename (e.g., 'aachen_000000_000019')
                basename = img_file.replace('_leftImg8bit.png', '')

                # Check if corresponding label exists
                label_file = f'{basename}_gtFine_labelIds.png'
                label_path = os.path.join(city_label_dir, label_file)

                if os.path.exists(label_path):
                    pairs.append((city, basename))

    return pairs


def create_splits(train_pairs, val_pairs, train_ratio, seed=42):
    """
    Split train into train/test, keep val as is

    Args:
        train_pairs: List of (city, basename) tuples for original train
        val_pairs: List of (city, basename) tuples for val
        train_ratio: Ratio for new train split (rest goes to test)
        seed: Random seed

    Returns:
        new_train_pairs, val_pairs, test_pairs
    """
    # Shuffle train pairs
    random.seed(seed)
    train_pairs_shuffled = train_pairs.copy()
    random.shuffle(train_pairs_shuffled)

    # Split train into new_train and test
    n_total = len(train_pairs_shuffled)
    n_train = int(n_total * train_ratio)

    new_train_pairs = train_pairs_shuffled[:n_train]
    test_pairs = train_pairs_shuffled[n_train:]

    return new_train_pairs, val_pairs, test_pairs


def label_id_to_train_id(label_id_mask):
    """Convert labelId mask to trainId mask"""
    train_id_mask = np.full(label_id_mask.shape, 255, dtype=np.uint8)

    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        train_id_mask[label_id_mask == label_id] = train_id

    return train_id_mask


def compute_class_distribution(pairs, gtfine_dir, num_classes):
    """Compute pixel count for each class"""
    class_counts = np.zeros(num_classes, dtype=np.int64)

    print("Computing class distribution...")
    for city, basename in tqdm(pairs):
        label_file = f'{basename}_gtFine_labelIds.png'
        label_path = os.path.join(gtfine_dir, city, label_file)

        if os.path.exists(label_path):
            label_id_mask = np.array(Image.open(label_path))
            train_id_mask = label_id_to_train_id(label_id_mask)

            # Count pixels for each class (ignore 255)
            for class_idx in range(num_classes):
                class_counts[class_idx] += np.sum(train_id_mask == class_idx)

    return class_counts


def compute_class_weights(class_counts):
    """
    Compute class weights for handling class imbalance
    Uses Median Frequency Balancing: weight[c] = median_freq / freq[c]

    Args:
        class_counts: numpy array of pixel counts for each class

    Returns:
        list: Class weights
    """
    class_counts = np.array(class_counts)
    class_weights = np.zeros(len(class_counts))

    mask = class_counts > 0
    if mask.sum() > 0:
        # Calculate frequencies
        total_pixels = class_counts.sum()
        freq = class_counts[mask] / total_pixels

        # Median Frequency Balancing
        median_freq = np.median(freq)
        class_weights[mask] = median_freq / freq

    return class_weights.tolist()


def save_split_files(output_dir, train_pairs, val_pairs, test_pairs):
    """Save train/val/test file lists"""
    os.makedirs(output_dir, exist_ok=True)

    # Format: city/basename (e.g., aachen/aachen_000000_000019)
    def format_pair(pair):
        return f"{pair[0]}/{pair[1]}"

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in train_pairs]))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in val_pairs]))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join([format_pair(p) for p in test_pairs]))

    print(f"Split files saved to {output_dir}")


def save_dataset_info(output_dir, label_id_to_train_id, class_names,
                     class_counts, class_weights, color_mapping):
    """Save dataset metadata"""
    info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'ignore_index': 255,
        'label_id_to_train_id': label_id_to_train_id,
        'color_mapping': {str(k): v for k, v in color_mapping.items()},
        'class_counts': class_counts.tolist(),
        'class_weights': class_weights,
    }

    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Dataset info saved to {output_dir}/dataset_info.json")


def main():
    parser = argparse.ArgumentParser(description='Prepare Cityscapes dataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of Cityscapes dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of train split from original train (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')
    args = parser.parse_args()

    # Setup paths
    leftimg_dir = os.path.join(args.data_root, 'leftImg8bit')
    gtfine_dir = os.path.join(args.data_root, 'gtFine')
    output_dir = os.path.join(args.data_root, 'splits')
    train_ratio = args.train_ratio
    test_ratio = 1.0 - train_ratio

    print("=" * 80)
    print("Cityscapes Dataset Preparation (19 Classes)")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Original image size: 2048×1024 (no resizing)")
    print(f"Train/Test split: {train_ratio*100:.0f}% / {test_ratio*100:.0f}%")

    # Verify paths exist
    if not os.path.exists(leftimg_dir):
        print(f"Error: {leftimg_dir} does not exist!")
        sys.exit(1)
    if not os.path.exists(gtfine_dir):
        print(f"Error: {gtfine_dir} does not exist!")
        sys.exit(1)

    # Get color mapping
    print("\n1. Loading official Cityscapes 34 -> 19 class mapping...")
    color_mapping = get_color_mapping()
    print(f"   Found {len(TRAINING_CLASSES)} classes: {', '.join(TRAINING_CLASSES)}")

    # Find image pairs
    print("\n2. Finding image-label pairs...")
    train_pairs = find_image_pairs(leftimg_dir, gtfine_dir, 'train')
    val_pairs = find_image_pairs(leftimg_dir, gtfine_dir, 'val')
    print(f"   Found {len(train_pairs)} training pairs")
    print(f"   Found {len(val_pairs)} validation pairs")

    # Create splits
    print("\n3. Creating train/val/test splits...")
    print(f"   Splitting train: {train_ratio*100:.0f}% train, {test_ratio*100:.0f}% test")
    new_train_pairs, val_pairs, test_pairs = create_splits(
        train_pairs, val_pairs, train_ratio, args.seed
    )
    print(f"   Train: {len(new_train_pairs)} images")
    print(f"   Val:   {len(val_pairs)} images")
    print(f"   Test:  {len(test_pairs)} images")

    # Save split files first
    print("\n4. Saving split files...")
    save_split_files(output_dir, new_train_pairs, val_pairs, test_pairs)

    # Compute class distribution
    print("\n5. Computing class distribution...")
    train_gtfine_dir = os.path.join(gtfine_dir, 'train')
    class_counts = compute_class_distribution(
        new_train_pairs, train_gtfine_dir, len(TRAINING_CLASSES)
    )

    # Compute class weights using Median Frequency Balancing
    print("\n6. Computing class weights (Median Frequency Balancing)...")
    class_weights = compute_class_weights(class_counts)

    # Show class distribution
    print("\n   Class distribution (training set):")
    total_pixels = class_counts.sum()
    for idx, (name, count, weight) in enumerate(zip(TRAINING_CLASSES, class_counts, class_weights)):
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"   {idx:2d}. {name:20s}: {count:10d} ({percentage:5.2f}%) weight: {weight:.4f}")

    # Save everything
    print("\n7. Saving dataset info...")
    save_dataset_info(output_dir, LABEL_ID_TO_TRAIN_ID, TRAINING_CLASSES,
                     class_counts, class_weights, color_mapping)

    print("\n" + "=" * 80)
    print("Preparation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Extract features: python common/tools/extract_features.py \\")
    print(f"       --task segmentation \\")
    print(f"       --data-root {args.data_root}")
    print(f"\n  Note: Images will be resized to 1024x512 during feature extraction")
    print(f"  Features will be saved to: ./features/segmentation/")


if __name__ == '__main__':
    main()
