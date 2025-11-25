"""
Feature Dataset for Loading Pre-computed Backbone Features
Shared across all tasks (segmentation, object detection, lane detection)
"""

import os
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Generic dataset for loading pre-computed ResNet101 backbone features.

    All tasks use the same feature format from ResNet101 (standard FPN notation):
        - c2: 256 channels, H/4, W/4   (stride 4)
        - c3: 512 channels, H/8, W/8   (stride 8)
        - c4: 1024 channels, H/16, W/16 (stride 16)
        - c5: 2048 channels, H/32, W/32 (stride 32)

    Each task selects which feature levels it needs during training.
    The 'target' field is task-specific (masks/boxes/lanes).

    Expected directory structure:
        feature_dir/
            train/
                00000.pt
                00001.pt
                ...
            val/
                00000.pt
                ...
            test/
                00000.pt
                ...

    Args:
        feature_dir: Base directory containing split subdirectories
        split: 'train', 'val', or 'test'
    """

    def __init__(self, feature_dir, split):
        self.feature_dir = feature_dir
        self.split = split

        # Path to this split's subdirectory
        split_dir = os.path.join(feature_dir, split)

        if not os.path.exists(split_dir):
            raise ValueError(
                f"Split directory does not exist: {split_dir}\n"
                f"Expected structure: {feature_dir}/{split}/*.pt"
            )

        # Find all feature files for this split
        self.feature_files = sorted([
            f for f in os.listdir(split_dir)
            if f.endswith('.pt')
        ])

        if len(self.feature_files) == 0:
            raise ValueError(
                f"No feature files found in {split_dir}\n"
                f"Expected files matching pattern: *.pt"
            )

        # Store full paths for faster loading
        self.feature_paths = [
            os.path.join(split_dir, f) for f in self.feature_files
        ]

        print(f"  Loaded {len(self.feature_files)} cached features for '{split}' split")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        """
        Load pre-computed features and targets.

        Returns:
            dict with keys:
                'c2': (256, H/4, W/4) - stride 4 features
                'c3': (512, H/8, W/8) - stride 8 features
                'c4': (1024, H/16, W/16) - stride 16 features
                'c5': (2048, H/32, W/32) - stride 32 features
                'target': Task-specific target (mask/boxes/lanes)
        """
        feature_path = self.feature_paths[idx]
        data = torch.load(feature_path, weights_only=False)
        return data

    def __repr__(self):
        return (
            f"FeatureDataset(\n"
            f"  Split: {self.split}\n"
            f"  Feature directory: {self.feature_dir}\n"
            f"  Number of samples: {len(self.feature_files)}\n"
            f")"
        )


if __name__ == '__main__':
    """Test the FeatureDataset"""
    import argparse

    parser = argparse.ArgumentParser(description='Test FeatureDataset')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing cached features')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split (default: train)')
    args = parser.parse_args()

    print("Testing FeatureDataset...")
    print(f"Feature directory: {args.feature_dir}")

    # Create dataset
    dataset = FeatureDataset(args.feature_dir, args.split)
    print(f"\n{dataset}")

    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]

        print(f"\nSample keys: {list(sample.keys())}")
        print(f"\nFeature shapes:")
        print(f"  x1: {sample['x1'].shape}")
        print(f"  x2: {sample['x2'].shape}")
        print(f"  x3: {sample['x3'].shape}")
        print(f"  x4: {sample['x4'].shape}")
        print(f"\nTarget type: {type(sample['target'])}")
        if isinstance(sample['target'], torch.Tensor):
            print(f"Target shape: {sample['target'].shape}")

        print("\nFeatureDataset test passed!")
    else:
        print("\nNo samples found in dataset!")
