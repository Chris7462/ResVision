"""
Feature DataLoader Creation
Creates dataloaders from cached backbone features for all tasks
"""

from torch.utils.data import DataLoader
from .feature_dataset import FeatureDataset


def create_feature_dataloaders(
    feature_dir,
    batch_size=16,
    num_workers=4,
    splits=['train', 'val', 'test']
):
    """
    Create dataloaders from cached features.

    Args:
        feature_dir: Directory containing cached feature files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        splits: List of splits to create dataloaders for (default: ['train', 'val', 'test'])

    Returns:
        dict with dataloaders for each split
    """

    print(f"Creating feature dataloaders from: {feature_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Splits: {splits}")

    dataloaders = {}

    for split in splits:
        # Create dataset
        dataset = FeatureDataset(feature_dir, split)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # Only shuffle training set
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        dataloaders[split] = dataloader
        print(f"  {split.capitalize()}: {len(dataloader)} batches ({len(dataset)} samples)")

    return dataloaders


if __name__ == '__main__':
    """Test feature dataloaders"""
    import argparse

    parser = argparse.ArgumentParser(description='Test feature dataloaders')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing cached features')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='Splits to load (default: train val test)')
    args = parser.parse_args()

    # Create dataloaders
    dataloaders = create_feature_dataloaders(
        feature_dir=args.feature_dir,
        batch_size=args.batch_size,
        num_workers=4,
        splits=args.splits
    )

    # Test loading one batch from train
    if 'train' in dataloaders:
        train_loader = dataloaders['train']

        print("\nTesting train loader...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Keys: {list(batch.keys())}")
            print(f"  x1 shape: {batch['x1'].shape}")
            print(f"  x2 shape: {batch['x2'].shape}")
            print(f"  x3 shape: {batch['x3'].shape}")
            print(f"  x4 shape: {batch['x4'].shape}")
            print(f"  Target shape: {batch['target'].shape}")
            break

        print("\n✓ Feature dataloaders test passed!")
