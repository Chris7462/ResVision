"""
Cityscapes DataLoader Creation
Creates train, validation, and test dataloaders
Supports both simple interface (for feature extraction) and detailed interface
"""

import os
import json
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .cityscapes_dataset import CityscapesDataset


# ImageNet normalization (required for pretrained ResNet101)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_training_transform(target_size, mean=IMAGENET_MEAN, std=IMAGENET_STD, use_augmentation=False):
    """
    Get training transforms

    Args:
        target_size: (width, height) tuple
        mean: RGB mean for normalization (default: ImageNet)
        std: RGB std for normalization (default: ImageNet)
        use_augmentation: If True, apply data augmentation (for trainable backbone).
                         If False, only resize and normalize (for frozen backbone/feature extraction).
    """
    if use_augmentation:
        # Full augmentation for end-to-end training with trainable backbone
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        # No augmentation for feature extraction with frozen backbone
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


def get_validation_transform(target_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Get validation/test transforms (no augmentation)

    Args:
        target_size: (width, height) tuple
        mean: RGB mean for normalization (default: ImageNet)
        std: RGB std for normalization (default: ImageNet)
    """
    return A.Compose([
        A.Resize(height=target_size[1], width=target_size[0]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def create_cityscapes_dataloaders(
    data_root,
    batch_size=8,
    num_workers=4,
    target_size=(1024, 512),
    use_augmentation=False
):
    """
    Simple interface for feature extraction and standard use cases.
    Assumes standard Cityscapes directory structure:
        data_root/
            leftImg8bit/
                train/
                val/
            gtFine/
                train/
                val/
            splits/
                train.txt
                val.txt
                test.txt
                dataset_info.json

    Args:
        data_root: Root directory of Cityscapes dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        target_size: (width, height) tuple (default: 1024×512)
        use_augmentation: If True, apply data augmentation to training set (default: False)

    Returns:
        dict with 'train', 'val', 'test' dataloaders and metadata
    """
    leftimg_dir = os.path.join(data_root, 'leftImg8bit')
    gtfine_dir = os.path.join(data_root, 'gtFine')
    splits_dir = os.path.join(data_root, 'splits')
    dataset_info_path = os.path.join(data_root, 'splits', 'dataset_info.json')

    return create_cityscapes_dataloaders_detailed(
        leftimg_dir=leftimg_dir,
        gtfine_dir=gtfine_dir,
        splits_dir=splits_dir,
        dataset_info_path=dataset_info_path,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size,
        use_augmentation=use_augmentation
    )


def create_cityscapes_dataloaders_detailed(
    leftimg_dir,
    gtfine_dir,
    splits_dir,
    dataset_info_path,
    batch_size=8,
    num_workers=4,
    target_size=(1024, 512),
    use_augmentation=False
):
    """
    Detailed interface with explicit paths.
    Use this if your directory structure is non-standard.

    Args:
        leftimg_dir: Base directory with leftImg8bit images (contains train/val subdirs)
        gtfine_dir: Base directory with gtFine labels (contains train/val subdirs)
        splits_dir: Directory containing train.txt, val.txt, test.txt
        dataset_info_path: Path to dataset_info.json
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        target_size: (width, height) tuple (default: 1024×512)
        use_augmentation: If True, apply data augmentation to training set (default: False)

    Returns:
        dict with 'train', 'val', 'test' dataloaders and metadata
    """

    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    print(f"Creating Cityscapes dataloaders...")
    print(f"  Target size: {target_size[0]}×{target_size[1]}")
    print(f"  Normalization: ImageNet (mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
    print(f"  Data augmentation: {'Enabled' if use_augmentation else 'Disabled'}")

    # Create transforms
    train_transform = get_training_transform(target_size=target_size, use_augmentation=use_augmentation)
    val_transform = get_validation_transform(target_size=target_size)

    # Create datasets
    train_dataset = CityscapesDataset(
        split_file=os.path.join(splits_dir, 'train.txt'),
        leftimg_dir=os.path.join(leftimg_dir, 'train'),
        gtfine_dir=os.path.join(gtfine_dir, 'train'),
        dataset_info_path=dataset_info_path,
        transform=train_transform
    )

    val_dataset = CityscapesDataset(
        split_file=os.path.join(splits_dir, 'val.txt'),
        leftimg_dir=os.path.join(leftimg_dir, 'val'),
        gtfine_dir=os.path.join(gtfine_dir, 'val'),
        dataset_info_path=dataset_info_path,
        transform=val_transform
    )

    test_dataset = CityscapesDataset(
        split_file=os.path.join(splits_dir, 'test.txt'),
        leftimg_dir=os.path.join(leftimg_dir, 'train'),  # Test split comes from train
        gtfine_dir=os.path.join(gtfine_dir, 'train'),    # Test split comes from train
        dataset_info_path=dataset_info_path,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} images)")
    print(f"  Batch size: {batch_size}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': info['class_weights'],
        'num_classes': info['num_classes'],
        'class_names': info['class_names'],
        'ignore_index': info['ignore_index']
    }


if __name__ == '__main__':
    """Test the dataloader creation"""
    import argparse
    import torch

    parser = argparse.ArgumentParser(description='Test Cityscapes dataloaders')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of Cityscapes dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[1024, 512],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Target image size (default: 1024 512)')
    args = parser.parse_args()

    # Create dataloaders using simple interface
    dataloaders = create_cityscapes_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        target_size=tuple(args.target_size)
    )

    # Access dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    class_weights = dataloaders['class_weights']

    print(f"\nDataset info:")
    print(f"  Number of classes: {dataloaders['num_classes']}")
    print(f"  Class names: {dataloaders['class_names']}")
    print(f"  Ignore index: {dataloaders['ignore_index']}")
    print(f"  Class weights (first 5): {class_weights[:5]}")

    # Test loading one batch
    print("\nTesting train loader...")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        targets = batch['target']

        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Unique target values: {len(torch.unique(targets))} classes")
        break

    print("\nCityscapes dataloaders test passed!")
