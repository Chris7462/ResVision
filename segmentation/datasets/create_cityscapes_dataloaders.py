"""
Cityscapes DataLoader Creation
Creates train, validation, and test dataloaders with data augmentation for training
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


def get_transform(target_size, mean=IMAGENET_MEAN, std=IMAGENET_STD, train=False):
    """
    Get transforms for training or validation/test.

    Training uses data augmentation (horizontal flip).
    Validation/test use only resize and normalization.

    Args:
        target_size: (width, height) tuple
        mean: RGB mean for normalization (default: ImageNet)
        std: RGB std for normalization (default: ImageNet)
        train: If True, apply training augmentations

    Returns:
        Albumentations Compose object
    """
    if train:
        # Training: with data augmentation
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            # TODO: Experiment with additional augmentations if needed:
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        # Validation/Test: no augmentation
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


def create_cityscapes_dataloaders(
    data_root,
    batch_size=8,
    num_workers=4,
    target_size=(1024, 512)
):
    """
    Create train, validation, and test dataloaders for Cityscapes.

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

    Returns:
        dict with 'train', 'val', 'test' dataloaders and metadata
    """
    # Setup paths
    leftimg_dir = os.path.join(data_root, 'leftImg8bit')
    gtfine_dir = os.path.join(data_root, 'gtFine')
    splits_dir = os.path.join(data_root, 'splits')
    dataset_info_path = os.path.join(data_root, 'splits', 'dataset_info.json')

    # Load dataset info
    with open(dataset_info_path, 'r') as f:
        info = json.load(f)

    print(f"Creating Cityscapes dataloaders...")
    print(f"  Target size: {target_size[0]}x{target_size[1]}")
    print(f"  Normalization: ImageNet (mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
    print(f"  Training augmentation: HorizontalFlip(p=0.5)")

    # Create transforms
    train_transform = get_transform(target_size=target_size, train=True)
    val_transform = get_transform(target_size=target_size, train=False)

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
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} images) - with augmentation")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} images) - no augmentation")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} images) - no augmentation")
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

    # Create dataloaders
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

    # Test loading one batch from train
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
