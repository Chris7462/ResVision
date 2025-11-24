"""
Generic Feature Extraction Script
Extracts ResNet101 backbone features from any dataset and caches them to disk
Used by all tasks (segmentation, object detection, lane detection)
"""

import os
import torch
import argparse
from tqdm import tqdm


def extract_features(dataloader, backbone, output_dir, split, device='cuda'):
    """
    Extract and save ResNet101 features from a dataset.

    Args:
        dataloader: PyTorch DataLoader with raw images
                   Each batch should have 'image' and 'target' keys
        backbone: ResNet101Backbone instance (should be frozen)
        output_dir: Directory to save extracted features
        split: Split name ('train', 'val', 'test')
        device: Device to run extraction on ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    backbone = backbone.to(device)
    backbone.eval()

    print(f"\nExtracting features for '{split}' split...")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split}"):
            images = batch['image'].to(device)
            batch_size = images.size(0)

            # Extract features through backbone
            features = backbone(images)

            # Save each sample in the batch
            for i in range(batch_size):
                # Prepare save dictionary
                save_dict = {
                    'x1': features['x1'][i].cpu(),
                    'x2': features['x2'][i].cpu(),
                    'x3': features['x3'][i].cpu(),
                    'x4': features['x4'][i].cpu(),
                }

                # Add target (task-specific: mask/boxes/lanes)
                if isinstance(batch['target'], dict):
                    # For detection: target is a dict with boxes, labels, etc.
                    save_dict['target'] = {
                        k: v[i].cpu() if isinstance(v, torch.Tensor) else v[i]
                        for k, v in batch['target'].items()
                    }
                else:
                    # For segmentation/lanes: target is a tensor
                    save_dict['target'] = batch['target'][i].cpu()

                # Save to disk
                save_path = os.path.join(output_dir, f"{split}_{sample_count:05d}.pt")
                torch.save(save_dict, save_path)

                sample_count += 1

    print(f"✓ Extracted {sample_count} samples for '{split}' split")
    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ResNet101 features from a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Segmentation (Cityscapes)
  python extract_features.py \\
    --task segmentation \\
    --data-root ./data/Cityscapes \\
    --batch-size 8

  # Object Detection (COCO)
  python extract_features.py \\
    --task detection \\
    --data-root ./data/COCO \\
    --batch-size 8

  # Custom output directory
  python extract_features.py \\
    --task segmentation \\
    --data-root ./data/Cityscapes \\
    --output-dir ./features/my_custom_features
        """
    )

    parser.add_argument('--task', type=str, required=True,
                        choices=['segmentation', 'detection', 'lane_detection'],
                        help='Task type (determines which dataset to use)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save extracted features (default: ./features/{task})')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for feature extraction (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help='Splits to extract (default: train val test)')

    args = parser.parse_args()

    # Auto-generate output directory based on task if not specified
    if args.output_dir is None:
        args.output_dir = f'./features/{args.task}'

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    print("="*80)
    print("ResVision Feature Extraction")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Splits: {args.splits}")
    print(f"Device: {args.device}")

    # Import task-specific dataloader creator based on task
    if args.task == 'segmentation':
        from segmentation.datasets import create_cityscapes_dataloaders
        dataloader_fn = create_cityscapes_dataloaders

    elif args.task == 'detection':
        from object_detection.datasets import create_coco_dataloaders
        dataloader_fn = create_coco_dataloaders

    elif args.task == 'lane_detection':
        from lane_detection.datasets import create_tusimple_dataloaders
        dataloader_fn = create_tusimple_dataloaders

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Create ResNet101 backbone (frozen, pretrained)
    from common.backbone import create_resnet101_backbone

    print("\nCreating ResNet101 backbone...")
    backbone = create_resnet101_backbone(pretrained=True, freeze=True)
    print(backbone)

    # Extract features for each split
    for split in args.splits:
        print(f"\n{'='*80}")
        print(f"Processing '{split}' split")
        print(f"{'='*80}")

        # Create dataloader for this split
        # Note: Each task's dataloader function should accept these arguments
        dataloaders = dataloader_fn(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        if split not in dataloaders:
            print(f"Warning: Split '{split}' not found in dataloaders, skipping...")
            continue

        dataloader = dataloaders[split]

        # Extract and save features
        extract_features(
            dataloader=dataloader,
            backbone=backbone,
            output_dir=args.output_dir,
            split=split,
            device=args.device
        )

    print("\n" + "="*80)
    print("Feature extraction complete!")
    print("="*80)
    print(f"Features saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
