"""
FCN Testing/Evaluation Script for Cityscapes with Cached Features
Evaluates trained FCN decoder and visualizes results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import argparse

from common.datasets import create_feature_dataloaders
from segmentation.head import FCNDecoder
from segmentation.utils import batch_iou, batch_pixel_acc


# Output directories
OUTPUT_DIR = './outputs/segmentation'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(model, dataloader, device, num_classes, ignore_index):
    """
    Evaluate the model on a dataset.

    Args:
        model: FCNDecoder model
        dataloader: Dataloader with cached features
        device: Device to evaluate on
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation

    Returns:
        mean_iou: Mean IoU across all samples
        mean_pixel_acc: Mean pixel accuracy
        all_predictions: List of prediction arrays
        all_targets: List of target arrays
    """
    model.eval()

    all_ious = []
    all_pixel_accs = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            # Load cached features
            c2 = batch['c2'].to(device)
            c3 = batch['c3'].to(device)
            c4 = batch['c4'].to(device)
            c5 = batch['c5'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            outputs = model(c2, c3, c4, c5)

            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Calculate metrics
            batch_iou_score = batch_iou(preds, targets_np, num_classes, ignore_index)
            batch_pix_acc = batch_pixel_acc(preds, targets_np, ignore_index)

            all_ious.append(batch_iou_score)
            all_pixel_accs.append(batch_pix_acc)

            # Store predictions and targets for visualization
            all_predictions.extend(preds)
            all_targets.extend(targets_np)

            # Update progress bar
            pbar.set_postfix({
                'iou': f'{batch_iou_score:.4f}',
                'acc': f'{batch_pix_acc:.4f}'
            })

    mean_iou = np.mean(all_ious)
    mean_pixel_acc = np.mean(all_pixel_accs)

    return mean_iou, mean_pixel_acc, all_predictions, all_targets


def visualize_predictions(predictions, targets, color_map, num_samples=5, save_path=None):
    """
    Visualize segmentation predictions.

    Args:
        predictions: List of prediction arrays (H, W)
        targets: List of target arrays (H, W)
        color_map: Dict mapping class IDs to RGB colors
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    import matplotlib
    matplotlib.use('Agg')

    num_samples = min(num_samples, len(predictions))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        pred = predictions[i]
        target = targets[i]

        # Convert class IDs to RGB colors
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        target_rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)

        for class_id, color in color_map.items():
            pred_rgb[pred == class_id] = color
            target_rgb[target == class_id] = color

        # Plot ground truth
        axes[i, 0].imshow(target_rgb)
        axes[i, 0].set_title(f'Sample {i+1}: Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Plot prediction
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title(f'Sample {i+1}: Prediction', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")

    plt.close()


def load_model(checkpoint_path, num_classes, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on

    Returns:
        model: Loaded FCNDecoder model
        checkpoint: Full checkpoint dict
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FCNDecoder(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_iou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_iou']:.4f}")

    return model, checkpoint


def main():
    ap = argparse.ArgumentParser(description='Test FCN decoder with cached features')

    # Model and data arguments
    ap.add_argument('--checkpoint', type=str, required=True,
                     help='Path to model checkpoint')
    ap.add_argument('--feature-dir', type=str, default='./features/segmentation',
                    help='Directory containing cached features (default: ./features/segmentation)')
    ap.add_argument('--dataset-info', type=str, default='./data/Cityscapes/splits/dataset_info.json',
                    help='Path to dataset_info.json')

    # Evaluation arguments
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Batch size (default: 16)')
    ap.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loading workers (default: 4)')

    # Visualization arguments
    ap.add_argument('--visualize', action='store_true',
                    help='Generate visualization of predictions')
    ap.add_argument('--num-vis', type=int, default=5,
                    help='Number of samples to visualize (default: 5)')

    args = ap.parse_args()

    # Always evaluate on test set
    split = 'test'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset info
    print(f"\nLoading dataset info from: {args.dataset_info}")
    with open(args.dataset_info, 'r') as f:
        dataset_info = json.load(f)

    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    ignore_index = dataset_info['ignore_index']
    color_map = dataset_info.get('color_mapping', {})

    # Convert color_map keys from string to int
    color_map = {int(k): v for k, v in color_map.items()}

    print(f"Number of classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")

    # Load model
    model, checkpoint = load_model(args.checkpoint, num_classes, device)
    print(model)

    # Create dataloader for test split
    print(f"\nCreating dataloader for '{split}' split...")
    dataloaders = create_feature_dataloaders(
        feature_dir=args.feature_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=[split]
    )

    test_loader = dataloaders[split]

    # Evaluate
    print("\n" + "="*80)
    print(f"Evaluating on {split} split")
    print("="*80)

    mean_iou, mean_pixel_acc, predictions, targets = evaluate(
        model, test_loader, device, num_classes, ignore_index
    )

    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Pixel Accuracy: {mean_pixel_acc:.4f}")

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'split': split,
        'mean_iou': float(mean_iou),
        'pixel_accuracy': float(mean_pixel_acc),
        'num_samples': len(predictions)
    }

    results_path = os.path.join(OUTPUT_DIR, f'results_{split}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Visualize predictions
    if args.visualize:
        print(f"\nGenerating visualizations ({args.num_vis} samples)...")
        vis_path = os.path.join(OUTPUT_DIR, f'predictions_{split}.png')
        visualize_predictions(
            predictions, targets, color_map,
            num_samples=args.num_vis, save_path=vis_path
        )

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
