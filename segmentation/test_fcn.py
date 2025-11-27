"""
FCN Testing/Evaluation Script for Cityscapes
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

from segmentation.datasets import create_cityscapes_dataloaders
from segmentation.model import create_fcn_resnet101
from segmentation.utils import pixel_accuracy, mean_pixel_accuracy, mean_iou, frequency_weighted_iou, iou_per_class


# Output directories
OUTPUT_DIR = './outputs/segmentation'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(model, dataloader, device, num_classes, ignore_index):
    """
    Evaluate the model on a dataset.

    Args:
        model: FCN model
        dataloader: Dataloader with images
        device: Device to evaluate on
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation

    Returns:
        miou: Mean IoU across all samples
        pixel_acc: Pixel accuracy
        mean_acc: Mean pixel accuracy
        fwiou: Frequency weighted IoU
        per_class_iou: IoU for each class
        all_predictions: List of prediction arrays
        all_targets: List of target arrays
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Store predictions and targets for final evaluation
            all_predictions.extend(preds)
            all_targets.extend(targets_np)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics on entire dataset
    miou = mean_iou(all_predictions, all_targets, num_classes, ignore_index)
    pixel_acc = pixel_accuracy(all_predictions, all_targets, ignore_index)
    mean_acc = mean_pixel_accuracy(all_predictions, all_targets, num_classes, ignore_index)
    fwiou = frequency_weighted_iou(all_predictions, all_targets, num_classes, ignore_index)

    # Calculate per-class IoU on entire dataset
    # Need to compute across all images using accumulation
    total_intersection = np.zeros(num_classes, dtype=np.int64)
    total_union = np.zeros(num_classes, dtype=np.int64)

    for pred, target in zip(all_predictions, all_targets):
        valid_mask = (target != ignore_index)
        for cls in range(num_classes):
            pred_inds = (pred == cls) & valid_mask
            target_inds = (target == cls) & valid_mask

            intersection = pred_inds[target_inds].sum()
            union = pred_inds.sum() + target_inds.sum() - intersection

            total_intersection[cls] += intersection
            total_union[cls] += union

    per_class_iou = []
    for cls in range(num_classes):
        if total_union[cls] == 0:
            per_class_iou.append(float('nan'))
        else:
            per_class_iou.append(float(total_intersection[cls]) / total_union[cls])

    return miou, pixel_acc, mean_acc, fwiou, per_class_iou, all_predictions, all_targets


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
    Only loads decoder weights (backbone from pretrained ImageNet).

    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on

    Returns:
        model: Loaded FCN model
        checkpoint: Full checkpoint dict
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model (backbone from pretrained, decoder random init)
    model = create_fcn_resnet101(num_classes=num_classes)

    # Load decoder weights from checkpoint
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_iou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_iou']:.4f}")

    return model, checkpoint


def main():
    ap = argparse.ArgumentParser(description='Test FCN with frozen ResNet101 backbone')

    # Model and data arguments
    ap.add_argument('--checkpoint', type=str, required=True,
                     help='Path to model checkpoint')
    ap.add_argument('--data-root', type=str, required=True,
                    help='Root directory of Cityscapes dataset')

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
    dataset_info_path = os.path.join(args.data_root, 'splits', 'dataset_info.json')
    print(f"\nLoading dataset info from: {dataset_info_path}")
    with open(dataset_info_path, 'r') as f:
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
    dataloaders = create_cityscapes_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(1024, 512)
    )

    test_loader = dataloaders[split]

    # Evaluate
    print("\n" + "="*80)
    print(f"Evaluating on {split} split")
    print("="*80)

    miou, pixel_acc, mean_acc, fwiou, per_class_iou, predictions, targets = evaluate(
        model, test_loader, device, num_classes, ignore_index
    )

    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Mean IoU (mIoU):              {miou:.4f}")
    print(f"Pixel Accuracy:               {pixel_acc:.4f}")
    print(f"Mean Pixel Accuracy:          {mean_acc:.4f}")
    print(f"Frequency Weighted IoU:       {fwiou:.4f}")

    # Print per-class IoU
    print(f"\nPer-Class IoU:")
    for cls_id, (cls_name, cls_iou) in enumerate(zip(class_names, per_class_iou)):
        if np.isnan(cls_iou):
            print(f"  {cls_id:2d}. {cls_name:20s}: N/A (not in dataset)")
        else:
            print(f"  {cls_id:2d}. {cls_name:20s}: {cls_iou:.4f}")

    # Interpretation guidance
    if pixel_acc > mean_acc + 0.05:
        print("\nNote: Pixel Accuracy >> Mean Pixel Accuracy")
        print("  This suggests the model performs well on frequent classes")
        print("  but may struggle with rare classes.")

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'split': split,
        'mean_iou': float(miou),
        'pixel_accuracy': float(pixel_acc),
        'mean_pixel_accuracy': float(mean_acc),
        'frequency_weighted_iou': float(fwiou),
        'per_class_iou': {name: float(iou) if not np.isnan(iou) else None 
                          for name, iou in zip(class_names, per_class_iou)},
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
