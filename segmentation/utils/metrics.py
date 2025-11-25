"""
Evaluation Metrics for Semantic Segmentation
Includes IoU (Intersection over Union) and pixel accuracy
Properly handles ignore_index for unlabeled pixels
"""

import numpy as np


def iou(pred, target, n_class, ignore_index=255):
    """
    Calculate Intersection over Union for each class.

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        list: IoU for each class (nan if class not present in ground truth)
    """
    ious = []

    # Create mask for valid pixels (not ignored)
    valid_mask = (target != ignore_index)

    for cls in range(n_class):
        # Only consider valid pixels
        pred_inds = (pred == cls) & valid_mask
        target_inds = (target == cls) & valid_mask

        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            # If there is no ground truth for this class, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))

    return ious


def pixel_acc(pred, target, ignore_index=255):
    """
    Calculate pixel accuracy.

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Pixel accuracy
    """
    # Only evaluate on valid pixels (not ignored)
    valid_mask = (target != ignore_index)

    correct = ((pred == target) & valid_mask).sum()
    total = valid_mask.sum()

    if total == 0:
        return 0.0

    return correct / total


def batch_iou(preds, targets, n_class, ignore_index=255):
    """
    Calculate IoU for a batch of predictions.

    Args:
        preds: Batch of predictions (N, H, W)
        targets: Batch of ground truth (N, H, W)
        n_class: Number of classes
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Mean IoU across all classes and batch
    """
    total_ious = []
    for pred, target in zip(preds, targets):
        total_ious.append(iou(pred, target, n_class, ignore_index))

    # Shape: (batch_size, n_class)
    total_ious = np.array(total_ious)

    # Average across batch and classes (ignoring nan values)
    mean_iou = np.nanmean(total_ious)

    return mean_iou


def batch_pixel_acc(preds, targets, ignore_index=255):
    """
    Calculate pixel accuracy for a batch.

    Args:
        preds: Batch of predictions (N, H, W)
        targets: Batch of ground truth (N, H, W)
        ignore_index: Index to ignore in evaluation (default: 255)

    Returns:
        float: Mean pixel accuracy across the batch
    """
    pixel_accs = []
    for pred, target in zip(preds, targets):
        pixel_accs.append(pixel_acc(pred, target, ignore_index))

    return np.array(pixel_accs).mean()


if __name__ == '__main__':
    """Test the metrics"""
    print("Testing segmentation metrics...")

    # Create dummy predictions and targets
    n_class = 5
    ignore_index = 255

    pred = np.random.randint(0, n_class, size=(10, 10))
    target = np.random.randint(0, n_class, size=(10, 10))

    # Add some ignored pixels
    target[0:2, 0:2] = ignore_index

    # Test single sample metrics
    print("\nSingle sample test:")
    ious = iou(pred, target, n_class, ignore_index)
    print(f"  IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in ious]}")
    print(f"  Mean IoU: {np.nanmean(ious):.4f}")

    pix_acc = pixel_acc(pred, target, ignore_index)
    print(f"  Pixel accuracy: {pix_acc:.4f}")

    # Test batch metrics
    print("\nBatch test:")
    batch_preds = np.random.randint(0, n_class, size=(4, 10, 10))
    batch_targets = np.random.randint(0, n_class, size=(4, 10, 10))

    # Add ignored pixels to batch
    batch_targets[:, 0:2, 0:2] = ignore_index

    batch_mean_iou = batch_iou(batch_preds, batch_targets, n_class, ignore_index)
    print(f"  Batch mean IoU: {batch_mean_iou:.4f}")

    batch_pix_acc = batch_pixel_acc(batch_preds, batch_targets, ignore_index)
    print(f"  Batch pixel accuracy: {batch_pix_acc:.4f}")

    # Test edge case: all pixels ignored
    print("\nEdge case test (all pixels ignored):")
    all_ignored_target = np.full((10, 10), ignore_index, dtype=np.int64)
    edge_ious = iou(pred, all_ignored_target, n_class, ignore_index)
    print(f"  IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in edge_ious]}")
    edge_pix_acc = pixel_acc(pred, all_ignored_target, ignore_index)
    print(f"  Pixel accuracy: {edge_pix_acc:.4f}")

    print("\nMetrics test passed!")
