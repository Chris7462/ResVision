"""
Evaluation Metrics for Semantic Segmentation
Includes IoU (Intersection over Union) and pixel accuracy
"""

import numpy as np


def iou(pred, target, n_class):
    """
    Calculate Intersection over Union for each class.

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices
        n_class: Number of classes

    Returns:
        list: IoU for each class (nan if class not present in ground truth)
    """
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection

        if union == 0:
            # If there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))

    return ious


def pixel_acc(pred, target):
    """
    Calculate pixel accuracy.

    Args:
        pred: Prediction mask (H, W) with class indices
        target: Ground truth mask (H, W) with class indices

    Returns:
        float: Pixel accuracy
    """
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


def batch_iou(preds, targets, n_class):
    """
    Calculate IoU for a batch of predictions.

    Args:
        preds: Batch of predictions (N, H, W)
        targets: Batch of ground truth (N, H, W)
        n_class: Number of classes

    Returns:
        numpy array: Mean IoU per class across the batch (n_class,)
    """
    total_ious = []
    for pred, target in zip(preds, targets):
        total_ious.append(iou(pred, target, n_class))

    # Shape: (n_class, batch_size)
    total_ious = np.array(total_ious).T
    # Average across batch for each class
    ious = np.nanmean(total_ious, axis=1)

    return ious


def batch_pixel_acc(preds, targets):
    """
    Calculate pixel accuracy for a batch.

    Args:
        preds: Batch of predictions (N, H, W)
        targets: Batch of ground truth (N, H, W)

    Returns:
        float: Mean pixel accuracy across the batch
    """
    pixel_accs = []
    for pred, target in zip(preds, targets):
        pixel_accs.append(pixel_acc(pred, target))

    return np.array(pixel_accs).mean()


if __name__ == '__main__':
    """Test the metrics"""
    print("Testing segmentation metrics...")

    # Create dummy predictions and targets
    n_class = 5
    pred = np.random.randint(0, n_class, size=(10, 10))
    target = np.random.randint(0, n_class, size=(10, 10))

    # Test single sample metrics
    print("\nSingle sample test:")
    ious = iou(pred, target, n_class)
    print(f"  IoU per class: {[f'{x:.4f}' if not np.isnan(x) else 'nan' for x in ious]}")
    print(f"  Mean IoU: {np.nanmean(ious):.4f}")

    pix_acc = pixel_acc(pred, target)
    print(f"  Pixel accuracy: {pix_acc:.4f}")

    # Test batch metrics
    print("\nBatch test:")
    batch_preds = np.random.randint(0, n_class, size=(4, 10, 10))
    batch_targets = np.random.randint(0, n_class, size=(4, 10, 10))

    batch_ious = batch_iou(batch_preds, batch_targets, n_class)
    print(f"  Batch IoU per class: {[f'{x:.4f}' for x in batch_ious]}")
    print(f"  Batch mean IoU: {np.nanmean(batch_ious):.4f}")

    batch_pix_acc = batch_pixel_acc(batch_preds, batch_targets)
    print(f"  Batch pixel accuracy: {batch_pix_acc:.4f}")

    print("\nMetrics test passed!")
