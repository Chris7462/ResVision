"""
FCN Training Script for Cityscapes with Frozen ResNet101 Backbone
Trains FCN decoder using transfer learning (backbone frozen, decoder trainable)
Following original FCN paper training settings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import argparse

from segmentation.datasets import create_cityscapes_dataloaders
from segmentation.model import create_fcn_resnet101
from segmentation.utils import pixel_accuracy, mean_pixel_accuracy, mean_iou, frequency_weighted_iou


# ================== Configuration ==================
# Training settings (following original FCN paper)
BATCH_SIZE = 16  # Adjust if OOM occurs with backbone in memory
EPOCHS = 200
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ReduceLROnPlateau settings
LR_PATIENCE = 10        # Reduce LR if no improvement for 10 epochs
LR_FACTOR = 0.5         # Multiply LR by 0.5 when reducing
LR_MIN = 1e-6           # Minimum learning rate

# Data settings
NUM_WORKERS = 20
TARGET_SIZE = (1024, 512)  # (width, height)

# Output directories
MODEL_DIR = './checkpoints/segmentation'
PLOT_DIR = './plots/segmentation'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    num_classes, ignore_index, epoch, total_epochs):
    """Train for one epoch"""
    model.train()  # Sets decoder to train mode, backbone stays in eval mode

    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)

        # Forward pass through model (backbone + decoder)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass (only decoder parameters updated)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect predictions and targets
        with torch.no_grad():
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets_np)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
            #'mIoU': f'{batch_iou_score:.4f}',
            #'acc': f'{batch_pix_acc:.4f}'
        })

    # Compute metrics on all collected predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = running_loss / len(train_loader)
    avg_iou = mean_iou(all_preds, all_targets, num_classes, ignore_index)
    avg_pixel_acc = pixel_accuracy(all_preds, all_targets, ignore_index)
    avg_mean_acc = mean_pixel_accuracy(all_preds, all_targets, num_classes, ignore_index)
    avg_fwiou = frequency_weighted_iou(all_preds, all_targets, num_classes, ignore_index)

    return avg_loss, avg_iou, avg_pixel_acc, avg_mean_acc, avg_fwiou


def validate(model, val_loader, criterion, device, num_classes,
             ignore_index, epoch, total_epochs):
    """Validate the model"""
    model.eval()  # Sets decoder to eval mode, backbone already in eval mode

    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]  ")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)

            # Forward pass through model
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Collect predictions and targets
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets_np)

    # Compute metrics on all collected predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = running_loss / len(val_loader)
    avg_iou = mean_iou(all_preds, all_targets, num_classes, ignore_index)
    avg_pixel_acc = pixel_accuracy(all_preds, all_targets, ignore_index)
    avg_mean_acc = mean_pixel_accuracy(all_preds, all_targets, num_classes, ignore_index)
    avg_fwiou = frequency_weighted_iou(all_preds, all_targets, num_classes, ignore_index)

    return avg_loss, avg_iou, avg_pixel_acc, avg_mean_acc, avg_fwiou


def plot_training_history(history, save_dir, experiment_name):
    """
    Plot training and validation loss and metrics curves.

    Args:
        history: Dictionary with keys 'train_loss', 'train_iou', 'train_pixel_acc',
                'val_loss', 'val_iou', 'val_pixel_acc'
        save_dir: Directory to save the plot
        experiment_name: Name for the saved file
    """
    import matplotlib
    matplotlib.use('Agg')
    plt.style.use('ggplot')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    num_epochs = len(history['train_loss'])
    epochs_range = np.arange(1, num_epochs + 1)

    # Plot 1: Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train Loss',
                    marker='o', markersize=3, linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Val Loss',
                    marker='s', markersize=3, linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch #', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Pixel Accuracy
    axes[0, 1].plot(epochs_range, history['train_pixel_acc'], label='Train Pixel Acc',
                    marker='o', markersize=3, linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_pixel_acc'], label='Val Pixel Acc',
                    marker='s', markersize=3, linewidth=2)
    axes[0, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch #', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mean Accuracy
    axes[1, 0].plot(epochs_range, history['train_mean_acc'], label='Train Mean Acc',
                    marker='o', markersize=3, linewidth=2)
    axes[1, 0].plot(epochs_range, history['val_mean_acc'], label='Val Mean Acc',
                    marker='s', markersize=3, linewidth=2)
    axes[1, 0].set_title('Mean Accuracy (Per-Class Average)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch #', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: IoU Metrics (mIoU and f.w. IoU)
    axes[1, 1].plot(epochs_range, history['train_iou'], label='Train mIoU',
                    marker='o', markersize=3, linewidth=2)
    axes[1, 1].plot(epochs_range, history['val_iou'], label='Val mIoU',
                    marker='s', markersize=3, linewidth=2)
    axes[1, 1].plot(epochs_range, history['train_fwiou'], label='Train f.w. IoU',
                    marker='o', markersize=3, linewidth=2, linestyle='--')
    axes[1, 1].plot(epochs_range, history['val_fwiou'], label='Val f.w. IoU',
                    marker='s', markersize=3, linewidth=2, linestyle='--')
    axes[1, 1].set_title('IoU Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch #', fontsize=12)
    axes[1, 1].set_ylabel('IoU Score', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f'{experiment_name}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load checkpoint to resume training.
    Only loads decoder weights (backbone always from pretrained ImageNet).

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to map checkpoint to

    Returns:
        start_epoch: Epoch to resume from
        best_iou: Best mIoU achieved so far
        history: Training history dict
    """
    print(f"\n[INFO] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load only decoder weights (backbone is always from pretrained)
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print('[INFO] Decoder state loaded')

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('[INFO] Optimizer state loaded')

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('[INFO] Scheduler state loaded')

    start_epoch = checkpoint['epoch']
    best_iou = checkpoint['best_iou']
    history = checkpoint['history']

    print(f"[INFO] Resuming from epoch {start_epoch}")
    print(f"[INFO] Best mIoU so far: {best_iou:.4f}")

    return start_epoch, best_iou, history


def main():
    ap = argparse.ArgumentParser(description='Train FCN with frozen ResNet101 backbone')
    ap.add_argument('--data-root', type=str, required=True,
                    help='Root directory of Cityscapes dataset')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                    help=f'Batch size (default: {BATCH_SIZE}). Reduce if OOM occurs.')
    ap.add_argument('--epochs', type=int, default=EPOCHS,
                    help=f'Number of epochs (default: {EPOCHS})')
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume training from')
    ap.add_argument('--override-lr', type=float, default=None,
                    help='Override learning rate when resuming training')
    args = ap.parse_args()

    # Update config from args
    batch_size = args.batch_size
    epochs = args.epochs

    # Experiment name
    EXPERIMENT_NAME = f"FCN-ResNet101_cityscapes_batch{batch_size}_epoch{epochs}_SGD_lr{LR}"
    print(f"Experiment: {EXPERIMENT_NAME}")

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
    class_weights = dataset_info.get('class_weights', None)

    print(f"Number of classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_cityscapes_dataloaders(
        data_root=args.data_root,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        target_size=TARGET_SIZE
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create FCN model (frozen backbone + trainable decoder)
    print(f"\nCreating FCN model...")
    model = create_fcn_resnet101(num_classes=num_classes)
    model = model.to(device)
    print(model)

    # Setup loss and optimizer (following original FCN paper)
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"\nUsing class weights (Median Frequency Balancing)")
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    else:
        print(f"\nWarning: No class weights found in dataset_info.json, using uniform weights")
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Only optimize decoder parameters (backbone is frozen)
    optimizer = optim.SGD(model.decoder.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Use ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',              # Maximize validation mIoU
        factor=LR_FACTOR,        # Multiply LR by this factor
        patience=LR_PATIENCE,    # Wait this many epochs before reducing
        min_lr=LR_MIN            # Minimum LR
    )

    print(f"\nOptimizer: SGD (decoder only)")
    print(f"  Learning rate: {LR}")
    print(f"  Momentum: {MOMENTUM}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  LR Scheduler: ReduceLROnPlateau")
    print(f"    Mode: max (based on validation mIoU)")
    print(f"    Patience: {LR_PATIENCE} epochs")
    print(f"    Factor: {LR_FACTOR}")
    print(f"    Min LR: {LR_MIN}")

    # Training state
    start_epoch = 0
    best_iou = 0.0
    history = {
        'train_loss': [],
        'train_iou': [],
        'train_pixel_acc': [],
        'train_mean_acc': [],
        'train_fwiou': [],
        'val_loss': [],
        'val_iou': [],
        'val_pixel_acc': [],
        'val_mean_acc': [],
        'val_fwiou': []
    }

    # Load checkpoint if resuming
    if args.resume:
        start_epoch, best_iou, history = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

        # Override learning rate if specified
        if args.override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.override_lr
            print(f'[INFO] Overriding learning rate to {args.override_lr}')

    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, train_iou, train_pixel_acc, train_mean_acc, train_fwiou = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            num_classes, ignore_index, epoch, epochs
        )

        # Validate
        val_loss, val_iou, val_pixel_acc, val_mean_acc, val_fwiou = validate(
            model, val_loader, criterion, device, num_classes,
            ignore_index, epoch, epochs
        )

        # Update learning rate based on validation mIoU
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['train_mean_acc'].append(train_mean_acc)
        history['train_fwiou'].append(train_fwiou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_pixel_acc'].append(val_pixel_acc)
        history['val_mean_acc'].append(val_mean_acc)
        history['val_fwiou'].append(val_fwiou)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Train - loss: {train_loss:.4f}, Pixel Acc: {train_pixel_acc:.4f}, Mean Acc: {train_mean_acc:.4f}, mIoU: {train_iou:.4f}, f.w. IoU: {train_fwiou:.4f}")
        print(f"  Val   - loss: {val_loss:.4f}, Pixel Acc: {val_pixel_acc:.4f}, Mean Acc: {val_mean_acc:.4f}, mIoU: {val_iou:.4f}, f.w. IoU: {val_fwiou:.4f}")

        # Check if this is the best model
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou

        # Check if periodic checkpoint (every 10 epochs)
        is_periodic = (epoch + 1) % 10 == 0

        # Prepare checkpoint (save decoder only, not backbone)
        checkpoint = {
            'epoch': epoch + 1,  # Next epoch to resume from
            'decoder_state_dict': model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_iou': best_iou,
            'history': history
        }

        # Always save last checkpoint
        last_model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_last.pth')
        torch.save(checkpoint, last_model_path)

        # Save best checkpoint
        if is_best:
            best_model_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved! (mIoU: {best_iou:.4f})")

        # Save periodic checkpoint
        if is_periodic:
            periodic_path = os.path.join(MODEL_DIR, f'{EXPERIMENT_NAME}_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
            print(f"  ✓ Periodic checkpoint saved (epoch {epoch+1})")

        # Plot training history after each epoch
        plot_training_history(history, PLOT_DIR, EXPERIMENT_NAME)

        print("-" * 80)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_iou:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
