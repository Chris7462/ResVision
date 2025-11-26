# Semantic Segmentation on Cityscapes

This module implements FCN (Fully Convolutional Networks) for semantic segmentation on the Cityscapes dataset using transfer learning with a frozen ResNet101 backbone.

## Overview

**Architecture:**
- **Backbone**: ResNet101 (frozen, pretrained on ImageNet)
- **Decoder**: FCN with transposed convolutions and skip connections
- **Strategy**: Transfer learning - backbone frozen, only decoder trained

**Key Features:**
- Frozen ResNet101 backbone (pretrained on ImageNet)
- Train only the FCN decoder (fast and memory efficient)
- Uses standard FPN notation (C2, C3, C4, C5)
- Median Frequency Balancing for class imbalance
- SGD optimizer following original FCN paper

## Dataset Setup

### 1. Download Cityscapes

Download the Cityscapes dataset from [https://www.cityscapes-dataset.com](https://www.cityscapes-dataset.com):
- `leftImg8bit_trainvaltest.zip` (11GB)
- `gtFine_trainvaltest.zip` (241MB)

Extract to:
```
data/Cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### 2. Prepare Dataset

Run the preparation script to create train/val/test splits and compute class weights:
```bash
python tools/prepare_cityscapes_data.py \
  --data-root ./data/Cityscapes
```

**What this does:**
- Creates 90/10 train/test split from training data
- Keeps validation set as-is
- Converts 34 label classes to 19 training classes
- Computes class distribution and Median Frequency Balancing weights
- Saves splits to `data/Cityscapes/splits/`

**Output:**
```
data/Cityscapes/splits/
├── train.txt       # 2677 images (90% of original train)
├── val.txt         # 500 images (official val set)
├── test.txt        # 298 images (10% of original train)
└── dataset_info.json  # Class info, weights, mappings
```

## Training Pipeline

### Train FCN

Train the FCN model (frozen backbone + trainable decoder):

```bash
python segmentation/train_fcn.py \
  --data-root ./data/Cityscapes \
  --batch-size 4 \
  --epochs 100
```

**Arguments:**
- `--data-root`: Root directory of Cityscapes dataset (required)
- `--batch-size`: Batch size (default: 4). Reduce if OOM occurs.
- `--epochs`: Number of epochs (default: 100)
- `--resume`: Path to checkpoint to resume training from
- `--override-lr`: Override learning rate when resuming training

**Training Configuration:**
- Batch size: 4 (adjust based on GPU memory)
- Epochs: 100
- Optimizer: SGD (lr=1e-3, momentum=0.9, weight\_decay=5e-4)
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Loss: CrossEntropyLoss with class weights (Median Frequency Balancing)

**What gets saved:**
- Only decoder weights (backbone is always from pretrained ImageNet)
- Optimizer and scheduler states
- Training history

**Outputs:**
```
checkpoints/segmentation/
├── FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_best.pth
├── FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_last.pth
└── FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_epoch_10.pth

plots/segmentation/
└── FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_history.png
```

**Resume Training:**
```bash
python segmentation/train_fcn.py \
  --data-root ./data/Cityscapes \
  --resume ./checkpoints/segmentation/FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_last.pth \
  --override-lr 0.0001
```

### Evaluate

Evaluate the trained model on the test set:
```bash
python segmentation/test_fcn.py \
  --checkpoint ./checkpoints/segmentation/FCN-ResNet101_cityscapes_batch4_epoch100_SGD_lr0.001_best.pth \
  --data-root ./data/Cityscapes \
  --visualize \
  --num-vis 10
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--data-root`: Root directory of Cityscapes dataset (required)
- `--visualize`: Generate visualization of predictions
- `--num-vis`: Number of samples to visualize (default: 5)
- `--batch-size`: Batch size for evaluation (default: 16)

**Outputs:**
```
outputs/segmentation/
├── results_test.json         # Evaluation metrics
└── predictions_test.png      # Visualizations
```

## Cityscapes Classes

The dataset uses 19 training classes (mapped from 34 original classes):

| ID | Class Name      | ID | Class Name       |
|----|-----------------|----|------------------|
| 0  | road            | 10 | sky              |
| 1  | sidewalk        | 11 | person           |
| 2  | building        | 12 | rider            |
| 3  | wall            | 13 | car              |
| 4  | fence           | 14 | truck            |
| 5  | pole            | 15 | bus              |
| 6  | traffic light   | 16 | train            |
| 7  | traffic sign    | 17 | motorcycle       |
| 8  | vegetation      | 18 | bicycle          |
| 9  | terrain         |    |                  |

**Ignore Index:** 255 (for unlabeled/void pixels)

## Design Decisions

### Why Frozen Backbone?

Transfer learning with a frozen backbone:
- **Fast**: No backward pass through backbone, only through decoder
- **Memory efficient**: No gradients stored for backbone
- **Effective**: Pretrained ImageNet features are strong for segmentation
- **Simple**: Single-stage training, no need for feature caching

### Why SGD instead of Adam?

Following the original FCN paper which uses SGD with momentum. SGD often generalizes better for vision tasks.

### Why Median Frequency Balancing?

Cityscapes has severe class imbalance (e.g., "road" appears much more than "train"). Class weights help the model learn rare classes better.

### Checkpoint Saving Strategy

We only save decoder weights because:
- Backbone never changes (always frozen pretrained ResNet101)
- Smaller checkpoint files (~10-20MB vs ~170MB+)
- Cleaner - you only trained the decoder
- Easy to reproduce - anyone can load: pretrained backbone + your decoder

## File Structure
```
segmentation/
├── datasets/
│   ├── cityscapes_dataset.py          # Cityscapes dataset class
│   ├── create_cityscapes_dataloaders.py  # Dataloader creation
│   └── __init__.py
├── head/
│   ├── decoder.py                     # FCN decoder architecture
│   └── __init__.py
├── models/
│   ├── fcn.py                         # FCN model (backbone + decoder)
│   └── __init__.py
├── utils/
│   ├── metrics.py                     # IoU and pixel accuracy
│   └── __init__.py
├── train_fcn.py                       # Training script
├── test_fcn.py                        # Evaluation script
└── ReadMe.md                          # This file
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (try 2 or 1)
- Close other GPU processes

### Training Loss Not Decreasing
- Check if class weights are loaded correctly
- Verify dataset preparation completed successfully
- Try lowering learning rate with `--override-lr`

### Low mIoU on Validation
- Train for more epochs (100 may not be enough)
- Check if training/validation loss curves look reasonable
- Use `--visualize` to inspect predictions

## References

- [FCN Paper](https://arxiv.org/abs/1411.4038) - Long et al., "Fully Convolutional Networks for Semantic Segmentation"
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) - Cordts et al.
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., "Deep Residual Learning for Image Recognition"
