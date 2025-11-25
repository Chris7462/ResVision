# Semantic Segmentation on Cityscapes

This module implements FCN (Fully Convolutional Networks) for semantic segmentation on the Cityscapes dataset using transfer learning with a frozen ResNet101 backbone.

## Overview

**Architecture:**
- **Backbone**: ResNet101 (frozen, pretrained on ImageNet)
- **Decoder**: FCN with transposed convolutions and skip connections
- **Strategy**: Transfer learning with cached features

**Key Features:**
- Pre-compute and cache ResNet101 features once
- Train only the FCN decoder (much faster, 3-5x speedup)
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
├── train.txt       # 2475 images (90% of original train)
├── val.txt         # 500 images (official val set)
├── test.txt        # 275 images (10% of original train)
└── dataset_info.json  # Class info, weights, mappings
```

## Training Pipeline

### Step 1: Extract Features

Extract ResNet101 features from all images and cache to disk:

```bash
python common/tools/extract_features.py \
  --task segmentation \
  --data-root ./data/Cityscapes \
  --batch-size 8 \
  --device cuda
```

**Arguments:**
- `--task`: Task type (segmentation/detection/lane_detection)
- `--data-root`: Root directory of Cityscapes dataset
- `--batch-size`: Batch size for feature extraction (default: 8)
- `--output-dir`: Custom output directory (default: `./features/segmentation`)
- `--splits`: Which splits to extract (default: train val test)

**Output:**
```
features/segmentation/
├── train/
│   ├── 00000.pt
│   ├── 00001.pt
│   └── ...
├── val/
└── test/
```

Each `.pt` file contains:
- `c2`: (256, H/4, W/4) - stride 4 features
- `c3`: (512, H/8, W/8) - stride 8 features
- `c4`: (1024, H/16, W/16) - stride 16 features
- `c5`: (2048, H/32, W/32) - stride 32 features
- `target`: (H, W) - segmentation mask

### Step 2: Train FCN Decoder

Train the FCN decoder using cached features:

```bash
python segmentation/train_fcn_features.py \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json
```

**Arguments:**
- `--feature-dir`: Directory containing cached features
- `--dataset-info`: Path to dataset\_info.json
- `--resume`: Path to checkpoint to resume training
- `--override-lr`: Override learning rate when resuming

**Training Configuration:**
- Batch size: 8
- Epochs: 100
- Optimizer: SGD (lr=1e-3, momentum=0.9, weight\_decay=5e-4)
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Loss: CrossEntropyLoss with class weights (Median Frequency Balancing)

**Outputs:**
```
checkpoints/segmentation/
├── FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_best.pth
├── FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_last.pth
└── FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_epoch_10.pth

plots/segmentation/
└── FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_history.png
```

**Resume Training:**
```bash
python segmentation/train_fcn_features.py \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json \
  --resume ./checkpoints/segmentation/FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_last.pth \
  --override-lr 0.0001
```

### Step 3: Evaluate

Evaluate the trained model on the test set:

```bash
python segmentation/test_fcn.py \
  --checkpoint ./checkpoints/segmentation/FCN-ResNet101_cityscapes_cached_batch8_epoch100_SGD_lr0.001_best.pth \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json \
  --visualize \
  --num-vis 10
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--feature-dir`: Directory containing cached features
- `--dataset-info`: Path to dataset\_info.json
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

### Why Transfer Learning with Cached Features?

1. **Speed**: Extract features once, train decoder multiple times (3-5x faster)
2. **Memory**: Can use larger batch sizes since backbone not in GPU memory
3. **Efficiency**: ResNet101 is frozen anyway, no need to recompute features
4. **Flexibility**: Easy to experiment with different decoder architectures

### Why No Data Augmentation?

Since we're using cached features extracted once:
- Features are computed from fixed images
- No augmentation during feature extraction (features cached once)
- Augmentation would require re-extracting features each time

For end-to-end training with trainable backbone, augmentation would be beneficial.

### Why SGD instead of Adam?

Following the original FCN paper which uses SGD with momentum. SGD often generalizes better for vision tasks.

### Why Median Frequency Balancing?

Cityscapes has severe class imbalance (e.g., "road" appears much more than "train"). Class weights help the model learn rare classes better.

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
├── utils/
│   ├── metrics.py                     # IoU and pixel accuracy
│   └── __init__.py
├── train_fcn_features.py              # Training script
├── test_fcn.py                        # Evaluation script
└── ReadMe.md                          # This file
```

## Troubleshooting

### CUDA Out of Memory during Feature Extraction
- Reduce `--batch-size` (try 4 or 2)
- Use smaller input resolution (modify `target_size` in code)

### Training Loss Not Decreasing
- Check if class weights are loaded correctly
- Verify feature extraction completed successfully
- Try lowering learning rate with `--override-lr`

### Low mIoU on Validation
- Train for more epochs (100 may not be enough)
- Check if training/validation loss curves look reasonable
- Visualize predictions to debug

## References

- [FCN Paper](https://arxiv.org/abs/1411.4038) - Long et al., "Fully Convolutional Networks for Semantic Segmentation"
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) - Cordts et al.
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - He et al., "Deep Residual Learning for Image Recognition"
