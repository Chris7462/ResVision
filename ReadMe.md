# ResVision

Multi-task computer vision framework using transfer learning with a shared ResNet101 backbone. Supports semantic segmentation, object detection, and lane detection with pre-computed feature caching for efficient training.

## Overview

ResVision implements a transfer learning approach where:
1. **Feature Extraction**: Extract ResNet101 features once and cache to disk
2. **Task Training**: Train task-specific heads using cached features (3-5x faster)
3. **Multi-Task**: Share the same backbone across multiple vision tasks

**Key Benefits:**
- Extract features once, train multiple times
- Larger batch sizes (backbone not in GPU memory)
- Faster experimentation with decoder architectures
- Consistent feature representation across tasks

## Supported Tasks

### 1. Semantic Segmentation (Cityscapes)
FCN decoder for pixel-wise classification on Cityscapes dataset.

**Status:** Complete
**Documentation:** [segmentation/ReadMe.md](segmentation/ReadMe.md)

### 2. Object Detection (COCO)
FCOS decoder for anchor-free object detection.

**Status:** Coming Soon

### 3. Lane Detection (TuSimple)
SCNN decoder for lane line detection.

**Status:** Coming Soon

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Chris7462/ResVision.git
cd ResVision

# Create virtual environment
python -m venv res
source res/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Getting Started with Segmentation

See detailed instructions in [segmentation/ReadMe.md](segmentation/ReadMe.md)

**Quick workflow:**
```bash
# 1. Prepare dataset
python tools/prepare_cityscapes_data.py --data-root ./data/Cityscapes

# 2. Extract features
python common/tools/extract_features.py \
  --task segmentation \
  --data-root ./data/Cityscapes

# 3. Train decoder
python segmentation/train_fcn_features.py \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json

# 4. Evaluate
python segmentation/test_fcn.py \
  --checkpoint ./checkpoints/segmentation/FCN-ResNet101_*_best.pth \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json \
  --visualize
```

## Project Structure

```
ResVision/
├── common/
│   ├── backbone/               # Shared ResNet101 backbone
│   │   ├── resnet.py           # ResNet101 implementation
│   │   └── __init__.py
│   ├── datasets/               # Shared dataset utilities
│   │   ├── feature_dataset.py  # Dataset for cached features
│   │   ├── create_feature_dataloaders.py  # Feature dataloader creation
│   │   └── __init__.py
│   └── tools/                  # Common utility scripts
│       └── extract_features.py # Feature extraction script
├── segmentation/               # Semantic segmentation (FCN on Cityscapes)
│   ├── datasets/
│   │   ├── cityscapes_dataset.py
│   │   ├── create_cityscapes_dataloaders.py
│   │   └── __init__.py
│   ├── head/                  # FCN decoder
│   │   ├── decoder.py
│   │   └── __init__.py
│   ├── utils/                 # Segmentation metrics
│   │   ├── metrics.py
│   │   └── __init__.py
│   ├── train_fcn_features.py  # Training script
│   ├── test_fcn.py            # Evaluation script
│   └── ReadMe.md              # Segmentation documentation
├── object_detection/          # Object detection (FCOS on COCO) - TBD
├── lane_detection/            # Lane detection (SCNN on TuSimple) - TBD
├── tools/                     # Utility scripts
│   └── prepare_cityscapes_data.py
├── data/                      # Raw datasets (gitignored)
├── features/                  # Cached features (gitignored)
├── checkpoints/               # Model weights (gitignored)
├── outputs/                   # Experiment results (gitignored)
├── plots/                     # Training plots (gitignored)
├── .gitignore
├── requirements.txt
└── ReadMe.md                  # This file
```

## Design Philosophy

### Why Transfer Learning with Cached Features?

1. **Speed**: Extract features once, train decoder multiple times (3-5x faster)
2. **Memory**: Can use larger batch sizes since backbone not in GPU memory
3. **Efficiency**: ResNet101 is frozen anyway, no need to recompute features
4. **Flexibility**: Easy to experiment with different decoder architectures

### Why No End-to-End Training?

This project focuses exclusively on transfer learning:
- Backbone is always frozen (pretrained ResNet101)
- Features are cached once and reused
- Only task-specific heads are trained

For end-to-end training with trainable backbones, this is not the right framework.

### Feature Representation (FPN Notation)

All tasks use the same multi-scale features from ResNet101:
- **C2** (256 channels, stride 4): Early features with high resolution
- **C3** (512 channels, stride 8): Mid-level features
- **C4** (1024 channels, stride 16): Higher-level features
- **C5** (2048 channels, stride 32): Deepest features with semantic information

Different tasks can select which feature levels they need:
- Segmentation (FCN): Uses all 4 levels (C2, C3, C4, C5)
- Object Detection (FCOS): Typically uses C3, C4, C5
- Lane Detection (SCNN): Typically uses C5

## Contributing

When adding new tasks:
1. Create task directory (e.g., `object_detection/`)
2. Implement dataset class that returns images and targets
3. Create dataloader using standard Cityscapes pattern
4. Implement task-specific head/decoder
5. Create training and testing scripts
6. Add task-specific ReadMe with documentation
7. Update `common/tools/extract_features.py` to support new task

## References

- **FCN**: Long et al., "Fully Convolutional Networks for Semantic Segmentation" (CVPR 2015)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- **Cityscapes**: Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding" (CVPR 2016)
- **FPN**: Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017)
