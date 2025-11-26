# ResVision

Multi-task computer vision framework using transfer learning with a shared frozen ResNet101 backbone. Supports semantic segmentation, object detection, and lane detection.

## Overview

ResVision implements a transfer learning approach where:
1. **Frozen Backbone**: ResNet101 pretrained on ImageNet (always frozen)
2. **Task-Specific Heads**: Train only lightweight decoder heads for each task
3. **Multi-Task**: Share the same frozen backbone across multiple vision tasks

**Key Benefits:**
- Fast training (only decoder is trained, backbone frozen)
- Memory efficient (no gradients stored for backbone)
- Consistent feature representation across tasks
- Easy to add new tasks with same backbone

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

# 2. Train model
python segmentation/train_fcn.py --data-root ./data/Cityscapes

# 3. Evaluate
python segmentation/test_fcn.py \
  --checkpoint ./checkpoints/segmentation/FCN-ResNet101_*_best.pth \
  --data-root ./data/Cityscapes \
  --visualize
```

## Project Structure
```
ResVision/
├── common/
│   ├── backbone/               # Shared ResNet101 backbone
│   │   ├── resnet.py           # ResNet101 implementation
│   │   └── __init__.py
│   └── __init__.py
│
├── segmentation/               # Semantic segmentation (FCN on Cityscapes)
│   ├── datasets/
│   │   ├── cityscapes_dataset.py
│   │   ├── create_cityscapes_dataloaders.py
│   │   └── __init__.py
│   ├── head/                  # FCN decoder
│   │   ├── decoder.py
│   │   └── __init__.py
│   ├── models/                # FCN model (backbone + decoder)
│   │   ├── fcn_model.py
│   │   └── __init__.py
│   ├── utils/                 # Segmentation metrics
│   │   ├── metrics.py
│   │   └── __init__.py
│   ├── train_fcn.py           # Training script
│   ├── test_fcn.py            # Evaluation script
│   └── ReadMe.md              # Segmentation documentation
│
├── object_detection/          # Object detection (FCOS on COCO) - TBD
├── lane_detection/            # Lane detection (SCNN on TuSimple) - TBD
│
├── tools/                     # Utility scripts
│   └── prepare_cityscapes_data.py
│
├── data/                      # Raw datasets (gitignored)
├── checkpoints/               # Model weights (gitignored)
├── outputs/                   # Experiment results (gitignored)
├── plots/                     # Training plots (gitignored)
│
├── .gitignore
├── requirements.txt
└── ReadMe.md                  # This file
```

## Design Philosophy

### Why Transfer Learning with Frozen Backbone?

This project focuses exclusively on transfer learning:
- Backbone is always frozen (pretrained ResNet101)
- Only task-specific heads are trained
- Fast, memory-efficient, and effective

**Benefits:**
1. **Fast Training**: No backward pass through ResNet101, only through lightweight decoder
2. **Memory Efficient**: No gradients stored for backbone parameters
3. **Small Checkpoints**: Only save decoder weights (~10-20MB vs ~170MB+ for full model)
4. **Easy Experimentation**: Quick to try different decoder architectures
5. **Proven Approach**: Standard practice in modern computer vision

### Feature Representation (FPN Notation)

All tasks use the same multi-scale features from ResNet101:
- **C2** (256 channels, stride 4): Early features with high resolution
- **C3** (512 channels, stride 8): Mid-level features
- **C4** (1024 channels, stride 16): Higher-level features
- **C5** (2048 channels, stride 32): Deepest features with semantic information

Different tasks can select which feature levels they need:
- **Segmentation (FCN)**: Uses all 4 levels (C2, C3, C4, C5)
- **Object Detection (FCOS)**: Typically uses C3, C4, C5
- **Lane Detection (SCNN)**: Typically uses C5

### Checkpoint Strategy

We save only decoder weights because:
- Backbone never changes (always frozen pretrained ResNet101)
- Smaller files (~10-20MB instead of ~170MB+)
- Cleaner conceptually (you only trained the decoder)
- Easy to reproduce (pretrained backbone + your decoder weights)

## Adding New Tasks

When adding new tasks:
1. Create task directory (e.g., `object_detection/`)
2. Implement dataset class that returns images and targets
3. Create dataloader factory function
4. Implement task-specific head/decoder
5. Create model class combining frozen backbone + decoder
6. Create training and testing scripts
7. Add task-specific ReadMe with documentation

**Example structure:**
```
new_task/
├── datasets/
│   ├── dataset.py
│   ├── create_dataloaders.py
│   └── __init__.py
├── head/
│   ├── decoder.py
│   └── __init__.py
├── models/
│   ├── model.py              # Combines backbone + decoder
│   └── __init__.py
├── utils/
│   ├── metrics.py
│   └── __init__.py
├── train.py
├── test.py
└── ReadMe.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- albumentations
- tqdm
- matplotlib

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please ensure:
- New tasks follow the existing structure
- Code is well-documented
- Training scripts follow the frozen backbone pattern
- ReadMes are comprehensive

## References

- **FCN**: Long et al., "Fully Convolutional Networks for Semantic Segmentation" (CVPR 2015)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- **Cityscapes**: Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding" (CVPR 2016)
- **FPN**: Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017)

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:
```
[Add citation information if applicable]
```
