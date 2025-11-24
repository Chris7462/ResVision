# ResVision: Multi-Task Computer Vision with Transfer Learning

ResVision is a multi-task computer vision framework for semantic segmentation, object detection, and lane detection, all sharing a ResNet101 backbone through transfer learning.

## Project Overview

ResVision uses a **feature extraction approach** for efficient transfer learning:
1. Extract features once using a frozen pretrained ResNet101 backbone
2. Cache features to disk
3. Train task-specific heads using cached features

This approach enables:
- ✅ Fast experimentation with different head architectures
- ✅ Efficient training (no repeated backbone forward passes)
- ✅ Easy multi-task learning with shared representations

## Project Structure

```
ResVision/
├── common/
│   ├── backbone/                 # Shared ResNet101 backbone
│   │   ├── resnet.py             # ResNet101 implementation
│   │   └── __init__.py
│   ├── datasets/                 # Shared dataset utilities
│   │   ├── feature_dataset.py    # Dataset for cached features
│   │   ├── create_feature_dataloaders.py   # Feature dataloader creation
│   │   └── __init__.py
│   └── tools/                 # Common utility scripts
│       └── extract_features.py   # Feature extraction script
├── segmentation/                 # Semantic segmentation (FCN on Cityscapes)
│   ├── datasets/
│   │   ├── cityscapes_dataset.py
│   │   ├── create_cityscapes_dataloaders.py
│   │   └── __init__.py
│   ├── head/                     # FCN decoder
│   ├── models/                   # Full FCN model
│   ├── utils/                    # Segmentation metrics
│   └── __init__.py
├── object_detection/             # Object detection (FCOS on COCO) - TBD
├── lane_detection/               # Lane detection (SCNN on TuSimple) - TBD
├── tools/                        # Utility scripts
│   └── prepare_cityscapes_data.py
├── data/                         # Raw datasets (gitignored)
├── features/                     # Cached features (gitignored)
├── checkpoints/                  # Model weights (gitignored)
└── outputs/                      # Experiment results (gitignored)
```

## Requirements

```bash
pip install torch torchvision
pip install albumentations
pip install pillow numpy tqdm
```

## Getting Started: Semantic Segmentation on Cityscapes

### Step 1: Download Cityscapes Dataset

1. Register and download from [Cityscapes website](https://www.cityscapes-dataset.com/)
2. Download these packages:
   - `leftImg8bit_trainvaltest.zip` (11GB)
   - `gtFine_trainvaltest.zip` (241MB)
3. Extract to `./data/Cityscapes/`:

```
data/Cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

### Step 2: Prepare Dataset

Run the preparation script to create splits and compute statistics:

```bash
python tools/prepare_cityscapes_data.py \
  --data-root ./data/Cityscapes \
  --train-ratio 0.9 \
  --seed 42
```

**What this does:**
- Finds all image-label pairs
- Splits original train into 90% train / 10% test
- Keeps original val as validation set
- Computes class distribution and weights (Median Frequency Balancing)
- Creates `splits/` directory under `data/Cityscapes/` with:
  - `train.txt` - Training file list
  - `val.txt` - Validation file list
  - `test.txt` - Test file list
  - `dataset_info.json` - Dataset metadata

**Output:**
```
Processing 2,975 images...
✓ Train: 2,677 images
✓ Val: 500 images
✓ Test: 298 images
✓ 19 classes with computed weights
```

### Step 3: Extract Features

Extract ResNet101 features from all images and cache to disk:

```bash
python common/tools/extract_features.py \
  --task segmentation \
  --data-root ./data/Cityscapes \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda
```

**Optional arguments:**
- `--output-dir` - Custom output directory (default: `./features/segmentation`)
- `--splits` - Which splits to extract (default: `train val test`)

**What this does:**
- Loads pretrained ResNet101 (frozen)
- Processes images at 1024×512 resolution
- Extracts multi-scale features: x1, x2, x3, x4
- Saves each sample as `{split}_{index:05d}.pt`

**Feature format:**
Each `.pt` file contains:
```python
{
    'x1': torch.Tensor,  # (256, 128, 256) - stride 4
    'x2': torch.Tensor,  # (512, 64, 128)  - stride 8
    'x3': torch.Tensor,  # (1024, 32, 64)  - stride 16
    'x4': torch.Tensor,  # (2048, 16, 32)  - stride 32
    'target': torch.Tensor,  # (512, 1024) - segmentation mask
}
```

**Output:**
```
features/segmentation/
├── train_00000.pt
├── train_00001.pt
├── ...
├── val_00000.pt
└── test_00000.pt
```

**Time estimate:** ~30-60 minutes for full Cityscapes (depends on GPU)

### Step 4: Train FCN Head (TBD)

```bash
python segmentation/train_fcn_features.py \
  --feature-dir ./features/segmentation \
  --dataset-info ./data/Cityscapes/splits/dataset_info.json \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-3
```

### Step 5: Evaluate (TBD)

```bash
python segmentation/test_fcn.py \
  --checkpoint ./checkpoints/segmentation/best_model.pth \
  --feature-dir ./features/segmentation \
  --output-dir ./outputs/segmentation/
```

## Design Decisions

### Why Feature Caching?

**Traditional approach:** Train end-to-end
- ❌ Backbone forward pass every epoch
- ❌ Slow iteration when experimenting with heads
- ❌ Harder to share backbone across tasks

**Our approach:** Extract features once, train heads
- ✅ Backbone forward pass only once
- ✅ Fast head experimentation (3-5x speedup)
- ✅ Easy multi-task learning (shared features)

### Why No Data Augmentation During Extraction?

For feature extraction with a **frozen backbone**, augmentation is problematic:
- Features are cached once and reused for all epochs
- Any augmentation is "baked in" to the cached features
- Can't change augmentation strategy after extraction

**Solution:**
- No augmentation during feature extraction
- Optional augmentation for end-to-end training (if backbone is trainable)

### ImageNet Normalization

We use ImageNet statistics for normalization:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

**Why?** The ResNet101 backbone is pretrained on ImageNet with these statistics. Using dataset-specific normalization would hurt performance.

### Image Resizing

Original Cityscapes: 2048×1024 → Resized to 1024×512

**Why resize?**
- Reduces memory usage (4x less)
- Faster training
- Still provides good segmentation quality

**Note:** ResNet101 has no fixed input size requirement. Both dimensions must be divisible by 32 (due to 5 downsampling layers).

## Configuration

Key parameters in `create_cityscapes_dataloaders.py`:

```python
# Image size for feature extraction
target_size = (1024, 512)  # (width, height)

# Data augmentation (only for end-to-end training)
use_augmentation = False  # Default: disabled for feature extraction

# Batch size
batch_size = 8  # Adjust based on GPU memory
```

## Cityscapes Classes

19 training classes (from official 34→19 mapping):

| ID | Class Name | ID | Class Name |
|----|------------|----|------------|
| 0 | road | 10 | sky |
| 1 | sidewalk | 11 | person |
| 2 | building | 12 | rider |
| 3 | wall | 13 | car |
| 4 | fence | 14 | truck |
| 5 | pole | 15 | bus |
| 6 | traffic light | 16 | train |
| 7 | traffic sign | 17 | motorcycle |
| 8 | vegetation | 18 | bicycle |
| 9 | terrain | | |

## TODO

- [x] Implement `create_feature_dataloaders.py`
- [ ] Implement FCN head architecture
- [ ] Implement FCN model (backbone + head)
- [ ] Implement training script for cached features
- [ ] Implement testing/evaluation script
- [ ] Add COCO object detection task
- [ ] Add TuSimple lane detection task
- [ ] Add multi-task training support

## Troubleshooting

**Issue:** `CUDA out of memory` during feature extraction
- Solution: Reduce `--batch-size` (try 4 or 2)

**Issue:** `FileNotFoundError` for Cityscapes
- Solution: Check directory structure matches expected format
- Run `prepare_cityscapes_data.py` first

**Issue:** Feature extraction is slow
- Solution: Increase `--num-workers` for faster data loading
- Check GPU utilization with `nvidia-smi`

## Citation

If you use ResVision in your research, please cite:

```bibtex
@software{resvision2025,
  title = {ResVision: Multi-Task Computer Vision with Transfer Learning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ResVision}
}
```

## License

MIT License (or your preferred license)

## Acknowledgments

- Cityscapes Dataset: [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FCN: [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
