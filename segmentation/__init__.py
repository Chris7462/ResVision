"""
Segmentation module for ResVision
Provides FCN-based semantic segmentation on Cityscapes dataset with frozen ResNet101 backbone:
- datasets: Cityscapes dataset loaders
- head: FCN decoder architecture
- models: Full FCN model (backbone + decoder)
- utils: Evaluation metrics (IoU, pixel accuracy)
"""
