"""
Common datasets module for ResVision
Provides shared dataset implementations across all tasks:
- FeatureDataset: For loading cached backbone features
- Feature dataloader creation utilities
"""


from .feature_dataset import FeatureDataset
from .create_feature_dataloaders import create_feature_dataloaders

__all__ = [
    'FeatureDataset',
    'create_feature_dataloaders',
]
