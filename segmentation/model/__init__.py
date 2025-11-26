"""
Segmentation models module
Provides full model architectures (backbone + head) for semantic segmentation
"""

from .fcn import FCN, create_fcn_resnet101


__all__ = [
    'FCN',
    'create_fcn_resnet101',
]
