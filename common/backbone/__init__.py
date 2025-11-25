"""
Backbone module for ResVision
Provides shared ResNet101 backbone and feature extraction utilities
"""


from .resnet import (
    ResNet101Backbone,
    create_resnet101_backbone,
    RESNET101_FEATURE_CHANNELS
)

__all__ = [
    'ResNet101Backbone',
    'create_resnet101_backbone',
    'RESNET101_FEATURE_CHANNELS',
]
