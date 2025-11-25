"""
Backbone module for ResVision
Provides shared ResNet101 backbone and feature extraction utilities
"""


from .resnet import ResNet101Backbone, create_resnet101_backbone

# ResNet101 feature channels (constant across all tasks)
# Using standard FPN notation: C2, C3, C4, C5
RESNET101_FEATURE_CHANNELS = {
    'c2': 256,   # stride 4:  layer1 output (conv2_x)
    'c3': 512,   # stride 8:  layer2 output (conv3_x)
    'c4': 1024,  # stride 16: layer3 output (conv4_x)
    'c5': 2048,  # stride 32: layer4 output (conv5_x)
}

__all__ = [
    'ResNet101Backbone',
    'create_resnet101_backbone',
    'RESNET101_FEATURE_CHANNELS',
]
