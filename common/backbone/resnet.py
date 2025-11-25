"""
ResNet101 Backbone for Multi-Task Learning
Extracts multi-scale features (C2, C3, C4, C5) from ResNet101
Shared across segmentation, object detection, and lane detection tasks
"""

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter


# ResNet101 feature channels (constant across all tasks)
# Using standard FPN notation: C2, C3, C4, C5
RESNET101_FEATURE_CHANNELS = {
    'c2': 256,   # stride 4:  layer1 output (conv2_x)
    'c3': 512,   # stride 8:  layer2 output (conv3_x)
    'c4': 1024,  # stride 16: layer3 output (conv4_x)
    'c5': 2048,  # stride 32: layer4 output (conv5_x)
}


class ResNet101Backbone(nn.Module):
    """
    ResNet101 backbone that extracts multi-scale features.

    Features extracted (using standard FPN notation):
        - c2: 256 channels, stride 4  (after layer1/conv2_x)
        - c3: 512 channels, stride 8  (after layer2/conv3_x)
        - c4: 1024 channels, stride 16 (after layer3/conv4_x)
        - c5: 2048 channels, stride 32 (after layer4/conv5_x)

    Args:
        pretrained: If True, load ImageNet pretrained weights
        freeze: If True, freeze all backbone parameters
    """

    def __init__(self, pretrained=True, freeze=False):
        super().__init__()

        # Load pretrained ResNet101
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet101(weights=weights)

        # Define which layers to extract features from
        return_layers = {
            'layer1': 'c2',  # stride 4:  (B, 256, H/4, W/4)
            'layer2': 'c3',  # stride 8:  (B, 512, H/8, W/8)
            'layer3': 'c4',  # stride 16: (B, 1024, H/16, W/16)
            'layer4': 'c5',  # stride 32: (B, 2048, H/32, W/32)
        }

        # Create feature extractor
        self.backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # Use shared feature channels constant
        self.feature_channels = RESNET101_FEATURE_CHANNELS

        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through backbone.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            dict: Multi-scale features
                {
                    'c2': (B, 256, H/4, W/4),
                    'c3': (B, 512, H/8, W/8),
                    'c4': (B, 1024, H/16, W/16),
                    'c5': (B, 2048, H/32, W/32)
                }
        """
        return self.backbone(x)

    def __repr__(self):
        """String representation showing backbone configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_status = "frozen" if trainable_params == 0 else "trainable"

        return (
            f"ResNet101Backbone(\n"
            f"  Status: {frozen_status}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
            f"  Feature channels: c2={self.feature_channels['c2']}, "
            f"c3={self.feature_channels['c3']}, "
            f"c4={self.feature_channels['c4']}, "
            f"c5={self.feature_channels['c5']}\n"
            f")"
        )


def create_resnet101_backbone(pretrained=True, freeze=False):
    """
    Factory function to create ResNet101 backbone.

    Args:
        pretrained: If True, load ImageNet pretrained weights
        freeze: If True, freeze all backbone parameters

    Returns:
        ResNet101Backbone instance
    """
    return ResNet101Backbone(pretrained=pretrained, freeze=freeze)


if __name__ == '__main__':
    """Test the backbone"""
    print("Testing ResNet101 Backbone...")

    # Create backbone (unfrozen)
    backbone = create_resnet101_backbone(pretrained=True, freeze=False)
    print(f"\n{backbone}")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        features = backbone(dummy_input)

    print("\nOutput features:")
    for key, feat in features.items():
        print(f"  {key}: {feat.shape}")

    # Test frozen backbone
    print("\nTesting frozen backbone...")
    backbone_frozen = create_resnet101_backbone(pretrained=True, freeze=True)
    print(f"\n{backbone_frozen}")
