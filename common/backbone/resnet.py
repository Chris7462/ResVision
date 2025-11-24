"""
ResNet101 Backbone for Multi-Task Learning
Extracts multi-scale features (C2, C3, C4, C5) from ResNet101
Shared across segmentation, object detection, and lane detection tasks
"""

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter


class ResNet101Backbone(nn.Module):
    """
    ResNet101 backbone that extracts multi-scale features.

    Features extracted:
        - x1 (C2): 256 channels, stride 4  (after layer1/conv2_x)
        - x2 (C3): 512 channels, stride 8  (after layer2/conv3_x)
        - x3 (C4): 1024 channels, stride 16 (after layer3/conv4_x)
        - x4 (C5): 2048 channels, stride 32 (after layer4/conv5_x)

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
            'layer1': 'x1',  # C2: (B, 256, H/4, W/4)
            'layer2': 'x2',  # C3: (B, 512, H/8, W/8)
            'layer3': 'x3',  # C4: (B, 1024, H/16, W/16)
            'layer4': 'x4',  # C5: (B, 2048, H/32, W/32)
        }

        # Create feature extractor
        self.backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # Feature channels for each level
        self.feature_channels = {
            'x1': 256,
            'x2': 512,
            'x3': 1024,
            'x4': 2048,
        }

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
                    'x1': (B, 256, H/4, W/4),
                    'x2': (B, 512, H/8, W/8),
                    'x3': (B, 1024, H/16, W/16),
                    'x4': (B, 2048, H/32, W/32)
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
            f"  Feature channels: x1={self.feature_channels['x1']}, "
            f"x2={self.feature_channels['x2']}, "
            f"x3={self.feature_channels['x3']}, "
            f"x4={self.feature_channels['x4']}\n"
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
