"""
FCN Model for Semantic Segmentation
Combines frozen ResNet101 backbone with trainable FCN decoder
"""

import torch
import torch.nn as nn
from common.backbone import create_resnet101_backbone
from segmentation.head import FCNDecoder


class FCN(nn.Module):
    """
    Fully Convolutional Network for semantic segmentation.

    Combines a frozen ResNet101 backbone with a trainable FCN decoder.
    The backbone extracts multi-scale features (c2, c3, c4, c5) which are
    then passed to the decoder for segmentation.

    Args:
        backbone: ResNet101Backbone instance (should be frozen)
        decoder: FCNDecoder instance (trainable)
    """

    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        """
        Forward pass through backbone and decoder.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Extract features through frozen backbone (no gradient)
        with torch.no_grad():
            features = self.backbone(x)

        # Decode to segmentation map (with gradient)
        output = self.decoder(
            features['c2'],
            features['c3'],
            features['c4'],
            features['c5']
        )

        return output

    def train(self, mode=True):
        """
        Set training mode.
        Backbone always stays in eval mode (frozen).
        Only decoder switches between train/eval.
        """
        super().train(mode)
        # Keep backbone in eval mode always
        self.backbone.eval()
        return self

    def __repr__(self):
        """String representation showing model configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"FCN(\n"
            f"  Backbone: ResNet101 (frozen)\n"
            f"  Decoder: FCNDecoder (trainable)\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
            f"  Frozen parameters: {total_params - trainable_params:,}\n"
            f")"
        )


def create_fcn_resnet101(num_classes, pretrained=True):
    """
    Factory function to create FCN with frozen ResNet101 backbone.

    This is the recommended way to instantiate the model. It ensures:
    - Backbone is pretrained on ImageNet and frozen
    - Decoder is randomly initialized and trainable

    Args:
        num_classes: Number of output segmentation classes
        pretrained: If True, use ImageNet pretrained weights for backbone (default: True)

    Returns:
        FCN model instance

    Example:
        >>> model = create_fcn_resnet101(num_classes=19)
        >>> model = model.to(device)
        >>> outputs = model(images)
    """
    # Create frozen ResNet101 backbone (pretrained on ImageNet)
    backbone = create_resnet101_backbone(pretrained=pretrained, freeze=True)

    # Create trainable FCN decoder
    decoder = FCNDecoder(num_classes=num_classes)

    # Combine into FCN model
    model = FCN(backbone, decoder)

    return model


if __name__ == '__main__':
    """Test the FCN model"""
    print("Testing FCN Model...")

    # Create model for Cityscapes (19 classes)
    num_classes = 19
    model = create_fcn_resnet101(num_classes=num_classes)
    print(f"\n{model}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    H, W = 512, 1024

    dummy_images = torch.randn(batch_size, 3, H, W)

    # Test in eval mode
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)

    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected: ({batch_size}, {num_classes}, {H}, {W})")

    # Verify output shape
    assert outputs.shape == (batch_size, num_classes, H, W), "Output shape mismatch!"

    # Test in train mode
    model.train()
    outputs_train = model(dummy_images)
    print(f"\nTrain mode output shape: {outputs_train.shape}")

    # Verify backbone is still in eval mode
    assert not model.backbone.training, "Backbone should always be in eval mode!"
    assert model.decoder.training, "Decoder should be in train mode!"

    # Verify requires_grad settings
    print("\nVerifying parameter settings:")
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"  Backbone trainable params: {backbone_trainable:,} (should be 0)")
    print(f"  Decoder trainable params: {decoder_trainable:,} (should be > 0)")
    
    assert backbone_trainable == 0, "Backbone should have 0 trainable parameters!"
    assert decoder_trainable > 0, "Decoder should have trainable parameters!"
    
    print("\nFCN Model test passed!")
