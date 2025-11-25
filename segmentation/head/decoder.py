"""
FCN Decoder for Semantic Segmentation
Decoder with transposed convolutions and skip connections
Works with pre-computed ResNet101 features (transfer learning)
"""

import torch
import torch.nn as nn
from common.backbone import RESNET101_FEATURE_CHANNELS


class FCNDecoder(nn.Module):
    """
    FCN decoder for semantic segmentation.

    Takes multi-scale features from ResNet101 backbone and produces
    segmentation predictions through transposed convolutions with skip connections.

    Architecture:
        c5 (2048, H/32) -> deconv -> + c4 (1024, H/16)
                        -> deconv -> + c3 (512, H/8)
                        -> deconv -> + c2 (256, H/4)
                        -> deconv -> (128, H/2)
                        -> deconv -> (64, H)
                        -> classifier -> (num_classes, H, W)

    Args:
        num_classes: Number of output classes
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.feature_channels = RESNET101_FEATURE_CHANNELS

        # Upsampling path with skip connections
        # c5 (2048) -> c4 (1024)
        self.deconv1 = nn.ConvTranspose2d(
            self.feature_channels['c5'], self.feature_channels['c4'],
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.feature_channels['c4'])

        # c4 (1024) -> c3 (512)
        self.deconv2 = nn.ConvTranspose2d(
            self.feature_channels['c4'], self.feature_channels['c3'],
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.feature_channels['c3'])

        # c3 (512) -> c2 (256)
        self.deconv3 = nn.ConvTranspose2d(
            self.feature_channels['c3'], self.feature_channels['c2'],
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(self.feature_channels['c2'])

        # c2 (256) -> H/2 (128)
        self.deconv4 = nn.ConvTranspose2d(
            self.feature_channels['c2'], 128,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)

        # H/2 (128) -> H (64)
        self.deconv5 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(64)

        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c2, c3, c4, c5):
        """
        Forward pass with pre-computed features.

        Args:
            c2: Features from ResNet layer1 (B, 256, H/4, W/4) - stride 4
            c3: Features from ResNet layer2 (B, 512, H/8, W/8) - stride 8
            c4: Features from ResNet layer3 (B, 1024, H/16, W/16) - stride 16
            c5: Features from ResNet layer4 (B, 2048, H/32, W/32) - stride 32

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Upsample c5 and add c4 (skip connection)
        out = self.relu(self.bn1(self.deconv1(c5)))  # (B, 1024, H/16, W/16)
        out = out + c4  # Skip connection

        # Upsample and add c3 (skip connection)
        out = self.relu(self.bn2(self.deconv2(out)))  # (B, 512, H/8, W/8)
        out = out + c3  # Skip connection

        # Upsample and add c2 (skip connection)
        out = self.relu(self.bn3(self.deconv3(out)))  # (B, 256, H/4, W/4)
        out = out + c2  # Skip connection

        # Upsample to H/2
        out = self.relu(self.bn4(self.deconv4(out)))  # (B, 128, H/2, W/2)

        # Upsample to H
        out = self.relu(self.bn5(self.deconv5(out)))  # (B, 64, H, W)

        # Final classification
        out = self.classifier(out)  # (B, num_classes, H, W)

        return out

    def __repr__(self):
        """String representation showing decoder configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"FCNDecoder(\n"
            f"  Number of classes: {self.num_classes}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
            f"  Feature channels: c2={self.feature_channels['c2']}, "
            f"c3={self.feature_channels['c3']}, "
            f"c4={self.feature_channels['c4']}, "
            f"c5={self.feature_channels['c5']}\n"
            f")"
        )


if __name__ == '__main__':
    """Test the FCN decoder"""
    print("Testing FCN Decoder...")

    # Create FCN decoder for 19 classes (Cityscapes)
    num_classes = 19
    decoder = FCNDecoder(num_classes=num_classes)
    print(f"\n{decoder}")

    # Test forward pass with dummy features
    print("\nTesting forward pass...")
    batch_size = 2
    H, W = 512, 1024  # Input image size

    # Create dummy features (as if from ResNet101 at 512×1024 input)
    c2 = torch.randn(batch_size, 256, H//4, W//4)    # (2, 256, 128, 256)
    c3 = torch.randn(batch_size, 512, H//8, W//8)    # (2, 512, 64, 128)
    c4 = torch.randn(batch_size, 1024, H//16, W//16) # (2, 1024, 32, 64)
    c5 = torch.randn(batch_size, 2048, H//32, W//32) # (2, 2048, 16, 32)

    print(f"\nInput feature shapes:")
    print(f"  c2: {c2.shape}")
    print(f"  c3: {c3.shape}")
    print(f"  c4: {c4.shape}")
    print(f"  c5: {c5.shape}")

    # Forward pass
    with torch.no_grad():
        output = decoder(c2, c3, c4, c5)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: ({batch_size}, {num_classes}, {H}, {W})")

    # Check output dimensions
    assert output.shape == (batch_size, num_classes, H, W), "Output shape mismatch!"

    print("\nFCN Decoder test passed!")
