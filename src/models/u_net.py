import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2

    U-Net building block that applies two 3 x 3 convolutions,
    each followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Bilinear upsample (no channel reduction)"""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """U-Net architecture for binary image segmentation."""

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super().__init__()

        features = init_features

        # Encoder (contracting path)
        self.encoder1 = DoubleConv(in_channels, features)
        self.encoder2 = Down(features, features * 2)
        self.encoder3 = Down(features * 2, features * 4)
        self.encoder4 = Down(features * 4, features * 8)

        # Bottleneck with dropout for regularization
        self.bottleneck = nn.Sequential(
            Down(features * 8, features * 16), nn.Dropout2d(0.5)
        )

        # Decoder (expanding path) - upsampling only (no channel changes)
        self.up4 = Up()
        self.up3 = Up()
        self.up2 = Up()
        self.up1 = Up()

        # Decoder convolutions - after concatenation with skip connections
        # Input channels = upsampled channels + skip connection channels
        self.decoder4 = DoubleConv(
            features * 16 + features * 8, features * 8
        )  # 1024 + 512 -> 512
        self.decoder3 = DoubleConv(
            features * 8 + features * 4, features * 4
        )  # 512 + 256 -> 256
        self.decoder2 = DoubleConv(
            features * 4 + features * 2, features * 2
        )  # 256 + 128 -> 128
        self.decoder1 = DoubleConv(features * 2 + features, features)  # 128 + 64 -> 64

        # Final 1x1 convolution to produce output logits
        self.out_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.encoder1(x)  # 512x512x64
        enc2 = self.encoder2(enc1)  # 256x256x128
        enc3 = self.encoder3(enc2)  # 128x128x256
        enc4 = self.encoder4(enc3)  # 64x64x512

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 32x32x1024

        # Decoder with skip connections
        up4 = self.up4(bottleneck)  # 64x64x1024 (just upsample, no channel change)
        concat4 = torch.cat([up4, enc4], dim=1)  # 64x64x1536 (1024 + 512)
        dec4 = self.decoder4(concat4)  # 64x64x512

        up3 = self.up3(dec4)  # 128x128x512
        concat3 = torch.cat([up3, enc3], dim=1)  # 128x128x768 (512 + 256)
        dec3 = self.decoder3(concat3)  # 128x128x256

        up2 = self.up2(dec3)  # 256x256x256
        concat2 = torch.cat([up2, enc2], dim=1)  # 256x256x384 (256 + 128)
        dec2 = self.decoder2(concat2)  # 256x256x128

        up1 = self.up1(dec2)  # 512x512x128
        concat1 = torch.cat([up1, enc1], dim=1)  # 512x512x192 (128 + 64)
        dec1 = self.decoder1(concat1)  # 512x512x64

        # Output
        logits = self.out_conv(dec1)  # 512x512x1

        return logits


if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=3, out_channels=1, init_features=64)

    # Print model summary
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)

    logger.info(f"\nInput shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
