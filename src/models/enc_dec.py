import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EncDec(nn.Module):
    """Simple encoder-decoder without skip connections (baseline for U-Net comparison)."""

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super().__init__()

        features = init_features

        # Encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 512 -> 256

        self.enc_conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 -> 128

        self.enc_conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 -> 64

        self.enc_conv3 = nn.Conv2d(features, features, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 -> 32

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(features, features, 3, padding=1)

        # Decoder (upsampling) - use scale_factor for flexibility
        self.upsample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dec_conv0 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dec_conv1 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dec_conv2 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dec_conv3 = nn.Conv2d(features, features, 3, padding=1)

        # Final output layer
        self.out_conv = nn.Conv2d(features, out_channels, 1)

    def forward(self, x):
        # Encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # Decoder (NO skip connections - key difference from U-Net)
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = F.relu(self.dec_conv3(self.upsample3(d2)))

        # Output logits
        out = self.out_conv(d3)
        return out


if __name__ == "__main__":
    model = EncDec(in_channels=3, out_channels=1, init_features=64)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with 512x512
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)

    logger.info(f"\nInput shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
