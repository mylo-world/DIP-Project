"""
  Encoder  : 3 levels  (32 → 64 → 128 filters)
             each level = Conv3×3+BN+ReLU × 2 → MaxPool 2×2

  Bottleneck: Conv3×3+BN+ReLU × 2  (256 filters)

  Decoder  : 3 levels  (128 → 64 → 32 filters)
             each level = ConvTranspose2×2 + skip-cat + Conv3×3+BN+ReLU × 2

  Output   : Conv1×1 + (Sigmoid applied externally / by loss fn)

Default in_channels = 9  (RGB + LBP + Canny + HSV + BBox)
"""

import torch
import torch.nn as nn

\
class _ConvBlock(nn.Sequential):
    """Two Conv3×3 + BN + ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _EncoderBlock(nn.Module):
    """ConvBlock + MaxPool. Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    """ConvTranspose2×2 + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch,
                                       kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_ch * 2, out_ch)  

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:],
                                           mode="bilinear",
                                           align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)



class UNetMini(nn.Module):
    def __init__(self,
                 in_channels:  int = 9,
                 out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters   # shorthand: 32

        # Encoder
        self.enc1 = _EncoderBlock(in_channels, f)        # 32
        self.enc2 = _EncoderBlock(f,           f * 2)    # 64
        self.enc3 = _EncoderBlock(f * 2,       f * 4)    # 128

        # Bottleneck
        self.bottleneck = _ConvBlock(f * 4, f * 8)       # 256

        # Decoder
        self.dec3 = _DecoderBlock(f * 8, f * 4)          # 128
        self.dec2 = _DecoderBlock(f * 4, f * 2)          # 64
        self.dec1 = _DecoderBlock(f * 2, f)              # 32

        # Output
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
        # Output bias = -2.0 → sigmoid(-2) ≈ 0.12, matching ~10% polyp prevalence.
        # Prevents the model collapsing to "predict everything" on early epochs.
        nn.init.constant_(self.out_conv.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 9, H, W) input tensor
        Returns:
            (B, 1, H, W) raw logits  ← sigmoid applied by loss / evaluate
        """
        # Encoder
        x, s1 = self.enc1(x)   # s1: (B, 32, H,   W  )
        x, s2 = self.enc2(x)   # s2: (B, 64, H/2, W/2)
        x, s3 = self.enc3(x)   # s3: (B,128, H/4, W/4)

        # Bottleneck
        x = self.bottleneck(x)  #    (B,256, H/8, W/8)

        # Decoder
        x = self.dec3(x, s3)   #    (B,128, H/4, W/4)
        x = self.dec2(x, s2)   #    (B, 64, H/2, W/2)
        x = self.dec1(x, s1)   #    (B, 32, H,   W  )

        return self.out_conv(x) #    (B,  1, H,   W  )