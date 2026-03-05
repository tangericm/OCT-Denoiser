"""2D UNet for complex spectrum denoising with lateral A-line context.

Input:  [B, 4, L, W]  (real/imag for w1, real/imag for w2; L=spectral pixels, W=lateral A-lines)
Output: [B, 2, L, W]  (real/imag denoised full-bandwidth spectrum)

Asymmetric stem: wide kernel along spectral axis (L), narrow along lateral (W),
reflecting that spectral coherence spans many pixels while lateral context is
used for structural consistency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResBlock2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DownBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Sequential(ResBlock2d(out_ch), ResBlock2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.bn(self.down(x))))


class UpBlock2d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_ch)
        self.res = nn.Sequential(ResBlock2d(out_ch), ResBlock2d(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))
        return self.res(x)


class SpectrumUNet2D(nn.Module):
    """2D UNet for complex OCT spectrum denoising with lateral context.

    Accepts variable W at inference (sliding-window friendly).
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 2, base: int = 32):
        super().__init__()
        # Asymmetric stem: 15-wide along spectral axis, 3-wide along lateral
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=(15, 3), padding=(7, 1), bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
        )
        self.enc1 = nn.Sequential(ResBlock2d(base), ResBlock2d(base))
        self.down1 = DownBlock2d(base, base * 2)
        self.down2 = DownBlock2d(base * 2, base * 4)
        self.down3 = DownBlock2d(base * 4, base * 8)

        self.bottleneck = nn.Sequential(ResBlock2d(base * 8), ResBlock2d(base * 8))

        self.up2 = UpBlock2d(base * 8, base * 4, base * 4)
        self.up1 = UpBlock2d(base * 4, base * 2, base * 2)
        self.up0 = UpBlock2d(base * 2, base, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, W = x.shape[-2], x.shape[-1]
        # Pad both dims to multiples of 8
        pad_l = (8 - L % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_l > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_l), mode="reflect")

        x0 = self.stem(x)
        s0 = self.enc1(x0)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b = self.bottleneck(s3)

        x = self.up2(b, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        out = self.head(x)
        return out[..., :L, :W]


@register_model("spectrum_unet_2d")
def build_spectrum_unet_2d(*, base: int = 32, **_kw) -> nn.Module:
    return SpectrumUNet2D(in_channels=4, out_channels=2, base=base)
