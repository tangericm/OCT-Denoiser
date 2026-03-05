"""1D UNet for complex spectrum denoising.

Input:  [B, 4, L]  (real/imag for w1, real/imag for w2)
Output: [B, 2, L]  (real/imag denoised full-bandwidth spectrum)

Designed for OCT spectral-domain denoising where each A-line spectrum
is processed independently as a 1D signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResBlock1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DownBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Sequential(ResBlock1d(out_ch), ResBlock1d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.bn(self.down(x))))


class UpBlock1d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(out_ch)
        self.res = nn.Sequential(ResBlock1d(out_ch), ResBlock1d(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))
        return self.res(x)


class SpectrumUNet1D(nn.Module):
    """1D UNet for complex OCT spectrum denoising."""

    def __init__(self, in_channels: int = 4, out_channels: int = 2, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, padding=7, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU(inplace=True),
        )
        self.enc1 = nn.Sequential(ResBlock1d(base), ResBlock1d(base))
        self.down1 = DownBlock1d(base, base * 2)
        self.down2 = DownBlock1d(base * 2, base * 4)
        self.down3 = DownBlock1d(base * 4, base * 8)

        self.bottleneck = nn.Sequential(ResBlock1d(base * 8), ResBlock1d(base * 8))

        self.up2 = UpBlock1d(base * 8, base * 4, base * 4)
        self.up1 = UpBlock1d(base * 4, base * 2, base * 2)
        self.up0 = UpBlock1d(base * 2, base, base)
        self.head = nn.Conv1d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        # Pad to multiple of 8 for clean downsampling
        pad_total = (8 - L % 8) % 8
        if pad_total > 0:
            x = F.pad(x, (0, pad_total), mode="reflect")

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
        return out[..., :L]


@register_model("spectrum_unet_1d")
def build_spectrum_unet_1d(*, base: int = 64, **_kw) -> nn.Module:
    return SpectrumUNet1D(in_channels=4, out_channels=2, base=base)
