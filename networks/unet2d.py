"""Standard 2D U-Net denoiser — canonical encoder/decoder baseline.

Single-channel in/out. Four resolution levels with double-conv blocks and
skip connections. Predicts the clean image directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1 = DoubleConv(base, base * 2)
        self.d2 = DoubleConv(base * 2, base * 4)
        self.d3 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, in_ch, 1)

    @staticmethod
    def _cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.inc(x)
        s1 = self.d1(self.pool(s0))
        s2 = self.d2(self.pool(s1))
        b = self.d3(self.pool(s2))

        x = self.c3(self._cat(self.u3(b), s2))
        x = self.c2(self._cat(self.u2(x), s1))
        x = self.c1(self._cat(self.u1(x), s0))
        return self.outc(x)


@register_model("unet2d")
def build_unet2d(*, base: int = 64, in_ch: int = 1) -> nn.Module:
    return UNet2D(in_ch=in_ch, base=base)
