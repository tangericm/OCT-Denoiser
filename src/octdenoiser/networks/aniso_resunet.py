"""Anisotropic ResUNet — asymmetric kernels for OCT's asymmetric physics.

Almost every image-restoration backbone is isotropic: square kernels, equal
receptive field in both axes. An OCT B-scan is not isotropic in any respect
that matters.

  * AXIAL (depth, z) resolution comes from the source spectral bandwidth via the
    Fourier relationship -- roughly 8 px FWHM here, and the axis carrying the
    layer structure a clinician reads.
  * LATERAL (x) resolution comes from the beam optics and the scan sampling --
    a different mechanism, a different scale, and on this data a measured
    structure correlation length several times the axial one.

Speckle inherits the same asymmetry: its grain is elongated, not round.

This variant uses kernels taller than they are wide, so the receptive field
grows faster along depth than laterally, and spends parameters where the
structure actually varies. It is the cheapest architectural change here that
encodes something true about OCT rather than importing a natural-image prior.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


def _pad(kernel: tuple[int, int]) -> tuple[int, int]:
    return (kernel[0] // 2, kernel[1] // 2)


class AnisoResBlock(nn.Module):
    def __init__(self, channels: int, kernel: tuple[int, int] = (5, 3)):
        super().__init__()
        p = _pad(kernel)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, padding=p, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel, padding=p, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: tuple[int, int]):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, stride=2, padding=_pad(kernel), bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
        self.b1 = AnisoResBlock(cout, kernel)
        self.b2 = AnisoResBlock(cout, kernel)

    def forward(self, x):
        return self.b2(self.b1(self.op(x)))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int, kernel: tuple[int, int]):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(cin, cout * 4, 1, bias=False), nn.PixelShuffle(2))
        self.fuse = nn.Sequential(
            nn.Conv2d(cout + cskip, cout, kernel, padding=_pad(kernel), bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
        self.b1 = AnisoResBlock(cout, kernel)
        self.b2 = AnisoResBlock(cout, kernel)

    def forward(self, x, skip):
        y = self.up(x)
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.b2(self.b1(self.fuse(torch.cat([y, skip], dim=1))))


class AnisoResUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32,
                 kernel: tuple[int, int] = (5, 3)):
        super().__init__()
        p = _pad(kernel)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel, padding=p, bias=False),
            nn.BatchNorm2d(base), nn.SiLU(inplace=True))
        self.enc = AnisoResBlock(base, kernel)
        self.d1 = Down(base, base * 2, kernel)
        self.d2 = Down(base * 2, base * 4, kernel)
        self.d3 = Down(base * 4, base * 8, kernel)
        self.mid = AnisoResBlock(base * 8, kernel)
        self.u3 = Up(base * 8, base * 4, base * 4, kernel)
        self.u2 = Up(base * 4, base * 2, base * 2, kernel)
        self.u1 = Up(base * 2, base, base, kernel)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        s0 = self.enc(self.stem(x))
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        y = self.mid(self.d3(s2))
        y = self.u3(y, s2)
        y = self.u2(y, s1)
        y = self.u1(y, s0)
        return self.head(y)[:, :, :h, :w]


@register_model("aniso_resunet")
def build_aniso_resunet(*, base: int = 32, in_ch: int = 1,
                        kernel_h: int = 5, kernel_w: int = 3, **_) -> nn.Module:
    return AnisoResUNet(in_ch=in_ch, base=base, kernel=(kernel_h, kernel_w))
