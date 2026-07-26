"""Dual-domain ResUNet: Fast Fourier Convolution blocks (Chi et al., NeurIPS 2020).

Physics motivation specific to OCT
----------------------------------
The data ORIGINATES as a spectrum. Axial resolution is set directly by the
spectral window, and the reconstruction is an IFFT along k. A network with an
explicit frequency-domain branch is therefore operating in a basis the signal
was actually formed in, rather than one imposed on it.

Practically, an FFC block sees the ENTIRE image at every layer: the Fourier
branch's receptive field is global by construction. That is aimed at the
vertical streaking visible in the current method's output, which a stack of
3x3 convolutions has no efficient way to suppress -- streaks are long-range
structure and local kernels only see them a few pixels at a time.

The repository previously contained spectrum-domain networks that were removed.
They were trained under the leaking full-band target; this revisits the idea
with the supervision defect fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class FourierUnit(nn.Module):
    """Convolve in the frequency domain: rfft2 -> 1x1 conv on (real, imag) -> irfft2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch * 2, out_ch * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # The WHOLE frequency branch runs in float32, not just the transforms.
        # Casting only the rfft2 input is not enough: self.conv still executes
        # under autocast and returns bf16, and torch.complex rejects bf16 --
        # "Expected both inputs to be Half, Float or Double tensors". Disabling
        # autocast for the block also keeps phase out of a 8-bit mantissa,
        # which is the reason for the float32 in the first place.
        with torch.amp.autocast(x.device.type, enabled=False):
            xf = x.float()
            ffted = torch.fft.rfft2(xf, norm="ortho")
            ffted = torch.cat([ffted.real, ffted.imag], dim=1)
            ffted = F.relu(self.bn(self.conv(ffted)), inplace=True)
            real, imag = ffted.chunk(2, dim=1)
            out = torch.fft.irfft2(torch.complex(real, imag), s=(h, w), norm="ortho")
        return out.to(x.dtype)


class FFCBlock(nn.Module):
    """Split channels into a local (conv) branch and a global (Fourier) branch."""

    def __init__(self, channels: int, global_ratio: float = 0.5):
        super().__init__()
        self.c_global = max(1, int(channels * global_ratio))
        self.c_local = channels - self.c_global

        self.local_conv = nn.Conv2d(self.c_local, self.c_local, 3, padding=1, bias=False)
        self.local_bn = nn.BatchNorm2d(self.c_local)
        self.global_unit = FourierUnit(self.c_global, self.c_global)
        # Let the two branches exchange information.
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl, xg = torch.split(x, [self.c_local, self.c_global], dim=1)
        yl = F.silu(self.local_bn(self.local_conv(xl)))
        yg = self.global_unit(xg)
        y = self.bn(self.fuse(torch.cat([yl, yg], dim=1)))
        return F.silu(x + y)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
        self.block = FFCBlock(cout)

    def forward(self, x):
        return self.block(self.op(x))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(cin, cout * 4, 1, bias=False), nn.PixelShuffle(2))
        self.fuse = nn.Sequential(nn.Conv2d(cout + cskip, cout, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(cout), nn.SiLU(inplace=True))
        self.block = FFCBlock(cout)

    def forward(self, x, skip):
        y = self.up(x)
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.fuse(torch.cat([y, skip], dim=1)))


class FFCResUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base), nn.SiLU(inplace=True))
        self.enc = FFCBlock(base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.mid = FFCBlock(base * 8)
        self.u3 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u1 = Up(base * 2, base, base)
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


@register_model("ffc_resunet")
def build_ffc_resunet(*, base: int = 32, in_ch: int = 1, **_) -> nn.Module:
    return FFCResUNet(in_ch=in_ch, base=base)
