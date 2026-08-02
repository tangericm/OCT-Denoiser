"""NAFNet image-restoration architecture (Chen et al., ECCV 2022)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over ``[B, C, H, W]``."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(self.norm1(x)))
        y = self.sg(y)
        y = y * self.sca(y)
        x = x + self.conv3(y) * self.beta

        y = self.sg(self.conv4(self.norm2(x)))
        return x + self.conv5(y) * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base: int = 32,
        enc_blocks: Sequence[int] = (1, 1, 1, 2),
        middle_blocks: int = 2,
        dec_blocks: Sequence[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, base, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        c = base
        for n_blocks in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])

        for n_blocks in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n_blocks)]))

        self.ending = nn.Conv2d(base, out_ch, 3, padding=1)
        self.padder = 2 ** len(enc_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_h = (self.padder - height % self.padder) % self.padder
        pad_w = (self.padder - width % self.padder) % self.padder
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        y = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs, strict=True):
            y = encoder(y)
            skips.append(y)
            y = down(y)
        y = self.middle(y)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1], strict=True):
            y = up(y)
            y = y + skip
            y = decoder(y)
        out = self.ending(y)
        return out[:, :, :height, :width]


@register_model("nafnet")
def build_nafnet(*, base: int = 32, in_ch: int = 1, **_: object) -> nn.Module:
    return NAFNet(in_ch=in_ch, out_ch=1, base=base)
