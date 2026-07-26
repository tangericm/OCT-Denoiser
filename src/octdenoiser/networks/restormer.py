"""Restormer — transformer restoration with transposed attention (CVPR 2022).

Attention is computed across CHANNELS rather than spatial positions, so cost is
linear in pixel count instead of quadratic. That is what makes a transformer
usable at OCT B-scan sizes at all: standard spatial self-attention over
1024x2048 is out of the question, and patch tokenisation would destroy exactly
the fine speckle-scale detail under study.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nafnet import LayerNorm2d
from .registry import register_model


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention: a C x C attention map, not HW x HW."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project(out)


class GDFN(nn.Module):
    """Gated-Dconv feed-forward: depthwise conv plus a multiplicative gate."""

    def __init__(self, dim: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Restormer(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32,
                 blocks=(2, 3, 3, 4), heads=(1, 2, 4, 8)):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, base, 3, padding=1, bias=False)
        dims = [base * (2 ** i) for i in range(len(blocks))]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(len(blocks) - 1):
            self.encoders.append(nn.Sequential(
                *[TransformerBlock(dims[i], heads[i]) for _ in range(blocks[i])]))
            self.downs.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1] // 4, 3, padding=1, bias=False),
                nn.PixelUnshuffle(2)))

        self.middle = nn.Sequential(
            *[TransformerBlock(dims[-1], heads[-1]) for _ in range(blocks[-1])])

        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(blocks) - 1, 0, -1):
            self.ups.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i - 1] * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2)))
            self.reduces.append(nn.Conv2d(dims[i - 1] * 2, dims[i - 1], 1, bias=False))
            self.decoders.append(nn.Sequential(
                *[TransformerBlock(dims[i - 1], heads[i - 1]) for _ in range(blocks[i - 1])]))

        self.output = nn.Conv2d(dims[0], out_ch, 3, padding=1, bias=False)
        self.padder = 2 ** (len(blocks) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        y = self.patch(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs, strict=True):
            y = enc(y)
            skips.append(y)
            y = down(y)
        y = self.middle(y)
        for up, red, dec, skip in zip(self.ups, self.reduces, self.decoders, skips[::-1], strict=True):
            y = up(y)
            y = dec(red(torch.cat([y, skip], dim=1)))
        out = self.output(y)
        return out[:, :, :h, :w]


@register_model("restormer")
def build_restormer(*, base: int = 32, in_ch: int = 1, **_) -> nn.Module:
    return Restormer(in_ch=in_ch, out_ch=1, base=base)
