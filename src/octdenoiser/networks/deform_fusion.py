"""Deformable multi-frame fusion — learned alignment, EDVR/BasicVSR++ lineage.

Why this, for this dataset specifically
---------------------------------------
Every scheme compared so far is single-frame in, single-frame out, and they all
sit under the same ceiling: a single B-scan does not contain the information a
50-frame average does. The measured gap is stark -- one network pass is worth
roughly 8-16 averaged frames, but 50-frame averaging still produces a visibly
better image than any of them.

The obvious answer is to feed several frames. The obvious obstacle is that they
are misaligned, and explicit registration is unreliable here: phase correlation
made two of the four Maestro2 stacks WORSE until a "reject worsening shifts"
guard was added, and even then a global rigid shift cannot absorb the tilt and
shear that eye motion leaves within a single scan.

Deformable convolution sidesteps that. Rather than estimating one shift per
frame, the network predicts a per-pixel sampling offset from the feature
difference between each neighbour and the reference frame, then samples the
neighbour at those offsets. Alignment becomes local, learned, and trained
jointly with the denoising objective instead of being a fragile preprocessing
step that can fail silently.

Offsets are initialised at zero, so the module starts as plain averaging and has
to earn any warping it applies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .registry import register_model


class DeformAlign(nn.Module):
    """Align one neighbour's features to the reference frame's."""

    def __init__(self, channels: int, groups: int = 4, kernel: int = 3):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.kernel = kernel
        n = groups * kernel * kernel

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.to_offset = nn.Conv2d(channels, n * 2, 3, padding=1)
        self.to_mask = nn.Conv2d(channels, n, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(channels, channels // 1, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Start as identity: zero offsets means "no warping", so the module
        # begins as plain averaging and must learn to deviate.
        for layer in (self.to_offset, self.to_mask):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, neighbour: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        feat = self.offset_conv(torch.cat([neighbour, reference], dim=1))
        offset = self.to_offset(feat)
        mask = torch.sigmoid(self.to_mask(feat)) * 2.0  # centred on 1 at init
        return deform_conv2d(neighbour, offset, self.weight, self.bias,
                             padding=self.kernel // 2, mask=mask)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class DeformFusionNet(nn.Module):
    """Input [B, K, H, W] of K neighbouring frames; predicts one denoised frame.

    The reference frame is the middle of the stack. Neighbours are aligned to it
    with deformable convolution, fused by a learned attention weight, then
    decoded by a small U-Net.
    """

    def __init__(self, in_ch: int = 5, base: int = 32):
        super().__init__()
        self.n_frames = in_ch
        self.ref_index = in_ch // 2

        self.extract = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.SiLU(inplace=True),
            ResBlock(base))
        self.align = DeformAlign(base)
        # Per-frame, per-pixel fusion weight from similarity to the reference.
        self.attn = nn.Conv2d(base * 2, 1, 3, padding=1)

        self.enc = ResBlock(base)
        self.d1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(base * 2), nn.SiLU(inplace=True), ResBlock(base * 2))
        self.d2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(base * 4), nn.SiLU(inplace=True), ResBlock(base * 4))
        self.mid = ResBlock(base * 4)
        self.u2 = nn.Sequential(nn.Conv2d(base * 4, base * 2 * 4, 1, bias=False), nn.PixelShuffle(2))
        self.f2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False),
                                nn.BatchNorm2d(base * 2), nn.SiLU(inplace=True), ResBlock(base * 2))
        self.u1 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 1, bias=False), nn.PixelShuffle(2))
        self.f1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1, bias=False),
                                nn.BatchNorm2d(base), nn.SiLU(inplace=True), ResBlock(base))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, h, w = x.shape
        ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        feats = [self.extract(x[:, i:i + 1]) for i in range(k)]
        ref = feats[self.ref_index]

        aligned, weights = [], []
        for i, f in enumerate(feats):
            a = ref if i == self.ref_index else self.align(f, ref)
            aligned.append(a)
            weights.append(self.attn(torch.cat([a, ref], dim=1)))
        w_stack = torch.softmax(torch.cat(weights, dim=1), dim=1)
        fused = sum(w_stack[:, i:i + 1] * aligned[i] for i in range(k))

        s0 = self.enc(fused)
        s1 = self.d1(s0)
        y = self.mid(self.d2(s1))
        y = self.f2(torch.cat([self.u2(y), s1], dim=1))
        y = self.f1(torch.cat([self.u1(y), s0], dim=1))
        return self.head(y)[:, :, :h, :w]


@register_model("deform_fusion")
def build_deform_fusion(*, base: int = 32, in_ch: int = 5, **_) -> nn.Module:
    return DeformFusionNet(in_ch=in_ch, base=base)
