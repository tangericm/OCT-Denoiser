"""1D-optimized multi-level ResUNet for single A-line style inputs.

This model keeps the same high-level design as resunet_pseudo3d_multilevel,
but uses anisotropic kernels/strides so processing happens primarily along depth
(H) while preserving width (W), which is typically 1 for single A-line strips.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResBlock1D2D(nn.Module):
    """Residual block using depth-wise anisotropic 2D convs: kernel (3,1)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class Down1D2D(nn.Module):
    """Downsample depth only: stride (2,1)."""

    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.down = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Sequential(*[ResBlock1D2D(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        return self.res(x)


class Up1D2D(nn.Module):
    """Upsample depth only: transposed conv stride (2,1)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=(2, 1),
            stride=(2, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Conv2d(
            out_ch + skip_ch,
            out_ch,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Sequential(*[ResBlock1D2D(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(self.fuse(x)))
        return self.res(x)


class Pseudo3DStem1D2D(nn.Module):
    """Pseudo-3D stem adapted for single A-line style widths (W≈1)."""

    def __init__(self, out_ch: int):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(
            1,
            8,
            kernel_size=(2, 3, 1),
            padding=(0, 1, 0),
            bias=False,
        )
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.act = nn.SiLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(
            8,
            16,
            kernel_size=(1, 3, 1),
            padding=(0, 1, 0),
            bias=False,
        )
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv2d = nn.Conv2d(16, out_ch, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2d = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,1,2,H,W]
        x = self.act(self.bn3d_1(self.conv3d_1(x)))
        x = self.act(self.bn3d_2(self.conv3d_2(x)))
        x = x.squeeze(2)
        x = self.act(self.bn2d(self.conv2d(x)))
        return x


class Level2Stem1D2D(nn.Module):
    """Level-2 stem for sub-window channels using anisotropic convs."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(out_ch // 2, 8)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class MultiLevelStem1D2D(nn.Module):
    """Fuse level-1 and level-2 inputs for 1D-optimized encoder input."""

    def __init__(self, out_ch: int, n_sub_channels: int = 16):
        super().__init__()
        self.n_level1_ch = 2
        self.level1 = Pseudo3DStem1D2D(out_ch=out_ch)
        self.level2 = Level2Stem1D2D(in_ch=n_sub_channels, out_ch=out_ch)
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l1 = x[:, : self.n_level1_ch]
        x_l2 = x[:, self.n_level1_ch :]
        f1 = self.level1(x_l1)
        f2 = self.level2(x_l2)
        fused = torch.cat([f1, f2], dim=1)
        return self.act(self.bn(self.fuse(fused)))


class ResUNetPseudo3DMultiLevel1D(nn.Module):
    """ResUNet variant optimized for depth-only context (single A-line strips)."""

    def __init__(self, base: int = 64, n_sub_channels: int = 16):
        super().__init__()
        self.stem = MultiLevelStem1D2D(out_ch=base, n_sub_channels=n_sub_channels)
        self.enc1 = nn.Sequential(ResBlock1D2D(base), ResBlock1D2D(base))
        self.down1 = Down1D2D(base, base * 2, n_res=2)
        self.down2 = Down1D2D(base * 2, base * 4, n_res=2)
        self.down3 = Down1D2D(base * 4, base * 8, n_res=2)
        self.bot = nn.Sequential(ResBlock1D2D(base * 8), ResBlock1D2D(base * 8))
        self.up2 = Up1D2D(base * 8, base * 4, base * 4, n_res=2)
        self.up1 = Up1D2D(base * 4, base * 2, base * 2, n_res=2)
        self.up0 = Up1D2D(base * 2, base, base, n_res=2)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        s0 = self.enc1(x0)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bot(s3)

        x = self.up2(b, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


@register_model("resunet_pseudo3d_multilevel_1d")
def build_resunet_pseudo3d_multilevel_1d(*, base: int = 64, n_sub_channels: int = 16) -> nn.Module:
    return ResUNetPseudo3DMultiLevel1D(base=base, n_sub_channels=n_sub_channels)
