"""1D-optimized multi-level ResUNet for single A-line style inputs.

This architecture follows the existing multi-level ResUNet design but uses
anisotropic operations that emphasize depth-axis context while preserving
lateral width for very narrow strips (including width=1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResidualBlock(nn.Module):
    """Residual block with depth-wise anisotropic kernels."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DownsampleBlock(nn.Module):
    """Downsample depth only (stride=(2,1)), preserve width."""

    def __init__(self, in_channels: int, out_channels: int, n_res: int = 2):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.residual_stack = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        return self.residual_stack(x)


class UpsampleBlock(nn.Module):
    """Upsample depth only (stride=(2,1)), preserve width."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, n_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(2, 1),
            stride=(2, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )
        self.bn_fuse = nn.BatchNorm2d(out_channels)
        self.residual_stack = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))
        return self.residual_stack(x)


class DualInputStem(nn.Module):
    """Stem for the first two channels using lightweight pseudo-3D mixing."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(2, 3, 1), padding=(0, 1, 0), bias=False)
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.act = nn.SiLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False)
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv2d = nn.Conv2d(16, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2d = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,1,2,H,W]
        x = self.act(self.bn3d_1(self.conv3d_1(x)))
        x = self.act(self.bn3d_2(self.conv3d_2(x)))
        x = x.squeeze(2)
        return self.act(self.bn2d(self.conv2d(x)))


class SubbandStem(nn.Module):
    """Stem for additional multi-level subband channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = max(out_channels // 2, 8)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(x)))


class MultiLevelStem(nn.Module):
    """Fuse dual-input channels and subband channels into shared features."""

    def __init__(self, out_channels: int, n_sub_channels: int = 16):
        super().__init__()
        self.n_primary_channels = 2
        self.primary_stem = DualInputStem(out_channels=out_channels)
        self.subband_stem = SubbandStem(in_channels=n_sub_channels, out_channels=out_channels)
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_primary = x[:, : self.n_primary_channels]
        x_subband = x[:, self.n_primary_channels :]
        f_primary = self.primary_stem(x_primary)
        f_subband = self.subband_stem(x_subband)
        fused = torch.cat([f_primary, f_subband], dim=1)
        return self.act(self.bn(self.fuse(fused)))


class ResUNetMultiLevel1D(nn.Module):
    """Multi-level ResUNet optimized for single A-line style input strips."""

    def __init__(self, base: int = 64, n_sub_channels: int = 16):
        super().__init__()
        self.stem = MultiLevelStem(out_channels=base, n_sub_channels=n_sub_channels)
        self.enc1 = nn.Sequential(ResidualBlock(base), ResidualBlock(base))
        self.down1 = DownsampleBlock(base, base * 2, n_res=2)
        self.down2 = DownsampleBlock(base * 2, base * 4, n_res=2)
        self.down3 = DownsampleBlock(base * 4, base * 8, n_res=2)
        self.bottleneck = nn.Sequential(ResidualBlock(base * 8), ResidualBlock(base * 8))
        self.up2 = UpsampleBlock(base * 8, base * 4, base * 4, n_res=2)
        self.up1 = UpsampleBlock(base * 4, base * 2, base * 2, n_res=2)
        self.up0 = UpsampleBlock(base * 2, base, base, n_res=2)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        s0 = self.enc1(x0)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)

        x = self.up2(b, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


@register_model("resunet_multilevel_1d")
def build_resunet_multilevel_1d(*, base: int = 64, n_sub_channels: int = 16) -> nn.Module:
    return ResUNetMultiLevel1D(base=base, n_sub_channels=n_sub_channels)
