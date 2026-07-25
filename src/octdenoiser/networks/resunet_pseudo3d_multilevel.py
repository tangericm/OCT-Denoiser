"""ResUNet with Pseudo-3D stem extended for multi-level spectral input."""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model
from .resunet_pseudo3d import ResUNetPseudo3D, Pseudo3DStem


class Level2Stem(nn.Module):
    """Process multi-channel sub-window input (Level 2) with 2D convolutions."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(out_ch // 2, 8)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class MultiLevelStem(nn.Module):
    """
    Fuses Level 1 (2-channel dual-window) and Level 2 (sub-window) inputs.

    Input:  [B, 2 + n_sub_channels, H, W]
    Output: [B, out_ch, H, W]

    Level 1 (channels 0:2) is processed by the existing Pseudo3DStem.
    Level 2 (channels 2:) is processed by a Conv2D-based stem.
    Both are fused via a 1x1 convolution.
    """

    def __init__(self, out_ch: int, n_sub_channels: int = 16):
        super().__init__()
        self.n_level1_ch = 2
        self.n_sub_channels = n_sub_channels
        self.level1 = Pseudo3DStem(out_ch=out_ch)
        self.level2 = Level2Stem(in_ch=n_sub_channels, out_ch=out_ch)
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l1 = x[:, : self.n_level1_ch]  # [B, 2, H, W]
        x_l2 = x[:, self.n_level1_ch :]  # [B, n_sub_channels, H, W]
        f1 = self.level1(x_l1)  # [B, out_ch, H, W]
        f2 = self.level2(x_l2)  # [B, out_ch, H, W]
        fused = torch.cat([f1, f2], dim=1)  # [B, 2*out_ch, H, W]
        return self.act(self.bn(self.fuse(fused)))


class ResUNetPseudo3DMultiLevel(ResUNetPseudo3D):
    """
    ResUNet with multi-level spectral input support.

    Inherits the full encoder-decoder from ResUNetPseudo3D but replaces
    the stem with MultiLevelStem to handle both Level 1 (2-channel)
    and Level 2 (sub-window) inputs.

    Input:  [B, 2 + n_sub_channels, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(self, base: int = 64, n_sub_channels: int = 16):
        super().__init__(base=base)
        # Replace the standard Pseudo3DStem with the multi-level version
        self.stem = MultiLevelStem(out_ch=base, n_sub_channels=n_sub_channels)


@register_model("resunet_pseudo3d_multilevel")
def build_resunet_pseudo3d_multilevel(
    *, base: int = 64, n_sub_channels: int = 16
) -> nn.Module:
    return ResUNetPseudo3DMultiLevel(base=base, n_sub_channels=n_sub_channels)
