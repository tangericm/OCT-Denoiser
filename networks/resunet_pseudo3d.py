from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

class ResBlock2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Sequential(*[ResBlock2D(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        return self.res(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.res = nn.Sequential(*[ResBlock2D(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1) # Skip connection concatination
        x = self.act(self.bn2(self.fuse(x)))
        return self.res(x)

# class FiLM(nn.Module):
#     def __init__(self, ch: int, hidden: int = 64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, hidden),
#             nn.SiLU(),
#             nn.Linear(hidden, 2 * ch),
#         )

#     def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
#         gam_bet = self.net(g)
#         gamma, beta = gam_bet.chunk(2, dim=1)
#         return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

class Pseudo3DStem(nn.Module):
    """
    Mix the two input B-scans with a small 3D conv stack, then collapse to 2D features.

    Input:  x [B,2,H,W]
    Treat as volume [B,1,D=2,H,W] so 3D conv sees the inter-input offset like "pseudo-3D".
    """
    def __init__(self, out_ch: int):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(2, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.act = nn.SiLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv2d = nn.Conv2d(16, out_ch, 3, padding=1, bias=False)
        self.bn2d = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,1,2,H,W]
        x = self.act(self.bn3d_1(self.conv3d_1(x)))  # [B,8,1,H,W]
        x = self.act(self.bn3d_2(self.conv3d_2(x)))  # [B,16,1,H,W]
        x = x.squeeze(2)                              # [B,16,H,W]
        x = self.act(self.bn2d(self.conv2d(x)))       # [B,out_ch,H,W]
        return x

class ResUNetPseudo3D(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()

        self.stem = Pseudo3DStem(out_ch=base)
        self.enc1 = nn.Sequential(ResBlock2D(base), ResBlock2D(base))
        self.down1 = Down(base, base * 2, n_res=2)
        self.down2 = Down(base * 2, base * 4, n_res=2)
        self.down3 = Down(base * 4, base * 8, n_res=2)
        self.bot = nn.Sequential(ResBlock2D(base * 8), ResBlock2D(base * 8))
        self.up2 = Up(base * 8, base * 4, base * 4, n_res=2)
        self.up1 = Up(base * 4, base * 2, base * 2, n_res=2)
        self.up0 = Up(base * 2, base, base, n_res=2)
        self.head = nn.Conv2d(base, 1, 1)


    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
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

@register_model("resunet_pseudo3d")
def build_resunet_pseudo3d(*, base: int = 64, **_) -> nn.Module:
    return ResUNetPseudo3D(base=base)
