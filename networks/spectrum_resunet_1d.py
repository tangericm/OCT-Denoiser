"""1D ResUNet for real OCT spectrum denoising.

Input:  [B, 2, L]  (spec_w1, spec_w2) — real-valued windowed spectra
Output: [B, 1, L]  — real-valued denoised full-bandwidth spectrum

Architecture:
  Stem (Conv1d 2→base)  →  enc1  →  down1  →  down2  →  down3
                                                            ↓
                                                           bot
  head  ←  up0  ←  up1  ←  up2  ←──────────────────────┘
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(ch)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class Down1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.SiLU(inplace=True)
        self.res  = nn.Sequential(*[ResBlock1d(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.bn(self.down(x))))


class Up1d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_res: int = 2):
        super().__init__()
        self.up      = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn      = nn.BatchNorm1d(out_ch)
        self.act     = nn.SiLU(inplace=True)
        self.fuse    = nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn_fuse = nn.BatchNorm1d(out_ch)
        self.res     = nn.Sequential(*[ResBlock1d(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn_fuse(self.fuse(x)))
        return self.res(x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SpectrumResUNet1D(nn.Module):
    """
    1D ResUNet for real spectrum denoising.

    Input:  [B, 2, L]  — (spec_w1, spec_w2) real-valued windowed spectra
    Output: [B, 1, L]  — denoised full-bandwidth real spectrum
    """
    def __init__(self, base: int = 64):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv1d(2, base, 3, padding=1, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU(inplace=True),
        )
        self.enc1  = nn.Sequential(ResBlock1d(base), ResBlock1d(base))

        self.down1 = Down1d(base,     base * 2, n_res=2)
        self.down2 = Down1d(base * 2, base * 4, n_res=2)
        self.down3 = Down1d(base * 4, base * 8, n_res=2)
        self.bot   = nn.Sequential(ResBlock1d(base * 8), ResBlock1d(base * 8))

        self.up2 = Up1d(base * 8, base * 4, base * 4, n_res=2)
        self.up1 = Up1d(base * 4, base * 2, base * 2, n_res=2)
        self.up0 = Up1d(base * 2, base,     base,     n_res=2)

        self.head = nn.Conv1d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        # Pad to multiple of 8 so 3 stride-2 downs stay aligned
        pad = (8 - L % 8) % 8
        if pad > 0:
            x = F.pad(x, (0, pad), mode="reflect")

        x0 = self.stem(x)    # [B, base,   L]
        s0 = self.enc1(x0)   # [B, base,   L]   — skip for up0
        s1 = self.down1(s0)  # [B, 2*base, L/2] — skip for up1
        s2 = self.down2(s1)  # [B, 4*base, L/4] — skip for up2
        s3 = self.down3(s2)  # [B, 8*base, L/8]
        b  = self.bot(s3)    # [B, 8*base, L/8]

        x = self.up2(b,  s2)  # [B, 4*base, L/4]
        x = self.up1(x,  s1)  # [B, 2*base, L/2]
        x = self.up0(x,  s0)  # [B, base,   L]
        return self.head(x)[..., :L]  # [B, 1, L]


@register_model("spectrum_resunet_1d")
def build_spectrum_resunet_1d(*, base: int = 64, **_kw) -> nn.Module:
    return SpectrumResUNet1D(base=base)
