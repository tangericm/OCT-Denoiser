"""Pseudo-2D-stem 1D ResUNet for complex OCT spectrum denoising.

Directly mirrors ResUNetPseudo3D in structure, adapted for 1D spectral inputs
with a physics-informed IFFT/FFT encoding branch.

Input:  [B, 4, L]  (w1_re, w1_im, w2_re, w2_im)
Output: [B, 2, L]  (re, im of denoised full-bandwidth complex spectrum)

Architecture map (mirrors ResUNetPseudo3D):
  Pseudo2DStem  →  enc1  →  down1  →  down2  →  down3
                                                   ↓
                                                  bot
  head  ←  up0  ←  up1  ←  up2  ←──────────────┘
              ↑        ↑       ↑
           s0+phys    s1      s2

Physics branch: at enc1 resolution, IFFT features into depth domain,
refine with a small 1D CNN, FFT back — injected as an additive skip on s0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# Building blocks (1D analogs of the 2D blocks in resunet_pseudo3d)
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
# Pseudo2DStem — 1D analog of Pseudo3DStem
# ---------------------------------------------------------------------------

class Pseudo2DStem(nn.Module):
    """
    Fuse two complex sub-band spectra via a Conv2d that treats the window
    index as a collapsible 'depth' dimension — exactly as Pseudo3DStem uses
    Conv3d(k=(2,3,3)) to collapse two B-scan slices into one feature map.

    The first 4 input channels (w1_re, w1_im, w2_re, w2_im) go through the
    pseudo-2D path. Any remaining channels (e.g. w1_mask, w2_mask) are
    concatenated after the squeeze before the final Conv1d.

    Input:  x [B, in_ch, L]  — in_ch >= 4; first 4 are (w1_re, w1_im, w2_re, w2_im)

    Complex path on x[:, :4]:
      view → [B, 2(re/im), 2(windows), L]
      Conv2d(2→8,  k=(2,3), pad=(0,1))  collapses window dim → [B, 8,  1, L]
      Conv2d(8→16, k=(1,3), pad=(0,1))  refines along L      → [B, 16, 1, L]
      squeeze → [B, 16, L]
      cat with x[:, 4:]                                       → [B, 16+extra, L]
      Conv1d(16+extra→out_ch, k=3)                            → [B, out_ch, L]
    """
    def __init__(self, out_ch: int, in_ch: int = 4):
        super().__init__()
        extra_ch = in_ch - 4  # channels beyond the 4 complex ones
        # k=(2,3): height-2 collapses the two sub-band windows, width-3 smooths L
        self.conv2d_1 = nn.Conv2d(2,  8,  kernel_size=(2, 3), padding=(0, 1), bias=False)
        self.bn2d_1   = nn.BatchNorm2d(8)
        self.act      = nn.SiLU(inplace=True)
        self.conv2d_2 = nn.Conv2d(8,  16, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2d_2   = nn.BatchNorm2d(16)
        self.conv1d   = nn.Conv1d(16 + extra_ch, out_ch, 3, padding=1, bias=False)
        self.bn1d     = nn.BatchNorm1d(out_ch)
        self.extra_ch = extra_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        cplx  = x[:, :4]                                     # [B, 4, L]
        extra = x[:, 4:] if self.extra_ch > 0 else None      # [B, extra_ch, L] or None
        # Pseudo-2D fusion of the two complex sub-band spectra
        # view maps: ch0→[:,0,0,:]=w1_re, ch1→[:,0,1,:]=w1_im,
        #            ch2→[:,1,0,:]=w2_re, ch3→[:,1,1,:]=w2_im
        cplx = cplx.view(B, 2, 2, L)                         # [B, 2(re/im), 2(windows), L]
        cplx = self.act(self.bn2d_1(self.conv2d_1(cplx)))    # [B, 8, 1, L]
        cplx = self.act(self.bn2d_2(self.conv2d_2(cplx)))    # [B, 16, 1, L]
        cplx = cplx.squeeze(2)                                # [B, 16, L]
        if extra is not None:
            cplx = torch.cat([cplx, extra], dim=1)            # [B, 16+extra_ch, L]
        return self.act(self.bn1d(self.conv1d(cplx)))         # [B, out_ch, L]


# ---------------------------------------------------------------------------
# Physics-informed depth refinement block
# ---------------------------------------------------------------------------

class DepthRefineBlock(nn.Module):
    """
    IFFT → depth-domain CNN refinement → FFT bridge.

    Treats interleaved re/im channel pairs as complex features, maps them
    into the depth domain via IFFT, refines with a small 1D CNN (where the
    network can learn coherence structure, layer reflections, speckle), then
    maps back to spectral domain via FFT. The result is added onto the skip
    connection at enc1 resolution.

    in_ch must be even (paired as re/im).
    """
    def __init__(self, in_ch: int):
        if in_ch % 2 != 0:
            raise ValueError("in_ch must be even for re/im complex pairing")
        super().__init__()
        half = in_ch // 2
        self.refine = nn.Sequential(
            nn.Conv1d(half, half, 5, padding=2, bias=False),
            nn.BatchNorm1d(half),
            nn.SiLU(inplace=True),
            ResBlock1d(half),
            ResBlock1d(half),
        )

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        # Interleaved re/im pairs → complex: ch 0,2,4,... are real; 1,3,5,... are imag
        return torch.complex(x[:, 0::2].float(), x[:, 1::2].float())

    @staticmethod
    def _from_complex(x: torch.Tensor) -> torch.Tensor:
        # complex → interleaved re/im pairs
        return torch.stack([x.real, x.imag], dim=2).flatten(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, L]
        cplx  = self._to_complex(x)                             # [B, in_ch//2, L] complex
        depth = torch.fft.ifft(cplx, dim=-1).abs()              # [B, in_ch//2, L] real magnitude
        depth = self.refine(depth)                               # [B, in_ch//2, L]
        spec  = torch.fft.fft(depth.to(torch.complex64), dim=-1) # [B, in_ch//2, L] complex
        return self._from_complex(spec)                          # [B, in_ch, L]


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class SpectrumResUNet1D(nn.Module):
    """
    1D spectrum-domain ResUNet with Pseudo2DStem and physics-informed IFFT/FFT.

    Mirrors ResUNetPseudo3D exactly:
      - Same stem → enc1 → 3× down → bot → 3× up → head structure
      - Same n_res=2 ResBlocks at each encoder/decoder stage
      - Same additive physics skip (depth_refine) fused at the highest-res skip
    """
    def __init__(self, base: int = 64, in_ch: int = 6):
        super().__init__()
        if base % 2 != 0:
            raise ValueError("base must be even for re/im complex pairing")
        if in_ch < 4:
            raise ValueError("in_ch must be >= 4 (w1_re, w1_im, w2_re, w2_im)")

        self.stem  = Pseudo2DStem(out_ch=base, in_ch=in_ch)
        self.enc1  = nn.Sequential(ResBlock1d(base), ResBlock1d(base))

        self.down1 = Down1d(base,     base * 2, n_res=2)
        self.down2 = Down1d(base * 2, base * 4, n_res=2)
        self.down3 = Down1d(base * 4, base * 8, n_res=2)
        self.bot   = nn.Sequential(ResBlock1d(base * 8), ResBlock1d(base * 8))

        # Physics bridge: applied at full enc1 resolution, injected back into up0 skip
        self.depth_refine = DepthRefineBlock(in_ch=base)

        self.up2 = Up1d(base * 8, base * 4, base * 4, n_res=2)
        self.up1 = Up1d(base * 4, base * 2, base * 2, n_res=2)
        self.up0 = Up1d(base * 2, base,     base,     n_res=2)

        # Output: re + im of denoised full-bandwidth complex spectrum
        self.head = nn.Conv1d(base, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        # Pad to multiple of 8 so 3 stride-2 downs stay aligned
        pad = (8 - L % 8) % 8
        if pad > 0:
            x = F.pad(x, (0, pad), mode="reflect")

        x0 = self.stem(x)           # [B, base,   L]
        s0 = self.enc1(x0)          # [B, base,   L]   — skip for up0
        s1 = self.down1(s0)         # [B, 2*base, L/2] — skip for up1
        s2 = self.down2(s1)         # [B, 4*base, L/4] — skip for up2
        s3 = self.down3(s2)         # [B, 8*base, L/8]
        b  = self.bot(s3)           # [B, 8*base, L/8]

        # Physics: IFFT enc1 features to depth domain, refine, FFT back
        phys = self.depth_refine(s0)  # [B, base, L]

        x = self.up2(b,  s2)          # [B, 4*base, L/4]
        x = self.up1(x,  s1)          # [B, 2*base, L/2]
        x = self.up0(x, s0 + phys)    # [B, base,   L]   — fused physics skip
        return self.head(x)[..., :L]  # [B, 2,      L]


@register_model("spectrum_resunet_1d")
def build_spectrum_resunet_1d(*, base: int = 64, in_ch: int = 6, **_kw) -> nn.Module:
    return SpectrumResUNet1D(base=base, in_ch=in_ch)
