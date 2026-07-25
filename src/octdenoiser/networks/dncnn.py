"""DnCNN — standard residual CNN denoiser (Zhang et al., 2017).

Canonical baseline: a stack of Conv-BN-ReLU layers predicting the residual
(noise) component, with a global skip so the network output is the denoised
estimate. Input and output are single-channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


class DnCNN(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64, depth: int = 17):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(base, base, 3, padding=1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(base, in_ch, 3, padding=1, bias=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict residual; global skip yields the denoised estimate.
        return x - self.body(x)


@register_model("dncnn")
def build_dncnn(*, base: int = 64, in_ch: int = 1, depth: int = 17) -> nn.Module:
    return DnCNN(in_ch=in_ch, base=base, depth=depth)
