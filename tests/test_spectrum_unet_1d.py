from __future__ import annotations

import torch

from networks.registry import create_model


def test_spectrum_unet_1d_forward_shape():
    model = create_model("spectrum_unet_1d", base=16)
    x = torch.randn(2, 6, 288)
    y = model(x)
    assert y.shape == (2, 2, 288)
