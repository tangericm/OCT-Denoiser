from __future__ import annotations

import torch

from networks.registry import create_model


def test_resunet_multilevel_1d_forward_width1():
    model = create_model("resunet_multilevel_1d", base=16, n_sub_channels=8)
    x = torch.randn(2, 10, 288, 1)  # 2 primary + 8 subband channels
    y = model(x)
    assert y.shape == (2, 1, 288, 1)


def test_resunet_multilevel_1d_forward_width16():
    model = create_model("resunet_multilevel_1d", base=16, n_sub_channels=8)
    x = torch.randn(2, 10, 288, 16)
    y = model(x)
    assert y.shape == (2, 1, 288, 16)
