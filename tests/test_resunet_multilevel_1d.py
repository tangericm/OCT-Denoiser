"""Forward pass shape and finite-output tests for all registered ResUNet models."""
from __future__ import annotations

import torch
from networks.registry import create_model


def test_resunet_pseudo3d_forward():
    """Standard 2-channel model: [B, 2, H, W] -> [B, 1, H, W]."""
    model = create_model("resunet_pseudo3d", base=16)
    x = torch.randn(2, 2, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64), f"Unexpected output shape: {y.shape}"
    assert torch.isfinite(y).all(), "Output contains non-finite values"


def test_resunet_pseudo3d_multilevel_forward():
    """Multi-level model: [B, 2+n_sub, H, W] -> [B, 1, H, W]."""
    n_sub = 4
    model = create_model("resunet_pseudo3d_multilevel", base=16, n_sub_channels=n_sub)
    x = torch.randn(2, 2 + n_sub, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64), f"Unexpected output shape: {y.shape}"
    assert torch.isfinite(y).all(), "Output contains non-finite values"


def test_resunet_pseudo3d_multilevel_strip():
    """Multi-level model on A-line strips (W=1) used in strip patch_mode."""
    n_sub = 4
    model = create_model("resunet_pseudo3d_multilevel", base=16, n_sub_channels=n_sub)
    x = torch.randn(2, 2 + n_sub, 288, 1)
    y = model(x)
    assert y.shape == (2, 1, 288, 1), f"Unexpected output shape: {y.shape}"
    assert torch.isfinite(y).all(), "Output contains non-finite values"
