from __future__ import annotations

import torch

from networks.axial_deconv import AxialDeconvolution
from networks.registry import create_model


def test_axial_deconvolution_forward_1d_shape_with_bscan_baseline():
    mod = AxialDeconvolution(h_kernel=None, lam=1.0e-3, use_bscan_average_baseline=True)
    x = torch.randn(3, 2, 64)
    y = mod(x)
    assert y.shape == (3, 2, 64)


def test_axial_deconvolution_forward_2d_shape():
    mod = AxialDeconvolution(
        h_kernel=torch.ones(64, dtype=torch.complex64),
        lam=1.0e-3,
        learnable_correction=True,
    )
    x = torch.randn(2, 2, 64, 5)
    y = mod(x)
    assert y.shape == (2, 2, 64, 5)


def test_spectrum_unet_1d_forward_uses_axial_deconv():
    model = create_model(
        "spectrum_unet_1d",
        base=16,
        deconv_h=None,
        deconv_lambda=1.0e-3,
        learnable_deconv_correction=True,
        deconv_use_bscan_average_baseline=True,
    )
    x = torch.randn(2, 6, 288)
    y = model(x)
    assert y.shape == (2, 2, 288)
