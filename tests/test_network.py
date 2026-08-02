from __future__ import annotations

import torch

from octdenoiser.networks import create_model, list_models


def test_only_production_model_is_public() -> None:
    assert list_models() == ["nafnet"]


def test_nafnet_handles_non_multiple_image_size() -> None:
    model = create_model("nafnet", base=4, in_ch=1)
    image = torch.randn(1, 1, 19, 21)
    output = model(image)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_production_parameter_counts() -> None:
    base32 = create_model("nafnet", base=32, in_ch=1)
    assert sum(parameter.numel() for parameter in base32.parameters()) == 6_815_457
    del base32
    base64 = create_model("nafnet", base=64, in_ch=1)
    assert sum(parameter.numel() for parameter in base64.parameters()) == 27_110_849
