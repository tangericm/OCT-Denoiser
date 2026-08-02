from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch

from octdenoiser.engine.infer import denoise_path, load_model
from octdenoiser.networks import create_model


def _checkpoint(path: Path, *, include_base: bool) -> None:
    model = create_model("nafnet", base=4, in_ch=1)
    blob: dict[str, object] = {"arch": "nafnet", "in_ch": 1, "model": model.state_dict()}
    if include_base:
        blob["base"] = 4
    torch.save(blob, path)


def test_load_model_infers_base_for_release_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "release.pt"
    _checkpoint(checkpoint, include_base=False)
    model = load_model(checkpoint, torch.device("cpu"))
    assert model.intro.out_channels == 4  # type: ignore[attr-defined]


def test_denoise_tiff_stack_preserves_shape_and_dtype(
    tmp_path: Path,
    processed_volumes: tuple[Path, Path],
) -> None:
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "denoised.tiff"
    _checkpoint(checkpoint, include_base=True)
    denoise_path(
        processed_volumes[0],
        checkpoint,
        output,
        device="cpu",
        amp=False,
    )
    result = tifffile.imread(output)
    assert result.shape == (8, 32, 40)
    assert result.dtype == np.uint16
