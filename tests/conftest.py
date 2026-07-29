from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture
def processed_volumes(tmp_path: Path) -> tuple[Path, Path]:
    height, width = 32, 40
    yy, xx = np.mgrid[:height, :width]
    paths = []
    for volume_index in range(2):
        frames = []
        for frame_index in range(8):
            image = 1200 + 300 * np.sin((xx + frame_index) / 6) + 500 * np.exp(
                -((yy - 15 - volume_index) ** 2) / 30
            )
            frames.append(np.clip(image, 0, 65535).astype(np.uint16))
        path = tmp_path / f"volume_{volume_index}.tiff"
        tifffile.imwrite(path, np.stack(frames), photometric="minisblack")
        paths.append(path)
    return paths[0], paths[1]
