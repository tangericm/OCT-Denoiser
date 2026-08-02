from __future__ import annotations

import pytest

from octdenoiser.configs.default import TrainConfig


def test_config_rejects_split_that_cannot_make_pairs() -> None:
    config = TrainConfig(inputs=("volume.tiff",), pair_offset=4, group_size=4)
    with pytest.raises(ValueError, match="group_size"):
        config.validate()
