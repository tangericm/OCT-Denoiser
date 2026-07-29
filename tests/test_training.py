from __future__ import annotations

from pathlib import Path

import torch

from octdenoiser.configs.default import TrainConfig
from octdenoiser.engine.train import run_training


def test_one_epoch_training_writes_deployment_checkpoints(
    tmp_path: Path,
    processed_volumes: tuple[Path, Path],
) -> None:
    config = TrainConfig(
        inputs=tuple(str(path) for path in processed_volumes),
        runs_root=str(tmp_path / "runs"),
        experiment_name="smoke",
        base=4,
        group_size=4,
        train_fraction=0.5,
        patch_height=16,
        patch_width=16,
        batch_size=2,
        workers=0,
        epochs=1,
        device="cpu",
        amp=False,
        augment=False,
    )
    run_dir = run_training(config)
    best = run_dir / "checkpoints" / "best.pt"
    final = run_dir / "checkpoints" / "final.pt"
    assert best.is_file()
    assert final.is_file()
    blob = torch.load(best, map_location="cpu", weights_only=True)
    assert blob["arch"] == "nafnet"
    assert blob["base"] == 4
    assert "inputs" not in blob
