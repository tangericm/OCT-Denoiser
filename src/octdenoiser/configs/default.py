from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for self-supervised training on processed B-scans."""

    inputs: tuple[str, ...]
    runs_root: str = "runs"
    experiment_name: str = "oct-denoiser"

    model_name: str = "nafnet"
    base: int = 64
    pair_offset: int = 1
    group_size: int = 64
    train_fraction: float = 0.8

    patch_height: int = 256
    patch_width: int = 256
    patches_per_pair: int = 1
    augment: bool = True

    batch_size: int = 4
    workers: int = 4
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    w_charb: float = 0.8
    w_grad: float = 0.5
    val_every: int = 1
    patience: int = 10
    min_delta: float = 0.0

    device: str = "auto"
    amp: bool = True
    seed: int = 0
    deterministic: bool = False

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("At least one B-scan input path is required.")
        if self.model_name != "nafnet":
            raise ValueError("The public pipeline supports the production NAFNet model only.")
        if self.base <= 0:
            raise ValueError("base must be positive.")
        if self.pair_offset <= 0:
            raise ValueError("pair_offset must be positive.")
        if self.group_size <= self.pair_offset:
            raise ValueError("group_size must be larger than pair_offset.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("Patch dimensions must be positive.")
        if self.patches_per_pair <= 0:
            raise ValueError("patches_per_pair must be positive.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.workers < 0:
            raise ValueError("workers cannot be negative.")
        if self.val_every <= 0:
            raise ValueError("val_every must be positive.")
