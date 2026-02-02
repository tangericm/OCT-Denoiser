from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Paths
    npz_path: str
    runs_root: str
    experiment_name: str

    # Device
    device: str = "cuda"
    amp: bool = True

    # Repro / speed
    seed: int = 42
    deterministic: bool = True

    # Data
    train_frac: float = 0.85
    patch_h: int = 128
    patch_w: int = 128
    patches_per_frame: int = 16
    augment: bool = True
    batch_size: int = 32
    num_workers: int = 8

    # Model selection
    model_name: str = "resunet_pseudo3d"
    base: int = 64

    # Optim
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 5e-5
    grad_clip: float = 1.0

    # Loss weights
    w_charb: float = 0.8
    w_grad: float = 0.5

    # Logging/checkpoint cadence
    val_every: int = 5
    save_every: int = 25

    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_warmup_checks: int = 0

    # Inference outputs (relative to run dir)
    tiff_dtype: str = "uint16"
    also_save_float32: bool = False
