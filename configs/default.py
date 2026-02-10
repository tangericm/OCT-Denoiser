from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class FolderSpec:
    root_folder: str          # e.g. r"images\Maestro3"
    data_folder: str          # e.g. "6mm_1024Aline"
    pixels: int               # 2048
    alines: int               # 1024 or 2048
    clb_path: Optional[str] = None
    crop_depth: Tuple[int, int] = (1024, 2048)
    do_dc_subtract: bool = True
    window_type: str = "hann"
    use_log: bool = True
    log_eps: float = 1e-6
    apply_fftshift_depth: bool = False
    dispersion: Optional[List[float]] = None
    window_sigma: float = 0.08
    gap: float = 0.15

    def to_preprocess_config(self):
        from preprocess import Config as PreprocessConfig
        return PreprocessConfig(
            pixels=self.pixels,
            alines=self.alines,
            data_folder=self.data_folder,
            do_dc_subtract=self.do_dc_subtract,
            window_type=self.window_type,
            use_log=self.use_log,
            log_eps=self.log_eps,
            crop_depth=self.crop_depth,
            apply_fftshift_depth=self.apply_fftshift_depth,
            window_sigma=self.window_sigma,
            gap=self.gap,
            dispersion=self.dispersion,
        )


@dataclass
class TrainConfig:
    # Paths
    npz_path: Optional[str] = None
    runs_root: str = "runs"
    experiment_name: str = "experiment"

    folder_specs: Optional[List[FolderSpec]] = None
    cache_frames_per_worker: int = 1000

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
    patch_mode: str = "patch"
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
    w_snr_loss: float = 0.0          # smooth SNR loss weight (0 = disabled)
    w_snr_loss_start: Optional[float] = None  # epoch-schedule start weight (None => constant w_snr_loss)
    w_snr_loss_end: Optional[float] = None    # epoch-schedule end weight (None => constant w_snr_loss)
    w_snr_ramp_start_epoch: int = 1           # first epoch where SNR ramp begins
    w_snr_ramp_end_epoch: Optional[int] = None  # epoch where SNR ramp reaches end weight (None => epochs)
    snr_loss_t_peak: float = 0.1     # temperature for soft-peak selection
    snr_loss_t_bg: float = 0.1       # temperature for soft-background selection

    # ROI (y ranges) for SNR/CNR
    snr_sig_y0: int = 111
    snr_sig_y1: int = 600

    # Logging/checkpoint cadence
    val_every: int = 5
    save_every: int = 5

    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_warmup_checks: int = 0

    # Composite validation score (lower is better):
    # score = w_val_loss * norm(val_loss) - w_snr * norm(snr) - w_cnr * norm(cnr)
    score_w_val_loss: float = 1.0
    score_w_snr: float = 1.0
    score_w_cnr: float = 0.0

    # Inference outputs
    tiff_dtype: str = "uint16"
    also_save_float32: bool = False
