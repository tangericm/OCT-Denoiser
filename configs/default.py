from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class TrainConfig:
    # Paths
    npz_path: str
    runs_root: str
    experiment_name: str

    folder_specs: Optional[List["FolderSpec"]] = None
    cache_frames_per_worker: int = 2
    
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
    patch_mode: str = "patch"  # "strip" (random x, full depth) or "patch" (random x and y)
    augment: bool = True
    batch_size: int = 32
    num_workers: int = 8
    cache_frames_per_worker: int = 1000,

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

    # ROI (y ranges) for SNR/CNR loss
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
    # score = (score_w_val_loss * val_loss) - (score_w_snr * val_snr) - (score_w_cnr * val_cnr)
    # Increase score_w_val_loss to emphasize loss, or increase score_w_snr/score_w_cnr
    # to prioritize higher SNR/CNR in checkpoint selection.
    # score_norm can be:
    #   - "none": use raw metrics in score
    #   - "baseline_relative": use (metric - baseline) / (abs(baseline) + score_norm_eps)
    # Baseline values are taken at the first validation pass in each training run.
    score_norm: str = "baseline_relative"
    score_norm_eps: float = 1e-8
    score_w_val_loss: float = 1.0
    score_w_snr: float = 1.0
    score_w_cnr: float = 0.0

    # Inference outputs (relative to run dir)
    tiff_dtype: str = "uint16"
    also_save_float32: bool = False

@dataclass
class FolderSpec:
    root_folder: str          # e.g. r"images\Maestro3"
    data_folder: str          # e.g. "6mm_1024Aline"
    pixels: int               # 2048
    alines: int               # 1024 or 2048
    clb_path: Optional[str] = None  # optional override; if None use existing BscanProcessor logic
    crop_depth: Tuple[int,int] = (1024, 2048)
    do_dc_subtract: bool = True
    window_type: str = "hann"
    use_log: bool = True
    log_eps: float = 1e-6
    apply_fftshift_depth: bool = False
    dispersion: Optional[List[float]] = None

    # the sweep knobs you care about
    window_sigma: float = 0.08
    gap: float = 0.15
