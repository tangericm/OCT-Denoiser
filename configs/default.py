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
    window_sigma: float = 0.08
    gap: float = 0.15
    gap_offset: float = 0.0
    n_sub_windows: int = 0            # 0=disabled; e.g. 8 sub-windows per parent (16 total)
    sub_window_spread: float = 2.0    # sub-window center spread in sigma units



@dataclass
class TrainConfig:
    # Paths
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
    w_spec_mag: float = 0.05

    # ROI (y ranges) for SNR/CNR
    snr_sig_y0: int = 111
    snr_sig_y1: int = 600
    snr_sig_stat: str = "max"  # signal statistic for SNR: "max" or "p<percentile>" (e.g. "p95")

    # Logging/checkpoint cadence
    val_every: int = 5
    save_every: int = 5

    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_warmup_checks: int = 0

    # Inference outputs
    tiff_dtype: str = "uint16"
    also_save_float32: bool = False
    save_raw_spectra: bool = False

    # -----------------------------------------------------------------------
    # Physics-guided OCT model (physics_oct_net)
    # -----------------------------------------------------------------------
    # Architecture
    num_blocks: int = 3               # encoder/decoder depth (down/up stages)
    predict_logvar: bool = False      # add heteroscedastic log-variance head
    dispersion_mode: str = "polynomial"   # "polynomial" | "dense"
    dispersion_poly_order: int = 4    # Chebyshev poly order for dispersion head

    # Self-supervised masking
    use_masked_self_supervision: bool = True
    mask_ratio: float = 0.15          # fraction of k-pixels masked per A-line
    mask_span: int = 3                # width of each contiguous mask span
    mask_mode: str = "zero"           # "zero" | "noise" | "mean"

    # Physics loss weights
    # w_charb is reused as the masked-measurement Charbonnier weight
    w_bg_smooth: float = 0.01         # TV regularisation on predicted background
    w_gain_smooth: float = 0.01       # TV regularisation on predicted gain
    w_disp_smooth: float = 0.001      # TV regularisation on dispersion (dense mode only)
    w_depth: float = 0.01             # soft depth-domain consistency weight
    w_var: float = 0.0                # heteroscedastic NLL weight (needs predict_logvar)

