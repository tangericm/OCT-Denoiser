from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class FolderSpec:
    """Per-dataset specification used by BscanProcessor and the data pipeline."""
    root_folder: str                        # e.g. r"images\Maestro3"
    data_folder: str                        # e.g. "6mm_1024Aline"
    pixels: int                             # spectral samples per A-line (e.g. 2048)
    alines: int                             # A-lines per B-scan (e.g. 1024 or 2048)
    clb_path: Optional[str] = None          # override CLB path; auto-discovered if None
    crop_depth: Tuple[int, int] = (1024, 2048)  # [z0, z1) pixel crop after IFFT
    do_dc_subtract: bool = True
    use_log: bool = True
    log_eps: float = 1e-6
    apply_fftshift_depth: bool = False
    window_sigma: float = 0.08              # Gaussian width for spectral windows
    gap: float = 0.15                       # center separation of the two windows
    gap_offset: float = 0.0                 # shared offset for both window centers
    n_sub_windows: int = 0                  # sub-windows per parent; 0 = disabled
    sub_window_spread: float = 2.0          # sub-window center spread in sigma units



@dataclass
class TrainConfig:
    """Training configuration. All hyperparameters live here."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    runs_root: str = "runs"
    experiment_name: str = "experiment"

    folder_specs: Optional[List[FolderSpec]] = None
    cache_frames_per_worker: int = 1000     # LRU cache size per DataLoader worker

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: str = "cuda"
    amp: bool = True                        # automatic mixed precision (requires CUDA)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42
    deterministic: bool = True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    train_frac: float = 0.85               # fraction of frames used for training
    patch_h: int = 128                     # patch height ("strip" mode: full depth)
    patch_w: int = 128                     # patch width  ("strip" mode: 1 A-line)
    patches_per_frame: int = 16
    patch_mode: str = "patch"              # "patch" = random crop; "strip" = full-depth A-line
    augment: bool = True
    batch_size: int = 32
    num_workers: int = 8

    # ------------------------------------------------------------------
    # Input / target construction (baseline study)
    # ------------------------------------------------------------------
    input_mode: str = "bandgap"            # "bandgap" = [w1,w2] sub-bands; "fullband" = single full-band image (1ch)
    target_mode: str = "fullband"          # "fullband" = same-frame full band; "average" = temporal average target
    avg_leave_one_out: bool = True         # average excludes the input frame (no input/target leak)
    avg_cache_dir: str = "avg_cache"       # per-folder linear-magnitude sum cache (relative to runs_root)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_name: str = "resunet_pseudo3d"   # "resunet_pseudo3d" | "resunet_pseudo3d_multilevel" | "dncnn" | "unet2d"
    base: int = 64                         # base channel width

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 5e-5
    grad_clip: float = 1.0

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    w_charb: float = 0.8                   # Charbonnier loss weight
    w_grad: float = 0.5                    # gradient L1 loss weight

    # ------------------------------------------------------------------
    # Metrics — ROI (y pixel rows) for SNR/CNR evaluation
    # ------------------------------------------------------------------
    snr_sig_y0: int = 111
    snr_sig_y1: int = 600
    snr_sig_stat: str = "max"              # "max" or "p<percentile>" e.g. "p99.99"

    # ------------------------------------------------------------------
    # Validation / checkpoint cadence
    # ------------------------------------------------------------------
    val_every: int = 5                     # validate every N epochs
    save_every: int = 5                    # save periodic checkpoint every N epochs

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    early_stop_patience: int = 5           # validation checks without improvement
    early_stop_min_delta: float = 0.0
    early_stop_warmup_checks: int = 0

    # ------------------------------------------------------------------
    # Inference output
    # ------------------------------------------------------------------
    tiff_dtype: str = "uint16"             # "uint8" | "uint16" | "float32"
    also_save_float32: bool = False
