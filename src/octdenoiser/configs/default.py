import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Valid enum-like values. `model_name` is deliberately NOT validated here —
# networks/registry.py is the source of truth and create_model() raises on an
# unknown name; duplicating the list would drift.
PATCH_MODES = ("patch", "strip")
INPUT_MODES = ("bandgap", "fullband")
TARGET_MODES = ("fullband", "average")
TIFF_DTYPES = ("uint8", "uint16", "float32")
_SNR_STAT_RE = re.compile(r"^(max|p\d+(\.\d+)?)$")

MULTILEVEL_MODELS = ("resunet_pseudo3d_multilevel",)


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


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

    def __post_init__(self) -> None:
        if self.pixels <= 0 or self.alines <= 0:
            raise ConfigError(f"pixels and alines must be positive, got {self.pixels}, {self.alines}")
        z0, z1 = self.crop_depth
        if not (0 <= z0 < z1 <= self.pixels):
            raise ConfigError(
                f"crop_depth must satisfy 0 <= z0 < z1 <= pixels ({self.pixels}), got {self.crop_depth}"
            )
        if self.window_sigma <= 0:
            raise ConfigError(f"window_sigma must be positive, got {self.window_sigma}")
        if self.n_sub_windows < 0:
            raise ConfigError(f"n_sub_windows must be >= 0, got {self.n_sub_windows}")
        if self.log_eps <= 0:
            raise ConfigError(f"log_eps must be positive, got {self.log_eps}")

    @property
    def in_channels_bandgap(self) -> int:
        """Input channel count this spec produces in bandgap mode."""
        return 2 + 2 * self.n_sub_windows



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

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.patch_mode not in PATCH_MODES:
            raise ConfigError(f"patch_mode must be one of {PATCH_MODES}, got {self.patch_mode!r}")
        if self.input_mode not in INPUT_MODES:
            raise ConfigError(f"input_mode must be one of {INPUT_MODES}, got {self.input_mode!r}")
        if self.target_mode not in TARGET_MODES:
            raise ConfigError(f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}")
        if self.tiff_dtype not in TIFF_DTYPES:
            raise ConfigError(f"tiff_dtype must be one of {TIFF_DTYPES}, got {self.tiff_dtype!r}")
        if not _SNR_STAT_RE.match(self.snr_sig_stat):
            raise ConfigError(
                f'snr_sig_stat must be "max" or "p<percentile>" (e.g. "p99.99"), got {self.snr_sig_stat!r}'
            )
        if not 0.0 < self.train_frac < 1.0:
            raise ConfigError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if self.snr_sig_y0 >= self.snr_sig_y1:
            raise ConfigError(f"snr_sig_y0 must be < snr_sig_y1, got {self.snr_sig_y0}, {self.snr_sig_y1}")
        for name in ("epochs", "batch_size", "patch_h", "patch_w", "patches_per_frame",
                     "val_every", "save_every", "base"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must be >= 0, got {self.num_workers}")

        self._validate_model_input_consistency()

    def _validate_model_input_consistency(self) -> None:
        """Catch model/input mismatches that would otherwise fail at the first
        forward pass, or — worse — train a different architecture than intended.

        engine/train.py branches on model_name before consulting input_mode, and
        only reads folder_specs[0].n_sub_windows, so a heterogeneous or
        inconsistent config silently builds the wrong stem.
        """
        if not self.folder_specs:
            return  # nothing to cross-check yet; the dataloader will raise if it stays None

        n_subs = {fs.n_sub_windows for fs in self.folder_specs}
        if len(n_subs) > 1:
            raise ConfigError(
                f"all folder_specs must share n_sub_windows (only folder_specs[0] is used to "
                f"size the model stem), got {sorted(n_subs)}"
            )
        n_sub = n_subs.pop()

        is_multilevel = self.model_name in MULTILEVEL_MODELS
        if is_multilevel:
            if n_sub <= 0:
                raise ConfigError(
                    f"model_name={self.model_name!r} needs n_sub_windows > 0 on every FolderSpec, got {n_sub}"
                )
            if self.input_mode != "bandgap":
                raise ConfigError(
                    f"model_name={self.model_name!r} builds a multi-level spectral stem and requires "
                    f'input_mode="bandgap", got {self.input_mode!r}'
                )
        elif self.input_mode == "fullband" and n_sub > 0:
            raise ConfigError(
                f'input_mode="fullband" produces a 1-channel input, so n_sub_windows must be 0, got {n_sub}'
            )
