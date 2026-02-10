# CLAUDE.md — OCT-Denoiser

## Project Overview

Deep learning system for denoising Optical Coherence Tomography (OCT) B-scan images. Uses a ResUNet architecture with a pseudo-3D stem to process dual-channel spectral OCT data and produce denoised output. The project covers the full pipeline: raw spectral data preprocessing, neural network training, hyperparameter tuning, and inference.

**Author:** Eric Tang (tangericm)

## Quick Reference

```
# Environment setup
conda create --name OCTDenoiser python=3.14
conda activate OCTDenoiser
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt

# Train a model
python model_train.py

# Run inference with a trained checkpoint
python model_predict.py

# Hyperparameter tuning
python tune.py
```

- **Python:** 3.14 (Miniconda)
- **PyTorch:** 2.10.0+cu128 (CUDA)
- **No test suite, CI/CD, linter, or formatter is configured**

## Repository Structure

```
OCT-Denoiser/
├── model_train.py           # Main training entry point
├── model_predict.py         # Standalone inference script
├── preprocess.py            # OCT signal processing pipeline (581 lines)
├── tune.py                  # Optuna hyperparameter search
├── requirements.txt         # Pinned pip dependencies
│
├── configs/
│   └── default.py           # TrainConfig and FolderSpec dataclasses
│
├── networks/
│   ├── registry.py          # Model registration decorator + factory
│   └── resunet_pseudo3d.py  # ResUNet with Pseudo-3D stem
│
├── engine/
│   ├── train.py             # Training loop (AMP, early stopping, composite scoring)
│   ├── eval.py              # Patch and full-frame validation
│   ├── infer.py             # Inference pipeline → TIFF output
│   ├── losses.py            # Charbonnier + gradient L1 losses
│   ├── metrics.py           # SNR/CNR computation in dB
│   └── early_stopping.py    # Patience-based early stopping dataclass
│
├── data/
│   ├── datamodule.py        # DataModule factory for DataLoaders
│   ├── dataset.py           # RawBscanPatchDataset (lazy init, per-worker caching)
│   └── full_frame_dataset.py# Full-frame dataset for validation/inference
│
└── utils/
    ├── run_manager.py       # Timestamped run directory creation
    ├── seed.py              # Deterministic seeding (random, numpy, torch, CUDA)
    ├── json_logging.py      # Config/metrics JSON serialization
    ├── live_plot.py          # Real-time loss curve plotting
    ├── io_tiff.py           # TIFF I/O with percentile-based scaling
    └── keypoint_registration.py  # ORB-based frame registration
```

## Architecture & Data Flow

```
Raw .raw files
  → BscanProcessor (preprocess.py)
    [DC subtract → k-linear resample → spectral windowing → dispersion comp → FFT → log compress]
  → X: [F, 2, H, W] (dual spectral windows)  Y: [F, 1, H, W] (full-bandwidth target)
  → Dataset (patches or full frames)
  → DataLoader
  → ResUNet Pseudo-3D model
  → Loss (w_charb * Charbonnier + w_grad * gradient_L1)
  → Validation (patch loss + full-frame SNR/CNR)
  → Composite score for checkpointing & early stopping
```

### Model: ResUNet Pseudo-3D

- **Input:** `[B, 2, H, W]` — two spectral-window B-scans
- **Stem:** `Pseudo3DStem` — Conv3D `(2,3,3)` kernel fuses the 2 channels into a 2D feature map
- **Encoder:** 4-level with residual blocks (base → 2x → 4x → 8x channels)
- **Decoder:** 3-level with skip connections and ConvTranspose2d upsampling
- **Output:** `[B, 1, H, W]` — denoised prediction
- **Activation:** SiLU (Swish) throughout; BatchNorm2d normalization
- **Default base channels:** 64 (often overridden to 32 in scripts)

### Model Registry

New models are added via decorator and looked up by name:
```python
@register_model("my_model")
def build_my_model(*, base: int = 64) -> nn.Module:
    ...
```
Used as: `create_model(cfg.model_name, base=cfg.base)`

## Configuration System

All configuration uses Python **dataclasses** in `configs/default.py`. No YAML/JSON config files.

- `TrainConfig` — all training hyperparameters, paths, loss weights, early stopping, ROI bounds
- `FolderSpec` — per-dataset specification: raw data location, dimensions, spectral processing parameters (window_sigma, gap, dispersion coefficients)

Key parameters in `TrainConfig`:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `patch_mode` | `"patch"` | `"strip"` (full-depth, random x) or `"patch"` (random x,y) |
| `w_charb` / `w_grad` | 0.8 / 0.5 | Charbonnier and gradient loss weights |
| `score_w_val_loss` / `score_w_snr` / `score_w_cnr` | 1.0 / 1.0 / 0.0 | Composite score weights (lower = better) |
| `amp` | `True` | Automatic mixed precision |
| `early_stop_patience` | 5 | Validation checks without improvement before stopping |

## Key Conventions

### Code Style
- No linter or formatter is enforced; code uses standard Python conventions
- Type hints via `typing` and `__future__ annotations`
- Dataclasses over dicts for configuration
- `from __future__ import annotations` in library modules

### Project Patterns
- **Lazy dataset initialization:** Datasets create per-worker `BscanProcessor` instances in `__getitem__` (not `__init__`) to work with multiprocessing DataLoaders
- **LRU frame caching:** Workers cache processed frames to avoid redundant I/O
- **Composite validation score:** `score = w_loss * val_loss - w_snr * snr - w_cnr * cnr` (lower is better; normalized against first-check baselines)
- **Checkpoint naming:** `best.pt` (best composite score), `epoch_NNN.pt` (periodic)
- **Run directories:** Auto-versioned via `make_run_dir()` — creates `runs/<experiment>/<experiment>_v001/`

### Tensor Conventions
- Input shape: `[B, 2, H, W]` (batch, dual-window channels, height, width)
- Target shape: `[B, 1, H, W]`
- Images are log-compressed floats in [0, 1] range
- ROI for SNR/CNR: signal region defined by `snr_sig_y0`/`snr_sig_y1` (pixel rows); background is bottom 50 rows

### File I/O
- Raw input: `.raw` uint16 binary files (spectral domain)
- Calibration: `.clb` files for k-linear resampling
- Output: multi-page TIFF stacks (uint16 or float32) via `tifffile`
- Training artifacts: JSON logs, PNG loss plots, CSV metrics

## Common Development Tasks

### Adding a New Model
1. Create a file in `networks/` (e.g., `networks/my_model.py`)
2. Implement the model class, accepting `[B, 2, H, W]` input and returning `[B, 1, H, W]`
3. Decorate the builder function with `@register_model("my_model")`
4. Import the module in `networks/__init__.py` so registration runs
5. Set `model_name="my_model"` in `TrainConfig`

### Adding a New Loss Function
1. Add the function to `engine/losses.py`
2. Integrate it into the training step in `engine/train.py` (search for `w_charb` / `w_grad` usage)
3. Add corresponding weight to `TrainConfig`

### Adding a New Dataset / Scan Type
1. Create a new `FolderSpec` with the correct raw data dimensions, crop, and dispersion coefficients
2. Add it to the `folder_specs` list in `model_train.py`
3. The `BscanProcessor` in `preprocess.py` handles all spectral processing generically

### Running Hyperparameter Tuning
`tune.py` uses Optuna to search over spectral parameters (window_sigma, gap) and optionally training hyperparameters. Results are stored in per-trial CSV files under the run directory.

## Important Notes

- **No automated tests exist.** Validate changes by running training on a small subset or verifying inference output visually.
- **GPU required for practical use.** Training and inference are designed for CUDA. CPU mode is supported but slow.
- **Windows paths in scripts.** `model_train.py` and `model_predict.py` use Windows-style backslash paths (e.g., `r"images\Maestro3"`). These may need adjustment on Linux.
- **Data not included.** Raw OCT `.raw` files and calibration `.clb` files are not in the repo. They are expected under an `images/` directory.
- **`runs/` directory** is generated at runtime for checkpoints, logs, and predictions.
