# CLAUDE.MD — OCT-Denoiser Contributor Guide

This file is a high-signal operating manual for Claude (and other coding agents) working in this repository.
Use it to maximize correctness, speed, and consistency when making changes.

---
## Project Overview

Deep learning system for denoising Optical Coherence Tomography (OCT) B-scan images. Uses a ResUNet architecture with a pseudo-3D stem to process dual-channel spectral OCT data and produce denoised output. The project covers the full pipeline: raw spectral data preprocessing, neural network training, hyperparameter tuning, and inference.

**Author:** Eric Tang (tangericm)

## Quick Reference

```
# Environment setup
conda create --name OCTDenoiser python=3.14
conda activate OCTDenoiser
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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

## Repository Structure

```
OCT-Denoiser/
├── model_train.py           # Main training entry point
├── model_predict.py         # Standalone inference script
├── preprocess.py            # OCT signal processing pipeline
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
│   ├── losses.py            # Loss functions (Charbonnier, gradient L1) + unpack_batch
│   ├── metrics.py           # SNR/CNR computation in dB
│   └── early_stopping.py    # Patience-based early stopping dataclass
│
├── data/
│   ├── datamodule.py        # DataModule factory for DataLoaders
│   └── dataset.py           # RawBscanDataset (lazy init, per-worker caching)
│
└── utils/
    ├── run_manager.py       # Timestamped run directory creation
    ├── helpers.py            # Seeding (seed_all) and JSON serialization (save_json)
    ├── live_plot.py          # Real-time loss curve plotting
    └── io_tiff.py           # TIFF I/O with percentile-based scaling
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
- `FolderSpec` — per-dataset specification: raw data location, dimensions, spectral processing parameters. Also used directly by `BscanProcessor` (no separate preprocess config).

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
- **Centralized loss computation:** `compute_total_loss()` in `engine/losses.py` computes `w_charb * Charbonnier + w_grad * gradient_L1`, used by both training and evaluation
- **Composite validation score:** `score = w_loss * val_loss - w_snr * snr - w_cnr * cnr` (lower is better; normalized against first-check baselines)
- **Checkpoint naming:** `best.pt` (best val loss), `best_by_score.pt` (best composite score), `epoch_NNN.pt` (periodic)
- **Run directories:** Auto-versioned via `make_run_dir()` — creates `runs/<experiment>/<timestamp>/`

### Tensor Conventions
- Input shape: `[B, 2, H, W]` (batch, dual-window channels, height, width)
- Target shape: `[B, 1, H, W]`
- Images are log-compressed floats in [0, 1] range
- ROI for SNR/CNR: signal region defined by `snr_sig_y0`/`snr_sig_y1` (pixel rows); background is bottom 20 rows

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
2. Integrate it into `compute_total_loss()` in `engine/losses.py`
3. Add corresponding weight to `TrainConfig`

### Adding a New Dataset / Scan Type
1. Create a new `FolderSpec` with the correct raw data dimensions, crop, and dispersion coefficients
2. Add it to the `folder_specs` list in `model_train.py`
3. `BscanProcessor` in `preprocess.py` accepts `FolderSpec` directly and handles all spectral processing

### Running Hyperparameter Tuning
`tune.py` uses Optuna to search over spectral parameters (window_sigma, gap) and optionally training hyperparameters. Results are stored in per-trial CSV files under the run directory.

## Important Notes

- **Tests:** `python tests/test_optimizations.py` runs preprocessing equivalence, resampling, Gaussian window invariant, and train-step smoke tests.
- **GPU required for practical use.** Training and inference are designed for CUDA. CPU mode is supported but slow.
- **Windows paths in scripts.** `model_train.py` and `model_predict.py` use Windows-style backslash paths (e.g., `r"images\Maestro3"`). These may need adjustment on Linux.
- **Data not included.** Raw OCT `.raw` files and calibration `.clb` files are not in the repo. They are expected under an `images/` directory.
- **`runs/` directory** is generated at runtime for checkpoints, logs, and predictions.

## 1) Project mission and constraints

- **Goal:** Train and run an OCT denoising model from raw spectral data through reconstruction, patching, training, validation, and TIFF export.
- **Core priorities (in order):**
  1. Preserve scientific/data-processing correctness.
  2. Keep training/inference reproducible.
  3. Avoid breaking data paths and run-output conventions.
  4. Keep performance-sensitive code efficient (especially preprocessing and spectral operations).

When unsure, prefer **readability + correctness** over micro-optimizations unless a hotspot is clear.

---

## 2) Repository map (quick mental model)

- `model_train.py` — primary training entrypoint (config assembly, run setup, train, optional inference).
- `model_predict.py` — standalone inference script from raw data + checkpoint.
- `preprocess.py` — preprocessing pipeline and computational OCT reconstruction helpers.
- `configs/default.py` — dataclass configuration contracts (`TrainConfig`, `FolderSpec`).
- `data/` — datasets/datamodule, patch extraction, frame loading.
- `engine/` — train/eval/infer loops, metrics, losses, early stopping.
- `networks/` — model definitions and model registry.
- `utils/` — IO, reproducibility, plotting/logging, run directory utilities.
- `runs/` (generated) — experiment outputs/checkpoints/predictions.

If adding new modules, keep this layering direction:
`configs -> data/networks/utils -> engine -> entry scripts`.

---

## 3) Environment + dependencies

- Python target is controlled by local setup docs; prefer a modern 3.x interpreter with installed `requirements.txt`.
- This project is GPU-aware (`torch` CUDA builds), but changes should not hard-crash on CPU-only environments.
- Heavy dependencies include: `torch`, `numpy`, `scipy`, `tifffile`, `matplotlib`, `optuna`.

### Recommended setup commands

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

If CUDA-specific wheels are required, follow README guidance and do not silently swap CUDA/CPU packages without noting it.

---

## 4) Standard run commands

Use these defaults unless task requires otherwise:

### Train
```bash
python model_train.py
```

### Predict (from raw data + checkpoint)
```bash
python model_predict.py
```

### Fast syntax sanity check
```bash
python -m compileall .
```

### Optional targeted checks
```bash
python -m py_compile model_train.py model_predict.py preprocess.py
```

When runtime/data is unavailable, still run non-data checks (`compileall`, import/smoke checks).

---

## 5) Coding standards for this repo

### General

- Follow existing style in touched files; avoid unrelated refactors.
- Keep function signatures stable unless task explicitly requires API changes.
- Use explicit names over clever abbreviations.
- Add short comments for non-obvious math or OCT-specific logic.

### Typing and docs

- Use type hints for new/modified functions where practical.
- Add/update docstrings for public or non-trivial functions.
- Document array shapes in comments/docstrings for numerical code.

### Error handling

- Fail early with clear exceptions for invalid shapes/paths/parameters.
- Include actionable error text (expected vs actual shape, missing file path, etc.).

### Logging

- Reuse existing logging patterns in `utils/helpers.py` and engine modules.
- Avoid noisy per-iteration prints in performance-critical loops.

### Performance-sensitive code

- In `preprocess.py` and similar hotspots:
  - Avoid unnecessary copies.
  - Reuse precomputed operators/buffers where possible.
  - Prefer vectorized operations over Python loops.
  - Benchmark before/after if changing algorithmic core.

---

## 6) Data and experiment safety rules

- **Do not** hardcode private/local absolute paths in committed code.
- Preserve existing run folder semantics (`runs/<experiment>/<run_id>/...`).
- Keep checkpoint naming and result-file conventions backward-compatible.
- Do not commit generated artifacts (`runs/`, large TIFFs, caches) unless explicitly requested.

When changing configs, keep defaults conservative and reproducible.

---

## 7) Reproducibility rules

- Respect seeding utilities (`utils/helpers.py: seed_all()`) and deterministic flags.
- Any new stochastic behavior should be seed-aware.
- If a change can alter numerical results, note it in commit/PR summary.

---

## 8) Model/training change checklist

Before finalizing training-related changes, verify:

1. Config fields exist, are typed, and are wired end-to-end.
2. Data pipeline outputs expected tensor shapes/dtypes.
3. Loss/metric changes are reflected in `compute_total_loss()` and both train + validation paths.
4. Checkpoint load/save compatibility is preserved.
5. Inference path still works with best checkpoint output.

If one of these cannot be validated locally, explicitly note the limitation.

---

## 9) Inference/output change checklist

- Validate output dtype/range assumptions (`uint16` vs float32 output paths).
- Confirm output directory creation logic remains robust.
- Keep frame ordering deterministic.
- For TIFF writing changes, verify at least one read/write roundtrip in a small local test where possible.

---

## 10) How Claude should execute tasks (operational playbook)

1. **Scan relevant files first** (`README`, entry script, touched module).
2. **Minimize blast radius**: patch only what is needed.
3. **Implement incrementally** with small coherent edits.
4. **Run checks** appropriate to environment constraints.
5. **Summarize clearly**:
   - what changed,
   - why,
   - how validated,
   - any risks/limitations.

For ambiguous requirements, choose the most conservative behavior that preserves current workflows.

---

## 11) Preferred response format for coding tasks

When reporting changes, include:

- **Summary** (bullet list by file)
- **Validation** (commands run + pass/fail)
- **Follow-ups** (only if truly needed)

Keep claims tied to concrete checks or code references.

---

## 12) Anti-patterns to avoid

- Silent behavior changes in preprocessing math.
- Mixing refactors with feature fixes in one patch.
- Renaming widely used config fields without migration handling.
- Introducing new dependencies for minor utilities.
- Committing debug prints, scratch notebooks, or local data artifacts.

---

## 13) If you need to add tests

Given this repo's script-oriented workflow, prefer lightweight checks:

- Unit-like checks for pure numerical helpers.
- Small synthetic-array tests for shape/range invariants.
- Smoke execution of entrypoints with minimal/mock config where feasible.

Do not add slow or data-heavy tests by default.

---

## 14) Definition of done

A change is "done" when:

- Code is correct and minimal.
- Existing workflows are preserved.
- At least one relevant sanity check has been run.
- Limitations are explicitly disclosed.
- Diff is clean and free of unrelated edits.
