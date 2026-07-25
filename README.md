# OCT Denoiser

Deep learning pipeline for denoising OCT B-scans using a ResUNet with pseudo-3D spectral stem. Raw `.raw` spectral data is preprocessed on-the-fly (k-linearisation → spectral windowing → IFFT → log compress → z-score) and fed to the network as dual-channel (or multi-level sub-window) inputs.

**Author:** Eric Tang (tangericm) · eric.tang22@gmail.com

---

## Results

![Reference versus network prediction](docs/result.jpg)

Self-supervised throughout: no clean reference and no repeat acquisition. Supervision
comes from splitting the raw interferogram into two Gaussian sub-bands separated by a
tunable gap, which yields two reconstructions with identical structure but decorrelated
speckle.

A single model trained across four acquisitions, evaluated against the full-bandwidth
reference on each:

| Acquisition | ΔSNR | ΔCNR |
|---|---|---|
| Macula, 6 mm, 1024 A-lines | +9.45 dB | +4.72 dB |
| Macula centre | +9.95 dB | +5.37 dB |
| Optic disc | +13.82 dB | +5.72 dB |
| Line, 6 mm, 2048 A-lines | +13.71 dB | +8.33 dB |

A run dedicated to a single volume (`window_sigma 0.03`, `gap 0.60`) reaches
**+16.45 dB SNR** and **+12.27 dB CNR** averaged over 64 frames.

Resolution is preserved rather than smoothed away: on a mirror phantom with a known
point spread function, the method measures a **PSF FWHM of 8.0 px** against 8.45 px for
the noisy input, the narrowest of the architectures compared.

> Display note: reference and prediction above are rendered through **one shared
> window** with a shared gamma. The black point is anchored to the prediction's 1st
> percentile so its noise floor survives; anchoring to the reference clips 39% of the
> prediction to pure black, which both misrepresents the output and, counter-intuitively,
> *reduces* the visible difference as the two images collapse toward the same black.

---

## Environment Setup

Python 3.14 · PyTorch 2.10.0+cu128 · CUDA required for practical training

```bash
conda create --name OCTDenoiser python=3.14
conda activate OCTDenoiser
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Quick Start

### 1. Configure your dataset

Edit the `USER CONFIGURATION` section in [model_train.py](model_train.py):

- Set `root_folder` / `data_folder` to point at your `bscan*.raw` files
- Set `pixels` / `alines` to match your OCT system
- Adjust `crop_depth`, `window_sigma`, and `gap` as needed
- Choose `model_name`:
  - `"resunet_pseudo3d"` — standard 2-channel input (no sub-windows)
  - `"resunet_pseudo3d_multilevel"` — multi-level input (requires `n_sub_windows > 0`)

### 2. Train

```bash
python model_train.py
```

Outputs written to `runs/<experiment_name>/<timestamp>/`:

| Path | Contents |
|------|----------|
| `checkpoints/best.pt` | Best validation checkpoint |
| `predictions_tiff/` | Post-training TIFF stacks |
| `val_outputs/` | Per-epoch validation images + progression TIFF |
| `config.json` | Full run configuration |
| `history.json` | Training / validation metrics history |

### 3. Inference from a checkpoint

Edit the `USER CONFIGURATION` section in [model_predict.py](model_predict.py) with the checkpoint path and matching `FolderSpec`, then:

```bash
python model_predict.py
```

### 4. Hyperparameter tuning (optional)

Tune `window_sigma` and `gap` with Optuna:

```bash
python tune.py
```

Results are saved as `runs/optuna/<timestamp>/study_results.csv`.

### 5. Sanity checks

```bash
python tests/test_optimizations.py          # preprocessing + model forward pass
python -m compileall .                      # syntax check (no data needed)
```

---

## Project Structure

```
OCT-Denoiser/
├── model_train.py                          # training entry point
├── model_predict.py                        # standalone inference
├── preprocess.py                           # BscanProcessor: raw -> B-scan tensor
├── tune.py                                 # Optuna window parameter search
├── configs/default.py                      # TrainConfig, FolderSpec dataclasses
├── data/
│   ├── dataset.py                          # RawBscanDataset (lazy init, LRU cache)
│   └── datamodule.py                       # DataLoader factory
├── engine/
│   ├── train.py                            # AMP training loop, checkpointing
│   ├── eval.py                             # patch + full-frame validation
│   ├── infer.py                            # raw -> TIFF inference pipeline
│   ├── losses.py                           # Charbonnier + gradient L1
│   ├── metrics.py                          # SNR/CNR (physical domain)
│   └── early_stopping.py                   # patience-based early stopping
├── networks/
│   ├── registry.py                         # @register_model decorator
│   ├── resunet_pseudo3d.py                 # base ResUNet with Pseudo-3D stem
│   └── resunet_pseudo3d_multilevel.py      # + multi-level spectral input
├── utils/
│   ├── helpers.py                          # seed_all, save_json, nanmean
│   ├── run_manager.py                      # run directory management
│   ├── io_tiff.py                          # TIFF stack I/O
│   └── live_plot.py                        # live loss curve (PNG + optional window)
└── tests/
    ├── test_optimizations.py               # preprocessing + model validation
    └── test_resunet_multilevel_1d.py       # model forward pass shape tests
```

---

## Configuration Reference

All configuration is defined in Python dataclasses — no YAML or JSON files.

### `FolderSpec` — per-dataset specification

| Field | Default | Description |
|-------|---------|-------------|
| `root_folder` | — | Root path containing the dataset folder and `.CLB` file |
| `data_folder` | — | Subfolder containing `bscan*.raw` files |
| `pixels` | — | Spectral samples per A-line (e.g. 2048) |
| `alines` | — | A-lines per B-scan (e.g. 1024) |
| `crop_depth` | `(1024, 2048)` | `[z0, z1)` pixel crop after IFFT |
| `window_sigma` | `0.08` | Gaussian spectral window width |
| `gap` | `0.15` | Separation between the two window centres |
| `gap_offset` | `0.0` | Shared shift of both window centres |
| `n_sub_windows` | `0` | Sub-windows per parent; `0` = disabled |
| `sub_window_spread` | `2.0` | Sub-window centre spread in sigma units |

### `TrainConfig` — training hyperparameters

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `"resunet_pseudo3d"` | Model to train |
| `base` | `64` | Base channel width |
| `epochs` | `300` | Maximum training epochs |
| `lr` | `3e-4` | AdamW learning rate |
| `batch_size` | `32` | Training batch size |
| `patch_mode` | `"patch"` | `"patch"` = random crop; `"strip"` = full-depth A-line |
| `w_charb` / `w_grad` | `0.8 / 0.5` | Charbonnier and gradient loss weights |
| `early_stop_patience` | `5` | Validation checks without improvement before stopping |
| `snr_sig_stat` | `"max"` | Signal statistic: `"max"` or `"p<N>"` e.g. `"p99.99"` |

---

## Data Flow

```
Raw .raw (uint16, Fortran order)
  -> DC subtract -> k-linear resample (natural cubic spline, precomputed gttrs)
  -> Gaussian spectral windowing (w1, w2, optional sub-windows)
  -> Batched IFFT -> magnitude -> log10 compress -> z-score normalise
  -> X: [B, 2 + 2*n_sub_windows, H, W]   (float32, normalised)
  -> Y: [B, 1, H, W]                      (full-bandwidth target)
  -> Loss: w_charb * Charbonnier + w_grad * gradient_L1
  -> best.pt checkpoint -> TIFF export
```
