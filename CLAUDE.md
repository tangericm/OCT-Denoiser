# CLAUDE.md — OCT-Denoiser

Deep learning pipeline for denoising OCT B-scans. ResUNet with pseudo-3D stem processes dual-channel spectral data → denoised output. Full pipeline: raw spectral preprocessing → training → hyperparameter tuning → inference.

**Author:** Eric Tang (tangericm)

---

## Quick Commands
```bash
pip install -e ".[dev]"                 # after installing torch from the PyTorch index
oct-train                               # train
oct-predict --checkpoint <path>         # inference from checkpoint
oct-tune                                # Optuna hyperparameter search
oct-mirror-study                        # mirror-phantom baseline study
pytest                                  # full suite, needs no OCT data
ruff check . && mypy                    # lint + types
python -m compileall -q src tests       # syntax check
```

Use the explicit `src tests` form: `compileall` descends into `.git/refs/`, so a
branch whose name ends in `.py` is parsed as source and fails. Three such
branches existed and have been pruned, but the explicit form cannot break again.

**Stack:** Python ≥3.10 · PyTorch ≥2.4 · CUDA required for practical use

---

## Repository Map
```
OCT-Denoiser/
├── pyproject.toml                    # packaging + ruff/mypy/pytest config
├── .github/workflows/ci.yml          # ruff → mypy → pytest → compileall
├── src/octdenoiser/
│   ├── preprocess.py                 # BscanProcessor: DC subtract → k-linear resample → window → FFT → log compress
│   ├── cli/{train,predict,tune}.py   # console scripts oct-train / oct-predict / oct-tune
│   ├── configs/default.py            # TrainConfig, FolderSpec (+ __post_init__ validation)
│   ├── data/dataset.py               # RawBscanDataset (lazy init, per-worker LRU cache)
│   ├── data/datamodule.py            # DataLoader factory
│   ├── data/avg_targets.py           # temporal-average target cache
│   ├── engine/train.py               # AMP training loop, early stopping, checkpointing
│   ├── engine/eval.py                # patch + full-frame validation
│   ├── engine/infer.py               # raw → TIFF inference pipeline
│   ├── engine/losses.py              # Charbonnier + gradient L1, unpack_batch
│   ├── engine/metrics.py             # SNR/CNR in dB, ROI helpers
│   ├── engine/early_stopping.py      # patience-based early stopping
│   ├── networks/registry.py          # @register_model decorator + create_model()
│   ├── networks/*.py                 # resunet_pseudo3d, _multilevel, dncnn, unet2d
│   ├── experiments/                  # run_mirror_study, run_retina_compare (CLIs)
│   ├── tools/eval_mirror.py          # PSNR/SSIM/FWHM harness
│   └── utils/                        # run dirs, seeding, TIFF I/O, live plot
└── tests/conftest.py                 # synthetic raw + .CLB fixtures — suite needs no data
```

Layer direction: `configs → data/networks/utils → engine → cli`

**Calibration is per instrument.** Maestro2 and Maestro3 ship different `.CLB`
k-linearisation LUTs. Applying the wrong one resamples onto the wrong k-grid and
yields a plausible-looking reconstruction with the wrong depth scale and PSF, with
no error raised. `root_folder` must point at the matching instrument directory, or
set `FolderSpec.clb_path` explicitly (it now wins over auto-discovery).

---

## Data Flow
```
Raw .raw (uint16)
  → BscanProcessor.process_one()
    DC subtract → k-linear resample (natural cubic spline, precomputed gttrs)
    → spectral windowing (w1, w2, optional sub-windows)
    → batched IFFT → log compress → z-score normalize
  → X: [B, 2 + 2*n_sub_windows, H, W]  (float32, z-score normalized per frame)
  → Y: [B, 1, H, W]  (full-bandwidth target)
  → Loss: w_charb * Charbonnier + w_grad * gradient_L1
  → best.pt checkpoint → TIFF export
```

**Tensor conventions:**
- Input: `[B, 2+n_sub, H, W]` — log-compressed floats, z-score normalized per frame
- Target: `[B, 1, H, W]`
- Raw files are uint16; all model tensors are float32

---

## Configuration

All config lives in Python dataclasses in `configs/default.py`. No YAML/JSON.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `patch_mode` | `"patch"` | `"strip"` = full-depth random-x; `"patch"` = random x,y |
| `w_charb` / `w_grad` | 0.8 / 0.5 | loss weights |
| `amp` | `True` | automatic mixed precision |
| `early_stop_patience` | 5 | counts validation checks, not epochs |
| `snr_sig_stat` | `"max"` | signal statistic: `"max"` or `"p<N>"` e.g. `"p99.99"` |
| `n_sub_windows` | 0 | sub-windows per parent window; 0 = disabled |

`FolderSpec` is used directly by `BscanProcessor` — it is both a config object and a preprocessing contract.

---

## Model Registry
```python
@register_model("my_model")
def build_my_model(*, base: int = 64) -> nn.Module:
    ...  # must accept [B, 2+n_sub, H, W], return [B, 1, H, W]
```

Register → import in `networks/__init__.py` → set `model_name` in `TrainConfig`.

**Available models:** `resunet_pseudo3d` · `resunet_pseudo3d_multilevel`

For multi-level models, pass `n_sub_channels = 2 * n_sub_windows` to `create_model()`.

---

## Critical Project-Specific Patterns

**Lazy worker init** — `BscanProcessor` is instantiated inside `__getitem__`, not `__init__`, to survive multiprocessing fork. Do not move initialization earlier.

**Batched FFT** — Always use `recon_bscan_batch()` for multi-spectrum reconstruction. Never call `recon_bscan_from_spectrum()` in hot paths.

**Precomputed spline** — `_spline_pre` holds the LAPACK `gttrs` factorization, computed once per `BscanProcessor`. Never recreate per frame. The `_rhs` buffer inside it is reused across calls.

**LRU frame cache** — Keyed by `(fidx, frame_idx)` per worker instance. Size controlled by `cache_frames_per_worker`.

**Negative strides** — Flip augmentations produce non-contiguous arrays. Always call `np.ascontiguousarray()` or use the `_to_torch_float32()` helper before `torch.from_numpy()`.

**SNR/CNR domain** — Metrics are computed in linear physical intensity, not log domain. Call `to_physical_intensity(pred, sample_meta)` before `roi_snr_cnr()`. `sample_meta` carries `target_mu`, `target_sd`, `log_eps`.

**Windows paths** — Entry scripts use Windows backslash paths (`r"images\Maestro3"`). Adjust for Linux.

---

## Common Tasks

### Add a new model
1. Create `networks/my_model.py`, decorate builder with `@register_model("my_model")`
2. Import in `networks/__init__.py`
3. Set `model_name="my_model"` in `TrainConfig`

### Add a new loss
1. Add function to `engine/losses.py`
2. Wire into `compute_total_loss()` — used by both train and eval paths
3. Add weight parameter to `TrainConfig`

### Add a new dataset
1. Create a `FolderSpec` with correct dims, crop, window parameters
2. Append to `folder_specs` in `src/octdenoiser/cli/train.py`

---

## Before Submitting Changes

- [ ] New config fields typed and wired end-to-end in `TrainConfig`/`FolderSpec`
- [ ] Tensor shapes correct: input `[B, C, H, W]`, target `[B, 1, H, W]`
- [ ] Loss/metric changes reflected in `compute_total_loss()` and both train + val paths
- [ ] Checkpoint load/save compatibility preserved
- [ ] Inference path verified with `best.pt`
- [ ] Output dtype assumptions preserved (`uint16` vs `float32`)
- [ ] `python -m compileall .` passes
- [ ] `python tests/test_optimizations.py` passes where feasible

---

## Hard Rules

- No silent changes to preprocessing math
- No mixing refactors with feature fixes in one patch
- No renaming config fields without migration handling
- No new dependencies for minor utilities
- No hardcoded absolute paths in committed code
- No committing `runs/`, TIFFs, or debug artifacts
- No per-iteration debug prints in training loops
- Preserve `runs/<experiment>/<timestamp>/` directory structure
