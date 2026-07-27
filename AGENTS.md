# AGENTS.md — OCT-Denoiser

See `CLAUDE.md` (repository map, data flow, model registry, patterns) and `README.md`
(environment setup, quick start, config reference) for the full project guide. Standard
dev commands (`ruff check .`, `mypy`, `pytest`, `python -m compileall -q src tests`,
`oct-train` / `oct-predict` / `oct-tune`) live there — this file only records the
Cursor Cloud specifics.

## Cursor Cloud specific instructions

- **Python env is a venv at `/home/ubuntu/.venvs/octdenoiser`** (system Python is not used).
  Run tools via that venv, e.g. `/home/ubuntu/.venvs/octdenoiser/bin/pytest`, or prepend it
  to `PATH`: `export PATH=/home/ubuntu/.venvs/octdenoiser/bin:$PATH`. The startup update
  script refreshes dependencies into this venv.
- **No GPU in the cloud VM — PyTorch is the CPU build.** `torch` is installed from the CPU
  wheel index (`https://download.pytorch.org/whl/cpu`); `torch.cuda.is_available()` is
  `False`. The training engine auto-falls back to CPU when CUDA is absent
  (`engine/train.py`), and AMP is a no-op on CPU (`amp=True` needs CUDA). Full-scale training
  is not practical on CPU here; use small configs / few epochs for smoke runs.
- **The `oct-*` console entry points do not run as-is.** The `USER CONFIGURATION` blocks in
  `cli/train.py` / `cli/predict.py` hardcode Windows paths (`r"images\Maestro3"`) and
  `device="cuda"`, and point at instrument data that is not in the repo. To exercise the
  pipeline you must supply a `FolderSpec` with a real path + `.CLB`, or drive the engine
  programmatically on synthetic data.
- **No instrument data is required for tests or a smoke run.** `tests/conftest.py`
  synthesises raw `.raw` interferograms + a synthetic `.CLB`; `pytest` runs fully offline
  (one test is skipped — it is `needs_cuda`). A minimal end-to-end driver (synthesise data →
  `run_training` → `predict_from_config`) reproduces the full raw→denoised→TIFF path on CPU.
- **No display needed** — this is a CLI/library pipeline (Matplotlib writes PNGs via the Agg
  backend); there is no interactive GUI to launch.
