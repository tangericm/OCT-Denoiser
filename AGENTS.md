# AGENTS.md

Project overview, code layout, and standard commands live in `CLAUDE.md` and
`README.md`. Read those first. This file only adds Cursor Cloud specific setup
notes.

## Cursor Cloud specific instructions

### Environment
- Dependencies are installed into a virtualenv at `.venv/` (the startup update
  script creates it and installs CPU-only `torch`/`torchvision` plus the package
  in editable mode with `[dev]` extras). Run tools via `.venv/bin/<tool>` (e.g.
  `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/oct-train`)
  or `source .venv/bin/activate` first.
- This VM is **CPU-only (no GPU)**. Everything needed for development — lint,
  types, tests, build, and the full train→infer pipeline — runs on CPU.

### Lint / types / tests / build
- Standard gate commands are in `README.md` ("Sanity checks") and
  `CLAUDE.md`; run them through `.venv/bin/...`. CI mirrors these
  (`.github/workflows/ci.yml`).
- The whole `pytest` suite uses synthetic raw + `.CLB` fixtures
  (`tests/conftest.py`), so it needs no GPU and no instrument data.

### Running the app (non-obvious)
- `oct-train` / `oct-predict` etc. read a hardcoded `USER CONFIGURATION` block in
  `src/octdenoiser/cli/{train,predict}.py` that points at **Windows backslash
  paths** (e.g. `images\Maestro3`) for real instrument `.raw` + `.CLB` data that
  is **not** in the repo. They are therefore not runnable out-of-the-box here;
  editing that block or committing data is required to run them on real data.
- `TrainConfig(device="cuda", amp=True)` is safe on this CPU VM: `engine/train.py`
  falls back to CPU when CUDA is absent and disables AMP automatically.
- To exercise the real engine end-to-end without instrument data, build a small
  `TrainConfig`/`FolderSpec` over a synthetic acquisition (same generator as
  `tests/conftest.py`) and call `engine.train.run_training` +
  `engine.infer.predict_from_config` with `device="cpu"`. This produces a
  `best.pt` checkpoint and denoised TIFFs under a temp `runs/` dir.
