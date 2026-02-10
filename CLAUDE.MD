# CLAUDE.MD — OCT-Denoiser Contributor Guide

This file is a high-signal operating manual for Claude (and other coding agents) working in this repository.
Use it to maximize correctness, speed, and consistency when making changes.

---

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
- `configs/default.py` — dataclass configuration contracts (`TrainConfig`, `FolderSpec`, etc.).
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

- Reuse existing logging patterns in `utils/json_logging.py` and engine modules.
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

- Respect seeding utilities (e.g., `utils/seed.py`) and deterministic flags.
- Any new stochastic behavior should be seed-aware.
- If a change can alter numerical results, note it in commit/PR summary.

---

## 8) Model/training change checklist

Before finalizing training-related changes, verify:

1. Config fields exist, are typed, and are wired end-to-end.
2. Data pipeline outputs expected tensor shapes/dtypes.
3. Loss/metric changes are reflected in train + validation paths.
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

Given this repo’s script-oriented workflow, prefer lightweight checks:

- Unit-like checks for pure numerical helpers.
- Small synthetic-array tests for shape/range invariants.
- Smoke execution of entrypoints with minimal/mock config where feasible.

Do not add slow or data-heavy tests by default.

---

## 14) Definition of done

A change is “done” when:

- Code is correct and minimal.
- Existing workflows are preserved.
- At least one relevant sanity check has been run.
- Limitations are explicitly disclosed.
- Diff is clean and free of unrelated edits.

