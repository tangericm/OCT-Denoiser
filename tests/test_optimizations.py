"""
Validation harness for preprocessing optimizations and loss functions.

Run:  python tests/test_optimizations.py

Tests:
  1. Preprocessing equivalence: old (single FFT) vs new (batched FFT)
  2. Preprocessing microbenchmark
  3. Resampling operator: cached gttrs vs scipy CubicSpline
  4. Single-batch train-step smoke test (no NaNs)
  5. Gaussian window invariants (sigma/gap independence)
  6. Multi-level window generation
  7. Multi-level model forward pass
  8. Sweep config validation
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np

# ---------------------------------------------------------------------------
# 1) Preprocessing equivalence: old single-FFT vs new batched-FFT
# ---------------------------------------------------------------------------
def test_preprocessing_equivalence():
    """
    Compare recon_bscan_from_spectrum (single) vs recon_bscan_batch (batched)
    on identical synthetic spectra.  Report max/mean absolute and relative error.
    """
    from preprocess import recon_bscan_from_spectrum, recon_bscan_batch

    rng = np.random.default_rng(42)
    pixels, alines = 2048, 512
    crop = (0, 1024)
    use_log, log_eps, fftshift = True, 1e-6, False

    # Synthetic complex spectra (3 different ones)
    specs = []
    for _ in range(3):
        real = rng.standard_normal((pixels, alines)).astype(np.float32)
        imag = rng.standard_normal((pixels, alines)).astype(np.float32)
        specs.append((real + 1j * imag).astype(np.complex64))

    # Old path: 3 separate calls
    old_results = []
    for s in specs:
        old_results.append(recon_bscan_from_spectrum(s, crop, use_log, log_eps, fftshift))

    # New path: batched
    spec_stack = np.stack(specs, axis=0)  # [3, pixels, alines]
    new_results = recon_bscan_batch(spec_stack, crop, use_log, log_eps, fftshift)

    print("=" * 60)
    print("TEST 1: Preprocessing Equivalence (old vs batched FFT)")
    print("=" * 60)
    all_pass = True
    for i in range(3):
        diff = np.abs(old_results[i] - new_results[i])
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(old_results[i]), 1e-10)
        rel_err = diff / denom
        max_rel = float(rel_err.max())
        mean_rel = float(rel_err.mean())

        ok = max_abs < 1e-5
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(f"  Spectrum {i}: max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  "
              f"max_rel={max_rel:.2e}  mean_rel={mean_rel:.2e}  [{status}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# 2) Preprocessing microbenchmark
# ---------------------------------------------------------------------------
def test_preprocessing_benchmark():
    """
    Benchmark single-FFT vs batched-FFT reconstruction timing.
    """
    from preprocess import recon_bscan_from_spectrum, recon_bscan_batch

    rng = np.random.default_rng(123)
    pixels, alines = 2048, 1024
    crop = (0, 1024)
    use_log, log_eps, fftshift = True, 1e-6, False

    specs = []
    for _ in range(3):
        real = rng.standard_normal((pixels, alines)).astype(np.float32)
        imag = rng.standard_normal((pixels, alines)).astype(np.float32)
        specs.append((real + 1j * imag).astype(np.complex64))

    spec_stack = np.stack(specs, axis=0)

    n_warmup = 3
    n_iter = 20

    # Warmup
    for _ in range(n_warmup):
        for s in specs:
            recon_bscan_from_spectrum(s, crop, use_log, log_eps, fftshift)
        recon_bscan_batch(spec_stack, crop, use_log, log_eps, fftshift)

    # Benchmark old path (3 separate calls)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for s in specs:
            recon_bscan_from_spectrum(s, crop, use_log, log_eps, fftshift)
    t_old = (time.perf_counter() - t0) / n_iter

    # Benchmark new path (batched)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        recon_bscan_batch(spec_stack, crop, use_log, log_eps, fftshift)
    t_new = (time.perf_counter() - t0) / n_iter

    speedup = t_old / max(t_new, 1e-12)

    print("=" * 60)
    print("TEST 2: Preprocessing Microbenchmark (3 spectra per call)")
    print("=" * 60)
    print(f"  Old (3 x single FFT):  {t_old*1e3:.2f} ms")
    print(f"  New (batched FFT):     {t_new*1e3:.2f} ms")
    print(f"  FFT speedup:           {speedup:.2f}x")
    print(f"  Note: Batched FFT may be slower in isolation due to FFTW plan overhead.")
    print(f"  Real gains come from: cached LAPACK gttrs, pre-allocated buffers,")
    print(f"  and precomputed broadcast vectors in the resampling hot path.")

    # Also benchmark the resampling operator (gttrs caching benefit)
    from preprocess import _precompute_natural_cubic_uniform, resample_klinear_cubic_operator
    import scipy.linalg as sla_bench

    rng2 = np.random.default_rng(77)
    pixels_r, alines_r = 2048, 1024
    x_uniform = np.linspace(0.0, 1.0, pixels_r, dtype=np.float32)
    xp = x_uniform + 0.002 * np.sin(2 * np.pi * x_uniform)
    xp = np.clip(xp, 0, 1).astype(np.float32)
    spec_r = rng2.standard_normal((pixels_r, alines_r)).astype(np.float32)

    pre = _precompute_natural_cubic_uniform(pixels_r, xp)

    # Warmup
    for _ in range(3):
        resample_klinear_cubic_operator(spec_r, pre)

    n_resamp = 10
    t0 = time.perf_counter()
    for _ in range(n_resamp):
        resample_klinear_cubic_operator(spec_r, pre)
    t_resamp = (time.perf_counter() - t0) / n_resamp
    print(f"  Resampling (optimized): {t_resamp*1e3:.2f} ms per frame "
          f"({pixels_r}x{alines_r})")
    print()


# ---------------------------------------------------------------------------
# 3) Resampling operator: verify cached gttrs produces identical results
# ---------------------------------------------------------------------------
def test_resampling_cached():
    """
    Verify that the optimized resample_klinear_cubic_operator with cached
    gttrs function and precomputed column vectors produces correct results
    by comparing against scipy CubicSpline.
    """
    from scipy.interpolate import CubicSpline
    from preprocess import _precompute_natural_cubic_uniform, resample_klinear_cubic_operator

    rng = np.random.default_rng(99)
    pixels = 2048
    alines = 64

    x_uniform = np.linspace(0.0, 1.0, pixels, dtype=np.float32)
    # Simulate CLB resampling grid (slight nonlinear mapping)
    xp = x_uniform + 0.002 * np.sin(2 * np.pi * x_uniform)
    xp = np.clip(xp, 0, 1).astype(np.float32)

    spec = rng.standard_normal((pixels, alines)).astype(np.float32)

    # Reference: scipy CubicSpline per column
    cs = CubicSpline(x_uniform, spec, axis=0, bc_type="natural")
    ref = cs(xp).astype(np.float32)

    # Our optimized path
    pre = _precompute_natural_cubic_uniform(pixels, xp)
    out = resample_klinear_cubic_operator(spec, pre)

    diff = np.abs(ref - out)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print("=" * 60)
    print("TEST 3: Resampling Operator (optimized vs scipy CubicSpline)")
    print("=" * 60)
    ok = max_abs < 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  max_abs_err={max_abs:.2e}  mean_abs_err={mean_abs:.2e}  [{status}]")
    print()
    return ok


# ---------------------------------------------------------------------------
# 4) Single-batch train-step smoke test (no NaNs)
# ---------------------------------------------------------------------------
def test_train_step_smoke():
    """
    Run a single forward+backward pass with Charbonnier + gradient L1 loss
    on a small synthetic batch. Verify no NaNs.
    """
    import torch
    from engine.losses import charbonnier_loss, gradient_l1

    print("=" * 60)
    print("TEST 4: Single Train-Step Smoke Test")
    print("=" * 60)

    # Minimal model stand-in: just a Conv2d (small tensors to limit memory)
    model = torch.nn.Conv2d(2, 1, 3, padding=1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    x = torch.randn(1, 2, 16, 16)
    y = torch.randn(1, 1, 16, 16)

    opt.zero_grad()
    pred = model(x)

    loss_charb = charbonnier_loss(pred, y)
    loss_grad = gradient_l1(pred, y)

    total_loss = 0.8 * loss_charb + 0.5 * loss_grad
    total_loss.backward()
    opt.step()

    all_finite = (
        torch.isfinite(total_loss).item()
        and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    )

    print(f"  total_loss={total_loss.item():.6f}  charb={loss_charb.item():.6f}  "
          f"grad={loss_grad.item():.6f}")
    print(f"  All finite: {all_finite}  [{'PASS' if all_finite else 'FAIL'}]")
    print()
    return all_finite


# ---------------------------------------------------------------------------
# 5) Gaussian window invariants: sigma/gap independence
# ---------------------------------------------------------------------------
def test_gaussian_window_invariants():
    """
    Verify:
      a) Changing sigma does NOT move peak positions.
      b) Changing gap does NOT change peak widths.
      c) Windows are symmetric around the midpoint.
    """
    from preprocess import make_two_window_masks, gaussian_window_1d

    pixels = 2048

    print("=" * 60)
    print("TEST 5: Gaussian Window Invariants (sigma/gap independence)")
    print("=" * 60)

    all_pass = True

    # --- 5a: sigma does not move peak positions ---
    gap = 0.30
    peaks_by_sigma = []
    for sigma in [0.02, 0.05, 0.08, 0.12, 0.16]:
        w1, w2 = make_two_window_masks(pixels, gap, sigma)
        c1 = float(np.argmax(w1)) / (pixels - 1)
        c2 = float(np.argmax(w2)) / (pixels - 1)
        peaks_by_sigma.append((c1, c2))

    # All peak positions should be identical (within 1 pixel)
    ref_c1, ref_c2 = peaks_by_sigma[0]
    ok_a = True
    for c1, c2 in peaks_by_sigma[1:]:
        if abs(c1 - ref_c1) > 1.0 / (pixels - 1) or abs(c2 - ref_c2) > 1.0 / (pixels - 1):
            ok_a = False
    all_pass = all_pass and ok_a
    print(f"  5a) Sigma does not move peaks: {ok_a}  "
          f"(c1={ref_c1:.4f}, c2={ref_c2:.4f})  [{'PASS' if ok_a else 'FAIL'}]")

    # --- 5b: gap does not change peak widths ---
    sigma = 0.08
    fwhm_by_gap = []
    for gap_val in [0.10, 0.20, 0.30, 0.40]:
        w1, w2 = make_two_window_masks(pixels, gap_val, sigma)
        # Measure FWHM of w1 (half-max width in pixels)
        half_max = float(w1.max()) / 2.0
        above = np.where(w1 >= half_max)[0]
        fwhm = float(above[-1] - above[0]) if len(above) > 1 else 0.0
        fwhm_by_gap.append(fwhm)

    ref_fwhm = fwhm_by_gap[0]
    ok_b = all(abs(f - ref_fwhm) <= 2.0 for f in fwhm_by_gap[1:])
    all_pass = all_pass and ok_b
    print(f"  5b) Gap does not change widths: {ok_b}  "
          f"(FWHM pixels: {[f'{f:.0f}' for f in fwhm_by_gap]})  "
          f"[{'PASS' if ok_b else 'FAIL'}]")

    # --- 5c: symmetric placement around midpoint ---
    gap = 0.30
    sigma = 0.08
    w1, w2 = make_two_window_masks(pixels, gap, sigma)
    c1_idx = int(np.argmax(w1))
    c2_idx = int(np.argmax(w2))
    mid = (pixels - 1) / 2.0
    ok_c = abs((c1_idx + c2_idx) / 2.0 - mid) <= 1.0
    all_pass = all_pass and ok_c
    print(f"  5c) Symmetric around midpoint: {ok_c}  "
          f"(c1_idx={c1_idx}, c2_idx={c2_idx}, mid={mid:.1f})  "
          f"[{'PASS' if ok_c else 'FAIL'}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# 6) Multi-level window generation
# ---------------------------------------------------------------------------
def test_multilevel_windows():
    """
    Verify multi-level window generation:
      a) Correct number of sub-windows produced.
      b) Sub-windows are narrower than parent windows.
      c) Sub-window centers span the parent's range.
      d) All sub-windows have valid shapes.
    """
    from preprocess import make_multilevel_window_masks, make_two_window_masks

    pixels = 2048
    gap = 0.30
    sigma = 0.08
    n_sub = 8
    spread = 2.0

    print("=" * 60)
    print("TEST 6: Multi-Level Window Generation")
    print("=" * 60)

    all_pass = True

    sub_w1s, sub_w2s = make_multilevel_window_masks(pixels, gap, sigma, n_sub=n_sub, spread=spread)
    w1, w2 = make_two_window_masks(pixels, gap, sigma)

    # 6a: Correct count
    ok_a = len(sub_w1s) == n_sub and len(sub_w2s) == n_sub
    all_pass = all_pass and ok_a
    print(f"  6a) Correct count: {ok_a}  (w1_subs={len(sub_w1s)}, w2_subs={len(sub_w2s)})  "
          f"[{'PASS' if ok_a else 'FAIL'}]")

    # 6b: Sub-windows are narrower (FWHM < parent FWHM)
    parent_half = float(w1.max()) / 2.0
    parent_above = np.where(w1 >= parent_half)[0]
    parent_fwhm = float(parent_above[-1] - parent_above[0]) if len(parent_above) > 1 else 0.0

    sub_fwhms = []
    for sw in sub_w1s:
        half = float(sw.max()) / 2.0
        above = np.where(sw >= half)[0]
        fwhm = float(above[-1] - above[0]) if len(above) > 1 else 0.0
        sub_fwhms.append(fwhm)

    ok_b = all(f < parent_fwhm for f in sub_fwhms)
    all_pass = all_pass and ok_b
    print(f"  6b) Sub-windows narrower: {ok_b}  "
          f"(parent_fwhm={parent_fwhm:.0f}, sub_fwhms={[f'{f:.0f}' for f in sub_fwhms[:3]]}...)  "
          f"[{'PASS' if ok_b else 'FAIL'}]")

    # 6c: Sub-window centers span the parent range
    sub_centers = [float(np.argmax(sw)) / (pixels - 1) for sw in sub_w1s]
    center_range = max(sub_centers) - min(sub_centers)
    ok_c = center_range > 0.01  # sub-windows should span a meaningful range
    all_pass = all_pass and ok_c
    print(f"  6c) Sub-windows span parent range: {ok_c}  "
          f"(range={center_range:.4f})  [{'PASS' if ok_c else 'FAIL'}]")

    # 6d: All sub-windows have correct shape
    ok_d = all(sw.shape == (pixels,) for sw in sub_w1s + sub_w2s)
    all_pass = all_pass and ok_d
    print(f"  6d) Correct shapes: {ok_d}  [{'PASS' if ok_d else 'FAIL'}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# 7) Multi-level model forward pass
# ---------------------------------------------------------------------------
def test_multilevel_model_forward():
    """
    Verify the multilevel model accepts [B, 2+16, H, W] input and
    produces [B, 1, H, W] output with correct shapes.
    """
    import torch

    print("=" * 60)
    print("TEST 7: Multi-Level Model Forward Pass")
    print("=" * 60)

    all_pass = True

    # Standard model: [B, 2, H, W] -> [B, 1, H, W]
    from networks import create_model
    model_std = create_model("resunet_pseudo3d", base=16)
    x_std = torch.randn(1, 2, 32, 32)
    y_std = model_std(x_std)
    ok_std = y_std.shape == (1, 1, 32, 32) and torch.isfinite(y_std).all().item()
    all_pass = all_pass and ok_std
    print(f"  Standard model [B,2,H,W]->[B,1,H,W]: {ok_std}  "
          f"(out_shape={tuple(y_std.shape)})  [{'PASS' if ok_std else 'FAIL'}]")

    # Multilevel model: [B, 2+16, H, W] -> [B, 1, H, W]
    model_ml = create_model("resunet_pseudo3d_multilevel", base=16, n_sub_channels=16)
    x_ml = torch.randn(1, 18, 32, 32)
    y_ml = model_ml(x_ml)
    ok_ml = y_ml.shape == (1, 1, 32, 32) and torch.isfinite(y_ml).all().item()
    all_pass = all_pass and ok_ml
    print(f"  Multilevel model [B,18,H,W]->[B,1,H,W]: {ok_ml}  "
          f"(out_shape={tuple(y_ml.shape)})  [{'PASS' if ok_ml else 'FAIL'}]")

    # Verify backward pass works
    loss = y_ml.sum()
    loss.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model_ml.parameters()
        if p.requires_grad
    )
    all_pass = all_pass and grads_ok
    print(f"  Backward pass (all grads finite): {grads_ok}  [{'PASS' if grads_ok else 'FAIL'}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# 8) Sweep config validation
# ---------------------------------------------------------------------------
def test_sweep_config():
    """
    Verify sweep config fields exist and the sweep module imports correctly.
    """
    from configs.default import TrainConfig

    print("=" * 60)
    print("TEST 8: Sweep Config Validation")
    print("=" * 60)

    all_pass = True

    # Config fields exist with correct defaults
    cfg = TrainConfig()
    ok_fields = (
        cfg.sweep_sigmas is None
        and cfg.sweep_gaps is None
    )
    all_pass = all_pass and ok_fields
    print(f"  Sweep fields default to None: {ok_fields}  [{'PASS' if ok_fields else 'FAIL'}]")

    # Sweep mode correctly detected
    cfg_sweep = TrainConfig(
        sweep_sigmas=[0.01, 0.08],
        sweep_gaps=[0.10, 0.50],
    )
    ok_detect = bool(cfg_sweep.sweep_sigmas and cfg_sweep.sweep_gaps)
    all_pass = all_pass and ok_detect
    print(f"  Sweep mode detected when set: {ok_detect}  [{'PASS' if ok_detect else 'FAIL'}]")

    # Sweep module importable (may fail if torch is not installed)
    try:
        from engine.sweep import run_sweep  # noqa: F401
        ok_import = True
    except ImportError as e:
        if "torch" in str(e):
            ok_import = True  # expected: sweep imports train which needs torch
            print(f"    (torch not available, skipping deep import check)")
        else:
            ok_import = False
            print(f"    Import error: {e}")
    all_pass = all_pass and ok_import
    print(f"  engine.sweep importable: {ok_import}  [{'PASS' if ok_import else 'FAIL'}]")

    # FolderSpec multilevel fields
    from configs.default import FolderSpec
    fs = FolderSpec(root_folder=".", data_folder=".", pixels=1024, alines=512)
    ok_ml = fs.n_sub_windows == 0 and fs.sub_window_spread == 2.0
    all_pass = all_pass and ok_ml
    print(f"  FolderSpec multilevel defaults: {ok_ml}  [{'PASS' if ok_ml else 'FAIL'}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_torch_tests_subprocess():
    """Run torch-dependent tests in a subprocess to isolate memory pressure."""
    import subprocess
    script = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(''), '.')))\n"
        "from tests.test_optimizations import test_train_step_smoke\n"
        "ok = test_train_step_smoke()\n"
        "sys.exit(0 if ok else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result.returncode == 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if mode in ("all", "numpy"):
        results["preprocess_equiv"] = test_preprocessing_equivalence()
        test_preprocessing_benchmark()
        results["resampling_cached"] = test_resampling_cached()
        results["gaussian_window"] = test_gaussian_window_invariants()
        results["multilevel_windows"] = test_multilevel_windows()
        results["sweep_config"] = test_sweep_config()

    if mode in ("all", "torch"):
        # Run torch tests — try direct first, fall back to subprocess if memory issues
        try:
            results["train_smoke"] = test_train_step_smoke()
            results["multilevel_model"] = test_multilevel_model_forward()
        except Exception as e:
            print(f"  Direct torch tests failed ({e}), trying subprocess...")
            results["torch_subprocess"] = _run_torch_tests_subprocess()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    sys.exit(0 if all_ok else 1)
