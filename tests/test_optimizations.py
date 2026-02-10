"""
Validation harness for preprocessing optimizations and smooth SNR loss.

Run:  python tests/test_optimizations.py

Tests:
  1. Preprocessing equivalence: old (single FFT) vs new (batched FFT)
  2. Preprocessing microbenchmark
  3. Smooth SNR loss: finite outputs, finite gradients, monotonic behavior
  4. Single-batch train-step smoke test (no NaNs)
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
# 4) Smooth SNR loss: finite outputs, gradients, monotonic behavior
# ---------------------------------------------------------------------------
def test_smooth_snr_loss():
    """
    Verify:
      a) Finite outputs and finite gradients on random input.
      b) Higher peak / lower noise => lower loss (better SNR).
    """
    import torch
    from engine.losses import smooth_snr_loss

    print("=" * 60)
    print("TEST 4: Smooth SNR Loss — Gradients & Behavior")
    print("=" * 60)

    all_pass = True

    # --- 4a: finite outputs and gradients ---
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(4, 1, 64, 64, generator=rng, requires_grad=True)
    loss, info = smooth_snr_loss(x, t_peak=0.1, t_bg=0.1)
    loss.backward()

    loss_finite = torch.isfinite(loss).item()
    grad_finite = x.grad is not None and torch.isfinite(x.grad).all().item()
    ok_a = loss_finite and grad_finite
    all_pass = all_pass and ok_a
    print(f"  4a) Finite loss: {loss_finite}  Finite grad: {grad_finite}  "
          f"loss={loss.item():.4f}  [{('PASS' if ok_a else 'FAIL')}]")
    print(f"      info: soft_peak={info['soft_peak'].item():.4f}  "
          f"soft_std_bg={info['soft_std_bg'].item():.4f}  "
          f"snr={info['snr'].item():.4f}")

    # --- 4b: higher peak => lower loss ---
    # Create two images: one with a strong bright region, one uniform
    base = torch.zeros(1, 1, 64, 64)
    bright = base.clone()
    bright[:, :, 10:20, 10:20] = 5.0  # strong signal region

    loss_uniform, _ = smooth_snr_loss(base + 0.01 * torch.randn_like(base), t_peak=0.1, t_bg=0.1)
    loss_bright, _ = smooth_snr_loss(bright + 0.01 * torch.randn_like(bright), t_peak=0.1, t_bg=0.1)

    ok_b = loss_bright.item() < loss_uniform.item()
    all_pass = all_pass and ok_b
    print(f"  4b) Bright signal has lower loss: {ok_b}  "
          f"(uniform={loss_uniform.item():.4f}, bright={loss_bright.item():.4f})  "
          f"[{'PASS' if ok_b else 'FAIL'}]")

    # --- 4c: lower noise => lower loss ---
    low_noise = torch.zeros(1, 1, 64, 64)
    low_noise[:, :, 10:20, 10:20] = 5.0
    low_noise += 0.01 * torch.randn_like(low_noise)

    high_noise = torch.zeros(1, 1, 64, 64)
    high_noise[:, :, 10:20, 10:20] = 5.0
    high_noise += 1.0 * torch.randn_like(high_noise)

    loss_low, _ = smooth_snr_loss(low_noise, t_peak=0.1, t_bg=0.1)
    loss_high, _ = smooth_snr_loss(high_noise, t_peak=0.1, t_bg=0.1)

    ok_c = loss_low.item() < loss_high.item()
    all_pass = all_pass and ok_c
    print(f"  4c) Low noise has lower loss: {ok_c}  "
          f"(low_noise={loss_low.item():.4f}, high_noise={loss_high.item():.4f})  "
          f"[{'PASS' if ok_c else 'FAIL'}]")

    # --- 4d: various temperatures produce finite results ---
    ok_d = True
    for tp in [0.01, 0.1, 1.0, 10.0]:
        for tb in [0.01, 0.1, 1.0, 10.0]:
            x_t = torch.randn(1, 1, 16, 16, requires_grad=True)
            l, _ = smooth_snr_loss(x_t, t_peak=tp, t_bg=tb)
            l.backward()
            if not (torch.isfinite(l).item() and torch.isfinite(x_t.grad).all().item()):
                ok_d = False
                print(f"    FAIL at t_peak={tp}, t_bg={tb}")
    all_pass = all_pass and ok_d
    print(f"  4d) All temperature combos finite: {ok_d}  [{'PASS' if ok_d else 'FAIL'}]")

    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# 5) Single-batch train-step smoke test (no NaNs)
# ---------------------------------------------------------------------------
def test_train_step_smoke():
    """
    Run a single forward+backward pass with all three loss components
    on a small synthetic batch. Verify no NaNs.
    """
    import torch
    from engine.losses import charbonnier_loss, gradient_l1, smooth_snr_loss

    print("=" * 60)
    print("TEST 5: Single Train-Step Smoke Test")
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
    loss_snr, snr_info = smooth_snr_loss(pred, t_peak=0.1, t_bg=0.1)

    total_loss = 0.8 * loss_charb + 0.5 * loss_grad + 0.1 * loss_snr
    total_loss.backward()
    opt.step()

    all_finite = (
        torch.isfinite(total_loss).item()
        and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    )

    print(f"  total_loss={total_loss.item():.6f}  charb={loss_charb.item():.6f}  "
          f"grad={loss_grad.item():.6f}  snr={loss_snr.item():.6f}")
    print(f"  All finite: {all_finite}  [{'PASS' if all_finite else 'FAIL'}]")
    print()
    return all_finite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_torch_tests_subprocess():
    """Run torch-dependent tests in a subprocess to isolate memory pressure."""
    import subprocess
    script = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(''), '.')))\n"
        "from tests.test_optimizations import test_smooth_snr_loss, test_train_step_smoke\n"
        "ok1 = test_smooth_snr_loss()\n"
        "ok2 = test_train_step_smoke()\n"
        "sys.exit(0 if (ok1 and ok2) else 1)\n"
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

    if mode in ("all", "torch"):
        # Run torch tests — try direct first, fall back to subprocess if memory issues
        try:
            results["snr_loss"] = test_smooth_snr_loss()
            results["train_smoke"] = test_train_step_smoke()
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
