"""Characterise the adjacent-position N2N bias.

With mismatched signal, N2N converges to E[s2 | x1] rather than s1. Two
consequences are measurable without training anything:

  A) BIAS BUDGET. The irreducible excess error is Var(s2 - s1), the structure
     difference between the paired positions. It only matters relative to the
     noise being removed, Var(speckle). Small ratio -> the bias is cheap.

  B) SLOW-AXIS BLUR. The converged estimator is the conditional mean over the
     paired neighbourhood, so it blurs along the slow (B-scan) axis. Measured as
     the broadening of the en-face slow-axis autocorrelation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
FOLDER = "M3_Macula_6x6mm_512x512x2"
CROP = (0, 1024)
TISSUE = slice(120, 700)
LATERAL_SIGMA = 3.0
START, N_POS = 380, 80
UM_PER_POS = 6000 / 1024

cfg = FolderSpec(root_folder=ROOT, data_folder=FOLDER, pixels=2048, alines=512,
                 crop_depth=CROP, window_sigma=0.05, gap=0.60, gap_offset=0.015)
proc = BscanProcessor(cfg)

print(f"{FOLDER}: loading positions {START}..{START+N_POS-1} ({UM_PER_POS:.2f} um apart)")
vol = np.stack([
    proc.process_one(proc.bscan_paths[START + i], frame_idx=START + i, fft_workers=-1)["target_full"][TISSUE]
    for i in range(N_POS)
]).astype(np.float64)
print(f"  volume {vol.shape}  [positions, depth, alines]\n")

struct = gaussian_filter1d(vol, LATERAL_SIGMA, axis=2, mode="nearest")
speck = vol - struct
var_speck = float(speck.var())
print(f"speckle variance (the quantity being removed) = {var_speck:.5f}\n")

print("=" * 74)
print("A) BIAS BUDGET — structure difference vs speckle being removed")
print("=" * 74)
print(f"{'sep':>5}{'um':>8}{'Var(s2-s1)':>13}{'/Var(speck)':>13}{'speckle corr':>14}")
rows = []
for d in (1, 2, 3, 4, 6, 8, 12, 16):
    a, b = struct[:-d], struct[d:]
    var_diff = float(((b - a) ** 2).mean())
    sa, sb = speck[:-d].ravel(), speck[d:].ravel()
    sc = float(((sa - sa.mean()) @ (sb - sb.mean())) / (np.linalg.norm(sa - sa.mean()) * np.linalg.norm(sb - sb.mean())))
    rows.append((d, var_diff, var_diff / var_speck, sc))
    print(f"{d:>5}{d*UM_PER_POS:>8.1f}{var_diff:>13.5f}{var_diff/var_speck:>13.4f}{sc:>14.4f}")

print()
print("=" * 74)
print("B) SLOW-AXIS BLUR — what the converged estimator costs")
print("=" * 74)


def slow_axis_acf_width(v):
    """FWHM of the slow-axis autocorrelation of the structure, in positions."""
    x = v - v.mean(axis=0, keepdims=True)
    n = x.shape[0]
    acf = np.array([
        float((x[: n - k] * x[k:]).mean() / max((x * x).mean(), 1e-12))
        for k in range(min(24, n))
    ])
    half = 0.5
    for k in range(1, acf.size):
        if acf[k] < half:
            t = (acf[k - 1] - half) / max(acf[k - 1] - acf[k], 1e-12)
            return 2.0 * (k - 1 + t)
    return float("nan")


base_w = slow_axis_acf_width(struct)
print(f"  structure slow-axis ACF FWHM (unblurred): {base_w:.2f} positions "
      f"= {base_w*UM_PER_POS:.1f} um")
print(f"\n{'sep':>5}{'um':>8}{'ACF FWHM':>11}{'broadening':>12}")
for d in (1, 2, 3, 4, 6, 8):
    # Converged estimator ~ mean of the two paired views' structure.
    est = 0.5 * (struct[:-d] + struct[d:])
    w = slow_axis_acf_width(est)
    print(f"{d:>5}{d*UM_PER_POS:>8.1f}{w:>11.2f}{w/base_w:>11.2f}x")

print()
print("=" * 74)
print("VERDICT")
print("=" * 74)
for d, vd, ratio, sc in rows:
    if d in (2, 4, 8):
        print(f"  sep {d} ({d*UM_PER_POS:5.1f} um): bias/noise={ratio:.3f}  speckle corr={sc:+.4f}")
best = min(rows, key=lambda r: r[2] + abs(r[3]) * 5)
print(f"\n  lowest combined bias+residual-correlation cost: separation {best[0]} "
      f"({best[0]*UM_PER_POS:.1f} um)")
