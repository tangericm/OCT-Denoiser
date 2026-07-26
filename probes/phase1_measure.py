"""Phase 1: measured PSF values vs analytical predictions, and real-data noise
calibration across every acquisition that ships background frames."""
import glob
import os

import numpy as np

from octdenoiser.physics.masks import (
    make_complementary_masks,
    psf_main_lobe_width,
    sidelobe_ratio,
)
from octdenoiser.physics.noise_model import calibrate_folder

N = 2048

print("=" * 74)
print("A) PSF MAIN LOBE WIDTH — measured vs predicted")
print("=" * 74)
full = np.ones(N, dtype=np.float32)
binary, _ = make_complementary_masks(N, kind="binary", seed=3)
smooth, _ = make_complementary_masks(N, kind="smooth", seed=3)
power, _ = make_complementary_masks(N, kind="power", seed=3)
half = np.zeros(N, dtype=np.float32)
half[: N // 2] = 1.0

rows = [
    ("full band (ones)", full, 1.00),
    ("random binary 50%", binary, 1.00),
    ("smooth random", smooth, None),
    ("power-complementary", power, None),
    ("contiguous half-band", half, 2.41),
]
print(f"{'mask':<24}{'FWHM px':>10}{'predicted':>11}{'sidelobe':>11}")
for name, m, pred in rows:
    w = psf_main_lobe_width(m)
    s = sidelobe_ratio(m)
    p = f"{pred:.2f}" if pred else "-"
    print(f"{name:<24}{w:>10.3f}{p:>11}{s:>11.5f}")

w_full = psf_main_lobe_width(full)
w_half = psf_main_lobe_width(half)
print(f"\n  contiguous sub-band resolution penalty: {w_half / w_full:.2f}x the full band")
print(f"  random mask penalty:                    {psf_main_lobe_width(binary) / w_full:.2f}x")

print()
print("=" * 74)
print("B) SPECKLE DECORRELATION — mask overlap drives it")
print("=" * 74)
rng = np.random.default_rng(0)
# A synthetic spectrum with speckle: random-phase scatterers.
spec = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)


def recon(mask):
    return np.abs(np.fft.ifft(spec * mask))


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


for label, kind in (("binary (disjoint)", "binary"), ("smooth (overlapping)", "smooth"),
                    ("power (overlapping)", "power")):
    m1, m2 = make_complementary_masks(N, kind=kind, seed=11)
    overlap = float(np.sum(np.minimum(m1, m2)) / np.sum(np.maximum(m1, m2)))
    print(f"  {label:<22} support overlap={overlap:.3f}   corr(recon1,recon2)={corr(recon(m1), recon(m2)):+.4f}")
print("\n  Reference: repeat B-scans of the SAME tissue correlate 0.975 in linear")
print("  magnitude, i.e. repeats barely decorrelate speckle at all.")

print()
print("=" * 74)
print("C) REAL-DATA NOISE CALIBRATION (background frames)")
print("=" * 74)
ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data"
results = []
for inst in ("Maestro2", "Maestro3"):
    base = os.path.join(ROOT, inst)
    if not os.path.isdir(base):
        continue
    for folder in sorted(os.listdir(base)):
        d = os.path.join(base, folder)
        backs = sorted(glob.glob(os.path.join(d, "back*.raw")))
        if not backs:
            continue
        alines = os.path.getsize(backs[0]) // (2048 * 2)
        try:
            model, curve = calibrate_folder(d, 2048, int(alines))
        except Exception as e:                              # noqa: BLE001
            print(f"  {folder:<34} FAILED: {type(e).__name__}: {e}")
            continue
        results.append((folder, model))
        flag = "OK " if model.is_physical() else "BAD"
        print(f"  [{flag}] {folder:<32} n={model.n_frames} alines={alines}")
        print(f"          read_var={model.read_var:10.2f}  gain={model.gain:7.4f}  "
              f"rin={model.rin:.3e}  R2={model.r_squared:.5f}")

if results:
    print()
    print("  " + "-" * 70)
    gains = np.array([m.gain for _, m in results])
    reads = np.array([m.read_var for _, m in results])
    rins = np.array([m.rin for _, m in results])
    n_phys = sum(m.is_physical() for _, m in results)
    print(f"  {len(results)} acquisitions calibrated, {n_phys} physical (read_var >= 0)")
    print(f"  gain     : {gains.min():.4f} .. {gains.max():.4f}   (spread {gains.max()/max(gains.min(),1e-9):.2f}x)")
    print(f"  read_var : {reads.min():.1f} .. {reads.max():.1f}")
    print(f"  rin      : {rins.min():.3e} .. {rins.max():.3e}")
    print("\n  Per-acquisition calibration is required if gain varies materially.")
else:
    print("  no folders with background frames found")

print()
print("  NOTE: Maestro2 line-scan folders ship no back*.raw, so the 50-frame")
print("  repeat stacks cannot be PTC-calibrated directly from dark frames.")
