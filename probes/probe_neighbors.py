"""Are adjacent positions in the OVERSAMPLED M3 volumes usable as N2N pairs?

The masks lost on speckle decorrelation and the sub-bands lose on resolution and
signal consistency. Repeat frames won on every physical axis but exist for only
4 acquisitions x 50 frames.

The M3 "x2" volumes are 1024 sequential positions across the same FOV -- ~5.9 um
apart at 6 mm/1024, well inside the beam width. Adjacent positions therefore see
nearly the same tissue with a DIFFERENT scatterer realisation: full bandwidth,
decorrelated speckle, and ~7,800 frames available. The cost is that the signal
is not identical, which is a bias term.

Measures how the structure difference grows with position separation, so the
bias can be traded against decorrelation.
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
FOLDER = "M3_Macula_6x6mm_512x512x2"
CROP = (0, 1024)
TISSUE = slice(120, 700)
LATERAL_SIGMA = 3.0

cfg = FolderSpec(root_folder=ROOT, data_folder=FOLDER, pixels=2048, alines=512,
                 crop_depth=CROP, window_sigma=0.05, gap=0.60, gap_offset=0.015)
proc = BscanProcessor(cfg)
print(f"{FOLDER}: {len(proc.bscan_paths)} frames, CLB={os.path.basename(proc.clb_path)}")
print(f"6 mm across 1024 positions -> {6000/1024:.2f} um between adjacent B-scans\n")


def frame(i):
    return proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)["target_full"]


def structure(a):
    return gaussian_filter1d(a[TISSUE].astype(np.float64), LATERAL_SIGMA, axis=1, mode="nearest")


def speckle(a):
    x = a[TISSUE].astype(np.float64)
    return x - gaussian_filter1d(x, LATERAL_SIGMA, axis=1, mode="nearest")


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


BASE = 400
base = frame(BASE)
sb, kb = structure(base), speckle(base)

print(f"{'separation':>11}{'um':>8}{'full':>9}{'structure':>11}{'speckle':>10}")
rows = []
for d in (1, 2, 3, 4, 6, 8, 16, 32):
    o = frame(BASE + d)
    row = (d, d * 6000 / 1024, corr(base[TISSUE], o[TISSUE]), corr(sb, structure(o)), corr(kb, speckle(o)))
    rows.append(row)
    print(f"{row[0]:>11}{row[1]:>8.1f}{row[2]:>9.4f}{row[3]:>11.4f}{row[4]:>10.4f}")

print()
print("Reference points measured earlier on Maestro2 (same treatment):")
print(f"  {'repeat frames':<34} structure~1.0   speckle=+0.0075")
print(f"  {'gaussian bandgap w1 vs w2':<34} (2.43x PSF)     speckle=+0.0034")
print(f"  {'complementary masks':<34} (1.01x PSF)     speckle=+0.2568")
print()
d1 = rows[0]
print(f"Adjacent positions ({d1[1]:.1f} um): structure {d1[3]:+.4f}, speckle {d1[4]:+.4f}")
print("N2N wants structure correlation near 1 and speckle correlation near 0.")
