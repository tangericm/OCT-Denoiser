"""Decisive test, like-for-like on REAL data.

Compares, on identical frames and through the identical reconstruction path:
  (a) repeat-frame decorrelation  -- what repeat-based N2N would give
  (b) mask-pair decorrelation     -- what the proposed method gives
  (c) Gaussian bandgap            -- what the existing method gives

Earlier numbers were not comparable: 0.975 came from a crude reconstruction
without k-linearisation whose DC artifact inflated it, and -0.022 came from a
synthetic spectrum.
"""
import os

import numpy as np

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import correlation
from octdenoiser.physics.masks import make_complementary_masks
from octdenoiser.preprocess import (
    BscanProcessor,
    make_two_window_masks,
    recon_bscan_batch,
    resample_klinear_cubic_operator,
)

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
FOLDER = "Line_6mm_2048Aline_135degCW_50frame_gain165"
CROP = (0, 1024)
TISSUE = slice(146, 360)   # measured energy band for this stack

spec_cfg = FolderSpec(root_folder=ROOT, data_folder=FOLDER, pixels=2048, alines=2048,
                      crop_depth=CROP, window_sigma=0.05, gap=0.60, gap_offset=0.015)
proc = BscanProcessor(spec_cfg)


def klin(path, idx):
    """Raw -> DC subtract -> k-linear resample, i.e. process_one's spectrum."""
    cfg = proc.cfg
    raw = np.fromfile(path, dtype=np.uint16).reshape((cfg.pixels, cfg.alines), order="F").astype(np.float32)
    raw[0:3, :] = raw[3, :]
    raw[:, 0] = raw[:, 1]
    raw -= raw.mean(axis=1, keepdims=True)
    return resample_klinear_cubic_operator(raw, proc._spline_pre).astype(np.complex64)


def recon(specs):
    batch = np.stack(specs).astype(np.complex64)
    return recon_bscan_batch(batch, CROP, True, 1e-6, False, fft_workers=-1)


s0 = klin(proc.bscan_paths[0], 0)
s1 = klin(proc.bscan_paths[1], 1)
N = s0.shape[0]

print("=" * 74)
print(f"{FOLDER}")
print(f"tissue rows {TISSUE.start}:{TISSUE.stop}")
print("=" * 74)


def report(label, a, b):
    full = correlation(a, b)
    tis = correlation(a[TISSUE], b[TISSUE])
    print(f"  {label:<42} full={full:+.4f}  tissue={tis:+.4f}")
    return tis


print("\n(a) REPEAT FRAMES — two acquisitions, full bandwidth each")
f0, f1 = recon([s0, s1])
rep = report("frame 0 vs frame 1", f0, f1)

print("\n(b) COMPLEMENTARY MASKS — one frame, disjoint k-support")
mask_vals = []
for seed in (0, 1, 2):
    m1, m2 = make_complementary_masks(N, kind="binary", seed=seed)
    a, b = recon([s0 * m1[:, None], s0 * m2[:, None]])
    mask_vals.append(report(f"mask pair, seed {seed}", a, b))

print("\n(c) GAUSSIAN BANDGAP — the existing method's two views")
w1, w2 = make_two_window_masks(N, spec_cfg.gap, spec_cfg.window_sigma, spec_cfg.gap_offset)
a, b = recon([s0 * w1[:, None], s0 * w2[:, None]])
bg = report(f"w1 vs w2 (sigma={spec_cfg.window_sigma}, gap={spec_cfg.gap})", a, b)
overlap = float(np.sum(np.minimum(w1, w2)) / np.sum(np.maximum(w1, w2)))
print(f"  {'gaussian window support overlap':<42} {overlap:.4f}")

print("\n(d) EXISTING METHOD'S TARGET LEAK — sub-band vs full band")
full_recon, w1_recon = recon([s0, s0 * w1[:, None]])
leak = report("w1 vs full band (input vs its target)", w1_recon, full_recon)

print()
print("=" * 74)
print("SUMMARY — tissue-band correlation, lower = better decorrelation")
print("=" * 74)
print(f"  repeat frames                {rep:+.4f}")
print(f"  complementary masks (mean)   {np.mean(mask_vals):+.4f}")
print(f"  gaussian bandgap w1 vs w2    {bg:+.4f}")
print(f"  bandgap input vs its target  {leak:+.4f}   <- N2N requires this near 0")
