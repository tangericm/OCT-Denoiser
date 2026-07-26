"""Two deciding measurements, on real data, identical treatment for all methods.

1. SPECKLE DECORRELATION, structure removed.
   Full-frame correlation is structure-dominated and cannot separate the
   methods (it gave 0.612 masks vs 0.606 bandgap). Speckle is high-frequency
   along the A-line axis while layer structure is not, so subtracting a
   laterally-smoothed version isolates the speckle-scale component. Applied
   identically to every construction, so the comparison is fair.

2. SIGNAL CONSISTENCY.
   N2N requires E[target | input] = signal. Contiguous sub-bands sit at
   different k, and scattering is wavelength-dependent, so their expected
   signals may differ systematically -- that is bias, not noise, and training
   cannot average it away. Random full-range masks sample the same spectral
   range statistically, so their expected signals should match.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
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
TISSUE = slice(146, 360)
LATERAL_SIGMA = 3.0   # speckle grain scale along x

cfg = FolderSpec(root_folder=ROOT, data_folder=FOLDER, pixels=2048, alines=2048,
                 crop_depth=CROP, window_sigma=0.05, gap=0.60, gap_offset=0.015)
proc = BscanProcessor(cfg)


def klin(path):
    raw = np.fromfile(path, dtype=np.uint16).reshape((cfg.pixels, cfg.alines), order="F").astype(np.float32)
    raw[0:3, :] = raw[3, :]
    raw[:, 0] = raw[:, 1]
    raw -= raw.mean(axis=1, keepdims=True)
    return resample_klinear_cubic_operator(raw, proc._spline_pre).astype(np.complex64)


def recon(specs):
    return recon_bscan_batch(np.stack(specs).astype(np.complex64), CROP, True, 1e-6, False, fft_workers=-1)


def speckle_part(img):
    """Remove the laterally-smooth component; what remains is speckle-scale."""
    a = img[TISSUE].astype(np.float64)
    return a - gaussian_filter1d(a, LATERAL_SIGMA, axis=1, mode="nearest")


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


s0 = klin(proc.bscan_paths[0])
s1 = klin(proc.bscan_paths[1])
N = s0.shape[0]

print("=" * 76)
print("1) SPECKLE DECORRELATION — lateral structure removed (sigma=%.1f px)" % LATERAL_SIGMA)
print("=" * 76)
print(f"{'construction':<40}{'full corr':>11}{'speckle corr':>14}")

f0, f1 = recon([s0, s1])
print(f"{'repeat frames (f0 vs f1)':<40}{corr(f0[TISSUE], f1[TISSUE]):>11.4f}"
      f"{corr(speckle_part(f0), speckle_part(f1)):>14.4f}")

mask_full, mask_spk = [], []
for seed in range(5):
    m1, m2 = make_complementary_masks(N, kind="binary", seed=seed)
    a, b = recon([s0 * m1[:, None], s0 * m2[:, None]])
    mask_full.append(corr(a[TISSUE], b[TISSUE]))
    mask_spk.append(corr(speckle_part(a), speckle_part(b)))
print(f"{'complementary masks (5 seeds)':<40}{np.mean(mask_full):>11.4f}{np.mean(mask_spk):>14.4f}")

w1, w2 = make_two_window_masks(N, cfg.gap, cfg.window_sigma, cfg.gap_offset)
a, b = recon([s0 * w1[:, None], s0 * w2[:, None]])
bg_full, bg_spk = corr(a[TISSUE], b[TISSUE]), corr(speckle_part(a), speckle_part(b))
print(f"{'gaussian bandgap (w1 vs w2)':<40}{bg_full:>11.4f}{bg_spk:>14.4f}")

fb, w1r = recon([s0, s0 * w1[:, None]])
print(f"{'bandgap input vs FULL-BAND target':<40}"
      f"{corr(w1r[TISSUE], fb[TISSUE]):>11.4f}{corr(speckle_part(w1r), speckle_part(fb)):>14.4f}")

print()
print("=" * 76)
print("2) SIGNAL CONSISTENCY — do the two views see the same expected signal?")
print("=" * 76)


def profile_stats(label, a, b):
    """Mean A-line profile agreement between two views."""
    pa = a[TISSUE].mean(axis=1)
    pb = b[TISSUE].mean(axis=1)
    rel = float(np.mean(np.abs(pa - pb)) / (np.mean(np.abs(pa)) + 1e-12))
    # Systematic tilt: does the difference trend with depth?
    z = np.arange(pa.size)
    slope = float(np.polyfit(z, pa - pb, 1)[0])
    print(f"  {label:<38} profile_corr={np.corrcoef(pa, pb)[0,1]:+.5f}  "
          f"mean|rel diff|={rel:.4f}  depth_slope={slope:+.3e}")
    return rel


aw, bw = recon([s0 * w1[:, None], s0 * w2[:, None]])
rel_bg = profile_stats("gaussian bandgap w1 vs w2", aw, bw)

rels = []
for seed in range(5):
    m1, m2 = make_complementary_masks(N, kind="binary", seed=seed)
    am, bm = recon([s0 * m1[:, None], s0 * m2[:, None]])
    rels.append(profile_stats(f"complementary masks seed {seed}", am, bm))

print()
print("=" * 76)
print("VERDICT")
print("=" * 76)
print(f"  speckle correlation   masks={np.mean(mask_spk):+.4f}   bandgap={bg_spk:+.4f}")
print(f"  signal mismatch       masks={np.mean(rels):.4f}    bandgap={rel_bg:.4f}"
      f"   ratio={rel_bg / max(np.mean(rels), 1e-12):.1f}x")
