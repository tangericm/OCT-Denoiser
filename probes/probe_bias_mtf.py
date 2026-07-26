"""The bias is an IN-PLANE transfer function, not a slow-axis blur.

A 2D network sees one B-scan: axes (z, x). The slow axis y is in neither its
input nor its output, so it cannot blur along y. Averaging adjacent positions is
something the VOLUME can do; the network cannot.

What it converges to is E[s_{i+d} | x_i]. Writing s_{i+d} = rho * s_i + e with
e orthogonal to s_i, the output is rho * s_i. If rho were one scalar that is a
pure contrast shrink. It is not: rho depends on in-plane spatial frequency,
because fine in-plane features have small y-extent and therefore decorrelate
faster with position displacement than coarse ones.

So rho(kx, kz) IS the bias filter, acting in the plane. This measures it, and
converts it to an equivalent in-plane PSF broadening.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
FOLDER = "M3_Macula_6x6mm_512x512x2"
CROP = (0, 1024)
TISSUE = slice(120, 632)     # 512 rows, power of two for the FFT
LATERAL_SIGMA = 3.0
START, N_POS = 380, 80

cfg = FolderSpec(root_folder=ROOT, data_folder=FOLDER, pixels=2048, alines=512,
                 crop_depth=CROP, window_sigma=0.05, gap=0.60, gap_offset=0.015)
proc = BscanProcessor(cfg)

vol = np.stack([
    proc.process_one(proc.bscan_paths[START + i], frame_idx=START + i,
                     fft_workers=-1)["target_full"][TISSUE]
    for i in range(N_POS)
]).astype(np.float64)
struct = gaussian_filter1d(vol, LATERAL_SIGMA, axis=2, mode="nearest")
nz, nx = struct.shape[1], struct.shape[2]
print(f"{FOLDER}: {vol.shape} [pos, z, x]\n")

win = np.hanning(nz)[:, None] * np.hanning(nx)[None, :]


def coherence(d):
    """rho(kx,kz) between structure at positions i and i+d, averaged over i."""
    cross = np.zeros((nz, nx), dtype=complex)
    pa = np.zeros((nz, nx))
    pb = np.zeros((nz, nx))
    for i in range(N_POS - d):
        A = np.fft.fft2((struct[i] - struct[i].mean()) * win)
        B = np.fft.fft2((struct[i + d] - struct[i + d].mean()) * win)
        cross += A * np.conj(B)
        pa += np.abs(A) ** 2
        pb += np.abs(B) ** 2
    return np.real(cross) / np.sqrt(pa * pb + 1e-30)


def radial(rho):
    """Average rho over radial in-plane frequency bins."""
    fz = np.fft.fftfreq(nz)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    r = np.sqrt(fz**2 + fx**2)
    edges = np.linspace(0, 0.5, 11)
    out = []
    for k in range(len(edges) - 1):
        m = (r >= edges[k]) & (r < edges[k + 1])
        out.append((0.5 * (edges[k] + edges[k + 1]), float(rho[m].mean())))
    return out


print("=" * 72)
print("BIAS TRANSFER FUNCTION rho(f) — flat means pure contrast shrink,")
print("falling means in-plane low-pass, i.e. genuine blur")
print("=" * 72)
print(f"{'freq (cyc/px)':>14}" + "".join(f"{f'd={d}':>10}" for d in (1, 2, 4, 8)))

curves = {d: radial(coherence(d)) for d in (1, 2, 4, 8)}
for k in range(10):
    f = curves[1][k][0]
    print(f"{f:>14.3f}" + "".join(f"{curves[d][k][1]:>10.4f}" for d in (1, 2, 4, 8)))

print()
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)
for d in (1, 2, 4, 8):
    vals = np.array([v for _, v in curves[d]])
    lo = vals[:3].mean()      # coarse structure
    hi = vals[-3:].mean()     # fine structure
    rho2d = coherence(d)
    # Equivalent in-plane PSF from the transfer function.
    psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.maximum(rho2d, 0))))
    prof = psf[psf.shape[0] // 2]
    pk = prof.max()
    above = np.flatnonzero(prof >= pk / 2)
    fwhm = float(above.max() - above.min() + 1)
    print(f"  d={d} ({d*5.86:5.1f} um): rho_coarse={lo:.4f}  rho_fine={hi:.4f}  "
          f"ratio={hi/max(lo,1e-9):.3f}   equiv in-plane PSF FWHM={fwhm:.1f} px")

print()
print("  A flat rho would mean the bias is a scalar contrast factor and the")
print("  correct fix is a single gain calibration. A falling rho means real")
print("  in-plane resolution loss, and the equivalent PSF FWHM is its size.")
