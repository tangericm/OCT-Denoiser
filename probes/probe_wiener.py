"""The transfer function every supervision scheme converges to, compared.

MSE training converges to E[target | input]. Its optimal linear form is the
Wiener filter, whose transfer function is the magnitude-squared coherence
between input and target as a function of in-plane spatial frequency. So
measuring raw coherence -- NO pre-smoothing -- directly predicts what each
scheme's trained network will and will not preserve.

Coherence near 1 at a frequency: that content is shared and will be preserved.
Coherence near 0: the target carries no information about it, so the network
learns to suppress it. For SPECKLE that is exactly what we want. For real
STRUCTURE it is resolution loss.

The previous attempt pre-smoothed laterally with sigma=3 px to separate speckle
from structure, which attenuates x-frequencies above ~0.1 cyc/px by 94% and made
the high-frequency numbers meaningless. This avoids that entirely.
"""
import numpy as np

from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import (
    BscanProcessor,
    make_two_window_masks,
    recon_bscan_batch,
    resample_klinear_cubic_operator,
)

NZ = NX = 512
BINS = np.linspace(0, 0.5, 11)


def coherence_curve(pairs):
    """Radially binned coherence over a list of (a, b) image pairs."""
    win = np.hanning(NZ)[:, None] * np.hanning(NX)[None, :]
    cross = np.zeros((NZ, NX), dtype=complex)
    pa = np.zeros((NZ, NX))
    pb = np.zeros((NZ, NX))
    for a, b in pairs:
        A = np.fft.fft2((a - a.mean()) * win)
        B = np.fft.fft2((b - b.mean()) * win)
        cross += A * np.conj(B)
        pa += np.abs(A) ** 2
        pb += np.abs(B) ** 2
    rho = np.real(cross) / np.sqrt(pa * pb + 1e-30)

    fz = np.fft.fftfreq(NZ)[:, None]
    fx = np.fft.fftfreq(NX)[None, :]
    r = np.sqrt(fz**2 + fx**2)
    return [float(rho[(r >= BINS[k]) & (r < BINS[k + 1])].mean()) for k in range(10)]


# ---------------------------------------------------------------- Maestro3
M3 = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
cfg3 = FolderSpec(root_folder=M3, data_folder="M3_Macula_6x6mm_512x512x2",
                  pixels=2048, alines=512, crop_depth=(0, 1024),
                  window_sigma=0.05, gap=0.60, gap_offset=0.015)
p3 = BscanProcessor(cfg3)
Z = slice(120, 632)

frames = {}
for i in range(380, 400):
    frames[i] = p3.process_one(p3.bscan_paths[i], frame_idx=i, fft_workers=-1)["target_full"][Z]

curves = {}
for d in (1, 2, 4):
    curves[f"adjacent d={d} ({d*5.86:.1f}um)"] = coherence_curve(
        [(frames[i], frames[i + d]) for i in range(380, 400 - d)]
    )

# Sub-bands on the SAME data, so the comparison is within-dataset.
w1, w2 = make_two_window_masks(2048, cfg3.gap, cfg3.window_sigma, cfg3.gap_offset)


def subband_pair(proc, path, idx):
    cfg = proc.cfg
    raw = np.fromfile(path, dtype=np.uint16).reshape((cfg.pixels, cfg.alines), order="F").astype(np.float32)
    raw[0:3, :] = raw[3, :]
    raw[:, 0] = raw[:, 1]
    raw -= raw.mean(axis=1, keepdims=True)
    s = resample_klinear_cubic_operator(raw, proc._spline_pre).astype(np.complex64)
    a, b = recon_bscan_batch(np.stack([s * w1[:, None], s * w2[:, None]]).astype(np.complex64),
                             (0, 1024), True, 1e-6, False, fft_workers=-1)
    return a[Z], b[Z]


curves["subband w1 vs w2"] = coherence_curve(
    [subband_pair(p3, p3.bscan_paths[i], i) for i in range(380, 392)]
)

# ---------------------------------------------------------------- Maestro2 repeats
M2 = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
cfg2 = FolderSpec(root_folder=M2, data_folder="Line_6mm_2048Aline_135degCW_50frame_gain165",
                  pixels=2048, alines=2048, crop_depth=(0, 1024),
                  window_sigma=0.05, gap=0.60, gap_offset=0.015)
p2 = BscanProcessor(cfg2)
ZR, XR = slice(120, 632), slice(700, 1212)
rep = [p2.process_one(p2.bscan_paths[i], frame_idx=i, fft_workers=-1)["target_full"][ZR, XR]
       for i in range(12)]
curves["repeat frames"] = coherence_curve([(rep[i], rep[i + 1]) for i in range(11)])

# ---------------------------------------------------------------- report
print("=" * 92)
print("COHERENCE vs IN-PLANE SPATIAL FREQUENCY — what each scheme preserves")
print("=" * 92)
names = list(curves)
print(f"{'freq':>7}" + "".join(f"{n[:20]:>21}" for n in names))
for k in range(10):
    f = 0.5 * (BINS[k] + BINS[k + 1])
    print(f"{f:>7.3f}" + "".join(f"{curves[n][k]:>21.4f}" for n in names))

print()
print("=" * 92)
print("HALF-COHERENCE CUTOFF — finest in-plane detail each scheme can preserve")
print("=" * 92)
for n in names:
    v = np.array(curves[n])
    fs = np.array([0.5 * (BINS[k] + BINS[k + 1]) for k in range(10)])
    below = np.flatnonzero(v < 0.5)
    if below.size == 0:
        print(f"  {n:<28} coherence stays above 0.5 across the band")
        continue
    k = below[0]
    if k == 0:
        print(f"  {n:<28} below 0.5 even at the coarsest scale")
        continue
    t = (v[k - 1] - 0.5) / max(v[k - 1] - v[k], 1e-12)
    fc = fs[k - 1] + t * (fs[k] - fs[k - 1])
    print(f"  {n:<28} f_c={fc:.3f} cyc/px  ->  finest period {1/max(fc,1e-9):.1f} px")
