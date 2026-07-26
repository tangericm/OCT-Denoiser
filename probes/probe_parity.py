"""Are the M3 "x2" volumes OCTA repeat pairs, or sequential oversampled positions?

My earlier test averaged separation-1 correlation over ALL starting indices. If
the layout is interleaved [p0r0, p0r1, p1r0, p1r1, ...] then half of those pairs
are repeats and half are cross-position, which halves the contrast and hides the
effect. This splits by parity instead.

  even-start pairs (0,1), (2,3), (4,5) ... -> REPEATS if interleaved
  odd-start  pairs (1,2), (3,4), (5,6) ... -> adjacent positions

512 positions x 2 repeats = 1024 B-scans, which matches the file count exactly,
and OCTA needs repeats at each position to detect flow.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
Z = slice(120, 632)
LATERAL_SIGMA = 3.0
START, N = 380, 60


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


for folder, alines in (("M3_Macula_6x6mm_512x512x2", 512),
                       ("M3_Disc_9x9mm_512x512x2", 512),
                       ("M3_Macula_3x3mm_320x320x2", 320)):
    cfg = FolderSpec(root_folder=ROOT, data_folder=folder, pixels=2048, alines=alines,
                     crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015)
    proc = BscanProcessor(cfg)
    fr = {}
    for i in range(START, START + N):
        fr[i] = proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)["target_full"][Z]

    st = {i: gaussian_filter1d(v.astype(np.float64), LATERAL_SIGMA, axis=1, mode="nearest")
          for i, v in fr.items()}
    sp = {i: v.astype(np.float64) - st[i] for i, v in fr.items()}

    def stats(idxs):
        full = [corr(fr[i], fr[i + 1]) for i in idxs]
        stc = [corr(st[i], st[i + 1]) for i in idxs]
        spc = [corr(sp[i], sp[i + 1]) for i in idxs]
        return np.mean(full), np.mean(stc), np.mean(spc)

    even = [i for i in range(START, START + N - 1) if i % 2 == 0]
    odd = [i for i in range(START, START + N - 1) if i % 2 == 1]

    print("=" * 74)
    print(f"{folder}   ({len(proc.bscan_paths)} frames, {alines} A-lines)")
    print("=" * 74)
    print(f"{'pair type':<34}{'full':>10}{'structure':>12}{'speckle':>10}")
    for name, idxs in (("even-start (i even -> i,i+1)", even), ("odd-start  (i odd  -> i,i+1)", odd)):
        f, s, k = stats(idxs)
        print(f"{name:<34}{f:>10.4f}{s:>12.4f}{k:>10.4f}")

    fe, se, ke = stats(even)
    fo, so, ko = stats(odd)
    print(f"{'difference (even - odd)':<34}{fe-fo:>+10.4f}{se-so:>+12.4f}{ke-ko:>+10.4f}")

    # If interleaved repeats, even-start pairs share a position and should show
    # markedly higher structure correlation than odd-start pairs.
    verdict = "REPEAT PAIRS (interleaved)" if (se - so) > 0.02 else "sequential positions"
    print(f"  -> {verdict}")
    print()
