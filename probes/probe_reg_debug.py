"""Why does registration make the anterior stack worse?"""
import numpy as np
from scipy.ndimage import shift as ndshift

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import auto_roi, correlation, phase_correlation_shift
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"

for folder in ("Line_3mm_2048Aline_135degCW_50frame_gain167_widefield_anterior_YM",
               "Line_6mm_2048Aline_135degCW_50frame_gain165"):
    cfg = FolderSpec(root_folder=ROOT, data_folder=folder, pixels=2048, alines=2048,
                     crop_depth=(0, 1024), window_sigma=0.05, gap=0.60)
    proc = BscanProcessor(cfg)
    fr = [proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)["target_full"]
          for i in range(6)]

    roi = auto_roi(fr[0].astype(np.float64))
    print("=" * 78)
    print(f"{folder[:70]}")
    print(f"  auto ROI = {roi}   (band height {roi[1]-roi[0]})")

    ref = fr[0].astype(np.float64)[roi[0]:roi[1]]
    print(f"{'frame':>6}{'dz':>9}{'dx':>9}{'corr before':>13}{'corr after':>12}")
    for i in range(1, 6):
        mov_full = fr[i].astype(np.float64)
        mov = mov_full[roi[0]:roi[1]]
        before = correlation(ref, mov)
        for ms in (None, 64):
            dz, dx = phase_correlation_shift(ref, mov, max_shift=ms)
            out = ndshift(mov_full, (dz, dx), order=3, mode="nearest")[roi[0]:roi[1]]
            after = correlation(ref, out)
            tag = "unbounded" if ms is None else f"max{ms}"
            print(f"{i:>6}{dz:>9.2f}{dx:>9.2f}{before:>13.4f}{after:>12.4f}   [{tag}]")

    # Is a pure x-shift or pure z-shift better than the joint estimate?
    print("  --- brute-force best single-axis shift, frame 1 ---")
    mov_full = fr[1].astype(np.float64)
    best = (None, -2.0)
    for dz in range(-40, 41, 4):
        c = correlation(ref, ndshift(mov_full, (dz, 0), order=1, mode="nearest")[roi[0]:roi[1]])
        if c > best[1]:
            best = (("dz", dz), c)
    for dx in range(-200, 201, 10):
        c = correlation(ref, ndshift(mov_full, (0, dx), order=1, mode="nearest")[roi[0]:roi[1]])
        if c > best[1]:
            best = (("dx", dx), c)
    print(f"    best single-axis: {best[0]} -> corr {best[1]:.4f}")
    print()
