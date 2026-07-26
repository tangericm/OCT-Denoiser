"""Where does the tissue actually sit in each Maestro2 stack?"""
import os

import numpy as np

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import correlation
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"

for folder in sorted(d for d in os.listdir(ROOT) if "50frame" in d):
    spec = FolderSpec(root_folder=ROOT, data_folder=folder, pixels=2048, alines=2048,
                      crop_depth=(0, 1024), window_sigma=0.05, gap=0.60)
    proc = BscanProcessor(spec)
    o0 = proc.process_one(proc.bscan_paths[0], frame_idx=0, need_linear_full=True, fft_workers=-1)
    o1 = proc.process_one(proc.bscan_paths[1], frame_idx=1, need_linear_full=True, fft_workers=-1)

    prof = o0["target_full_linear"].mean(axis=1)
    # Energy-weighted extent of the structure, ignoring the DC roll-off.
    p = prof.copy()
    p[:30] = 0.0
    thr = p.max() * 0.10
    rows = np.flatnonzero(p > thr)

    print(f"{folder}")
    print(f"  peak row={int(np.argmax(p))}  >10% band = [{rows.min()}, {rows.max()}]")
    deciles = [int(np.percentile(rows, q)) for q in (5, 25, 50, 75, 95)]
    print(f"  row percentiles of energy: {deciles}")
    print(f"  full-frame corr(f0,f1) linear = {correlation(o0['target_full_linear'], o1['target_full_linear']):+.4f}")
    for lo, hi in ((110, 600), (30, 1024), (rows.min(), rows.max() + 1)):
        a = o0["target_full"][lo:hi]
        b = o1["target_full"][lo:hi]
        print(f"    log-domain corr over rows [{lo},{hi}) = {correlation(a, b):+.4f}")
    print()
