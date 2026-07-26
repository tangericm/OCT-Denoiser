"""Phase 1 deliverable: register + average the Maestro2 50-frame stacks.

Gate: post-registration correlation must beat the 0.975 unregistered baseline,
and the frame-4 motion outlier (0.700) must be recovered or excluded.

Memory: 50 frames of 1024x2048 float32 is 400 MB per stack. Only the log stack
is held; linear frames are re-derived and accumulated one at a time.
"""
import os
import sys

import numpy as np
from scipy.ndimage import shift as ndshift

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import correlation, register_stack, speckle_contrast
from octdenoiser.preprocess import BscanProcessor

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
ROI = (110, 600)             # tissue band; MATLAB prior work used 130:600
BG = (700, 900, 200, 1800)   # deep background box for speckle contrast
PROBE = (1, 2, 4, 8)


def log(msg=""):
    print(msg, flush=True)


folders = sorted(
    d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and "50frame" in d
)
log(f"Maestro2 stacks: {len(folders)}\n")

summary = []
for folder in folders:
    spec = FolderSpec(
        root_folder=ROOT, data_folder=folder,
        pixels=2048, alines=2048, crop_depth=(0, 1024),
        window_sigma=0.05, gap=0.60,
    )
    proc = BscanProcessor(spec)
    n = len(proc.bscan_paths)
    log("=" * 76)
    log(f"{folder}  ({n} frames)   CLB={os.path.basename(proc.clb_path)}")

    # Pass 1: log stack for registration (400 MB), plus frame 0 linear + probes.
    log_stack = np.empty((n, 1024, 2048), dtype=np.float32)
    lin0 = None
    probe_before = {}
    for i, p in enumerate(proc.bscan_paths):
        out = proc.process_one(p, frame_idx=i, need_linear_full=True, fft_workers=-1)
        log_stack[i] = out["target_full"]
        if i == 0:
            lin0 = out["target_full_linear"].copy()
            sc_single = speckle_contrast(lin0, BG)
        elif i in PROBE or i == n - 1:
            probe_before[i] = correlation(lin0, out["target_full_linear"])
    log("  linear corr vs frame 0 BEFORE: "
        + "  ".join(f"f{j}={c:+.4f}" for j, c in sorted(probe_before.items())))

    registered, res = register_stack(
        log_stack, reference_index=0, roi="auto", max_shift=64, min_correlation=None
    )
    del registered, log_stack
    log("  " + res.summary().replace("\n", "\n  "))

    # Pass 2: re-derive linear frames, apply the shifts, accumulate.
    acc = np.zeros((1024, 2048), dtype=np.float64)
    naive = np.zeros((1024, 2048), dtype=np.float64)
    lin0_reg = None
    probe_after = {}
    for i, p in enumerate(proc.bscan_paths):
        out = proc.process_one(p, frame_idx=i, need_linear_full=True, fft_workers=-1)
        lin = out["target_full_linear"]
        naive += lin
        dz, dx = res.shifts[i]
        reg = ndshift(lin, (dz, dx), order=3, mode="nearest")
        if res.kept[i]:
            acc += reg
        if i == 0:
            lin0_reg = reg.copy()
        elif i in PROBE or i == n - 1:
            probe_after[i] = correlation(lin0_reg, reg)
    log("  linear corr vs frame 0 AFTER : "
        + "  ".join(f"f{j}={c:+.4f}" for j, c in sorted(probe_after.items())))

    ref_img = acc / res.n_kept
    naive_img = naive / n
    sc_naive = speckle_contrast(naive_img, BG)
    sc_reg = speckle_contrast(ref_img, BG)
    looks = (sc_single / max(sc_reg, 1e-9)) ** 2

    log("  speckle contrast (deep background):")
    log(f"    single frame       {sc_single:.4f}")
    log(f"    naive average      {sc_naive:.4f}")
    log(f"    registered average {sc_reg:.4f}")
    log(f"    ideal 1/sqrt({res.n_kept})     {sc_single / np.sqrt(res.n_kept):.4f}")
    log(f"    effective independent looks: {looks:.1f} of {res.n_kept}")

    summary.append(dict(
        folder=folder, kept=res.n_kept,
        before=float(np.mean(res.correlations_before[res.kept])),
        after=float(np.mean(res.correlations[res.kept])),
        probe_before=probe_before, probe_after=probe_after,
        sc_single=sc_single, sc_naive=sc_naive, sc_reg=sc_reg, looks=looks,
    ))
    log()

log("=" * 76)
log("SUMMARY")
log("=" * 76)
log(f"{'stack':<50}{'kept':>6}{'corr b':>9}{'corr a':>9}{'looks':>8}")
for s in summary:
    log(f"{s['folder'][:48]:<50}{s['kept']:>6}{s['before']:>9.4f}{s['after']:>9.4f}{s['looks']:>8.1f}")

gate = all(s["after"] > 0.975 for s in summary)
log(f"\nGATE post-registration correlation > 0.975 : {gate}")
log(f"GATE every stack kept >= 2 frames           : {all(s['kept'] >= 2 for s in summary)}")
sys.exit(0 if gate else 1)
