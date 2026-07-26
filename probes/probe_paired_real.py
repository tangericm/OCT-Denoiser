"""Smoke the paired dataset on a real OCTA volume.

Validation: the pairs it yields should reproduce the correlations measured
directly -- speckle ~0.017 for position pairing, ~0.105 for repeat pairing.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.data.paired_dataset import PairedFrameDataset

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"
spec = FolderSpec(root_folder=ROOT, data_folder="M3_Macula_6x6mm_512x512x2",
                  pixels=2048, alines=512, crop_depth=(0, 1024),
                  window_sigma=0.05, gap=0.60, gap_offset=0.015)


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def split_parts(img):
    st = gaussian_filter1d(img.astype(np.float64), 3.0, axis=1, mode="nearest")
    return st, img.astype(np.float64) - st


for mode, step in (("position", 1), ("position", 2), ("repeat", 1)):
    ds = PairedFrameDataset(
        folder_specs=[spec], split="train", train_frac=0.9,
        pair_mode=mode, position_step=step,
        patch_h=256, patch_w=256, patches_per_frame=1, patch_mode="patch",
        augment=False, seed=0, cache_frames_per_worker=16,
    )
    ds._build_index()
    label = f"{mode} step={step}" if mode == "position" else "repeat"
    print(f"{label:<22} offset={ds.frame_offset}  pairs={len(ds)}")

    scs, sts = [], []
    for i in range(0, 24):
        x, y, _ = ds[i]
        a, b = x.numpy()[0], y.numpy()[0]
        sa, ka = split_parts(a)
        sb, kb = split_parts(b)
        sts.append(corr(sa, sb))
        scs.append(corr(ka, kb))
    print(f"{'':<22} structure={np.mean(sts):+.4f}   speckle={np.mean(scs):+.4f}")
    print()

print("Directly measured earlier on the same volume:")
print("  repeats   (2k, 2k+1)   speckle +0.105   structure 0.963")
print("  positions (i,  i+2)    speckle +0.017   structure 0.958")
