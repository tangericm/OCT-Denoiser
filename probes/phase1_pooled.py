"""Pooled noise calibration across Maestro3 acquisitions (shared detector)."""
import glob
import os

import numpy as np

from octdenoiser.physics.noise_model import (
    fit_noise_model,
    fit_pooled_noise_model,
    load_background_frames,
)

ROOT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro3"

curves, nframes, indep = {}, {}, {}
for folder in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, folder)
    backs = sorted(glob.glob(os.path.join(d, "back*.raw")))
    if not backs:
        continue
    alines = os.path.getsize(backs[0]) // (2048 * 2)
    frames = load_background_frames(d, 2048, int(alines))
    model, curve = fit_noise_model(frames, source=folder)
    curves[folder] = curve
    nframes[folder] = frames.shape[0]
    indep[folder] = model

pooled = fit_pooled_noise_model(curves, n_frames=nframes)

g = next(iter(pooled.values())).gain
r = next(iter(pooled.values())).rin
print("=" * 78)
print("POOLED FIT — shared gain/rin across Maestro3, per-acquisition read_var")
print("=" * 78)
print(f"  shared gain = {g:.5f} ADU/e-")
print(f"  shared rin  = {r:.4e}")
print()
print(f"{'acquisition':<34}{'read_var':>10}{'indep gain':>12}{'R2 pooled':>11}{'R2 indep':>10}")
for name, m in pooled.items():
    print(f"{name:<34}{m.read_var:>10.2f}{indep[name].gain:>12.4f}{m.r_squared:>11.5f}{indep[name].r_squared:>10.5f}")

reads = np.array([m.read_var for m in pooled.values()])
r2p = np.array([m.r_squared for m in pooled.values()])
r2i = np.array([m.r_squared for m in indep.values()])
gi = np.array([m.gain for m in indep.values()])
print()
print(f"  read_var spread (pooled) : {reads.min():.2f} .. {reads.max():.2f}")
print(f"  gain spread (independent): {gi.min():.4f} .. {gi.max():.4f}   <- degenerate")
print(f"  mean R2  pooled={r2p.mean():.5f}   independent={r2i.mean():.5f}")
print(f"  all physical (pooled)    : {all(m.is_physical() for m in pooled.values())}")
print(f"  all physical (independent): {all(m.is_physical() for m in indep.values())}")
print()
print("  Read noise sigma = sqrt(read_var):")
for name, m in pooled.items():
    print(f"    {name:<34}{m.read_noise_std:>7.2f} ADU")
