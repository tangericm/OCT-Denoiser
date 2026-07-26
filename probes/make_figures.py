"""Qualitative comparison panels from the trained checkpoints.

Display honesty: every panel is z-scored over the SAME tissue ROI and then
mapped through ONE shared window. Per-frame normalisation destroys absolute
scale, so without this a model could look better purely from a brightness or
contrast shift -- the exact failure mode that makes raw SNR untrustworthy.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from octdenoiser.configs.default import FolderSpec
from octdenoiser.experiments.run_fair_eval import SCHEME_INPUTS
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor

M2 = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
STACK = "Line_6mm_2048Aline_135degCW_50frame_gain165"
CKPT_DIR = "runs/supervision_ablation"
REF_CACHE = f"runs/references/{STACK}.npz"
OUT = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Denoiser\runs\figures"

Z = slice(140, 480)          # tissue band for this stack
X = slice(300, 1300)         # lateral crop for a readable aspect ratio
ZOOM_Z = slice(230, 330)
ZOOM_X = slice(700, 1000)
VMIN, VMAX = -1.8, 3.2       # shared display window, in sigma

LABELS = {
    "noisy": "Noisy input (single frame)",
    "A_bandgap_fullband": "A  bandgap -> fullband (current method)",
    "B_complementary": "B  complementary sub-band",
    "C_pair_position": "C  frame pair, position",
    "D_pair_repeat": "D  frame pair, repeat",
    "reference": "Reference (50-frame registered average)",
}

os.makedirs(OUT, exist_ok=True)
proc = BscanProcessor(FolderSpec(root_folder=M2, data_folder=STACK, pixels=2048,
                                 alines=2048, crop_depth=(0, 1024),
                                 window_sigma=0.05, gap=0.60, gap_offset=0.015))
out = proc.process_one(proc.bscan_paths[0], frame_idx=0, fft_workers=-1)
ref = np.load(REF_CACHE)["reference"]


def zs(a):
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() or 1.0)


panels = {"noisy": zs(out["target_full"][Z, X])}

for scheme in ("A_bandgap_fullband", "B_complementary", "C_pair_position", "D_pair_repeat"):
    p = os.path.join(CKPT_DIR, f"{scheme}.pt")
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=True)
    model = create_model("resunet_pseudo3d", base=32, in_ch=ck["in_ch"]).cuda().eval()
    model.load_state_dict(ck["model"], strict=True)
    chans = [out[k] for k in SCHEME_INPUTS[scheme]]
    with torch.no_grad():
        x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).cuda()
        pred = model(x)[0, 0].float().cpu().numpy()
    panels[scheme] = zs(pred[Z, X])
    del model
    torch.cuda.empty_cache()

panels["reference"] = zs(ref[Z, X])

order = ["noisy", "A_bandgap_fullband", "B_complementary",
         "C_pair_position", "D_pair_repeat", "reference"]
order = [k for k in order if k in panels]

# ---- full B-scan comparison -------------------------------------------------
fig, axes = plt.subplots(len(order), 1, figsize=(14, 2.1 * len(order)))
for ax, key in zip(axes, order):
    ax.imshow(panels[key], cmap="gray", vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_ylabel(LABELS[key], fontsize=8, rotation=0, ha="right", va="center")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"OCT denoising comparison — {STACK}\n"
             f"all panels z-scored over the same ROI and shown through one shared window "
             f"[{VMIN}, {VMAX}] sigma", fontsize=10)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
p1 = os.path.join(OUT, "comparison_full.png")
fig.savefig(p1, dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- zoomed detail ----------------------------------------------------------
zz = slice(ZOOM_Z.start - Z.start, ZOOM_Z.stop - Z.start)
zx = slice(ZOOM_X.start - X.start, ZOOM_X.stop - X.start)
n = len(order)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows))
for ax, key in zip(axes.ravel(), order):
    ax.imshow(panels[key][zz, zx], cmap="gray", vmin=VMIN, vmax=VMAX, aspect="auto")
    ax.set_title(LABELS[key], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("Zoomed detail — same shared window", fontsize=11)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
p2 = os.path.join(OUT, "comparison_zoom.png")
fig.savefig(p2, dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- residual maps: what each model REMOVED ---------------------------------
res_keys = [k for k in order if k not in ("noisy", "reference")]
fig, axes = plt.subplots(1, len(res_keys), figsize=(4.4 * len(res_keys), 4.0))
axes = np.atleast_1d(axes)
for ax, key in zip(axes, res_keys):
    src = panels["A_bandgap_fullband" if key == "A_bandgap_fullband" else "noisy"]
    r = panels["noisy"] - panels[key]
    ax.imshow(r[zz, zx], cmap="gray", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.set_title(f"{LABELS[key]}\nremoved (input - output)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Residuals — visible anatomy here means structure was removed, not just noise",
             fontsize=10)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
p3 = os.path.join(OUT, "comparison_residuals.png")
fig.savefig(p3, dpi=140, bbox_inches="tight")
plt.close(fig)

for p in (p1, p2, p3):
    print(p, os.path.getsize(p), "bytes")
