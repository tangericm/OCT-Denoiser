"""Is D's apparent sharpness real detail, or retained speckle?

Both look like texture and mean opposite things. The discriminator: take each
output's FINE-SCALE component and ask what it correlates with.

  corr(output_fine, reference_fine)  -> real structure the reference also has
  corr(output_fine, input_speckle)   -> speckle carried through from the input

A model whose fine detail tracks the reference is genuinely sharp. One whose
fine detail tracks the input's speckle only looks sharp.
"""
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec
from octdenoiser.experiments.run_fair_eval import SCHEME_INPUTS
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor

M2 = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
STACK = "Line_6mm_2048Aline_135degCW_50frame_gain165"
CKPT_DIR = "runs/supervision_ablation"
Z, SIG = slice(140, 480), 3.0

proc = BscanProcessor(FolderSpec(root_folder=M2, data_folder=STACK, pixels=2048,
                                 alines=2048, crop_depth=(0, 1024),
                                 window_sigma=0.05, gap=0.60, gap_offset=0.015))
ref_full = np.load(f"runs/references/{STACK}.npz")["reference"]


def fine(a):
    a = a.astype(np.float64)
    return a - gaussian_filter1d(a, SIG, axis=1, mode="nearest")


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


FRAMES = [0, 12, 24, 36]
acc: dict[str, dict[str, list]] = {}

for scheme in ("A_bandgap_fullband", "B_complementary", "C_pair_position", "D_pair_repeat"):
    p = os.path.join(CKPT_DIR, f"{scheme}.pt")
    if not os.path.exists(p):
        continue
    ck = torch.load(p, map_location="cpu", weights_only=True)
    model = create_model("resunet_pseudo3d", base=32, in_ch=ck["in_ch"]).cuda().eval()
    model.load_state_dict(ck["model"], strict=True)
    a = acc.setdefault(scheme, {"vs_ref": [], "vs_speckle": [], "fine_energy": []})

    for fi in FRAMES:
        out = proc.process_one(proc.bscan_paths[fi], frame_idx=fi, fft_workers=-1)
        chans = [out[k] for k in SCHEME_INPUTS[scheme]]
        with torch.no_grad():
            x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).cuda()
            pred = model(x)[0, 0].float().cpu().numpy()

        f_out = fine(pred[Z])
        f_ref = fine(ref_full[Z])
        f_in = fine(chans[0][Z])
        a["vs_ref"].append(corr(f_out, f_ref))
        a["vs_speckle"].append(corr(f_out, f_in))
        a["fine_energy"].append(float(f_out.var() / max(f_in.var(), 1e-12)))
    del model
    torch.cuda.empty_cache()

# The noisy input itself, as the reference point for "all speckle".
raw = {"vs_ref": [], "vs_speckle": []}
for fi in FRAMES:
    out = proc.process_one(proc.bscan_paths[fi], frame_idx=fi, fft_workers=-1)
    f_in = fine(out["target_full"][Z])
    raw["vs_ref"].append(corr(f_in, fine(ref_full[Z])))
    raw["vs_speckle"].append(1.0)

print("=" * 80)
print("IS THE FINE DETAIL REAL? correlation of each output's fine-scale component")
print("=" * 80)
print(f"{'scheme':<24}{'vs REFERENCE':>15}{'vs INPUT speckle':>19}{'fine energy':>14}")
print("-" * 80)
print(f"{'noisy input':<24}{np.mean(raw['vs_ref']):>15.4f}{1.0:>19.4f}{1.0:>14.4f}")
for scheme, a in acc.items():
    print(f"{scheme:<24}{np.mean(a['vs_ref']):>15.4f}"
          f"{np.mean(a['vs_speckle']):>19.4f}{np.mean(a['fine_energy']):>14.4f}")

print()
print("Reference-tracking detail is real structure. Input-tracking detail is")
print("speckle carried through. The noisy input row shows how much of its own")
print("fine detail the 50-frame average actually agrees with.")
