"""Does a network beat simply registering and averaging the same K frames?

If a model consuming K frames cannot beat a plain K-frame average, it is not
earning its complexity. Our own data already shows averaging is strong: 50
frames took speckle contrast from 0.53 to 0.09.

CIRCULARITY GUARD: the reference here is built from frames 25-49 ONLY, and every
input is drawn from frames 0-24. Averaging frames that also went into the
reference would trivially correlate with it.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import shift as ndshift

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import register_stack
from octdenoiser.experiments.run_fair_eval import SCHEME_INPUTS, _to_unit, _zscore
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor
from octdenoiser.tools.eval_mirror import psnr, ssim

M2 = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Data\Maestro2"
STACKS = [
    "Line_6mm_2048Aline_135degCW_50frame_gain165",
    "Line_6mm_2048Aline_135degCW_50frame_gain167_widefield_ET",
    "Line_6mm_2048Aline_135degCW_50frame_gain167_widefield_YM",
]
Z = slice(140, 480)
KS = (1, 2, 4, 8, 16)


def fine(a):
    a = a.astype(np.float64)
    return a - gaussian_filter1d(a, 3.0, axis=1, mode="nearest")


def corr(a, b):
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


rows: dict[str, dict[str, list]] = {}


def add(name, key, val):
    rows.setdefault(name, {}).setdefault(key, []).append(val)


for stack in STACKS:
    proc = BscanProcessor(FolderSpec(root_folder=M2, data_folder=stack, pixels=2048,
                                     alines=2048, crop_depth=(0, 1024),
                                     window_sigma=0.05, gap=0.60, gap_offset=0.015))
    n = len(proc.bscan_paths)
    logs, lins = [], []
    for i in range(n):
        o = proc.process_one(proc.bscan_paths[i], frame_idx=i,
                             need_linear_full=True, fft_workers=-1)
        logs.append(o["target_full"])
        lins.append(o["target_full_linear"])
    logs = np.stack(logs)
    lins = np.stack(lins)

    # Reference from the SECOND half only.
    ref_idx = list(range(25, n))
    _, r_ref = register_stack(logs[ref_idx], reference_index=0, roi="auto",
                              max_shift=64, reject_worsening=True)
    acc = np.zeros_like(lins[0], dtype=np.float64)
    for j, gi in enumerate(ref_idx):
        dz, dx = r_ref.shifts[j]
        acc += ndshift(lins[gi], (dz, dx), order=3, mode="nearest")
    reference = np.log10(acc / len(ref_idx) + 1e-6)
    ref_z = _zscore(reference[Z])
    ref_fine = fine(reference[Z])

    # Inputs from the FIRST half only.
    in_idx = list(range(0, 25))
    _, r_in = register_stack(logs[in_idx], reference_index=0, roi="auto",
                             max_shift=64, reject_worsening=True)
    reg_lin = np.stack([ndshift(lins[gi], tuple(r_in.shifts[j]), order=3, mode="nearest")
                        for j, gi in enumerate(in_idx)])

    for k in KS:
        avg = np.log10(reg_lin[:k].mean(axis=0) + 1e-6)
        a_z = _zscore(avg[Z])
        name = f"average {k:2d} frame(s)"
        add(name, "psnr", psnr(_to_unit(a_z), _to_unit(ref_z), 1.0))
        add(name, "ssim", ssim(_to_unit(a_z), _to_unit(ref_z), 1.0))
        add(name, "fine_vs_ref", corr(fine(avg[Z]), ref_fine))

    # Networks, single frame 0 (from the input half).
    o0 = proc.process_one(proc.bscan_paths[0], frame_idx=0, fft_workers=-1)
    for scheme in ("A_bandgap_fullband", "B_complementary",
                   "C_pair_position", "D_pair_repeat"):
        p = f"runs/supervision_ablation/{scheme}.pt"
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            continue
        model = create_model("resunet_pseudo3d", base=32, in_ch=ck["in_ch"]).cuda().eval()
        model.load_state_dict(ck["model"], strict=True)
        chans = [o0[k] for k in SCHEME_INPUTS[scheme]]
        with torch.no_grad():
            x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).cuda()
            pred = model(x)[0, 0].float().cpu().numpy()
        p_z = _zscore(pred[Z])
        add(f"net {scheme}", "psnr", psnr(_to_unit(p_z), _to_unit(ref_z), 1.0))
        add(f"net {scheme}", "ssim", ssim(_to_unit(p_z), _to_unit(ref_z), 1.0))
        add(f"net {scheme}", "fine_vs_ref", corr(fine(pred[Z]), ref_fine))
        del model
        torch.cuda.empty_cache()

print("=" * 78)
print("AVERAGING BASELINE vs NETWORKS")
print("  reference = frames 25-49 registered+averaged")
print("  inputs    = frames 0-24 only (disjoint, so no circularity)")
print("=" * 78)
print(f"{'method':<30}{'PSNR':>10}{'SSIM':>10}{'fine vs ref':>14}")
print("-" * 78)
for name in list(rows):
    r = rows[name]
    print(f"{name:<30}{np.mean(r['psnr']):>10.3f}{np.mean(r['ssim']):>10.4f}"
          f"{np.mean(r['fine_vs_ref']):>14.4f}")
