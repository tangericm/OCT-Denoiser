"""Fair comparison of trained supervision schemes.

Why the first ablation could not settle anything
------------------------------------------------
Each scheme consumes a different input -- A takes both sub-bands, B one
sub-band, C and D a full-band frame -- so its metrics compared denoisers AND
denoising problems at once. B's apparent lead was partly that a band-limited
input is an easier target.

This scores every checkpoint against ONE common target instead.

Primary: full reference vs the Maestro2 50-frame averages
---------------------------------------------------------
Registered and averaged repeat stacks are the only near-clean reference this
dataset admits (25-36 effective looks, speckle contrast 0.53 -> 0.09). Crucially
a 50-frame average matches NO scheme's training objective, so it is the least
circular target available -- unlike scoring against a shifted frame, which
structurally resembles what scheme C was trained to predict and would flatter it.

Caveat: Maestro2 is a different instrument from the M3 training data (its own
.CLB, 2048 A-lines, line scans rather than volumes). That is out-of-distribution
for all four models equally -- fair between them, but not an in-domain result.

Comparison domain: predictions are z-scored log images whose absolute scale is
destroyed by per-frame normalisation, so both prediction and reference are
z-scored over the tissue ROI before scoring. The comparison is therefore
scale- and offset-invariant by construction.

Secondary: B's domain shift
---------------------------
B trains on a sub-band but would deploy on the full band. Feeding it both
measures how much that costs -- the risk flagged as ablation 7 in the plan and
never tested. If B collapses on full-band input its lead is not deployable.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import shift as ndshift

from octdenoiser.configs.default import FolderSpec
from octdenoiser.eval.reference import register_stack
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor
from octdenoiser.tools.eval_mirror import psnr, ssim

# The anterior stack yields only 3.0 effective looks of 50 and is not a usable
# reference; the other three give 25-36.
REFERENCE_STACKS = [
    ("Line_6mm_2048Aline_135degCW_50frame_gain165", 2048),
    ("Line_6mm_2048Aline_135degCW_50frame_gain167_widefield_ET", 2048),
    ("Line_6mm_2048Aline_135degCW_50frame_gain167_widefield_YM", 2048),
]

SCHEME_INPUTS = {
    "A_bandgap_fullband": ("input_w1", "input_w2"),
    "B_complementary": ("input_w1",),
    "C_pair_position": ("target_full",),
    "D_pair_repeat": ("target_full",),
}

TISSUE = slice(120, 700)
LATERAL_SIGMA = 3.0


def _spec(root: str, folder: str, alines: int) -> FolderSpec:
    return FolderSpec(root_folder=root, data_folder=folder, pixels=2048, alines=alines,
                      crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015)


def _zscore(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)


def _to_unit(a: np.ndarray) -> np.ndarray:
    """Map to [0,1] by percentile, for SSIM's bounded data range."""
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


@dataclass
class Reference:
    """A registered 50-frame average, plus the shifts that built it.

    The average lives in frame 0's coordinates. Scoring a prediction made from
    frame i against it compares two images that are physically displaced --
    these stacks carry real, non-monotonic motion (mean |dz| of several pixels,
    one stack reaching 22) -- so the metric would charge every model for
    eye movement on top of whatever denoising it did.

    That is not a constant offset that cancels in a ranking. Misalignment
    penalises a SHARP output more than a blurred one, which is the same
    direction PSNR and SSIM are already biased in (see docs/FINDINGS.md section
    9). Keeping the shifts and undoing them per frame removes the term.
    """

    image: np.ndarray          # [H, W] log-domain average, in frame 0's coordinates
    shifts: np.ndarray         # [N, 2] (dz, dx) carrying frame i onto frame 0

    def aligned_to(self, frame_index: int) -> np.ndarray:
        """The reference resampled into frame `frame_index`'s coordinates.

        The REFERENCE is what gets resampled, never the prediction: a 50-frame
        average is already smooth, so cubic interpolation costs it essentially
        nothing, whereas resampling the prediction would blur exactly the fine
        structure the comparison is meant to measure.
        """
        if frame_index >= len(self.shifts):
            return self.image
        dz, dx = self.shifts[frame_index]
        if abs(dz) < 1e-3 and abs(dx) < 1e-3:
            return self.image
        return ndshift(self.image, (-float(dz), -float(dx)), order=3, mode="nearest")


# Bumped when the cache payload changed (shifts added). Old caches lack the
# shifts and would silently fall back to the unregistered comparison.
_CACHE_VERSION = "v2"


def build_reference(root: str, folder: str, alines: int, cache_dir: str) -> tuple[Reference, dict]:
    """Register and average one repeat stack. Cached -- this is the slow part."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{folder}_{_CACHE_VERSION}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return (Reference(z["reference"], z["shifts"]),
                {"cached": True, "n_kept": int(z["n_kept"])})

    proc = BscanProcessor(_spec(root, folder, alines))
    n = len(proc.bscan_paths)
    log_stack = np.empty((n, 1024, alines), dtype=np.float32)
    for i, p in enumerate(proc.bscan_paths):
        log_stack[i] = proc.process_one(p, frame_idx=i, fft_workers=-1)["target_full"]

    _, res = register_stack(log_stack, reference_index=0, roi="auto",
                            max_shift=64, reject_worsening=True)
    del log_stack

    acc = np.zeros((1024, alines), dtype=np.float64)
    for i, p in enumerate(proc.bscan_paths):
        lin = proc.process_one(p, frame_idx=i, need_linear_full=True,
                               fft_workers=-1)["target_full_linear"]
        dz, dx = res.shifts[i]
        acc += ndshift(lin, (dz, dx), order=3, mode="nearest")
    ref_linear = acc / n

    # Reference lives in the same log domain the models output.
    ref_log = np.log10(ref_linear + 1e-6).astype(np.float32)
    shifts = np.asarray(res.shifts, dtype=np.float32)
    np.savez_compressed(path, reference=ref_log, shifts=shifts, n_kept=res.n_kept)
    return (Reference(ref_log, shifts),
            {"cached": False, "n_kept": res.n_kept,
             "corr_after": float(np.mean(res.correlations))})


@torch.no_grad()
def score_scheme(model, scheme: str, root: str, device: str,
                 references: dict, n_frames: int, full_band_override: bool = False) -> dict:
    """Score one checkpoint against the near-clean references."""
    keys = SCHEME_INPUTS[scheme]
    if full_band_override:
        keys = ("target_full",) * len(keys)

    acc: dict[str, list[float]] = {"psnr": [], "ssim": [], "residual_leak": [], "speckle_ratio": []}
    for folder, alines in REFERENCE_STACKS:
        if folder not in references:
            continue
        reference = references[folder]
        proc = BscanProcessor(_spec(root, folder, alines))
        step = max(1, len(proc.bscan_paths) // n_frames)

        for i in range(0, min(len(proc.bscan_paths), n_frames * step), step):
            # Undo frame i's motion so the comparison measures denoising rather
            # than displacement. Shift the full frame, then crop, so rows moving
            # into the ROI carry real data.
            ref_z = _zscore(reference.aligned_to(i)[TISSUE])
            out = proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)
            chans = [out[k] for k in keys]
            x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).to(device)
            pred = model(x)[0, 0].float().cpu().numpy()[TISSUE]
            src = chans[0][TISSUE]

            pred_z = _zscore(pred)
            acc["psnr"].append(psnr(_to_unit(pred_z), _to_unit(ref_z), 1.0))
            acc["ssim"].append(ssim(_to_unit(pred_z), _to_unit(ref_z), 1.0))

            sm = gaussian_filter1d(src.astype(np.float64), LATERAL_SIGMA, axis=1, mode="nearest")
            resid = src.astype(np.float64) - pred.astype(np.float64)
            acc["residual_leak"].append(abs(_corr(resid, sm)))
            sp_in = src.astype(np.float64) - sm
            sp_out = pred.astype(np.float64) - gaussian_filter1d(
                pred.astype(np.float64), LATERAL_SIGMA, axis=1, mode="nearest")
            acc["speckle_ratio"].append(float(sp_out.var() / max(sp_in.var(), 1e-12)))

    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


@torch.no_grad()
def noisy_input_baseline(root: str, references: dict, n_frames: int) -> dict:
    """The do-nothing floor: a single raw frame scored against the reference.

    Any model that does not beat this is not denoising.
    """
    acc: dict[str, list[float]] = {"psnr": [], "ssim": []}
    for folder, alines in REFERENCE_STACKS:
        if folder not in references:
            continue
        reference = references[folder]
        proc = BscanProcessor(_spec(root, folder, alines))
        step = max(1, len(proc.bscan_paths) // n_frames)
        for i in range(0, min(len(proc.bscan_paths), n_frames * step), step):
            # Same alignment as score_scheme -- the floor has to be measured the
            # same way as what it is a floor for.
            ref_z = _zscore(reference.aligned_to(i)[TISSUE])
            src = proc.process_one(proc.bscan_paths[i], frame_idx=i,
                                   fft_workers=-1)["target_full"][TISSUE]
            src_z = _zscore(src)
            acc["psnr"].append(psnr(_to_unit(src_z), _to_unit(ref_z), 1.0))
            acc["ssim"].append(ssim(_to_unit(src_z), _to_unit(ref_z), 1.0))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m2-root", required=True, help="Maestro2 directory")
    p.add_argument("--ckpt-dir", default=os.path.join("runs", "supervision_ablation"))
    p.add_argument("--cache-dir", default=os.path.join("runs", "references"))
    p.add_argument("--out", default=os.path.join("runs", "fair_eval"))
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("building near-clean references (cached after the first run)", flush=True)
    references: dict[str, Reference] = {}
    for folder, alines in REFERENCE_STACKS:
        try:
            ref, info = build_reference(args.m2_root, folder, alines, args.cache_dir)
            references[folder] = ref
            print(f"  {folder[:52]:<54} {'cached' if info['cached'] else 'built'}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {folder[:52]:<54} FAILED {type(e).__name__}: {e}", flush=True)
    if not references:
        raise SystemExit("no references available")

    floor = noisy_input_baseline(args.m2_root, references, args.n_frames)
    print(f"\nnoisy-input floor: PSNR={floor['psnr']:.3f}  SSIM={floor['ssim']:.4f}\n", flush=True)

    results: dict[str, dict] = {"_noisy_floor": floor}
    for scheme in SCHEME_INPUTS:
        ck_path = os.path.join(args.ckpt_dir, f"{scheme}.pt")
        if not os.path.exists(ck_path):
            print(f"  {scheme:<22} SKIP (no checkpoint)", flush=True)
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        model = create_model("resunet_pseudo3d", base=args.base, in_ch=ck["in_ch"]).to(args.device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()

        results[scheme] = score_scheme(model, scheme, args.m2_root, args.device,
                                       references, args.n_frames)
        print(f"  {scheme:<22} PSNR={results[scheme]['psnr']:.3f}  "
              f"SSIM={results[scheme]['ssim']:.4f}", flush=True)

        # B trains on a sub-band but would deploy on the full band.
        if scheme == "B_complementary":
            shifted = score_scheme(model, scheme, args.m2_root, args.device,
                                   references, args.n_frames, full_band_override=True)
            results["B_complementary_fullband_input"] = shifted
            print(f"  {'B on FULL-BAND input':<22} PSNR={shifted['psnr']:.3f}  "
                  f"SSIM={shifted['ssim']:.4f}   <- domain shift", flush=True)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    with open(os.path.join(args.out, "fair_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 86)
    print(f"{'scheme':<34}{'PSNR':>9}{'SSIM':>9}{'speckle':>10}{'leak':>9}{'vs floor':>11}")
    print("=" * 86)
    print(f"{'noisy input (do nothing)':<34}{floor['psnr']:>9.3f}{floor['ssim']:>9.4f}"
          f"{'1.0000':>10}{'--':>9}{'--':>11}")
    for name, m in results.items():
        if name == "_noisy_floor":
            continue
        print(f"{name:<34}{m['psnr']:>9.3f}{m['ssim']:>9.4f}{m['speckle_ratio']:>10.4f}"
              f"{m['residual_leak']:>9.4f}{m['psnr'] - floor['psnr']:>+11.3f}")
    print("\nPrimary metric is PSNR/SSIM against the 50-frame averages, which match no")
    print("scheme's training objective. A model failing to beat the noisy floor is not")
    print("denoising. Maestro2 is a different instrument from the M3 training data, so")
    print("this is out-of-distribution for every model equally.")


if __name__ == "__main__":
    main()
