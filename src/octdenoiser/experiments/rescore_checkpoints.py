"""Re-score saved checkpoints against the ALIGNED reference, and show the bias.

Why this exists
---------------
`build_reference` used to compute registration shifts, average with them, and
then discard them, so a prediction made from frame i was scored against an
average sitting in frame 0's coordinates. Every reference-scored number
produced before that fix carries the error (docs/FINDINGS.md section 9).

The runs that produced those numbers save `.pt` files, so correcting them is an
evaluation pass rather than a retrain -- that is the entire point of this
script.

What it reports
---------------
Both scorings, side by side. The delta is the quantity of interest, not just
the corrected column: misalignment penalises a SHARP output more than a blurred
one, so the bias is not a constant that cancels in a ranking. If re-scoring
moves every model by the same amount the old ordering survives; if it moves
them by different amounts, the old ordering was partly an artefact. Reporting
only the corrected numbers would hide which of those happened.

Also writes qualitative panels, because PSNR and SSIM both reward blur and the
residual-structure work in this project came from looking at images.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from octdenoiser.configs.default import FolderSpec
from octdenoiser.experiments.run_fair_eval import (
    REFERENCE_STACKS,
    TISSUE,
    Reference,
    _to_unit,
    _zscore,
    build_reference,
)
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor
from octdenoiser.tools.eval_mirror import psnr, ssim
from octdenoiser.utils.helpers import save_json

# deform_fusion is the only multi-frame entry; it consumes K frames while every
# other architecture consumes one, so its row is not a like-for-like comparison.
MULTI_FRAME = {"deform_fusion"}
POSITION_STRIDE = 2          # repeats_per_position * position_step


def _spec(root: str, folder: str, alines: int) -> FolderSpec:
    return FolderSpec(root_folder=root, data_folder=folder, pixels=2048, alines=alines,
                      crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015)


def _input_for(proc: BscanProcessor, i: int, n_frames: int, multi: bool) -> np.ndarray:
    """The tensor the model was trained to consume, centred on frame i."""
    if not multi:
        one = proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)
        return np.stack([one["target_full"]])
    last = len(proc.bscan_paths) - 1
    idx = [min(max(i + (j - n_frames // 2) * POSITION_STRIDE, 0), last)
           for j in range(n_frames)]
    return np.stack([proc.process_one(proc.bscan_paths[j], frame_idx=j,
                                      fft_workers=-1)["target_full"] for j in idx])


@torch.no_grad()
def score_both_ways(model, multi: bool, args, references: dict[str, Reference]) -> dict:
    """Score every eval frame against the aligned AND the unaligned reference."""
    model.eval()
    acc: dict[str, list[float]] = {
        "psnr_aligned": [], "ssim_aligned": [],
        "psnr_unaligned": [], "ssim_unaligned": [],
        "shift_mag": [],
    }
    for folder, alines in REFERENCE_STACKS:
        if folder not in references:
            continue
        reference = references[folder]
        proc = BscanProcessor(_spec(args.m2_root, folder, alines))
        step = max(1, len(proc.bscan_paths) // args.eval_frames)
        for i in range(0, min(len(proc.bscan_paths), args.eval_frames * step), step):
            chans = _input_for(proc, i, args.n_frames, multi)
            x = torch.from_numpy(chans[None].astype(np.float32)).to(args.device)
            p_z = _to_unit(_zscore(model(x)[0, 0].float().cpu().numpy()[TISSUE]))

            aligned = _to_unit(_zscore(reference.aligned_to(i)[TISSUE]))
            unaligned = _to_unit(_zscore(reference.image[TISSUE]))

            acc["psnr_aligned"].append(psnr(p_z, aligned, 1.0))
            acc["ssim_aligned"].append(ssim(p_z, aligned, 1.0))
            acc["psnr_unaligned"].append(psnr(p_z, unaligned, 1.0))
            acc["ssim_unaligned"].append(ssim(p_z, unaligned, 1.0))
            dz, dx = (reference.shifts[i] if i < len(reference.shifts) else (0.0, 0.0))
            acc["shift_mag"].append(float(np.hypot(dz, dx)))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["d_psnr"] = out["psnr_aligned"] - out["psnr_unaligned"]
    out["d_ssim"] = out["ssim_aligned"] - out["ssim_unaligned"]
    return out


@torch.no_grad()
def noisy_floor_both_ways(args, references: dict[str, Reference]) -> dict:
    """The do-nothing baseline, scored the same two ways."""
    acc: dict[str, list[float]] = {"psnr_aligned": [], "psnr_unaligned": [],
                                   "ssim_aligned": [], "ssim_unaligned": []}
    for folder, alines in REFERENCE_STACKS:
        if folder not in references:
            continue
        reference = references[folder]
        proc = BscanProcessor(_spec(args.m2_root, folder, alines))
        step = max(1, len(proc.bscan_paths) // args.eval_frames)
        for i in range(0, min(len(proc.bscan_paths), args.eval_frames * step), step):
            raw = proc.process_one(proc.bscan_paths[i], frame_idx=i,
                                   fft_workers=-1)["target_full"]
            n_z = _to_unit(_zscore(raw[TISSUE]))
            aligned = _to_unit(_zscore(reference.aligned_to(i)[TISSUE]))
            unaligned = _to_unit(_zscore(reference.image[TISSUE]))
            acc["psnr_aligned"].append(psnr(n_z, aligned, 1.0))
            acc["ssim_aligned"].append(ssim(n_z, aligned, 1.0))
            acc["psnr_unaligned"].append(psnr(n_z, unaligned, 1.0))
            acc["ssim_unaligned"].append(ssim(n_z, unaligned, 1.0))
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def qualitative_panels(models: dict, args, references: dict[str, Reference],
                       folder: str, alines: int, frame: int) -> None:
    """One row per architecture: prediction plus a zoom, against input and reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = references[folder]
    proc = BscanProcessor(_spec(args.m2_root, folder, alines))
    frame = min(frame, len(proc.bscan_paths) - 1)

    raw = proc.process_one(proc.bscan_paths[frame], frame_idx=frame,
                           fft_workers=-1)["target_full"][TISSUE]
    ref = reference.aligned_to(frame)[TISSUE]

    x0, x1 = args.zoom_x, args.zoom_x + args.zoom_w
    z0, z1 = args.zoom_z, args.zoom_z + args.zoom_h

    tiles: list[tuple[str, np.ndarray]] = [("noisy input", raw)]
    for name, (model, multi) in models.items():
        chans = _input_for(proc, frame, args.n_frames, multi)
        xt = torch.from_numpy(chans[None].astype(np.float32)).to(args.device)
        tiles.append((name, model(xt)[0, 0].float().cpu().numpy()[TISSUE]))
    tiles.append(("50-frame average\n(reference)", ref))

    n = len(tiles)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6.4), constrained_layout=True)
    for col, (name, img) in enumerate(tiles):
        u = _to_unit(_zscore(img))
        axes[0, col].imshow(u, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[0, col].set_title(name, fontsize=9)
        axes[1, col].imshow(u[z0:z1, x0:x1], cmap="gray", vmin=0, vmax=1, aspect="auto")
        for r in (0, 1):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
    axes[0, 0].set_ylabel("full B-scan", fontsize=9)
    axes[1, 0].set_ylabel(f"zoom {args.zoom_w}x{args.zoom_h}", fontsize=9)
    fig.suptitle(f"{folder[:52]} — frame {frame}, scored against the aligned reference",
                 fontsize=10)
    out = os.path.join(args.out, f"compare_frame{frame}.png")
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def load_checkpoints(ckpt_dir: str, device: str, base: int) -> dict:
    """Rebuild each saved model from its checkpoint's own recorded shape."""
    models = {}
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.endswith(".pt"):
            continue
        name = fn[:-3]
        blob = torch.load(os.path.join(ckpt_dir, fn), map_location=device)
        arch = blob.get("arch", name)
        in_ch = blob["in_ch"]
        model = create_model(arch, base=base, in_ch=in_ch).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        models[name] = (model, name in MULTI_FRAME)
        print(f"  loaded {name:<22} arch={arch:<20} in_ch={in_ch}", flush=True)
    return models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m2-root", required=True)
    p.add_argument("--ckpt-dir", default=os.path.join("runs", "architecture_sweep"))
    p.add_argument("--out", default=os.path.join("runs", "rescored"))
    p.add_argument("--cache-dir", default=os.path.join("runs", "references"))
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--eval-frames", type=int, default=8)
    p.add_argument("--n-frames", type=int, default=5, help="deform_fusion input frames")
    p.add_argument("--device", default="cuda")
    p.add_argument("--panel-frame", type=int, default=0)
    p.add_argument("--zoom-x", type=int, default=900)
    p.add_argument("--zoom-w", type=int, default=320)
    p.add_argument("--zoom-z", type=int, default=120)
    p.add_argument("--zoom-h", type=int, default=200)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-panels", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("re-scoring saved checkpoints against the ALIGNED reference")
    print(f"  checkpoints: {args.ckpt_dir}")
    print(f"  eval frames per stack: {args.eval_frames}\n", flush=True)

    references: dict[str, Reference] = {}
    for folder, alines in REFERENCE_STACKS:
        try:
            ref, info = build_reference(args.m2_root, folder, alines, args.cache_dir)
            references[folder] = ref
            mag = float(np.mean(np.hypot(ref.shifts[:, 0], ref.shifts[:, 1])))
            print(f"  reference {folder[:46]:<46} mean |shift| {mag:.2f} px", flush=True)
        except Exception as e:                                        # noqa: BLE001
            print(f"  reference {folder[:46]} FAILED {type(e).__name__}: {e}", flush=True)
    if not references:
        raise SystemExit("no references available")

    print("\nloading checkpoints", flush=True)
    models = load_checkpoints(args.ckpt_dir, args.device, args.base)
    if not models:
        raise SystemExit(f"no .pt checkpoints in {args.ckpt_dir}")

    print("\nscoring", flush=True)
    floor = noisy_floor_both_ways(args, references)
    results: dict[str, dict] = {"_noisy_floor": floor, "_config": vars(args)}
    for name, (model, multi) in models.items():
        results[name] = score_both_ways(model, multi, args, references)
        m = results[name]
        print(f"  {name:<22} aligned {m['psnr_aligned']:.3f} / {m['ssim_aligned']:.4f}   "
              f"unaligned {m['psnr_unaligned']:.3f} / {m['ssim_unaligned']:.4f}   "
              f"delta {m['d_psnr']:+.3f} dB", flush=True)

    print("\n" + "=" * 96)
    print(f"{'architecture':<22}{'PSNR now':>10}{'PSNR was':>10}{'delta':>9}"
          f"{'SSIM now':>10}{'SSIM was':>10}{'delta':>9}{'frames':>8}")
    print("=" * 96)
    print(f"{'noisy input':<22}{floor['psnr_aligned']:>10.3f}{floor['psnr_unaligned']:>10.3f}"
          f"{floor['psnr_aligned'] - floor['psnr_unaligned']:>+9.3f}"
          f"{floor['ssim_aligned']:>10.4f}{floor['ssim_unaligned']:>10.4f}"
          f"{floor['ssim_aligned'] - floor['ssim_unaligned']:>+9.4f}{'--':>8}")
    for name in models:
        m = results[name]
        print(f"{name:<22}{m['psnr_aligned']:>10.3f}{m['psnr_unaligned']:>10.3f}"
              f"{m['d_psnr']:>+9.3f}{m['ssim_aligned']:>10.4f}{m['ssim_unaligned']:>10.4f}"
              f"{m['d_ssim']:>+9.4f}{args.n_frames if name in MULTI_FRAME else 1:>8}")

    deltas = np.array([results[n]["d_psnr"] for n in models])
    print(f"\ndelta spread across models: {deltas.min():+.3f} to {deltas.max():+.3f} dB "
          f"(range {deltas.ptp():.3f})")
    print("A uniform delta would mean the old ordering survived the bug; a spread")
    print("means part of the old ordering was an artefact of the misalignment.")

    save_json(os.path.join(args.out, "rescored.json"), results)
    print(f"wrote {os.path.join(args.out, 'rescored.json')}")

    if not args.no_panels:
        print("\nqualitative panels", flush=True)
        folder, alines = REFERENCE_STACKS[0]
        if folder in references:
            qualitative_panels(models, args, references, folder, alines, args.panel_frame)


if __name__ == "__main__":
    main()
