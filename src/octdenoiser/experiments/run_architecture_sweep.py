"""Architecture sweep: modern and OCT-specific backbones vs the current one.

Supervision is held FIXED at frame-pair position (scheme C) for the
single-frame models, so any difference is architectural. deform_fusion is the
exception and is labelled as such below.

FAIRNESS NOTE, read before comparing rows
------------------------------------------
deform_fusion consumes K neighbouring frames; every other row consumes one.
That is the whole point -- a single B-scan does not contain what a 50-frame
average does, and one network pass measured worth only 8-16 averaged frames --
but it means deform_fusion is solving an EASIER problem with more information.
Its row is not a like-for-like architecture win and must not be read as one.
It also costs K frames of buffering and preprocessing at deployment.

Quick pass: modest step count, single seed. Enough for a general ranking, not
enough to separate close candidates. The controlled B-vs-C run showed seed
spread near 0.08 dB PSNR, so treat gaps under ~0.2 dB here as unresolved.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from octdenoiser.configs.default import TrainConfig
from octdenoiser.data.datamodule import RawBscanDataModule
from octdenoiser.data.paired_dataset import MultiFrameDataset
from octdenoiser.engine.losses import compute_total_loss, unpack_batch
from octdenoiser.experiments.run_fair_eval import (
    REFERENCE_STACKS,
    _to_unit,
    _zscore,
    build_reference,
    noisy_input_baseline,
)
from octdenoiser.experiments.run_supervision_ablation import SCHEMES, TRAIN_VOLUMES, specs
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor
from octdenoiser.tools.eval_mirror import psnr, ssim
from octdenoiser.utils.helpers import save_json, seed_all

# name -> (registry key, multi-frame?)
ARCHITECTURES = {
    "resunet_pseudo3d": ("resunet_pseudo3d", False),   # current baseline
    "nafnet": ("nafnet", False),
    "restormer": ("restormer", False),
    "ffc_resunet": ("ffc_resunet", False),
    "aniso_resunet": ("aniso_resunet", False),
    "deform_fusion": ("deform_fusion", True),          # K frames in -- see note
}

TISSUE = slice(120, 700)


def make_loader(arch_multi: bool, args):
    if not arch_multi:
        cfg = TrainConfig(
            folder_specs=specs(args.m3_root, TRAIN_VOLUMES),
            model_name="resunet_pseudo3d", base=args.base,
            patch_h=args.patch, patch_w=args.patch,
            patches_per_frame=args.patches_per_frame, patch_mode="patch",
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_frac=0.9, augment=True, cache_frames_per_worker=args.cache_frames,
            device=args.device, seed=args.seed, lr=args.lr,
            **SCHEMES["C_pair_position"],
        )
        dm = RawBscanDataModule(cfg)
        dm.setup()
        return dm.train_loader(), dm

    ds = MultiFrameDataset(
        folder_specs=specs(args.m3_root, TRAIN_VOLUMES), split="train", train_frac=0.9,
        n_input_frames=args.n_frames, position_step=1, repeats_per_position=2,
        patch_h=args.patch, patch_w=args.patch,
        patches_per_frame=args.patches_per_frame, patch_mode="patch",
        augment=True, seed=args.seed, cache_frames_per_worker=args.cache_frames,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        collate_fn=_collate)
    return loader, None


def _collate(batch):
    xs, ys, metas = zip(*batch, strict=True)
    return torch.stack(xs, 0), torch.stack(ys, 0), metas


@torch.no_grad()
def score(model, arch_multi: bool, args, references: dict) -> dict:
    """Score against the near-clean references, in the tissue ROI."""
    model.eval()
    acc: dict[str, list[float]] = {"psnr": [], "ssim": []}
    s = 2  # repeats_per_position * position_step
    for folder, alines in REFERENCE_STACKS:
        if folder not in references:
            continue
        ref_z = _zscore(references[folder][TISSUE])
        from octdenoiser.configs.default import FolderSpec
        proc = BscanProcessor(FolderSpec(
            root_folder=args.m2_root, data_folder=folder, pixels=2048, alines=alines,
            crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015))
        step = max(1, len(proc.bscan_paths) // args.eval_frames)
        for i in range(0, min(len(proc.bscan_paths), args.eval_frames * step), step):
            if arch_multi:
                idx = [min(max(i + (j - args.n_frames // 2) * s, 0),
                           len(proc.bscan_paths) - 1) for j in range(args.n_frames)]
                chans = [proc.process_one(proc.bscan_paths[j], frame_idx=j,
                                          fft_workers=-1)["target_full"] for j in idx]
            else:
                chans = [proc.process_one(proc.bscan_paths[i], frame_idx=i,
                                          fft_workers=-1)["target_full"]]
            x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).to(args.device)
            pred = model(x)[0, 0].float().cpu().numpy()[TISSUE]
            p_z = _zscore(pred)
            acc["psnr"].append(psnr(_to_unit(p_z), _to_unit(ref_z), 1.0))
            acc["ssim"].append(ssim(_to_unit(p_z), _to_unit(ref_z), 1.0))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_one(name: str, args, references: dict) -> dict:
    key, multi = ARCHITECTURES[name]
    seed_all(args.seed, deterministic=False)
    loader, dm = make_loader(multi, args)

    in_ch = next(iter(loader))[0].shape[1]
    model = create_model(key, base=args.base, in_ch=in_ch).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    hist, it, t0 = [], iter(loader), time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y, _ = unpack_batch(batch, args.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
            loss = compute_total_loss(model(x), y, w_charb=0.8, w_grad=0.5)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))
        if step % args.log_every == 0 or step == args.steps:
            print(f"  [{name}] {step}/{args.steps} loss={np.mean(hist[-args.log_every:]):.5f} "
                  f"{(time.time()-t0)/step:.3f}s/step", flush=True)
    train_s_per_step = (time.time() - t0) / args.steps

    # Inference latency at a realistic B-scan size.
    model.eval()
    probe = torch.randn(1, in_ch, 512, 512, device=args.device)
    with torch.no_grad():
        for _ in range(3):
            model(probe)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            model(probe)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.time() - t) / 10 * 1000

    metrics = score(model, multi, args, references)
    metrics.update(params_M=n_params / 1e6, in_ch=in_ch, multi_frame=multi,
                   final_loss=float(np.mean(hist[-100:])),
                   train_s_per_step=train_s_per_step, latency_ms_512=latency_ms)
    torch.save({"model": model.state_dict(), "in_ch": in_ch, "arch": key},
               os.path.join(args.out, f"{name}.pt"))
    del model, opt, loader, dm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m3-root", required=True)
    p.add_argument("--m2-root", required=True)
    p.add_argument("--out", default=os.path.join("runs", "architecture_sweep"))
    p.add_argument("--cache-dir", default=os.path.join("runs", "references"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patches-per-frame", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--cache-frames", type=int, default=64)
    p.add_argument("--n-frames", type=int, default=5, help="deform_fusion input frames")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-frames", type=int, default=6)
    p.add_argument("--archs", default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    chosen = [a.strip() for a in args.archs.split(",") if a.strip()] or list(ARCHITECTURES)

    print("architecture sweep — supervision fixed at frame-pair position")
    print(f"  {args.steps} steps, base={args.base}, patch={args.patch}, batch={args.batch_size}")
    print("  deform_fusion consumes K frames; all others consume one. Not like-for-like.\n",
          flush=True)

    references: dict[str, np.ndarray] = {}
    for folder, alines in REFERENCE_STACKS:
        try:
            ref, _ = build_reference(args.m2_root, folder, alines, args.cache_dir)
            references[folder] = ref
        except Exception as e:                                    # noqa: BLE001
            print(f"  reference {folder[:44]} FAILED {type(e).__name__}: {e}", flush=True)
    floor = noisy_input_baseline(args.m2_root, references, args.eval_frames)
    print(f"noisy-input floor: PSNR={floor['psnr']:.3f} SSIM={floor['ssim']:.4f}\n", flush=True)

    results: dict[str, dict] = {"_noisy_floor": floor}
    for name in chosen:
        print(f"=== {name} ===", flush=True)
        try:
            results[name] = run_one(name, args, references)
            m = results[name]
            print(f"  -> PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
                  f"{m['params_M']:.2f}M {m['latency_ms_512']:.1f}ms", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results[name] = {"error": f"{type(e).__name__}: {e}"}
        save_json(os.path.join(args.out, "results.json"), results)
        print(flush=True)

    print("=" * 94)
    print(f"{'architecture':<22}{'PSNR':>8}{'SSIM':>9}{'params M':>10}"
          f"{'latency ms':>12}{'s/step':>9}{'frames':>8}")
    print("=" * 94)
    print(f"{'noisy input':<22}{floor['psnr']:>8.3f}{floor['ssim']:>9.4f}"
          f"{'--':>10}{'--':>12}{'--':>9}{'--':>8}")
    for name, m in results.items():
        if name.startswith("_"):
            continue
        if "error" in m:
            print(f"{name:<22}  FAILED: {m['error'][:60]}")
            continue
        print(f"{name:<22}{m['psnr']:>8.3f}{m['ssim']:>9.4f}{m['params_M']:>10.2f}"
              f"{m['latency_ms_512']:>12.1f}{m['train_s_per_step']:>9.3f}"
              f"{m['in_ch']:>8}")
    print("\ndeform_fusion sees K frames; the rest see one. Its row reflects more")
    print("information at inference, not a like-for-like architecture win.")
    print("Single seed at a modest step count: gaps under ~0.2 dB are unresolved.")


if __name__ == "__main__":
    main()
