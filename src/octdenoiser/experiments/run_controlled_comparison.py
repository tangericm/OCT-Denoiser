"""Controlled head-to-head: complementary sub-band (B) vs frame-pair position (C).

The first ablation was confounded (each scheme scored on its own input) and the
fair evaluation ran a single seed at 1500 steps, leaving C ahead of B by 0.56 dB
PSNR -- a gap too small to call against unknown run-to-run variance.

This runs both schemes at depth under matched conditions, across several seeds,
so the comparison has an error bar.

MATCHED ACROSS EVERY RUN
------------------------
Training volumes, held-out volumes, architecture, base width, patch size,
patches per frame, batch size, learning rate, weight decay, optimiser, cosine
schedule, gradient clipping, augmentation, step count, and the evaluation
protocol. Only the supervision scheme and the seed vary.

WHAT CANNOT BE MATCHED, AND IS NOT
----------------------------------
B consumes one sub-band; C consumes a full-band frame. That difference IS the
thing under test, so equalising it is not possible. Two consequences to keep in
view when reading the table:

  * C touches two raw frames per sample, B one. Runs are matched on GRADIENT
    STEPS, which is the standard control, not on raw frames read.
  * B's input is band-limited (2.43x the axial PSF width). Its denoising problem
    is genuinely easier; a win for B is a win on an easier task.

Primary metric is PSNR/SSIM against the registered Maestro2 50-frame averages,
which match no scheme's training objective and so cannot flatter either. The
in-domain no-reference numbers are reported as descriptors, not as the decision
metric -- the residual-leak measure was observed to REVERSE ordering between
Maestro2 and M3, so it is not trusted to rank.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from octdenoiser.configs.default import TrainConfig
from octdenoiser.data.datamodule import RawBscanDataModule
from octdenoiser.engine.losses import compute_total_loss, unpack_batch
from octdenoiser.experiments.run_fair_eval import (
    REFERENCE_STACKS,
    build_reference,
    noisy_input_baseline,
    score_scheme,
)
from octdenoiser.experiments.run_supervision_ablation import (
    SCHEMES,
    TRAIN_VOLUMES,
    specs,
)
from octdenoiser.experiments.run_supervision_ablation import (
    evaluate as evaluate_in_domain,
)
from octdenoiser.networks import create_model
from octdenoiser.utils.helpers import save_json, seed_all

CONTENDERS = ("B_complementary", "C_pair_position")


def train_one(scheme: str, seed: int, args) -> tuple[torch.nn.Module, dict]:
    seed_all(seed, deterministic=False)
    cfg = TrainConfig(
        folder_specs=specs(args.m3_root, TRAIN_VOLUMES),
        model_name="resunet_pseudo3d", base=args.base,
        patch_h=args.patch, patch_w=args.patch,
        patches_per_frame=args.patches_per_frame, patch_mode="patch",
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_frac=0.9, augment=True, cache_frames_per_worker=args.cache_frames,
        device=args.device, seed=seed, lr=args.lr, **SCHEMES[scheme],
    )
    dm = RawBscanDataModule(cfg)
    dm.setup()
    loader = dm.train_loader()

    in_ch = next(iter(loader))[0].shape[1]
    model = create_model("resunet_pseudo3d", base=args.base, in_ch=in_ch).to(args.device)
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
            print(f"  [{scheme} s{seed}] {step}/{args.steps} "
                  f"loss={np.mean(hist[-args.log_every:]):.5f} "
                  f"{(time.time()-t0)/step:.3f}s/step", flush=True)

    del dm, loader
    return model, {"final_loss": float(np.mean(hist[-100:])), "in_ch": in_ch}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m3-root", required=True)
    p.add_argument("--m2-root", required=True)
    p.add_argument("--out", default=os.path.join("runs", "controlled_comparison"))
    p.add_argument("--cache-dir", default=os.path.join("runs", "references"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patches-per-frame", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-frames", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-frames", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("controlled comparison: B_complementary vs C_pair_position")
    print(f"  seeds={seeds}  steps={args.steps}  base={args.base}  "
          f"patch={args.patch}  batch={args.batch_size}  lr={args.lr}")
    print(f"  train volumes: {[v for v, _ in TRAIN_VOLUMES]}")
    print("  matched: volumes, architecture, all hyperparameters, step count, eval")
    print("  differs: supervision scheme and seed only\n", flush=True)

    references: dict[str, np.ndarray] = {}
    for folder, alines in REFERENCE_STACKS:
        try:
            ref, _ = build_reference(args.m2_root, folder, alines, args.cache_dir)
            references[folder] = ref
        except Exception as e:                                    # noqa: BLE001
            print(f"  reference {folder[:44]} FAILED {type(e).__name__}: {e}", flush=True)
    if not references:
        raise SystemExit("no references available")
    floor = noisy_input_baseline(args.m2_root, references, args.eval_frames)
    print(f"noisy-input floor: PSNR={floor['psnr']:.3f} SSIM={floor['ssim']:.4f}\n", flush=True)

    results: dict[str, dict] = {"_noisy_floor": floor, "_config": vars(args)}
    for scheme in CONTENDERS:
        for seed in seeds:
            key = f"{scheme}__seed{seed}"
            print(f"=== {key} ===", flush=True)
            try:
                model, info = train_one(scheme, seed, args)
                model.eval()
                ref_scores = score_scheme(model, scheme, args.m2_root, args.device,
                                          references, args.eval_frames)
                dom = evaluate_in_domain(model, args.m3_root, args.device, scheme,
                                         n_frames=args.eval_frames)
                results[key] = {**info, "reference": ref_scores, "in_domain": dom}
                torch.save({"model": model.state_dict(), "in_ch": info["in_ch"],
                            "scheme": scheme, "seed": seed},
                           os.path.join(args.out, f"{key}.pt"))
                print(f"  -> PSNR={ref_scores['psnr']:.3f} SSIM={ref_scores['ssim']:.4f}",
                      flush=True)
                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as e:                                # noqa: BLE001
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                results[key] = {"error": f"{type(e).__name__}: {e}"}
            save_json(os.path.join(args.out, "results.json"), results)
            print(flush=True)

    print("=" * 88)
    print(f"{'scheme':<22}{'PSNR mean+-sd':>20}{'SSIM mean+-sd':>20}{'speckle':>11}{'n':>5}")
    print("=" * 88)
    summary: dict[str, dict] = {}
    for scheme in CONTENDERS:
        runs = [v for k, v in results.items()
                if k.startswith(scheme) and "error" not in v]
        if not runs:
            print(f"{scheme:<22}  no successful runs")
            continue
        ps = np.array([r["reference"]["psnr"] for r in runs])
        ss = np.array([r["reference"]["ssim"] for r in runs])
        sp = np.array([r["reference"]["speckle_ratio"] for r in runs])
        summary[scheme] = {"psnr_mean": float(ps.mean()), "psnr_sd": float(ps.std(ddof=1) if ps.size > 1 else 0.0),
                           "ssim_mean": float(ss.mean()), "ssim_sd": float(ss.std(ddof=1) if ss.size > 1 else 0.0),
                           "n": int(ps.size)}
        print(f"{scheme:<22}{ps.mean():>13.3f} +-{ps.std(ddof=1) if ps.size > 1 else 0.0:>5.3f}"
              f"{ss.mean():>13.4f} +-{ss.std(ddof=1) if ss.size > 1 else 0.0:>5.4f}"
              f"{sp.mean():>11.4f}{ps.size:>5}")

    if len(summary) == 2:
        b, c = summary["B_complementary"], summary["C_pair_position"]
        d_psnr = c["psnr_mean"] - b["psnr_mean"]
        pooled = np.sqrt((b["psnr_sd"] ** 2 + c["psnr_sd"] ** 2) / 2) or 1e-9
        print(f"\n  C - B on PSNR: {d_psnr:+.3f} dB   pooled sd {pooled:.3f}   "
              f"|diff|/sd = {abs(d_psnr)/pooled:.2f}")
        print("  A gap smaller than the run-to-run spread is not a result.")
    results["_summary"] = summary
    save_json(os.path.join(args.out, "results.json"), results)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
