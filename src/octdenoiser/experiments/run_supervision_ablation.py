"""Supervision-scheme ablation on the M3 OCTA volumes.

Trains one architecture under four supervision schemes and scores them on
held-out VOLUMES with a reference-free battery.

    A  bandgap -> fullband      the existing method. Its target CONTAINS its
                                input, measured speckle correlation +0.138.
    B  complementary sub-band   fixes that leak (+0.003) but both views are
                                narrow-band: 2.43x the axial PSF width, and a
                                0.269 chromatic signal mismatch.
    C  frame pair, position     two full-bandwidth frames one position apart
                                (11.7 um). Speckle corr +0.017, no chromatic
                                bias, no resolution cost.
    D  frame pair, repeat       the two OCTA repeats at one position. Speckle
                                corr +0.105 -- same scatterers, so speckle only
                                decorrelates as far as motion carries it.

FAIRNESS CAVEAT, read before comparing rows
-------------------------------------------
Scheme B consumes a SUB-BAND input, so its output is band-limited by
construction. Any resolution or detail metric partly measures its input rather
than its denoiser. That handicap is the point -- it is the cost of the spectral
split -- but B must be read as handicapped by construction, not simply beaten.

Split is by whole volume. The 9x9 mm acquisitions are held out entirely, so the
test set also probes generalisation across scan geometry.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from octdenoiser.configs.default import FolderSpec, TrainConfig
from octdenoiser.data.datamodule import RawBscanDataModule
from octdenoiser.engine.losses import compute_total_loss, unpack_batch
from octdenoiser.networks import create_model
from octdenoiser.preprocess import BscanProcessor
from octdenoiser.utils.helpers import save_json, seed_all

ROOT_DEFAULT = r"images\Maestro3"

# Only the x2 OCTA volumes: frame-pair supervision needs the interleaved
# repeat layout, and mixing in single-repeat volumes would change the pair
# geometry per folder.
TRAIN_VOLUMES = [
    ("M3_Disc_3x3mm_320x320x2", 320),
    ("M3_Disc_6x6mm_512x512x2", 512),
    ("M3_Macula_3x3mm_320x320x2", 320),
    ("M3_Macula_6x6mm_512x512x2", 512),
]
VAL_VOLUMES = [("M3_Wide_12x9mm_1024x768x2", 1024)]
TEST_VOLUMES = [("M3_Disc_9x9mm_512x512x2", 512), ("M3_Macula_9x9mm_512x512x2", 512)]

SCHEMES: dict[str, dict] = {
    "A_bandgap_fullband": dict(supervision="spectral", input_mode="bandgap",
                               target_mode="fullband"),
    "B_complementary": dict(supervision="spectral", input_mode="bandgap",
                            target_mode="complementary"),
    "C_pair_position": dict(supervision="frame_pair", pair_mode="position",
                            position_step=1),
    "D_pair_repeat": dict(supervision="frame_pair", pair_mode="repeat"),
}

LATERAL_SIGMA = 3.0
TISSUE = slice(120, 700)
BACKGROUND = slice(760, 1000)


def specs(root: str, volumes) -> list[FolderSpec]:
    return [
        FolderSpec(root_folder=root, data_folder=name, pixels=2048, alines=al,
                   crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015)
        for name, al in volumes
    ]


def _parts(img: np.ndarray):
    st = gaussian_filter1d(img.astype(np.float64), LATERAL_SIGMA, axis=1, mode="nearest")
    return st, img.astype(np.float64) - st


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


@torch.no_grad()
def evaluate(model, root: str, device: str, scheme: str, n_frames: int = 12) -> dict:
    """Reference-free battery on held-out volumes.

    speckle_ratio      output speckle energy / input speckle energy. Lower means
                       more speckle removed.
    structure_keep     correlation of the output's smooth component with the
                       input's. Near 1 means structure survived.
    residual_leak      correlation of (input - output) with the input's smooth
                       component. Near 0 means the residual is structure-free,
                       i.e. the model removed noise and not anatomy. This is the
                       metric SNR cannot fake.
    """
    model.eval()
    acc: dict[str, list[float]] = {"speckle_ratio": [], "structure_keep": [], "residual_leak": []}

    for name, alines in TEST_VOLUMES:
        fs = FolderSpec(root_folder=root, data_folder=name, pixels=2048, alines=alines,
                        crop_depth=(0, 1024), window_sigma=0.05, gap=0.60, gap_offset=0.015)
        proc = BscanProcessor(fs)
        step = max(1, len(proc.bscan_paths) // n_frames)
        for i in range(0, min(len(proc.bscan_paths), n_frames * step), step):
            out = proc.process_one(proc.bscan_paths[i], frame_idx=i, fft_workers=-1)
            # Each scheme must be fed the input it was TRAINED on. A consumes
            # both sub-bands (its Pseudo3DStem treats them as a depth-2 volume,
            # so a 1-channel input fails outright); B consumes one sub-band;
            # the frame-pair schemes consume a full-band frame.
            if scheme == "A_bandgap_fullband":
                chans = [out["input_w1"], out["input_w2"]]
            elif scheme == "B_complementary":
                chans = [out["input_w1"]]
            else:
                chans = [out["target_full"]]

            # The reference for the metrics is the model's OWN first input
            # channel, so each scheme is judged against what it actually saw.
            src = chans[0]
            x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).to(device)
            pred = model(x)[0, 0].float().cpu().numpy()

            st_in, sp_in = _parts(src[TISSUE])
            st_out, sp_out = _parts(pred[TISSUE])
            resid = src[TISSUE].astype(np.float64) - pred[TISSUE].astype(np.float64)

            acc["speckle_ratio"].append(float(sp_out.var() / max(sp_in.var(), 1e-12)))
            acc["structure_keep"].append(_corr(st_in, st_out))
            acc["residual_leak"].append(abs(_corr(resid, st_in)))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def train_one(scheme: str, kw: dict, args) -> dict:
    seed_all(args.seed, deterministic=False)
    cfg = TrainConfig(
        folder_specs=specs(args.root, TRAIN_VOLUMES),
        model_name="resunet_pseudo3d", base=args.base,
        patch_h=args.patch, patch_w=args.patch, patches_per_frame=args.patches_per_frame,
        patch_mode="patch", batch_size=args.batch_size, num_workers=args.num_workers,
        train_frac=0.9, augment=True, cache_frames_per_worker=args.cache_frames,
        device=args.device, seed=args.seed, lr=args.lr, **kw,
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            loss = compute_total_loss(model(x), y, w_charb=0.8, w_grad=0.5)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss.detach()))

        if step % args.log_every == 0 or step == args.steps:
            recent = float(np.mean(hist[-args.log_every:]))
            print(f"  [{scheme}] step {step}/{args.steps}  loss={recent:.5f}  "
                  f"{(time.time()-t0)/step:.3f} s/step", flush=True)

    ckpt = os.path.join(args.out, f"{scheme}.pt")
    torch.save({"model": model.state_dict(), "in_ch": in_ch, "scheme": scheme}, ckpt)

    metrics = evaluate(model, args.root, args.device, scheme, n_frames=args.eval_frames)
    metrics.update(in_ch=in_ch, final_loss=float(np.mean(hist[-50:])),
                   steps=args.steps, checkpoint=ckpt)
    del model, opt, dm, loader
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT_DEFAULT)
    p.add_argument("--out", default=os.path.join("runs", "supervision_ablation"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patches-per-frame", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-frames", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--eval-frames", type=int, default=12)
    p.add_argument("--schemes", default="", help="comma-separated subset; default all")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    chosen = [s.strip() for s in args.schemes.split(",") if s.strip()] or list(SCHEMES)

    print(f"supervision ablation -> {args.out}")
    print(f"  train volumes: {[v for v, _ in TRAIN_VOLUMES]}")
    print(f"  test volumes : {[v for v, _ in TEST_VOLUMES]}  (unseen 9x9 geometry)")
    print(f"  {args.steps} steps, base={args.base}, patch={args.patch}, "
          f"batch={args.batch_size}\n", flush=True)

    results: dict[str, dict] = {}
    for scheme in chosen:
        print(f"=== {scheme} ===", flush=True)
        try:
            results[scheme] = train_one(scheme, SCHEMES[scheme], args)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [{scheme}] FAILED: {type(e).__name__}: {e}", flush=True)
            results[scheme] = {"error": f"{type(e).__name__}: {e}"}
        # Written after every scheme so a long run yields partial results.
        save_json(os.path.join(args.out, "results.json"), results)
        print(flush=True)

    print("=" * 84)
    print(f"{'scheme':<22}{'in_ch':>7}{'speckle_ratio':>15}{'structure_keep':>16}{'residual_leak':>15}")
    print("=" * 84)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<22}  FAILED: {m['error']}")
            continue
        print(f"{name:<22}{m['in_ch']:>7}{m['speckle_ratio']:>15.4f}"
              f"{m['structure_keep']:>16.4f}{m['residual_leak']:>15.4f}")
    print("\nspeckle_ratio  lower = more speckle removed")
    print("structure_keep near 1 = structure survived")
    print("residual_leak  near 0 = residual is structure-free, i.e. noise removed not anatomy")
    print("\nB consumes a sub-band input, so it is band-limited by construction and")
    print("is handicapped on any detail metric. That is the cost of the spectral split.")


if __name__ == "__main__":
    main()
