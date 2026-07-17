"""Mirror-data denoising baseline study (configs A-E).

Trains five configs with identical seed / split / schedule / data and scores
them against a clean temporal-average reference on a held-out mirror folder.

    A  resunet_pseudo3d  bandgap [w1,w2]  -> full-band frame target   (current method)
    B  resunet_pseudo3d  bandgap [w1,w2]  -> temporal-average target
    C  dncnn             full-band 1ch    -> temporal-average target
    D  unet2d            full-band 1ch    -> temporal-average target
    E  resunet_pseudo3d  full-band 1ch    -> temporal-average target  (isolates arch)

Run (GPU):   python experiments/run_mirror_study.py
Quick check: python experiments/run_mirror_study.py --smoke
Eval only:   python experiments/run_mirror_study.py --eval-only
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.default import TrainConfig, FolderSpec
from utils.helpers import seed_all
from utils.run_manager import make_run_dir, setup_run_dirs
from engine.train import run_training
from utils.helpers import save_json

ROOT = r"images\Maestro3"
# mirror_6mm_1024Aline_5 is excluded entirely: it was acquired while testing
# mirrors at different depths and has artifacts near the bottom of the frame.
# _4 (mild bottom artifacts) serves as the held-out test folder instead.
TRAIN_FOLDERS = [
    "mirror_6mm_1024Aline",
    "mirror_6mm_1024Aline_2",
    "mirror_6mm_1024Aline_3",
]
TEST_FOLDER = "mirror_6mm_1024Aline_4"

# Bandgap window params (their tuned values) + mirror geometry from the probe.
BAND = dict(window_sigma=0.05, gap=0.60, gap_offset=0.015)
CROP = (0, 1024)
PEAK_ROW = 117          # mirror peak depth (probe)
SIG_Y0, SIG_Y1 = 108, 127   # signal ROI around the peak

CONFIGS = [
    ("A_bandgap_fullband",   dict(model_name="resunet_pseudo3d", base=32, input_mode="bandgap",  target_mode="fullband")),
    ("B_bandgap_average",    dict(model_name="resunet_pseudo3d", base=32, input_mode="bandgap",  target_mode="average")),
    ("C_dncnn_average",      dict(model_name="dncnn",            base=64, input_mode="fullband", target_mode="average")),
    ("D_unet_average",       dict(model_name="unet2d",           base=32, input_mode="fullband", target_mode="average")),
    ("E_resunet1ch_average", dict(model_name="resunet_pseudo3d", base=32, input_mode="fullband", target_mode="average")),
]


def mspec(name: str) -> FolderSpec:
    return FolderSpec(root_folder=ROOT, data_folder=name, pixels=2048, alines=1024,
                      crop_depth=CROP, n_sub_windows=0, **BAND)


def build_traincfg(tag: str, c: dict, args) -> TrainConfig:
    return TrainConfig(
        runs_root=args.runs_root,
        experiment_name=f"{args.study_name}/{tag}",
        folder_specs=[mspec(n) for n in TRAIN_FOLDERS],
        model_name=c["model_name"], base=c["base"],
        input_mode=c["input_mode"], target_mode=c["target_mode"],
        avg_leave_one_out=True,
        avg_cache_dir="avg_cache",
        cache_frames_per_worker=args.cache_frames,   # ~16MB/frame x this x num_workers RAM; tune to your box
        epochs=args.epochs, lr=3e-4, weight_decay=5e-5, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=True,
        patch_mode="strip", patch_h=288, patch_w=32, patches_per_frame=args.patches_per_frame,
        w_charb=0.8, w_grad=0.5,
        snr_sig_y0=SIG_Y0, snr_sig_y1=SIG_Y1, snr_sig_stat="max",
        val_every=args.val_every, save_every=max(args.epochs, 1),
        early_stop_patience=args.patience,
        device=args.device,
    )


def ckpt_registry_path(runs_root: str, study_name: str = "mirror_study") -> str:
    return os.path.join(runs_root, study_name, "ckpts.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--study-name", default="mirror_study",
                    help="namespace for run dirs / ckpts.json / eval outputs; bump (e.g. mirror_study_v2) when the train/test split changes")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patches-per-frame", type=int, default=32)
    ap.add_argument("--cache-frames", type=int, default=128, help="LRU frames cached per worker; raise to fit all train frames")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run on CPU")
    ap.add_argument("--eval-only", action="store_true", help="skip training, evaluate cached checkpoints")
    ap.add_argument("--no-tiffs", action="store_true", help="skip prediction/comparison TIFF export during eval")
    ap.add_argument("--all-folders", action="store_true", help="eval every mirror folder (4 train + test), not just the held-out one")
    ap.add_argument("--resume", action="store_true", help="skip configs whose checkpoint already exists in ckpts.json")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 4
        args.num_workers = 0
        args.patches_per_frame = 2
        args.val_every = 1
        args.patience = 99

    import json
    reg_path = ckpt_registry_path(args.runs_root, args.study_name)
    ckpts: dict = {}
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            ckpts = json.load(f)

    if not args.eval_only:
        for tag, c in CONFIGS:
            if args.resume and tag in ckpts and os.path.exists(ckpts[tag]):
                print(f"\n{'='*70}\n[SKIP] {tag}: checkpoint exists ({ckpts[tag]})\n{'='*70}")
                continue
            print(f"\n{'='*70}\n[TRAIN] {tag}: {c}\n{'='*70}")
            cfg = build_traincfg(tag, c, args)
            seed_all(cfg.seed, deterministic=cfg.deterministic)
            run_dir = make_run_dir(cfg.runs_root, cfg.experiment_name)
            paths = setup_run_dirs(run_dir)
            result = run_training(cfg, paths)
            ckpts[tag] = result["best_ckpt_path"]
            os.makedirs(os.path.dirname(reg_path), exist_ok=True)
            save_json(reg_path, ckpts)

    # ---- Evaluate all configs ----
    from tools.eval_mirror import evaluate_all
    device = args.device if torch.cuda.is_available() else "cpu"
    ms_dir = os.path.join(args.runs_root, args.study_name)
    folders = (TRAIN_FOLDERS + [TEST_FOLDER]) if args.all_folders else [TEST_FOLDER]
    for fold in folders:
        held = (fold == TEST_FOLDER)
        kind = "held-out (unseen)" if held else "TRAIN folder (seen data — optimistic)"
        print(f"\n{'#'*70}\n[eval] folder {fold}  [{kind}]\n{'#'*70}")
        out_csv = os.path.join(ms_dir, f"summary_{fold}.csv")
        save_dir = None if args.no_tiffs else os.path.join(ms_dir, "eval_tiffs", fold)
        evaluate_all(CONFIGS, ckpts, mspec(fold), device, PEAK_ROW, SIG_Y0, SIG_Y1, out_csv,
                     avg_leave_one_out=True, save_dir=save_dir)


if __name__ == "__main__":
    main()
