"""Run mirror-study checkpoints (configs A-E) on real retina sample data.

Retina volumes have no clean reference: each frame is a different retinal
location, so the temporal-average trick from the mirror study is invalid here.
This script therefore produces a *qualitative + no-reference* comparison:

  - montage TIFF per frame  [noisy | A | B | C | D | E]  in one fixed display
    domain so panels are directly comparable
  - no-reference metrics per config: SNR/CNR (linear domain, retina signal ROI,
    p99.99 stat — matches model_train.py) and bg_sigma (display domain, bottom
    rows above the noise floor)

Back-transform caveat: average-target configs (B-E) were trained to output the
z-scored log of a temporal-average target, which does not exist for retina
frames. Predictions are un-normalized with the frame's own full-band stats
instead — a close approximation; absolute linear-domain levels may be slightly
offset, but the display-domain comparison is unaffected by a global offset.

Run:  python experiments/run_retina_compare.py
      python experiments/run_retina_compare.py --folder 6mm_1024Aline_disc --stride 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from run_mirror_study import CONFIGS, BAND, CROP, ROOT, ckpt_registry_path
from configs.default import FolderSpec
from engine.metrics import roi_bounds, bg_bounds, roi_snr_cnr, to_physical_intensity
from networks import create_model
from utils.io_tiff import save_tiff_stack

# Retina signal ROI (y rows) + stat, matching model_train.py's evaluation setup.
SIG_Y0, SIG_Y1 = 111, 600
SIG_STAT = "p99.99"


def _model_kwargs_for_cfg(cfg: dict, n_sub: int) -> dict:
    """Mirror engine/train.py model-kwargs logic so eval builds an identical model."""
    kw = {"base": cfg["base"]}
    if cfg["model_name"] == "resunet_pseudo3d_multilevel":
        kw["n_sub_channels"] = 2 * n_sub
    else:
        kw["in_ch"] = 1 if cfg["input_mode"] == "fullband" else (2 + (2 * n_sub if n_sub > 0 else 0))
    return kw


def _gather_input(cfg: dict, out: dict) -> np.ndarray:
    if cfg["input_mode"] == "fullband":
        return out["target_full"][None, ...].astype(np.float32)
    chans = [out["input_w1"], out["input_w2"]]
    if "input_sub_windows" in out:
        chans.extend(out["input_sub_windows"])
    return np.stack(chans, axis=0).astype(np.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--study-name", default="mirror_study",
                    help="which study's ckpts.json to use (e.g. mirror_study_v2)")
    ap.add_argument("--folder", default="6mm_1024Aline", help="retina sample folder under images\\Maestro3")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-preds", action="store_true", help="also write per-config full prediction stacks")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    import json
    with open(ckpt_registry_path(args.runs_root, args.study_name)) as f:
        ckpts = json.load(f)

    fs = FolderSpec(root_folder=ROOT, data_folder=args.folder, pixels=2048, alines=1024,
                    crop_depth=CROP, n_sub_windows=0, **BAND)

    from preprocess import BscanProcessor
    proc = BscanProcessor(fs)
    log_eps = float(proc.cfg.log_eps)
    paths = proc.bscan_paths[::max(1, args.stride)]
    if args.max_frames is not None:
        paths = paths[: args.max_frames]
    F = len(paths)

    out_dir = os.path.join(args.runs_root, args.study_name, "retina_eval", args.folder)
    os.makedirs(out_dir, exist_ok=True)

    # Build all models up front.
    models: list[tuple[str, dict, torch.nn.Module]] = []
    for tag, cfg in CONFIGS:
        ck = ckpts.get(tag)
        if not ck or not os.path.exists(ck):
            print(f"[SKIP] {tag}: missing checkpoint {ck}")
            continue
        m = create_model(cfg["model_name"], **_model_kwargs_for_cfg(cfg, 0)).to(device)
        state = torch.load(ck, map_location="cpu")
        m.load_state_dict(state["model"], strict=True)
        m.eval()
        models.append((tag, cfg, m))
        print(f"[LOAD] {tag}: {ck}")

    # Fixed display window from frame 0's noisy full-band log image, shared by
    # every panel and every frame so brightness/contrast are comparable.
    out0 = proc.process_one(paths[0], frame_idx=0, need_linear_full=True)
    noisy_log0 = np.log10(out0["target_full_linear"].astype(np.float64) + log_eps)
    disp_lo, disp_hi = np.percentile(noisy_log0, [1, 99])
    disp_rng = max(disp_hi - disp_lo, 1e-6)

    def disp(x_log: np.ndarray) -> np.ndarray:
        return np.clip((x_log - disp_lo) / disp_rng, 0.0, 1.0).astype(np.float32)

    H, W = noisy_log0.shape
    sig_roi = roi_bounds(H, W, SIG_Y0, SIG_Y1)
    bg_roi = bg_bounds(H, W, x0=sig_roi[2], x1=sig_roi[3])
    by0, by1, bx0, bx1 = bg_roi

    panel_tags = ["noisy_input"] + [t for t, _, _ in models]
    acc = {t: {"snr": [], "cnr": [], "bg_sigma": []} for t in panel_tags}
    montage_pages: list[np.ndarray] = []
    pred_stacks: dict[str, list[np.ndarray]] = {t: [] for t, _, _ in models} if args.save_preds else {}
    perframe_rows: list[dict] = []

    for i, p in enumerate(paths):
        out = proc.process_one(p, frame_idx=i, need_linear_full=True)
        mu, sd = float(out["target_mu"]), float(out["target_sd"])
        meta = {"target_mu": mu, "target_sd": sd, "log_eps": log_eps}

        noisy_lin = out["target_full_linear"].astype(np.float64)
        noisy_disp = disp(np.log10(noisy_lin + log_eps))
        panels = [noisy_disp]

        s, c = roi_snr_cnr(noisy_lin.astype(np.float32), sig_roi, bg_roi, sig_stat=SIG_STAT)
        bsig = float(np.std(noisy_disp[by0:by1, bx0:bx1]))
        acc["noisy_input"]["snr"].append(s); acc["noisy_input"]["cnr"].append(c)
        acc["noisy_input"]["bg_sigma"].append(bsig)
        perframe_rows.append({"frame": i, "tag": "noisy_input", "snr": s, "cnr": c, "bg_sigma": bsig})

        for tag, cfg, m in models:
            x = torch.from_numpy(np.ascontiguousarray(_gather_input(cfg, out))[None, ...]).to(device)
            pred_norm = m(x).cpu().numpy()[0, 0]
            pred_lin = to_physical_intensity(pred_norm, meta)
            pred_disp = disp(np.log10(np.maximum(pred_lin, 0) + log_eps))
            panels.append(pred_disp)
            if args.save_preds:
                pred_stacks[tag].append(pred_disp)

            s, c = roi_snr_cnr(pred_lin.astype(np.float32), sig_roi, bg_roi, sig_stat=SIG_STAT)
            bsig = float(np.std(pred_disp[by0:by1, bx0:bx1]))
            acc[tag]["snr"].append(s); acc[tag]["cnr"].append(c)
            acc[tag]["bg_sigma"].append(bsig)
            perframe_rows.append({"frame": i, "tag": tag, "snr": s, "cnr": c, "bg_sigma": bsig})

        montage_pages.append(np.concatenate(panels, axis=1))
        print(f"[frame {i + 1}/{F}] done")

    # Fixed [0,1] -> [0,65535] mapping (no per-frame stretch) keeps every frame
    # and panel in the same display domain.
    save_tiff_stack(os.path.join(out_dir, "compare_montage.tiff"),
                    np.stack(montage_pages, axis=0), dtype="uint16",
                    scale_per_slice=False, p_lo=0.0, p_hi=100.0)
    with open(os.path.join(out_dir, "panel_order.txt"), "w") as f:
        f.write("Left-to-right panel order in compare_montage.tiff:\n")
        f.write("\n".join(panel_tags) + "\n")

    if args.save_preds:
        for tag, pages in pred_stacks.items():
            save_tiff_stack(os.path.join(out_dir, f"{tag}_pred.tiff"),
                            np.stack(pages, axis=0), dtype="uint16",
                            scale_per_slice=False, p_lo=0.0, p_hi=100.0)

    cols = ["frame", "tag", "snr", "cnr", "bg_sigma"]
    with open(os.path.join(out_dir, "retina_perframe.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(perframe_rows)

    sum_csv = os.path.join(out_dir, f"summary_retina_{args.folder}.csv")
    with open(sum_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["tag", "snr", "cnr", "bg_sigma"])
        wtr.writeheader()
        for t in panel_tags:
            wtr.writerow({"tag": t,
                          "snr": float(np.nanmean(acc[t]["snr"])),
                          "cnr": float(np.nanmean(acc[t]["cnr"])),
                          "bg_sigma": float(np.nanmean(acc[t]["bg_sigma"]))})

    print(f"\n=== RETINA COMPARISON ({args.folder}, {F} frames, no-reference metrics) ===")
    print(f"{'config':<22}{'SNR_dB':>9}{'CNR_dB':>9}{'bg_sig':>9}")
    for t in panel_tags:
        print(f"{t:<22}{np.nanmean(acc[t]['snr']):>9.2f}{np.nanmean(acc[t]['cnr']):>9.2f}"
              f"{np.nanmean(acc[t]['bg_sigma']):>9.4f}")
    print(f"\n[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
