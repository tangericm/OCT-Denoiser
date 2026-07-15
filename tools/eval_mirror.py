"""Evaluate mirror-denoising configs against a clean temporal-average reference.

For a held-out mirror folder, the clean reference is the full-frame temporal
mean of the linear full-band magnitude. Each trained config is run on every
frame of the held-out folder and scored against that reference in a common,
GT-defined display domain so numbers are comparable across configs.

Metrics (per frame, then averaged):
  PSNR, SSIM        vs clean reference (log domain, GT-normalized to [0,1])
  bg_sigma          background noise std in the GT-normalized log domain (lower=better)
  SNR, CNR (dB)     mirror-peak ROI, computed in linear intensity
  psf_fwhm          axial FWHM (px) of the peak; compare to reference FWHM

A "noisy input" row (raw single full-band frame) is included as the no-denoise
floor.
"""
from __future__ import annotations

import os
import csv
import numpy as np
import torch

from networks import create_model
from engine.metrics import roi_bounds, bg_bounds, roi_snr_cnr, to_physical_intensity
from data.avg_targets import build_folder_sum


# ---------------------------------------------------------------------------
# Small dependency-free SSIM (Gaussian window, standard constants)
# ---------------------------------------------------------------------------
def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g).astype(np.float64)


def _filter2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    # separable-equivalent full 2D convolution via FFT (valid region trimmed)
    from scipy.signal import fftconvolve
    return fftconvolve(img, k, mode="valid")


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    k = _gaussian_kernel()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = _filter2(a, k); mu_b = _filter2(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _filter2(a * a, k) - mu_a2
    sb = _filter2(b * b, k) - mu_b2
    sab = _filter2(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float(np.mean(num / den))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def _axial_fwhm(img_lin: np.ndarray, peak_row: int, half_span: int = 40) -> float:
    prof = img_lin.mean(axis=1)
    lo = max(0, peak_row - half_span); hi = min(len(prof), peak_row + half_span)
    seg = prof[lo:hi]
    if seg.size == 0:
        return float("nan")
    pk = seg.max(); half = pk / 2.0
    above = np.where(seg >= half)[0]
    return float(above.max() - above.min() + 1) if above.size else float("nan")


# ---------------------------------------------------------------------------
def _model_kwargs_for_cfg(cfg, n_sub: int) -> dict:
    """Mirror engine/train.py model-kwargs logic so eval builds an identical model."""
    kw = {"base": cfg["base"]}
    if cfg["model_name"] == "resunet_pseudo3d_multilevel":
        kw["n_sub_channels"] = 2 * n_sub
    else:
        kw["in_ch"] = 1 if cfg["input_mode"] == "fullband" else (2 + (2 * n_sub if n_sub > 0 else 0))
    return kw


@torch.no_grad()
def evaluate_config(cfg: dict, ckpt_path: str, test_fs, device: str,
                    peak_row: int, sig_y0: int, sig_y1: int,
                    avg_leave_one_out: bool = True) -> dict:
    """Run one config's checkpoint over the held-out folder; return mean metrics."""
    from preprocess import BscanProcessor

    proc = BscanProcessor(test_fs)
    paths = proc.bscan_paths
    n_sub = getattr(test_fs, "n_sub_windows", 0)

    # Clean reference: full-frame temporal mean of linear magnitude.
    sum_mag, N = build_folder_sum(test_fs)
    clean_lin = (sum_mag / N).astype(np.float64)
    log_eps = float(proc.cfg.log_eps)
    gt_log = np.log10(clean_lin + log_eps)
    lo, hi = np.percentile(gt_log, [1, 99])
    rng = max(hi - lo, 1e-6)
    def gt_norm(x_log):  # common GT-defined [0,1] display scaling
        return np.clip((x_log - lo) / rng, 0.0, 1.0)
    gt_disp = gt_norm(gt_log)

    H, W = clean_lin.shape
    sig_roi = roi_bounds(H, W, sig_y0, sig_y1)
    sy0, sy1, sx0, sx1 = sig_roi
    bg_roi = bg_bounds(H, W, x0=sx0, x1=sx1)
    by0, by1, bx0, bx1 = bg_roi

    model = None
    if ckpt_path is not None:
        model = create_model(cfg["model_name"], **_model_kwargs_for_cfg(cfg, n_sub)).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

    acc = {k: [] for k in ("psnr", "ssim", "bg_sigma", "snr", "cnr", "psf_fwhm")}

    for i, p in enumerate(paths):
        out = proc.process_one(p, frame_idx=i, need_linear_full=True)
        mag_i = out["target_full_linear"].astype(np.float64)

        # Per-frame averaged-target normalization (matches training target domain).
        if avg_leave_one_out and N > 1:
            avg_i = (sum_mag - mag_i) / (N - 1)
        else:
            avg_i = clean_lin
        t = np.log10(avg_i + log_eps)
        tmu, tsd = float(t.mean()), float(t.std()) + 1e-6

        if model is None:
            # noisy-input floor: the raw single full-band frame
            pred_lin = mag_i
        else:
            if cfg["input_mode"] == "fullband":
                x = out["target_full"][None, None, ...]
            else:
                chans = [out["input_w1"], out["input_w2"]]
                if "input_sub_windows" in out:
                    chans.extend(out["input_sub_windows"])
                x = np.stack(chans, axis=0)[None, ...]
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
            pred_norm = model(xt).cpu().numpy()[0, 0]
            # Back-transform uses the normalization the network was TRAINED to output:
            # full-band-target configs use the single-frame stats; average-target configs
            # use the averaged-target stats.
            if cfg.get("target_mode", "average") == "fullband":
                bmu, bsd = float(out["target_mu"]), float(out["target_sd"])
            else:
                bmu, bsd = tmu, tsd
            pred_lin = to_physical_intensity(pred_norm, {"target_mu": bmu, "target_sd": bsd, "log_eps": log_eps})

        pred_log = np.log10(np.maximum(pred_lin, 0) + log_eps)
        pred_disp = gt_norm(pred_log)

        acc["psnr"].append(psnr(pred_disp, gt_disp, 1.0))
        acc["ssim"].append(ssim(pred_disp, gt_disp, 1.0))
        acc["bg_sigma"].append(float(np.std(pred_disp[by0:by1, bx0:bx1])))
        s, c = roi_snr_cnr(pred_lin.astype(np.float32), sig_roi, bg_roi, sig_stat="max")
        acc["snr"].append(s); acc["cnr"].append(c)
        acc["psf_fwhm"].append(_axial_fwhm(pred_lin, peak_row))

    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def evaluate_all(configs: list, ckpts: dict, test_fs, device: str,
                 peak_row: int, sig_y0: int, sig_y1: int, out_csv: str,
                 avg_leave_one_out: bool = True) -> list:
    """Score every config + a noisy-input floor; write and return a summary table."""
    rows = []
    # Noisy input floor (no model)
    print("[eval] noisy-input floor ...")
    m = evaluate_config({"model_name": "dncnn", "base": 1, "input_mode": "fullband"},
                        None, test_fs, device, peak_row, sig_y0, sig_y1, avg_leave_one_out)
    rows.append({"tag": "noisy_input", **m})

    for tag, cfg in configs:
        ck = ckpts.get(tag)
        if not ck or not os.path.exists(ck):
            print(f"[eval] SKIP {tag} (missing checkpoint {ck})")
            continue
        print(f"[eval] {tag}: {cfg['model_name']} input={cfg['input_mode']} target={cfg['target_mode']}")
        m = evaluate_config(cfg, ck, test_fs, device, peak_row, sig_y0, sig_y1, avg_leave_one_out)
        rows.append({"tag": tag, **m})

    cols = ["tag", "psnr", "ssim", "bg_sigma", "snr", "cnr", "psf_fwhm"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({c: r.get(c) for c in cols})

    print("\n=== MIRROR STUDY RESULTS (held-out folder) ===")
    print(f"{'config':<14}{'PSNR':>8}{'SSIM':>8}{'bg_sig':>9}{'SNR_dB':>9}{'CNR_dB':>9}{'FWHM':>7}")
    for r in rows:
        print(f"{r['tag']:<14}{r['psnr']:>8.2f}{r['ssim']:>8.4f}{r['bg_sigma']:>9.4f}"
              f"{r['snr']:>9.2f}{r['cnr']:>9.2f}{r['psf_fwhm']:>7.1f}")
    print(f"\n[OK] wrote {out_csv}")
    return rows
