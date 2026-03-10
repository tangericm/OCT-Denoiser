"""
Inspect and compare predicted spectra against training data.

Loads a saved pred_spec_*.raw alongside its companion _meta.json and runs
BscanProcessor.process_one_spectrum() on the corresponding raw B-scan to
produce side-by-side comparisons of:
  - Spectral magnitude  |S(k)|
  - Spectral phase      arg(S(k))
  - Depth profile       |IFFT(S(k))|   (log10, before crop)
  - Full B-scan         log-magnitude image

for four spectra: input W1, input W2, ground-truth (spec_full), prediction.

Usage
-----
Set the paths in the CONFIG block below, then run:
    python tests/inspect_spectrum_predictions.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft

# Make sure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import FolderSpec
from preprocess import BscanProcessor

# ---------------------------------------------------------------------------
# CONFIG — edit these to point at your run output
# ---------------------------------------------------------------------------
# Directory produced by predict_spectrum_raw_to_tiffs / predict_spectrum_from_config
PRED_DIR = r"runs\Spectrum\<timestamp>\pred_tiff\6mm_1024Aline"

# Frame index to inspect (0-based)
FRAME_IDX = 0

# A-line index to inspect for spectral / depth-profile plots (None → use middle)
ALINE_IDX = None

# Output directory for saved figures (None → show interactively)
FIG_OUT_DIR = None   # e.g. r"runs\Spectrum\<timestamp>\inspect"

# Log scale for depth-profile plot?
DEPTH_LOG = True

# dB clip for B-scan display (relative to per-frame max)
DB_CLIP = 40.0
# ---------------------------------------------------------------------------


def _load_pred_raw(pred_dir: str) -> tuple[np.ndarray, dict]:
    """
    Find and load pred_spec_*.raw + companion _meta.json from pred_dir.

    Returns
    -------
    spec_stack : complex64  [F, pixels, alines]
    meta       : dict from the JSON file
    """
    raws = [f for f in os.listdir(pred_dir) if f.startswith("pred_spec_") and f.endswith(".raw")]
    if not raws:
        raise FileNotFoundError(f"No pred_spec_*.raw found in {pred_dir}")
    if len(raws) > 1:
        raise FileNotFoundError(f"Multiple pred_spec_*.raw in {pred_dir}: {raws}. Set PRED_DIR more specifically.")

    raw_path  = os.path.join(pred_dir, raws[0])
    meta_path = raw_path.replace(".raw", "_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Companion metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    F, pixels, alines = meta["shape"]
    arr = np.fromfile(raw_path, dtype=np.complex64).reshape(F, pixels, alines, order="C")
    print(f"[OK] Loaded predicted spectra: shape={arr.shape}  file={raw_path}")
    return arr, meta


def _folder_spec_from_meta(meta: dict) -> FolderSpec:
    fs_d = meta["folder_spec"]
    return FolderSpec(
        root_folder=fs_d["root_folder"],
        data_folder=fs_d["data_folder"],
        pixels=int(fs_d["pixels"]),
        alines=int(fs_d["alines"]),
        crop_depth=tuple(fs_d["crop_depth"]),
        use_log=bool(fs_d["use_log"]),
        log_eps=float(fs_d["log_eps"]),
        apply_fftshift_depth=bool(fs_d["apply_fftshift_depth"]),
        dispersion=fs_d.get("dispersion"),
        window_sigma=float(fs_d["window_sigma"]),
        gap=float(fs_d["gap"]),
        gap_offset=float(fs_d.get("gap_offset", 0.0)),
    )


def _spec_to_bscan_log(spec: np.ndarray, use_log: bool, log_eps: float, apply_fftshift: bool) -> np.ndarray:
    """Complex spectrum [pixels, alines] → log-magnitude B-scan [pixels, alines] (full depth, no crop)."""
    depth = sfft.ifft(spec, axis=0, workers=-1)
    mag = np.abs(depth).astype(np.float32)
    if apply_fftshift:
        mag = sfft.fftshift(mag, axes=0).astype(np.float32)
    if use_log:
        return np.log10(mag + log_eps).astype(np.float32)
    return mag


def _db_clip_image(img: np.ndarray, db: float) -> np.ndarray:
    """Clip image to [max - db, max] for display."""
    vmax = float(img.max())
    return np.clip(img, vmax - db / 20.0, vmax)   # log10 domain


def _fig_path(name: str) -> str | None:
    if FIG_OUT_DIR is None:
        return None
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    return os.path.join(FIG_OUT_DIR, name)


def _savefig(fig: plt.Figure, name: str) -> None:
    path = _fig_path(name)
    if path is not None:
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"[OK] Saved: {path}")
    else:
        plt.show()


def main() -> None:
    # ---- Load predicted spectra ------------------------------------------
    pred_stack, meta = _load_pred_raw(PRED_DIR)
    F, pixels, alines = pred_stack.shape
    frame_paths       = meta["frame_paths"]
    norm_factors      = meta["norm_factors"]

    if FRAME_IDX >= F:
        raise IndexError(f"FRAME_IDX={FRAME_IDX} out of range (F={F})")

    pred_spec   = pred_stack[FRAME_IDX]          # [pixels, alines] complex64
    norm_factor = float(norm_factors[FRAME_IDX])
    bscan_path  = frame_paths[FRAME_IDX]
    print(f"[INFO] Frame {FRAME_IDX}: {os.path.basename(bscan_path)}  norm_factor={norm_factor:.6e}")

    # ---- Reconstruct training data for this frame ------------------------
    folder_spec = _folder_spec_from_meta(meta)
    proc        = BscanProcessor(folder_spec)
    out         = proc.process_one_spectrum(bscan_path, frame_idx=FRAME_IDX)

    # Scale all spectra to physical units (undo per-frame normalisation)
    nf          = float(out["norm_factor"])
    spec_w1     = out["spec_w1"] * nf   # [pixels, alines] complex64
    spec_w2     = out["spec_w2"] * nf
    spec_full   = out["spec_full"] * nf  # ground truth

    use_log        = folder_spec.use_log
    log_eps        = float(folder_spec.log_eps)
    apply_fftshift = folder_spec.apply_fftshift_depth

    aline = int(alines // 2) if ALINE_IDX is None else int(ALINE_IDX)
    print(f"[INFO] A-line index for 1D plots: {aline}")

    k_axis = np.linspace(0, 1, pixels)

    # ---- Spectral magnitude comparison (one A-line) ----------------------
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.set_title(f"Spectral Magnitude — frame {FRAME_IDX}, A-line {aline}")
    axes.plot(k_axis, np.abs(spec_w1[:, aline]),   label="W1 input",  alpha=0.8, linewidth=0.9)
    axes.plot(k_axis, np.abs(spec_w2[:, aline]),   label="W2 input",  alpha=0.8, linewidth=0.9)
    axes.plot(k_axis, np.abs(spec_full[:, aline]), label="GT full",   linewidth=1.2, color="steelblue")
    axes.plot(k_axis, np.abs(pred_spec[:, aline]), label="Predicted", linewidth=1.2, color="crimson", linestyle="--")
    axes.set_xlabel("Normalised k")
    axes.set_ylabel("|S(k)|")
    axes.legend()
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"spec_magnitude_frame{FRAME_IDX}_aline{aline}.png")
    plt.close(fig)

    # ---- Spectral phase comparison (one A-line) --------------------------
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.set_title(f"Spectral Phase — frame {FRAME_IDX}, A-line {aline}")
    axes.plot(k_axis, np.angle(spec_w1[:, aline]),   label="W1 input",  alpha=0.8, linewidth=0.9)
    axes.plot(k_axis, np.angle(spec_w2[:, aline]),   label="W2 input",  alpha=0.8, linewidth=0.9)
    axes.plot(k_axis, np.angle(spec_full[:, aline]), label="GT full",   linewidth=1.2, color="steelblue")
    axes.plot(k_axis, np.angle(pred_spec[:, aline]), label="Predicted", linewidth=1.2, color="crimson", linestyle="--")
    axes.set_xlabel("Normalised k")
    axes.set_ylabel("Phase (rad)")
    axes.legend()
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"spec_phase_frame{FRAME_IDX}_aline{aline}.png")
    plt.close(fig)

    # ---- Depth profile (|IFFT(S)|) one A-line ----------------------------
    depth_w1   = sfft.ifft(spec_w1[:, aline])
    depth_w2   = sfft.ifft(spec_w2[:, aline])
    depth_full = sfft.ifft(spec_full[:, aline])
    depth_pred = sfft.ifft(pred_spec[:, aline])

    if apply_fftshift:
        depth_w1   = sfft.fftshift(depth_w1)
        depth_w2   = sfft.fftshift(depth_w2)
        depth_full = sfft.fftshift(depth_full)
        depth_pred = sfft.fftshift(depth_pred)

    mag_w1, mag_w2   = np.abs(depth_w1), np.abs(depth_w2)
    mag_full, mag_pred = np.abs(depth_full), np.abs(depth_pred)
    z_axis = np.arange(pixels)

    if DEPTH_LOG:
        plot_w1   = np.log10(mag_w1   + log_eps)
        plot_w2   = np.log10(mag_w2   + log_eps)
        plot_full = np.log10(mag_full + log_eps)
        plot_pred = np.log10(mag_pred + log_eps)
        ylabel    = "log10(|IFFT(S)|)"
    else:
        plot_w1, plot_w2, plot_full, plot_pred = mag_w1, mag_w2, mag_full, mag_pred
        ylabel = "|IFFT(S)|"

    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    axes.set_title(f"Depth Profile — frame {FRAME_IDX}, A-line {aline}")
    axes.plot(z_axis, plot_w1,   label="W1 input",  alpha=0.8, linewidth=0.9)
    axes.plot(z_axis, plot_w2,   label="W2 input",  alpha=0.8, linewidth=0.9)
    axes.plot(z_axis, plot_full, label="GT full",   linewidth=1.2, color="steelblue")
    axes.plot(z_axis, plot_pred, label="Predicted", linewidth=1.2, color="crimson", linestyle="--")
    axes.set_xlabel("Depth pixel")
    axes.set_ylabel(ylabel)
    axes.legend()
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"depth_profile_frame{FRAME_IDX}_aline{aline}.png")
    plt.close(fig)

    # ---- Full B-scan comparison (2D images) ------------------------------
    bscan_w1   = _spec_to_bscan_log(spec_w1,   use_log, log_eps, apply_fftshift)
    bscan_w2   = _spec_to_bscan_log(spec_w2,   use_log, log_eps, apply_fftshift)
    bscan_full = _spec_to_bscan_log(spec_full, use_log, log_eps, apply_fftshift)
    bscan_pred = _spec_to_bscan_log(pred_spec, use_log, log_eps, apply_fftshift)

    vmax = float(bscan_full.max())
    vmin = vmax - DB_CLIP / 20.0

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, img, title in zip(
        axes,
        [bscan_w1, bscan_w2, bscan_full, bscan_pred],
        ["W1 input", "W2 input", "GT full", "Predicted"],
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"B-scan (full depth, no crop) — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"bscan_full_frame{FRAME_IDX}.png")
    plt.close(fig)

    # ---- Magnitude error map: pred vs GT ---------------------------------
    mag_err = np.abs(pred_spec) - np.abs(spec_full)   # [pixels, alines]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    im0 = axes[0].imshow(np.abs(spec_full), aspect="auto", cmap="inferno")
    axes[0].set_title("|GT full|"); axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], fraction=0.03)

    im1 = axes[1].imshow(np.abs(pred_spec), aspect="auto", cmap="inferno")
    axes[1].set_title("|Predicted|"); axes[1].set_axis_off()
    fig.colorbar(im1, ax=axes[1], fraction=0.03)

    vabs = float(np.percentile(np.abs(mag_err), 99))
    im2 = axes[2].imshow(mag_err, aspect="auto", cmap="seismic", vmin=-vabs, vmax=vabs)
    axes[2].set_title("|Pred| − |GT|"); axes[2].set_axis_off()
    fig.colorbar(im2, ax=axes[2], fraction=0.03)

    fig.suptitle(f"Spectral Magnitude Error — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"spec_mag_error_frame{FRAME_IDX}.png")
    plt.close(fig)

    # ---- Phase error map: pred vs GT (only where GT magnitude is large) --
    threshold = float(np.percentile(np.abs(spec_full), 80))
    mask = np.abs(spec_full) > threshold

    phase_err = np.angle(pred_spec * np.conj(spec_full))   # wrapped diff [-π, π]
    phase_err_masked = np.where(mask, phase_err, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    im0 = axes[0].imshow(phase_err, aspect="auto", cmap="seismic", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Phase error (all k)"); axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], fraction=0.03, label="rad")

    im1 = axes[1].imshow(phase_err_masked, aspect="auto", cmap="seismic", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Phase error (|GT| > p80 = {threshold:.2e})")
    axes[1].set_axis_off()
    fig.colorbar(im1, ax=axes[1], fraction=0.03, label="rad")

    fig.suptitle(f"Spectral Phase Error (pred vs GT) — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"spec_phase_error_frame{FRAME_IDX}.png")
    plt.close(fig)

    # ---- Summary stats ---------------------------------------------------
    mask_flat = mask.ravel()
    print("\n=== Spectral comparison summary (frame {}) ===".format(FRAME_IDX))
    print(f"  Pixels: {pixels}  A-lines: {alines}")
    print(f"  Predicted spec  —  |S| mean={np.abs(pred_spec).mean():.4e}  std={np.abs(pred_spec).std():.4e}")
    print(f"  GT full spec    —  |S| mean={np.abs(spec_full).mean():.4e}  std={np.abs(spec_full).std():.4e}")
    print(f"  Mag MAE (all k):  {np.abs(mag_err).mean():.4e}")
    print(f"  Mag MAE (|GT|>p80): {np.abs(mag_err.ravel()[mask_flat]).mean():.4e}")
    rms_phase = float(np.sqrt(np.nanmean(phase_err_masked ** 2)))
    print(f"  Phase RMSE (|GT|>p80, rad): {rms_phase:.4f}  ({np.degrees(rms_phase):.2f} deg)")


if __name__ == "__main__":
    main()
