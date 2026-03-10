"""
Inspect predicted spectra by running live inference from a checkpoint.

Loads a model checkpoint, processes a single raw B-scan frame through
BscanProcessor.process_one_spectrum(), runs the model, and produces
side-by-side comparisons of:
  - Spectral magnitude  |S(k)|
  - Spectral phase      arg(S(k))
  - Depth profile       |IFFT(S(k))|  (log10, full depth before crop)
  - Full B-scan         log-magnitude image
  - Spectral magnitude / phase error maps

for four spectra: input W1, input W2, ground-truth (spec_full), prediction.

Usage
-----
Set CKPT_PATH (and optionally BSCAN_PATH) in the CONFIG block, then run:
    python tests/inspect_spectrum_predictions.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import FolderSpec
from networks import create_model
from preprocess import BscanProcessor

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CKPT_PATH = r"runs\Spectrum\20260310_084519\checkpoints\best.pt"

# Which entry in cfg.folder_specs to use (0-based)
FOLDER_SPEC_IDX = 0

# Explicit path to a bscan*.raw to inspect.
# Set to None to use FRAME_IDX from the folder_spec's discovered bscan list.
BSCAN_PATH: str | None = None

# Frame index used when BSCAN_PATH is None
FRAME_IDX = 0

# A-line for 1-D spectral / depth plots (None → middle A-line)
ALINE_IDX: int | None = None

# Inference device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory for saved figures (None → auto: <run_dir>/inspect/)
FIG_OUT_DIR: str| None = None

# Log scale for depth-profile plot?
DEPTH_LOG = True

# dB clip for B-scan display (relative to per-frame max)
DB_CLIP = 40.0
# ---------------------------------------------------------------------------

_CHUNK = 512  # A-lines per forward pass


def _folder_spec_from_ckpt(cfg_dict: dict, idx: int) -> FolderSpec:
    """Reconstruct a FolderSpec from the cfg dict saved inside a checkpoint."""
    fs_d = cfg_dict["folder_specs"][idx]
    # asdict() converts tuples → lists; restore where needed
    fs_d = dict(fs_d)
    if "crop_depth" in fs_d and fs_d["crop_depth"] is not None:
        fs_d["crop_depth"] = tuple(fs_d["crop_depth"])
    valid = set(FolderSpec.__dataclass_fields__)
    return FolderSpec(**{k: v for k, v in fs_d.items() if k in valid})


def _spec_to_bscan_log(spec: np.ndarray, use_log: bool, log_eps: float, apply_fftshift: bool) -> np.ndarray:
    """Complex spectrum [pixels, alines] → log-magnitude B-scan [pixels, alines] (full depth)."""
    depth = sfft.ifft(spec, axis=0, workers=-1)
    mag = np.abs(depth).astype(np.float32)
    if apply_fftshift:
        mag = sfft.fftshift(mag, axes=0).astype(np.float32)
    if use_log:
        return np.log10(mag + log_eps).astype(np.float32)
    return mag


def _savefig(fig: plt.Figure, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[OK] Saved: {path}")


def main() -> None:
    # ---- Load checkpoint & reconstruct config ----------------------------
    out_dir = FIG_OUT_DIR if FIG_OUT_DIR is not None else os.path.join(
        os.path.dirname(os.path.dirname(CKPT_PATH)), "inspect"
    )

    print(f"[INFO] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg_dict   = ckpt["cfg"]
    model_name = cfg_dict["model_name"]
    base       = int(cfg_dict["base"])
    epoch      = int(ckpt.get("epoch", -1))
    best_val   = float(ckpt.get("best_val", float("nan")))
    print(f"[INFO] model={model_name}  base={base}  epoch={epoch}  best_val={best_val:.6f}")

    folder_spec = _folder_spec_from_ckpt(cfg_dict, FOLDER_SPEC_IDX)
    print(f"[INFO] FolderSpec: {folder_spec.root_folder}/{folder_spec.data_folder}")

    # ---- Build processor & resolve bscan path ----------------------------
    proc = BscanProcessor(folder_spec)

    bscan_path = BSCAN_PATH if BSCAN_PATH is not None else proc.bscan_paths[FRAME_IDX]
    print(f"[INFO] Bscan: {os.path.basename(bscan_path)}")

    # ---- Process frame ---------------------------------------------------
    out         = proc.process_one_spectrum(bscan_path, frame_idx=FRAME_IDX)
    norm_factor = float(out["norm_factor"])

    pixels = folder_spec.pixels
    alines = out["spec_full"].shape[1]

    # Build network input [6, pixels, alines]  (normalised, as during training)
    w1_mask_2d = np.broadcast_to(out["w1_mask"][:, None], out["spec_w1"].shape)
    w2_mask_2d = np.broadcast_to(out["w2_mask"][:, None], out["spec_w2"].shape)
    x_np = np.stack([
        out["spec_w1"].real, out["spec_w1"].imag,
        out["spec_w2"].real, out["spec_w2"].imag,
        w1_mask_2d.astype(np.float32),
        w2_mask_2d.astype(np.float32),
    ], axis=0).astype(np.float32)  # [6, pixels, alines]

    # ---- Load model & run inference --------------------------------------
    model = create_model(model_name, base=base).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[INFO] Model loaded ({sum(p.numel() for p in model.parameters()):,} params)  device={DEVICE}")

    with torch.no_grad():
        # Reshape to [alines, 6, pixels] for A-line batch inference
        x_alines = torch.from_numpy(
            np.ascontiguousarray(x_np.transpose(2, 0, 1))
        ).to(DEVICE)  # [alines, 6, pixels]

        pred_chunks = []
        for j in range(0, alines, _CHUNK):
            pred_chunks.append(model(x_alines[j:j + _CHUNK]).cpu())
        pred_alines = torch.cat(pred_chunks, dim=0).numpy()  # [alines, 2, pixels]

    # Reconstruct complex spectrum in physical units
    pred_spec = (pred_alines[:, 0, :] + 1j * pred_alines[:, 1, :]).T * norm_factor  # [pixels, alines]
    pred_spec = pred_spec.astype(np.complex64)

    # Scale inputs / GT to physical units (undo per-frame z-score normalisation)
    spec_w1   = (out["spec_w1"]   * norm_factor).astype(np.complex64)
    spec_w2   = (out["spec_w2"]   * norm_factor).astype(np.complex64)
    spec_full = (out["spec_full"] * norm_factor).astype(np.complex64)

    use_log        = folder_spec.use_log
    log_eps        = float(folder_spec.log_eps)
    apply_fftshift = folder_spec.apply_fftshift_depth

    aline  = alines // 2 if ALINE_IDX is None else int(ALINE_IDX)
    k_axis = np.arange(pixels)
    z_axis = np.arange(pixels)
    print(f"[INFO] Inspecting A-line {aline} (of {alines})")

    # ---- Spectral magnitude (one A-line) ---------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Spectral Magnitude — frame {FRAME_IDX}, A-line {aline}")
    ax.plot(k_axis, spec_w1[:, aline].real,   label="W1 input",  color="tomato",      alpha=0.8, linewidth=0.9)
    ax.plot(k_axis, spec_w2[:, aline].real,   label="W2 input",  color="mediumseagreen", alpha=0.8, linewidth=0.9)
    ax.plot(k_axis, spec_full[:, aline].real, label="GT full",   color="steelblue",   linewidth=1.2)
    ax.plot(k_axis, pred_spec[:, aline].real, label="Predicted", color="crimson",      linewidth=1.2, linestyle="--")
    ax.set_xlabel("Pixel"); ax.set_ylabel("Re(S(k))")
    ax_w = ax.twinx()
    ax_w.plot(k_axis, out["w1_mask"] - 0.5, color="tomato",         linewidth=0.8, linestyle=":", alpha=0.6, label="W1 window")
    ax_w.plot(k_axis, out["w2_mask"] - 0.5, color="mediumseagreen", linewidth=0.8, linestyle=":", alpha=0.6, label="W2 window")
    ax_w.set_ylabel("Window weight"); ax_w.set_ylim(-1.25, 1.25); ax_w.set_yticks([-0.5, 0, 0.5])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_w.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"spec_magnitude_frame{FRAME_IDX}_aline{aline}.png", out_dir)
    plt.close(fig)

    # ---- Spectral phase (one A-line) -------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Spectral Phase — frame {FRAME_IDX}, A-line {aline}")
    ax.plot(k_axis, np.angle(spec_w1[:, aline]),   label="W1 input",  color="tomato",         alpha=0.8, linewidth=0.9)
    ax.plot(k_axis, np.angle(spec_w2[:, aline]),   label="W2 input",  color="mediumseagreen", alpha=0.8, linewidth=0.9)
    ax.plot(k_axis, np.angle(spec_full[:, aline]), label="GT full",   color="steelblue",      linewidth=1.2)
    ax.plot(k_axis, np.angle(pred_spec[:, aline]), label="Predicted", color="crimson",         linewidth=1.2, linestyle="--")
    ax.set_xlabel("Pixel"); ax.set_ylabel("Phase (rad)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"spec_phase_frame{FRAME_IDX}_aline{aline}.png", out_dir)
    plt.close(fig)

    # ---- Depth profile |IFFT(S)| (one A-line) ----------------------------
    def _depth(spec_col: np.ndarray) -> np.ndarray:
        d = sfft.ifft(spec_col)
        if apply_fftshift:
            d = sfft.fftshift(d)
        return np.abs(d).astype(np.float32)

    mag_w1, mag_w2     = _depth(spec_w1[:, aline]),   _depth(spec_w2[:, aline])
    mag_full, mag_pred = _depth(spec_full[:, aline]), _depth(pred_spec[:, aline])

    if DEPTH_LOG:
        plot_fn = lambda m: np.log10(m + log_eps)
        ylabel = "log10(|IFFT(S)|)"
    else:
        plot_fn = lambda m: m
        ylabel = "|IFFT(S)|"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Depth Profile — frame {FRAME_IDX}, A-line {aline}")
    ax.plot(z_axis, plot_fn(mag_w1),   label="W1 input",  alpha=0.8, linewidth=0.9)
    ax.plot(z_axis, plot_fn(mag_w2),   label="W2 input",  alpha=0.8, linewidth=0.9)
    ax.plot(z_axis, plot_fn(mag_full), label="GT full",   color="steelblue", linewidth=1.2)
    ax.plot(z_axis, plot_fn(mag_pred), label="Predicted", color="crimson",   linewidth=1.2, linestyle="--")
    ax.set_xlabel("Depth pixel"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f"depth_profile_frame{FRAME_IDX}_aline{aline}.png", out_dir)
    plt.close(fig)

    # ---- Full B-scan (2D images) -----------------------------------------
    bscan_w1   = _spec_to_bscan_log(spec_w1,   use_log, log_eps, apply_fftshift)
    bscan_w2   = _spec_to_bscan_log(spec_w2,   use_log, log_eps, apply_fftshift)
    bscan_full = _spec_to_bscan_log(spec_full, use_log, log_eps, apply_fftshift)
    bscan_pred = _spec_to_bscan_log(pred_spec, use_log, log_eps, apply_fftshift)

    vmax = float(bscan_full.max())
    vmin = vmax - DB_CLIP / 20.0

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, img, title in zip(axes,
                               [bscan_w1, bscan_w2, bscan_full, bscan_pred],
                               ["W1 input", "W2 input", "GT full", "Predicted"]):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title); ax.set_axis_off()
    fig.suptitle(f"B-scan (full depth, no crop) — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"bscan_full_frame{FRAME_IDX}.png", out_dir)
    plt.close(fig)

    # ---- Spectral magnitude error map ------------------------------------
    mag_err = np.abs(pred_spec) - np.abs(spec_full)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    im = axes[0].imshow(np.abs(spec_full), aspect="auto", cmap="inferno")
    axes[0].set_title("|GT full|"); axes[0].set_axis_off()
    fig.colorbar(im, ax=axes[0], fraction=0.03)

    im = axes[1].imshow(np.abs(pred_spec), aspect="auto", cmap="inferno")
    axes[1].set_title("|Predicted|"); axes[1].set_axis_off()
    fig.colorbar(im, ax=axes[1], fraction=0.03)

    vabs = float(np.percentile(np.abs(mag_err), 99))
    im = axes[2].imshow(mag_err, aspect="auto", cmap="seismic", vmin=-vabs, vmax=vabs)
    axes[2].set_title("|Pred| − |GT|"); axes[2].set_axis_off()
    fig.colorbar(im, ax=axes[2], fraction=0.03)

    fig.suptitle(f"Spectral Magnitude Error — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"spec_mag_error_frame{FRAME_IDX}.png", out_dir)
    plt.close(fig)

    # ---- Spectral phase error map ----------------------------------------
    threshold        = float(np.percentile(np.abs(spec_full), 80))
    mask             = np.abs(spec_full) > threshold
    phase_err        = np.angle(pred_spec * np.conj(spec_full))   # wrapped [-π, π]
    phase_err_masked = np.where(mask, phase_err, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    im = axes[0].imshow(phase_err, aspect="auto", cmap="seismic", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("Phase error (all k)"); axes[0].set_axis_off()
    fig.colorbar(im, ax=axes[0], fraction=0.03, label="rad")

    im = axes[1].imshow(phase_err_masked, aspect="auto", cmap="seismic", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Phase error (|GT| > p80 = {threshold:.2e})")
    axes[1].set_axis_off()
    fig.colorbar(im, ax=axes[1], fraction=0.03, label="rad")

    fig.suptitle(f"Spectral Phase Error (pred vs GT) — frame {FRAME_IDX}", fontsize=13)
    fig.tight_layout()
    _savefig(fig, f"spec_phase_error_frame{FRAME_IDX}.png", out_dir)
    plt.close(fig)

    # ---- Summary stats ---------------------------------------------------
    mask_flat = mask.ravel()
    rms_phase = float(np.sqrt(np.nanmean(phase_err_masked ** 2)))
    print(f"\n=== Spectral comparison summary — frame {FRAME_IDX}, A-line {aline} ===")
    print(f"  pixels={pixels}  alines={alines}  norm_factor={norm_factor:.4e}")
    print(f"  Predicted  |S| mean={np.abs(pred_spec).mean():.4e}  std={np.abs(pred_spec).std():.4e}")
    print(f"  GT full    |S| mean={np.abs(spec_full).mean():.4e}  std={np.abs(spec_full).std():.4e}")
    print(f"  Mag MAE (all k):       {np.abs(mag_err).mean():.4e}")
    print(f"  Mag MAE (|GT| > p80):  {np.abs(mag_err.ravel()[mask_flat]).mean():.4e}")
    print(f"  Phase RMSE (|GT|>p80): {rms_phase:.4f} rad  ({np.degrees(rms_phase):.2f} deg)")


if __name__ == "__main__":
    main()
