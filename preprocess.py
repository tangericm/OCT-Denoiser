import os
import glob
import tifffile
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    pixels: int = 2048
    alines: int = 1024
    data_folder: str = "6mm_1024Aline"  # Name of data folder containing bscan*.raw files

    # Preprocessing knobs
    do_dc_subtract: bool = True
    window_type: str = "hann"  # "hann" or "ones"
    use_log: bool = True
    log_eps: float = 1e-6

    # Depth handling
    # Typically you only need half-depth; set to (0, 1024) if desired
    crop_depth: Tuple[int, int] = (1024, 2048)
    apply_fftshift_depth: bool = True  # closer to your MATLAB ifftshift usage

    # Spectral gap / windowing
    window_sigma: float = 0.08  # normalized width of Gaussian windows
    gap: float = 0.25           # relative gap in [0, 0.5], you can vary later

    # Dispersion compensation (optional)
    # In MATLAB you have [a2, a3] used with k^(p+1) style; here we keep it plumbed.
    dispersion: Optional[List[float]] = None  # e.g. [1.3e-6, 5.4e-10] or None
    
    # Debug mode: when True, no output files are written (NPZ, PNGs, etc.)
    debug_mode: bool = False


# -----------------------------
# Low-level IO
# -----------------------------
def read_bscan_raw(path: str, pixels: int, alines: int) -> np.ndarray:
    """
    Reads uint16 raw bscan file saved in row-major [pixels x alines] like MATLAB fread([pixels alines],'uint16').
    Returns float32 array [pixels, alines].
    """
    data = np.fromfile(path, dtype=np.uint16)
    expected = pixels * alines
    if data.size != expected:
        raise ValueError(f"{os.path.basename(path)} has {data.size} elements; expected {expected}.")
    data = data.reshape((pixels, alines), order="F")  # MATLAB fread([pixels alines]) fills column-major
    return data.astype(np.float32)


def read_clb_resampling(clb_path: str, pixels: int) -> np.ndarray:
    """
    Reads CLB float32 resampling array.
    Your MATLAB: fseek 64 bytes, fread float32 Inf, then resampling(1:pixels).
    """
    with open(clb_path, "rb") as f:
        f.seek(64, os.SEEK_SET)
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size < pixels:
        raise ValueError(f"CLB resampling array too small: {arr.size} < {pixels}")
    return arr[:pixels].astype(np.float32)


# -----------------------------
# Signal processing
# -----------------------------
def hann_window(pixels: int) -> np.ndarray:
    w = np.hanning(pixels).astype(np.float32)
    # Your MATLAB scales Hann to [0,1]; mimic that.
    w = w - w.min()
    w = w / (w.max() + 1e-12)
    return w


def dc_subtract(spec: np.ndarray) -> np.ndarray:
    """
    spec: [pixels, alines]
    subtract mean across alines for each pixel (row-wise), like MATLAB: OCT - mean(OCT,2)
    Removes DC spike at first pixel by setting it equal to second pixel.
    """
    spec[0, :] = spec[1, :]  # Remove first-pixel spike by copying second pixel
    return spec - spec.mean(axis=1, keepdims=True)


def resample_klinear(spec: np.ndarray, resampling: np.ndarray) -> np.ndarray:
    """
    spec: [pixels, alines], sampled at uniform index in [0,1]
    resampling: [pixels] values in [0,1] mapping to new k-linear index positions
    Performs cubic spline interpolation per A-line, similar to MATLAB interp1(index, OCT(:,n), resampling, 'cubic')
    """
    
    x = np.linspace(0.0, 1.0, spec.shape[0], dtype=np.float32)
    xp = resampling.astype(np.float32)

    out = np.empty_like(spec, dtype=np.float32)
    
    # Cubic spline interpolation per column
    for a in range(spec.shape[1]):
        cs = CubicSpline(x, spec[:, a], bc_type='natural')
        out[:, a] = cs(xp).astype(np.float32)
    
    return out


def apply_dispersion(spec: np.ndarray, dispersion: List[float]) -> np.ndarray:
    """
    spec: real-valued spectra [pixels, alines]
    Returns complex spectra with dispersion phase applied: spec * exp(1j * phase(k))
    We mimic your MATLAB approach:
      k = (0:pixels-1)' - pixels/2
      phase = sum_p dispersion[p] * k^(p+1)
    """
    pixels = spec.shape[0]
    k = (np.arange(pixels, dtype=np.float32) - pixels / 2.0)  # [pixels]
    phase = np.zeros((pixels,), dtype=np.float32)
    for p, c in enumerate(dispersion):
        phase += c * (k ** (p + 2))  # (p+1) in MATLAB with p starting at 1 -> exponent p+1 => here p+2
    phase_term = np.exp(1j * phase.astype(np.float32)).astype(np.complex64)  # [pixels]
    return (spec.astype(np.complex64) * phase_term[:, None]).astype(np.complex64)


def recon_bscan_from_spectrum(spec_complex: np.ndarray,
                              crop: Tuple[int, int],
                              use_log: bool,
                              log_eps: float,
                              apply_fftshift_depth: bool) -> np.ndarray:
    """
    spec_complex: complex64 [pixels, alines]
    Returns float32 B-scan [H, W] where H is depth and W is alines.
    """
    # IFFT along spectral axis (pixels)
    depth_c = np.fft.ifft(spec_complex, axis=0)  # [pixels, alines] complex
    mag = np.abs(depth_c).astype(np.float32)     # [pixels, alines]

    if apply_fftshift_depth:
        # MATLAB used ifftshift(...,1) after ifft; depending on convention, fftshift/ifftshift differs.
        # Empirically, OCT often uses fftshift to center DC; here we'll use fftshift to align structure.
        mag = np.fft.fftshift(mag, axes=0).astype(np.float32)

    z0, z1 = crop
    bscan = mag[z0:z1, :]  # [H,W]

    if use_log:
        bscan = np.log(bscan + log_eps).astype(np.float32)

    # Normalize per-frame for stable training (you can revise later)
    mu = float(bscan.mean())
    sd = float(bscan.std()) + 1e-6
    bscan = (bscan - mu) / sd
    return bscan.astype(np.float32)


def gaussian_window_1d(pixels: int, center: float, sigma: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, pixels, dtype=np.float32)
    w = np.exp(-0.5 * ((x - center) / sigma) ** 2).astype(np.float32)
    return w


def make_two_window_masks(pixels: int, gap: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two Gaussian windows separated by a central gap. Similar to the slide concept.
    gap is relative width in [0,0.5].
    """
    delta = 2.5 * sigma
    c1 = np.clip(0.5 - gap / 2.0 - delta, 0.05, 0.95)
    c2 = np.clip(0.5 + gap / 2.0 + delta, 0.05, 0.95)
    w1 = gaussian_window_1d(pixels, float(c1), sigma)
    w2 = gaussian_window_1d(pixels, float(c2), sigma)
    return w1, w2


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert float image to uint8 using robust percentile-based scaling.
    """
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


# -----------------------------
# High-level processor
# -----------------------------
class BscanProcessor:
    def __init__(self, root_folder: str, cfg: Config):
        """
        root_folder: project folder containing
          - data folder (default: "6mm_1024Aline") with bscan*.raw
          - CLB file "000003_3DOCT-1_FUNDUS.CLB" (can also be inside root_folder)
        cfg.data_folder: name of the data folder (default: "6mm_1024Aline")
        """
        self.root = root_folder
        self.cfg = cfg

        # Find data folder using config parameter
        self.data_dir = os.path.join(root_folder, cfg.data_folder)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        # Find CLB (either in root or inside data_dir)
        clb_candidates = [
            os.path.join(root_folder, "000003_3DOCT-1_FUNDUS.CLB"),
            os.path.join(self.data_dir, "000003_3DOCT-1_FUNDUS.CLB"),
            # os.path.join(root_folder, "9071984_3DOCT-1_FUNDUS.CLB"),
            # os.path.join(self.data_dir, "9071984_3DOCT-1_FUNDUS.CLB"),
        ]
        self.clb_path = None
        for c in clb_candidates:
            if os.path.isfile(c):
                self.clb_path = c
                break
        if self.clb_path is None:
            raise FileNotFoundError(f"Could not find CLB file in: {clb_candidates}")

        # Enumerate bscans
        self.bscan_paths = sorted(glob.glob(os.path.join(self.data_dir, "bscan*.raw")))
        if len(self.bscan_paths) == 0:
            raise FileNotFoundError(f"No bscan*.raw found in {self.data_dir}")

        # Load resampling LUT once
        self.resampling = read_clb_resampling(self.clb_path, cfg.pixels)

        # Precompute apodization window
        if cfg.window_type == "hann":
            self.apod = hann_window(cfg.pixels)
        elif cfg.window_type == "ones":
            self.apod = np.ones((cfg.pixels,), dtype=np.float32)
        elif cfg.window_type == "gaussian":
            self.apod = gaussian_window_1d(cfg.pixels, 0.5, cfg.window_sigma)
        else:
            raise ValueError(f"Unknown window_type={cfg.window_type}")

        # Precompute two spectral windows
        self.w1, self.w2 = make_two_window_masks(cfg.pixels, cfg.gap, cfg.window_sigma)

    def _debug_plot(self, step_name: str, data: np.ndarray, is_complex: bool = False, 
                    windows: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Concise debug plotting of a single processing step with optional window overlay.
        
        Args:
            step_name: Name of the processing step
            data: [pixels, alines] array to plot
            is_complex: If True, plot magnitude of complex data
            windows: Optional tuple (w1, w2) of window functions to overlay on center A-line
        """
        if is_complex:
            mag = np.abs(data)
        else:
            mag = np.abs(data)
        
        # Plot center A-line and 2D view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Center A-line (log scale) with optional window overlay
        aline_idx = mag.shape[1] // 2
        ax1.semilogy(mag[:, aline_idx], linewidth=0.8, color='steelblue', label='Spectrum')
        
        if windows is not None:
            w1, w2 = windows
            # Normalize windows to match spectrum scale for visibility
            spec_max = mag[:, aline_idx].max()
            ax1_twin = ax1.twinx()
            ax1_twin.plot(w1, linewidth=1.5, color='red', alpha=0.7, label='Window 1')
            ax1_twin.plot(w2, linewidth=1.5, color='orange', alpha=0.7, label='Window 2')
            ax1_twin.set_ylabel('Window Amplitude', color='red')
            ax1_twin.set_ylim([0, 1.1])
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
        
        ax1.set_title(f"{step_name} - Center A-line")
        ax1.set_xlabel("Pixel")
        ax1.set_ylabel("Magnitude (log)")
        ax1.grid(True, alpha=0.3)
        
        # 2D heatmap
        im = ax2.imshow(np.log(mag + 1e-6), aspect='auto', cmap='gray', origin='upper')
        ax2.set_title(f"{step_name} - All A-lines")
        ax2.set_xlabel("A-line")
        ax2.set_ylabel("Pixel")
        plt.colorbar(im, ax=ax2, label="Log Magnitude")
        
        plt.tight_layout()
        plt.show(block=False)

    def process_one(self, bscan_path: str) -> Dict[str, np.ndarray]:
        """
        Returns dict containing:
          - target_full: [H,W] float32
          - input_w1: [H,W] float32
          - input_w2: [H,W] float32
        """
        cfg = self.cfg

        raw = read_bscan_raw(bscan_path, cfg.pixels, cfg.alines)  # [pixels, alines]
        if cfg.debug_mode:
            self._debug_plot("1. Raw Spectrum", raw)

        if cfg.do_dc_subtract:
            raw = dc_subtract(raw)
            if cfg.debug_mode:
                self._debug_plot("2. After DC Subtraction", raw)

        # Resample to k-linear using CLB LUT
        resamp = resample_klinear(raw, self.resampling)  # [pixels, alines]
        if cfg.debug_mode:
            self._debug_plot("3. After Resampling", resamp)

        # Apodization (Hann)
        resamp = (resamp * self.apod[:, None]).astype(np.float32)
        if cfg.debug_mode:
            self._debug_plot("4. After Apodization", resamp)

        # Dispersion compensation
        if cfg.dispersion is not None and len(cfg.dispersion) > 0:
            spec_full = apply_dispersion(resamp, cfg.dispersion)  # complex64 [pixels,alines]
        else:
            spec_full = resamp.astype(np.complex64)  # treat as complex with imag=0
        if cfg.debug_mode:
            self._debug_plot("5. Full Spectrum (Complex)", spec_full, is_complex=True, 
                           windows=(self.w1, self.w2))

        # Full-spectrum target recon
        target_full = recon_bscan_from_spectrum(
            spec_full, cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth
        )

        # Two-window gapped inputs
        spec1 = (spec_full * self.w1[:, None]).astype(np.complex64)
        spec2 = (spec_full * self.w2[:, None]).astype(np.complex64)
        if cfg.debug_mode:
            self._debug_plot("6a. Window 1 Spectrum", spec1, is_complex=True)
            self._debug_plot("6b. Window 2 Spectrum", spec2, is_complex=True)

        input_w1 = recon_bscan_from_spectrum(
            spec1, cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth
        )
        input_w2 = recon_bscan_from_spectrum(
            spec2, cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth
        )
        if cfg.debug_mode:
            self._debug_plot("7a. Reconstructed Window 1", input_w1)
            self._debug_plot("7b. Reconstructed Window 2", input_w2)
            self._debug_plot("7c. Full Target", target_full)

        return {
            "target_full": target_full,  # [H,W]
            "input_w1": input_w1,        # [H,W]
            "input_w2": input_w2,        # [H,W]
        }

    def process_all(self,
                    out_npz: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Process all frames; optionally save dataset npz and debug PNGs.
        Returns dict with:
          X: [F,2,H,W], Y: [F,1,H,W]
        """
        cfg = self.cfg
        F = len(self.bscan_paths)

        # Determine H,W by running first frame
        first = self.process_one(self.bscan_paths[0])
        H, W = first["target_full"].shape

        X = np.zeros((F, 2, H, W), dtype=np.float32)
        Y = np.zeros((F, 1, H, W), dtype=np.float32)

        for i, p in enumerate(self.bscan_paths):
            out = self.process_one(p)
            X[i, 0] = out["input_w1"]
            X[i, 1] = out["input_w2"]
            Y[i, 0] = out["target_full"]

        if out_npz is not None:
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)
            np.savez_compressed(out_npz, X=X, Y=Y)
            print(f"[OK] Saved dataset: {out_npz}  X={X.shape} Y={Y.shape}")
            
            # Save full stacks as multi-page TIFFs
            base_dir = os.path.dirname(out_npz)
            
            # Window 1 stack: [F, H, W] -> uint8
            w1_stack = np.stack([to_uint8(X[i, 0]) for i in range(F)])
            w1_path = os.path.join(base_dir, f"{cfg.data_folder}_window1.tiff")
            tifffile.imwrite(w1_path, w1_stack, photometric='minisblack')
            print(f"[OK] Saved Window 1 stack: {w1_path} (shape: {w1_stack.shape})")
            
            # Window 2 stack: [F, H, W] -> uint8
            w2_stack = np.stack([to_uint8(X[i, 1]) for i in range(F)])
            w2_path = os.path.join(base_dir, f"{cfg.data_folder}_window2.tiff")
            tifffile.imwrite(w2_path, w2_stack, photometric='minisblack')
            print(f"[OK] Saved Window 2 stack: {w2_path} (shape: {w2_stack.shape})")
            
            # Target stack: [F, H, W] -> uint8
            target_stack = np.stack([to_uint8(Y[i, 0]) for i in range(F)])
            target_path = os.path.join(base_dir, f"{cfg.data_folder}_target.tiff")
            tifffile.imwrite(target_path, target_stack, photometric='minisblack')
            print(f"[OK] Saved Target stack: {target_path} (shape: {target_stack.shape})")
                

        return {"X": X, "Y": Y}


# -----------------------------
# Unit-like tests / sanity checks
# -----------------------------
def run_sanity_tests(dataset: Dict[str, np.ndarray], cfg: Config):
    X, Y = dataset["X"], dataset["Y"]

    assert X.ndim == 4 and Y.ndim == 4, "X,Y should be rank-4 tensors"
    assert X.shape[0] == Y.shape[0], "Frame count mismatch"
    assert X.shape[1] == 2, "Expected dual-input channels=2"
    assert Y.shape[1] == 1, "Expected target channels=1"

    F, _, H, W = X.shape
    print(f"[TEST] Shapes OK: X={X.shape}, Y={Y.shape}")

    # Numerical sanity: not all zeros / not NaN
    assert np.isfinite(X).all() and np.isfinite(Y).all(), "Found NaN/Inf"
    assert np.abs(X).mean() > 1e-4, "X seems near-zero; pipeline likely broken"
    assert np.abs(Y).mean() > 1e-4, "Y seems near-zero; pipeline likely broken"
    print("[TEST] Finite/non-trivial OK")

    # Optional: target should generally have *more detail* than windowed inputs
    # We’ll compare simple gradient energy.
    def grad_energy(img):
        dy = np.abs(img[1:, :] - img[:-1, :]).mean()
        dx = np.abs(img[:, 1:] - img[:, :-1]).mean()
        return float(dy + dx)

    idx = min(0, F - 1)
    ge_t = grad_energy(Y[idx, 0])
    ge_1 = grad_energy(X[idx, 0])
    ge_2 = grad_energy(X[idx, 1])
    print(f"[TEST] Gradient energy (frame0): target={ge_t:.4f}, w1={ge_1:.4f}, w2={ge_2:.4f}")
    # Not asserting ordering, because normalization/log may affect it. Just reporting.


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    # Update this to your project root that contains "6mm_1024Aline" and the CLB file.
    project_root = r"C:\Users\erict\OneDrive\Desktop\Projects\OCT Reconstruction\images"

    cfg = Config(
        data_folder="6mm_1024Aline", 
        pixels=2048,
        alines=1024,
        do_dc_subtract=True,
        window_type="hann",
        use_log=True,
        crop_depth=(1024, 2048),
        apply_fftshift_depth=True,
        window_sigma=0.08,
        gap=0.25,
        dispersion=[1.315892282e-06, 5.459678905e-10], # M3
        # dispersion=[4.778474717e-06, 6.475358372e-09], # M2
        debug_mode=False,
    )

    proc = BscanProcessor(project_root, cfg)

    # Use data_folder name for output NPZ filename
    out_npz = os.path.join(project_root, "processed", f"{cfg.data_folder}_gapped_dataset.npz")

    dataset = proc.process_all(out_npz=out_npz)
    run_sanity_tests(dataset, cfg)


if __name__ == "__main__":
    main()
