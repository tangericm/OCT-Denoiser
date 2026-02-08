import os
import glob
import tifffile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.fft as sfft
import scipy.linalg as sla


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
    crop_depth: Tuple[int, int] = (0, pixels//2)
    apply_fftshift_depth: bool = False

    # Spectral gap / windowing
    window_sigma: float = 0.08  # normalized width of Gaussian windows
    gap: float = 0.25           # relative gap in [0, 0.5], you can vary later

    # Dispersion compensation
    dispersion: Optional[List[float]] = None
    
    # Debug mode: when True, no output files are written
    debug_mode: bool = True

@contextmanager
def timer(name: str, enabled: bool = True):
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {name}: {dt*1e3:.2f} ms")

# -----------------------------
# Low-level IO
# -----------------------------
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

def _precompute_natural_cubic_uniform(pixels: int, xp: np.ndarray) -> dict:
    """
    Precompute everything that depends only on:
      - uniform grid x in [0,1] with `pixels` points
      - query points xp (CLB resampling), length `pixels`

    This implements the same natural cubic spline used by CubicSpline(x, y, bc_type="natural")
    on a uniform grid, but in a way that's fast for many RHS columns (A-lines).
    """
    n = int(pixels)
    if xp.shape[0] != n:
        raise ValueError(f"xp must have length {n}, got {xp.shape[0]}")

    # uniform spacing
    h = np.float32(1.0 / (n - 1))
    inv_h = np.float32((n - 1))  # 1/h

    # Build banded matrix for natural spline second derivatives M:
    # M0 = Mn-1 = 0 (natural)
    # interior: M_{i-1} + 4 M_i + M_{i+1} = (6/h^2) * (y_{i+1} - 2y_i + y_{i-1})
    # (uniform grid; the "h" factors cancel out except in rhs)
    ab = np.zeros((3, n), dtype=np.float32)  # (upper, diag, lower) for solve_banded((1,1), ab, b)

    # diag
    ab[1, :] = 4.0
    ab[1, 0] = 1.0
    ab[1, -1] = 1.0

    # upper diag (ab[0,1:] corresponds to A[i-1,i])
    ab[0, 1:] = 1.0
    ab[0, 1] = 0.0     # because row 0 is boundary
    ab[0, -1] = 0.0    # last row boundary

    # lower diag (ab[2,:-1] corresponds to A[i+1,i])
    ab[2, :-1] = 1.0
    ab[2, -2] = 0.0    # because last row boundary
    ab[2, 0] = 0.0     # first row boundary

    # Build explicit tridiagonal arrays (dl, d, du) and factorize once.
    # This avoids refactorizing the same system every frame (major speed-up vs solve_banded).
    # Matrix corresponds to the banded 'ab' above (natural spline: M0=Mn-1=0, dropped from first/last interior eqs).
    dl = np.ones(n - 1, dtype=np.float32)   # sub-diagonal: A[i+1,i]
    d  = np.full(n, 4.0, dtype=np.float32) # main diagonal
    du = np.ones(n - 1, dtype=np.float32)   # super-diagonal: A[i,i+1]

    # Natural boundary rows as identity, and decouple first/last interior from boundary unknowns.
    d[0] = 1.0
    d[-1] = 1.0
    du[0] = 0.0        # A[0,1] = 0
    dl[0] = 0.0        # A[1,0] = 0 (since M0 fixed to 0)
    du[-1] = 0.0       # A[n-2,n-1] = 0 (since Mn-1 fixed to 0)
    dl[-1] = 0.0       # A[n-1,n-2] = 0

    # Factorize tridiagonal once (LAPACK gttrf); store factors for fast repeated solves (gttrs).
    gttrf, = sla.lapack.get_lapack_funcs(("gttrf",), (d,))
    dl_f, d_f, du_f, du2, ipiv, info = gttrf(dl.copy(), d.copy(), du.copy())
    if info != 0:
        raise RuntimeError(f"gttrf failed with info={info}")
    
    # Precompute evaluation indices and weights for each xp
    # Map xp in [0,1] to interval index i in [0, n-2]
    # i = floor(xp / h) = floor(xp * (n-1))
    t_raw = (xp.astype(np.float32) * inv_h)
    i0 = np.floor(t_raw).astype(np.int32)
    i0 = np.clip(i0, 0, n - 2)
    # local coordinate t in [0,1]
    t = (t_raw - i0.astype(np.float32)).astype(np.float32)
    a = (1.0 - t).astype(np.float32)
    b = t

    # These appear in the cubic formula:
    # S = a*y_i + b*y_{i+1} + ((a^3-a) M_i + (b^3-b) M_{i+1}) * h^2 / 6
    a3ma = (a * a * a - a).astype(np.float32)
    b3mb = (b * b * b - b).astype(np.float32)
    c = (h * h / 6.0).astype(np.float32)

    return {
        "ab": ab,          # banded system matrix for M
        "h2_over_6": c,
        "i0": i0,
        "a": a,
        "b": b,
        "a3ma": a3ma,
        "b3mb": b3mb,
        "inv_h2_6": np.float32(6.0 / (h * h)),  # rhs scale
        "n": n,
        "dl_f": dl_f,
        "d_f": d_f,
        "du_f": du_f,
        "du2": du2,
        "ipiv": ipiv,
        "_rhs": None, 
    }


def resample_klinear_cubic_operator(spec: np.ndarray, pre: dict) -> np.ndarray:
    """
    Fast natural cubic spline resampling for many A-lines.
    spec: float32 [pixels, alines]
    returns float32 [pixels, alines]

    Exactly matches natural cubic spline on uniform x in [0,1].
    """
    n = pre["n"]
    if spec.shape[0] != n:
        raise ValueError(f"spec first dim must be {n}, got {spec.shape[0]}")

    # --- Build rhs for second derivatives M ---
    # rhs[0] = rhs[-1] = 0
    # rhs[1:-1] = (6/h^2) * (y[i+1] - 2y[i] + y[i-1])
    rhs = pre.get("_rhs", None)
    if rhs is None or rhs.shape != spec.shape:
        rhs = np.empty_like(spec, dtype=np.float32)
        pre["_rhs"] = rhs
    rhs.fill(0.0)
    scale = pre["inv_h2_6"]
    rhs[1:-1, :] = scale * (spec[2:, :] - 2.0 * spec[1:-1, :] + spec[:-2, :])

    # --- Solve banded tridiagonal for M (second derivatives), for all columns at once ---
    # sla.solve_banded supports multiple RHS with shape (n, alines)
    # M = sla.solve_banded((1, 1), pre["ab"], rhs, overwrite_ab=False, overwrite_b=False, check_finite=False).astype(np.float32)

    # Use precomputed factors to solve for M (faster than solve_banded)
    gttrs, = sla.lapack.get_lapack_funcs(("gttrs",), (pre["d_f"],))
    M, info = gttrs(pre["dl_f"], pre["d_f"], pre["du_f"], pre["du2"], pre["ipiv"], rhs, trans='N')
    if info != 0:
        raise RuntimeError(f"gttrs failed with info={info}")
    M = M.astype(np.float32, copy=False)

    # --- Evaluate spline at xp using precomputed indices/weights ---
    i0 = pre["i0"]
    i1 = i0 + 1

    # Gather y_i, y_{i+1}, M_i, M_{i+1}
    # Use take along axis 0 (vectorized over xp) then broadcast over columns
    y0 = spec[i0, :]    # [pixels, alines]
    y1 = spec[i1, :]
    m0 = M[i0, :]
    m1 = M[i1, :]

    a = pre["a"][:, None]
    b = pre["b"][:, None]
    a3ma = pre["a3ma"][:, None]
    b3mb = pre["b3mb"][:, None]
    c = pre["h2_over_6"]

    out = (a * y0 + b * y1 + (a3ma * m0 + b3mb * m1) * c).astype(np.float32)
    return out

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
    # depth_c = np.fft.ifft(spec_complex, axis=0)  # [pixels, alines] complex
    depth_c = sfft.ifft(spec_complex, axis=0, workers=-1)  # [pixels, alines] complex
    mag = np.abs(depth_c).astype(np.float32)     # [pixels, alines]

    if apply_fftshift_depth:
        # mag = np.fft.fftshift(mag, axes=0).astype(np.float32)
        mag = sfft.fftshift(mag, axes=0).astype(np.float32)

    z0, z1 = crop
    bscan = mag[z0:z1, :]  # [H,W]

    if use_log:
        bscan = np.log10(bscan + log_eps).astype(np.float32)

    # Normalize per-frame for stable training
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
        self.clb_path = None
        if self.clb_path is None:
            clbs = sorted(glob.glob(os.path.join(self.data_dir, "*.CLB")))
            if len(clbs) == 1:
                self.clb_path = clbs[0]
            elif len(clbs) > 1:
                raise FileNotFoundError(
                    f"Multiple .CLB files found in dataset folder {self.data_dir}: {clbs}\n"
                )
        # Search in root_folder
        if self.clb_path is None:
            clbs = sorted(glob.glob(os.path.join(root_folder, "*.CLB")))
            if len(clbs) == 1:
                self.clb_path = clbs[0]
            elif len(clbs) > 1:
                raise FileNotFoundError(
                    f"Multiple .CLB files found in root folder {root_folder}: {clbs}\n"
                )

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

        # Cache reusable grids and buffers
        self._x = np.linspace(0.0, 1.0, cfg.pixels, dtype=np.float32)           # uniform grid
        self._xp = self.resampling.astype(np.float32, copy=False)              # CLB grid

        # Precompute cubic spline operator for CLB resampling (natural spline on uniform grid)
        self._spline_pre = _precompute_natural_cubic_uniform(cfg.pixels, self.resampling.astype(np.float32, copy=False))

        # Precompute dispersion phase term if needed (saves per-frame exp)
        self._phase_term = None
        if cfg.dispersion is not None and len(cfg.dispersion) > 0:
            k = (np.arange(cfg.pixels, dtype=np.float32) - cfg.pixels / 2.0)
            phase = np.zeros((cfg.pixels,), dtype=np.float32)
            for p, c in enumerate(cfg.dispersion):
                phase += c * (k ** (p + 2))
            self._phase_term = np.exp(1j * phase).astype(np.complex64)  # [pixels]

        # Reuse buffers to reduce allocations (per processor instance)
        self._raw_u16 = np.empty((cfg.pixels * cfg.alines,), dtype=np.uint16)
        self._raw_f32 = np.empty((cfg.pixels, cfg.alines), dtype=np.float32)
        self._resamp_f32 = np.empty((cfg.pixels, cfg.alines), dtype=np.float32)
        self._spec_full_c64 = np.empty((cfg.pixels, cfg.alines), dtype=np.complex64)
        self._spec1_c64 = np.empty((cfg.pixels, cfg.alines), dtype=np.complex64)
        self._spec2_c64 = np.empty((cfg.pixels, cfg.alines), dtype=np.complex64)

    def _debug_plot(
        self,
        step_name: str,
        data: np.ndarray,
        is_complex: bool = False,
        windows: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        frame_idx: Optional[int] = None,
        save_png: bool = False,
    ) -> None:
        if is_complex:
            mag = np.abs(data)
        else:
            mag = np.abs(data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        aline_idx = mag.shape[1] // 2
        ax1.semilogy(mag[:, aline_idx], linewidth=0.8, color="steelblue")

        if windows is not None:
            w1, w2 = windows
            ax1_twin = ax1.twinx()
            ax1_twin.plot(w1, color="red", alpha=0.7, label="Window 1")
            ax1_twin.plot(w2, color="orange", alpha=0.7, label="Window 2")
            ax1_twin.set_ylim([0, 1.1])
            ax1_twin.legend(loc="upper right")

        ax1.set_title(f"{step_name} – Center A-line")
        ax1.set_xlabel("Pixel")
        ax1.set_ylabel("Magnitude (log)")
        ax1.grid(True, alpha=0.3)

        im = ax2.imshow(np.log10(mag + 1e-6), aspect="auto", cmap="gray", origin="upper")
        ax2.set_title(f"{step_name} – All A-lines")
        ax2.set_xlabel("A-line")
        ax2.set_ylabel("Pixel")
        plt.colorbar(im, ax=ax2, label="Log Magnitude")

        plt.tight_layout()

        if (save_png and frame_idx == 0 and hasattr(self, "_debug_out_dir") and self._debug_out_dir is not None):
            safe_name = step_name.replace(" ", "_").replace(".", "")
            # Prefer explicit dataset frame basename if available (set in `process_one`),
            # otherwise fall back to numeric frame index.
            frame_name = getattr(self, "_current_frame_name", f"frame{frame_idx:03d}")
            dataset_name = getattr(self, "_dataset_name", None)
            if dataset_name:
                out_path = os.path.join(self._debug_out_dir, f"{dataset_name}_{frame_name}_{safe_name}.png")
            else:
                out_path = os.path.join(self._debug_out_dir, f"{frame_name}_{safe_name}.png")
            plt.savefig(out_path, dpi=150)

        plt.close(fig)

    def process_one(self, bscan_path: str, frame_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Returns dict containing:
          - target_full: [H,W] float32
          - input_w1: [H,W] float32
          - input_w2: [H,W] float32
        """
        prof = False
        cfg = self.cfg
        self._current_frame_name = os.path.splitext(os.path.basename(bscan_path))[0]

        pixels = cfg.pixels
        alines = cfg.alines

        # 1) Read raw spectrum (fast path, minimal allocations)
        with timer("read raw", prof):
            data = np.fromfile(bscan_path, dtype=np.uint16)
        expected = pixels * alines
        if data.size != expected:
            raise ValueError(f"{os.path.basename(bscan_path)} has {data.size} elements; expected {expected}.")

        # Reshape
        with timer("reshape raw", prof):
            raw = data.reshape((pixels, alines), order="F").astype(np.float32, copy=False)

        # 2) DC subtract (in-place where safe)
        with timer("DC subtract", prof):
            if cfg.do_dc_subtract:
                raw[0, :] = raw[3, :]
                raw[1, :] = raw[3, :]
                raw[2, :] = raw[3, :]
                raw = raw - raw.mean(axis=1, keepdims=True)

        # 3) Resample to k-linear (cubic spline, vectorized across A-lines)
        # Use cached grids self._x and self._xp to avoid per-call linspace/casts
        # cs = CubicSpline(self._x, raw, axis=0, bc_type="natural")
        # resamp = cs(self._xp).astype(np.float32, copy=False)  # [pixels, alines]
        with timer("resample cubic", prof):
            resamp = resample_klinear_cubic_operator(raw, self._spline_pre)

        # 4) Apodization (in-place multiply)
        with timer("apodization", prof):
            resamp *= self.apod[:, None]

        # 5) Build complex spectrum (with optional dispersion)
        with timer("build complex spectrum", prof):
            if self._phase_term is not None:
                spec_full = resamp.astype(np.complex64, copy=True)
                spec_full *= self._phase_term[:, None]
            else:
                spec_full = resamp.astype(np.complex64, copy=True)

        if cfg.debug_mode:
            self._debug_plot("5. Full Spectrum (Complex)", spec_full, is_complex=True, windows=(self.w1, self.w2), frame_idx=frame_idx, save_png=True)

        # 6) Recon target + two windowed inputs
        with timer("reconstruct B-scan", prof):
            target_full = recon_bscan_from_spectrum(spec_full, cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth)

        # Window spectra (allocate once per call; could reuse buffers if you guarantee no aliasing)
        with timer("apply spectral windows", prof):
            spec1 = spec_full * self.w1[:, None]
            spec2 = spec_full * self.w2[:, None]

        with timer("reconstruct windowed B-scans", prof):
            input_w1 = recon_bscan_from_spectrum(spec1.astype(np.complex64, copy=False), cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth)
            input_w2 = recon_bscan_from_spectrum(spec2.astype(np.complex64, copy=False), cfg.crop_depth, cfg.use_log, cfg.log_eps, cfg.apply_fftshift_depth)

        return {
            "target_full": target_full.astype(np.float32, copy=False),
            "input_w1": input_w1.astype(np.float32, copy=False),
            "input_w2": input_w2.astype(np.float32, copy=False),
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

        self._debug_out_dir = None
        self._dataset_name = None
        if out_npz is not None:
            self._debug_out_dir = os.path.dirname(out_npz)
            os.makedirs(self._debug_out_dir, exist_ok=True)
            # Dataset basename (used to prefix debug PNG filenames)
            self._dataset_name = os.path.splitext(os.path.basename(out_npz))[0]

        # Determine H,W by running first frame
        first = self.process_one(self.bscan_paths[0], frame_idx=0)
        H, W = first["target_full"].shape

        X = np.zeros((F, 2, H, W), dtype=np.float32)
        Y = np.zeros((F, 1, H, W), dtype=np.float32)

        for i, p in enumerate(self.bscan_paths):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing frame {i + 1}/{F}")
            out = self.process_one(p, frame_idx=i)
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
