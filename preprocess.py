from __future__ import annotations

import os
import glob
from typing import TYPE_CHECKING, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
import scipy.linalg as sla

if TYPE_CHECKING:
    from configs.default import FolderSpec

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

    # Cache the LAPACK gttrs solver function once (avoids per-call lookup)
    gttrs, = sla.lapack.get_lapack_funcs(("gttrs",), (d_f,))

    # Precompute column-vector (2D) forms for broadcasting over alines
    a_col = a[:, None].copy()
    b_col = b[:, None].copy()
    a3ma_col = a3ma[:, None].copy()
    b3mb_col = b3mb[:, None].copy()

    return {
        "ab": ab,          # banded system matrix for M
        "h2_over_6": c,
        "i0": i0,
        "a": a,
        "b": b,
        "a3ma": a3ma,
        "b3mb": b3mb,
        "a_col": a_col,       # [pixels, 1] for broadcasting
        "b_col": b_col,
        "a3ma_col": a3ma_col,
        "b3mb_col": b3mb_col,
        "inv_h2_6": np.float32(6.0 / (h * h)),  # rhs scale
        "n": n,
        "dl_f": dl_f,
        "d_f": d_f,
        "du_f": du_f,
        "du2": du2,
        "ipiv": ipiv,
        "gttrs": gttrs,     # cached LAPACK solver
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
    # Use precomputed LAPACK gttrs solver (cached in precompute dict)
    M, info = pre["gttrs"](pre["dl_f"], pre["d_f"], pre["du_f"], pre["du2"], pre["ipiv"], rhs, trans='N')
    if info != 0:
        raise RuntimeError(f"gttrs failed with info={info}")
    M = M.astype(np.float32, copy=False)

    # --- Evaluate spline at xp using precomputed indices/weights ---
    i0 = pre["i0"]
    i1 = i0 + 1

    # Gather y_i, y_{i+1}, M_i, M_{i+1}
    y0 = spec[i0, :]    # [pixels, alines]
    y1 = spec[i1, :]
    m0 = M[i0, :]
    m1 = M[i1, :]

    # Use precomputed column-vector forms (avoids per-call reshape)
    out = (pre["a_col"] * y0 + pre["b_col"] * y1
           + (pre["a3ma_col"] * m0 + pre["b3mb_col"] * m1) * pre["h2_over_6"]).astype(np.float32)
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
    depth_c = sfft.ifft(spec_complex, axis=0, workers=-1)  # [pixels, alines] complex
    mag = np.abs(depth_c).astype(np.float32)     # [pixels, alines]

    if apply_fftshift_depth:
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


def recon_bscan_batch(spec_batch: np.ndarray,
                      crop: Tuple[int, int],
                      use_log: bool,
                      log_eps: float,
                      apply_fftshift_depth: bool,
                      return_stats: bool = False) -> Any:
    """
    Batch-reconstruct multiple spectra in a single FFT call.

    spec_batch: complex64 [K, pixels, alines] — K spectra to reconstruct
    Returns list of K float32 B-scans, each [H, W].
    If return_stats is True, returns tuple:
      (bscans, stats)
    where stats is a list of K dicts with keys: img_norm, mu, sd.

    Uses a single batched IFFT along axis=1 (spectral axis) for all K
    spectra simultaneously, reducing FFT overhead.
    """
    # Batched IFFT: axis=1 is the spectral axis for [K, pixels, alines]
    depth_c = sfft.ifft(spec_batch, axis=1, workers=-1)
    mag = np.abs(depth_c).astype(np.float32)  # [K, pixels, alines]

    if apply_fftshift_depth:
        mag = sfft.fftshift(mag, axes=1).astype(np.float32)

    z0, z1 = crop
    bscans_crop = mag[:, z0:z1, :]  # [K, H, W]

    results = []
    stats = []
    for k in range(bscans_crop.shape[0]):
        bscan = bscans_crop[k]  # [H, W]
        if use_log:
            bscan = np.log10(bscan + log_eps).astype(np.float32)
        mu = float(bscan.mean())
        sd = float(bscan.std()) + 1e-6
        bscan_norm = ((bscan - mu) / sd).astype(np.float32)
        results.append(bscan_norm)
        if return_stats:
            stats.append({
                "img_norm": bscan_norm,
                "mu": mu,
                "sd": sd,
            })
    if return_stats:
        return results, stats
    return results


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



# -----------------------------
# High-level processor
# -----------------------------
class BscanProcessor:
    def __init__(self, folder_spec: FolderSpec):
        """
        folder_spec: a FolderSpec containing root_folder, data_folder, pixels,
            alines, crop_depth, and all spectral processing parameters.
        """
        cfg = folder_spec
        root_folder = cfg.root_folder
        self.root = root_folder
        self.cfg = cfg

        # Find data folder using config parameter
        self.data_dir = os.path.join(root_folder, cfg.data_folder)
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        # Find CLB (either in data_dir or root_folder)
        self.clb_path = None
        clbs = sorted(glob.glob(os.path.join(self.data_dir, "*.CLB")))
        if len(clbs) == 1:
            self.clb_path = clbs[0]
        elif len(clbs) > 1:
            raise FileNotFoundError(
                f"Multiple .CLB files found in dataset folder {self.data_dir}: {clbs}\n"
            )
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
        # Pre-allocated batch buffer for 3-way batched FFT [full, w1, w2]
        self._spec_batch_c64 = np.empty((3, cfg.pixels, cfg.alines), dtype=np.complex64)

    def save_window_figure(self, out_path: str, bscan_path: Optional[str] = None) -> None:
        cfg = self.cfg
        if bscan_path is None:
            if not self.bscan_paths:
                raise ValueError("No bscan paths available to build a representative spectrum.")
            bscan_path = self.bscan_paths[0]

        data = np.fromfile(bscan_path, dtype=np.uint16)
        expected = cfg.pixels * cfg.alines
        if data.size != expected:
            raise ValueError(f"{os.path.basename(bscan_path)} has {data.size} elements; expected {expected}.")

        raw = data.reshape((cfg.pixels, cfg.alines), order="F").astype(np.float32, copy=False)
        if cfg.do_dc_subtract:
            raw[0, :] = raw[3, :]
            raw[1, :] = raw[3, :]
            raw[2, :] = raw[3, :]
            raw = raw - raw.mean(axis=1, keepdims=True)

        resamp = resample_klinear_cubic_operator(raw, self._spline_pre)
        # resamp *= self.apod[:, None]

        spec_full = resamp.astype(np.complex64, copy=False)
        if self._phase_term is not None:
            spec_full = spec_full.copy()
            spec_full *= self._phase_term[:, None]

        aline_idx = spec_full.shape[1] // 2
        spectrum_mag = np.abs(spec_full[:, aline_idx])
        spectrum_max = float(np.max(spectrum_mag)) if spectrum_mag.size else 1.0
        spectrum_norm = spectrum_mag / max(spectrum_max, 1e-12)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(spectrum_norm, color="steelblue", linewidth=0.9, label="Spectrum (center A-line)")
        ax.plot(self.w1, color="red", alpha=0.8, label="Window 1")
        ax.plot(self.w2, color="orange", alpha=0.8, label="Window 2")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(f"window_sigma={cfg.window_sigma:.4f}  gap={cfg.gap:.4f}")
        ax.set_ylim([-0.05, 1.1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        fig.tight_layout()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    def process_one(self, bscan_path: str, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Process a single raw B-scan file into denoising inputs and target.

        Returns dict with keys:
          target_full, input_w1, input_w2: [H,W] float32
          target_mu, target_sd, input_w1_mu, input_w1_sd, input_w2_mu, input_w2_sd: float
        """
        cfg = self.cfg
        pixels = cfg.pixels
        alines = cfg.alines

        # 1) Read raw spectrum
        data = np.fromfile(bscan_path, dtype=np.uint16)
        expected = pixels * alines
        if data.size != expected:
            raise ValueError(f"{os.path.basename(bscan_path)} has {data.size} elements; expected {expected}.")

        raw = data.reshape((pixels, alines), order="F").astype(np.float32, copy=False)

        # 2) DC subtract
        if cfg.do_dc_subtract:
            raw[0, :] = raw[3, :]
            raw[1, :] = raw[3, :]
            raw[2, :] = raw[3, :]
            raw = raw - raw.mean(axis=1, keepdims=True)

        # 3) Resample to k-linear (cubic spline, vectorized across A-lines)
        resamp = resample_klinear_cubic_operator(raw, self._spline_pre)

        # 4) Apodization
        # resamp *= self.apod[:, None]

        # 5) Build complex spectrum (with optional dispersion)
        if self._phase_term is not None:
            spec_full = resamp.astype(np.complex64, copy=True)
            spec_full *= self._phase_term[:, None]
        else:
            spec_full = resamp.astype(np.complex64, copy=True)

        # 6) Recon target + two windowed inputs via batched FFT
        np.multiply(spec_full, self.w1[:, None], out=self._spec1_c64)
        np.multiply(spec_full, self.w2[:, None], out=self._spec2_c64)

        self._spec_batch_c64[0] = spec_full
        self._spec_batch_c64[1] = self._spec1_c64
        self._spec_batch_c64[2] = self._spec2_c64
        batch_imgs, batch_stats = recon_bscan_batch(
            self._spec_batch_c64,
            cfg.crop_depth,
            cfg.use_log,
            cfg.log_eps,
            cfg.apply_fftshift_depth,
            return_stats=True,
        )
        target_full, input_w1, input_w2 = batch_imgs
        target_stats, input_w1_stats, input_w2_stats = batch_stats

        return {
            "target_full": target_full,
            "input_w1": input_w1,
            "input_w2": input_w2,
            "target_mu": target_stats["mu"],
            "target_sd": target_stats["sd"],
            "input_w1_mu": input_w1_stats["mu"],
            "input_w1_sd": input_w1_stats["sd"],
            "input_w2_mu": input_w2_stats["mu"],
            "input_w2_sd": input_w2_stats["sd"],
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
            from utils.io_tiff import save_tiff_stack

            os.makedirs(os.path.dirname(out_npz), exist_ok=True)
            np.savez_compressed(out_npz, X=X, Y=Y)
            print(f"[OK] Saved dataset: {out_npz}  X={X.shape} Y={Y.shape}")

            base_dir = os.path.dirname(out_npz)
            for label, stack in [("window1", X[:, 0]), ("window2", X[:, 1]), ("target", Y[:, 0])]:
                path = os.path.join(base_dir, f"{cfg.data_folder}_{label}.tiff")
                save_tiff_stack(path, stack, dtype="uint8")
                print(f"[OK] Saved {label} stack: {path} (shape: {stack.shape})")

        return {"X": X, "Y": Y}
