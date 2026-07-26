r"""Detector noise calibration from background frames (photon transfer curve).

Why this exists
---------------
The spectral-domain Recorrupted2Recorrupted component needs a calibrated noise
covariance to build provably unbiased training pairs. Getting it wrong makes the
corruption biased, and R2R's guarantee evaporates.

Two measurements on this data rule out the usual shortcuts:

1. `var/mean` is NOT constant. Measured on Maestro3 background stacks it climbs
   from ~0.24 at low signal to ~4.5 at high signal. Pure shot noise would give a
   flat ratio equal to the gain, so a Poisson-only model is wrong.
2. A straight line fit returns a NEGATIVE read-noise intercept (about -1518
   ADU^2), which is physically impossible and confirms the linear model does not
   describe the data.

The extra curvature is relative intensity noise (RIN) from the source, which is
multiplicative and therefore quadratic in mean. The model fitted here is

    var(mu) = read_var + gain * mu + rin * mu^2
              \_______/   \______/   \_______/
               readout      shot       source RIN

Calibration is PER ACQUISITION, not global: two Maestro3 folders measured gains
of 6.25 and 4.46. It is also distinct from the k-linearisation `.CLB`, which is
per instrument.

Why background frames
---------------------
`back*.raw` are captured with no sample light, so the only structure across the
spectral axis is the source envelope and fixed-pattern offset. That gives a wide
sweep of mean levels with NO object signal to contaminate the variance, which is
exactly what a photon transfer curve needs. Using sample frames instead would
fold real structural variation into the variance estimate and inflate the gain.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as sopt

# uint16 sensor: values approaching full scale are clipped and their variance
# collapses, which drags a fit badly if not excluded.
ADU_FULL_SCALE = 65535.0


@dataclass
class NoiseModel:
    """var(mu) = read_var + gain * mu + rin * mu^2, in ADU^2."""

    read_var: float
    gain: float
    rin: float

    # Diagnostics — carried so a bad fit is visible rather than silent.
    n_frames: int = 0
    n_bins_used: int = 0
    r_squared: float = float("nan")
    max_rel_residual: float = float("nan")
    mean_range: tuple[float, float] = (float("nan"), float("nan"))
    source: str = ""

    @property
    def read_noise_std(self) -> float:
        """Read noise in ADU. NaN if the fit produced a negative variance."""
        return float(np.sqrt(self.read_var)) if self.read_var > 0 else float("nan")

    def variance(self, mean: np.ndarray | float) -> np.ndarray | float:
        m = np.asarray(mean, dtype=np.float64)
        return self.read_var + self.gain * m + self.rin * m * m

    def std(self, mean: np.ndarray | float) -> np.ndarray | float:
        return np.sqrt(np.maximum(self.variance(mean), 0.0))

    def is_physical(self) -> bool:
        """A usable fit has non-negative read variance and gain.

        Phase 1's verification gate: a negative intercept means the model is
        still wrong, and the R2R corruption must not be built on it.
        """
        return self.read_var >= 0.0 and self.gain >= 0.0 and np.isfinite(self.rin)

    def summary(self) -> str:
        return (
            f"NoiseModel({self.source or 'unnamed'})\n"
            f"  var(mu) = {self.read_var:.2f} + {self.gain:.4f}*mu + {self.rin:.3e}*mu^2\n"
            f"  read noise      : {self.read_noise_std:.2f} ADU\n"
            f"  gain            : {self.gain:.4f} ADU/e-\n"
            f"  RIN coefficient : {self.rin:.3e}\n"
            f"  frames={self.n_frames}  bins={self.n_bins_used}  R2={self.r_squared:.5f}\n"
            f"  max |rel resid| : {self.max_rel_residual:.3f}\n"
            f"  mean range      : {self.mean_range[0]:.1f} .. {self.mean_range[1]:.1f} ADU\n"
            f"  physical        : {self.is_physical()}"
        )


@dataclass
class PTCCurve:
    """Binned photon transfer curve, kept for plotting and inspection."""

    mean: np.ndarray
    variance: np.ndarray
    used: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


def load_background_frames(
    data_dir: str, pixels: int, alines: int, pattern: str = "back*.raw"
) -> np.ndarray:
    """Load the background stack for one acquisition as [N, pixels, alines]."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No background frames matching {pattern!r} in {data_dir}. "
            f"Noise calibration requires dark frames captured with no sample light."
        )
    expected = pixels * alines
    frames = []
    for p in paths:
        a = np.fromfile(p, dtype=np.uint16)
        if a.size != expected:
            raise ValueError(
                f"{p}: expected {expected} uint16 samples for "
                f"pixels={pixels}, alines={alines}, got {a.size}"
            )
        frames.append(a.reshape((pixels, alines), order="F").astype(np.float64))
    return np.stack(frames)


def load_fpn(data_dir: str, pixels: int, alines: int) -> np.ndarray | None:
    """Load FPN.raw if present — the fixed-pattern offset reference.

    Measured to correlate 0.9989 with the background temporal mean, so either
    can serve as the offset; this is returned for cross-checking.
    """
    p = os.path.join(data_dir, "FPN.raw")
    if not os.path.isfile(p):
        return None
    a = np.fromfile(p, dtype=np.uint16)
    if a.size != pixels * alines:
        return None
    return a.reshape((pixels, alines), order="F").astype(np.float64)


def photon_transfer_curve(
    frames: np.ndarray,
    *,
    n_bins: int = 24,
    saturation_frac: float = 0.90,
    min_mean: float = 1.0,
) -> PTCCurve:
    """Per-pixel temporal mean/variance, binned by mean level.

    Variance is taken ACROSS frames at fixed pixel, so the source envelope and
    fixed-pattern offset — both constant in time — cancel out entirely.
    """
    if frames.ndim != 3 or frames.shape[0] < 3:
        raise ValueError(f"need at least 3 frames shaped [N, H, W], got {frames.shape}")

    mu = frames.mean(axis=0).ravel()
    var = frames.var(axis=0, ddof=1).ravel()

    keep = (mu >= min_mean) & (mu < saturation_frac * ADU_FULL_SCALE) & np.isfinite(var)
    mu, var = mu[keep], var[keep]
    if mu.size < n_bins * 4:
        raise ValueError(f"too few usable pixels after masking: {mu.size}")

    order = np.argsort(mu)
    mu, var = mu[order], var[order]

    edges = np.linspace(0, mu.size, n_bins + 1).astype(int)
    b_mean, b_var = [], []
    for k in range(n_bins):
        sl = slice(edges[k], edges[k + 1])
        if sl.stop > sl.start:
            b_mean.append(mu[sl].mean())
            # Median is deliberate: a handful of hot pixels in a bin would drag
            # the mean variance upward and bend the fitted curve.
            b_var.append(np.median(var[sl]))

    return PTCCurve(np.asarray(b_mean), np.asarray(b_var))


def fit_noise_model(
    frames: np.ndarray,
    *,
    n_bins: int = 24,
    saturation_frac: float = 0.90,
    trim_iters: int = 2,
    trim_sigma: float = 3.0,
    source: str = "",
) -> tuple[NoiseModel, PTCCurve]:
    """Fit var = read_var + gain*mu + rin*mu^2 to the photon transfer curve.

    Weighted least squares: a variance estimated from N frames has roughly
    constant RELATIVE error, so absolute error scales with the variance itself
    and the correct weight is 1/var^2. Unweighted fitting lets the high-signal
    bins dominate and is what drives the intercept negative.

    Outlier bins are trimmed iteratively — the topmost bin is typically
    contaminated by clipping at the spectrum edges even below the saturation cut.
    """
    curve = photon_transfer_curve(
        frames, n_bins=n_bins, saturation_frac=saturation_frac
    )
    m, v = curve.mean, curve.variance
    used = np.ones(m.size, dtype=bool)

    coef = np.zeros(3)
    for _ in range(max(trim_iters, 0) + 1):
        if used.sum() < 4:
            raise RuntimeError("too few PTC bins survived trimming to fit 3 parameters")
        A = np.vstack([np.ones_like(m[used]), m[used], m[used] ** 2]).T
        w = 1.0 / np.maximum(v[used], 1e-9) ** 2
        Aw = A * np.sqrt(w)[:, None]
        yw = v[used] * np.sqrt(w)
        # Non-negative least squares: all three terms are variances or
        # variance-per-unit-signal and cannot be negative. An unconstrained fit
        # returned gain = -0.012 on M3_Macula_3x3mm -- a good fit numerically
        # (R^2 0.992) but physically impossible. NNLS pins such a term at zero,
        # meaning "no shot contribution separable from RIN here", instead of
        # reporting a negative one.
        coef, _ = sopt.nnls(Aw, yw)

        pred_all = coef[0] + coef[1] * m + coef[2] * m * m
        rel = np.abs(v - pred_all) / np.maximum(pred_all, 1e-9)
        thresh = rel[used].mean() + trim_sigma * rel[used].std()
        new_used = used & (rel <= max(thresh, 1e-12))
        if new_used.sum() == used.sum():
            break
        used = new_used

    pred = coef[0] + coef[1] * m + coef[2] * m * m
    ss_res = float(np.sum((v[used] - pred[used]) ** 2))
    ss_tot = float(np.sum((v[used] - v[used].mean()) ** 2))
    curve.used = used

    model = NoiseModel(
        read_var=float(coef[0]),
        gain=float(coef[1]),
        rin=float(coef[2]),
        n_frames=int(frames.shape[0]),
        n_bins_used=int(used.sum()),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        max_rel_residual=float(np.max(np.abs(v[used] - pred[used]) / np.maximum(pred[used], 1e-9))),
        mean_range=(float(m.min()), float(m.max())),
        source=source,
    )
    return model, curve


def calibrate_folder(
    data_dir: str, pixels: int, alines: int, **fit_kw
) -> tuple[NoiseModel, PTCCurve]:
    """Calibrate one acquisition folder from its background frames."""
    frames = load_background_frames(data_dir, pixels, alines)
    return fit_noise_model(frames, source=os.path.basename(data_dir.rstrip("/\\")), **fit_kw)


def fit_pooled_noise_model(
    curves: dict[str, PTCCurve],
    *,
    n_frames: dict[str, int] | None = None,
) -> dict[str, NoiseModel]:
    """Fit a SHARED gain and RIN across acquisitions, with per-acquisition read_var.

        var_ij = read_var_i + gain * mu_ij + rin * mu_ij^2

    Linear in every parameter, so this is one exact weighted least-squares solve
    with an indicator column per acquisition plus two shared columns.

    THE MAESTRO3 DATA REJECTS THIS MODEL -- keep it for diagnosis, not for
    production calibration
    ----------------------------------------------------------------------
    The hypothesis was that gain scatter across acquisitions (0.016-0.061, with
    one negative) reflected fit degeneracy between the shot and RIN terms, since
    both grow with mean level and only ~9 dark frames are available.

    Pooling refuted it. Mean R^2 collapsed from 0.995 (independent fits) to
    0.323, with three acquisitions going NEGATIVE -- the shared-gain model fits
    worse than a constant. Detector gain is an adjustable acquisition setting on
    this instrument, which the Maestro2 folder names state outright
    ("...gain165", "...gain167"), so the spread is real and per-acquisition
    fitting is correct.

    Use this only to test whether a set of acquisitions shares a gain setting.
    A high pooled R^2 means they do; a collapse means they do not.
    """
    if not curves:
        raise ValueError("no curves supplied")

    names = list(curves)
    n_acq = len(names)

    rows, targets, weights, owner = [], [], [], []
    for i, name in enumerate(names):
        c = curves[name]
        used = c.used if c.used.size == c.mean.size else np.ones(c.mean.size, dtype=bool)
        m, v = c.mean[used], c.variance[used]
        for mj, vj in zip(m, v, strict=True):
            row = np.zeros(n_acq + 2)
            row[i] = 1.0          # this acquisition's read_var
            row[n_acq] = mj       # shared gain
            row[n_acq + 1] = mj * mj  # shared rin
            rows.append(row)
            targets.append(vj)
            # Constant relative error on a variance estimate -> absolute error
            # scales with the variance, so weight by 1/var^2.
            weights.append(1.0 / max(vj, 1e-9) ** 2)
            owner.append(i)

    A = np.asarray(rows)
    y = np.asarray(targets)
    w = np.sqrt(np.asarray(weights))
    if A.shape[0] < n_acq + 2:
        raise RuntimeError(f"only {A.shape[0]} bins for {n_acq + 2} parameters")

    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    gain, rin = float(coef[n_acq]), float(coef[n_acq + 1])

    pred = A @ coef
    owner_arr = np.asarray(owner)

    out: dict[str, NoiseModel] = {}
    for i, name in enumerate(names):
        sel = owner_arr == i
        ss_res = float(np.sum((y[sel] - pred[sel]) ** 2))
        ss_tot = float(np.sum((y[sel] - y[sel].mean()) ** 2))
        c = curves[name]
        out[name] = NoiseModel(
            read_var=float(coef[i]),
            gain=gain,
            rin=rin,
            n_frames=(n_frames or {}).get(name, 0),
            n_bins_used=int(sel.sum()),
            r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            max_rel_residual=float(
                np.max(np.abs(y[sel] - pred[sel]) / np.maximum(pred[sel], 1e-9))
            ),
            mean_range=(float(c.mean.min()), float(c.mean.max())),
            source=f"{name} (pooled)",
        )
    return out
