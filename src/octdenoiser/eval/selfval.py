r"""Held-out-mask self-validation — the primary model-selection metric.

The problem it solves
---------------------
There is no clean reference for OCT retina data. The existing validation loss is
computed on random frames drawn from the SAME acquisition as training, and
adjacent B-scans are spatially correlated, so it is optimistic. The four
Maestro2 50-frame averages are the only near-clean references available and
cover just four line positions — far too narrow to select models on.

The argument
------------
Partition the k-axis into three mutually disjoint masks. Train `recon(m1)` to
predict `recon(m2)`, then score against a third view `recon(m3)` that was never
used in training. Writing x_i = s + eta_i, where s is the underlying structure
and eta_i is that view's speckle-plus-detector noise:

    E|| f(x1) - x3 ||^2  =  E|| f(x1) - s ||^2  +  E|| eta_3 ||^2
                            \_______________/     \____________/
                              what we want         constant in f

The cross term vanishes because eta_3 is zero-mean and independent of x1, which
disjoint masks guarantee: detector noise is independent across k, and speckle
correlation scales with k-support overlap (measured -0.022 for disjoint binary
masks, versus 0.975 for repeat B-scans of the same tissue).

The second term does not depend on the model, so **ranking models by this score
matches ranking them by true MSE against a clean reference** — without ever
having one. That is the entire point, and `rank_agreement` in the tests
demonstrates it rather than assuming it.

IMPORTANT: this is a RANKING tool, not an absolute quality score. Its value
carries the irreducible E||eta_3||^2 offset, so it must never be reported as if
it were PSNR against clean data.

Two measured limits
-------------------
1. It needs enough frames. The metric has finite-sample variance, so models
   within that variance of each other can swap. On a single 64x48 frame, true
   MSEs of 0.00708 and 0.00745 (5% apart) scored 0.07120 and 0.07102 (0.25%
   apart) and inverted. Aggregated over 32 frames, ranking matched the clean
   reference exactly. Validate over enough frames to resolve close candidates.

2. `vs_identity` is a weak floor, not a usefulness test. The identity carries
   2*sigma^2 -- its own noise plus the held-out view's -- so nearly any
   smoothing beats it: a ruinous sigma=12 blur still scored 0.888. Only a
   degenerate constant output exceeds 1.0. Select on `held_out_mse`; treat
   `vs_identity` as a tripwire for degenerate outputs only.

Determinism
-----------
Validation masks are drawn from a FIXED seed set. Redrawing them each epoch
would inject mask noise into the very signal used for early stopping and
checkpoint selection, and would make two models incomparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from octdenoiser.physics.masks import make_mask_partition
from octdenoiser.preprocess import recon_bscan_batch

# Index convention within a partition. Training consumes INPUT and TARGET; the
# validator scores against HELD_OUT, which training never sees.
INPUT_VIEW = 0
TARGET_VIEW = 1
HELD_OUT_VIEW = 2


@dataclass
class ValidationViews:
    """Three disjoint reconstructions of one frame."""

    input_view: np.ndarray
    target_view: np.ndarray
    held_out_view: np.ndarray
    seed: int

    @property
    def shape(self) -> tuple[int, ...]:
        return self.input_view.shape


def make_validation_views(
    spectrum: np.ndarray,
    *,
    seed: int,
    crop: tuple[int, int],
    use_log: bool = True,
    log_eps: float = 1e-6,
    apply_fftshift_depth: bool = False,
    fft_workers: int = 1,
) -> ValidationViews:
    """Split a k-linearised spectrum into three disjoint views and reconstruct.

    `spectrum` is [pixels, alines], complex, AFTER k-linearisation — i.e. the
    `spec_full` that `BscanProcessor.process_one` builds. Masks apply along the
    spectral axis.

    All three reconstructions go through a single batched IFFT, per the codebase
    rule that multi-spectrum reconstruction never loops over the single-spectrum
    path.
    """
    if spectrum.ndim != 2:
        raise ValueError(f"expected [pixels, alines], got {spectrum.shape}")

    pixels = spectrum.shape[0]
    masks = make_mask_partition(pixels, 3, seed=seed)

    batch = np.empty((3, *spectrum.shape), dtype=np.complex64)
    for i, m in enumerate(masks):
        np.multiply(spectrum, m[:, None], out=batch[i])

    views = recon_bscan_batch(
        batch, crop, use_log, log_eps, apply_fftshift_depth, fft_workers=fft_workers
    )
    return ValidationViews(
        input_view=views[INPUT_VIEW],
        target_view=views[TARGET_VIEW],
        held_out_view=views[HELD_OUT_VIEW],
        seed=seed,
    )


def held_out_mse(prediction: np.ndarray, held_out_view: np.ndarray) -> float:
    """Mean squared error against the held-out view.

    Lower is better, and the ordering is meaningful; the absolute value is not,
    because it includes the irreducible noise term.
    """
    if prediction.shape != held_out_view.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape} vs held-out {held_out_view.shape}"
        )
    d = prediction.astype(np.float64) - held_out_view.astype(np.float64)
    return float(np.mean(d * d))


def held_out_psnr_proxy(prediction: np.ndarray, held_out_view: np.ndarray) -> float:
    """A dB-scaled monotone transform of `held_out_mse`, for readable logs.

    Monotone, so it ranks identically. NOT PSNR against clean data — the noise
    floor makes the absolute number meaningless on its own.
    """
    mse = held_out_mse(prediction, held_out_view)
    if mse <= 0:
        return float("inf")
    data_range = float(held_out_view.max() - held_out_view.min())
    if data_range <= 0:
        return float("nan")
    return 10.0 * np.log10((data_range**2) / mse)


class SelfValidator:
    """Fixed-seed held-out-mask validation across a set of frames.

    Seeds are derived once from `base_seed` and reused for every evaluation, so
    the score is comparable across epochs and across models. Redrawing masks per
    call would add variance to the early-stopping signal.
    """

    def __init__(self, n_frames: int, *, base_seed: int = 20260101) -> None:
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        rng = np.random.default_rng(base_seed)
        self.seeds: list[int] = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_frames)]
        self.base_seed = base_seed

    def seed_for(self, frame_index: int) -> int:
        return self.seeds[frame_index % len(self.seeds)]

    def views_for(self, spectrum: np.ndarray, frame_index: int, **kw) -> ValidationViews:
        return make_validation_views(spectrum, seed=self.seed_for(frame_index), **kw)

    def score(self, predictions: list[np.ndarray], views: list[ValidationViews]) -> dict[str, float]:
        """Aggregate over frames. Returns the ranking metric plus diagnostics."""
        if len(predictions) != len(views):
            raise ValueError(f"{len(predictions)} predictions vs {len(views)} view sets")
        if not predictions:
            raise ValueError("nothing to score")

        per_frame = [held_out_mse(p, v.held_out_view) for p, v in zip(predictions, views, strict=True)]
        # The identity baseline — feeding the input view straight through —
        # is the floor any useful model must beat.
        identity = [held_out_mse(v.input_view, v.held_out_view) for v in views]

        mse = float(np.mean(per_frame))
        return {
            "held_out_mse": mse,
            "held_out_psnr_proxy": float(np.mean([
                held_out_psnr_proxy(p, v.held_out_view)
                for p, v in zip(predictions, views, strict=True)
            ])),
            "identity_mse": float(np.mean(identity)),
            # >1 means the model is worse than doing nothing.
            "vs_identity": mse / max(float(np.mean(identity)), 1e-12),
            "n_frames": float(len(per_frame)),
            "std": float(np.std(per_frame)),
        }
