"""Hybrid inter-frame registration for OCT stacks.

Strategy per frame (vs. reference):
  1. Coarse DFT-based phase-correlation shift.
  2. AKAZE feature-based transform (affine/Euclidean) with RANSAC.
  3. Optional ECC refinement.
  4. Quality-gated fallback: feature -> DFT -> identity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────
class TransformType(Enum):
    IDENTITY = "identity"
    DFT = "dft"
    FEATURE = "feature"
    ECC = "ecc"


@dataclass
class RegistrationConfig:
    """Knobs for the hybrid registration pipeline."""

    # Reference selection
    ref_index: Optional[int] = None   # None -> auto-select
    ref_strategy: str = "middle"      # "middle" or "sharpness"

    # DFT phase-correlation
    dft_upsample_factor: int = 10

    # AKAZE feature matching
    akaze_threshold: float = 0.001
    ratio_test: float = 0.75
    min_inliers: int = 10
    ransac_reproj: float = 5.0
    transform_model: str = "euclidean"  # "euclidean" or "affine"

    # ECC refinement
    use_ecc: bool = False
    ecc_iterations: int = 200
    ecc_eps: float = 1e-5
    ecc_motion: str = "euclidean"

    # Quality gates
    min_ncc_gain: float = -0.05
    max_translation: float = 100.0
    max_rotation_deg: float = 5.0

    # Interpolation
    interp: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: float = 0.0

    # Preprocessing toggle
    register_on_preprocessed: bool = True


@dataclass
class FrameResult:
    """Per-frame registration result and quality metrics."""

    index: int
    method: TransformType = TransformType.IDENTITY
    matrix: np.ndarray = field(
        default_factory=lambda: np.eye(2, 3, dtype=np.float32)
    )
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    ncc_before: float = 0.0
    ncc_after: float = 0.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    fallback_reason: str = ""


# ── Metrics ───────────────────────────────────────────────────────────────
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two images."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _sharpness(img: np.ndarray) -> float:
    """Laplacian variance -- proxy for image sharpness."""
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(np.var(lap))


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Convert [0,1] float32 to uint8 for OpenCV."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ── Reference selection ───────────────────────────────────────────────────
def select_reference(stack: np.ndarray, strategy: str = "middle") -> int:
    """Choose the reference frame index."""
    N = stack.shape[0]
    if strategy == "sharpness":
        scores = [_sharpness(stack[i]) for i in range(N)]
        idx = int(np.argmax(scores))
        logger.info(
            "Reference by sharpness: frame %d (score=%.2f)", idx, scores[idx]
        )
        return idx
    return N // 2


# ── DFT phase-correlation ────────────────────────────────────────────────
def _dft_register(
    ref: np.ndarray, mov: np.ndarray, upsample: int = 10
) -> Tuple[float, float, float]:
    """Sub-pixel translation via DFT phase correlation.

    Returns ``(dy, dx, peak_value)``.
    """
    from skimage.registration import phase_cross_correlation

    shift_yx, error, _phasediff = phase_cross_correlation(
        ref, mov, upsample_factor=upsample, normalization=None
    )
    return float(shift_yx[0]), float(shift_yx[1]), float(1.0 - error)


# ── AKAZE feature-based registration ─────────────────────────────────────
def _akaze_register(
    ref_u8: np.ndarray,
    mov_u8: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[Optional[np.ndarray], int, float]:
    """AKAZE keypoint matching -> robust transform.

    Returns ``(M_2x3 | None, inlier_count, inlier_ratio)``.
    """
    akaze = cv2.AKAZE_create(threshold=cfg.akaze_threshold)
    kp1, des1 = akaze.detectAndCompute(ref_u8, None)
    kp2, des2 = akaze.detectAndCompute(mov_u8, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des2, des1, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < cfg.ratio_test * n.distance:
                good.append(m)

    if len(good) < cfg.min_inliers:
        return None, len(good), 0.0

    pts_mov = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if cfg.transform_model == "affine":
        M, mask = cv2.estimateAffine2D(
            pts_mov, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_reproj,
        )
    else:
        M, mask = cv2.estimateAffinePartial2D(
            pts_mov, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_reproj,
        )

    if M is None or mask is None:
        return None, 0, 0.0

    inliers = int(mask.sum())
    ratio = inliers / len(good) if len(good) > 0 else 0.0
    return M, inliers, ratio


# ── ECC refinement ────────────────────────────────────────────────────────
_ECC_MOTION = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}


def _ecc_refine(
    ref_u8: np.ndarray,
    mov_u8: np.ndarray,
    init_M: np.ndarray,
    cfg: RegistrationConfig,
) -> Optional[np.ndarray]:
    """Run ECC optimization starting from *init_M*."""
    motion = _ECC_MOTION.get(cfg.ecc_motion, cv2.MOTION_EUCLIDEAN)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg.ecc_iterations,
        cfg.ecc_eps,
    )
    warp = init_M.copy().astype(np.float32)
    try:
        _, refined = cv2.findTransformECC(
            ref_u8, mov_u8, warp, motion, criteria, None, 5
        )
        return refined
    except cv2.error:
        logger.debug("ECC refinement failed; keeping previous transform.")
        return None


# ── Transform helpers ─────────────────────────────────────────────────────
def _decompose_affine(M: np.ndarray) -> Tuple[float, float, float]:
    """Extract (dx, dy, rotation_deg) from 2x3 affine matrix."""
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return dx, dy, rot


def _transform_ok(
    M: np.ndarray, cfg: RegistrationConfig
) -> Tuple[bool, str]:
    """Check whether the estimated transform is within sane limits."""
    dx, dy, rot = _decompose_affine(M)
    if abs(dx) > cfg.max_translation or abs(dy) > cfg.max_translation:
        return False, f"translation ({dx:.1f},{dy:.1f}) exceeds limit"
    if abs(rot) > cfg.max_rotation_deg:
        return False, f"rotation {rot:.2f} deg exceeds limit"
    return True, ""


def _warp_frame(
    img: np.ndarray, M: np.ndarray, cfg: RegistrationConfig
) -> np.ndarray:
    """Apply 2x3 affine warp to *img*.

    *M* is a forward transform (mov -> ref).  ``cv2.warpAffine``
    without ``WARP_INVERSE_MAP`` internally inverts *M* so it can
    sample backward from the destination into the source.
    """
    H, W = img.shape[:2]
    return cv2.warpAffine(
        img,
        M,
        (W, H),
        flags=cfg.interp,
        borderMode=cfg.border_mode,
        borderValue=cfg.border_value,
    )


def _shift_to_matrix(dy: float, dx: float) -> np.ndarray:
    """Build 2x3 translation matrix from (dy, dx) pixel shift."""
    M = np.eye(2, 3, dtype=np.float32)
    M[0, 2] = dx
    M[1, 2] = dy
    return M


# ── Single-frame registration ────────────────────────────────────────────
def register_one(
    ref_raw: np.ndarray,
    mov_raw: np.ndarray,
    ref_prep: np.ndarray,
    mov_prep: np.ndarray,
    idx: int,
    cfg: RegistrationConfig,
) -> FrameResult:
    """Register *mov* to *ref* using the hybrid pipeline.

    Detection runs on preprocessed copies; the returned transform is
    meant to be applied to the original-intensity image.
    """
    res = FrameResult(index=idx)
    res.ncc_before = _ncc(ref_prep, mov_prep)

    # Step 1: coarse DFT shift
    dy_dft, dx_dft, _ = _dft_register(
        ref_prep, mov_prep, cfg.dft_upsample_factor
    )
    M_dft = _shift_to_matrix(dy_dft, dx_dft)

    # Step 2: AKAZE feature-based
    ref_u8 = _to_u8(ref_prep)
    mov_u8 = _to_u8(mov_prep)
    M_feat, inliers, inlier_ratio = _akaze_register(ref_u8, mov_u8, cfg)

    # Step 3: collect candidates
    candidates: List[Tuple[np.ndarray, TransformType]] = []

    if M_feat is not None and inliers >= cfg.min_inliers:
        ok, reason = _transform_ok(M_feat, cfg)
        if ok:
            candidates.append((M_feat, TransformType.FEATURE))
            res.inlier_count = inliers
            res.inlier_ratio = inlier_ratio
        else:
            res.fallback_reason = f"feature rejected: {reason}"

    ok_dft, reason_dft = _transform_ok(M_dft, cfg)
    if ok_dft:
        candidates.append((M_dft, TransformType.DFT))

    candidates.append(
        (np.eye(2, 3, dtype=np.float32), TransformType.IDENTITY)
    )

    # Step 4: pick candidate with best NCC after warp
    best_M = candidates[-1][0]
    best_method = candidates[-1][1]
    best_ncc = -1.0

    for M_cand, method in candidates:
        warped = _warp_frame(mov_prep, M_cand, cfg)
        ncc_val = _ncc(ref_prep, warped)
        if ncc_val > best_ncc:
            best_ncc = ncc_val
            best_M = M_cand
            best_method = method

    # Step 5: optional ECC refinement
    if cfg.use_ecc and best_method != TransformType.IDENTITY:
        refined = _ecc_refine(ref_u8, mov_u8, best_M, cfg)
        if refined is not None:
            warped_ecc = _warp_frame(mov_prep, refined, cfg)
            ncc_ecc = _ncc(ref_prep, warped_ecc)
            if ncc_ecc > best_ncc:
                best_M = refined
                best_method = TransformType.ECC
                best_ncc = ncc_ecc

    # Step 6: quality gate
    ncc_gain = best_ncc - res.ncc_before
    if ncc_gain < cfg.min_ncc_gain and best_method != TransformType.IDENTITY:
        res.fallback_reason = (
            f"NCC gain {ncc_gain:.4f} < threshold; fallback to identity"
        )
        best_M = np.eye(2, 3, dtype=np.float32)
        best_method = TransformType.IDENTITY
        best_ncc = res.ncc_before

    dx, dy, rot = _decompose_affine(best_M)
    res.method = best_method
    res.matrix = best_M
    res.dx = dx
    res.dy = dy
    res.rotation_deg = rot
    res.ncc_after = best_ncc
    return res


# ── Full stack registration ──────────────────────────────────────────────
def register_stack(
    raw_stack: np.ndarray,
    prep_stack: np.ndarray,
    cfg: Optional[RegistrationConfig] = None,
) -> Tuple[np.ndarray, List[FrameResult]]:
    """Register all frames to a common reference.

    Parameters
    ----------
    raw_stack : [N, H, W] original-intensity images (float32).
    prep_stack : [N, H, W] preprocessed for detection.
    cfg : registration settings.

    Returns
    -------
    registered : [N, H, W] aligned raw-intensity stack.
    results : per-frame FrameResult list.
    """
    if cfg is None:
        cfg = RegistrationConfig()

    N = raw_stack.shape[0]
    ref_idx = (
        cfg.ref_index
        if cfg.ref_index is not None
        else select_reference(prep_stack, cfg.ref_strategy)
    )
    ref_idx = max(0, min(ref_idx, N - 1))
    logger.info("Reference frame: %d", ref_idx)

    ref_raw = raw_stack[ref_idx]
    ref_prep = prep_stack[ref_idx]

    registered = np.empty_like(raw_stack)
    results: List[FrameResult] = []

    for i in range(N):
        if i == ref_idx:
            registered[i] = raw_stack[i]
            res = FrameResult(index=i, method=TransformType.IDENTITY)
            res.ncc_before = 1.0
            res.ncc_after = 1.0
            results.append(res)
            continue

        res = register_one(
            ref_raw, raw_stack[i], ref_prep, prep_stack[i], i, cfg
        )
        registered[i] = _warp_frame(raw_stack[i], res.matrix, cfg)
        results.append(res)

        lvl = logging.WARNING if res.fallback_reason else logging.DEBUG
        logger.log(
            lvl,
            "Frame %3d  method=%-8s  dx=%+6.2f dy=%+6.2f rot=%+5.2f  "
            "NCC %.4f->%.4f  inliers=%d  %s",
            i, res.method.value, res.dx, res.dy, res.rotation_deg,
            res.ncc_before, res.ncc_after, res.inlier_count,
            res.fallback_reason,
        )

    return registered, results


def apply_transforms_to_stack(
    stack: np.ndarray,
    results: List[FrameResult],
    cfg: Optional[RegistrationConfig] = None,
) -> np.ndarray:
    """Apply previously computed transforms to a different stack.

    Useful for registering ground-truth images using transforms
    computed on predictions.

    Parameters
    ----------
    stack : [N, H, W] float32 array to warp.
    results : per-frame FrameResult from :func:`register_stack`.
    cfg : registration config (for interpolation settings).

    Returns
    -------
    [N, H, W] warped stack.
    """
    if cfg is None:
        cfg = RegistrationConfig()

    N = stack.shape[0]
    out = np.empty_like(stack)
    for i in range(N):
        fr = results[i]
        if fr.method == TransformType.IDENTITY:
            out[i] = stack[i]
        else:
            out[i] = _warp_frame(stack[i], fr.matrix, cfg)
    return out
