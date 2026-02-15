"""
Tests for the image registration module (engine/registration.py).

Run:  python tests/test_registration.py

Tests use synthetic images — including OCT-like frames with a bright tissue
band surrounded by dark background — to verify that the registration pipeline
recovers orientation, rotation, and translation within acceptable tolerances.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import numpy as np
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift

# Load registration module directly from file to avoid engine/__init__.py
# pulling in torch-dependent modules (allows running this test without torch).
_reg_path = os.path.join(os.path.dirname(__file__), "..", "engine", "registration.py")
_spec = importlib.util.spec_from_file_location("engine.registration", _reg_path)
_reg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reg
_spec.loader.exec_module(_reg)

register_frame = _reg.register_frame
register_stack = _reg.register_stack
apply_registration_to_stack = _reg.apply_registration_to_stack
save_registration_csv = _reg.save_registration_csv
save_registration_json = _reg.save_registration_json
FrameRegistrationResult = _reg.FrameRegistrationResult
_ncc = _reg._ncc
_apply_orientation = _reg._apply_orientation
_apply_transform = _reg._apply_transform
_detect_tissue_rows = _reg._detect_tissue_rows
_edge_map = _reg._edge_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(shape: tuple[int, int] = (128, 128), seed: int = 42) -> np.ndarray:
    """Create a synthetic test image with multiple Gaussian blobs.

    Produces enough texture for phase cross-correlation to work reliably.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    img = np.zeros((H, W), dtype=np.float64)
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]

    # Place 5-8 blobs at random positions
    n_blobs = rng.integers(5, 9)
    for _ in range(n_blobs):
        cy = rng.uniform(H * 0.2, H * 0.8)
        cx = rng.uniform(W * 0.2, W * 0.8)
        sigma = rng.uniform(5, 15)
        amplitude = rng.uniform(0.5, 1.0)
        img += amplitude * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))

    # Add a bit of noise so it's not too smooth
    img += rng.normal(0, 0.02, (H, W))
    return img.astype(np.float32)


def _make_oct_like_image(
    shape: tuple[int, int] = (512, 256),
    tissue_rows: tuple[int, int] = (150, 350),
    seed: int = 42,
    noise_level: float = 0.02,
) -> np.ndarray:
    """Create a synthetic OCT-like B-scan: bright tissue band + dark noise.

    The tissue band contains layered horizontal structures (mimicking retinal
    layers).  The rest of the image is near-zero dark background with noise.
    This is the pattern that defeated the old full-frame registration.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    t_y0, t_y1 = tissue_rows
    img = np.zeros((H, W), dtype=np.float64)

    # Dark background noise
    img += rng.normal(0, noise_level, (H, W))

    # Tissue region: layered horizontal bands
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    n_layers = 6
    tissue_height = t_y1 - t_y0
    for k in range(n_layers):
        # Each layer is a horizontally-stretched Gaussian band with some curvature
        center_y = t_y0 + tissue_height * (k + 0.5) / n_layers
        # Slight curvature: center varies sinusoidally across columns
        curve = 3.0 * np.sin(2 * np.pi * xs / W)
        sigma_y = tissue_height / (n_layers * 3.0)
        amplitude = 0.3 + 0.5 * rng.random()
        layer = amplitude * np.exp(-((ys - center_y - curve) ** 2) / (2 * sigma_y ** 2))
        img += layer

    # Add some speckle noise within the tissue band
    tissue_noise = rng.normal(0, 0.08, (tissue_height, W))
    img[t_y0:t_y1, :] += tissue_noise

    img = np.clip(img, 0, None)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 1) Identity registration (pred == ref)
# ---------------------------------------------------------------------------
def test_registration_identity():
    """When pred equals ref, registration should produce zero transform."""
    print("=" * 60)
    print("TEST 1: Identity Registration (pred == ref)")
    print("=" * 60)

    ref = _make_test_image()
    registered, result = register_frame(ref, ref, frame_idx=0)

    ok_angle = abs(result.refined_angle_deg) < 1.0
    ok_shift = abs(result.dy) < 1.0 and abs(result.dx) < 1.0
    ok_score = result.score > 0.90
    ok_success = result.success
    ok_no_flip = not result.flip_lr

    all_ok = ok_angle and ok_shift and ok_score and ok_success and ok_no_flip
    print(f"  angle={result.refined_angle_deg:.2f} (expect ~0)  [{'PASS' if ok_angle else 'FAIL'}]")
    print(f"  shift=({result.dy:.2f}, {result.dx:.2f}) (expect ~0)  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.90)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  success={result.success}  [{'PASS' if ok_success else 'FAIL'}]")
    print(f"  flip_lr={result.flip_lr} (expect False)  [{'PASS' if ok_no_flip else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 2) Known translation
# ---------------------------------------------------------------------------
def test_registration_translation():
    """A known translation should be recovered within 0.5 pixel."""
    print("=" * 60)
    print("TEST 2: Known Translation Recovery")
    print("=" * 60)

    ref = _make_test_image(shape=(128, 128), seed=123)
    true_dy, true_dx = 5.0, -3.0
    pred = ndi_shift(ref, (-true_dy, -true_dx), order=3, mode="constant", cval=0.0)

    registered, result = register_frame(pred, ref, frame_idx=0, refine_angles=False)

    # The registration should find shift ≈ (true_dy, true_dx)
    dy_err = abs(result.dy - true_dy)
    dx_err = abs(result.dx - true_dx)
    ok_shift = dy_err < 0.5 and dx_err < 0.5
    ok_score = result.score > 0.85
    ok_angle = abs(result.refined_angle_deg) < 1.0

    all_ok = ok_shift and ok_score and ok_angle
    print(f"  true_shift=({true_dy}, {true_dx})")
    print(f"  found_shift=({result.dy:.2f}, {result.dx:.2f})  err=({dy_err:.3f}, {dx_err:.3f})  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.85)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  angle={result.refined_angle_deg:.2f} (expect ~0)  [{'PASS' if ok_angle else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 3) 180-degree rotation
# ---------------------------------------------------------------------------
def test_registration_rotation_180():
    """A 180-degree rotation should be recovered."""
    print("=" * 60)
    print("TEST 3: 180-Degree Rotation Recovery")
    print("=" * 60)

    ref = _make_test_image(shape=(128, 128), seed=77)
    pred = np.rot90(ref, k=2)

    registered, result = register_frame(pred, ref, frame_idx=0)

    ok_orient = result.orientation_deg == 180
    ok_score = result.score > 0.85
    ok_shift = abs(result.dy) < 1.0 and abs(result.dx) < 1.0

    all_ok = ok_orient and ok_score and ok_shift
    print(f"  orientation_deg={result.orientation_deg} (expect 180)  [{'PASS' if ok_orient else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.85)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  shift=({result.dy:.2f}, {result.dx:.2f}) (expect ~0)  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 4) 90-degree rotation (square images only)
# ---------------------------------------------------------------------------
def test_registration_rotation_90():
    """Undoing a 90-degree CCW rotation requires applying 270 CCW to pred."""
    print("=" * 60)
    print("TEST 4: 90-Degree Rotation Recovery (Square Image)")
    print("=" * 60)

    ref = _make_test_image(shape=(128, 128), seed=55)
    pred = np.rot90(ref, k=1)  # pred = ref rotated 90 CCW

    registered, result = register_frame(pred, ref, frame_idx=0)

    # To undo 90 CCW, we must apply 270 CCW (=90 CW) to pred.
    ok_orient = result.orientation_deg == 270
    ok_score = result.score > 0.85
    ok_shift = abs(result.dy) < 1.0 and abs(result.dx) < 1.0

    all_ok = ok_orient and ok_score and ok_shift
    print(f"  orientation_deg={result.orientation_deg} (expect 270)  [{'PASS' if ok_orient else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.85)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  shift=({result.dy:.2f}, {result.dx:.2f}) (expect ~0)  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 5) Combined rotation + translation
# ---------------------------------------------------------------------------
def test_registration_combined():
    """Combined 180-degree rotation + translation should be recovered."""
    print("=" * 60)
    print("TEST 5: Combined Rotation + Translation")
    print("=" * 60)

    ref = _make_test_image(shape=(128, 128), seed=99)
    # Apply: rotate 180 then shift
    pred = np.rot90(ref, k=2)
    true_dy, true_dx = 4.0, -2.0
    pred = ndi_shift(pred, (-true_dy, -true_dx), order=3, mode="constant", cval=0.0)

    registered, result = register_frame(pred, ref, frame_idx=0)

    # The registration should recover the 180 orientation
    ok_orient = result.orientation_deg == 180
    ok_score = result.score > 0.80
    ok_success = result.success

    # The NCC of the registered result with ref should be high
    ncc_registered = _ncc(registered, ref)
    ok_ncc = ncc_registered > 0.80

    all_ok = ok_orient and ok_score and ok_success and ok_ncc
    print(f"  orientation_deg={result.orientation_deg} (expect 180)  [{'PASS' if ok_orient else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.80)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  NCC(registered, ref)={ncc_registered:.4f} (expect >0.80)  [{'PASS' if ok_ncc else 'FAIL'}]")
    print(f"  success={result.success}  [{'PASS' if ok_success else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 6) Left-right flip
# ---------------------------------------------------------------------------
def test_registration_flip():
    """A horizontal flip should be recovered."""
    print("=" * 60)
    print("TEST 6: Horizontal Flip Recovery")
    print("=" * 60)

    ref = _make_test_image(shape=(128, 128), seed=33)
    pred = ref[:, ::-1].copy()

    registered, result = register_frame(pred, ref, frame_idx=0, include_flips=True)

    ok_flip = result.flip_lr is True
    ok_score = result.score > 0.85
    ok_shift = abs(result.dy) < 1.0 and abs(result.dx) < 1.0

    all_ok = ok_flip and ok_score and ok_shift
    print(f"  flip_lr={result.flip_lr} (expect True)  [{'PASS' if ok_flip else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.85)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  shift=({result.dy:.2f}, {result.dx:.2f}) (expect ~0)  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 7) Low-texture frame (graceful failure)
# ---------------------------------------------------------------------------
def test_registration_low_texture():
    """A nearly-constant frame should fail gracefully."""
    print("=" * 60)
    print("TEST 7: Low-Texture Frame (Graceful Failure)")
    print("=" * 60)

    ref = np.ones((64, 64), dtype=np.float32) * 0.5
    pred = np.ones((64, 64), dtype=np.float32) * 0.5

    registered, result = register_frame(pred, ref, frame_idx=0)

    ok_no_success = not result.success
    ok_note = result.note == "low_texture"
    ok_identity = np.allclose(registered, pred)

    all_ok = ok_no_success and ok_note and ok_identity
    print(f"  success={result.success} (expect False)  [{'PASS' if ok_no_success else 'FAIL'}]")
    print(f"  note={result.note!r} (expect 'low_texture')  [{'PASS' if ok_note else 'FAIL'}]")
    print(f"  returned original pred: {ok_identity}  [{'PASS' if ok_identity else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 8) Stack registration
# ---------------------------------------------------------------------------
def test_registration_stack():
    """Stack registration should work on multiple frames."""
    print("=" * 60)
    print("TEST 8: Stack Registration (3 Frames)")
    print("=" * 60)

    F = 3
    refs = np.stack([_make_test_image(seed=i * 10) for i in range(F)])
    preds = np.stack([
        refs[0],                                       # identity
        np.rot90(refs[1], k=2),                        # 180
        ndi_shift(refs[2], (-3.0, 2.0), order=3),     # translation
    ])

    reg_stack, results = register_stack(preds, refs, refine_angles=False)

    ok_len = len(results) == F
    ok_all_success = all(r.success for r in results)

    # Verify registration quality: NCC of each registered frame with ref
    nccs = [_ncc(reg_stack[i], refs[i]) for i in range(F)]
    ok_nccs = all(n > 0.80 for n in nccs)

    all_ok = ok_len and ok_all_success and ok_nccs
    print(f"  result count={len(results)} (expect {F})  [{'PASS' if ok_len else 'FAIL'}]")
    print(f"  all success={ok_all_success}  [{'PASS' if ok_all_success else 'FAIL'}]")
    for i, n in enumerate(nccs):
        print(f"  NCC frame {i}: {n:.4f}")
    print(f"  all NCC > 0.80: {ok_nccs}  [{'PASS' if ok_nccs else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 9) apply_registration_to_stack
# ---------------------------------------------------------------------------
def test_apply_registration_to_stack():
    """Applying registration transforms to a stack should produce valid output."""
    print("=" * 60)
    print("TEST 9: Apply Registration to Stack")
    print("=" * 60)

    F = 2
    refs = np.stack([_make_test_image(seed=i * 20 + 100) for i in range(F)])
    preds = np.stack([
        ndi_shift(refs[0], (-2.0, 3.0), order=3),
        np.rot90(refs[1], k=2),
    ])

    _, results = register_stack(preds, refs, refine_angles=False)

    # Apply same transforms to a different stack (e.g., GT)
    other_stack = np.stack([_make_test_image(seed=i * 20 + 200) for i in range(F)])
    transformed = apply_registration_to_stack(other_stack, results)

    ok_shape = transformed.shape == other_stack.shape
    ok_dtype = transformed.dtype == other_stack.dtype
    ok_finite = np.all(np.isfinite(transformed))

    all_ok = ok_shape and ok_dtype and ok_finite
    print(f"  shape preserved: {ok_shape}  [{'PASS' if ok_shape else 'FAIL'}]")
    print(f"  dtype preserved: {ok_dtype}  [{'PASS' if ok_dtype else 'FAIL'}]")
    print(f"  all finite: {ok_finite}  [{'PASS' if ok_finite else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 10) CSV/JSON output
# ---------------------------------------------------------------------------
def test_registration_io():
    """Registration CSV and JSON output should be valid files."""
    print("=" * 60)
    print("TEST 10: Registration CSV/JSON Output")
    print("=" * 60)

    results = [
        FrameRegistrationResult(
            frame_idx=0, orientation_deg=0, flip_lr=False,
            refined_angle_deg=0.5, dy=1.2, dx=-0.3,
            score=0.95, success=True, tissue_y0=100, tissue_y1=400,
        ),
        FrameRegistrationResult(
            frame_idx=1, orientation_deg=180, flip_lr=False,
            refined_angle_deg=180.0, dy=0.0, dx=0.0,
            score=0.88, success=True, tissue_y0=100, tissue_y1=400,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "reg.csv")
        json_path = os.path.join(tmpdir, "reg.json")

        save_registration_csv(csv_path, results)
        save_registration_json(json_path, results)

        ok_csv = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        ok_json = os.path.isfile(json_path) and os.path.getsize(json_path) > 0

        # Verify CSV content
        if ok_csv:
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            ok_csv = (
                len(rows) == 2
                and "frame_idx" in rows[0]
                and "score" in rows[0]
                and "tissue_y0" in rows[0]
            )

        # Verify JSON content
        if ok_json:
            import json
            with open(json_path) as f:
                data = json.load(f)
            ok_json = (
                len(data) == 2
                and data[0]["frame_idx"] == 0
                and data[0]["tissue_y0"] == 100
            )

    all_ok = ok_csv and ok_json
    print(f"  CSV valid: {ok_csv}  [{'PASS' if ok_csv else 'FAIL'}]")
    print(f"  JSON valid: {ok_json}  [{'PASS' if ok_json else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 11) Non-square image (90/270 skipped)
# ---------------------------------------------------------------------------
def test_registration_nonsquare():
    """For non-square images, 90/270 orientations should be skipped gracefully."""
    print("=" * 60)
    print("TEST 11: Non-Square Image Registration")
    print("=" * 60)

    ref = _make_test_image(shape=(96, 128), seed=44)
    # Apply 180 rotation (which preserves shape for non-square)
    pred = np.rot90(ref, k=2)

    registered, result = register_frame(pred, ref, frame_idx=0)

    ok_orient = result.orientation_deg == 180
    ok_score = result.score > 0.85
    ok_shape = registered.shape == ref.shape

    all_ok = ok_orient and ok_score and ok_shape
    print(f"  orientation_deg={result.orientation_deg} (expect 180)  [{'PASS' if ok_orient else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.85)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  shape preserved: {ok_shape}  [{'PASS' if ok_shape else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 12) Tissue band detection
# ---------------------------------------------------------------------------
def test_tissue_detection():
    """_detect_tissue_rows should find the tissue band in an OCT-like image."""
    print("=" * 60)
    print("TEST 12: Tissue Band Detection")
    print("=" * 60)

    true_y0, true_y1 = 150, 350
    img = _make_oct_like_image(shape=(512, 256), tissue_rows=(true_y0, true_y1), seed=42)

    det_y0, det_y1 = _detect_tissue_rows(img)

    # Detection should be within ±20 rows of the true tissue band
    ok_y0 = abs(det_y0 - true_y0) < 20
    ok_y1 = abs(det_y1 - true_y1) < 20
    # Detected band must cover most of the true band
    ok_coverage = det_y0 <= true_y0 + 20 and det_y1 >= true_y1 - 20

    all_ok = ok_y0 and ok_y1 and ok_coverage
    print(f"  true tissue: [{true_y0}:{true_y1}]")
    print(f"  detected:    [{det_y0}:{det_y1}]")
    print(f"  y0 error: {abs(det_y0 - true_y0)} (expect <20)  [{'PASS' if ok_y0 else 'FAIL'}]")
    print(f"  y1 error: {abs(det_y1 - true_y1)} (expect <20)  [{'PASS' if ok_y1 else 'FAIL'}]")
    print(f"  coverage OK: {ok_coverage}  [{'PASS' if ok_coverage else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 13) OCT-like registration with translation (critical test)
# ---------------------------------------------------------------------------
def test_oct_translation():
    """Registration should detect translation in an OCT-like image with
    a thin tissue band and dominant dark background.

    This is the scenario that failed with the old full-frame approach:
    phase correlation was dominated by the dark background, always
    returning (0,0) shift.
    """
    print("=" * 60)
    print("TEST 13: OCT-Like Image — Translation Recovery")
    print("=" * 60)

    tissue_rows = (150, 350)
    ref = _make_oct_like_image(shape=(512, 256), tissue_rows=tissue_rows, seed=42)

    true_dy, true_dx = 8.0, -5.0
    pred = ndi_shift(ref, (-true_dy, -true_dx), order=3, mode="constant", cval=0.0)

    # Add slightly different noise to pred (simulating denoised vs noisy)
    rng = np.random.default_rng(999)
    pred = pred + rng.normal(0, 0.01, pred.shape).astype(np.float32)

    registered, result = register_frame(pred, ref, frame_idx=0, refine_angles=False)

    dy_err = abs(result.dy - true_dy)
    dx_err = abs(result.dx - true_dx)
    ok_shift = dy_err < 1.5 and dx_err < 1.5
    ok_score = result.score > 0.5
    ok_success = result.success
    # Verify the shift is NOT (0,0) — that was the old bug
    ok_nonzero = abs(result.dy) > 0.5 or abs(result.dx) > 0.5

    all_ok = ok_shift and ok_score and ok_success and ok_nonzero
    print(f"  true_shift=({true_dy}, {true_dx})")
    print(f"  found_shift=({result.dy:.2f}, {result.dx:.2f})  err=({dy_err:.3f}, {dx_err:.3f})  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  non-zero shift detected: {ok_nonzero}  [{'PASS' if ok_nonzero else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.5)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  tissue=[{result.tissue_y0}:{result.tissue_y1}]")
    print(f"  success={result.success}  [{'PASS' if ok_success else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 14) OCT-like registration with different noise levels
# ---------------------------------------------------------------------------
def test_oct_noise_difference():
    """Registration should work when pred is denoised and ref is noisy.

    Edge enhancement should handle the noise-level difference by focusing
    on structural boundaries shared by both images.
    """
    print("=" * 60)
    print("TEST 14: OCT-Like Image — Different Noise Levels")
    print("=" * 60)

    tissue_rows = (150, 350)
    # "Ground truth" = noisy version
    ref = _make_oct_like_image(shape=(512, 256), tissue_rows=tissue_rows, seed=42, noise_level=0.05)

    # "Prediction" = denoised version with a known shift
    clean = _make_oct_like_image(shape=(512, 256), tissue_rows=tissue_rows, seed=42, noise_level=0.005)
    true_dy, true_dx = 6.0, -4.0
    pred = ndi_shift(clean, (-true_dy, -true_dx), order=3, mode="constant", cval=0.0)

    registered, result = register_frame(pred, ref, frame_idx=0, refine_angles=False)

    dy_err = abs(result.dy - true_dy)
    dx_err = abs(result.dx - true_dx)
    ok_shift = dy_err < 2.0 and dx_err < 2.0
    ok_score = result.score > 0.3
    ok_success = result.success
    ok_nonzero = abs(result.dy) > 0.5 or abs(result.dx) > 0.5

    all_ok = ok_shift and ok_score and ok_success and ok_nonzero
    print(f"  true_shift=({true_dy}, {true_dx})")
    print(f"  found_shift=({result.dy:.2f}, {result.dx:.2f})  err=({dy_err:.3f}, {dx_err:.3f})  [{'PASS' if ok_shift else 'FAIL'}]")
    print(f"  non-zero shift: {ok_nonzero}  [{'PASS' if ok_nonzero else 'FAIL'}]")
    print(f"  score={result.score:.4f} (expect >0.3)  [{'PASS' if ok_score else 'FAIL'}]")
    print(f"  success={result.success}  [{'PASS' if ok_success else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# 15) OCT-like registration with explicit tissue_roi parameter
# ---------------------------------------------------------------------------
def test_oct_explicit_roi():
    """Passing an explicit tissue_roi should produce equivalent results."""
    print("=" * 60)
    print("TEST 15: OCT-Like Image — Explicit tissue_roi")
    print("=" * 60)

    tissue_rows = (150, 350)
    ref = _make_oct_like_image(shape=(512, 256), tissue_rows=tissue_rows, seed=42)

    true_dy, true_dx = 5.0, 3.0
    pred = ndi_shift(ref, (-true_dy, -true_dx), order=3, mode="constant", cval=0.0)

    # Use auto-detect
    _, result_auto = register_frame(pred, ref, frame_idx=0, refine_angles=False)
    # Use explicit ROI
    _, result_expl = register_frame(
        pred, ref, frame_idx=0, refine_angles=False,
        tissue_roi=tissue_rows,
    )

    # Both should find a non-trivial shift close to truth
    dy_err_auto = abs(result_auto.dy - true_dy)
    dx_err_auto = abs(result_auto.dx - true_dx)
    dy_err_expl = abs(result_expl.dy - true_dy)
    dx_err_expl = abs(result_expl.dx - true_dx)

    ok_auto = dy_err_auto < 1.5 and dx_err_auto < 1.5 and result_auto.success
    ok_expl = dy_err_expl < 1.5 and dx_err_expl < 1.5 and result_expl.success

    all_ok = ok_auto and ok_expl
    print(f"  Auto-detect: shift=({result_auto.dy:.2f}, {result_auto.dx:.2f})  "
          f"err=({dy_err_auto:.3f}, {dx_err_auto:.3f})  [{'PASS' if ok_auto else 'FAIL'}]")
    print(f"  Explicit ROI: shift=({result_expl.dy:.2f}, {result_expl.dx:.2f})  "
          f"err=({dy_err_expl:.3f}, {dx_err_expl:.3f})  [{'PASS' if ok_expl else 'FAIL'}]")
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}

    results["identity"] = test_registration_identity()
    results["translation"] = test_registration_translation()
    results["rotation_180"] = test_registration_rotation_180()
    results["rotation_90"] = test_registration_rotation_90()
    results["combined"] = test_registration_combined()
    results["flip"] = test_registration_flip()
    results["low_texture"] = test_registration_low_texture()
    results["stack"] = test_registration_stack()
    results["apply_stack"] = test_apply_registration_to_stack()
    results["io"] = test_registration_io()
    results["nonsquare"] = test_registration_nonsquare()
    results["tissue_detect"] = test_tissue_detection()
    results["oct_translation"] = test_oct_translation()
    results["oct_noise_diff"] = test_oct_noise_difference()
    results["oct_explicit_roi"] = test_oct_explicit_roi()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    sys.exit(0 if all_ok else 1)
