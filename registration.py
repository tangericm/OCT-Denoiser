"""
registration.py
===============
Python translation of registration.m + registerImages3.m

Robust subpixel rigid registration (translation + rotation) + frame
averaging for a multi-page OCT TIFF stack.

Install:
    pip install SimpleITK tifffile numpy scipy matplotlib

MATLAB → Python mapping
-----------------------
imread(file, m)                     tifffile.imread  → (N, H, W) float32
single()                            .astype(np.float32)
normxcorr2(template, image)         fftconvolve(image, template[::-1,::-1], 'full')
imregconfig('monomodal')            SetMetricAsCorrelation
                                    + SetOptimizerAsRegularStepGradientDescent
imregtform(...,'rigid',...,         ImageRegistrationMethod + Euler2DTransform
           'PyramidLevels',4)       + SetShrinkFactorsPerLevel([8,4,2,1])
imwarp(img,tform,'OutputView',...)  sitk.Resample(img, fixed_grid, tform, sitkBSpline3)
Fast_BigTiff_Write / WriteIMG       tifffile.TiffWriter(bigtiff=True) / tw.write(frame)
"""

import os
import numpy as np
import SimpleITK as sitk
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


# ──────────────────────────────────────────────────────────────────────
#  USER SETTINGS
# ──────────────────────────────────────────────────────────────────────
IN_DIR = r'C:\Users\erict\OneDrive\Desktop\Projects\OCT Denoiser\runs\A-Line\1D_npatch=256\predictions_tiff\6mm_1024Aline'
FILE1  = 'gt_6mm_1024Aline_s005_g060.tiff'
FILE2  = 'pred_6mm_1024Aline_s005_g060.tiff'

# Frame count: None = auto-detect from TIFF (recommended).
# Set to an integer (e.g. 64) to process a fixed number of frames.
N_FRAMES = None

# Row crop: MATLAB gt(130:600,:,:) → Python [129:600]  (0-based, end-exclusive)
ROW_START = 129
ROW_END   = 600


# ──────────────────────────────────────────────────────────────────────
#  HELPER: min-max normalise to [0, 1]
#  Replicates the NaN/Inf handling + normalisation block in registerImages3.m
# ──────────────────────────────────────────────────────────────────────
def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.copy().astype(np.float32)
    img[np.isnan(img)]    = 0.0
    img[np.isposinf(img)] = 1.0
    img[np.isneginf(img)] = 0.0
    lo, hi = float(img.min()), float(img.max())
    if lo == hi:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────
#  HELPER: phase-correlation pre-alignment
#  Replicates the normxcorr2 block in registerImages3.m
# ──────────────────────────────────────────────────────────────────────
def _phase_correlation_shift(fixed: np.ndarray, moving: np.ndarray,
                              max_shift_frac: float = 0.10):
    """
    Estimate pixel shift (tx, ty) of MOVING relative to FIXED via
    cross-correlation of the centre horizontal strip.

    fftconvolve(A, B[::-1, ::-1], 'full') == cross-correlation of A with B
    which is the Python equivalent of MATLAB normxcorr2(B, A).

    Returns (tx, ty) clamped to ±(max_shift_frac * max(H, W)).
    """
    n_rows = fixed.shape[0]
    r1, r2 = n_rows // 4, 3 * n_rows // 4

    f_strip = fixed [r1:r2, :]
    m_strip = moving[r1:r2, :]

    c = fftconvolve(m_strip, f_strip[::-1, ::-1], mode='full')
    ypeak, xpeak = np.unravel_index(np.argmax(np.abs(c)), c.shape)

    # fftconvolve 'full' pads output by (template_size - 1)
    ty = int(ypeak) - (f_strip.shape[0] - 1)
    tx = int(xpeak) - (f_strip.shape[1] - 1)

    max_shift = round(max_shift_frac * max(fixed.shape))
    return int(np.clip(tx, -max_shift, max_shift)), \
           int(np.clip(ty, -max_shift, max_shift))


# ──────────────────────────────────────────────────────────────────────
#  CORE: rigid registration  (≡ registerImages3.m)
# ──────────────────────────────────────────────────────────────────────
def register_images(moving_np: np.ndarray, fixed_np: np.ndarray) -> dict:
    """
    Rigid (translation + rotation) registration of MOVING onto FIXED.

    Parameters
    ----------
    moving_np, fixed_np : (H, W) float32 numpy arrays

    Returns
    -------
    dict:
        'RegisteredImage'  – (H, W) float32, warped normalised MOVING
        'Transformation'   – sitk.Transform  (Euler2DTransform)
    """
    # normalise to [0, 1]
    fixed_n  = _normalize(fixed_np)
    moving_n = _normalize(moving_np)

    # SimpleITK images  (must be float32 for registration framework)
    # GetImageFromArray takes (row, col); SimpleITK physical coords are (x=col, y=row)
    fixed_s  = sitk.GetImageFromArray(fixed_n)
    moving_s = sitk.GetImageFromArray(moving_n)

    # coarse translation via phase correlation
    tx_px, ty_px = _phase_correlation_shift(fixed_n, moving_n)

    # initial Euler2DTransform
    #   center of rotation = geometric centre of fixed image
    #   angle              = 0
    #   translation        = phase-corr shift  (negated: SimpleITK maps fixed → moving)
    h, w = fixed_np.shape
    init_tf = sitk.Euler2DTransform()
    init_tf.SetCenter(((w - 1) / 2.0, (h - 1) / 2.0))
    init_tf.SetAngle(0.0)
    init_tf.SetTranslation((-float(tx_px), -float(ty_px)))

    # ── registration method ──────────────────────────────────────────
    reg = sitk.ImageRegistrationMethod()

    # Metric: Normalised Correlation  ←→  MATLAB imregconfig('monomodal')
    reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.NONE)   # use all pixels

    # Interpolator: cubic B-spline  ←→  MATLAB 'cubic' in imwarp
    reg.SetInterpolator(sitk.sitkBSpline3)

    # Optimizer: RegularStepGradientDescent  ←→  MATLAB optimizer
    # All values mirror registerImages3.m exactly
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate               = 6.25e-03,   # MaximumStepLength
        minStep                    = 1e-07,       # MinimumStepLength
        numberOfIterations         = 400,         # MaximumIterations
        relaxationFactor           = 0.5,         # RelaxationFactor
        gradientMagnitudeTolerance = 1e-06,       # GradientMagnitudeTolerance
        estimateLearningRate       = reg.Never,
    )

    # Auto-scale angle vs translation DOFs so each takes a similar-sized step.
    # Python equivalent of MATLAB imregtform's automatic DOF scaling.
    reg.SetOptimizerScalesFromPhysicalShift()

    # 4-level pyramid  ←→  MATLAB PyramidLevels=4
    reg.SetShrinkFactorsPerLevel( [8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(init_tf, inPlace=False)

    try:
        tform = reg.Execute(fixed_s, moving_s)
    except Exception as exc:
        print(f'  [WARNING] Registration failed ({exc}). Using initial transform.')
        tform = init_tf

    # Warp  ←→  imwarp(MOVING, tform, 'OutputView', imref2d(size(FIXED)))
    registered = sitk.Resample(
        moving_s, fixed_s, tform,
        sitk.sitkBSpline3, 0.0, moving_s.GetPixelID(),
    )

    return {
        'RegisteredImage': sitk.GetArrayFromImage(registered).astype(np.float32),
        'Transformation' : tform,
    }


# ──────────────────────────────────────────────────────────────────────
#  HELPER: apply a pre-computed transform to a raw (unnormalised) image
#  ←→  imwarp(gt_crop(:,:,m), tform, 'OutputView', imref2d(size(ref)))
# ──────────────────────────────────────────────────────────────────────
def _apply_transform(image_np: np.ndarray,
                     tform: sitk.Transform,
                     reference_np: np.ndarray) -> np.ndarray:
    img_s = sitk.GetImageFromArray(image_np.astype(np.float32))
    ref_s = sitk.GetImageFromArray(reference_np.astype(np.float32))
    out_s = sitk.Resample(img_s, ref_s, tform,
                          sitk.sitkBSpline3, 0.0, img_s.GetPixelID())
    return sitk.GetArrayFromImage(out_s).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    # read full stacks as (N, H, W) float32
    print('Reading stacks ...')
    gt_stack   = tifffile.imread(os.path.join(IN_DIR, FILE1)).astype(np.float32)
    pred_stack = tifffile.imread(os.path.join(IN_DIR, FILE2)).astype(np.float32)

    # handle single-frame TIFF edge case
    if gt_stack.ndim == 2:
        gt_stack, pred_stack = gt_stack[np.newaxis], pred_stack[np.newaxis]

    # auto-detect or cap frame count
    n = gt_stack.shape[0] if N_FRAMES is None else min(N_FRAMES, gt_stack.shape[0])
    print(f'Processing {n} frames (detected {gt_stack.shape[0]} in file).')

    # row-crop  ←→  MATLAB gt(130:600,:,:)
    gt_crop   = gt_stack  [:n, ROW_START:ROW_END, :]
    pred_crop = pred_stack[:n, ROW_START:ROW_END, :]

    # display first frame before registration
    plt.figure()
    plt.imshow(gt_crop[0], cmap='gray')
    plt.title('gt_crop – frame 0 (before registration)')
    plt.axis('off'); plt.tight_layout(); plt.show(block=False)

    # reference = first pred frame  ←→  MATLAB ref = pred_crop(:,:,1)
    ref      = pred_crop[0]
    gt_reg   = gt_crop.copy()
    pred_reg = pred_crop.copy()

    # output paths  ←→  MATLAB [inDir '\' file(1:end-5) '_registered.tiff']
    out_gt   = os.path.join(IN_DIR, os.path.splitext(FILE1)[0] + '_registered3.tiff')
    out_pred = os.path.join(IN_DIR, os.path.splitext(FILE2)[0] + '_registered3.tiff')

    # register frame-by-frame; write incrementally to BigTIFF
    # TiffWriter(bigtiff=True)          ←→  Fast_BigTiff_Write
    # tw.write(frame, contiguous=True)  ←→  WriteIMG (sequential pages)
    print(f'Registering {n} frames ...')
    with tifffile.TiffWriter(out_gt,   bigtiff=True) as tw_gt, \
         tifffile.TiffWriter(out_pred, bigtiff=True) as tw_pred:

        for m in range(n):
            res          = register_images(pred_crop[m], ref)
            tform        = res['Transformation']
            pred_reg[m]  = res['RegisteredImage']
            gt_reg[m]    = _apply_transform(gt_crop[m], tform, ref)

            if m == 0 or (m + 1) % 10 == 0:
                print(f'  frame {m + 1}/{n}')

            tw_gt  .write(gt_reg  [m], contiguous=True, photometric='minisblack')
            tw_pred.write(pred_reg[m], contiguous=True, photometric='minisblack')

    print(f'Saved: {out_gt}')
    print(f'Saved: {out_pred}')

    # mean-averaged result  ←→  MATLAB imagesc(mean(gt_reg, 3))
    plt.figure()
    plt.imshow(gt_reg.mean(axis=0), cmap='gray')
    plt.title('Mean registered GT')
    plt.axis('off'); plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()