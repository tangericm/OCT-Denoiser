import numpy as np
import cv2

def reg_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Convert to uint8, normalize, optionally enhance contrast for better keypoints.
    """
    x = img.astype(np.float32, copy=False)
    x = x - np.percentile(x, 1)
    x = x / (np.percentile(x, 99) - np.percentile(x, 1) + 1e-8)
    x = np.clip(x, 0, 1)
    u8 = (x * 255).astype(np.uint8)

    # CLAHE helps a lot on OCT-like imagery
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)

    # Optional: emphasize edges (uncomment if needed)
    # u8 = cv2.GaussianBlur(u8, (0, 0), 1.0)
    # u8 = cv2.Canny(u8, 30, 90)

    return u8

def estimate_similarity_orb(moving: np.ndarray, fixed: np.ndarray,
                            *, max_features: int = 4000,
                            good_match_frac: float = 0.25,
                            ransac_reproj_thresh: float = 3.0,
                            scale_locked: bool = False):
    """
    Returns 2x3 warp matrix M mapping moving -> fixed, plus diagnostics.
    """
    mov = reg_preprocess(moving)
    fix = reg_preprocess(fixed)

    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(mov, None)   # moving
    kp2, des2 = orb.detectAndCompute(fix, None)   # fixed

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None, {"ok": False, "reason": "not_enough_keypoints"}

    # Hamming matcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return None, {"ok": False, "reason": "not_enough_matches"}

    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(8, int(len(matches) * good_match_frac))
    matches = matches[:keep]

    pts_m = np.float32([kp1[m.queryIdx].pt for m in matches])  # moving points
    pts_f = np.float32([kp2[m.trainIdx].pt for m in matches])  # fixed points

    # Similarity/partial affine with RANSAC
    M, inliers = cv2.estimateAffinePartial2D(
        pts_m, pts_f, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh
    )
    if M is None:
        return None, {"ok": False, "reason": "ransac_failed"}

    inlier_count = int(inliers.sum()) if inliers is not None else 0

    if scale_locked:
        # Project M to pure rigid (rotation + translation, no scale)
        A = M[:, :2]
        t = M[:, 2]
        # SVD-based closest rotation
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        M = np.hstack([R.astype(np.float32), t.reshape(2, 1).astype(np.float32)])

    return M.astype(np.float32), {"ok": True, "inliers": inlier_count, "matches": len(matches)}

def warp_frame(img: np.ndarray, M: np.ndarray, out_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_shape_hw
    return cv2.warpAffine(
        img.astype(np.float32, copy=False),
        M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )

def register_stack_keypoints_to_pred0(pred_stack: np.ndarray, gt_stack: np.ndarray,
                                      *, scale_locked: bool = True):
    assert pred_stack.shape == gt_stack.shape and pred_stack.ndim == 3
    n, h, w = pred_stack.shape

    pred_ref = pred_stack[0].astype(np.float32, copy=False)

    pred_reg = np.zeros_like(pred_stack, dtype=np.float32)
    gt_reg   = np.zeros_like(gt_stack, dtype=np.float32)

    pred_reg[0] = pred_stack[0]
    gt_reg[0]   = gt_stack[0]

    rows = [{"frame": 0, "ok": True, "inliers": 0, "matches": 0,
             "m00": 1.0, "m01": 0.0, "m02": 0.0, "m10": 0.0, "m11": 1.0, "m12": 0.0,
             "reason": ""}]

    for i in range(1, n):
        M, info = estimate_similarity_orb(pred_stack[i], pred_ref, scale_locked=scale_locked)

        if M is None:
            # Fallback: identity (or you can add your phase-correlation fallback here)
            pred_reg[i] = pred_stack[i]
            gt_reg[i]   = gt_stack[i]
            rows.append({"frame": i, "ok": False, "inliers": 0, "matches": 0,
                         "m00": 1.0, "m01": 0.0, "m02": 0.0, "m10": 0.0, "m11": 1.0, "m12": 0.0,
                         "reason": info.get("reason", "unknown")})
            continue

        pred_reg[i] = warp_frame(pred_stack[i], M, (h, w))
        gt_reg[i]   = warp_frame(gt_stack[i],   M, (h, w))

        rows.append({"frame": i, "ok": True,
                     "inliers": info.get("inliers", 0),
                     "matches": info.get("matches", 0),
                     "m00": float(M[0,0]), "m01": float(M[0,1]), "m02": float(M[0,2]),
                     "m10": float(M[1,0]), "m11": float(M[1,1]), "m12": float(M[1,2]),
                     "reason": ""})

    return pred_reg, gt_reg, rows

