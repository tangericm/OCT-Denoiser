import os
import numpy as np
import tifffile as tiff
from .run_manager import ensure_dir

def _robust_scale(img: np.ndarray, out_max: float, p_lo: float, p_hi: float) -> np.ndarray:
    lo, hi = np.percentile(img, [p_lo, p_hi])
    img = (img - lo) / (hi - lo + 1e-6)
    img = np.clip(img, 0.0, 1.0)
    return img * out_max

def save_tiff_stack(
    path: str,
    stack: np.ndarray,
    *,
    imagej: bool = True,
    dtype: str = "uint16",          # "uint8" | "uint16" | "float32"
    scale_per_slice: bool = True,   # only relevant for uint8/uint16
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    overwrite: bool = True,
) -> None:
    if os.path.exists(path) and (not overwrite):
        raise FileExistsError(path)

    stack = np.asarray(stack)

    # Normalize shape -> [P,H,W] pages
    if stack.ndim == 4:
        n, c, h, w = stack.shape
        pages = stack.reshape(n * c, h, w)
    elif stack.ndim == 3:
        pages = stack
    elif stack.ndim == 2:
        pages = stack[None, ...]
    else:
        raise ValueError(f"Unsupported stack shape: {stack.shape}")

    if dtype == "float32":
        out = pages.astype(np.float32)
    elif dtype == "uint8":
        if scale_per_slice:
            out = np.stack([_robust_scale(p, 255.0, p_lo, p_hi).astype(np.uint8) for p in pages], axis=0)
        else:
            lo, hi = np.percentile(pages, [p_lo, p_hi])
            norm = np.clip((pages - lo) / (hi - lo + 1e-6), 0, 1)
            out = (norm * 255.0).astype(np.uint8)
    elif dtype == "uint16":
        if scale_per_slice:
            out = np.stack([_robust_scale(p, 65535.0, p_lo, p_hi).astype(np.uint16) for p in pages], axis=0)
        else:
            lo, hi = np.percentile(pages, [p_lo, p_hi])
            norm = np.clip((pages - lo) / (hi - lo + 1e-6), 0, 1)
            out = (norm * 65535.0).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    ensure_dir(os.path.dirname(path) or ".")
    tiff.imwrite(
        path,
        out,
        imagej=imagej,
        photometric="minisblack",
        metadata={"axes": "ZYX"} if imagej else None,
    )
