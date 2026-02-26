"""I/O utilities for loading / saving OCT image stacks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Supported extensions in priority order
_IMG_EXTS = {".tiff", ".tif", ".png", ".npy"}


def discover_images(directory: str | Path) -> List[Path]:
    """Return sorted list of image paths in *directory*.

    Supports multi-page TIFF stacks (returns the single file) and directories
    of individual slices (TIFF, PNG, NPY).
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Input directory not found: {d}")

    files = sorted(
        p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {d} (looked for {_IMG_EXTS})")
    return files


def load_stack(directory: str | Path) -> Tuple[np.ndarray, List[str]]:
    """Load all images in *directory* into a float32 ``[N, H, W]`` array.

    Returns ``(stack, filenames)`` where *filenames* preserves ordering.
    Multi-page TIFFs are expanded into individual frames.
    """
    import tifffile

    files = discover_images(directory)
    frames: List[np.ndarray] = []
    names: List[str] = []

    for p in files:
        ext = p.suffix.lower()
        if ext in {".tiff", ".tif"}:
            img = tifffile.imread(str(p))
            if img.ndim == 3:
                for i in range(img.shape[0]):
                    frames.append(img[i].astype(np.float32))
                    names.append(f"{p.name}[{i}]")
            else:
                frames.append(img.astype(np.float32))
                names.append(p.name)
        elif ext == ".npy":
            arr = np.load(str(p)).astype(np.float32)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    frames.append(arr[i])
                    names.append(f"{p.name}[{i}]")
            else:
                frames.append(arr)
                names.append(p.name)
        else:
            from skimage.io import imread as sk_imread

            img = sk_imread(str(p))
            if img.ndim == 3:
                img = np.mean(img[:, :, :3], axis=2)
            frames.append(img.astype(np.float32))
            names.append(p.name)

    stack = np.stack(frames, axis=0)
    logger.info("Loaded %d frames  shape=%s  dtype=float32", len(frames), stack.shape)
    return stack, names


def save_stack(
    stack: np.ndarray,
    outdir: str | Path,
    prefix: str = "reg",
    dtype: str = "float32",
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> Path:
    """Save ``[N, H, W]`` array as a multi-page TIFF stack.

    Parameters
    ----------
    stack : array [N, H, W]
    outdir : destination directory (created if needed)
    prefix : filename prefix
    dtype : "float32", "uint16", or "uint8"
    p_lo, p_hi : percentile bounds for integer scaling

    Returns
    -------
    Path to the written file.
    """
    import tifffile

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{prefix}.tiff"

    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]

    if dtype == "float32":
        out = arr
    else:
        out_max = 65535.0 if dtype == "uint16" else 255.0
        lo = np.percentile(arr, p_lo)
        hi = np.percentile(arr, p_hi)
        scaled = np.clip((arr - lo) / (hi - lo + 1e-7), 0.0, 1.0) * out_max
        out = scaled.astype(np.uint16 if dtype == "uint16" else np.uint8)

    tifffile.imwrite(
        str(path),
        out,
        imagej=True,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )
    logger.info("Saved %s  shape=%s  dtype=%s", path, out.shape, out.dtype)
    return path
