"""Frame-pair Noise2Noise supervision from OCTA volumes.

Why this exists
---------------
The spectral schemes all pay for their decorrelation. Contiguous sub-bands
decorrelate speckle essentially perfectly (+0.003) but cost 2.43x the axial PSF
width and carry a 0.269 chromatic signal mismatch, because the two bands sit at
different k and scattering is wavelength-dependent. Interleaved random masks
preserve the PSF (1.01x) but barely decorrelate speckle (+0.257) -- the same
property that keeps the bandwidth is what keeps the speckle.

Pairing two FRAMES avoids the trade entirely: both sides are full bandwidth,
same spectrum, no chromatic bias.

The M3 "x2" volumes are OCTA acquisitions storing interleaved repeat pairs:

    index 2k     -> position k, repeat 0
    index 2k + 1 -> position k, repeat 1

so 512 positions x 2 repeats = 1024 files, matching the file count exactly.

Which pairing? Position, not repeat
-----------------------------------
Counterintuitively, the repeat pairs are the WORSE choice. Measured with
structure removed, across three volumes:

    pairing                        speckle corr   structure corr
    repeats     (2k, 2k+1)              0.105          0.963
    positions   (i,  i+2)               0.017          0.958

Noise2Noise wants structure shared and speckle NOT shared. Repeats revisit the
same scatterers, so their speckle only decorrelates as far as motion carries it.
Different positions see different scatterers. Structure correlation is
essentially equal, so position pairing wins outright on the axis that matters.

Both are exposed via `pair_mode` so the comparison stays an ablation rather than
an assumption.

Residual bias
-------------
Position pairing is not free: paired positions are 11.7 um apart (512 positions
across 6 mm), so the structure is not identical and E[s2|x1] != s1. Measured,
the structure difference is 26% of the speckle variance being removed, and the
input/target coherence -- the transfer function MSE training converges to --
holds to a finer in-plane scale than either spectral scheme (16.7 px period,
against 32.9 for sub-bands and 37.3 for repeat frames).
"""
from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np

from octdenoiser.data.dataset import RawBscanDataset, _to_torch_float32

PAIR_MODES = ("position", "repeat")


class PairedFrameDataset(RawBscanDataset):
    """Noise2Noise over frame pairs drawn from one acquisition.

    Input and target are both full-bandwidth reconstructions of DIFFERENT
    frames, so neither contains the other and no spectral splitting is involved.

    pair_mode:
      "position" -- pair index i with i + repeats_per_position*position_step,
                    i.e. the same repeat index one or more positions away.
                    Default, and the measured best.
      "repeat"   -- pair the two repeats at one position, (2k, 2k+1).

    Reuses the parent's lazy per-worker init, LRU frame cache, cropping and
    augmentation. Only the index and item construction differ.
    """


    def __init__(
        self,
        folder_specs: list[Any],
        split: str,
        train_frac: float,
        *,
        pair_mode: str = "position",
        position_step: int = 1,
        repeats_per_position: int = 2,
        **kw,
    ):
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}, got {pair_mode!r}")
        if position_step < 1:
            raise ValueError(f"position_step must be >= 1, got {position_step}")
        if repeats_per_position < 1:
            raise ValueError(f"repeats_per_position must be >= 1, got {repeats_per_position}")
        if pair_mode == "repeat" and repeats_per_position < 2:
            raise ValueError('pair_mode="repeat" needs repeats_per_position >= 2')

        self.pair_mode = pair_mode
        self.position_step = position_step
        self.repeats_per_position = repeats_per_position
        self._swapped = False
        # Both sides are full-bandwidth frames; no spectral view is constructed.
        kw.setdefault("input_mode", "fullband")
        kw.setdefault("target_mode", "fullband")
        super().__init__(folder_specs, split, train_frac, **kw)

    # ------------------------------------------------------------------
    @property
    def frame_offset(self) -> int:
        """Index distance between the two frames of a pair."""
        if self.pair_mode == "repeat":
            return 1
        return self.repeats_per_position * self.position_step

    def _pair_starts(self, n_frames: int) -> list[int]:
        """Valid first-frame indices for a pair, within one acquisition."""
        off = self.frame_offset
        if self.pair_mode == "repeat":
            # Only even starts pair the two repeats of the SAME position; odd
            # starts would straddle a position boundary.
            return list(range(0, n_frames - off, self.repeats_per_position))
        return list(range(0, n_frames - off))

    def _build_index(self):
        if self._index is not None:
            return

        rng = np.random.RandomState(self.seed)
        index: list[tuple] = []

        for fidx, fs in enumerate(self.folder_specs):
            data_dir = os.path.join(fs.root_folder, fs.data_folder)
            paths = sorted(glob.glob(os.path.join(data_dir, "bscan*.raw")))
            n = len(paths)
            if n == 0:
                raise FileNotFoundError(f"No bscan*.raw found in {data_dir}")

            starts = self._pair_starts(n)
            if not starts:
                raise ValueError(
                    f"{fs.data_folder}: {n} frames is too few for pair_mode="
                    f"{self.pair_mode!r} with offset {self.frame_offset}"
                )

            # Split on PAIRS, so a frame cannot appear in both train and val
            # through different pairings.
            order = np.arange(len(starts))
            rng.shuffle(order)
            n_train = int(round(self.train_frac * len(starts)))
            chosen = order[:n_train] if self.split == "train" else order[n_train:]

            if not self.full_frame:
                z0, z1 = fs.crop_depth
                if fs.alines < self.patch_w:
                    raise ValueError(f"patch_w={self.patch_w} > alines={fs.alines}")
                if (z1 - z0) < self.patch_h:
                    raise ValueError(f"patch_h={self.patch_h} > cropped depth {z1 - z0}")

            for k in chosen:
                i = starts[int(k)]
                if self.full_frame:
                    index.append((fidx, i, i + self.frame_offset))
                else:
                    for pr in range(self.patches_per_frame):
                        index.append((fidx, i, i + self.frame_offset, pr))

        if self.full_frame and self.max_frames is not None and len(index) > self.max_frames:
            pick = np.random.RandomState(self.seed).permutation(len(index))[: self.max_frames]
            index = [index[i] for i in pick]

        self._index = index
        self._estimated_len = len(index)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        self._init_worker_state()   # builds the index if it is not built yet
        assert self._index is not None
        entry = self._index[idx]
        fidx, ia, ib = entry[0], entry[1], entry[2]

        # Alternate which frame is the input. The paired positions are 11.7 um
        # apart so their structure is not identical; alternating keeps that
        # residual bias symmetric instead of accumulating in one direction.
        # Held fixed for full-frame validation so the metric is deterministic.
        self._swapped = not self.full_frame and bool(self._rng.randint(0, 2))
        if self._swapped:
            ia, ib = ib, ia

        out_a = self._fetch_frame(fidx, ia)
        out_b = self._fetch_frame(fidx, ib)

        inputs = [out_a["target_full"]]
        tgt = out_b["target_full"]
        tmu = float(out_b["target_mu"])
        tsd = float(out_b["target_sd"])

        if self.full_frame:
            x = np.stack(inputs, axis=0).astype(np.float32)
            y = tgt[None, ...].astype(np.float32)
        else:
            x, y = self._random_crop(inputs, tgt)
            if self.augment:
                x, y = self._random_flips(x, y)
            else:
                x = np.ascontiguousarray(x)
                y = np.ascontiguousarray(y)

        return (
            _to_torch_float32(x),
            _to_torch_float32(y),
            self._build_meta(
                fidx, ia, out=out_b if self.full_frame else None,
                target_mu=tmu, target_sd=tsd,
            ),
        )
