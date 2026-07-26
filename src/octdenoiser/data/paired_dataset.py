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

    Train/val are split on contiguous blocks of `group_size` frames, not on
    pairs. Position pairs share frames -- (0,2) and (2,4) both use frame 2 -- so
    a pair-level split leaks frames across the boundary and inflates the
    validation loss that selects the checkpoint.

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
        group_size: int = 64,
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
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")

        self.pair_mode = pair_mode
        self.position_step = position_step
        self.repeats_per_position = repeats_per_position
        self.group_size = group_size
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

    def _starts_in_block(self, lo: int, hi: int) -> list[int]:
        """Pair starts whose every constituent frame lies inside [lo, hi)."""
        off = self.frame_offset
        if self.pair_mode == "repeat":
            # Only starts on the position grid pair the two repeats of the SAME
            # position; off-grid starts would straddle a position boundary. The
            # grid is global, so align to it rather than to the block edge.
            s = self.repeats_per_position
            first = ((lo + s - 1) // s) * s
            return list(range(first, hi - off, s))
        return list(range(lo, hi - off))

    def _blocks(self, n_frames: int) -> list[tuple[int, int]]:
        """Contiguous frame ranges to split on, big enough to hold a pair."""
        n_blocks = max(2, n_frames // self.group_size)
        edges = np.linspace(0, n_frames, n_blocks + 1).round().astype(int)
        return [
            (int(a), int(b))
            for a, b in zip(edges[:-1], edges[1:], strict=True)
            if b - a > self.frame_offset
        ]

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

            # Split on contiguous BLOCKS of frames, then form pairs inside each.
            #
            # Splitting on pairs is not enough: position pairs overlap in their
            # frames -- (0,2) and (2,4) both use frame 2 -- so shuffling starts
            # puts the same frame in train and val, and the validation loss that
            # drives early stopping and checkpoint selection reads optimistic.
            # Blocks also handle the correlation between neighbouring B-scans,
            # which a frame-level split ignores entirely.
            #
            # Cost: pairs straddling a block edge are dropped, `frame_offset` of
            # them per block (~3% at the default group_size=64, offset 2).
            blocks = self._blocks(n)
            if len(blocks) < 2:
                raise ValueError(
                    f"{fs.data_folder}: {n} frames cannot be split into two blocks "
                    f"holding a pair of offset {self.frame_offset}; lower group_size "
                    f"(currently {self.group_size}) or use a longer acquisition"
                )

            order = np.arange(len(blocks))
            rng.shuffle(order)
            # Clamp so neither split is empty -- an empty val loader fails later
            # and much less legibly.
            n_train = int(np.clip(round(self.train_frac * len(blocks)), 1, len(blocks) - 1))
            picked = order[:n_train] if self.split == "train" else order[n_train:]

            starts = [s for b in picked for s in self._starts_in_block(*blocks[int(b)])]
            if not starts:
                raise ValueError(
                    f"{fs.data_folder}: no valid pairs in the {self.split} blocks for "
                    f"pair_mode={self.pair_mode!r} with offset {self.frame_offset}"
                )

            if not self.full_frame:
                z0, z1 = fs.crop_depth
                if fs.alines < self.patch_w:
                    raise ValueError(f"patch_w={self.patch_w} > alines={fs.alines}")
                if (z1 - z0) < self.patch_h:
                    raise ValueError(f"patch_h={self.patch_h} > cropped depth {z1 - z0}")

            for i in starts:
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


class MultiFrameDataset(PairedFrameDataset):
    """K neighbouring frames in, one held-out frame as the target.

    Every scheme compared so far is single-frame in, single-frame out, and they
    share a ceiling: one B-scan does not contain what a 50-frame average does.
    Measured, one network pass is worth roughly 8-16 averaged frames, yet
    50-frame averaging still looks clearly better than any of them. Feeding
    several frames is the direct attack on that ceiling.

    Layout, with s = repeats_per_position so one position step is s indices:

        input  : c - m*s, ..., c, ..., c + m*s      (K frames, m = K//2)
        target : c + (m+1)*s                        (one position beyond)

    The target sits outside the input window, so it shares no frame with the
    input and its speckle is decorrelated (measured +0.017 at one position of
    separation). Keeping the same repeat index throughout avoids mixing in the
    repeat pairing, whose speckle correlation is 6x higher.
    """

    def __init__(self, folder_specs, split, train_frac, *, n_input_frames: int = 5, **kw):
        if n_input_frames < 2:
            raise ValueError(f"n_input_frames must be >= 2, got {n_input_frames}")
        self.n_input_frames = n_input_frames
        kw.setdefault("pair_mode", "position")
        super().__init__(folder_specs, split, train_frac, **kw)

    @property
    def _half(self) -> int:
        return self.n_input_frames // 2

    @property
    def frame_offset(self) -> int:
        """Index span from the first input frame to the target."""
        s = self.repeats_per_position * self.position_step
        return (self._half + self._half + 1) * s

    def _input_and_target(self, start: int) -> tuple[list[int], int]:
        s = self.repeats_per_position * self.position_step
        inputs = [start + i * s for i in range(self.n_input_frames)]
        return inputs, start + self.n_input_frames * s

    def __getitem__(self, idx: int):
        self._init_worker_state()
        assert self._index is not None
        entry = self._index[idx]
        fidx, start = entry[0], entry[1]

        in_idx, tgt_idx = self._input_and_target(start)
        # Reversing the stack is a free augmentation: the physics is symmetric
        # in scan direction, and it keeps the residual structure bias symmetric.
        self._swapped = not self.full_frame and bool(self._rng.randint(0, 2))
        if self._swapped:
            in_idx = in_idx[::-1]

        inputs = [self._fetch_frame(fidx, i)["target_full"] for i in in_idx]
        out_t = self._fetch_frame(fidx, tgt_idx)
        tgt = out_t["target_full"]

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
            self._build_meta(fidx, in_idx[len(in_idx) // 2],
                             out=out_t if self.full_frame else None,
                             target_mu=float(out_t["target_mu"]),
                             target_sd=float(out_t["target_sd"])),
        )
