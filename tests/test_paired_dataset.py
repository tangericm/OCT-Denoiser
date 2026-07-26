"""Frame-pair Noise2Noise dataset.

Pairing must respect the interleaved OCTA layout [p0r0, p0r1, p1r0, p1r1, ...]
and must not leak a frame across the train/val split via different pairings.
"""
from __future__ import annotations

import numpy as np
import pytest

from octdenoiser.data.paired_dataset import PairedFrameDataset


def _ds(spec, **kw):
    base = dict(folder_specs=[spec], split="train", train_frac=0.7,
                patch_h=32, patch_w=16, patches_per_frame=2, patch_mode="patch",
                augment=False, seed=0, cache_frames_per_worker=8)
    base.update(kw)
    return PairedFrameDataset(**base)


# --------------------------------------------------------------------------
# Pair geometry
# --------------------------------------------------------------------------
def test_position_pairing_steps_a_whole_position():
    """Repeats are interleaved, so one position step is TWO indices."""
    assert _ds(_spec(), pair_mode="position", position_step=1).frame_offset == 2


def test_position_step_scales():
    assert _ds(_spec(), pair_mode="position", position_step=3).frame_offset == 6


def test_repeat_pairing_steps_one_index():
    assert _ds(_spec(), pair_mode="repeat").frame_offset == 1


def test_repeat_pairs_start_on_even_indices_only():
    """Odd starts would straddle a position boundary, not pair a repeat."""
    ds = _ds(_spec(), pair_mode="repeat")
    assert ds._starts_in_block(0, 10) == [0, 2, 4, 6, 8]


def test_repeat_starts_align_to_the_global_position_grid():
    """A block boundary is not a position boundary; starts must snap to the grid."""
    ds = _ds(_spec(), pair_mode="repeat")
    assert ds._starts_in_block(3, 10) == [4, 6, 8], "start 3 would pair p1r1 with p2r0"


def test_position_pairs_use_every_start():
    ds = _ds(_spec(), pair_mode="position", position_step=1)
    assert ds._starts_in_block(0, 10) == list(range(8))


def test_block_starts_keep_both_frames_inside_the_block():
    ds = _ds(_spec(), pair_mode="position", position_step=1)
    starts = ds._starts_in_block(4, 12)
    assert starts, "block must yield pairs"
    assert min(starts) >= 4
    assert max(starts) + ds.frame_offset < 12


def test_rejects_bad_pair_mode():
    with pytest.raises(ValueError, match="pair_mode must be"):
        _ds(_spec(), pair_mode="adjacent")


def test_rejects_bad_step():
    with pytest.raises(ValueError, match="position_step must be"):
        _ds(_spec(), position_step=0)


def test_repeat_mode_needs_repeats():
    with pytest.raises(ValueError, match="repeats_per_position >= 2"):
        _ds(_spec(), pair_mode="repeat", repeats_per_position=1)


# --------------------------------------------------------------------------
# Index and splits
# --------------------------------------------------------------------------
def test_index_pairs_are_offset_correctly(synthetic_spec):
    ds = _ds(synthetic_spec, pair_mode="position", position_step=1, patches_per_frame=1)
    ds._build_index()
    assert ds._index, "index must not be empty"
    for _fidx, ia, ib, *_ in ds._index:
        assert ib - ia == 2, f"expected offset 2, got {ib - ia}"


def test_train_and_val_pairs_are_disjoint(synthetic_spec):
    common = dict(train_frac=0.6, patches_per_frame=1, pair_mode="position", position_step=1)
    tr = _ds(synthetic_spec, split="train", **common)
    va = _ds(synthetic_spec, split="val", **common)
    tr._build_index()
    va._build_index()

    tr_pairs = {(e[1], e[2]) for e in tr._index}
    va_pairs = {(e[1], e[2]) for e in va._index}
    assert tr_pairs and va_pairs
    assert not (tr_pairs & va_pairs), "a pair must not appear in both splits"


def test_train_and_val_share_no_frame(synthetic_spec):
    """The split must isolate FRAMES, not just pairs.

    Position pairs overlap in their constituent frames -- (0,2) and (2,4) both
    use frame 2 -- so a shuffled pair-level split puts the same frame on both
    sides. That inflates the validation loss driving early stopping and
    checkpoint selection. Blocks of contiguous frames are what prevent it.
    """
    common = dict(train_frac=0.6, patches_per_frame=1, pair_mode="position", position_step=1)
    tr = _ds(synthetic_spec, split="train", **common)
    va = _ds(synthetic_spec, split="val", **common)
    tr._build_index()
    va._build_index()

    tr_frames = {f for e in tr._index for f in (e[1], e[2])}
    va_frames = {f for e in va._index for f in (e[1], e[2])}
    assert tr_frames and va_frames
    assert not (tr_frames & va_frames), f"frames in both splits: {sorted(tr_frames & va_frames)}"


def test_too_few_frames_raises(synthetic_spec):
    ds = _ds(synthetic_spec, pair_mode="position", position_step=99)
    with pytest.raises(ValueError, match="cannot be split into two blocks"):
        ds._build_index()


def test_split_failure_advises_raising_group_size_not_lowering_it(synthetic_spec):
    """The remediation half of the message must point somewhere that works.

    `_blocks` uses n_blocks = max(2, n // group_size), so block WIDTH scales with
    group_size and a block survives only if it is wider than frame_offset.
    Lowering group_size narrows blocks and can only make the failure worse, so
    advice to lower it is never correct -- and group_size is floored at 2, so a
    user already at the floor would be told to do something the ctor rejects.
    """
    with pytest.raises(ValueError, match="raise group_size"):
        _ds(synthetic_spec, pair_mode="position", position_step=99)._build_index()


@pytest.mark.parametrize("n_frames,offset", [(64, 10), (128, 6), (1024, 2)])
def test_group_sizes_that_work_are_upward_closed(n_frames, offset):
    """The property that makes "raise group_size" sound advice.

    Not the surviving-block COUNT -- that peaks and then falls, since bigger
    blocks means fewer of them (n=64, offset=10 gives [0,0,0,4,2,2]). What holds
    is that the split SUCCEEDS on an upward-closed set: below a threshold every
    group_size fails, at and above it every group_size works. So raising can fix
    a failure and lowering never can.
    """
    sizes = (2, 4, 8, 16, 32, 64, 128)
    ok = []
    for gs in sizes:
        ds = _ds(_spec(), pair_mode="position", position_step=offset // 2, group_size=gs)
        assert ds.frame_offset == offset, "test setup must hit the intended offset"
        ok.append(len(ds._blocks(n_frames)) >= 2)

    assert any(ok), f"n={n_frames} offset={offset}: no group_size works, bad fixture"
    first = ok.index(True)
    assert all(ok[first:]), (
        f"n={n_frames} offset={offset}: success by group_size "
        f"{dict(zip(sizes, ok, strict=True))} "
        f"is not upward-closed, so 'raise group_size' would not be sound advice"
    )
    assert not any(ok[:first]), "failures must all sit below the threshold"


# --------------------------------------------------------------------------
# Item construction
# --------------------------------------------------------------------------
def test_yields_single_channel_full_band_pair(synthetic_spec):
    ds = _ds(synthetic_spec, pair_mode="position", position_step=1)
    x, y, meta = ds[0]
    assert x.shape == (1, 32, 16), "both sides are full-bandwidth, 1 channel"
    assert y.shape == (1, 32, 16)
    assert np.isfinite(x.numpy()).all() and np.isfinite(y.numpy()).all()
    assert isinstance(meta, dict)


def test_input_and_target_are_different_frames(synthetic_spec):
    ds = _ds(synthetic_spec, pair_mode="position", position_step=1,
             patch_h=64, patch_w=32)
    x, y, _ = ds[0]
    assert not np.allclose(x.numpy()[0], y.numpy()[0]), "pair must be two frames"


def test_direction_is_alternated(synthetic_spec):
    """Paired positions differ slightly in structure, so the direction of the
    mapping must alternate or the residual bias accumulates one way."""
    ds = _ds(synthetic_spec, pair_mode="position", position_step=1, patches_per_frame=16)
    seen = set()
    for i in range(min(len(ds), 60)):
        ds[i]
        seen.add(ds._swapped)
    assert seen == {True, False}, f"both directions must occur, saw {seen}"


def test_full_frame_direction_is_never_swapped(synthetic_spec):
    ds = _ds(synthetic_spec, split="val", full_frame=True,
             pair_mode="position", position_step=1)
    for i in range(min(len(ds), 5)):
        ds[i]
        assert ds._swapped is False


def test_full_frame_mode_is_deterministic(synthetic_spec):
    ds = _ds(synthetic_spec, split="val", full_frame=True, pair_mode="position",
             position_step=1)
    a1, b1, m1 = ds[0]
    a2, b2, m2 = ds[0]
    assert np.array_equal(a1.numpy(), a2.numpy())
    assert np.array_equal(b1.numpy(), b2.numpy())
    for key in ("target_mu", "target_sd", "log_eps"):
        assert key in m1, f"full-frame meta missing {key}"


def test_repeat_mode_produces_valid_items(synthetic_spec):
    ds = _ds(synthetic_spec, pair_mode="repeat")
    x, y, _ = ds[0]
    assert x.shape == (1, 32, 16)
    assert y.shape == (1, 32, 16)


def _spec():
    from octdenoiser.configs.default import FolderSpec

    return FolderSpec(root_folder="r", data_folder="d", pixels=256, alines=64,
                      crop_depth=(0, 128))
