"""Seeds are numbered the way Wolfram numbers rules; this pins the numbering."""

from __future__ import annotations

import numpy as np
import pytest

from goe.seedcode import (
    alphabet_for, canonical_key, crop, decode, distinct_codes, encode, place,
    seed_from_code,
)

FULL = alphabet_for(4, full=True)   # (0, 1, 2, 3, 4) -- base 5
BINARY = alphabet_for(3)            # (0, 3)          -- base 2


def test_code_zero_is_nothing():
    assert not decode(0, 3, BINARY).any()
    assert not decode(0, 3, FULL).any()


def test_the_numeral_is_a_picture_of_the_patch():
    # 625 == 1 * 5**4, and digit 4 of nine is the centre cell.
    assert np.array_equal(
        decode(625, 3, FULL),
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
    )


def test_encode_inverts_decode():
    for code in (0, 1, 7, 625, 4321, 5**9 - 1):
        assert encode(decode(code, 3, FULL), FULL) == code


def test_codes_outside_the_patch_are_rejected():
    with pytest.raises(ValueError):
        decode(2**9, 3, BINARY)
    with pytest.raises(ValueError):
        decode(-1, 3, BINARY)


def test_translation_does_not_change_the_organism():
    """A lone cell is the same seed wherever it sits in the patch."""
    keys = {canonical_key(decode(1 << i, 3, BINARY)) for i in range(9)}
    assert len(keys) == 1


def test_rotations_and_reflections_collapse_together():
    patch = np.array([[3, 3, 0], [0, 3, 0], [0, 0, 0]], dtype=np.int8)
    key = canonical_key(patch)
    for turns in range(4):
        assert canonical_key(np.rot90(patch, turns)) == key
        assert canonical_key(np.fliplr(np.rot90(patch, turns))) == key


def test_distinct_codes_dedupes_the_binary_3x3_space():
    codes = list(distinct_codes(3, BINARY))
    assert len(codes) == 86           # of 512 raw codes
    assert codes[0][0] == 0           # enumeration starts at the empty patch
    assert all(a[0] < b[0] for a, b in zip(codes, codes[1:]))  # ascending


def test_crop_removes_dead_margin():
    patch = np.zeros((3, 3), dtype=np.int8)
    patch[2, 1] = 3
    assert crop(patch).shape == (1, 1)
    assert crop(np.zeros((3, 3), dtype=np.int8)).size == 0


def test_place_centres_the_cropped_patch():
    grid = place(decode(1, 3, BINARY), (11, 11))
    assert grid[5, 5] == 3
    assert int((grid > 0).sum()) == 1


def test_seed_from_code_matches_the_single_seed():
    from goe.seeds import single

    assert np.array_equal(seed_from_code((21, 21), level=3, code=1), single((21, 21), 3))
