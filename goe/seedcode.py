"""Numbering every possible seed, the way Wolfram numbers elementary rules.

Wolfram's rule number is the truth table of a 1-D rule read as a base-2
numeral.  The same trick numbers *seeds* here: fix a ``size x size`` patch,
fix an alphabet of the levels a cell may take, and a seed is a base-``B``
numeral whose digits are the patch read top-left to bottom-right.

    code 0                  an empty patch -- nothing
    code 1                  the last cell of the patch, at the first live level
    code 5^4 = 625          (size 3, alphabet 0..4) the centre cell at level 1

Written out in base ``B``, the code *is* a picture of the seed::

    >>> decode(625, size=3, alphabet=(0, 1, 2, 3, 4))
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]], dtype=int8)

Two seeds that differ only by where they sit, or by a rotation or a flip, grow
the same organism -- the rule is isotropic and the grid is homogeneous.  So the
raw codes are massively redundant, and :func:`canonical_key` collapses each
orbit (8 dihedral symmetries x every translation) to one representative.  That
is what makes an exhaustive survey of a 3x3 patch worth running: 512 binary
codes contain far fewer than 512 distinct organisms.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


def alphabet_for(level: int, full: bool = False) -> tuple[int, ...]:
    """The digit alphabet: every level ``0..level``, or just dead-or-``level``.

    The binary alphabet is the useful default.  A lone cell only blooms when
    its level sits in the rule's growth band, so for a given rule there is
    normally exactly one level worth seeding with, and letting cells take
    intermediate levels mostly enumerates seeds that starve.
    """
    return tuple(range(level + 1)) if full else (0, level)


def decode(code: int, size: int = 3, alphabet: Sequence[int] = (0, 3)) -> np.ndarray:
    """Turn a code into its ``size x size`` patch.

    The most significant digit is the top-left cell, so the numeral reads in
    the same order as the picture.
    """
    if code < 0:
        raise ValueError("code must be non-negative")
    base = len(alphabet)
    cells = size * size
    if code >= base**cells:
        raise ValueError(f"code {code} does not fit in {size}x{size} base {base}")

    digits = np.zeros(cells, dtype=np.int64)
    for i in range(cells - 1, -1, -1):
        code, digits[i] = divmod(code, base)
    return np.asarray(alphabet, dtype=np.int8)[digits].reshape(size, size)


def encode(patch: np.ndarray, alphabet: Sequence[int] = (0, 3)) -> int:
    """The inverse of :func:`decode`."""
    lookup = {int(v): i for i, v in enumerate(alphabet)}
    code = 0
    for value in patch.reshape(-1):
        code = code * len(alphabet) + lookup[int(value)]
    return code


def crop(patch: np.ndarray) -> np.ndarray:
    """Trim dead rows and columns; position within the patch is not meaningful."""
    live = patch > 0
    if not live.any():
        return np.zeros((0, 0), dtype=np.int8)
    rows = np.flatnonzero(live.any(axis=1))
    cols = np.flatnonzero(live.any(axis=0))
    return patch[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _orbit(patch: np.ndarray) -> Iterator[np.ndarray]:
    """The eight dihedral images of a patch."""
    for flipped in (patch, np.fliplr(patch)):
        for turns in range(4):
            yield np.rot90(flipped, turns)


def canonical_key(patch: np.ndarray) -> tuple[int, int, bytes]:
    """A key identifying the seed's whole symmetry orbit.

    Two patches share a key exactly when one is a translation, rotation or
    reflection of the other -- that is, when they grow the same organism.
    """
    cropped = crop(patch)
    return min(
        (image.shape[0], image.shape[1], image.tobytes())
        for image in map(np.ascontiguousarray, _orbit(cropped))
    )


def distinct_codes(
    size: int = 3, alphabet: Sequence[int] = (0, 3), max_codes: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Walk the codes in order, yielding only the first of each symmetry orbit.

    Yields ``(code, patch)``.  The code is the smallest numeral in its orbit,
    so it is a stable name for that organism.
    """
    total = len(alphabet) ** (size * size)
    if max_codes is not None:
        total = min(total, max_codes)
    seen: set[tuple[int, int, bytes]] = set()
    for code in range(total):
        patch = decode(code, size, alphabet)
        key = canonical_key(patch)
        if key in seen:
            continue
        seen.add(key)
        yield code, patch


def place(patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Centre a patch on an empty grid, cropped so position is canonical."""
    grid = np.zeros(shape, dtype=np.int8)
    cropped = crop(patch)
    if cropped.size == 0:
        return grid
    h, w = cropped.shape
    y, x = (shape[0] - h) // 2, (shape[1] - w) // 2
    grid[y : y + h, x : x + w] = cropped
    return grid


def seed_from_code(
    shape: tuple[int, int],
    level: int = 3,
    code: int = 0,
    size: int = 3,
    full: bool = False,
) -> np.ndarray:
    """Seed generator for :data:`goe.seeds.SEEDS` -- ``--seed code``."""
    return place(decode(code, size, alphabet_for(level, full)), shape)
