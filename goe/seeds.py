"""Seed patterns.

Every seed returns an ``int8`` grid of the requested shape.  The interesting
property of this automaton is how little it takes: a *single* cell at level 3
on an empty grid blooms indefinitely into an eight-fold mandala, because the
rule is isotropic and the grid's own symmetry is all the structure it needs.

Seeds are registered in :data:`SEEDS` and take keyword arguments, so the CLI
can expose them as ``--seed mandala --seed-arg size=24``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Shape = tuple[int, int]


def _blank(shape: Shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int8)


def _centre(shape: Shape) -> tuple[int, int]:
    return shape[0] // 2, shape[1] // 2


def single(shape: Shape, level: int = 3) -> np.ndarray:
    """One cell.  The canonical seed -- everything else is a variation."""
    g = _blank(shape)
    cy, cx = _centre(shape)
    g[cy, cx] = level
    return g


def cross(shape: Shape, level: int = 3, arm: int = 3) -> np.ndarray:
    """A plus sign.  Four-fold rather than eight-fold, so growth is squarer."""
    g = _blank(shape)
    cy, cx = _centre(shape)
    g[cy - arm : cy + arm + 1, cx] = level
    g[cy, cx - arm : cx + arm + 1] = level
    return g


def ring(shape: Shape, level: int = 3, radius: int = 24, thickness: int = 1) -> np.ndarray:
    """A circle.  Grows inwards and outwards at once, and the two fronts meet."""
    g = _blank(shape)
    cy, cx = _centre(shape)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    d = np.hypot(yy - cy, xx - cx)
    g[(d >= radius) & (d < radius + thickness)] = level
    return g


def r_pentomino(shape: Shape, level: int = 3) -> np.ndarray:
    """Conway's famous chaotic starter, lifted into levels."""
    g = _blank(shape)
    cy, cx = _centre(shape)
    for dy, dx in ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)):
        g[cy + dy - 1, cx + dx - 1] = level
    return g


def random_patch(
    shape: Shape, level: int = 3, size: int = 48, density: float = 0.35, seed: int = 0
) -> np.ndarray:
    """An asymmetric blob.  Produces organic, non-repeating growth."""
    g = _blank(shape)
    rng = np.random.default_rng(seed)
    cy, cx = _centre(shape)
    h = size // 2
    patch = (rng.random((size, size)) < density) * level
    g[cy - h : cy - h + size, cx - h : cx - h + size] = patch
    return g


def mandala(
    shape: Shape,
    level: int = 3,
    size: int = 32,
    density: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """A random patch folded into eight-fold dihedral symmetry.

    Randomness supplies the motif, the folding supplies the order, and the rule
    turns both into a bloom that keeps the symmetry forever.  This is the seed
    that generates the most varied output for a given amount of luck.
    """
    g = _blank(shape)
    rng = np.random.default_rng(seed)
    q = (rng.random((size, size)) < density) * level
    q = np.tril(q)  # keep one octant, then unfold it
    q = np.maximum(q, q.T)  # mirror across the diagonal
    half = np.concatenate([np.fliplr(q), q], axis=1)
    full = np.concatenate([np.flipud(half), half], axis=0).astype(np.int8)
    cy, cx = _centre(shape)
    h = size
    g[cy - h : cy + h, cx - h : cx + h] = full
    return g


def scatter(
    shape: Shape,
    level: int = 3,
    count: int = 16,
    spread: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Several lone cells.  Each blooms, and the fronts interfere where they meet."""
    g = _blank(shape)
    rng = np.random.default_rng(seed)
    cy, cx = _centre(shape)
    ry, rx = int(shape[0] * spread / 2), int(shape[1] * spread / 2)
    ys = rng.integers(cy - ry, cy + ry, count)
    xs = rng.integers(cx - rx, cx + rx, count)
    g[ys, xs] = level
    return g


def lattice(shape: Shape, level: int = 3, pitch: int = 96, jitter: int = 0, seed: int = 0) -> np.ndarray:
    """A regular grid of lone cells: blooms on a lattice, tiling the plane."""
    g = _blank(shape)
    rng = np.random.default_rng(seed)
    ys = np.arange(pitch // 2, shape[0], pitch)
    xs = np.arange(pitch // 2, shape[1], pitch)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    if jitter:
        yy = yy + rng.integers(-jitter, jitter + 1, yy.shape)
        xx = xx + rng.integers(-jitter, jitter + 1, xx.shape)
    ok = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    g[yy[ok], xx[ok]] = level
    return g


from .seedcode import seed_from_code  # noqa: E402  (registry lives below)

SEEDS: dict[str, Callable[..., np.ndarray]] = {
    "single": single,
    # Any patch at all, addressed by number -- see goe/seedcode.py.
    "code": seed_from_code,
    "cross": cross,
    "ring": ring,
    "r-pentomino": r_pentomino,
    "random": random_patch,
    "mandala": mandala,
    "scatter": scatter,
    "lattice": lattice,
}
