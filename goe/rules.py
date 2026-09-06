"""The Game of Evolution rule family.

Conway's cells are binary: alive or dead.  A Game of Evolution cell carries a
*level* in ``0..limit``.  Two things follow from that, and they are what make
the automaton interesting:

1. A cell's influence on its neighbourhood is its level, not a flat 1.  The
   neighbourhood factor ``N`` is therefore the **sum** of the eight neighbours'
   levels (``0..8*limit``), not a population count (``0..8``).
2. A cell does not flip between two states; it walks up and down a ladder.
   ``N`` selects a *delta* applied to the current level, and the result is
   clamped back into ``0..limit``.

The original 2017 Java implementation (reivash/GOE, ``goe.maths.Matrix``) used

    N < 2  -> level - 1        starvation
    N == 2 -> level            equilibrium
    N == 3 -> level + 1        growth
    N >= 4 -> level - (N - 4)  overcrowding, increasingly punishing

which :func:`Rule.classic` reproduces exactly.  :class:`Rule` generalises it to
four named thresholds so the same engine can explore the neighbouring rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rule:
    """A level-based totalistic rule.

    The delta applied to a cell whose neighbours' levels sum to ``N`` is::

        -1                 if N < starve
         0                 if starve <= N < grow
        +1                 if grow   <= N < crowd
        -slope * (N-crowd) if N >= crowd

    Note the overcrowding branch is continuous with equilibrium: at exactly
    ``N == crowd`` the delta is zero, and it gets steeper from there.
    """

    limit: int = 4
    starve: int = 2
    grow: int = 3
    crowd: int = 4
    slope: int = 1
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if not 0 <= self.starve <= self.grow <= self.crowd:
            raise ValueError("thresholds must satisfy 0 <= starve <= grow <= crowd")

    @property
    def max_factor(self) -> int:
        """Largest neighbourhood factor reachable: eight neighbours at ``limit``."""
        return 8 * self.limit

    def delta_table(self) -> np.ndarray:
        """Delta for every reachable ``N``, as ``int8`` indexed by ``N``.

        Precomputing this is what lets both backends reduce the whole rule to a
        single table lookup -- on the GPU it lives in constant memory.
        """
        n = np.arange(self.max_factor + 1, dtype=np.int32)
        delta = np.where(
            n < self.starve,
            -1,
            np.where(
                n < self.grow,
                0,
                np.where(n < self.crowd, 1, -self.slope * (n - self.crowd)),
            ),
        )
        # A single step can never move a cell further than the full ladder.
        return np.clip(delta, -self.limit, self.limit).astype(np.int8)

    @classmethod
    def classic(cls, limit: int = 4) -> "Rule":
        """The original 2017 rule.  ``limit=4`` is the historical setting."""
        return cls(limit=limit, starve=2, grow=3, crowd=4, slope=1, name="classic")


#: Named rules worth looking at.
#:
#: These are not guesses.  Sweeping the threshold space and scoring each rule by
#: the multi-scale variation of its cell density -- which separates structured
#: lace from the uniform noise discs that most live rules settle into -- leaves
#: a few hundred survivors out of a few thousand, and the ones below are the
#: top of that ranking.  ``classic`` places 7th, which is a decent sign that
#: the 2017 thresholds were not an accident.
PRESETS: dict[str, Rule] = {
    # The original.  A lone level-3 cell blooms into an eight-fold mandala.
    "classic": Rule.classic(),
    # The classic thresholds on a longer ladder: identical structure, but six
    # levels in play instead of four, so the shading has somewhere to go.
    # Ranked first in the sweep.
    "deep": Rule(limit=6, starve=2, grow=3, crowd=4, slope=1, name="deep"),
    # Wider bands all round: a looser, lacier weave that spreads faster.
    "lace": Rule(limit=10, starve=3, grow=4, crowd=6, slope=1, name="lace"),
    # Slow and finely graded -- growth needs heavy support and overcrowding
    # bites twice as hard, so the front stays thin and the interior thins out.
    "veil": Rule(limit=12, starve=3, grow=5, crowd=8, slope=2, name="veil"),
    # The densest of the set: fills its disc rather than filigreeing it.
    "flood": Rule(limit=6, starve=2, grow=4, crowd=6, slope=1, name="flood"),
    # A single level of starvation tolerance, which lets thin filaments survive
    # in the wake of the front instead of being pruned.
    "filament": Rule(limit=5, starve=1, grow=3, crowd=4, slope=1, name="filament"),
}
