"""NumPy reference backend.

Correctness lives here.  The CUDA backend is validated against this one
(see ``tests/test_backends.py``), so this implementation stays deliberately
plain: a padded halo buffer and eight shifted views summed together.
"""

from __future__ import annotations

import numpy as np

from ..rules import Rule


class CpuEngine:
    """Evolves a level grid on the CPU with NumPy."""

    name = "numpy"

    def __init__(self, rule: Rule, state: np.ndarray, wrap: bool = False) -> None:
        self.rule = rule
        self.wrap = wrap
        self._state = np.ascontiguousarray(state, dtype=np.int8)
        h, w = self._state.shape
        # int16 because the neighbourhood factor reaches 8*limit, past int8.
        self._pad = np.zeros((h + 2, w + 2), dtype=np.int16)
        self._delta = rule.delta_table().astype(np.int16)

    @property
    def state(self) -> np.ndarray:
        return self._state

    def _factors(self) -> np.ndarray:
        """Sum of the eight neighbours' levels, for every cell."""
        p = self._pad
        p[1:-1, 1:-1] = self._state
        if self.wrap:
            # Mirror the opposite edges into the halo so the grid is a torus.
            p[0, 1:-1] = self._state[-1]
            p[-1, 1:-1] = self._state[0]
            p[1:-1, 0] = self._state[:, -1]
            p[1:-1, -1] = self._state[:, 0]
            p[0, 0] = self._state[-1, -1]
            p[0, -1] = self._state[-1, 0]
            p[-1, 0] = self._state[0, -1]
            p[-1, -1] = self._state[0, 0]
        # else: the halo stays zero, i.e. the original's dead border.
        return (
            p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:]
            + p[1:-1, 0:-2] + p[1:-1, 2:]
            + p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
        )

    def step(self, generations: int = 1) -> None:
        for _ in range(generations):
            nxt = self._state + self._delta[self._factors()]
            self._state = np.clip(nxt, 0, self.rule.limit).astype(np.int8)

    def download(self) -> np.ndarray:
        return self._state.copy()
