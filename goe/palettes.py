"""Colour ramps.

A cell's level is a scalar, so colouring is a 1-D ramp lookup.  Each palette is
a handful of control points that get interpolated into a 256-entry ``uint8``
LUT once, at import-time cost of nothing.

Ramps are written dark-to-bright: index 0 is the dead background, index 255 a
saturated cell.  ``glow`` is the tint used by the bloom pass, normally close to
the ramp's bright end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Stop = tuple[float, tuple[int, int, int]]


@dataclass(frozen=True)
class Palette:
    name: str
    stops: tuple[Stop, ...]
    glow: tuple[int, int, int] = (255, 255, 255)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def lut(self) -> np.ndarray:
        """A ``(256, 3)`` uint8 lookup table."""
        if "lut" not in self._cache:
            xs = np.linspace(0.0, 1.0, 256)
            pos = np.array([p for p, _ in self.stops], dtype=np.float64)
            cols = np.array([c for _, c in self.stops], dtype=np.float64)
            table = np.stack(
                [np.interp(xs, pos, cols[:, ch]) for ch in range(3)], axis=1
            )
            self._cache["lut"] = np.clip(table, 0, 255).astype(np.uint8)
        return self._cache["lut"]

    @property
    def background(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.lut()[0])


PALETTES: dict[str, Palette] = {
    # The original's green CRT phosphor, given a proper ramp.
    "phosphor": Palette(
        "phosphor",
        (
            (0.00, (2, 4, 3)),
            (0.25, (7, 54, 27)),
            (0.50, (34, 142, 62)),
            (0.75, (129, 224, 124)),
            (1.00, (233, 255, 222)),
        ),
        glow=(120, 255, 150),
    ),
    "ember": Palette(
        "ember",
        (
            (0.00, (5, 2, 6)),
            (0.25, (74, 12, 40)),
            (0.50, (176, 42, 44)),
            (0.75, (243, 144, 45)),
            (1.00, (255, 246, 196)),
        ),
        glow=(255, 150, 60),
    ),
    "ice": Palette(
        "ice",
        (
            (0.00, (3, 5, 14)),
            (0.25, (13, 49, 96)),
            (0.50, (28, 124, 168)),
            (0.75, (110, 208, 224)),
            (1.00, (233, 251, 255)),
        ),
        glow=(120, 210, 255),
    ),
    "magma": Palette(
        "magma",
        (
            (0.00, (4, 3, 18)),
            (0.25, (48, 18, 98)),
            (0.50, (122, 42, 140)),
            (0.75, (216, 92, 106)),
            (1.00, (252, 226, 158)),
        ),
        glow=(255, 140, 120),
    ),
    "spectral": Palette(
        "spectral",
        (
            (0.00, (4, 4, 12)),
            (0.20, (32, 28, 120)),
            (0.40, (16, 150, 150)),
            (0.60, (140, 200, 60)),
            (0.80, (240, 150, 40)),
            (1.00, (255, 244, 220)),
        ),
        glow=(200, 220, 180),
    ),
    "bone": Palette(
        "bone",
        (
            (0.00, (4, 4, 6)),
            (0.50, (110, 112, 124)),
            (1.00, (255, 255, 255)),
        ),
        glow=(220, 225, 255),
    ),
}
