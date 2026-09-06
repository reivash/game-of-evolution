"""Turning a level grid into an image worth looking at.

A straight ``level -> colour`` lookup is legible but flat: five colours on
black, which is roughly what the 2017 build drew.  Two cheap additions carry
most of the visual weight here:

**Echo.**  The automaton's growth front is thin, and everything interesting
about the pattern is *where the front has been*.  The echo buffer keeps a
decaying maximum of past energy, so the structure the bloom carved out stays
faintly visible behind the live cells instead of vanishing each generation.

**Bloom.**  Bright cells bleed light into their neighbours at two scales -- a
tight core and a wide halo -- which is what makes a lattice of single pixels
read as something emitting light rather than a noisy bitmap.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .palettes import Palette


def _blur(a: np.ndarray, radius: float, downscale: int = 1) -> np.ndarray:
    """Gaussian blur via PIL, optionally at reduced resolution.

    Wide halos are blurred at 1/N resolution and scaled back up: the result is
    visually identical once it is this soft, and it costs a fraction as much.
    """
    h, w = a.shape
    img = Image.fromarray((np.clip(a, 0.0, 1.0) * 255).astype(np.uint8), "L")
    if downscale > 1:
        img = img.resize((max(1, w // downscale), max(1, h // downscale)), Image.BILINEAR)
        radius = radius / downscale
    img = img.filter(ImageFilter.GaussianBlur(radius))
    if downscale > 1:
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def colorize(
    state: np.ndarray,
    limit: int,
    palette: Palette,
    echo: np.ndarray | None = None,
    echo_weight: float = 0.55,
    bloom: float = 0.8,
    bloom_radius: float = 2.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Map a level grid to an ``(H, W, 3)`` uint8 image."""
    energy = state.astype(np.float32) / float(limit)

    field = energy
    if echo is not None and echo_weight > 0:
        # (1 - energy) keeps the echo out of the way of live cells: it fills the
        # dead space behind the front rather than washing out the front itself.
        field = np.clip(energy + echo_weight * echo * (1.0 - energy), 0.0, 1.0)
    if gamma != 1.0:
        field = np.power(field, gamma, dtype=np.float32)

    rgb = palette.lut()[(field * 255.0).astype(np.uint8)].astype(np.float32)

    if bloom > 0:
        core = _blur(energy * energy, bloom_radius, downscale=1)
        halo = _blur(energy, bloom_radius * 4.0, downscale=4)
        glow = np.clip(core + 0.55 * halo, 0.0, 1.0)[..., None]
        rgb = rgb + glow * np.asarray(palette.glow, dtype=np.float32) * bloom

    return np.clip(rgb, 0, 255).astype(np.uint8)


def to_image(rgb: np.ndarray, scale: int = 1) -> Image.Image:
    """Wrap a pixel array, magnifying by an integer factor with hard edges.

    Nearest-neighbour on purpose: at scale > 1 the cells should stay crisp
    squares, the way the original rendered them as quads.
    """
    img = Image.fromarray(rgb, "RGB")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save(rgb: np.ndarray, path, scale: int = 1) -> None:
    to_image(rgb, scale).save(path)
