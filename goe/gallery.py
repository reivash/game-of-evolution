"""The curated screenshot set.

Framing is most of the work with this automaton.  A bloom is a speck for its
first hundred generations and overruns the grid a few hundred after that, and
the window in between -- where the whole organism is visible *and* individual
cells are still resolvable -- is where it looks like anything.  So the shots
below specify a grid in cells and a magnification, and let
:meth:`~goe.automaton.Automaton.run_to_extent` decide when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .automaton import Automaton
from .palettes import PALETTES
from .render import colorize, to_image


@dataclass
class Shot:
    """One screenshot: what to simulate, how far, and how to colour it."""

    name: str
    caption: str
    rule: str = "classic"
    seed: str = "single"
    seed_args: dict = field(default_factory=dict)
    palette: str = "phosphor"
    cells: tuple[int, int] = (560, 560)
    scale: int = 2
    fit: float | None = 0.92
    generations: int | None = None
    wrap: bool = False
    echo: float = 0.94
    echo_weight: float = 0.45
    bloom: float = 0.55
    bloom_radius: float = 1.4
    gamma: float = 1.0
    crop: int | None = None
    crop_offset: tuple[int, int] = (0, 0)

    def render(self, backend: str = "auto") -> tuple[Image.Image, dict]:
        automaton = Automaton.from_seed(
            self.cells, rule=self.rule, seed=self.seed, seed_args=dict(self.seed_args),
            wrap=self.wrap, backend=backend, echo_decay=self.echo,
        )
        if self.generations is not None:
            automaton.run(self.generations)
        else:
            automaton.run_to_extent(self.fit or 0.92, check_every=10)

        state, echo = automaton.snapshot(), automaton.echo()
        if self.crop:
            h, w = state.shape
            oy, ox = self.crop_offset
            y = (h - self.crop) // 2 + oy
            x = (w - self.crop) // 2 + ox
            region = (slice(y, y + self.crop), slice(x, x + self.crop))
            state = state[region]
            echo = echo[region] if echo is not None else None

        rgb = colorize(
            state, automaton.rule.limit, PALETTES[self.palette], echo=echo,
            echo_weight=self.echo_weight, bloom=self.bloom,
            bloom_radius=self.bloom_radius, gamma=self.gamma,
        )
        return to_image(rgb, scale=self.scale), automaton.metrics()


#: The set rendered by ``python -m goe gallery``.
GALLERY: list[Shot] = [
    Shot(
        "genesis",
        "The original rule, from a single cell at level 3. Everything here -- "
        "the eight-fold symmetry, the concentric rings, the scalloped rim -- "
        "is the grid's own symmetry made visible; nothing in the seed asked "
        "for any of it.",
        rule="classic", seed="single", palette="phosphor",
        cells=(340, 340), scale=4,
    ),
    Shot(
        "deep-rings",
        "The same thresholds on a six-level ladder. The extra headroom lets "
        "the interior thin out into standing rings instead of saturating.",
        rule="deep", seed="single", palette="bone",
        cells=(340, 340), scale=4, echo=0.95,
    ),
    Shot(
        "mandala",
        "A random blob folded into eight-fold symmetry before it is planted. "
        "The randomness picks the motif, the rule keeps it symmetric forever.",
        rule="lace", seed="mandala", seed_args={"size": 28, "seed": 5},
        palette="ember", cells=(360, 360), scale=4, fit=0.90,
    ),
    Shot(
        "collision",
        "Five lone cells, far apart. Each blooms on its own until the fronts "
        "meet, and where they interfere the symmetry breaks into something "
        "neither of them would have produced.",
        rule="classic", seed="scatter", seed_args={"count": 5, "spread": 0.62, "seed": 11},
        palette="magma", cells=(560, 560), scale=3, fit=0.92,
    ),
    Shot(
        "veil",
        "A ring, not a point. Growth needs heavy support here and crowding "
        "bites twice as hard, so the two fronts -- one running inward, one "
        "outward -- stay thin, and meet at the centre.",
        rule="veil", seed="ring", seed_args={"radius": 40, "level": 3}, palette="ice",
        cells=(360, 360), scale=4,
    ),
    Shot(
        "weave",
        "A lattice of single cells on a torus. Every bloom grows into its "
        "neighbours at once, and the interference tiles the whole plane.",
        rule="deep", seed="lattice", seed_args={"pitch": 108}, palette="spectral",
        cells=(432, 432), scale=3, wrap=True, generations=380, fit=None,
    ),
    Shot(
        "filigree",
        "The growth front of a classic bloom at seven times magnification, "
        "with "
        "the echo turned off so nothing but the levels themselves is drawn. "
        "The lace is not a texture; it is individual cells.",
        rule="classic", seed="single", palette="phosphor",
        cells=(1200, 1200), scale=7, fit=0.72, crop=170, crop_offset=(-388, 0),
        echo=0.0, echo_weight=0.0, bloom=0.35, bloom_radius=1.0,
    ),
]


def render_timelapse(backend: str = "auto", cells: int = 420, scale: int = 1) -> Image.Image:
    """A strip showing one bloom at six ages, to make the growth legible."""
    marks = [60, 140, 260, 420, 620, 860]
    automaton = Automaton.from_seed(
        (cells, cells), rule="classic", seed="single", backend=backend, echo_decay=0.94
    )
    frames = automaton.run(marks[-1], capture_at=marks)

    gap = 8
    tile = cells * scale
    strip = Image.new("RGB", (len(frames) * tile + (len(frames) - 1) * gap, tile), (6, 8, 7))
    for i, frame in enumerate(frames):
        rgb = colorize(frame.state, automaton.rule.limit, PALETTES["phosphor"],
                       echo=frame.echo, echo_weight=0.45, bloom=0.5, bloom_radius=1.2)
        strip.paste(to_image(rgb, scale), (i * (tile + gap), 0))
    return strip


def cmd_gallery(args) -> int:
    """``python -m goe gallery`` -- render every shot into a directory."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or [])

    for shot in GALLERY:
        if wanted and shot.name not in wanted:
            continue
        image, metrics = shot.render(backend=args.backend)
        path = out / f"{shot.name}.png"
        image.save(path)
        print(f"{path}  {image.width}x{image.height}  "
              f"g={metrics['generation']}  cov={metrics['coverage']:.1%}")

    if not wanted or "timelapse" in wanted:
        strip = render_timelapse(backend=args.backend)
        strip.save(out / "timelapse.png")
        print(f"{out / 'timelapse.png'}  {strip.width}x{strip.height}")
    return 0
