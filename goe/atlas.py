"""Exhaustive survey of the seed codes: run every one, see which live.

With a numbering for seeds (:mod:`goe.seedcode`) the obvious question is what
the whole space does.  For a 3x3 patch and a binary alphabet that is 512 codes,
which collapse to 86 distinct organisms once translations, rotations and
reflections are folded together -- small enough to simply run all of them and
look at the results side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .automaton import Automaton
from .palettes import PALETTES
from .render import colorize, to_image
from .rules import PRESETS, Rule
from .seedcode import alphabet_for, distinct_codes, place


@dataclass
class Outcome:
    """What one seed code did."""

    code: int
    patch: np.ndarray
    generation: int
    coverage: float
    verdict: str  # "empty", "extinct", "static" or "bloom"
    state: np.ndarray
    echo: np.ndarray | None

    @property
    def live_cells(self) -> int:
        return int((self.patch > 0).sum())


def survey(
    rule: Rule,
    size: int = 3,
    full: bool = False,
    cells: int = 220,
    target: float = 0.82,
    max_generations: int = 4000,
    backend: str = "auto",
    max_codes: int | None = None,
) -> list[Outcome]:
    """Run every distinct seed code and classify how it ends up."""
    alphabet = alphabet_for(rule.grow, full=full)
    outcomes: list[Outcome] = []

    for code, patch in distinct_codes(size, alphabet, max_codes):
        grid = place(patch, (cells, cells))
        automaton = Automaton(rule, grid, backend=backend, echo_decay=0.94)
        verdict = "bloom"
        if not patch.any():
            verdict = "empty"  # code 0: the empty patch, not a seed that died
        try:
            automaton.run_to_extent(target, max_generations=max_generations, check_every=10)
        except RuntimeError:
            if verdict != "empty":
                verdict = "extinct"
        if verdict == "bloom" and automaton.extent() < target:
            # run_to_extent gave up on a still life or an oscillator.
            verdict = "static"
        metrics = automaton.metrics()
        outcomes.append(
            Outcome(code, patch, metrics["generation"], metrics["coverage"],
                    verdict, automaton.snapshot(), automaton.echo())
        )
    return outcomes


def render_atlas(
    outcomes: list[Outcome], palette: str = "phosphor", columns: int = 10,
    tile: int = 200, blooms_only: bool = False,
) -> Image.Image:
    """Lay the survey out as a labelled contact sheet."""
    shown = [o for o in outcomes if not blooms_only or o.verdict == "bloom"]
    if not shown:
        raise ValueError("nothing to draw")

    limit = max(int(o.state.max()) for o in shown) or 1
    label = 16
    rows = (len(shown) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label)), (8, 9, 10))
    draw = ImageDraw.Draw(sheet)

    for i, outcome in enumerate(shown):
        rgb = colorize(outcome.state, limit, PALETTES[palette], echo=outcome.echo,
                       echo_weight=0.45, bloom=0.5, bloom_radius=1.2)
        img = to_image(rgb).resize((tile, tile), Image.LANCZOS)
        x, y = (i % columns) * tile, (i // columns) * (tile + label)
        sheet.paste(img, (x, y))

        tag = f"{outcome.code}"
        if outcome.verdict != "bloom":
            tag += f"  {outcome.verdict}"
        draw.text((x + 4, y + tile + 3), tag, fill=(190, 210, 195))

        # A thumbnail of the seed itself, so the code is readable as a picture.
        ph, pw = outcome.patch.shape
        dot = 3
        ox, oy = x + tile - pw * dot - 4, y + 4
        for r in range(ph):
            for c in range(pw):
                if outcome.patch[r, c] > 0:
                    draw.rectangle(
                        [ox + c * dot, oy + r * dot, ox + c * dot + dot - 1,
                         oy + r * dot + dot - 1], fill=(235, 245, 235))
    return sheet


def cmd_seeds(args) -> int:
    """``python -m goe seeds`` -- survey the seed-code space."""
    rule = PRESETS[args.rule]
    alphabet = alphabet_for(rule.grow, full=args.full)
    total = len(alphabet) ** (args.size * args.size)
    print(f"rule {rule.name}: {args.size}x{args.size} patch, alphabet {alphabet} "
          f"-> {total:,} codes")

    outcomes = survey(rule, size=args.size, full=args.full, cells=args.cells,
                      backend=args.backend, max_codes=args.max_codes)
    counts = {v: n for v in ("bloom", "static", "extinct", "empty")
              if (n := sum(1 for o in outcomes if o.verdict == v))}
    print(f"{len(outcomes)} distinct up to symmetry and translation: "
          + ", ".join(f"{n} {v}" for v, n in counts.items()))

    blooms = [o for o in outcomes if o.verdict == "bloom"]
    if blooms:
        print(f"\n{'code':>8}{'seed':>6}{'gens':>7}{'cov':>8}   patch")
        for o in sorted(blooms, key=lambda o: o.generation)[: args.top]:
            # Show the level digit, so --full seeds stay readable.
            rows = ["".join("." if v == 0 else str(int(v)) for v in row)
                    for row in o.patch]
            print(f"{o.code:>8}{o.live_cells:>6}{o.generation:>7}"
                  f"{o.coverage * 100:>7.1f}%   {' / '.join(rows)}")

    if args.out:
        sheet = render_atlas(outcomes, palette=args.palette,
                             blooms_only=args.blooms_only)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.out)
        print(f"\nwrote {args.out}  {sheet.width}x{sheet.height}")
    return 0
