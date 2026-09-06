"""The simulation front end: rule + seed + backend, plus the echo buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .backends import make_engine
from .rules import PRESETS, Rule
from .seeds import SEEDS


@dataclass
class Frame:
    """A captured generation: the levels, and the echo behind them."""

    generation: int
    state: np.ndarray
    echo: np.ndarray | None


class Automaton:
    """Runs a :class:`~goe.rules.Rule` over a grid on the chosen backend.

    The echo buffer, when enabled, lives on the same device as the grid, so a
    thousand-generation GPU run still costs exactly one host transfer.
    """

    def __init__(
        self,
        rule: Rule,
        state: np.ndarray,
        wrap: bool = False,
        backend: str = "auto",
        echo_decay: float = 0.0,
    ) -> None:
        self.rule = rule
        self.engine = make_engine(rule, state, wrap=wrap, backend=backend)
        self.echo_decay = float(echo_decay)
        self.generation = 0
        self._echo = None
        if self.echo_decay > 0:
            xp = self._xp()
            self._echo = xp.zeros(state.shape, dtype=xp.float32)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_seed(
        cls,
        shape: tuple[int, int],
        rule: Rule | str = "classic",
        seed: str = "single",
        seed_args: dict | None = None,
        level: int | None = None,
        **kwargs,
    ) -> "Automaton":
        """Build from named parts, the way the CLI and gallery do."""
        if isinstance(rule, str):
            if rule not in PRESETS:
                raise ValueError(f"unknown rule {rule!r}; have {sorted(PRESETS)}")
            rule = PRESETS[rule]
        if seed not in SEEDS:
            raise ValueError(f"unknown seed {seed!r}; have {sorted(SEEDS)}")
        args = dict(seed_args or {})
        # The seed level is not a free choice. A lone cell at level L presents
        # each of its eight neighbours with a neighbourhood factor of exactly L,
        # so those neighbours are only born if L lands in the rule's growth band
        # (grow <= L < crowd). Seed below it and the pattern starves on the
        # first generation; seed above it and overcrowding kills it just as
        # fast. The bottom of the growth band is the level that blooms -- which
        # is why 3 was the magic number in the original.
        args.setdefault("level", level if level is not None else rule.grow)
        grid = SEEDS[seed](shape, **args)
        return cls(rule, grid, **kwargs)

    # -- running ---------------------------------------------------------

    def _xp(self):
        if self.engine.name == "cuda":
            import cupy

            return cupy
        return np

    def _update_echo(self) -> None:
        xp = self._xp()
        energy = self.engine.state.astype(xp.float32) / float(self.rule.limit)
        self._echo = xp.maximum(self._echo * self.echo_decay, energy)

    def run(self, generations: int, capture_at: Sequence[int] | None = None) -> list[Frame]:
        """Advance the grid, optionally capturing intermediate generations.

        ``capture_at`` holds absolute generation numbers.  Without an echo
        buffer and without captures, the whole run is a single batch of kernel
        launches with no host synchronisation at all.
        """
        wanted = {int(g) for g in (capture_at or [])}
        frames: list[Frame] = []
        target = self.generation + generations

        if self._echo is not None:
            # The echo decays per generation, so it has to see every one of them.
            # These stay unsynchronised kernel launches regardless.
            for _ in range(generations):
                self.engine.step(1)
                self.generation += 1
                self._update_echo()
                if self.generation in wanted:
                    frames.append(Frame(self.generation, self.snapshot(), self.echo()))
            return frames

        # No echo: batch the run into the largest chunks the captures allow.
        stops = sorted(g for g in wanted if self.generation < g <= target)
        for g in stops + ([target] if not stops or stops[-1] != target else []):
            self.engine.step(g - self.generation)
            self.generation = g
            if g in wanted:
                frames.append(Frame(g, self.snapshot(), None))
        return frames

    def extent(self) -> float:
        """How far the pattern reaches, as a fraction of the grid's short side.

        Computed on the device, so it is cheap enough to poll during a run.
        """
        xp = self._xp()
        alive = self.engine.state > 0
        rows = xp.any(alive, axis=1)
        cols = xp.any(alive, axis=0)
        if not bool(rows.any()):
            return 0.0
        ry = xp.nonzero(rows)[0]
        rx = xp.nonzero(cols)[0]
        span = max(int(ry[-1] - ry[0]), int(rx[-1] - rx[0])) + 1
        return span / float(min(self.engine.state.shape))

    def run_to_extent(
        self, fraction: float = 0.92, max_generations: int = 20000, check_every: int = 25
    ) -> int:
        """Grow until the pattern spans ``fraction`` of the grid, then stop.

        Framing is the whole game with this automaton: too few generations and
        the organism is a speck, too many and it runs off the edge and the
        symmetry is lost.  Rather than hand-tuning a generation count per rule,
        grow until it fills the frame.  Returns the generation reached.
        """
        best, stalled = 0.0, 0
        while self.generation < max_generations:
            self.run(check_every)
            reach = self.extent()
            if reach >= fraction:
                break
            if reach > best:
                best, stalled = reach, 0
            else:
                # A still life or an oscillator will never reach the target.
                stalled += 1
                if stalled >= 20:
                    break
            if reach == 0.0:
                # Extinct. Nothing can revive an empty grid, so stop rather
                # than spin to max_generations.
                raise RuntimeError(
                    f"pattern died out at generation {self.generation}; the seed "
                    f"level must land in the rule's growth band "
                    f"({self.rule.grow} <= level < {self.rule.crowd}) and a dense "
                    f"seed can still overcrowd itself on the first step"
                )
        return self.generation

    # -- reading out -----------------------------------------------------

    def snapshot(self) -> np.ndarray:
        """The current levels, as a host-side ``int8`` array."""
        return self.engine.download()

    def echo(self) -> np.ndarray | None:
        if self._echo is None:
            return None
        xp = self._xp()
        return self._echo if xp is np else xp.asnumpy(self._echo)

    def metrics(self) -> dict[str, int | float]:
        """Cheap summary statistics, useful for spotting dead runs."""
        s = self.snapshot()
        alive = int((s > 0).sum())
        return {
            "generation": self.generation,
            "alive": alive,
            "mass": int(s.sum(dtype=np.int64)),
            "coverage": alive / float(s.size),
        }
