"""Command line entry point: ``python -m goe ...``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .automaton import Automaton
from .backends import device_name, is_available
from .palettes import PALETTES
from .render import colorize, save
from .rules import PRESETS
from .seeds import SEEDS


def _size(text: str) -> tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` into the ``(height, width)`` arrays want."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from exc
    return h, w


def _seed_args(pairs: list[str] | None) -> dict:
    """Parse ``key=value`` seed arguments, keeping ints and floats typed."""
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not _:
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        for cast in (int, float):
            try:
                out[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rule", default="classic", choices=sorted(PRESETS))
    p.add_argument("--seed", default="single", choices=sorted(SEEDS))
    p.add_argument("--seed-arg", action="append", metavar="KEY=VALUE")
    p.add_argument("--size", type=_size, default=(1080, 1920), metavar="WIDTHxHEIGHT")
    p.add_argument("-g", "--generations", type=int, default=600)
    p.add_argument("--wrap", action="store_true", help="torus instead of a dead border")
    p.add_argument("--palette", default="phosphor", choices=sorted(PALETTES))
    p.add_argument("--echo", type=float, default=0.97, metavar="DECAY",
                   help="echo decay per generation; 0 disables the trail")
    p.add_argument("--echo-weight", type=float, default=0.55)
    p.add_argument("--bloom", type=float, default=0.8)
    p.add_argument("--bloom-radius", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--scale", type=int, default=1, help="integer magnification")
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu"])


def cmd_render(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    automaton = Automaton.from_seed(
        args.size,
        rule=args.rule,
        seed=args.seed,
        seed_args=_seed_args(args.seed_arg),
        wrap=args.wrap,
        backend=args.backend,
        echo_decay=args.echo,
    )
    print(f"backend: {automaton.engine.name}"
          + (f" ({device_name()})" if automaton.engine.name == "cuda" else ""))

    capture = sorted({int(g) for g in (args.capture or [])})
    t0 = time.perf_counter()
    frames = automaton.run(args.generations, capture_at=capture)
    elapsed = time.perf_counter() - t0

    cells = args.size[0] * args.size[1] * args.generations
    print(f"{args.generations} generations of {args.size[1]}x{args.size[0]} "
          f"in {elapsed:.2f}s ({cells / elapsed / 1e9:.2f} Gcell/s)")

    palette = PALETTES[args.palette]
    shots = [(f.generation, f.state, f.echo) for f in frames]
    shots.append((automaton.generation, automaton.snapshot(), automaton.echo()))

    for generation, state, echo in shots:
        path = out if generation == automaton.generation else \
            out.with_name(f"{out.stem}-g{generation:05d}{out.suffix}")
        rgb = colorize(
            state, automaton.rule.limit, palette,
            echo=echo, echo_weight=args.echo_weight,
            bloom=args.bloom, bloom_radius=args.bloom_radius, gamma=args.gamma,
        )
        save(rgb, path, scale=args.scale)
        print(f"wrote {path}")

    m = automaton.metrics()
    print(f"alive {m['alive']:,} cells ({m['coverage']:.1%} coverage), mass {m['mass']:,}")
    return 0


def _time_engine(engine, generations: int) -> float:
    """Seconds per generation, measured on the device where there is one.

    Wall-clock timing around a GPU run measures the launch queue and the
    read-back, not the kernel; CUDA events measure the kernel.
    """
    if engine.name == "cuda":
        import cupy as cp

        # Warm up until the clocks have actually ramped, not just until the
        # kernel has compiled. An idle GPU reports a fraction of its throughput
        # for the first few milliseconds of work.
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            engine.step(50)
            engine.synchronize()

        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        engine.step(generations)
        stop.record()
        stop.synchronize()
        return cp.cuda.get_elapsed_time(start, stop) / 1e3 / generations

    engine.step(5)  # let NumPy's buffers settle
    t0 = time.perf_counter()
    engine.step(generations)
    return (time.perf_counter() - t0) / generations


def cmd_bench(args: argparse.Namespace) -> int:
    from .backends import make_engine
    from .seeds import SEEDS

    print(f"cuda available: {is_available()}"
          + (f" -- {device_name()}" if is_available() else ""))
    print(f"{'grid':>10} {'backend':>8} {'ms/gen':>10} {'Gcell/s':>9} "
          f"{'GB/s':>8} {'speedup':>8}")

    rule = PRESETS["classic"]
    for side in args.sizes:
        grid = SEEDS["mandala"]((side, side), level=3, size=32, seed=1)
        cpu_per_gen = None
        for backend in ("cpu", "cuda"):
            if backend == "cuda" and not is_available():
                continue
            if backend == "cpu" and side > args.cpu_limit:
                continue  # NumPy past ~2048^2 is minutes, not seconds
            engine = make_engine(rule, grid, backend=backend)
            per_gen = _time_engine(engine, args.generations)
            if backend == "cpu":
                cpu_per_gen = per_gen
            cells = side * side
            speedup = (f"{cpu_per_gen / per_gen:.0f}x"
                       if backend == "cuda" and cpu_per_gen else "-")
            # One byte in, one byte out per cell: the traffic floor for a
            # stencil that reuses its reads perfectly.
            print(f"{side}^2".rjust(10)
                  + f"{backend:>9}{per_gen * 1e3:>10.3f}{cells / per_gen / 1e9:>10.2f}"
                  + f"{cells * 2 / per_gen / 1e9:>9.1f}{speedup:>9}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("rules:")
    for name, rule in PRESETS.items():
        print(f"  {name:<10} limit={rule.limit:<3} starve={rule.starve} "
              f"grow={rule.grow} crowd={rule.crowd} slope={rule.slope}")
    print("\nseeds:   " + ", ".join(sorted(SEEDS)))
    print("palettes:" + ", ".join(sorted(PALETTES)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="goe", description="Game of Evolution: levelled cellular automaton.")
    parser.add_argument("--version", action="version", version=f"goe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_render = sub.add_parser("render", help="run the automaton and write a PNG")
    _add_common(p_render)
    p_render.add_argument("-o", "--out", default="out.png")
    p_render.add_argument("--capture", type=int, action="append",
                          help="also write this generation; repeatable")
    p_render.set_defaults(func=cmd_render)

    p_gallery = sub.add_parser("gallery", help="render the curated screenshot set")
    p_gallery.add_argument("-o", "--out", default="screenshots")
    p_gallery.add_argument("--only", action="append", help="render just these shots")
    p_gallery.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu"])
    from .gallery import cmd_gallery
    p_gallery.set_defaults(func=cmd_gallery)

    p_bench = sub.add_parser("bench", help="compare the CUDA and NumPy backends")
    p_bench.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p_bench.add_argument("-g", "--generations", type=int, default=200)
    p_bench.add_argument("--cpu-limit", type=int, default=2048,
                         help="skip the NumPy backend above this grid side")
    p_bench.set_defaults(func=cmd_bench)

    p_list = sub.add_parser("list", help="show the available rules, seeds and palettes")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
