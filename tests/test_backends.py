"""The CUDA kernel is only useful if it is indistinguishable from the reference.

These are exact-equality tests, not approximate ones: the automaton is integer
arithmetic end to end, so any divergence is a bug, not rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from goe.backends import CpuEngine, is_available
from goe.backends.cuda import CudaEngine
from goe.rules import PRESETS, Rule
from goe.seeds import SEEDS

requires_cuda = pytest.mark.skipif(not is_available(), reason="no CUDA device")


def test_classic_rule_matches_the_2017_java_thresholds():
    # From goe.maths.Matrix.applyRules: N<2 -> -1, N==2 -> 0, N==3 -> +1,
    # N>=4 -> -(N-4).
    table = Rule.classic().delta_table()
    assert list(table[:9]) == [-1, -1, 0, 1, 0, -1, -2, -3, -4]


def test_a_single_cell_blooms_rather_than_dying():
    """The headline behaviour: one level-3 cell is enough."""
    a = CpuEngine(Rule.classic(), SEEDS["single"]((201, 201), level=3))
    a.step(200)
    s = a.state
    assert (s > 0).sum() > 500
    # And it stays eight-fold symmetric forever.
    assert np.array_equal(s, np.fliplr(s))
    assert np.array_equal(s, np.flipud(s))
    assert np.array_equal(s, s.T)


def test_dead_border_is_the_default():
    """A seed on the edge must not wrap around when wrap is off."""
    g = np.zeros((32, 32), np.int8)
    g[0, 0] = 3
    a = CpuEngine(Rule.classic(), g, wrap=False)
    a.step(1)
    assert a.state[-1, -1] == 0


def test_wrap_makes_it_a_torus():
    g = np.zeros((32, 32), np.int8)
    g[0, 0] = 3
    a = CpuEngine(Rule.classic(), g, wrap=True)
    a.step(1)
    assert a.state[-1, -1] > 0


@requires_cuda
@pytest.mark.parametrize("rule_name", sorted(PRESETS))
@pytest.mark.parametrize("wrap", [False, True])
def test_cuda_matches_cpu_exactly(rule_name, wrap):
    rule = PRESETS[rule_name]
    rng = np.random.default_rng(7)
    # A non-square, non-block-aligned grid, to catch indexing and halo bugs.
    grid = rng.integers(0, rule.limit + 1, size=(137, 251), dtype=np.int8)

    cpu = CpuEngine(rule, grid, wrap=wrap)
    gpu = CudaEngine(rule, grid, wrap=wrap)
    for _ in range(20):
        cpu.step(1)
        gpu.step(1)
        assert np.array_equal(cpu.state, gpu.download())


@requires_cuda
def test_cuda_matches_cpu_on_a_long_bloom():
    """Batched launches with buffer ping-pong must not drift from the reference."""
    rule = PRESETS["classic"]
    grid = SEEDS["mandala"]((256, 256), level=3, size=24, seed=3)
    cpu = CpuEngine(rule, grid)
    gpu = CudaEngine(rule, grid)
    cpu.step(300)
    gpu.step(300)
    assert np.array_equal(cpu.state, gpu.download())
