import numpy as np
import pytest

from volsurface import optim


def test_nelder_mead_finds_quadratic_minimum():
    def f(x):
        return (x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2 + 5.0

    x, fval = optim.nelder_mead(f, np.array([0.0, 0.0]), bounds=[(-10, 10), (-10, 10)], max_iter=500)
    assert x[0] == pytest.approx(3.0, abs=1e-3)
    assert x[1] == pytest.approx(-2.0, abs=1e-3)
    assert fval == pytest.approx(5.0, abs=1e-3)


def test_nelder_mead_respects_bounds():
    def f(x):
        return -(x[0])  # unbounded minimum would run to +infinity

    x, _ = optim.nelder_mead(f, np.array([0.0]), bounds=[(-1.0, 1.0)], max_iter=300)
    assert x[0] <= 1.0 + 1e-6


def test_multi_start_beats_or_matches_single_start_on_a_multimodal_function():
    """Rastrigin-like 1D function with many local minima; multi-start should
    find a result at least as good as a single fixed start."""
    def f(x):
        return 10 + (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]))

    x_single, f_single = optim.nelder_mead(f, np.array([4.5]), bounds=[(-5, 5)], max_iter=200)
    x_multi, f_multi = optim.multi_start_minimize(f, bounds=[(-5, 5)], n_starts=10, seed=0, max_iter=200)
    assert f_multi <= f_single + 1e-6


def test_multi_start_uses_extra_starts():
    """A caller-supplied extra_starts point at the exact global optimum
    should be picked up (best_f == optimum) even with zero random starts."""
    def f(x):
        return (x[0] - 1.234) ** 2

    x, fval = optim.multi_start_minimize(f, bounds=[(-10, 10)], n_starts=1, seed=0,
                                          extra_starts=[[1.234]], max_iter=50)
    assert fval < 1e-6
