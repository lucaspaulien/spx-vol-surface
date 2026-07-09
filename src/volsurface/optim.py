"""A small, dependency-free Nelder-Mead simplex optimiser.

Both the SVI slice calibration and the Heston surface calibration are
non-linear least-squares problems in 4-5 parameters. ``scipy.optimize`` would
be the standard industry tool for this (and remains a drop-in upgrade -- see
README "Possible extensions"), but this repo intentionally ships its own
minimiser so the whole calibration pipeline runs with numpy/pandas alone,
with no network access and no compiled dependency required to reproduce a
single result end to end.

Multi-start wrapper included: Nelder-Mead is a local, derivative-free method,
so ``multi_start_minimize`` restarts it from several random points inside the
given bounds and keeps the best converged result, to reduce the risk of
reporting a local optimum as the global calibration.
"""
from __future__ import annotations

import numpy as np


def nelder_mead(fun, x0, bounds=None, max_iter=2000, xatol=1e-8, fatol=1e-10,
                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Minimise ``fun`` (R^n -> R) starting from ``x0``.

    ``bounds``: optional list of (lo, hi) per dimension. Out-of-bounds points
    are penalised with a large finite value rather than rejected outright, so
    the simplex can still move smoothly along a boundary.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    def penalised(x):
        if bounds is not None:
            penalty = 0.0
            for xi, (lo, hi) in zip(x, bounds):
                if xi < lo:
                    penalty += 1e6 * (lo - xi) ** 2
                elif xi > hi:
                    penalty += 1e6 * (xi - hi) ** 2
            if penalty > 0:
                return 1e8 + penalty
        val = fun(x)
        return val if np.isfinite(val) else 1e12

    # Initial simplex: x0 plus n perturbed points. When bounds are supplied the
    # perturbation is a fraction of each dimension's bound width (robust even
    # when x0[i] is 0 or near a bound); otherwise fall back to a relative 5%
    # bump of x0 itself.
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        step_sizes = 0.15 * (b[:, 1] - b[:, 0])
    else:
        step_sizes = np.where(x0 != 0, np.abs(x0) * 0.05, 0.05)
    for i in range(n):
        perturbed = x0.copy()
        perturbed[i] += step_sizes[i]
        simplex[i + 1] = perturbed
    f_vals = np.array([penalised(p) for p in simplex])

    for _ in range(max_iter):
        order = np.argsort(f_vals)
        simplex, f_vals = simplex[order], f_vals[order]

        if np.abs(f_vals[-1] - f_vals[0]) < fatol and \
           np.max(np.abs(simplex[1:] - simplex[0])) < xatol:
            break

        centroid = simplex[:-1].mean(axis=0)
        worst, f_worst = simplex[-1], f_vals[-1]

        xr = centroid + alpha * (centroid - worst)
        f_r = penalised(xr)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1], f_vals[-1] = xr, f_r
        elif f_r < f_vals[0]:
            xe = centroid + gamma * (xr - centroid)
            f_e = penalised(xe)
            if f_e < f_r:
                simplex[-1], f_vals[-1] = xe, f_e
            else:
                simplex[-1], f_vals[-1] = xr, f_r
        else:
            xc = centroid + rho * (worst - centroid)
            f_c = penalised(xc)
            if f_c < f_worst:
                simplex[-1], f_vals[-1] = xc, f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = penalised(simplex[i])

    order = np.argsort(f_vals)
    return simplex[order[0]], f_vals[order[0]]


def multi_start_minimize(fun, bounds, n_starts=8, seed=0, extra_starts=None, **kwargs):
    """Random-restart wrapper around :func:`nelder_mead`.

    Returns ``(best_x, best_f)`` across ``n_starts`` random initial points
    drawn uniformly from ``bounds``, plus the midpoint of the bounds and any
    caller-supplied ``extra_starts`` (e.g. a financially-sensible initial
    guess -- pure random search over wide parameter bounds is a poor way to
    find a needle-in-a-haystack optimum such as Heston's, see heston.py).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    starts = [0.5 * (lo + hi)]
    if extra_starts:
        starts.extend(np.asarray(s, dtype=float) for s in extra_starts)
    n_random = max(n_starts - len(starts), 0)
    for _ in range(n_random):
        starts.append(rng.uniform(lo, hi))

    best_x, best_f = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, lo, hi)
        x, f = nelder_mead(fun, x0, bounds=bounds, **kwargs)
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f
