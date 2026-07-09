"""Raw SVI (Stochastic Volatility Inspired) parameterization and per-maturity
slice calibration, following Gatheral (2004) / Gatheral & Jacquier (2014).

Total implied variance as a function of log-moneyness k = log(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)**2 + sigma**2) )

with implied vol sigma_impl(k, T) = sqrt(w(k) / T).

Each expiry is calibrated independently (a "slice"); ``calibrate_svi_surface``
loops this over every maturity in the dataset. This is the standard
industry-practice granularity: SVI slices are almost always fit maturity by
maturity, then optionally checked/repaired for calendar-spread arbitrage
across maturities (see :func:`check_calendar_arbitrage`).

Known property, not a bug: raw SVI's 5 parameters are NOT uniquely
identified from a single noiseless curve -- several visibly different
(a, b, rho, m, sigma) tuples can reproduce essentially the same w(k) curve
over the traded strike range (see tests/test_svi.py, which deliberately
asserts on the *fitted curve*, not on individual recovered parameters, for
exactly this reason). This is why the raw parameters here should not be
over-interpreted term by term; what is stable and meaningful is the curve
itself, and the (better-identified) at-the-money level/skew/curvature it
implies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import optim

# Parameter order used throughout this module: (a, b, rho, m, sigma)
_BOUNDS = [
    (-2.0, 2.0),   # a: overall variance level
    (1e-6, 4.0),   # b: angle/slope of the wings
    (-0.999, 0.999),  # rho: correlation / skew
    (-2.0, 2.0),   # m: horizontal shift of the smile minimum
    (1e-4, 2.0),   # sigma: curvature at the money
]


@dataclass
class SVISlice:
    ttm: float
    forward: float
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse_vol_pts: float  # calibration fit quality, in *volatility points* (not variance)
    n_quotes: int

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) +
                                   np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k):
        w = np.maximum(self.total_variance(k), 1e-12)
        return np.sqrt(w / self.ttm)

    def is_arbitrage_free_here(self, k_grid=None) -> bool:
        if k_grid is None:
            k_grid = np.linspace(self.m - 3 * self.sigma - 1, self.m + 3 * self.sigma + 1, 400)
        return bool(np.all(_butterfly_g(self, k_grid) >= -1e-6))


def _min_total_variance(params) -> float:
    """Minimum of w(k) over all k, attained at k* = m + rho*sigma/sqrt(1-rho^2).
    Gatheral's no (butterfly) arbitrage necessary condition is w_min >= 0.
    """
    a, b, rho, m, sigma = params
    return a + b * sigma * np.sqrt(max(1.0 - rho ** 2, 0.0))


def calibrate_svi_slice(log_moneyness, total_variance, ttm, forward,
                         weights=None, n_starts=8, seed=0, max_iter=500) -> SVISlice:
    """Least-squares calibration of one SVI slice to market (k, w) points.

    Objective is a *weighted* sum of squared errors in total variance, with a
    soft penalty enforcing the butterfly no-arbitrage floor (w_min >= 0) so
    the optimiser is steered away from arbitrageable fits rather than merely
    checked after the fact.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w_mkt = np.asarray(total_variance, dtype=float)
    if weights is None:
        weights = np.ones_like(k)
    weights = np.asarray(weights, dtype=float)

    a0 = max(np.median(w_mkt), 1e-4)
    x0 = np.array([a0 * 0.5, 0.3, -0.4, 0.0, 0.2])

    def loss(params):
        a, b, rho, m, sigma = params
        w_model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        resid = weights * (w_model - w_mkt)
        sse = float(np.sum(resid ** 2))
        arb_floor = _min_total_variance(params)
        penalty = 1e4 * min(arb_floor, 0.0) ** 2
        return sse + penalty

    best_x, _ = optim.multi_start_minimize(loss, bounds=_BOUNDS, n_starts=n_starts, seed=seed,
                                            extra_starts=[x0], max_iter=max_iter,
                                            fatol=1e-9, xatol=1e-7)
    a, b, rho, m, sigma = best_x

    w_fit = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    vol_fit = np.sqrt(np.maximum(w_fit, 1e-12) / ttm)
    vol_mkt = np.sqrt(np.maximum(w_mkt, 1e-12) / ttm)
    rmse_vol_pts = float(np.sqrt(np.mean((vol_fit - vol_mkt) ** 2)))

    return SVISlice(ttm=ttm, forward=forward, a=a, b=b, rho=rho, m=m, sigma=sigma,
                     rmse_vol_pts=rmse_vol_pts, n_quotes=len(k))


def calibrate_svi_surface(iv_df: pd.DataFrame, seed=0, **calib_kwargs) -> dict[float, SVISlice]:
    """Calibrate one SVI slice per unique maturity in ``iv_df``.

    ``iv_df`` must have columns: ttm, forward, strike, implied_vol (see
    data.py for the expected schema coming out of the data-cleaning step).
    ``**calib_kwargs`` (e.g. ``n_starts``, ``max_iter``) are forwarded to
    :func:`calibrate_svi_slice` for every maturity -- handy to trade fit
    quality for speed (e.g. in a test suite or a quick interactive check).
    """
    slices = {}
    for ttm, grp in iv_df.groupby("ttm"):
        forward = float(grp["forward"].iloc[0])
        k = np.log(grp["strike"].to_numpy() / forward)
        w = (grp["implied_vol"].to_numpy() ** 2) * ttm
        slices[float(ttm)] = calibrate_svi_slice(k, w, ttm, forward, seed=seed, **calib_kwargs)
    return slices


def _butterfly_g(svi_slice: SVISlice, k) -> np.ndarray:
    """Gatheral & Jacquier (2014) butterfly-arbitrage density proxy g(k).

    g(k) >= 0 for all k is a necessary and sufficient condition (given the
    slice is otherwise well-behaved) for the *risk-neutral density implied by
    this single slice* to be non-negative everywhere, i.e. no static
    butterfly arbitrage within that maturity. Derivatives of w are computed
    by central finite differences -- deliberately simple and easy to audit,
    since this is a correctness check, not a hot path.
    """
    k = np.asarray(k, dtype=float)
    h = 1e-5
    w = svi_slice.total_variance(k)
    wp = (svi_slice.total_variance(k + h) - svi_slice.total_variance(k - h)) / (2 * h)
    wpp = (svi_slice.total_variance(k + h) - 2 * w + svi_slice.total_variance(k - h)) / (h ** 2)

    w_safe = np.maximum(w, 1e-10)
    term1 = (1 - k * wp / (2 * w_safe)) ** 2
    term2 = (wp ** 2 / 4) * (1 / w_safe + 0.25)
    term3 = wpp / 2
    return term1 - term2 + term3


def check_calendar_arbitrage(slices: dict[float, SVISlice], k_grid=None) -> pd.DataFrame:
    """Check that total variance w(k) is non-decreasing in T at fixed k
    (necessary condition for no calendar-spread arbitrage across maturities).

    Returns a small report: for each pair of consecutive maturities, the worst
    (most negative) violation of w(T2, k) - w(T1, k) >= 0 over the grid.
    """
    ttms = sorted(slices.keys())
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 200)
    rows = []
    for t1, t2 in zip(ttms[:-1], ttms[1:]):
        w1 = slices[t1].total_variance(k_grid)
        w2 = slices[t2].total_variance(k_grid)
        diff = w2 - w1
        rows.append({"ttm_short": t1, "ttm_long": t2,
                      "min_diff": float(diff.min()), "violated": bool(diff.min() < -1e-6)})
    return pd.DataFrame(rows)
