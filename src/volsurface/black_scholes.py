"""Black-Scholes(-Merton) pricing, Greeks, and a hand-rolled implied-vol solver.

Deliberately dependency-light: the implied-vol inversion is a robust
Newton-Raphson iteration (using the analytic vega) with a bisection fallback,
rather than a call to ``scipy.optimize.brentq``. Implied-vol inversion is a
foundational building block for everything else in this repo (the whole SVI
surface is built on it) -- it is worth owning outright and testing in
isolation rather than trusting a black box for it.

All functions are vectorised over numpy arrays.
"""
from __future__ import annotations

import math

import numpy as np

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_erf = np.vectorize(math.erf)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # erf-based CDF, accurate to ~1e-15, no scipy needed.
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def d1_d2(spot: np.ndarray, strike: np.ndarray, rate: float, div_yield: float,
          vol: np.ndarray, ttm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vol = np.maximum(vol, 1e-8)
    ttm = np.maximum(ttm, 1e-8)
    sqrt_t = np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * ttm) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def bs_price(spot, strike, rate, div_yield, vol, ttm, option_type="call"):
    """European option price under Black-Scholes-Merton (continuous dividend yield)."""
    spot, strike, vol, ttm = map(lambda a: np.asarray(a, dtype=float), (spot, strike, vol, ttm))
    d1, d2 = d1_d2(spot, strike, rate, div_yield, vol, ttm)
    disc_r = np.exp(-rate * ttm)
    disc_q = np.exp(-div_yield * ttm)
    call = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    if option_type == "call":
        return call
    if option_type == "put":
        # Put-call parity, avoids re-deriving a second closed form.
        return call - spot * disc_q + strike * disc_r
    raise ValueError(f"unknown option_type={option_type!r}")


def vega(spot, strike, rate, div_yield, vol, ttm):
    """d(price)/d(vol), same for calls and puts."""
    spot, strike, vol, ttm = map(lambda a: np.asarray(a, dtype=float), (spot, strike, vol, ttm))
    d1, _ = d1_d2(spot, strike, rate, div_yield, vol, ttm)
    disc_q = np.exp(-div_yield * ttm)
    return spot * disc_q * _norm_pdf(d1) * np.sqrt(np.maximum(ttm, 1e-8))


def greeks(spot, strike, rate, div_yield, vol, ttm, option_type="call"):
    """Closed-form Delta/Gamma/Vega/Theta/Rho."""
    spot, strike, vol, ttm = map(lambda a: np.asarray(a, dtype=float), (spot, strike, vol, ttm))
    d1, d2 = d1_d2(spot, strike, rate, div_yield, vol, ttm)
    disc_r = np.exp(-rate * ttm)
    disc_q = np.exp(-div_yield * ttm)
    pdf_d1 = _norm_pdf(d1)
    sqrt_t = np.sqrt(np.maximum(ttm, 1e-8))

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    veg = spot * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
                 - rate * strike * disc_r * _norm_cdf(d2)
                 + div_yield * spot * disc_q * _norm_cdf(d1))
        rho = strike * ttm * disc_r * _norm_cdf(d2)
    elif option_type == "put":
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
                 + rate * strike * disc_r * _norm_cdf(-d2)
                 - div_yield * spot * disc_q * _norm_cdf(-d1))
        rho = -strike * ttm * disc_r * _norm_cdf(-d2)
    else:
        raise ValueError(f"unknown option_type={option_type!r}")

    return {"delta": delta, "gamma": gamma, "vega": veg, "theta": theta, "rho": rho}


def implied_vol(price, spot, strike, rate, div_yield, ttm, option_type="call",
                 lo=1e-4, hi=5.0, tol=1e-8, max_newton_iter=50, max_bisect_iter=100):
    """Invert Black-Scholes for implied vol, scalar-by-scalar (vectorise via np.vectorize
    at the call site if needed).

    Strategy: Newton-Raphson seeded at a Brenner-Subrahmanyam-style initial guess,
    using the analytic vega. Newton on implied vol is well known to occasionally
    diverge or overshoot into vol<=0 for deep ITM/OTM or near-expiry quotes (vega
    -> 0), so every iteration is guarded, and a bisection fallback on the
    (monotonic in vol) pricing function guarantees convergence whenever a
    no-arbitrage price is bracketed by [lo, hi].
    """
    price = float(price)
    spot, strike, ttm = float(spot), float(strike), float(ttm)

    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan  # no time value left / arbitrage-violating quote, cannot invert

    # Initial guess (Brenner-Subrahmanyam approximation).
    vol = np.sqrt(2 * np.pi / max(ttm, 1e-8)) * price / spot
    vol = min(max(vol, lo), hi)

    for _ in range(max_newton_iter):
        px = float(bs_price(spot, strike, rate, div_yield, vol, ttm, option_type))
        v = float(vega(spot, strike, rate, div_yield, vol, ttm))
        diff = px - price
        if abs(diff) < tol:
            return vol
        if v < 1e-10:
            break  # vega too small, Newton step unreliable -> fall through to bisection
        step = diff / v
        new_vol = vol - step
        if not (lo < new_vol < hi):
            break
        vol = new_vol

    # Bisection fallback: pricing function is monotonically increasing in vol.
    f_lo = float(bs_price(spot, strike, rate, div_yield, lo, ttm, option_type)) - price
    f_hi = float(bs_price(spot, strike, rate, div_yield, hi, ttm, option_type)) - price
    if f_lo * f_hi > 0:
        return np.nan  # price not bracketed in [lo, hi] -> reject rather than guess
    a, b = lo, hi
    for _ in range(max_bisect_iter):
        mid = 0.5 * (a + b)
        f_mid = float(bs_price(spot, strike, rate, div_yield, mid, ttm, option_type)) - price
        if abs(f_mid) < tol or (b - a) < 1e-10:
            return mid
        if f_lo * f_mid <= 0:
            b, f_hi = mid, f_mid
        else:
            a, f_lo = mid, f_mid
    return 0.5 * (a + b)
