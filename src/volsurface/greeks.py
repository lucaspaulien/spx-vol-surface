"""Greeks off the calibrated SVI surface and off the calibrated Heston model.

Two genuinely different notions of "Greek" are computed here, and the
difference is the point, not an oversight:

* **SVI "smile-adjusted" Greeks** -- bump the spot (or time), re-read the
  implied vol at the *new* log-moneyness off the SAME calibrated SVI slice,
  and reprice with Black-Scholes at that vol. This captures the sticky-strike
  smile dynamics (Delta already "knows" that moving spot changes which point
  of the smile you are sitting on), unlike the naive textbook BS Delta which
  holds vol fixed. This is deliberately the sticky-strike convention, not
  sticky-delta -- documented rather than silently assumed, see README.
* **Heston finite-difference Greeks** -- Heston has closed-form Delta/Gamma
  available via the characteristic function (differentiate under the
  integral sign), but this repo computes them by straightforward
  bump-and-reprice on the already-implemented, already-tested pricing
  function instead. That is a deliberate simplicity-over-elegance choice
  (one code path to trust, not two), explicitly logged as a limitation in
  README rather than presented as a closed-form Greek it is not.
"""
from __future__ import annotations

import numpy as np

from . import black_scholes as bs
from . import heston as hst


def svi_smile_greeks(svi_slice, S0, r, q, option_type="call", bump_rel=1e-3, bump_t_days=1.0, strikes=None):
    """Delta/Gamma/Vega/Theta at-the-strike-grid of ``svi_slice``, using the
    calibrated smile to re-price after each bump (sticky-strike).

    ``strikes``: if omitted, defaults to a 25-point grid spanning +/-3 sigma
    around the fitted slice's own (m, sigma) -- convenient for plotting the
    whole smile, but note this grid is centered on the *fitted curve's*
    minimum, not necessarily on-the-money, and its resolution near any
    specific strike (e.g. spot) varies with the fitted (m, sigma) -- which
    is not uniquely identified (see svi.py). Callers that need Greeks at a
    *specific* strike (e.g. true at-the-money, ``forward``) should pass it
    explicitly rather than picking the nearest point out of the auto-grid.
    """
    ttm = svi_slice.ttm
    forward = svi_slice.forward
    if strikes is None:
        strikes = forward * np.exp(np.linspace(svi_slice.m - 3 * svi_slice.sigma,
                                                svi_slice.m + 3 * svi_slice.sigma, 25))
    else:
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    def price_at(spot, ttm_):
        fwd = spot * np.exp((r - q) * ttm_) if ttm_ > 0 else spot
        k = np.log(strikes / fwd)
        vol = svi_slice.implied_vol(k) if ttm_ == ttm else \
            np.sqrt(np.maximum(svi_slice.total_variance(k), 1e-12) / max(ttm_, 1e-8))
        return bs.bs_price(spot, strikes, r, q, vol, ttm_, option_type)

    h_s = S0 * bump_rel
    p_up = price_at(S0 + h_s, ttm)
    p_mid = price_at(S0, ttm)
    p_dn = price_at(S0 - h_s, ttm)
    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * p_mid + p_dn) / (h_s ** 2)

    h_t = bump_t_days / 365.0
    theta = (price_at(S0, max(ttm - h_t, 1e-6)) - p_mid) / h_t  # per year; divide by 365 for per-day

    h_vol = 1e-4
    vol_at_mid = svi_slice.implied_vol(np.log(strikes / forward))
    p_vol_up = bs.bs_price(S0, strikes, r, q, vol_at_mid + h_vol, ttm, option_type)
    vega = (p_vol_up - p_mid) / h_vol

    return {"strike": strikes, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def heston_fd_greeks(params: hst.HestonParams, S0, K, r, q, T, option_type="call",
                      bump_rel=1e-3, bump_v0=1e-4, bump_t_days=1.0, **pricer_kwargs):
    """Bump-and-reprice Delta/Gamma/Vega(-like, bump v0)/Theta from a
    calibrated Heston model. See module docstring for why finite differences
    rather than the closed-form characteristic-function derivative."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    kappa, theta, sigma_v, rho, v0 = params.as_tuple()
    pricer_kwargs = dict(pricer_kwargs)
    pricer_kwargs.setdefault("n_points", 600)

    def price(spot, vzero, ttm_):
        if ttm_ <= 0:
            return np.maximum(spot - K, 0.0) if option_type == "call" else np.maximum(K - spot, 0.0)
        return np.atleast_1d(hst.heston_call_price(spot, K, r, q, ttm_, kappa, theta, sigma_v, rho, vzero,
                                                     **pricer_kwargs))

    h_s = S0 * bump_rel
    p_up = price(S0 + h_s, v0, T)
    p_mid = price(S0, v0, T)
    p_dn = price(S0 - h_s, v0, T)
    delta = (p_up - p_dn) / (2 * h_s)
    gamma = (p_up - 2 * p_mid + p_dn) / (h_s ** 2)

    v0_bumped = min(v0 + bump_v0, 0.9)
    p_v0_up = price(S0, v0_bumped, T)
    vega_v0 = (p_v0_up - p_mid) / (v0_bumped - v0)

    h_t = bump_t_days / 365.0
    theta_g = (price(S0, v0, max(T - h_t, 1e-6)) - p_mid) / h_t

    return {"strike": K, "price": p_mid, "delta": delta, "gamma": gamma,
            "vega_v0": vega_v0, "theta": theta_g}
