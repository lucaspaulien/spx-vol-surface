"""Heston (1993) stochastic volatility model: characteristic function pricing
and whole-surface calibration.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t,      dW1 dW2 = rho dt

Pricing uses the "Little Heston Trap" characteristic-function formulation
(Albrecher, Mayer, Schoutens & Tistaert, 2007) -- the numerically stable
branch choice for the complex square root/logarithm in the original Heston
(1993) formula, well documented to blow up for long maturities / certain
parameter regions if implemented the "naive" way. The call price is then
recovered via the standard P1/P2 decomposition, integrated numerically on a
fixed grid (no scipy.integrate dependency, see optim.py for the
calibration-side rationale).

Performance note: the characteristic function phi(u) does NOT depend on
strike, only on maturity and the model parameters -- so for a batch of
strikes at the same maturity, phi is evaluated ONCE on the u-grid and the
strike-dependent part (``exp(-i u ln K)``) is applied as a vectorised outer
product. This is what makes whole-surface calibration (hundreds of
(K, T) quotes, each loss evaluation re-pricing everything) tractable without
a compiled/FFT backend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import optim


# Parameter order: (kappa, theta, sigma_v, rho, v0). Bounds are deliberately
# tight around financially-plausible equity-index values (rather than the
# widest range the model formally allows): Heston calibration is a famously
# non-convex, multi-basin problem (see calibrate_heston_surface docstring),
# and a derivative-free local optimiser restarted inside an unrealistically
# wide box (e.g. theta up to 2.0 <=> 141% long-run vol) wastes almost all of
# its restarts in regions no real index surface would ever calibrate to.
_BOUNDS = [
    (0.2, 8.0),      # kappa: mean-reversion speed
    (1e-3, 0.25),    # theta: long-run variance (<=> up to 50% long-run vol)
    (1e-2, 1.5),     # sigma_v: vol-of-vol
    (-0.95, 0.95),   # rho: spot/vol correlation
    (1e-3, 0.25),    # v0: initial variance (<=> up to 50% instantaneous vol)
]


def _char_func(u, S0, r, q, kappa, theta, sigma_v, rho, v0, T):
    """phi(u) = E[exp(i u ln S_T)] under the risk-neutral measure, Little-Trap form.
    Vectorised over ``u`` (any shape, dtype promoted to complex)."""
    u = np.asarray(u, dtype=complex)
    x0 = np.log(S0)
    xi = kappa - rho * sigma_v * 1j * u
    d = np.sqrt(xi ** 2 + (sigma_v ** 2) * (u ** 2 + 1j * u))
    g2 = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    C = ((r - q) * 1j * u * T
         + (kappa * theta / sigma_v ** 2) * ((xi - d) * T - 2.0 * np.log((1 - g2 * exp_dT) / (1 - g2))))
    D = ((xi - d) / sigma_v ** 2) * ((1 - exp_dT) / (1 - g2 * exp_dT))
    return np.exp(C + D * v0 + 1j * u * x0)


def heston_call_price(S0, K, r, q, T, kappa, theta, sigma_v, rho, v0,
                       u_max=150.0, n_points=1000):
    """European call price(s) under Heston for one maturity T and one or more
    strikes K, sharing a single characteristic-function evaluation on the
    u-grid across all strikes (see module docstring)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    ln_K = np.log(K)

    u = np.linspace(1e-6, u_max, n_points)
    phi_2 = _char_func(u, S0, r, q, kappa, theta, sigma_v, rho, v0, T)              # (n_points,)
    phi_num = _char_func(u - 1j, S0, r, q, kappa, theta, sigma_v, rho, v0, T)
    phi_den = _char_func(np.array([-1j]), S0, r, q, kappa, theta, sigma_v, rho, v0, T)[0]
    phi_1 = phi_num / phi_den

    # Outer product over (u, strike): shape (n_points, n_strikes).
    phase = np.exp(-1j * np.outer(u, ln_K))
    integrand_1 = np.real(phase * (phi_1 / (1j * u))[:, None])
    integrand_2 = np.real(phase * (phi_2 / (1j * u))[:, None])

    p1 = 0.5 + np.trapezoid(integrand_1, u, axis=0) / np.pi
    p2 = 0.5 + np.trapezoid(integrand_2, u, axis=0) / np.pi

    prices = S0 * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    return prices if prices.size > 1 else float(prices[0])


@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float
    rmse_vol_pts: float
    feller_ratio: float  # 2*kappa*theta / sigma_v^2 ; >= 1 satisfies the Feller condition

    def as_tuple(self):
        return (self.kappa, self.theta, self.sigma_v, self.rho, self.v0)


def calibrate_heston_surface(iv_df: pd.DataFrame, S0: float, r: float, q: float,
                              n_starts=8, seed=0, max_iter=250,
                              u_max=150.0, n_points=800) -> HestonParams:
    """Single global calibration of the 5 Heston parameters against every
    (strike, maturity) quote in ``iv_df`` at once (unlike SVI, which is fit
    maturity by maturity) -- Heston's whole premise is that ONE parameter set
    explains the entire surface, so that is exactly what is optimised here.

    Performance: the objective is a *vega-weighted price* least-squares
    (weight = 1/market-vega), not a full implied-vol re-inversion at every
    evaluation. To first order, price_error / vega ~= vol_error, so this is
    the standard trick (see e.g. Cont & da Fonseca-style calibration setups)
    for approximating an implied-vol-space objective while paying for a
    Newton/bisection IV solve only ONCE per quote (up front, to get the
    market vega), not thousands of times inside the optimiser loop. The
    reported ``rmse_vol_pts`` is nonetheless computed from a *real* IV
    inversion of the final calibrated prices -- the speed shortcut only
    applies to what the optimiser sees while searching, not to the number
    that gets reported.
    """
    from . import black_scholes as bs

    groups = []
    for T, g in iv_df.groupby("ttm"):
        strikes = g["strike"].to_numpy()
        iv_mkt = g["implied_vol"].to_numpy()
        mkt_prices = np.array([bs.bs_price(S0, k, r, q, v, T, "call") for k, v in zip(strikes, iv_mkt)])
        mkt_vega = np.array([bs.vega(S0, k, r, q, v, T) for k, v in zip(strikes, iv_mkt)])
        mkt_vega = np.maximum(mkt_vega, 1e-6)
        groups.append((float(T), strikes, mkt_prices, mkt_vega, iv_mkt))

    # Financially-sensible starting point: v0 = theta = ATM variance (from the
    # shortest-dated slice), kappa/sigma_v/rho at typical equity-index values.
    # This one deterministic start matters far more than extra random restarts
    # for a landscape this non-convex (see module & _BOUNDS docstrings).
    shortest_ttm, shortest_strikes, _sp, _sv, shortest_iv = min(groups, key=lambda g: g[0])
    atm_idx = int(np.argmin(np.abs(shortest_strikes - S0)))
    atm_var = float(shortest_iv[atm_idx] ** 2)
    smart_start = [2.0, atm_var, 0.5, -0.5, atm_var]

    def loss(params):
        kappa, theta, sigma_v, rho, v0 = params
        sse, n = 0.0, 0
        for T, strikes, mkt_prices, mkt_vega, _iv_mkt in groups:
            model_prices = np.atleast_1d(heston_call_price(
                S0, strikes, r, q, T, kappa, theta, sigma_v, rho, v0, u_max=u_max, n_points=n_points))
            vol_err_approx = (model_prices - mkt_prices) / mkt_vega
            sse += float(np.sum(vol_err_approx ** 2))
            n += len(strikes)
        return sse / max(n, 1)

    best_x, _ = optim.multi_start_minimize(loss, bounds=_BOUNDS, n_starts=n_starts, seed=seed,
                                            extra_starts=[smart_start], max_iter=max_iter)
    kappa, theta, sigma_v, rho, v0 = best_x

    # Final, accurate RMSE in true vol-points: real IV inversion, done once.
    sq_errs = []
    for T, strikes, _mkt_prices, _mkt_vega, iv_mkt in groups:
        model_prices = np.atleast_1d(heston_call_price(
            S0, strikes, r, q, T, kappa, theta, sigma_v, rho, v0, u_max=u_max, n_points=max(n_points, 1000)))
        iv_model = np.array([bs.implied_vol(px, S0, k, r, q, T, "call") for px, k in zip(model_prices, strikes)])
        valid = np.isfinite(iv_model)
        sq_errs.extend(((iv_model[valid] - iv_mkt[valid]) ** 2).tolist())
    rmse_vol_pts = float(np.sqrt(np.mean(sq_errs))) if sq_errs else float("nan")
    feller = 2 * kappa * theta / sigma_v ** 2

    return HestonParams(kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, v0=v0,
                         rmse_vol_pts=rmse_vol_pts, feller_ratio=feller)
