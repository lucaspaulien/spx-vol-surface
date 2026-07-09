"""Data layer: live SPX/SPY option chains via yfinance, a synthetic-but-Heston-
true surface generator for offline reproducibility, and the mid-price ->
implied-vol cleaning step shared by both.

Why a synthetic generator ships alongside the live fetcher: this repo's CI
and test suite must be able to run with zero network access (see
.github/workflows/ci.yml) -- exactly the same design choice as most
production quant codebases, which never let their test suite depend on a
live market data feed being reachable. ``generate_synthetic_surface`` draws
its "market" prices from a Heston process with a *known* parameter set,
which doubles as the ground truth used to prove the calibration is correct
in tests/test_heston.py, not just self-consistent.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import black_scholes as bs
from . import heston as hst

# A parameter set broadly in line with published SPX Heston calibrations
# (see e.g. Gatheral, "The Volatility Surface", ch. 3-4): moderate mean
# reversion, ~20% long-run vol, strong negative spot/vol correlation (the
# equity "leverage effect"), meaningful vol-of-vol.
SPX_LIKE_HESTON_PARAMS = dict(kappa=2.5, theta=0.045, sigma_v=0.55, rho=-0.75, v0=0.035)


def generate_synthetic_surface(spot=5300.0, r=0.045, q=0.013,
                                maturities_days=(7, 30, 60, 91, 182, 365),
                                n_strikes=21, moneyness_range=0.35,
                                heston_params=None, quote_noise_vol=0.0015,
                                bid_ask_half_spread=0.01, seed=0) -> pd.DataFrame:
    """Simulate a full SPX-like option chain snapshot from a known Heston
    process, in the same raw shape a live chain would arrive in (bid/ask,
    not implied vol yet) -- so it exercises exactly the same cleaning code
    path as :func:`fetch_live_spx_chain`.

    Returns columns: ttm, strike, forward, bid, ask, option_type.
    """
    heston_params = heston_params or SPX_LIKE_HESTON_PARAMS
    rng = np.random.default_rng(seed)
    rows = []
    for d in maturities_days:
        ttm = d / 365.0
        forward = spot * np.exp((r - q) * ttm)
        strikes = forward * np.exp(np.linspace(-moneyness_range, moneyness_range, n_strikes))
        true_prices = np.atleast_1d(hst.heston_call_price(
            spot, strikes, r, q, ttm, **heston_params, n_points=1200))
        # Multiplicative noise on price (mimics quote-to-quote market noise),
        # plus a symmetric bid/ask spread around the noisy mid.
        noisy_mid = true_prices * (1.0 + rng.normal(0, quote_noise_vol, size=true_prices.shape))
        noisy_mid = np.maximum(noisy_mid, 1e-6)
        half_spread = np.maximum(noisy_mid * bid_ask_half_spread, 0.01)
        for K, mid, hs in zip(strikes, noisy_mid, half_spread):
            rows.append(dict(ttm=ttm, strike=float(K), forward=float(forward),
                              bid=float(mid - hs), ask=float(mid + hs), option_type="call"))
    df = pd.DataFrame(rows)
    df.attrs["spot"] = spot
    df.attrs["rate"] = r
    df.attrs["div_yield"] = q
    df.attrs["true_heston_params"] = heston_params
    return df


def clean_and_compute_iv(chain: pd.DataFrame, spot: float, r: float, q: float,
                          min_price=0.05, max_rel_spread=0.35, min_ttm_days=3) -> pd.DataFrame:
    """Raw (bid, ask, strike, ttm) chain -> clean (ttm, forward, strike,
    implied_vol) surface ready for svi.py / heston.py.

    Filters applied, each of which is a real, named source of bad quotes
    rather than a blanket "drop anything weird":
    - ``min_price``: penny/near-worthless quotes carry almost no information
      and blow up implied-vol inversion (division by ~0 vega).
    - ``max_rel_spread``: quotes with bid-ask spread > ``max_rel_spread`` of
      the mid are too illiquid to trust as a real market opinion.
    - ``min_ttm_days``: options expiring within days have implied vols that
      are extremely noisy / dominated by microstructure, not the smile.
    - any mid price failing to invert to a finite vol (intrinsic-violating
      or simply unbracketed) is dropped rather than silently imputed.
    """
    df = chain.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"].clip(lower=1e-6)

    before = len(df)
    df = df[(df["mid"] >= min_price) & (df["rel_spread"] <= max_rel_spread) &
            (df["ttm"] >= min_ttm_days / 365.0)]
    dropped_liquidity = before - len(df)

    ivs = []
    for row in df.itertuples(index=False):
        iv = bs.implied_vol(row.mid, spot, row.strike, r, q, row.ttm, row.option_type)
        ivs.append(iv)
    df = df.assign(implied_vol=ivs)

    before2 = len(df)
    df = df[np.isfinite(df["implied_vol"]) & (df["implied_vol"] > 0.01) & (df["implied_vol"] < 3.0)]
    dropped_inversion = before2 - len(df)

    df.attrs["n_dropped_liquidity"] = dropped_liquidity
    df.attrs["n_dropped_inversion"] = dropped_inversion
    return df[["ttm", "strike", "forward", "implied_vol"]].reset_index(drop=True)


def fetch_live_spx_chain(ticker="^SPX", max_maturities=8, r=0.045, q=0.013) -> pd.DataFrame:
    """Pull a live option chain via yfinance. Requires a normal internet
    connection -- NOT expected to work inside a network-restricted sandbox,
    intended to be run on a workstation with regular internet access
    (``python scripts/fetch_live_data.py``).

    ``ticker="^SPX"`` targets the index directly; if Yahoo does not serve a
    chain for the index on your feed, fall back to ``ticker="SPY"`` (the
    S&P 500 ETF -- same shape of smile, strikes/spot ~10x smaller) and pass
    the ETF's own dividend yield.
    """
    import yfinance as yf  # deferred import: not required unless this function is called

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["last_price"])
    expiries = tk.options[:max_maturities]

    rows = []
    today = date.today()
    for expiry in expiries:
        ttm_days = (date.fromisoformat(expiry) - today).days
        if ttm_days <= 0:
            continue
        chain = tk.option_chain(expiry)
        calls = chain.calls
        for _, opt in calls.iterrows():
            if opt["bid"] <= 0 or opt["ask"] <= 0:
                continue
            rows.append(dict(ttm=ttm_days / 365.0, strike=float(opt["strike"]),
                              forward=spot * np.exp((r - q) * ttm_days / 365.0),
                              bid=float(opt["bid"]), ask=float(opt["ask"]), option_type="call"))
    df = pd.DataFrame(rows)
    df.attrs["spot"] = spot
    df.attrs["rate"] = r
    df.attrs["div_yield"] = q
    return df
