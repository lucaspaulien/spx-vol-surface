"""Brute-force finite-difference audit of the closed-form Black-Scholes
Greeks in black_scholes.greeks().

test_black_scholes.py already spot-checks delta/gamma/vega at a single
point; this file widens that into a systematic cross-check across strikes,
vols, maturities, and both option types, and adds theta and rho (previously
untested against finite differences at all). The point is to catch a sign
error or a wrong convention in the closed-form formulas, not to test the
finite-difference method itself.

Theta sign convention: black_scholes.greeks()'s theta is dPrice/dt where t
is *calendar* time (t = T_expiry - ttm), i.e. calendar time decay -- the
standard "theta is negative for a long option" convention. Since
dPrice/dt = -dPrice/dttm, the finite-difference estimate is built from
price(ttm - h) and price(ttm + h) with the sign flipped accordingly.
"""
import numpy as np
import pytest

from volsurface import black_scholes as bs

GRID = [
    # (spot, strike, rate, div_yield, vol, ttm)
    (100.0, 100.0, 0.03, 0.01, 0.20, 1.0),    # ATM
    (100.0, 80.0, 0.03, 0.01, 0.20, 1.0),     # ITM call / OTM put
    (100.0, 130.0, 0.03, 0.01, 0.20, 1.0),    # OTM call / ITM put
    (100.0, 100.0, 0.00, 0.00, 0.35, 0.05),   # short-dated, high vol, zero rates
    (100.0, 105.0, 0.05, 0.02, 0.15, 3.0),    # long-dated
    (753.0, 753.0, 0.045, 0.013, 0.55, 0.057),  # SPX/SPY-like: high vol, near-weekly
]


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(S, K, r, q, vol, T, option_type):
    h = S * 1e-5
    analytic = float(bs.greeks(S, K, r, q, vol, T, option_type)["delta"])
    fd = (float(bs.bs_price(S + h, K, r, q, vol, T, option_type))
          - float(bs.bs_price(S - h, K, r, q, vol, T, option_type))) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-5, rel=1e-4)


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(S, K, r, q, vol, T, option_type):
    h = S * 1e-3  # gamma is a 2nd derivative -- needs a coarser step than delta/vega for float precision
    analytic = float(bs.greeks(S, K, r, q, vol, T, option_type)["gamma"])
    fd = (float(bs.bs_price(S + h, K, r, q, vol, T, option_type))
          - 2 * float(bs.bs_price(S, K, r, q, vol, T, option_type))
          + float(bs.bs_price(S - h, K, r, q, vol, T, option_type))) / (h ** 2)
    assert analytic == pytest.approx(fd, abs=1e-4, rel=1e-3)


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(S, K, r, q, vol, T, option_type):
    h = 1e-5
    analytic = float(bs.greeks(S, K, r, q, vol, T, option_type)["vega"])
    fd = (float(bs.bs_price(S, K, r, q, vol + h, T, option_type))
          - float(bs.bs_price(S, K, r, q, vol - h, T, option_type))) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-5, rel=1e-4)


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(S, K, r, q, vol, T, option_type):
    h = min(T * 1e-3, 1e-4)
    analytic = float(bs.greeks(S, K, r, q, vol, T, option_type)["theta"])
    # dPrice/dt (calendar time) = -dPrice/dttm
    fd = -(float(bs.bs_price(S, K, r, q, vol, T + h, option_type))
           - float(bs.bs_price(S, K, r, q, vol, T - h, option_type))) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-3, rel=1e-3)


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(S, K, r, q, vol, T, option_type):
    h = 1e-5
    analytic = float(bs.greeks(S, K, r, q, vol, T, option_type)["rho"])
    fd = (float(bs.bs_price(S, K, r + h, q, vol, T, option_type))
          - float(bs.bs_price(S, K, r - h, q, vol, T, option_type))) / (2 * h)
    assert analytic == pytest.approx(fd, abs=1e-4, rel=1e-4)


@pytest.mark.parametrize("S,K,r,q,vol,T", GRID)
def test_call_put_greek_parity_relations(S, K, r, q, vol, T):
    """Textbook put-call parity identities for the Greeks themselves --
    independent of finite differences, these must hold *exactly* (up to
    float error) from the formulas alone."""
    gc = bs.greeks(S, K, r, q, vol, T, "call")
    gp = bs.greeks(S, K, r, q, vol, T, "put")

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    assert float(gc["delta"] - gp["delta"]) == pytest.approx(disc_q, abs=1e-10)
    assert float(gc["gamma"]) == pytest.approx(float(gp["gamma"]), abs=1e-10)
    assert float(gc["vega"]) == pytest.approx(float(gp["vega"]), abs=1e-10)
    assert float(gc["rho"] - gp["rho"]) == pytest.approx(K * T * disc_r, abs=1e-8)
