import numpy as np
import pytest

from volsurface import black_scholes as bs


def test_call_put_parity():
    S, K, r, q, vol, T = 100.0, 95.0, 0.03, 0.01, 0.25, 0.75
    call = bs.bs_price(S, K, r, q, vol, T, "call")
    put = bs.bs_price(S, K, r, q, vol, T, "put")
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


@pytest.mark.parametrize("true_vol,option_type", [
    (0.10, "call"), (0.22, "call"), (0.55, "call"),
    (0.10, "put"), (0.22, "put"), (0.55, "put"),
])
def test_implied_vol_recovers_known_vol(true_vol, option_type):
    S, K, r, q, T = 100.0, 105.0, 0.03, 0.01, 0.5
    price = bs.bs_price(S, K, r, q, true_vol, T, option_type)
    iv = bs.implied_vol(price, S, K, r, q, T, option_type)
    assert iv == pytest.approx(true_vol, abs=1e-6)


def test_implied_vol_across_moneyness_grid():
    """Sanity: recovers vol across a wide range of strikes without diverging."""
    S, r, q, T, true_vol = 100.0, 0.02, 0.0, 1.0, 0.30
    strikes = np.linspace(50, 200, 40)
    for K in strikes:
        price = bs.bs_price(S, K, r, q, true_vol, T, "call")
        iv = bs.implied_vol(price, S, K, r, q, T, "call")
        if np.isnan(iv):
            continue  # far wings can legitimately have ~0 time value, see docstring
        assert iv == pytest.approx(true_vol, abs=1e-4)


def test_implied_vol_rejects_arbitrage_violating_price():
    """A price below intrinsic value cannot be inverted -- must return NaN,
    never a fabricated number."""
    S, K, r, q, T = 100.0, 80.0, 0.0, 0.0, 1.0
    intrinsic = S - K
    iv = bs.implied_vol(intrinsic - 1.0, S, K, r, q, T, "call")
    assert np.isnan(iv)


def test_vega_matches_finite_difference_of_price():
    S, K, r, q, vol, T = 100.0, 100.0, 0.02, 0.0, 0.2, 1.0
    h = 1e-5
    analytic = float(bs.vega(S, K, r, q, vol, T))
    fd = (float(bs.bs_price(S, K, r, q, vol + h, T, "call"))
          - float(bs.bs_price(S, K, r, q, vol - h, T, "call"))) / (2 * h)
    assert analytic == pytest.approx(fd, rel=1e-4)


def test_greeks_match_finite_differences():
    S, K, r, q, vol, T = 100.0, 105.0, 0.03, 0.01, 0.25, 0.5
    g = bs.greeks(S, K, r, q, vol, T, "call")

    h = 1e-4
    delta_fd = (float(bs.bs_price(S + h, K, r, q, vol, T, "call"))
                - float(bs.bs_price(S - h, K, r, q, vol, T, "call"))) / (2 * h)
    assert float(g["delta"]) == pytest.approx(delta_fd, abs=1e-4)

    gamma_fd = (float(bs.bs_price(S + h, K, r, q, vol, T, "call"))
                - 2 * float(bs.bs_price(S, K, r, q, vol, T, "call"))
                + float(bs.bs_price(S - h, K, r, q, vol, T, "call"))) / (h ** 2)
    assert float(g["gamma"]) == pytest.approx(gamma_fd, abs=1e-2)

    assert 0.0 < float(g["delta"]) < 1.0
    assert float(g["gamma"]) > 0.0
    assert float(g["vega"]) > 0.0


def test_deep_itm_call_delta_near_one_deep_otm_near_zero():
    S, r, q, vol, T = 100.0, 0.02, 0.0, 0.2, 1.0
    g_itm = bs.greeks(S, 20.0, r, q, vol, T, "call")
    g_otm = bs.greeks(S, 400.0, r, q, vol, T, "call")
    assert float(g_itm["delta"]) > 0.99
    assert float(g_otm["delta"]) < 0.01
