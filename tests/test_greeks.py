import numpy as np
import pytest

from volsurface import black_scholes as bs
from volsurface import greeks
from volsurface import heston
from volsurface import svi


def test_svi_smile_greeks_explicit_strikes_overrides_default_grid():
    """svi_smile_greeks' default 25-point grid is centered on the fitted
    slice's own (m, sigma), which is not uniquely identified (see svi.py) --
    so picking "the nearest grid point to spot" as a stand-in for "the ATM
    Greek" can land a couple percent away from true at-the-money. Callers
    that need Greeks at a specific strike (e.g. the true ATM point,
    ``forward``) must be able to request it directly and get it exactly,
    not the nearest point of an unrelated auto-generated grid."""
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    forward = S * np.exp((r - q) * T)
    sl = svi.SVISlice(ttm=T, forward=forward, a=0.02, b=0.2, rho=-0.4, m=0.6, sigma=0.15,
                       rmse_vol_pts=0.0, n_quotes=10)
    g_default = greeks.svi_smile_greeks(sl, S, r, q)
    assert not np.any(np.isclose(g_default["strike"], forward, rtol=1e-9))  # forward isn't in the default grid

    g_explicit = greeks.svi_smile_greeks(sl, S, r, q, strikes=np.array([forward]))
    assert len(g_explicit["strike"]) == 1
    assert float(g_explicit["strike"][0]) == pytest.approx(forward)
    assert np.isfinite(g_explicit["delta"][0])


def test_svi_smile_delta_equals_bs_delta_on_a_flat_smile():
    """Degenerate case: if b=0 (flat smile, no skew, no curvature), the SVI
    surface is just a single constant vol, and the sticky-strike smile Delta
    must collapse exactly to the plain Black-Scholes Delta at that vol."""
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    flat_vol = 0.22
    forward = S * np.exp((r - q) * T)
    sl = svi.SVISlice(ttm=T, forward=forward, a=flat_vol ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1,
                       rmse_vol_pts=0.0, n_quotes=10)
    g = greeks.svi_smile_greeks(sl, S, r, q)
    i_atm = int(np.argmin(np.abs(g["strike"] - S)))
    bs_delta = float(bs.greeks(S, float(g["strike"][i_atm]), r, q, flat_vol, T, "call")["delta"])
    assert float(g["delta"][i_atm]) == pytest.approx(bs_delta, abs=2e-3)


def test_svi_smile_gamma_positive_near_the_money():
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    forward = S * np.exp((r - q) * T)
    sl = svi.SVISlice(ttm=T, forward=forward, a=0.02, b=0.2, rho=-0.4, m=0.0, sigma=0.15,
                       rmse_vol_pts=0.0, n_quotes=10)
    g = greeks.svi_smile_greeks(sl, S, r, q)
    i_atm = int(np.argmin(np.abs(g["strike"] - S)))
    assert float(g["gamma"][i_atm]) > 0


def test_heston_fd_delta_matches_bs_delta_in_the_zero_vol_of_vol_limit():
    S, r, q, T, vol = 100.0, 0.02, 0.0, 0.5, 0.25
    hp = heston.HestonParams(kappa=2.0, theta=vol ** 2, sigma_v=1e-3, rho=-0.2, v0=vol ** 2,
                              rmse_vol_pts=0.0, feller_ratio=999.0)
    strikes = np.array([90.0, 100.0, 110.0])
    hg = greeks.heston_fd_greeks(hp, S, strikes, r, q, T)
    for i, K in enumerate(strikes):
        bs_delta = float(bs.greeks(S, K, r, q, vol, T, "call")["delta"])
        assert hg["delta"][i] == pytest.approx(bs_delta, abs=5e-3)


def test_heston_fd_delta_between_zero_and_one_for_calls():
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    hp = heston.HestonParams(kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.6, v0=0.04,
                              rmse_vol_pts=0.0, feller_ratio=0.64)
    hg = greeks.heston_fd_greeks(hp, S, np.array([80.0, 100.0, 130.0]), r, q, T)
    assert np.all((hg["delta"] > 0) & (hg["delta"] < 1))
    assert np.all(hg["gamma"] > 0)
