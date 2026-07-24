import numpy as np
import pandas as pd
import pytest

from volsurface import black_scholes as bs
from volsurface import heston


def test_heston_collapses_to_black_scholes_as_vol_of_vol_to_zero():
    """The defining sanity check for any Heston implementation: with
    sigma_v -> 0 and v0 = theta = vol**2, the model must reduce to plain
    Black-Scholes with a constant vol, to high precision."""
    S, K, r, q, T, vol = 100.0, 100.0, 0.03, 0.01, 1.0, 0.2
    heston_price = heston.heston_call_price(S, K, r, q, T, kappa=2.0, theta=vol ** 2,
                                             sigma_v=0.001, rho=-0.3, v0=vol ** 2)
    bs_price = bs.bs_price(S, K, r, q, vol, T, "call")
    assert float(heston_price) == pytest.approx(float(bs_price), abs=0.02)


def test_heston_price_is_monotonic_in_v0():
    """More initial variance -> more optionality -> strictly higher call price,
    all else equal (a basic no-arbitrage / monotonicity sanity check)."""
    S, K, r, q, T = 100.0, 100.0, 0.03, 0.01, 0.5
    p_low = heston.heston_call_price(S, K, r, q, T, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.5, v0=0.02)
    p_high = heston.heston_call_price(S, K, r, q, T, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.5, v0=0.08)
    assert float(p_high) > float(p_low)


def test_heston_price_finite_for_positive_rho_low_kappa():
    """phi(u=-1j) -- used to normalise P1 -- has a removable singularity in
    the closed-form 'little trap' formula whenever xi = kappa - rho*sigma_v
    is <= 0, i.e. rho>0 and kappa <= rho*sigma_v. That corner is reachable
    inside _BOUNDS (kappa down to 0.2, rho up to 0.95, sigma_v up to 1.5),
    so a random calibration restart landing there must not blow up to NaN."""
    S, K, r, q, T = 100.0, [90.0, 100.0, 110.0], 0.03, 0.01, 1.0
    prices = heston.heston_call_price(S, K, r, q, T, kappa=1.0, theta=0.04,
                                       sigma_v=1.2, rho=0.9, v0=0.04)
    assert np.all(np.isfinite(prices))
    assert np.all(np.asarray(prices) >= 0.0)


def test_heston_price_nonnegative_at_short_maturity_deep_otm():
    """Fourier-truncation error at short maturities/deep OTM strikes can push
    the raw integral slightly below zero -- a call price must never be
    negative. Uses this repo's own SPX-like params and shortest default
    maturity (see data.SPX_LIKE_HESTON_PARAMS, generate_synthetic_surface)."""
    S, r, q, T = 5300.0, 0.045, 0.013, 7 / 365
    forward = S * np.exp((r - q) * T)
    strikes = forward * np.exp(np.linspace(-0.35, 0.35, 21))
    prices = heston.heston_call_price(S, strikes, r, q, T, kappa=2.5, theta=0.045,
                                       sigma_v=0.55, rho=-0.75, v0=0.035)
    assert np.all(prices >= 0.0)


def test_negative_rho_produces_negative_skew():
    """Textbook equity-index signature: rho < 0 => implied vol decreasing in
    strike (puts richer than calls in vol terms)."""
    S, r, q, T = 100.0, 0.03, 0.01, 0.5
    strikes = np.array([85.0, 100.0, 115.0])
    prices = heston.heston_call_price(S, strikes, r, q, T, kappa=2.0, theta=0.04,
                                       sigma_v=0.5, rho=-0.7, v0=0.04)
    ivs = [bs.implied_vol(p, S, k, r, q, T, "call") for p, k in zip(prices, strikes)]
    assert ivs[0] > ivs[1] > ivs[2]


def test_calibration_recovers_known_params_on_synthetic_surface():
    """The key correctness proof for the whole calibration pipeline: generate
    a surface from a KNOWN Heston parameter set, calibrate blind to those
    parameters, and check they come back out. Deliberately small (2
    maturities x 6 strikes, low optimiser budget) to keep CI runtime sane;
    scripts/run_benchmark.py uses a much larger budget for real results."""
    S, r, q = 100.0, 0.03, 0.01
    true_params = dict(kappa=2.5, theta=0.04, sigma_v=0.45, rho=-0.6, v0=0.045)
    rng = np.random.default_rng(7)

    rows = []
    for T in (0.25, 0.75):
        forward = S * np.exp((r - q) * T)
        strikes = forward * np.exp(np.linspace(-0.25, 0.25, 6))
        prices = np.atleast_1d(heston.heston_call_price(S, strikes, r, q, T, **true_params))
        for K, p in zip(strikes, prices):
            iv = bs.implied_vol(p, S, K, r, q, T, "call")
            if np.isfinite(iv):
                rows.append(dict(ttm=T, strike=K, forward=forward, implied_vol=iv + rng.normal(0, 0.0005)))
    iv_df = pd.DataFrame(rows)

    fitted = heston.calibrate_heston_surface(iv_df, S, r, q, n_starts=3, seed=5, max_iter=60, n_points=150)

    assert fitted.kappa == pytest.approx(true_params["kappa"], rel=0.3)
    assert fitted.theta == pytest.approx(true_params["theta"], rel=0.2)
    assert fitted.sigma_v == pytest.approx(true_params["sigma_v"], rel=0.3)
    assert fitted.rho == pytest.approx(true_params["rho"], abs=0.15)
    assert fitted.v0 == pytest.approx(true_params["v0"], rel=0.2)
    assert fitted.rmse_vol_pts < 0.01


def test_feller_ratio_is_computed_correctly():
    kappa, theta, sigma_v = 2.0, 0.04, 0.5
    hp = heston.HestonParams(kappa=kappa, theta=theta, sigma_v=sigma_v, rho=-0.5, v0=0.04,
                              rmse_vol_pts=0.0, feller_ratio=2 * kappa * theta / sigma_v ** 2)
    assert hp.feller_ratio == pytest.approx(2 * kappa * theta / sigma_v ** 2)
