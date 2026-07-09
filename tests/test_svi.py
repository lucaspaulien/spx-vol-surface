import numpy as np
import pandas as pd

from volsurface import svi


def _synthetic_slice(true_params, ttm=0.5, forward=100.0, n=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    k = np.log(forward * np.exp(np.linspace(-0.4, 0.4, n)) / forward)
    w_true = true_params["a"] + true_params["b"] * (
        true_params["rho"] * (k - true_params["m"])
        + np.sqrt((k - true_params["m"]) ** 2 + true_params["sigma"] ** 2))
    vol_true = np.sqrt(w_true / ttm)
    vol_obs = vol_true + rng.normal(0, noise, size=vol_true.shape)
    w_obs = vol_obs ** 2 * ttm
    return k, w_obs


def test_calibration_recovers_known_curve_noiseless():
    """Raw SVI's 5 parameters are not uniquely identified from one curve (see
    svi.py module docstring) -- several different-looking parameter tuples
    can fit the same w(k) essentially exactly. So the correctness check here
    is on the FITTED CURVE (implied vol at each traded log-moneyness),
    which IS well identified, rather than on the individual raw parameters."""
    true_params = dict(a=0.02, b=0.25, rho=-0.6, m=0.05, sigma=0.15)
    k, w = _synthetic_slice(true_params, noise=0.0)
    sl = svi.calibrate_svi_slice(k, w, ttm=0.5, forward=100.0, seed=1, n_starts=10, max_iter=800)

    true_vol = np.sqrt(w / 0.5)
    fitted_vol = sl.implied_vol(k)
    assert np.max(np.abs(fitted_vol - true_vol)) < 2e-3
    assert sl.rmse_vol_pts < 2e-3


def test_calibration_robust_to_small_noise():
    true_params = dict(a=0.02, b=0.25, rho=-0.6, m=0.05, sigma=0.15)
    k, w = _synthetic_slice(true_params, noise=0.001, seed=2)
    sl = svi.calibrate_svi_slice(k, w, ttm=0.5, forward=100.0, seed=1, n_starts=10, max_iter=800)
    # Fit quality should sit close to the injected noise floor, not blow up.
    assert sl.rmse_vol_pts < 0.006


def test_total_variance_is_nonnegative_on_a_wide_grid():
    true_params = dict(a=0.02, b=0.25, rho=-0.6, m=0.05, sigma=0.15)
    k, w = _synthetic_slice(true_params, noise=0.0005, seed=3)
    sl = svi.calibrate_svi_slice(k, w, ttm=0.5, forward=100.0, seed=1)
    k_grid = np.linspace(-2, 2, 500)
    assert np.all(sl.total_variance(k_grid) >= -1e-9)


def test_calendar_arbitrage_flags_a_deliberately_crossed_surface():
    """Build two slices where the LONGER maturity has deliberately LOWER
    total variance everywhere -- a textbook calendar arbitrage -- and check
    the detector actually flags it."""
    short = svi.SVISlice(ttm=0.25, forward=100.0, a=0.05, b=0.2, rho=-0.3, m=0.0, sigma=0.2,
                          rmse_vol_pts=0.0, n_quotes=10)
    long_ = svi.SVISlice(ttm=1.0, forward=100.0, a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.2,
                          rmse_vol_pts=0.0, n_quotes=10)
    report = svi.check_calendar_arbitrage({0.25: short, 1.0: long_})
    assert bool(report["violated"].iloc[0])


def test_calendar_arbitrage_clean_when_variance_increases_with_maturity():
    short = svi.SVISlice(ttm=0.25, forward=100.0, a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2,
                          rmse_vol_pts=0.0, n_quotes=10)
    long_ = svi.SVISlice(ttm=1.0, forward=100.0, a=0.05, b=0.2, rho=-0.3, m=0.0, sigma=0.2,
                          rmse_vol_pts=0.0, n_quotes=10)
    report = svi.check_calendar_arbitrage({0.25: short, 1.0: long_})
    assert not bool(report["violated"].iloc[0])


def test_calibrate_svi_surface_returns_one_slice_per_maturity():
    rows = []
    for ttm in (0.25, 0.5, 1.0):
        k, w = _synthetic_slice(dict(a=0.02, b=0.25, rho=-0.5, m=0.0, sigma=0.15), ttm=ttm, noise=0.001, seed=1)
        vol = np.sqrt(np.maximum(w, 1e-12) / ttm)
        strikes = 100.0 * np.exp(k)
        rows.extend(dict(ttm=ttm, strike=s, forward=100.0, implied_vol=v) for s, v in zip(strikes, vol))
    iv_df = pd.DataFrame(rows)
    slices = svi.calibrate_svi_surface(iv_df, seed=1, n_starts=4, max_iter=200)
    assert set(slices.keys()) == {0.25, 0.5, 1.0}
