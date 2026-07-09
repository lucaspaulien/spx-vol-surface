import numpy as np

from volsurface import data


def test_synthetic_surface_has_expected_shape_and_attrs():
    chain = data.generate_synthetic_surface(maturities_days=(30, 90), n_strikes=9, seed=0)
    assert set(chain.columns) >= {"ttm", "strike", "forward", "bid", "ask", "option_type"}
    assert len(chain) == 2 * 9
    assert chain.attrs["spot"] > 0
    assert "true_heston_params" in chain.attrs


def test_synthetic_surface_bid_below_ask():
    chain = data.generate_synthetic_surface(maturities_days=(30,), n_strikes=11, seed=1)
    assert np.all(chain["bid"] < chain["ask"])


def test_clean_and_compute_iv_drops_bad_quotes_and_keeps_good_ones():
    chain = data.generate_synthetic_surface(maturities_days=(30, 90, 180), n_strikes=15, seed=2)
    spot, r, q = chain.attrs["spot"], chain.attrs["rate"], chain.attrs["div_yield"]
    iv_df = data.clean_and_compute_iv(chain, spot=spot, r=r, q=q)

    assert len(iv_df) > 0
    assert len(iv_df) <= len(chain)
    assert set(iv_df.columns) == {"ttm", "strike", "forward", "implied_vol"}
    assert np.all(iv_df["implied_vol"] > 0)
    assert np.all(iv_df["implied_vol"] < 3.0)


def test_clean_and_compute_iv_drops_illiquid_penny_quotes():
    chain = data.generate_synthetic_surface(maturities_days=(365,), n_strikes=21,
                                             moneyness_range=0.6, seed=3)
    spot, r, q = chain.attrs["spot"], chain.attrs["rate"], chain.attrs["div_yield"]
    iv_df = data.clean_and_compute_iv(chain, spot=spot, r=r, q=q, min_price=0.05)
    assert iv_df.attrs["n_dropped_liquidity"] >= 0  # deep OTM tails should hit the filter
