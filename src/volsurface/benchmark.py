"""Multi-date benchmark: SVI vs Heston fit quality, and both vs a flat-vol
Black-Scholes baseline, measured as out-of-sample repricing error.

Methodology per "date" (a market snapshot):
  1. Split the surface's strikes into a calibration set and a held-out test
     set (default 80/20, stratified per maturity so every expiry contributes
     held-out points).
  2. Calibrate SVI (per-slice) and Heston (whole surface) on the calibration
     set only.
  3. Reprice the held-out test strikes from each calibrated model and from a
     flat-vol Black-Scholes baseline (ATM implied vol of each maturity,
     applied uniformly across all its strikes -- the naive "one vol per
     expiry" approach this whole project is trying to improve on).
  4. Record RMSE (in implied-vol points) for all three on the held-out set.

This out-of-sample split is the honest way to compare "fit quality": fitting
the calibration set itself would always favour the model with more free
parameters locally (SVI, 5 params PER slice) over Heston (5 params for the
WHOLE surface), which is not the comparison that matters for a pricing/
hedging desk -- what matters is how well the calibrated surface predicts
strikes it was not fit to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import black_scholes as bs
from . import data as data_mod
from . import heston as hst
from . import svi


@dataclass
class DateBenchmarkResult:
    date_id: int
    n_calib: int
    n_test: int
    svi_rmse_vol_pts: float
    heston_rmse_vol_pts: float
    flat_bs_rmse_vol_pts: float
    heston_feller_ratio: float


def _train_test_split_per_maturity(iv_df: pd.DataFrame, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _T, grp in iv_df.groupby("ttm"):
        idx = grp.index.to_numpy(copy=True)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return iv_df.loc[train_idx].reset_index(drop=True), iv_df.loc[test_idx].reset_index(drop=True)


def _flat_bs_rmse(train_df, test_df, spot, r, q) -> float:
    """Flat-vol baseline: one ATM implied vol per maturity (from the
    calibration set), applied to every held-out strike of that maturity."""
    atm_vol_by_ttm = {}
    for T, grp in train_df.groupby("ttm"):
        atm_idx = int(np.argmin(np.abs(grp["strike"].to_numpy() - grp["forward"].to_numpy())))
        atm_vol_by_ttm[T] = grp["implied_vol"].to_numpy()[atm_idx]

    errs = []
    for row in test_df.itertuples(index=False):
        flat_vol = atm_vol_by_ttm.get(row.ttm)
        if flat_vol is None:
            continue
        errs.append((flat_vol - row.implied_vol) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def _svi_rmse(train_df, test_df) -> float:
    slices = svi.calibrate_svi_surface(train_df)
    errs = []
    for row in test_df.itertuples(index=False):
        sl = slices.get(row.ttm)
        if sl is None:
            continue
        k = np.log(row.strike / row.forward)
        model_vol = float(sl.implied_vol(np.array([k]))[0])
        errs.append((model_vol - row.implied_vol) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")


def _heston_rmse(train_df, test_df, spot, r, q, **calib_kwargs) -> tuple[float, float]:
    hp = hst.calibrate_heston_surface(train_df, spot, r, q, **calib_kwargs)
    errs = []
    for T, grp in test_df.groupby("ttm"):
        strikes = grp["strike"].to_numpy()
        model_prices = np.atleast_1d(hst.heston_call_price(spot, strikes, r, q, T, *hp.as_tuple(),
                                                             n_points=1000))
        for K, px, iv_mkt in zip(strikes, model_prices, grp["implied_vol"].to_numpy()):
            iv_model = bs.implied_vol(px, spot, K, r, q, T, "call")
            if np.isfinite(iv_model):
                errs.append((iv_model - iv_mkt) ** 2)
    rmse = float(np.sqrt(np.mean(errs))) if errs else float("nan")
    return rmse, hp.feller_ratio


def run_single_date(date_id: int, spot=5300.0, r=0.045, q=0.013,
                     maturities_days=(30, 91, 182), n_strikes=11,
                     test_frac=0.2, heston_kwargs=None, seed=None) -> DateBenchmarkResult:
    """Generate one synthetic 'trading date' snapshot and score SVI/Heston/flat-BS
    on a held-out set of strikes. See :func:`run_benchmark` for looping this
    over several dates and :mod:`scripts.fetch_live_data` for swapping the
    synthetic snapshot for a real one."""
    heston_kwargs = heston_kwargs or {}
    seed = date_id if seed is None else seed

    chain = data_mod.generate_synthetic_surface(
        spot=spot, r=r, q=q, maturities_days=maturities_days, n_strikes=n_strikes, seed=seed)
    iv_df = data_mod.clean_and_compute_iv(chain, spot=spot, r=r, q=q)

    train_df, test_df = _train_test_split_per_maturity(iv_df, test_frac=test_frac, seed=seed)

    svi_rmse = _svi_rmse(train_df, test_df)
    heston_rmse, feller = _heston_rmse(train_df, test_df, spot, r, q, seed=seed, **heston_kwargs)
    flat_rmse = _flat_bs_rmse(train_df, test_df, spot, r, q)

    return DateBenchmarkResult(date_id=date_id, n_calib=len(train_df), n_test=len(test_df),
                                svi_rmse_vol_pts=svi_rmse, heston_rmse_vol_pts=heston_rmse,
                                flat_bs_rmse_vol_pts=flat_rmse, heston_feller_ratio=feller)


def run_benchmark(n_dates=10, **kwargs) -> pd.DataFrame:
    """Loop :func:`run_single_date` over ``n_dates`` independent synthetic
    snapshots (different random seed = different simulated day) and return
    one row per date. This is the table the README's headline numbers come
    from -- see scripts/run_benchmark.py for the CLI entry point."""
    rows = [run_single_date(i, **kwargs) for i in range(n_dates)]
    return pd.DataFrame([r.__dict__ for r in rows])


def summarize(results_df: pd.DataFrame) -> dict:
    out = {
        "n_dates": len(results_df),
        "svi_rmse_mean": float(results_df["svi_rmse_vol_pts"].mean()),
        "heston_rmse_mean": float(results_df["heston_rmse_vol_pts"].mean()),
        "flat_bs_rmse_mean": float(results_df["flat_bs_rmse_vol_pts"].mean()),
    }
    out["heston_vs_flat_bs_improvement_pct"] = float(
        100 * (1 - out["heston_rmse_mean"] / out["flat_bs_rmse_mean"]))
    out["svi_vs_flat_bs_improvement_pct"] = float(
        100 * (1 - out["svi_rmse_mean"] / out["flat_bs_rmse_mean"]))
    out["heston_vs_svi_diff_pct"] = float(
        100 * (out["heston_rmse_mean"] - out["svi_rmse_mean"]) / out["svi_rmse_mean"])
    return out
