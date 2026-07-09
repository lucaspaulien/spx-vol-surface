#!/usr/bin/env python3
"""End-to-end demo: load a surface (live CSV if --csv is given, otherwise a
synthetic SPX-like snapshot with a KNOWN Heston ground truth), calibrate
both SVI and Heston, print a fit-quality report, compute Greeks, and
(optionally) plot the smiles.

    python scripts/run_calibration.py                     # synthetic demo
    python scripts/run_calibration.py --csv data/live_iv_surface_SPX.csv --spot 5300 --rate 0.045 --div-yield 0.013
    python scripts/run_calibration.py --plot               # requires matplotlib (pip install -e ".[plot]")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import data, greeks, heston, svi  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="path to a cleaned iv_df CSV (ttm,strike,forward,implied_vol)")
    parser.add_argument("--spot", type=float, default=5300.0)
    parser.add_argument("--rate", type=float, default=0.045)
    parser.add_argument("--div-yield", type=float, default=0.013)
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--n-points", type=int, default=1000, help="Heston integration grid size")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.csv:
        iv_df = pd.read_csv(args.csv)
        spot, r, q = args.spot, args.rate, args.div_yield
        print(f"Loaded {len(iv_df)} quotes from {args.csv}")
    else:
        print("No --csv given: using a synthetic SPX-like surface with a KNOWN Heston ground truth "
              "(see volsurface.data.SPX_LIKE_HESTON_PARAMS).")
        chain = data.generate_synthetic_surface(spot=args.spot, r=args.rate, q=args.div_yield)
        iv_df = data.clean_and_compute_iv(chain, spot=args.spot, r=args.rate, q=args.div_yield)
        spot, r, q = args.spot, args.rate, args.div_yield
        print(f"Ground truth Heston params: {chain.attrs['true_heston_params']}")

    print(f"\nSurface: {len(iv_df)} quotes across {iv_df['ttm'].nunique()} maturities "
          f"(spot={spot}, r={r}, q={q})\n")

    print("=" * 70)
    print("SVI calibration (per maturity slice)")
    print("=" * 70)
    slices = svi.calibrate_svi_surface(iv_df, n_starts=args.n_starts, max_iter=args.max_iter)
    for T, sl in sorted(slices.items()):
        print(f"  T={T:6.3f}y  n={sl.n_quotes:3d}  RMSE={sl.rmse_vol_pts * 100:6.3f} vol-pts  "
              f"arb-free={sl.is_arbitrage_free_here()!s:5}  "
              f"(a={sl.a:.4f} b={sl.b:.4f} rho={sl.rho:+.4f} m={sl.m:+.4f} sigma={sl.sigma:.4f})")
    cal_report = svi.check_calendar_arbitrage(slices)
    if len(cal_report):
        n_violations = int(cal_report["violated"].sum())
        print(f"  Calendar-arbitrage check: {n_violations}/{len(cal_report)} adjacent maturity pairs violated")

    print("\n" + "=" * 70)
    print("Heston calibration (whole surface, one parameter set)")
    print("=" * 70)
    hp = heston.calibrate_heston_surface(iv_df, spot, r, q, n_starts=args.n_starts,
                                          max_iter=args.max_iter, n_points=args.n_points)
    print(f"  kappa={hp.kappa:.4f}  theta={hp.theta:.4f}  sigma_v={hp.sigma_v:.4f}  "
          f"rho={hp.rho:+.4f}  v0={hp.v0:.4f}")
    print(f"  RMSE={hp.rmse_vol_pts * 100:.3f} vol-pts   Feller ratio (2*kappa*theta/sigma_v^2)={hp.feller_ratio:.3f} "
          f"({'satisfied' if hp.feller_ratio >= 1 else 'NOT satisfied -- variance can touch zero'})")

    print("\n" + "=" * 70)
    print("Greeks at the ATM strike of the shortest maturity")
    print("=" * 70)
    shortest_T = min(slices)
    sl = slices[shortest_T]
    g = greeks.svi_smile_greeks(sl, spot, r, q)
    i_atm = int(np.argmin(np.abs(g["strike"] - spot)))
    print(f"  SVI smile-adjusted (T={shortest_T:.3f}): "
          f"Delta={g['delta'][i_atm]:.4f}  Gamma={g['gamma'][i_atm]:.6f}  "
          f"Vega={g['vega'][i_atm]:.3f}  Theta={g['theta'][i_atm]:.3f}")

    hg = greeks.heston_fd_greeks(hp, spot, np.array([spot]), r, q, shortest_T)
    print(f"  Heston finite-difference (T={shortest_T:.3f}): "
          f"Delta={hg['delta'][0]:.4f}  Gamma={hg['gamma'][0]:.6f}  "
          f"Vega(v0)={hg['vega_v0'][0]:.3f}  Theta={hg['theta'][0]:.3f}")

    if args.plot:
        _plot(iv_df, slices, hp, spot, r, q)


def _plot(iv_df, slices, hp, spot, r, q):
    import math

    import matplotlib.pyplot as plt

    from volsurface import black_scholes as bs
    from volsurface import heston as hst

    n = len(slices)
    ncols = min(n, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=True, squeeze=False)
    flat_axes = axes.flatten()
    for ax, (T, sl) in zip(flat_axes, sorted(slices.items())):
        grp = iv_df[iv_df["ttm"] == T]
        strikes_plot = np.linspace(grp["strike"].min(), grp["strike"].max(), 200)
        k_plot = np.log(strikes_plot / sl.forward)
        ax.scatter(grp["strike"], grp["implied_vol"] * 100, label="market", color="black", s=10, zorder=3)
        ax.plot(strikes_plot, sl.implied_vol(k_plot) * 100, label="SVI", color="tab:blue")
        heston_prices = np.atleast_1d(hst.heston_call_price(spot, strikes_plot, r, q, T, *hp.as_tuple(), n_points=1000))
        heston_ivs = [bs.implied_vol(p, spot, k, r, q, T, "call") for p, k in zip(heston_prices, strikes_plot)]
        ax.plot(strikes_plot, np.array(heston_ivs) * 100, label="Heston", color="tab:red", linestyle="--")
        ax.set_title(f"T={T:.3f}y", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat_axes[:n]:
        ax.legend(fontsize=6)
    for ax in flat_axes[n:]:
        ax.axis("off")
    fig.supxlabel("Strike")
    fig.supylabel("Implied vol (%)")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "vol_smiles.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
