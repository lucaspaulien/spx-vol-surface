#!/usr/bin/env python3
"""Pull a LIVE SPX (or SPY) option chain via yfinance and save the cleaned
implied-vol surface to data/live_iv_surface.csv.

Run this on a machine with normal internet access (NOT inside a
network-restricted sandbox/CI runner):

    pip install -e ".[live]"
    python scripts/fetch_live_data.py --ticker ^SPX
    python scripts/fetch_live_data.py --ticker SPY   # fallback if ^SPX has no chain on your feed

This is the one script in the repo that talks to the network. Everything
downstream (svi.py, heston.py, greeks.py, benchmark.py) only ever consumes
the resulting CSV, live or synthetic, through the exact same
``data.clean_and_compute_iv`` schema.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volsurface import data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="^SPX", help="^SPX (index) or SPY (ETF fallback)")
    parser.add_argument("--rate", type=float, default=0.045, help="risk-free rate (continuous, annualised)")
    parser.add_argument("--div-yield", type=float, default=0.013, help="dividend yield (continuous, annualised)")
    parser.add_argument("--max-maturities", type=int, default=8)
    parser.add_argument("--out", default=None, help="output CSV path (default: data/live_iv_surface_<ticker>.csv)")
    args = parser.parse_args()

    print(f"Fetching live option chain for {args.ticker} via yfinance...")
    try:
        chain = data.fetch_live_spx_chain(ticker=args.ticker, max_maturities=args.max_maturities,
                                           r=args.rate, q=args.div_yield)
    except Exception as e:
        print(f"\nError: could not fetch a live option chain for ticker {args.ticker!r}.")
        print("This usually means the ticker is invalid/delisted, or Yahoo Finance is temporarily "
              "rate-limiting/unreachable. Double-check the ticker (e.g. '^SPX' or 'SPY') and retry.")
        print(f"Underlying error: {type(e).__name__}: {e}")
        sys.exit(1)

    if len(chain) == 0:
        print(f"\nError: 0 quotes with a valid bid/ask came back for {args.ticker!r} "
              f"(--max-maturities {args.max_maturities}). Try a larger --max-maturities, "
              "a different ticker, or retry later (illiquid strikes often show a stale 0/0 quote).")
        sys.exit(1)
    print(f"  raw chain: {len(chain)} quotes across {chain['ttm'].nunique()} maturities, spot={chain.attrs['spot']:.2f}")

    iv_df = data.clean_and_compute_iv(chain, spot=chain.attrs["spot"], r=args.rate, q=args.div_yield)
    print(f"  cleaned surface: {len(iv_df)} quotes "
          f"(dropped {iv_df.attrs['n_dropped_liquidity']} on liquidity, "
          f"{iv_df.attrs['n_dropped_inversion']} on IV inversion)")
    if len(iv_df) == 0:
        print("\nWarning: every quote was filtered out by the liquidity/spread/inversion filters -- "
              "the saved CSV is empty and unusable for calibration. Try a larger --max-maturities "
              "or a more liquid ticker.")

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data" / f"live_iv_surface_{args.ticker.strip('^')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iv_df.to_csv(out_path, index=False)
    meta_path = out_path.with_suffix(".meta.txt")
    meta_path.write_text(f"spot={chain.attrs['spot']}\nrate={args.rate}\ndiv_yield={args.div_yield}\nticker={args.ticker}\n")
    print(f"Saved: {out_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
